"""Train from a local cache built by build_cache.py. Fast enough to iterate.

Reading from /workspace starved the GPU to 16%. With the cache on local disk the
run is GPU-bound, so experiments take minutes instead of an hour.

  python3 train_cached.py --cache train_s8_i256_c160_k60_f-1 --epochs 20
  python3 train_cached.py --cache ... --arch resnet34 --use 4 --seed 1

Two knobs worth understanding:

  --use N   how many of the cached slices per plane to feed per step. Caching 8
            and using 4 means a different random 4 each epoch, which is real
            augmentation along the slice axis and costs nothing. Set equal to
            the cached count to disable.
  --labels  the pseudo-label file. Everything downstream is bounded by this, so
            it is the highest-leverage flag here.
"""
import argparse
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ZIP = os.environ.get("RSNA_ZIP", "/workspace/rsna-knee-abnormality-detection.zip")   # train.csv source; an OAI run points this at a tiny zip whose train.csv has Report := subject id (folds by subject)
WORK = os.environ.get("WORK", "/workspace")
CACHE_DIR = os.environ.get("CACHE_DIR", "/root/cache")
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
          'Contusion', 'Fracture']
# the three focal findings: measured to prefer top-k pooling over the mean
# v4 teacher AUC against the 58 gold studies, per label (measured).
TEACHER_AUC = {"ACL": 0.9871, "MCL": 0.9683, "Medial Meniscus": 0.9483,
               "Lateral Meniscus": 0.8789, "Medial OA": 0.9318, "Lateral OA": 0.8327,
               "PF OA": 0.9015, "Effusion": 0.8770, "Synovitis": 0.7903,
               "Baker's": 0.9438, "Contusion": 0.8596, "Fracture": 0.7931}
_lw = np.array([2 * TEACHER_AUC[c] - 1 for c in LABELS], dtype=np.float32)
LABEL_W = _lw / _lw.mean()

PAIRCH = False   # set from --pairch before the model is built
FOCAL = {"Baker's", "Contusion", "Fracture"}
FOCAL_MASK = [c in FOCAL for c in LABELS]


MED_COLS, LAT_COLS = [2, 4], [3, 5]     # Medial/Lateral Meniscus, Medial/Lateral OA
_CS_CACHE = {}


# ---------------------------------------------------------------- --corsplit (coronal half-split)
# WHICH COLUMN HALF OF A CANONICAL CORONAL SLICE IS MEDIAL - the reasoning, from build_cache.py:
#  1. Patient +x is the patient's LEFT (DICOM LPS; build_cache.py:163, geo_side docstring).
#  2. read_one() never reorients pixels by ImageOrientationPatient: it centre-crops and
#     resizes the stored pixel array as-is (build_cache.py:358-384, crop at :375). So image
#     column index increases along the IOP row direction iop[0:3].
#  3. --lat-norm mirrors RIGHT knees' coronal/axial slices horizontally (build_cache.py:659-668,
#     `im[:, ::-1]` at :668), so the canonical orientation is a LEFT knee as acquired.
#  4. For a LEFT knee the medial compartment faces the midline = patient's right = -x
#     (the same rule SKM-TEA verified 40/40 for the sagittal axis: "LEFT knee, medial side is
#     DECREASING DICOM x"). With the standard coronal row direction +x (IOP 1\0\0\...),
#     low column = -x = MEDIAL, i.e. the image-LEFT half (columns < W/2) is medial. This matches
#     CompartmentPool.LEFT_IS_MEDIAL = True below.
#  NOT CERTAIN: step 4 assumes iop[0] ~ +1. Nothing in build_cache.py forces it - geo_side even
#  handles rx_ ~ -1 explicitly (`abs(rx_) > 0.9`, build_cache.py:177-183) - so a scanner that
#  writes coronal rows along -x would put medial on the right. Hence --corsplit-medial
#  left|right (default left). A header census of iop[0] sign on coronal series settles it.
CORSPLIT_CLASH = ("focal_k", "train_kper", "posw", "compsplit", "use_qhead", "planepool",
                  "clsattn", "attn", "seq", "gru", "compool", "is3d")


def coronal_slots(meta, n_slots):
    """Slot indices holding CORONAL series, from the cache manifest. build_cache.py:509-514
    orders slots plane-major: [(pl, f) for pl in planes for f in (1, 0)] with --contrast-slots,
    so planes Sagittal,Coronal,Axial x fluid (1, 0) -> slots 2 (Cor|f1) and 3 (Cor|f0).
    Prefers meta["slots"] ("Coronal|f1", ...; build_cache.py:681); falls back to meta["planes"]."""
    slots = meta.get("slots")
    if slots:
        if len(slots) != n_slots:
            raise ValueError(f"meta slots {slots} do not match n_slots={n_slots}")
        idx = [i for i, sl in enumerate(slots) if str(sl).split("|")[0] == "Coronal"]
    else:
        planes = list(meta.get("planes", ["Sagittal", "Coronal", "Axial"]))
        if "Coronal" not in planes or n_slots % len(planes):
            raise ValueError(f"cannot place Coronal in planes={planes} n_slots={n_slots}")
        k = n_slots // len(planes)
        c = planes.index("Coronal")
        idx = list(range(c * k, (c + 1) * k))
    if not idx:
        raise ValueError(f"no Coronal slot in cache meta (slots={slots})")
    return idx


def sideaux_target(n, n_slots, device):
    """(target, mask) over the n slice positions. For the two SAGITTAL slots the target is
    the slice's normalised position 0..1 along the medial-lateral axis (0 = most medial),
    which is exact after --lat-norm. Coronal and axial encode side in-plane, not by index,
    so they are masked out of this loss entirely."""
    per = max(n // n_slots, 1)
    t = torch.zeros(n, device=device)
    msk = torch.zeros(n, device=device)
    pos = torch.linspace(0, 1, per, device=device)
    for si in (0, 1):
        t[si * per:(si + 1) * per] = pos
        msk[si * per:(si + 1) * per] = 1.0
    return t, msk


def compsplit_mask(n, n_slots, device):
    """(n, n_labels) keep-mask. Only SAGITTAL slots carry the medial-lateral axis in the
    slice index; slots are [(plane, fluid) for plane in Sag,Cor,Ax for fluid in (1,0)],
    so slots 0 and 1 are the sagittal pair. Coronal and axial encode side IN-PLANE and are
    left fully on for every label."""
    key = (n, n_slots, str(device))
    if key in _CS_CACHE:
        return _CS_CACHE[key]
    per = max(n // n_slots, 1)
    M = torch.ones(n, len(LABELS), device=device)
    for si in (0, 1):
        b, h = si * per, per // 2
        for c in MED_COLS:
            M[b + h:b + per, c] = 0.0
        for c in LAT_COLS:
            M[b:b + h, c] = 0.0
    _CS_CACHE[key] = M
    return M


# ---------------------------------------------------------------- --compswap (compartment-swap mixing)
# RSNA 2024 lumbar 2nd place "mix sides across patients", transposed to the knee: study i keeps
# its MEDIAL compartment, takes study j's LATERAL compartment, and the labels follow the anatomy.
#
# ORIENTATION, per plane (after --lat-norm every study is a LEFT knee as acquired):
#  * CORONAL: image-LEFT column half = MEDIAL (coronal rows run along +x on 574/575 RSNA coronal
#    series; see the --corsplit comment above). --corsplit-medial sets it, shared with --corsplit.
#  * SAGITTAL: NOT determinable from the code. build_cache.py:622 orders slices by
#    InstanceNumber, never by position, and InstanceNumber direction is protocol-dependent: all
#    three SKM-TEA (GE) sagittal DESS series read here run InstanceNumber 1 -> N along DECREASING
#    x, and kernels/atlas*/infer_kaggle.py med_low() reads the sense per study from IPP for the same
#    reason. The cache keeps no IPP, so the lateral half of a cached sagittal stack is unknown per
#    study. Hence --compswap-sag drop (default: mixed rows lose their sagittal slots via the mask),
#    or high|low once a header census proves the direction is uniform.
#  * AXIAL: same image convention as coronal in principle (rows +x), but no census exists, so
#    --compswap-ax defaults to drop; swap uses the coronal column rule; keep leaves i's full axial
#    in place (i's lateral compartment shown under j's lateral labels = label noise).
# LABELS: Lateral Meniscus / Lateral OA (+ weights) from j; Medial Meniscus / Medial OA and MCL
# (a medial structure) stay with i. Weight 0 on mixed rows for labels that are not
# compartment-local or could sit in either half: ACL (intercondylar, straddles the split), PF OA
# (patellofemoral, central), Effusion / Synovitis (joint-wide), Baker's (posteromedial but can
# extend; follows the brief), Contusion / Fracture (either compartment).
COMPSWAP_ZERO = ["ACL", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
COMPSWAP_ZERO_COLS = [LABELS.index(c) for c in COMPSWAP_ZERO]
COMPSWAP_CLASH = ("mlflip", "pairch", "nvum", "report_clip")


def plane_slots(meta, n_slots, plane):
    """Slot indices of one plane from the cache manifest (same rule as coronal_slots())."""
    slots = meta.get("slots")
    if slots:
        if len(slots) != n_slots:
            raise ValueError(f"meta slots {slots} do not match n_slots={n_slots}")
        return [i for i, sl in enumerate(slots) if str(sl).split("|")[0] == plane]
    planes = list(meta.get("planes", ["Sagittal", "Coronal", "Axial"]))
    if plane not in planes or n_slots % len(planes):
        return []
    k = n_slots // len(planes)
    c = planes.index(plane)
    return list(range(c * k, (c + 1) * k))


def compswap_partner(b):
    """Derangement of range(b): a cyclic shift of a random permutation, so no row is paired
    with itself (a fixed point would zero 7 labels for nothing). b=2 -> the swap."""
    o = torch.randperm(b)
    p = torch.empty(b, dtype=torch.long)
    p[o] = o.roll(-1)
    return p


def compswap_apply(x, m, y, w, partner, n_slots, sag_slots, cor_slots, ax_slots,
                   sag="drop", ax="drop", cor_medial="left", vflip=None, extra=()):
    """Row i keeps its medial compartment and takes row partner[i]'s lateral one.
    x [b, n, (3,) H, W] slot-major tokens, m [b, n], y/w [b, L]. `extra` are per-token
    [b, n] tensors that travel with the SAGITTAL tokens (e.g. --lm-slice-aux targets).
    vflip [b] is Cached's random vertical-flip state; a partner flipped differently is flipped
    back first so both halves of a mixed slice share one orientation. A slot absent in either
    study is dropped (mask 0) for the mixed row. A row whose mask would become empty is left
    unmixed. Returns x, m, y, w, extra, mixed[b] (new tensors; inputs untouched)."""
    b, n = x.shape[:2]
    per = n // n_slots
    Wd = x.shape[-1]
    lat_cols = slice(Wd // 2, Wd) if cor_medial == "left" else slice(0, Wd // 2)
    h = per // 2
    x2, m2, y2, w2 = x.clone(), m.clone(), y.clone(), w.clone()
    ex2 = [e.clone() for e in extra]
    mixed = torch.zeros(b, dtype=torch.bool)
    for i in range(b):
        j = int(partner[i])
        if j == i:
            continue
        xj = x[j]
        if vflip is not None and bool(vflip[i]) != bool(vflip[j]):
            xj = torch.flip(xj, dims=[-2])
        xi, mi = x[i].clone(), m[i].clone()
        eis = [e[i].clone() for e in extra]
        for s in sag_slots:
            tok = slice(s * per, (s + 1) * per)
            if sag == "drop":
                mi[tok] = 0
                continue
            lat = (slice(s * per + per - h, (s + 1) * per) if sag == "high"
                   else slice(s * per, s * per + h))
            xi[lat] = xj[lat]
            for ei, e in zip(eis, extra):
                ei[lat] = e[j, lat]
            mi[tok] = m[i, tok] * m[j, tok]
        for s, mode in [(s, "swap") for s in cor_slots] + [(s, ax) for s in ax_slots]:
            tok = slice(s * per, (s + 1) * per)
            if mode == "drop":
                mi[tok] = 0
            elif mode == "swap":
                xi[tok, ..., lat_cols] = xj[tok, ..., lat_cols]
                mi[tok] = m[i, tok] * m[j, tok]
        if float(mi.sum()) <= 0:
            continue
        x2[i] = xi * mi.reshape(-1, *([1] * (xi.dim() - 1))).to(xi.dtype)
        m2[i] = mi
        for e2, ei in zip(ex2, eis):
            e2[i] = ei
        y2[i, LAT_COLS] = y[j, LAT_COLS]
        w2[i, LAT_COLS] = w[j, LAT_COLS]
        w2[i, COMPSWAP_ZERO_COLS] = 0.0
        mixed[i] = True
    return x2, m2, y2, w2, ex2, mixed


def load_lm_slice_targets(path, studies, rows, n_cached):
    """--lm-slice-targets npz (oai_pull/make_lm_slice_targets.py): ids [N], t [N, 2, S] 0/1 per
    cached SAGITTAL slice (slot order = the cache's two sagittal slots), ok [N, 2] slot validity.
    Returns (t [len(rows), 2, S] float32, ok [len(rows), 2] float32); unmatched rows ok = 0."""
    z = np.load(path, allow_pickle=False)
    ids = [str(s) for s in z["ids"]]
    t, ok = z["t"].astype(np.float32), z["ok"].astype(np.float32)
    if t.shape[-1] != n_cached:
        raise ValueError(f"--lm-slice-targets has {t.shape[-1]} slices per slot, cache has {n_cached}")
    pos = {s: k for k, s in enumerate(ids)}
    T = np.zeros((len(rows), t.shape[1], n_cached), np.float32)
    OK = np.zeros((len(rows), t.shape[1]), np.float32)
    for r, i in enumerate(rows):
        k = pos.get(studies[i])
        if k is not None:
            T[r], OK[r] = t[k], ok[k]
    return T, OK


class Cached(Dataset):
    def __init__(self, arr, mask, rows, y=None, w=None, use=None, train=True, aug=False,
                 aug_mult=1.0, drop_slots=(), in25d=False, dsample=0, pairch=False,
                 ret_vflip=False, lm_t=None, lm_ok=None, lm_slots=(0, 1)):
        self.arr, self.mask, self.rows = arr, mask, rows
        # Opt-in extra outputs, appended after ridx only when asked for (off = old tuple):
        # ret_vflip -> the random vertical-flip bit (--compswap re-aligns partners with it);
        # lm_t/lm_ok -> per-token --lm-slice-aux target and validity over the sagittal slots.
        self.ret_vflip, self.lm_t, self.lm_ok, self.lm_slots = ret_vflip, lm_t, lm_ok, tuple(lm_slots)
        self.y, self.w, self.train, self.aug = y, w, train, aug
        self.aug_mult = aug_mult
        self.dsample = dsample
        self.in25d = in25d
        self.pairch = pairch
        if drop_slots:
            # Zeroing the mask is enough: x is multiplied by it and the pooling divides by
            # the mask sum, so a dropped slot contributes nothing and is not counted.
            self.mask = np.array(self.mask)
            self.mask[:, list(drop_slots)] = 0
        self.n_cached = arr.shape[2]
        self.use = min(use or self.n_cached, self.n_cached)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        block = np.asarray(self.arr[r])                     # [plane, slice, H, W]
        if self.use < self.n_cached:
            if self.train:
                sel = np.sort(np.random.choice(self.n_cached, self.use, replace=False))
            else:
                sel = np.linspace(0, self.n_cached - 1, self.use).round().astype(int)
        else:
            sel = np.arange(self.n_cached)
        if getattr(self, "pairch", False):
            # Slots are [Sag-f1, Sag-f0, Cor-f1, Cor-f0, Ax-f1, Ax-f0]: each plane's
            # fluid-sensitive and non-fluid series sit adjacent. Stack the pair as image
            # channels instead of running them as two independent passes that only meet
            # at the final average - a radiologist compares bright fluid on the T2-like
            # series against the same anatomy on the T1-like one within a single look.
            # ch0 = fluid-sensitive, ch1 = non-fluid, ch2 = their mean (keeps the
            # pretrained 3-channel stem fully used).
            blk = block[:, sel]                       # [slot, use, H, W]
            pair = blk.reshape(3, 2, *blk.shape[1:])  # [plane, fluid-state, use, H, W]
            a_, b_ = pair[:, 0], pair[:, 1]
            mk = self.mask[r].reshape(3, 2)
            # if one of the pair is absent, fall back to the present one in both channels
            a2 = np.where(mk[:, 0:1, None, None] > 0, a_, b_)
            b2 = np.where(mk[:, 1:2, None, None] > 0, b_, a_)
            stack = np.stack([a2, b2, (a2.astype(np.float32) + b2) / 2], axis=2)
            x = torch.from_numpy(
                stack.reshape(-1, 3, *stack.shape[3:])).float().div_(255)
            m = torch.from_numpy(mk.max(1)).repeat_interleave(self.use).float()
            x = x * m.reshape(-1, 1, 1, 1)
            if self.train and self.aug:
                pass
            if self.y is None:
                return x, m
            return (x, m, torch.from_numpy(self.y[i]), torch.from_numpy(self.w[i]),
                    torch.tensor(i, dtype=torch.long))
        if self.in25d:
            # 2.5D: each selected slice becomes [prev, cur, next] from the FULL cached
            # stack (adjacency in cache order, clamped at the ends), so the net sees
            # through-plane context. Channel dim carries the triplet.
            lo = np.clip(sel - 1, 0, self.n_cached - 1)
            hi = np.clip(sel + 1, 0, self.n_cached - 1)
            trip = np.stack([block[:, lo], block[:, sel], block[:, hi]], axis=2)
            x = torch.from_numpy(
                trip.reshape(-1, 3, *trip.shape[3:])).float().div_(255)
        else:
            block = block[:, sel]
            x = torch.from_numpy(block.reshape(-1, *block.shape[2:])).float().div_(255)
        m = torch.from_numpy(self.mask[r]).repeat_interleave(self.use)
        if self.dsample:
            x = torch.nn.functional.interpolate(
                x[:, None] if x.dim() == 3 else x, size=(self.dsample, self.dsample),
                mode="bilinear", align_corners=False)
            x = x[:, 0] if x.shape[1] == 1 else x
        x = x * m.reshape(-1, *([1] * (x.dim() - 1)))       # zero out absent planes
        vflip = 0
        if self.train and self.aug:
            import torchvision.transforms.functional as TVF
            k = self.aug_mult
            ang = float((torch.rand(1) - 0.5) * 16 * k)
            sc = float(1 + (torch.rand(1) - 0.5) * 0.16 * k)
            tx = int(float((torch.rand(1) - 0.5) * 0.10 * k) * x.shape[-1])
            ty = int(float((torch.rand(1) - 0.5) * 0.10 * k) * x.shape[-2])
            x = TVF.affine(x, angle=ang, translate=[tx, ty], scale=sc, shear=[0.0, 0.0])
            if np.random.rand() < 0.5:
                x = (x * (1 + (torch.rand(1) - 0.5) * 0.2 * k)).clamp(0, 1)
        elif self.train and np.random.rand() < 0.5:
            x = torch.flip(x, dims=[-2])
            vflip = 1
        if self.y is None:
            return x, m
        extras = ()
        if self.ret_vflip:
            extras += (torch.tensor(vflip, dtype=torch.long),)
        if self.lm_t is not None:
            nsl = self.mask.shape[1]
            lt = torch.zeros(nsl * self.use)
            lk = torch.zeros(nsl * self.use)
            for k_, s_ in enumerate(self.lm_slots):
                lt[s_ * self.use:(s_ + 1) * self.use] = torch.from_numpy(self.lm_t[i, k_, sel])
                lk[s_ * self.use:(s_ + 1) * self.use] = float(self.lm_ok[i, k_])
            extras += (lt, lk * m.float())
        return (x, m, torch.from_numpy(self.y[i]), torch.from_numpy(self.w[i]),
                torch.tensor(i, dtype=torch.long)) + extras



class ClsAttnPool(nn.Module):
    """Per-label spatial pooling. Global average pooling reads the whole 12x12 map the
    same way for every finding, but a fracture line occupies one or two cells while an
    effusion spans most of them. One learned query per label produces its own attention
    map over the spatial positions, so each label pools the region it needs from shared
    features. Slice-axis pooling (top-k for focal labels) already does this one axis over
    and is worth +0.004 on the LB; this is the in-plane analogue."""

    def __init__(self, dim, n_labels):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_labels, dim) * dim ** -0.5)
        self.w = nn.Parameter(torch.randn(n_labels, dim) * dim ** -0.5)
        self.b = nn.Parameter(torch.zeros(n_labels))
        self.scale = dim ** -0.5

    def forward(self, fmap):                       # [N, C, H, W] -> [N, n_labels]
        n, c, h, w = fmap.shape
        f = fmap.flatten(2).transpose(1, 2)        # [N, HW, C]
        att = torch.softmax((f @ self.q.t()) * self.scale, dim=1)   # [N, HW, L]
        pooled = torch.einsum("nsc,nsl->nlc", f, att)               # [N, L, C]
        return (pooled * self.w).sum(-1) + self.b



class CompartmentPool(nn.Module):
    """Medial/lateral compartment-restricted mean pooling for Meniscus and OA labels.

    Census 2026-08-23: averaged the coronal mid-slice across all 4,248 studies with a
    coronal series. The knee remains visible even averaged across every patient (our
    centroid-crop + laterality-normalisation genuinely aligns studies), and the
    intensity-weighted horizontal center sits at x=190.3 of 384 - 1.7px from the true
    geometric center. So a fixed left/right split of the coronal feature map at the
    midline is a fair medial/lateral compartment split with no per-study localisation.

    Unlike ClsAttnPool (which lost, -0.003 to -0.004: a LEARNED spatial split found
    nothing), this is a fixed geometric prior motivated by the corpus-level alignment
    measurement above, applied only to the 5 labels with real medial/lateral anatomy.
    Everything else keeps the existing masked-mean/top-k pooling untouched.

    HARD SPLIT RESULT 2026-08-24: replacing the global feature with one compartment LOST
    on both folds - f0 0.8734 vs 0.8819 baseline, f1 0.8703 vs 0.8774, mean -0.008. The
    geometry was fine (see the census above); the pooling was too aggressive. Restricting
    a label to its own half discards the CONTRALATERAL compartment, but medial-vs-lateral
    comparison is exactly how OA and meniscus tears are read - the opposite compartment is
    the reference standard. Hence the soft variant (--compool-soft): concatenate global +
    medial + lateral and let the head weigh them, rather than substituting one for global.
    """

    COMPARTMENT_LABELS = {
        "Medial Meniscus": "medial", "Lateral Meniscus": "lateral",
        "Medial OA": "medial", "Lateral OA": "lateral",
    }
    # laterality-normalised orientation: after our lat_norm flip, patient-left is always
    # on the same side of the crop, matching medial/lateral consistently across studies.
    # Medial compartment = image-left half of the coronal crop (x < midline).
    LEFT_IS_MEDIAL = True
    CORONAL_SLOTS = (2, 3)   # Cor-FS, Cor-T1, per the SLOT_PRIOR ordering above

    def __init__(self, dim, n_slots, soft=False):
        super().__init__()
        self.soft = soft
        want = [i for i, c in enumerate(LABELS) if c in self.COMPARTMENT_LABELS]
        self.register_buffer("comp_idx", torch.tensor(want, dtype=torch.long))
        side = [self.COMPARTMENT_LABELS[LABELS[i]] for i in want]
        left = [(s == "medial") == self.LEFT_IS_MEDIAL for s in side]
        self.register_buffer("want_left", torch.tensor(left, dtype=torch.bool))
        self.n_slots = n_slots

    def forward(self, fmap, mean_p, n_per_slot):
        """fmap: [B*n, C, H, W] backbone output for every slice in every slot, in slot
        order. mean_p: [B, n_labels] the existing whole-crop prediction, used unchanged
        for every label this module does not touch. Returns mean_p with the 4
        compartment labels overwritten by their compartment-restricted counterpart."""
        w = fmap.shape[-1]
        left_half = fmap[..., : w // 2].mean(dim=(2, 3))    # [B*n, C]
        right_half = fmap[..., w // 2:].mean(dim=(2, 3))    # [B*n, C]
        b = mean_p.shape[0]
        left_half = left_half.reshape(b, self.n_slots, n_per_slot, -1)
        right_half = right_half.reshape(b, self.n_slots, n_per_slot, -1)
        cor = list(self.CORONAL_SLOTS)
        left_cor = left_half[:, cor].mean(dim=(1, 2))       # [B, C]
        right_cor = right_half[:, cor].mean(dim=(1, 2))     # [B, C]
        return left_cor, right_cor


class Net(nn.Module):
    def __init__(self, arch="resnet18", attn=False, init=None, gray_fold=False, focal_k=0, seq=False, in25d=False, qhead=False, maxpool=False, planepool=False, n_slots=6, gru=False, msc=False, clsattn=False, compool=False, compool_soft=False):
        super().__init__()
        import torchvision
        m = (None if arch.startswith(("dinov2", "mricore", "resnetblur", "r3d", "r2plus1d", "mc3", "timm:"))
             else getattr(torchvision.models, arch)(
                 weights=None if init else "IMAGENET1K_V1"))
        if init:
            # RadImageNet: 1.35M CT/MR/US images instead of ImageNet photos.
            # Keras ResNet50 is v1, so the stride sits on the leading 1x1 of each
            # downsampling block; torchvision is v1.5. Move it before loading or
            # the net silently subsamples at the wrong depth.
            ck = torch.load(init, map_location="cpu")
            if ck.get("keras_v1_strides"):
                for layer in (m.layer2, m.layer3, m.layer4):
                    layer[0].conv1.stride, layer[0].conv2.stride = (2, 2), (1, 1)
            miss, unexp = m.load_state_dict(ck["state_dict"], strict=False)
            miss = [k for k in miss if not k.startswith("fc.")]
            assert not miss and not unexp, (miss[:4], unexp[:4])
            print(f"init from {init}", flush=True)
        self.use_qhead = qhead
        if arch.startswith("r3d") or arch.startswith("r2plus1d"):
            # True 3D convolution over the slice axis - the one architecture family never
            # tried here. Each SLOT becomes a video: [3, slices, H, W]. Unlike 2.5D (which
            # only gave the first conv a 3-slice window) or the cross-slice transformer
            # (which mixed pooled features), this convolves through depth at every layer.
            import torchvision
            import torchvision.models.video as tvv
            m = getattr(tvv, arch)(weights="KINETICS400_V1")
            dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
            self.head = nn.Linear(dim, len(LABELS))
            self.dim = dim
            self.attn = attn
            self.focal_k = 0
            self.seq = False
            self.expand3 = False
            self.is_dino = False
            self.is3d = True
            self.n_slots = n_slots
            self.use_qhead = False
            self.maxpool = False
            self.planepool = False
            return
        self.is3d = False
        if arch.startswith("dinov2"):
            # Fine-tuned DINOv2, not frozen features. The public ablation that moves the
            # needle here (sadamtorres) reports +0.171 Medial Meniscus, +0.118 MCL,
            # +0.113 ACL from fine-tuning at UNCHANGED 224px - i.e. domain adaptation,
            # not resolution. Those are exactly the labels where our resnet trails its
            # own teacher (MCL -0.121, menisci -0.04).
            #
            # Patch size 14, so the cache image size must be a multiple of 14 (336 = 14*24).
            # Features = CLS + patch-mean concatenated, which the public work finds beats
            # either alone. Grey input is expanded to 3 channels to keep the pretrained
            # patch embedding intact.
            # NB: do NOT "import torch.hub" here - that binds `torch` as a LOCAL name
            # for the whole of __init__, so torch.load() in the RadImageNet --init branch
            # above dies with UnboundLocalError before this branch is ever reached. It
            # silently killed every --init run. torch is already imported at module level.
            m = torch.hub.load("facebookresearch/dinov2", arch)
            dim = m.embed_dim * 2
            self.backbone = m
            self.head = nn.Linear(dim, len(LABELS))
            self.clsattn = None
            self.dim = dim
            self.attn = attn
            self.focal_k = focal_k
            self.seq = seq
            self.expand3 = not PAIRCH
            self.is_dino = True
            if qhead:
                self._build_qhead()
            if seq:
                layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8,
                                                   dim_feedforward=dim * 2, dropout=0.1,
                                                   batch_first=True, norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.pos = nn.Parameter(torch.randn(1, 512, dim) * 0.02)
            return
        if arch.startswith("timm:"):
            # Generic timm backbone under OUR pipeline - built for the CoAtNet screen
            # (timm:coatnet_rmlp_2_rw_384). Same cache, pooling, head and folds as every
            # other lineage, so the delta vs wBCNX isolates the ARCHITECTURE alone. The
            # 3-channel pretrained stem is kept; grey input expands at forward like the
            # convnext branch (expand3). num_classes=0 returns pooled features.
            import timm
            m = timm.create_model(arch[5:], pretrained=True, num_classes=0)
            # 48 slices/study through a 73M hybrid at 384px overflows 24GB even at
            # batch 1 (attention activations). Checkpointing recomputes them in the
            # backward pass: ~3-4x less memory for ~30% slower epochs. Measured
            # necessary for coatnet_rmlp_2; harmless no-op if unsupported.
            if hasattr(m, "set_grad_checkpointing"):
                m.set_grad_checkpointing(True)
                print("grad checkpointing ON", flush=True)
            dim = m.num_features
            self.backbone = m
            self.head = nn.Linear(dim, len(LABELS))
            self.dim = dim
            self.attn = attn
            self.focal_k = focal_k
            self.seq = seq
            self.expand3 = not PAIRCH
            self.is_dino = False
            self.use_qhead = False
            self.maxpool = False
            self.planepool = False
            return
        if arch == "resnetblur18":
            # ELNet ablation: BlurPool anti-aliased downsampling gave meniscus
            # 0.848->0.904, ACL 0.923->0.960 at ~1.1k studies. timm's resnetblur18 is
            # resnet18 with blur_pool on the stride-2 paths; in_chans=1 folds the
            # pretrained RGB stem to grey, num_classes=0 returns pooled features.
            import timm
            from timm.layers import BlurPool2d
            # timm has no pretrained resnetblur18; standard resnet18 weights +
            # aa_layer injects the (weight-free) blur filters into stride-2 paths.
            m = timm.create_model("resnet18", pretrained=True, in_chans=1,
                                  num_classes=0, aa_layer=BlurPool2d)
            dim = m.num_features
            self.backbone = m
            self.head = nn.Linear(dim, len(LABELS))
            self.dim = dim
            self.attn = attn
            self.focal_k = focal_k
            self.seq = seq
            self.expand3 = False
            self.is_dino = False
            self.use_qhead = False
            self.maxpool = False
            self.planepool = False
            return
        if arch == "mricore":
            # MRI-CORE ViT-B/16 (Duke, DINO-pretrained on MRI slices instead of web
            # photos). Same fine-tune recipe as the dinov2 branch. Weights converted
            # offline to flat timm naming with pos_embed pre-interpolated to 336px
            # (patch 16, 336 = 16*21) - /workspace/mricore_vitb_336.pt.
            import timm
            m = timm.create_model("vit_base_patch16_224", img_size=336, num_classes=0)
            ck = torch.load("/workspace/mricore_vitb_336.pt", map_location="cpu")
            miss, unexp = m.load_state_dict(ck, strict=False)
            assert not miss and not unexp, (miss[:4], unexp[:4])
            print("init from mricore_vitb_336.pt", flush=True)
            dim = m.embed_dim * 2
            self.backbone = m
            self.head = nn.Linear(dim, len(LABELS))
            self.dim = dim
            self.attn = attn
            self.focal_k = focal_k
            self.seq = seq
            self.expand3 = True
            self.is_dino = True
            if qhead:
                self._build_qhead()
            return
        self.use_qhead = qhead
        self.is_dino = False
        if not hasattr(m, "fc"):
            # efficientnet/convnext have no .fc and no .conv1 - their stem lives inside
            # .features and the head inside .classifier. Rather than surgically rebuilding
            # a 1-channel stem for each family (easy to get silently wrong), keep the
            # pretrained 3-channel stem and expand the grey input to 3 channels at forward.
            mods = dict(m.named_modules())
            lin = [k for k, v in mods.items() if isinstance(v, nn.Linear)][-1]
            dim = mods[lin].in_features
            parent = m if "." not in lin else mods[lin.rsplit(".", 1)[0]]
            setattr(parent, lin.rsplit(".", 1)[-1], nn.Identity())
            self.backbone = m
            self.msc = msc
            self.n_slots = n_slots
            self._last = {}
            with torch.no_grad():
                probe_c = m.features(torch.zeros(1, 3, 64, 64)).shape[1]
            self.clsattn = ClsAttnPool(probe_c, len(LABELS)) if clsattn else None
            if clsattn:
                m.features.register_forward_hook(
                    lambda _m, _i, o: self._last.__setitem__("f", o))
            self.compool = CompartmentPool(probe_c, n_slots, soft=compool_soft) if compool else None
            if compool:
                self._comp = {}
                m.features.register_forward_hook(
                    lambda _m, _i, o: self._comp.__setitem__("f", o))
            if msc:
                # Multi-scale concat. Measured on convnext_tiny at 384px: the final map is
                # 12x12, i.e. 10.8mm per cell - wider than a meniscus, so a tear or fracture
                # line cannot be localised there. features[5] is 24x24 = 5.4mm/cell. Pool
                # both and let the head see fine and coarse together. This is the mechanism
                # behind MRPyrNet's published FPN gains (+0.021 ACL, +0.046 meniscus), but
                # motivated here by our own receptive-field measurement.
                fine_idx = 5 if len(m.features) > 6 else len(m.features) - 2
                self._fine = {}
                m.features[fine_idx].register_forward_hook(
                    lambda _m, _i, o: self._fine.__setitem__("f", o))
                with torch.no_grad():
                    probe = m.features[:fine_idx + 1](torch.zeros(1, 3, 64, 64))
                dim = dim + probe.shape[1]
            self.head = nn.Linear(dim, len(LABELS))
            if compool and compool_soft:
                # Head sees global + medial + lateral instead of global being REPLACED by
                # one compartment (which lost, see CompartmentPool). Initialised to
                # reproduce the baseline head exactly - compartment columns zeroed, global
                # columns and bias copied from the corresponding head rows - so the run
                # starts AT the baseline and can only add compartment signal on top.
                nc = int(self.compool.comp_idx.numel())
                self.comp_head = nn.Linear(dim + 2 * probe_c, nc)
                with torch.no_grad():
                    self.comp_head.weight.zero_()
                    self.comp_head.weight[:, :dim] = self.head.weight[self.compool.comp_idx]
                    self.comp_head.bias.copy_(self.head.bias[self.compool.comp_idx])
            self.dim = dim
            self.attn = attn
            self.focal_k = focal_k
            self.seq = seq
            self.expand3 = True
            if qhead:
                self._build_qhead()
            if seq:
                layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8,
                                                   dim_feedforward=dim * 2, dropout=0.1,
                                                   batch_first=True, norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.pos = nn.Parameter(torch.randn(1, 512, dim) * 0.02)
            return
        self.expand3 = False
        self.in25d = in25d
        dim = m.fc.in_features
        # Our input is single-channel. Replacing conv1 with a fresh Conv2d throws
        # away the pretrained first layer - the one place domain pretraining is
        # most concrete. Summing over RGB is what a grey image replicated into
        # three channels would have produced, so it preserves the filters exactly.
        if in25d:
            # three REAL input channels (prev/cur/next slice) - the pretrained RGB stem
            # is the right shape already and its centre-surround filters transfer.
            pass
        else:
            rgb = m.conv1.weight.data.clone()
            m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            if gray_fold:
                m.conv1.weight.data = rgb.sum(1, keepdim=True)
        self.maxpool = maxpool
        self.planepool = planepool
        self.n_slots = n_slots
        if maxpool:
            # drop the final avgpool: we pool the spatial map ourselves so both
            # average and max are available.
            self.backbone = nn.Sequential(*list(m.children())[:-2])
            dim = dim * 2
        else:
            self.backbone = nn.Sequential(*list(m.children())[:-1])
        head_in = dim * (n_slots if planepool else 1)
        self.head = nn.Linear(head_in, len(LABELS))
        self.dim = dim
        self.attn = attn
        self.focal_k = focal_k
        self.seq = seq
        self.gru = None
        if gru:
            # Bidirectional GRU over the slice sequence (slot-major order), hidden size
            # dim//2 so the concatenated output keeps width dim and the head is unchanged.
            self.gru = nn.GRU(dim, dim // 2, num_layers=1, batch_first=True,
                              bidirectional=True)
        if qhead:
            self._build_qhead()
        if seq:
            layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 2,
                                               dropout=0.1, batch_first=True,
                                               norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            # Without this the attention is permutation-invariant: it could check whether a
            # signal is corroborated somewhere in the bag, but not whether the corroboration
            # is on ADJACENT slices - which is the whole anatomical argument (a tear spans
            # neighbouring slices, an artifact does not). One learned embedding per (slot,
            # slice-within-slot) position, small init so it perturbs rather than dominates.
            self.pos = nn.Parameter(torch.randn(1, 512, dim) * 0.02)
        if attn:
            # one learned query per diagnosis: each label attends over the images
            # it cares about instead of averaging all 96 equally. The mean-pool
            # is the known information bottleneck of this architecture.
            self.q = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
            self.proj = nn.Linear(dim, dim)
            self.cls = nn.Parameter(torch.zeros(len(LABELS), dim))
            nn.init.normal_(self.cls, std=0.02)

    # Which slots each finding actually lives in. Slot order is plane-major over
    # (Sagittal, Coronal, Axial) x (fluid-sensitive, non-fluid): 0 Sag-FS, 1 Sag-T1,
    # 2 Cor-FS, 3 Cor-T1, 4 Ax-FS, 5 Ax-T1. This is a soft, LEARNED bias on the attention
    # logits, not a mask - the model can override it, it just does not start from scratch.
    SLOT_PRIOR = {
        "ACL":              (0, 1),        # sagittal, best seen on the oblique sag
        "MCL":              (2, 3),        # coronal - the classic MCL view
        "Medial Meniscus":  (0, 1, 2),     # sag body/horns + coronal
        "Lateral Meniscus": (0, 1, 2),
        "Medial OA":        (2, 3),        # coronal joint space
        "Lateral OA":       (2, 3),
        "PF OA":            (4, 5, 0),     # axial patellofemoral + sagittal trochlea
        "Effusion":         (0, 2, 4),     # any fluid-sensitive
        "Synovitis":        (0, 2, 4),
        "Baker's":          (4, 0),        # posterior: axial + sagittal, fluid-sensitive
        "Contusion":        (0, 2, 4),     # marrow oedema needs fat-sat
        "Fracture":         (0, 2, 4),
    }

    def _build_qhead(self, n_slots=6):
        # one learned query per finding; single-head cross-attention over slice features.
        d = self.dim
        self.q = nn.Parameter(torch.randn(len(LABELS), d) * (d ** -0.5))
        self.kproj = nn.Linear(d, d)
        self.vproj = nn.Linear(d, d)
        self.qout = nn.Parameter(torch.zeros(len(LABELS), d))
        nn.init.normal_(self.qout, std=0.02)
        prior = torch.zeros(len(LABELS), n_slots)
        for i, c in enumerate(LABELS):
            for sl in self.SLOT_PRIOR.get(c, ()):
                prior[i, sl] = 1.0
        self.slot_prior = nn.Parameter(prior)
        self.n_slots = n_slots


    def _apply_compool(self, base_pred, gfeat, b, n):
        if getattr(self, "compool", None) is None:
            return base_pred
        fmap = self._comp["f"]                             # [b*n, C, H, W]
        left_cor, right_cor = self.compool(fmap, base_pred, n // self.n_slots)
        comp_pred = base_pred.clone()
        idx = self.compool.comp_idx
        if getattr(self, "comp_head", None) is not None:
            # soft: keep the whole-crop view AND both compartments, let the head weigh them.
            comp_pred[:, idx] = self.comp_head(torch.cat([gfeat, left_cor, right_cor], 1))
            return comp_pred
        left_logit = self.head(left_cor)                   # [b, n_labels]
        right_logit = self.head(right_cor)
        want_left = self.compool.want_left
        chosen = torch.where(want_left[None], left_logit[:, idx], right_logit[:, idx])
        comp_pred[:, idx] = chosen
        return comp_pred

    def _corsplit_expand(self, x, m):
        """--corsplit token rewrite, in the forward pass (the cache is untouched).
        x [b, n, H, W] (or [b, n, 3, H, W] for in25d), slot-major tokens (token = slot*per + j).
        Returns x', m' with tokens [non-coronal (original order), MEDIAL-half coronal,
        LATERAL-half coronal]; each half is resized back to the full HxW with the same
        bilinear F.interpolate that Cached's --dsample uses. Also records the layout."""
        b, n = x.shape[:2]
        ns = self.n_slots
        if n % ns:
            raise ValueError(f"--corsplit: {n} tokens not divisible by n_slots={ns}")
        per = n // ns
        cs = set(int(c) for c in self.corsplit_slots)
        cor = [t for t in range(n) if t // per in cs]
        nc = [t for t in range(n) if t // per not in cs]
        dev = x.device
        cor_i = torch.tensor(cor, device=dev, dtype=torch.long)
        nc_i = torch.tensor(nc, device=dev, dtype=torch.long)
        xc = x.index_select(1, cor_i)                       # [b, ncor, ..., H, W]
        H, W = xc.shape[-2:]
        left, right = xc[..., : W // 2], xc[..., W // 2:]
        med, lat = (left, right) if self.corsplit_medial == "left" else (right, left)

        def _full(t):
            sh = t.shape
            t = F.interpolate(t.reshape(-1, 1, sh[-2], sh[-1]), size=(H, W),
                              mode="bilinear", align_corners=False)
            return t.reshape(*sh[:-1], W)
        xn = torch.cat([x.index_select(1, nc_i), _full(med), _full(lat)], dim=1)
        mc = m.index_select(1, cor_i)
        mn = torch.cat([m.index_select(1, nc_i), mc, mc], dim=1)
        k, c = len(nc), len(cor)
        gm = torch.cat([torch.ones(k + c, device=dev), torch.zeros(c, device=dev)])
        gl = torch.cat([torch.ones(k, device=dev), torch.zeros(c, device=dev),
                        torch.ones(c, device=dev)])
        self._cs = {"n": n, "nc": nc_i, "cor": cor_i, "k": k, "c": c,
                    "gmed": gm.to(m.dtype), "glat": gl.to(m.dtype)}
        return xn, mn

    def _corsplit_check(self):
        bad = [f_ for f_ in CORSPLIT_CLASH if getattr(self, f_, None) not in (None, False, 0, 0.0)]
        if bad:
            raise ValueError("--corsplit handles only the masked-mean pooling path; it cannot be "
                             "combined with: " + ", ".join(bad))
        if PAIRCH:
            raise ValueError("--corsplit cannot be combined with --pairch (3-plane token layout)")
        if self.corsplit_medial not in ("left", "right"):
            raise ValueError(f"corsplit_medial must be left|right, got {self.corsplit_medial!r}")

    def forward(self, x, m):
        if getattr(self, "corsplit", False):
            self._corsplit_check()
            x, m = self._corsplit_expand(x, m)
        b, n = x.shape[:2]
        if getattr(self, "is3d", False):
            ns = self.n_slots
            per = max(n // ns, 1)
            v = x.reshape(b * ns, 1, per, *x.shape[2:]).expand(-1, 3, -1, -1, -1)
            f = self.backbone(v).reshape(b, ns, self.dim)
            sm = m.reshape(b, ns, per).amax(2)[..., None]
            return self.head((f * sm).sum(1) / sm.sum(1).clamp(min=1))
        if getattr(self, "in25d", False) or x.dim() == 5:
            # in25d and pairch both arrive as [b, n, 3, H, W] - already 3-channel
            xin = x.reshape(b * n, 3, *x.shape[3:])
        else:
            xin = x.reshape(b * n, 1, *x.shape[2:])
            if getattr(self, "expand3", False):
                xin = xin.expand(-1, 3, -1, -1)
        if getattr(self, "is_dino", False):
            o = self.backbone.forward_features(xin)
            if isinstance(o, dict):
                f = torch.cat([o["x_norm_clstoken"],
                               o["x_norm_patchtokens"].mean(1)], dim=-1)
            else:
                # timm ViT (mricore): (B, 1+N, C) with CLS first, already normed
                f = torch.cat([o[:, 0], o[:, 1:].mean(1)], dim=-1)
            f = f.reshape(b, n, self.dim)
        elif getattr(self, "maxpool", False):
            fm = self.backbone(xin)                       # [b*n, C, H, W]
            f = torch.cat([fm.mean((2, 3)), fm.amax((2, 3))], dim=1).reshape(b, n, self.dim)
        elif getattr(self, "clsattn", None) is not None:
            self.backbone(xin)                                  # populates self._last
            L = self.clsattn(self._last["f"]).reshape(b, n, len(LABELS))
            w = m[..., None]
            mean_p = (L * w).sum(1) / w.sum(1).clamp(min=1)
            if not self.focal_k:
                return mean_p
            Lm = L.masked_fill(m[:, :, None] < 0.5, -1e4)
            topk = Lm.topk(min(self.focal_k, n), dim=1).values.mean(1)
            sel = torch.tensor(FOCAL_MASK, device=L.device, dtype=torch.bool)
            return torch.where(sel[None], topk, mean_p)
        elif getattr(self, "msc", False):
            coarse = self.backbone(xin)                       # [b*n, C_last]
            fine = self._fine["f"]                            # [b*n, C_fine, H, W]
            f = torch.cat([coarse, fine.mean((2, 3))], dim=1).reshape(b, n, self.dim)
        else:
            f = self.backbone(xin).reshape(b, n, self.dim)
        if self.seq:
            # residual so the settled mean-pool behaviour is the starting point and the
            # encoder only has to learn corrections; absent slots masked out of attention
            f = f + self.encoder(f + self.pos[:, :n], src_key_padding_mask=(m < 0.5))
            f = f * m[..., None]              # keep padded positions from leaking into the mean
        if getattr(self, "sideaux", 0) > 0 or getattr(self, "lm_slice_aux", 0) > 0:
            self._feat = f            # [b, n, dim] - per-slice, before any pooling
            if getattr(self, "corsplit", False):
                # sideaux_target() and the batch mask expect the ORIGINAL slot-major layout.
                # Coronal positions get the half-mean; sideaux masks them out anyway.
                cs = self._cs
                fo = f.new_zeros(b, cs["n"], f.shape[-1])
                fo[:, cs["nc"]] = f[:, :cs["k"]]
                fo[:, cs["cor"]] = 0.5 * (f[:, cs["k"]:cs["k"] + cs["c"]] + f[:, cs["k"] + cs["c"]:])
                self._feat = fo
        if self.focal_k:
            # Per-label pooling. A Baker's cyst, bone contusion or fracture occupies a
            # handful of the 48 slices, so a flat mean dilutes it ~16x - in the forward
            # pass AND in the gradient, so the features never learn it properly. Measured
            # on a mean-pooled model, switching those three to top-k at inference alone
            # recovered +0.029 / +0.020 / +0.004, which says the signal is present and
            # pooling is what discards it. Here the top-k is inside training, so the
            # gradient concentrates on the slices that actually contain the finding.
            # Diffuse targets (effusion, OA, meniscus, ligaments) keep the mean - they
            # lost badly under top-k (effusion 0.940 -> 0.857).
            L = self.head(f)                                   # [b, n, labels]
            L = L.masked_fill(m[:, :, None] < 0.5, -1e4)
            w = m[..., None]
            gfeat = (f * w).sum(1) / w.sum(1).clamp(min=1)
            mean_p = self.head(gfeat)
            k = min(self.focal_k, n)
            topk_p = L.topk(k, dim=1).values.mean(1)
            sel = torch.tensor(FOCAL_MASK, device=f.device, dtype=torch.bool)
            out = torch.where(sel[None], topk_p, mean_p)
            if getattr(self, "compsplit", False):
                # the four compartment labels pool ONLY their own half of the sagittal slots
                cm = compsplit_mask(n, getattr(self, "n_slots", 6), f.device)[None]
                lw = w * cm
                mp2 = (self.head(f) * lw).sum(1) / lw.sum(1).clamp(min=1)
                out = out.clone()
                cols = MED_COLS + LAT_COLS
                out[:, cols] = mp2[:, cols].to(out.dtype)
            return self._apply_compool(out, gfeat, b, n)
        if getattr(self, "gru", None) is not None:
            # order-aware refinement of slice features; masked positions contribute
            # nothing downstream because every pooling path multiplies by m.
            f, _ = self.gru(f)
        if getattr(self, "use_qhead", False):
            # one learned query per finding, cross-attending over the slice features.
            # Absent slots are masked OUT of the softmax rather than contributing zeros,
            # so a study with no axial series redistributes its attention instead of
            # diluting it.
            k = self.kproj(f)                              # [b, n, d]
            v = self.vproj(f)
            att = torch.einsum("ld,bnd->bln", self.q, k) * (self.dim ** -0.5)
            # add the per-(finding, slot) prior. Images are laid out slot-major, so
            # image j belongs to slot j // (n / n_slots).
            per = max(n // self.n_slots, 1)
            att = att + self.slot_prior.repeat_interleave(per, dim=1)[None, :, :n]
            att = att.masked_fill(m[:, None, :] < 0.5, float("-inf"))
            att = att.softmax(-1)
            pooled = torch.einsum("bln,bnd->bld", att, v)  # [b, 12, d]
            return (pooled * self.qout[None]).sum(-1) + self.head.bias
        if getattr(self, "planepool", False):
            # mean WITHIN each slot, then concatenate - the head sees each plane
            # separately instead of one blended average.
            per = max(n // self.n_slots, 1)
            w = m[..., None]
            fw = (f * w).reshape(b, self.n_slots, per, self.dim).sum(2)
            cw = w.reshape(b, self.n_slots, per, 1).sum(2).clamp(min=1)
            return self.head((fw / cw).reshape(b, -1))
        if not self.attn:
            w = m[..., None]
            gfeat = (f * w).sum(1) / w.sum(1).clamp(min=1)
            self._gfeat = gfeat           # pooled features, read by --joint-oa / --diff-oa heads
            out = self.head(gfeat)
            if getattr(self, "corsplit", False):
                # Medial labels pool non-coronal + MEDIAL-half coronal tokens only; lateral
                # likewise. Same head weights, no new parameters. gfeat (all tokens, both
                # halves) still drives every other label and _gfeat.
                cs = self._cs
                wm = w * cs["gmed"][None, :, None]
                wl = w * cs["glat"][None, :, None]
                g_med = (f * wm).sum(1) / wm.sum(1).clamp(min=1)
                g_lat = (f * wl).sum(1) / wl.sum(1).clamp(min=1)
                out = out.clone()
                out[:, MED_COLS] = self.head(g_med)[:, MED_COLS].to(out.dtype)
                out[:, LAT_COLS] = self.head(g_lat)[:, LAT_COLS].to(out.dtype)
                return out
            if getattr(self, "train_kper", None) is not None:
                # KERNEL-MATCHED POOLING. Every ConvNeXt lineage trains under the masked
                # mean above (focal_k == 0) but infer_kaggle.py scores it with PER-LABEL
                # top-k (K_PER = 6/6/6/6/16/16/16/12/12/3/3/3), so the head has never seen
                # the pooling it is graded under - the likeliest OA-specific weakness.
                # This is the kernel's rule verbatim (CNXNet.forward): masked slices ->
                # -1e4, top-k(min(k, n)) over the slot axis, mean; k == 0 keeps the mean.
                # It runs in the validation forward too, so teach/truth are measured
                # under the same pooling. Gradient reaches only the selected slices.
                L = self.head(f).masked_fill(m[:, :, None] < 0.5, -1e4)   # [b, n, labels]
                out = out.clone()
                for kk in sorted({v for v in self.train_kper if v > 0}):
                    cols = [i for i, v in enumerate(self.train_kper) if v == kk]
                    tp = L.topk(min(kk, n), dim=1).values.mean(1)
                    out[:, cols] = tp[:, cols].to(out.dtype)
                return self._apply_compool(out, gfeat, b, n)
            if getattr(self, "posw", False):
                # per-label softmax over slice position, SAGITTAL slots only. Coronal and
                # axial encode side in-plane, so their positions carry no M/L meaning.
                per = max(n // getattr(self, "n_slots", 6), 1)
                L = self.head(f)                                   # [b, n, labels]
                wcol = torch.ones(n, len(LABELS), device=f.device)
                pt = self.pos_table.softmax(1) * per                # [labels, per], mean 1
                for si in (0, 1):
                    wcol[si * per:(si + 1) * per] = pt.T
                lw = w * wcol[None]
                out = (L * lw).sum(1) / lw.sum(1).clamp(min=1)
                return self._apply_compool(out, gfeat, b, n)
            if getattr(self, "compsplit", False):
                # THIS is the path taken with focal_k == 0, which is the default and what
                # every one of our ConvNeXt lineages trains under. The first --compsplit
                # implementation only patched the focal_k branch, so the flag was dead code
                # and the run came back byte-identical to its control.
                cm = compsplit_mask(n, getattr(self, "n_slots", 6), f.device)[None]
                lw = w * cm
                mp2 = (self.head(f) * lw).sum(1) / lw.sum(1).clamp(min=1)
                out = out.clone()
                cols = MED_COLS + LAT_COLS
                out[:, cols] = mp2[:, cols].to(out.dtype)
            return self._apply_compool(out, gfeat, b, n)
        h = self.proj(f)                                   # [b, n, d]
        logits = torch.einsum("bnd,cd->bcn", h, self.q) / (self.dim ** 0.5)
        logits = logits.masked_fill(m[:, None, :] < 0.5, float("-inf"))
        a = torch.softmax(logits, dim=-1)                  # [b, labels, n]
        ctx = torch.einsum("bcn,bnd->bcd", a, f)           # per-label context
        return (ctx * self.cls[None]).sum(-1) + self.head.bias


# ---------------------------------------------------------------- --report-clip (image-report contrastive aux)
# Opt-in auxiliary loss: align the pooled image feature (Net._gfeat, the same tensor
# --joint-oa / --diff-oa read) with a frozen multilingual embedding of the study's own
# report (embed_reports.py, BAAI/bge-m3, 1024-d). The projection head model.rclip_proj is
# training-only: it is stripped from ck["model"] at save time (kept under ck["rclip_proj"])
# so inference kernels and oai_pull/score_moaks.py load the checkpoint with no new keys.
#
# MEMORY QUEUE. Our runs use batch 2 (x accum 8), so an in-batch InfoNCE would have ONE
# negative per row - almost no signal. The image->text direction therefore also scores
# against a FIFO queue of the last RCLIP_QUEUE text embeddings from earlier batches
# (detached, constant targets: the text side is a frozen encoder, so the queue never goes
# stale the way a MoCo key queue does). Queue entries that are (near-)identical to a row's
# own positive (cos > 0.999: the same study seen again, or a duplicated boilerplate report)
# are masked out so they are never counted as negatives. The text->image direction stays
# in-batch only: queued image features would come from older weights.
RCLIP_QUEUE = 256
RCLIP_DIM = 1024


def load_report_emb(path, studies, rows, key="emb"):
    """Report embeddings aligned to dataset rows. Returns (E [len(rows), D] float32,
    ok [len(rows)] bool); rows without an embedding get zeros and ok=False."""
    z = np.load(path, allow_pickle=False)
    ids = [str(s) for s in z["ids"]]
    emb = z[key].astype(np.float32)
    pos = {s: k for k, s in enumerate(ids)}
    E = np.zeros((len(rows), emb.shape[1]), np.float32)
    ok = np.zeros(len(rows), bool)
    for j, r in enumerate(rows):
        k = pos.get(studies[r])
        if k is not None and np.isfinite(emb[k]).all() and np.abs(emb[k]).sum() > 0:
            E[j], ok[j] = emb[k], True
    return E, ok


def report_clip_loss(img, txt, ok, tau=0.07, queue=None):
    """Symmetric InfoNCE between projected image features and report embeddings.

    img   [b, D] projected image features (any dtype; computed in float32)
    txt   [b, D] report embeddings for the same rows
    ok    [b]    bool, rows that have an embedding; the others are skipped
    queue [q, D] or None, L2-normalised detached text negatives for image->text
    Returns (loss or None when no row has an embedding, normalised txt of the used rows)."""
    ok = ok.bool()
    if not bool(ok.any()):
        return None, None
    q = F.normalize(img[ok].float(), dim=-1)
    t = F.normalize(txt[ok].float(), dim=-1)
    k = q.shape[0]
    tgt = torch.arange(k, device=q.device)
    keys = t
    if queue is not None and queue.shape[0] > 0:
        keys = torch.cat([t, queue.to(t.dtype)], 0)
    li = q @ keys.t() / tau                                      # [k, k + q]
    if keys.shape[0] > k:
        dup = (t @ queue.to(t.dtype).t()) > 0.999                # queue copies of a positive
        li = torch.cat([li[:, :k], li[:, k:].masked_fill(dup, float("-inf"))], 1)
    lt = t @ q.t() / tau                                         # [k, k] text->image, in-batch
    loss = 0.5 * (F.cross_entropy(li, tgt) + F.cross_entropy(lt, tgt))
    return loss, t.detach()


def rclip_enqueue(queue, t, size=RCLIP_QUEUE):
    """FIFO: newest first, keep the last `size` detached text embeddings."""
    if t is None:
        return queue
    return torch.cat([t.detach(), queue], 0)[:size]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--labels", default="report_labels_v1.csv")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--use", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--fold", type=int, default=-1,
                   help="0..folds-1: hold out this report-hash fold as validation. "
                        "-1 keeps the small gold holdout. Grouping by report hash "
                        "keeps byte-identical reports (177 studies share text) in "
                        "one fold, since they share one derived target vector.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--attn", action="store_true",
                   help="per-label attention pooling instead of a flat mean over all "
                        "images: each diagnosis learns which views/slices to weigh.")
    p.add_argument("--aug", action="store_true",
                   help="rigid-jitter augmentation (rot +-8deg, scale +-8%%, shift "
                        "+-10%%, intensity) - the public notebook family. No flips: "
                        "laterality is normalized upstream and a flip would undo it.")
    p.add_argument("--focal-k", dest="focal_k", type=int, default=0,
                   help="top-k pooling for Baker's/Contusion/Fracture, mean for the rest. "
                        "0 disables (flat mean everywhere, the settled baseline).")
    p.add_argument("--seq", action="store_true",
                   help="2-layer transformer encoder over the per-image features before "
                        "pooling, so slices can inform each other - a tear spanning three "
                        "adjacent slices is one finding, not three independent votes. The "
                        "mean-pool treats all 48 images as an unordered bag; this is the "
                        "only architecture idea never yet tested (attention POOLING was "
                        "tested and is different: it reweights the bag, it does not let "
                        "slices exchange information).")
    p.add_argument("--conf-weight", dest="conf_weight", action="store_true",
                   help="loss weight = conf directly, instead of 0.25 + 0.75*conf. The "
                        "label author encodes 'the report does not address this finding' "
                        "as p=0.5 (19%% of cells in v2) and prescribes weight 2*|p-0.5|, "
                        "which sends those to ZERO. Our floor of 0.25 trains on them as "
                        "if they were confident negatives. Averaging into v4 destroys "
                        "the 0.5 marker, so the mask has to be carried over from v2 - "
                        "see pubv4_masked.csv.")
    p.add_argument("--compool-soft", dest="compool_soft", action="store_true",
                   help="soft compartment concat (global+medial+lateral) instead of replacing global")
    p.add_argument("--compool", action="store_true",
                   help="medial/lateral compartment split for Meniscus/OA labels")
    p.add_argument("--clsattn", action="store_true",
                   help="per-label spatial attention pooling instead of global average")
    p.add_argument("--pairch", action="store_true",
                   help="stack each plane's fluid/non-fluid pair as image channels")
    p.add_argument("--msc", action="store_true",
                   help="concat a finer-scale pooled feature map into the head")
    p.add_argument("--upsample", default="",
                   help='rare-positive upsampling, e.g. "Fracture:3,Contusion:3"')
    p.add_argument("--no-gold", dest="no_gold", action="store_true",
                   help="never use the 58 expert labels - gold studies fall through to "
                        "their pseudo-labels at normal weight. Required for generating "
                        "distillation labels: if the label-making models train on gold, "
                        "study A's label carries information about study B's expert "
                        "annotation, and the 58-study gate then scores its own answers. "
                        "That leak made distillation read +0.0163 at the gate and -0.006 "
                        "on the leaderboard.")
    p.add_argument("--full", action="store_true",
                   help="train on ALL studies, no held-out fold. Uses every label instead "
                        "of 80%%, but there is then no honest validation signal, so "
                        "best-epoch selection is disabled and --epochs decides when to "
                        "stop. Set it to the epoch the fold runs peaked at (~22).")
    p.add_argument("--best", action="store_true", default=True,
                   help="save the best-validation epoch rather than the last one")
    p.add_argument("--last", dest="best", action="store_false",
                   help="save the final epoch instead (the old behaviour)")
    p.add_argument("--bf16", action="store_true",
                   help="autocast in bfloat16 instead of float16. resnet50 at 384px "
                        "NaNs inside one epoch where resnet18 never does: fp16 tops "
                        "out at 65504, and a single overflowed forward writes NaN "
                        "into BatchNorm's *running* stats, so training keeps looking "
                        "healthy on batch stats while eval returns all-NaN. bf16 has "
                        "fp32's exponent range and cannot overflow there.")
    p.add_argument("--clip", type=float, default=0.0,
                   help="max grad norm. resnet50 at 384px diverged to NaN inside one "
                        "epoch on real data where a 6-step synthetic probe was clean, "
                        "so the blow-up is data-dependent, not an init/scale problem. "
                        "Off by default: the resnet18 lineage never needed it and "
                        "enabling it there would silently change a settled baseline.")
    p.add_argument("--init", default=None,
                   help="path to a converted non-ImageNet backbone (RadImageNet). "
                        "Sets weights=None and loads this instead.")
    p.add_argument("--gray-fold", dest="gray_fold", action="store_true",
                   help="seed the 1-channel conv1 by summing the pretrained RGB "
                        "filters instead of random-init. Off by default so runs "
                        "stay comparable to every earlier arm; hold it fixed on "
                        "both sides when A/B-ing a backbone.")
    p.add_argument("--fast", action="store_true",
                   help="channels_last memory format + torch.compile. Same model, same "
                        "math, ~35%% faster on tensor-core GPUs. Numerically reordered "
                        "accumulation makes results differ within seed noise, so never "
                        "compare a --fast run against a non--fast baseline: re-run the "
                        "baseline fold once under --fast when adopting.")
    p.add_argument("--nvum", type=float, default=0.0,
                   help="NVUM (Liu et al. MICCAI22): per-sample EMA memory of debiased "
                        "logits used as extra soft targets, lambda=this value. Best-"
                        "evidenced robust-training lever for report-derived noisy labels: "
                        "+2.3-2.6 AUC over strong baselines on NLP-labelled CXR. 0=off.")
    p.add_argument("--nvum-beta", dest="nvum_beta", type=float, default=0.9,
                   help="EMA decay for the NVUM logit memory.")
    p.add_argument("--fixmatch", type=float, default=0.0,
                   help="multi-label FixMatch: confidence threshold TAU for accepting a "
                        "pseudo-label on LOW-CONFIDENCE cells (0 disables). 0.95 is standard.")
    p.add_argument("--fm-lambda", dest="fm_lambda", type=float, default=1.0,
                   help="weight on the FixMatch unlabeled loss.")
    p.add_argument("--fm-conf", dest="fm_conf", type=float, default=0.3,
                   help="cells with extraction confidence BELOW this are treated as unlabeled.")
    p.add_argument("--da", action="store_true",
                   help="distribution alignment: rescale pseudo-label odds so their running "
                        "marginal matches the labelled marginal. The verified result (+0.021) "
                        "was FixMatch WITH this; plain FixMatch collapses to the majority class.")
    p.add_argument("--backbone-init", dest="backbone_init", default="",
                   help="Load backbone weights from a checkpoint after the model is built. "
                        "Arch-agnostic, unlike --init which is hard-wired to the RadImageNet "
                        "ResNet path. Used to start from a backbone pretrained on SKM-TEA "
                        "per-slice TISSUE PRESENCE - the only anatomical slice-level "
                        "supervision available, since tib_MED and tib_LAT are near-mutually "
                        "exclusive by slice and a model that predicts them must learn "
                        "medial vs lateral.")
    p.add_argument("--posw-init", dest="posw_init", default="",
                   help="JSON of per-label position weights to INITIALISE --posw with, "
                        "instead of uniform. posw_init.json holds the SKM-TEA measured tissue "
                        "profiles resampled onto our 16 slot positions (Medial OA CoM 0.279, "
                        "Lateral OA 0.800). Learning the split from scratch produced uniform "
                        "for all twelve labels; this starts at the anatomical truth and lets "
                        "the data move it.")
    p.add_argument("--override", default="",
                   help="CSV of STUDENT out-of-fold predictions. Where the report never "
                        "discusses a structure, the teacher is not labelling it - it is "
                        "guessing (measured on gold-58: teacher AUC 0.556 on silent Lateral "
                        "Meniscus, i.e. a coin flip) while the image model already scores "
                        "0.676 there. This REPLACES the fabricated teacher target with the "
                        "student's own estimate on exactly those cells. Masking them (see "
                        "--silmask) lost, because it threw the student's advantage away too. "
                        "Requires --silmask to say which cells are silent.")
    p.add_argument("--eval-labels", dest="eval_labels", default="",
                   help="Label CSV used for the VALIDATION fold only. Use when --labels is a "
                        "student-prediction file (full self-distillation): training rows take "
                        "--labels, validation rows take this, so `teach` stays the same yardstick "
                        "as every other run (agreement with the teacher on held-out studies) "
                        "instead of the model's agreement with its own OOF.")
    p.add_argument("--override-all", dest="override_all", action="store_true",
                   help="FULL SELF-DISTILLATION: ignore the spoken mask and replace EVERY non-gold "
                        "teacher target with the student's calibrated prediction. A teammate measured on "
                        "8,027 expert-read studies that self-distilled targets beat report extraction "
                        "by +0.027 and that labels outweigh backbone 1.8x. --override alone (silent "
                        "cells only) gave +0.0163 on teach; this is the whole lever.")
    p.add_argument("--override-w", dest="override_w", type=float, default=1.0,
                   help="Loss weight for an overridden cell (default 1.0).")
    p.add_argument("--silmask", default="",
                   help="CSV of per-study SPOKEN flags (spoken_mask.csv). Where the report "
                        "never discusses a structure, the teacher's 0 is not a negative - it "
                        "is an absence of information, and training on it manufactures a "
                        "false negative. Measured on the 58 gold studies: of the 24 whose "
                        "report never mentions the lateral meniscus, SIX are positive. "
                        "54-77%% of all compartment labels are fabricated this way, which is "
                        "why the learned position table came out flat and why six "
                        "architectural attempts at medial/lateral all did nothing.")
    p.add_argument("--silw", type=float, default=0.0,
                   help="Loss weight for a label whose report is silent (default 0.0 = drop "
                        "it). Use a small positive value to keep a weak prior instead.")
    p.add_argument("--mlflip", type=float, default=0.0,
                   help="MIRROR AUGMENTATION WITH LABEL SWAP. With this probability, reverse "
                        "the SAGITTAL slice order and swap Medial<->Lateral on the targets "
                        "(Meniscus 2<->3, OA 4<->5). Unlike every other attempt this neither "
                        "restricts what the model sees nor adds a target - it makes the task "
                        "UNSOLVABLE without side discrimination, because the same pixels now "
                        "demand the opposite answer depending on orientation. Aimed directly "
                        "at the measured defect: the model predicts Med/Lat OA at r=0.747 when "
                        "the truth is r=0.317, i.e. it detects OA and splits it across both "
                        "outputs. That shortcut cannot survive this.")
    p.add_argument("--posw", action="store_true",
                   help="LEARNED PER-LABEL POSITION POOLING. Each label gets a learnable "
                        "weight over slice position within the two SAGITTAL slots, applied "
                        "as a softmax instead of the flat mean. --compsplit hard-masked each "
                        "compartment to its own half and LOST 0.0073; this lets the model "
                        "learn its own shape and can fall back to uniform, so it cannot be "
                        "worse by construction. The table is only (n_labels x slices_per_slot) "
                        "= 192 parameters, and it is READABLE - printing it says whether the "
                        "compartment story is true, independent of the score.")
    p.add_argument("--joint-oa", dest="joint_oa", type=float, default=0.0,
                   help="JOINT COMPARTMENT HEAD (OA). Extra softmax head on the pooled features over "
                        "{neither, medial only, lateral only, both}; soft target from the Medial/Lateral OA "
                        "labels (independent product). Forces a committed compartment pattern; auxiliary only, "
                        "inference outputs unchanged. Weight of the soft cross-entropy.")
    p.add_argument("--diff-oa", dest="diff_oa", type=float, default=0.0,
                   help="COMPARTMENT DIFFERENCE TARGET (OA). Extra scalar head on the pooled features trained "
                        "to predict (Lateral OA - Medial OA) with tanh + MSE, so features must separate the two "
                        "compartments. Auxiliary only. Weight of the MSE.")
    p.add_argument("--report-clip", dest="report_clip", type=float, default=0.0,
                   help="IMAGE-REPORT CONTRASTIVE AUX. Weight of a symmetric InfoNCE loss between "
                        "rclip_proj(pooled image feature) and the study's frozen report embedding "
                        "(--report-emb). Image->text also uses a 256-entry queue of recent text "
                        "embeddings as extra negatives (batch 2 has one in-batch negative). Rows "
                        "without an embedding are skipped. 0 = off, bit-identical to no flag. "
                        "Training-only: rclip_proj is not in ck['model'].")
    p.add_argument("--report-emb", dest="report_emb", default="",
                   help="npz from embed_reports.py (keys ids, emb[, emb_mean]), read from WORK")
    p.add_argument("--report-key", dest="report_key", default="emb",
                   help="which npz array to use: emb (whole report) or emb_mean (--chunk sentence mean)")
    p.add_argument("--report-tau", dest="report_tau", type=float, default=0.07,
                   help="InfoNCE temperature for --report-clip")
    p.add_argument("--sideaux", type=float, default=0.0,
                   help="AUXILIARY SLICE-LEVEL SUPERVISION. Predicts, per sagittal slice, "
                        "its normalised medial-lateral position, weighted by this factor. "
                        "This is the ONLY slice-level label available in the whole problem: "
                        "after --lat-norm the sagittal slice index IS the M/L coordinate "
                        "(build_cache reverses right knees; verified 40/40 vs SKM-TEA), so "
                        "the target is free and exact. Every other signal here is "
                        "study-level, which is why nothing spatial has worked - no gradient "
                        "ever distinguished one slice from another. Masking (--compsplit) "
                        "REMOVES information; this ADDS a reason for the features to encode "
                        "which side they came from. 0.1 is a sensible starting weight.")
    p.add_argument("--corsplit", action="store_true",
                   help="CORONAL HALF-SPLIT. In the forward pass every coronal slice becomes two "
                        "tokens, its MEDIAL half and its LATERAL half (split at the image centre, "
                        "each resized back to full HxW); the full coronal token is dropped. "
                        "Medial Meniscus/OA pool non-coronal + medial-half tokens, Lateral "
                        "Meniscus/OA non-coronal + lateral-half tokens, everything else all "
                        "tokens. Same head, no new parameters. Coronal tokens double, so a "
                        "6-slot study at --use U costs 8U backbone passes instead of 6U. "
                        "Needs a --lat-norm cache. Mean pooling only.")
    p.add_argument("--corsplit-medial", dest="corsplit_medial", default="left",
                   choices=["left", "right"],
                   help="which image-column half of a canonical coronal slice is MEDIAL. "
                        "'left' (columns < W/2) assumes coronal rows run along patient +x; see "
                        "the comment above coronal_slots() for why that is not certain.")
    p.add_argument("--compswap", type=float, default=0.0,
                   help="COMPARTMENT-SWAP MIXING (RSNA-2024 lumbar 2nd place 'mix sides across "
                        "patients'). With this probability per training batch (batch >= 2), every "
                        "study i is paired with another study j of the batch (a derangement) and "
                        "takes j's LATERAL compartment: the lateral column half of every coronal "
                        "slice (--corsplit-medial), plus sagittal/axial per --compswap-sag/-ax. "
                        "Lateral Meniscus/OA targets+weights come from j, Medial + MCL from i; ACL, "
                        "PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture get weight 0 on "
                        "mixed rows. 0 = off (bit-identical). See the COMPSWAP comment block.")
    p.add_argument("--compswap-sag", dest="compswap_sag", default="drop",
                   choices=["drop", "high", "low"],
                   help="sagittal slots on mixed rows. The lateral end of a cached sagittal stack "
                        "is NOT known (InstanceNumber order, protocol-dependent), so the default "
                        "drops them from mixed rows. high|low = the lateral half is the high|low "
                        "slice indices; use only after a header census shows one direction.")
    p.add_argument("--compswap-ax", dest="compswap_ax", default="drop",
                   choices=["drop", "keep", "swap"],
                   help="axial slots on mixed rows: drop (default), keep i's axial untouched "
                        "(label noise on the lateral labels), or swap the lateral column half "
                        "with the coronal rule (unverified for axial).")
    p.add_argument("--lm-slice-aux", dest="lm_slice_aux", type=float, default=0.0,
                   help="per-sagittal-slice auxiliary BCE: a Linear(dim,1) on each sagittal slice "
                        "feature predicts whether that cached slice overlaps the LATERAL MENISCUS "
                        "box (3DReasonKnee on OAI, oai_pull/make_lm_slice_targets.py). Masked to "
                        "studies/slots with a target; weight = this value. 0 = off.")
    p.add_argument("--lm-slice-targets", dest="lm_slice_targets", default="",
                   help="lm_slice_targets.npz (ids, t [N,2,S], ok [N,2]), read from WORK")
    p.add_argument("--compsplit", action="store_true",
                   help="TRAINING-TIME compartment split. The medial/lateral outputs are "
                        "pooled only over their own half of the SAGITTAL slots, so the "
                        "gradient forces each to depend on its own side. Diagnosed cause: "
                        "the model predicts Medial/Lateral OA at r=0.747 when the truth is "
                        "r=0.317 - it detects OA and misassigns the compartment. Masking "
                        "at INFERENCE (atlas1) did nothing because the features were "
                        "already merged by then; this puts the constraint in the gradient. "
                        "Requires --lat-norm caches: build_cache reverses sagittal order "
                        "for right knees, so low index = MEDIAL (verified 40/40 vs SKM-TEA).")
    p.add_argument("--train-kper", dest="train_kper", action="store_true",
                   help="TRAIN WITH THE KERNEL'S POOLING. Every ConvNeXt lineage trains with "
                        "masked-mean pooling (focal_k=0) but infer_kaggle.py scores it with "
                        "per-label top-k (K_PER), a train/inference mismatch flagged as the "
                        "likeliest OA-specific weakness. This pools each label as the mean of "
                        "its top-k slice logits over the valid slices, in BOTH the training and "
                        "validation forward, exactly as the kernel does. Composes with --sideaux; "
                        "not with --posw / --compsplit / --focal-k (error).")
    p.add_argument("--kper", default="6,6,6,6,16,16,16,12,12,3,3,3",
                   help="comma-separated per-label k for --train-kper, in LABELS order; 0 keeps "
                        "the mean for that label. Default is the kernel's K_PER.")
    p.add_argument("--usmooth", type=float, default=0.0,
                   help="stochastic uncertainty-targeted label smoothing: half-width of the "
                        "per-epoch target jitter at confidence 0 (0.25 is the tested value)")
    p.add_argument("--repgls", action="store_true",
                   help="Rep-GLS: per-sample per-label smoothing rate derived from the "
                        "LLM extraction confidence (recovered from the loss weight w). "
                        "Uncertain labels smooth toward 0.5; confident ones SHARPEN "
                        "(negative rate) - the negative-rate range alone was worth "
                        "+2.58pp in the source paper.")
    p.add_argument("--gru", action="store_true",
                   help="bidirectional GRU over slice features before pooling - the "
                        "RSNA-2023 abdominal-trauma winner head. Order-aware without "
                        "positional embeddings, lighter than the transformer aggregator "
                        "we already tested (which lost -0.008).")
    p.add_argument("--gold-ft", dest="gold_ft", type=int, default=0,
                   help="after normal training, fine-tune for N epochs at lr/20 on the "
                        "gold studies in the TRAINING folds only. Training on gold as a "
                        "co-equal target at weight 3.0 lost (-0.002): 58 hard 0/1 labels "
                        "shout over 4349 soft ones for 30 epochs. A short low-LR pass at "
                        "the END lets the model settle toward expert opinion without "
                        "letting it dominate the whole optimisation. Validation-fold gold "
                        "is never touched, so the metric stays honest.")
    p.add_argument("--dsample", type=int, default=0,
                   help="bilinearly downsample cached slices to NxN before the backbone. "
                        "Needed for 3D convs, whose activations scale with slices*H*W and "
                        "will not fit at 384. Run the 2D control at the SAME size or the "
                        "comparison confounds architecture with resolution.")
    p.add_argument("--maxpool", action="store_true",
                   help="concatenate global-MAX pooling with global-average pooling per "
                        "slice, doubling the feature width. Average pooling destroys a "
                        "peak that occupies a few pixels of a few slices, which is exactly "
                        "what a focal tear looks like - and focal findings are where this "
                        "model trails its own teacher (MCL -0.12, menisci -0.04).")
    p.add_argument("--planepool", action="store_true",
                   help="pool WITHIN each plane and concatenate, instead of averaging all "
                        "six slots into one vector. Sagittal has the most series and "
                        "currently dominates the blend; four of twelve labels are "
                        "medial/lateral, so plane identity is signal we are averaging away.")
    p.add_argument("--label-weight", dest="label_weight", action="store_true",
                   help="scale each label's loss by how reliably the TEACHER reads it, "
                        "measured against the 58 gold studies: w = (2*AUC-1) normalised "
                        "to mean 1. Teacher AUC spans 0.79 (Synovitis, Fracture) to 0.99 "
                        "(ACL), so equal weighting spends the same capacity fitting noise "
                        "as signal - and capacity is this pipeline's binding constraint. "
                        "Uses gold for guidance, not as training targets.")
    p.add_argument("--qhead", action="store_true",
                   help="12 per-diagnosis attention queries instead of one shared mean-pool. "
                        "Each finding gets a learned query that cross-attends over the 96 "
                        "slice features, with absent slots masked out of the softmax, so "
                        "ACL can look at the sagittal slices where it lives while Baker's "
                        "looks posteriorly - instead of all twelve reading one global "
                        "average. Matches the public 0.90+ head design and targets exactly "
                        "the focal labels where we trail our own teacher.")
    p.add_argument("--in25d", action="store_true",
                   help="2.5D input: each slice fed as [prev, cur, next] channel triplet "
                        "from the cached stack. Only tested before at 12 epochs under the "
                        "weak teacher - that verdict class is known-unreliable.")
    p.add_argument("--aug-mult", dest="aug_mult", type=float, default=1.0,
                   help="scale every rigid-jitter magnitude. 1.0 is the settled recipe "
                        "(rot +-8deg); 2.0 doubles rotation/scale/shift/intensity. The "
                        "strength has never been varied, only on/off.")
    p.add_argument("--drop-slots", dest="drop_slots", default="",
                   help="comma-separated slot indices to zero out. Slot order is "
                        "plane-major over (Sagittal,Coronal,Axial) x (fluid,non-fluid), "
                        "so '4,5' drops axial entirely. Tests whether a plane earns its "
                        "third of the input budget.")
    p.add_argument("--accum", type=int, default=1,
                   help="gradient accumulation steps: batch 2 x accum 4 reproduces "
                        "the effective batch 8 the other arms trained under, so "
                        "VRAM-forced small batches stay comparable")
    a = p.parse_args()
    if a.report_clip > 0 and not a.report_emb:
        p.error("--report-clip needs --report-emb (npz from embed_reports.py)")
    _kper = None
    if a.train_kper:
        _kper = [int(v) for v in a.kper.split(",")]
        if len(_kper) != len(LABELS):
            p.error(f"--kper needs {len(LABELS)} comma-separated ints (LABELS order), got {len(_kper)}: {a.kper}")
        _clash = [f_ for f_ in ("posw", "compsplit", "focal_k", "attn", "qhead", "planepool", "clsattn")
                  if getattr(a, f_)]
        if _clash:
            p.error("--train-kper replaces the pooling and cannot be combined with --"
                    + ", --".join(c.replace("_", "-") for c in _clash))
    if a.corsplit:
        _clash = [f_ for f_ in ("focal_k", "train_kper", "posw", "compsplit", "qhead", "planepool",
                                "clsattn", "attn", "seq", "gru", "compool", "compool_soft",
                                "pairch", "mlflip") if getattr(a, f_)]
        if a.arch.startswith(("r3d", "r2plus1d", "mc3")):
            _clash.append("arch " + a.arch)
        if _clash:
            p.error("--corsplit handles only the masked-mean pooling and cannot be combined with --"
                    + ", --".join(c.replace("_", "-") for c in _clash))
    if a.compswap < 0 or a.compswap > 1:
        p.error("--compswap is a probability in [0, 1]")
    if a.compswap > 0:
        # mlflip mirrors the sagittal order and swaps M/L labels: a second, conflicting M/L
        # remap of the same row. pairch changes the token layout (3 plane tokens, no slots).
        # nvum keeps a per-study logit EMA and report-clip a per-study text target; a mixed
        # row is no longer that study. --corsplit composes (it only re-tokenises coronal
        # halves, using the same --corsplit-medial side this swap uses).
        _clash = [f_ for f_ in COMPSWAP_CLASH if getattr(a, f_)]
        if _clash:
            p.error("--compswap cannot be combined with --"
                    + ", --".join(c.replace("_", "-") for c in _clash))
        if a.batch < 2:
            p.error("--compswap needs --batch >= 2 (pairs come from within the batch)")
    if a.lm_slice_aux > 0:
        if not a.lm_slice_targets:
            p.error("--lm-slice-aux needs --lm-slice-targets")
        _clash = [f_ for f_ in ("pairch", "clsattn") if getattr(a, f_)]
        if a.arch.startswith(("r3d", "r2plus1d", "mc3")):
            _clash.append("arch " + a.arch)
        if _clash:
            p.error("--lm-slice-aux needs per-slice features and cannot be combined with --"
                    + ", --".join(c.replace("_", "-") for c in _clash))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = a.out or f"{a.cache}_{a.arch}_u{a.use}_s{a.seed}.pt"

    base = os.path.join(CACHE_DIR, a.cache)
    meta = json.load(open(f"{base}.json"))
    arr = np.load(f"{base}.npy", mmap_mode="r")
    mask = np.load(f"{base}.mask.npy")
    studies = meta["studies"]
    print(f"cache {arr.shape} planes={meta['planes']} img={meta['img']}", flush=True)

    z = zipfile.ZipFile(ZIP)
    train = pd.read_csv(io.BytesIO(z.read("train.csv")))
    rex = pd.read_csv(os.path.join(WORK, a.labels)).set_index("StudyInstanceUID")
    gold = train[train[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS]

    keep = [i for i, s in enumerate(studies) if mask[i].sum() > 0]
    Y = np.zeros((len(studies), len(LABELS)), np.float32)
    W = np.zeros_like(Y)
    conf = [c for c in rex.columns if c.endswith("__conf")]
    for i, s in enumerate(studies):
        if s in gold.index and not a.no_gold:
            Y[i], W[i] = gold.loc[s].values, 3.0
        elif s in rex.index:
            Y[i] = rex.loc[s, LABELS].values
            _c = rex.loc[s, conf].values if conf else 1.0
            W[i] = _c if a.conf_weight else (0.25 + 0.75 * _c)

    if a.eval_labels and a.fold >= 0:
        import hashlib as _hl2
        _rex_ev = pd.read_csv(os.path.join(WORK, a.eval_labels)).set_index("StudyInstanceUID")
        _conf_ev = [c for c in _rex_ev.columns if c.endswith("__conf")]
        _rep2 = train.set_index("StudyInstanceUID")["Report"].fillna("")
        _nev = 0
        for i, s_ in enumerate(studies):
            if W[i].sum() == 0 or s_ not in _rex_ev.index: continue
            if s_ in gold.index and not a.no_gold: continue
            if int(_hl2.md5(_rep2.get(s_, s_).encode()).hexdigest()[:8], 16) % a.folds != a.fold: continue
            Y[i] = _rex_ev.loc[s_, LABELS].values
            _c = _rex_ev.loc[s_, _conf_ev].values if _conf_ev else 1.0
            W[i] = _c if a.conf_weight else (0.25 + 0.75 * _c)
            _nev += 1
        print(f"EVAL-LABELS: {_nev} validation-fold studies scored against {a.eval_labels} "
              f"(training rows keep {a.labels})", flush=True)

    if a.silmask:
        # A teacher 0 on a structure the report never discusses is not a negative, it is an
        # absence of information. Measured on the 58 gold studies: of the 24 whose report
        # never mentions the lateral meniscus, SIX are positive. Training those as negatives
        # manufactures false negatives across 54-77% of all compartment labels, which is why
        # the learned position table came out flat and six architectural attempts at
        # medial/lateral produced nothing. GOLD rows are exempt - those labels come from an
        # expert reading the IMAGES, so the report's silence says nothing about them.
        _sm = pd.read_csv(os.path.join(WORK, a.silmask)).set_index("StudyInstanceUID")
        _cols = [c for c in _sm.columns if c in LABELS]
        # --override-all is a FULL self-distill: every label, not just the ones the mask
        # file happens to carry (the 5-column mask silently limited the first fd run to
        # the compartment labels - 17,630 cells instead of ~42,000).
        _ci = {c: LABELS.index(c) for c in _cols}            # silmask loop: mask columns only
        _n_cell = _n_mask = _n_hit = 0
        for i, s_ in enumerate(studies):
            if W[i].sum() == 0:            # no label at all
                continue
            if s_ in gold.index and not a.no_gold:   # expert image read - leave alone
                continue
            if s_ not in _sm.index:
                continue
            _n_hit += 1
            _row = _sm.loc[s_]
            for c, ci in _ci.items():
                _n_cell += 1
                if int(_row[c]) == 0 and Y[i, ci] < 0.5:   # silent AND labelled negative
                    W[i, ci] *= a.silw
                    _n_mask += 1
        if a.override:
            # Replace, do not mask. Rank-normalised student scores are recalibrated so their
            # mean matches the teacher's prevalence on the SPOKEN studies for that label -
            # the best available estimate of the true rate. Without this the soft target
            # would assert ~50% prevalence where the truth is nearer 20%.
            _ov = pd.read_csv(os.path.join(WORK, a.override)).set_index("StudyInstanceUID")
            _nov = 0
            _ci_ov = {c: LABELS.index(c) for c in (LABELS if a.override_all else _cols)}
            # NEVER override a VALIDATION-fold row. Otherwise the val targets become the
            # student's own OOF predictions and `teach` measures self-agreement - circular,
            # and useless for picking the best epoch. Same md5 rule as the fold split below.
            import hashlib as _hl
            _rep = train.set_index("StudyInstanceUID")["Report"].fillna("")
            _valf = {s_ for s_ in studies
                     if a.fold >= 0 and int(_hl.md5(_rep.get(s_, s_).encode()).hexdigest()[:8], 16) % a.folds == a.fold}
            print(f"OVERRIDE: {len(_valf)} validation-fold studies left on teacher targets (teach stays honest)", flush=True)
            for c, ci in _ci_ov.items():
                if c not in _ov.columns: continue
                _spoken = [] if (a.override_all or c not in _sm.columns) else \
                          [i for i, s_ in enumerate(studies)
                           if s_ in _sm.index and int(_sm.loc[s_][c]) == 1 and W[i].sum() > 0]
                if a.override_all or not _spoken:
                    # calibrate to the teacher's OVERALL prevalence on labelled, non-gold studies
                    _lab = [i for i, s_ in enumerate(studies)
                            if W[i].sum() > 0 and not (s_ in gold.index and not a.no_gold)]
                    _prev = float(np.mean([Y[i, ci] for i in _lab])) if _lab else 0.2
                else:
                    _prev = float(np.mean([Y[i, ci] for i in _spoken]))
                for i, s_ in enumerate(studies):
                    if W[i].sum() == 0 or s_ not in _sm.index or s_ not in _ov.index: continue
                    if s_ in gold.index and not a.no_gold: continue
                    if s_ in _valf: continue
                    if a.override_all or (c in _sm.columns and int(_sm.loc[s_][c]) == 0):
                        _r = float(_ov.loc[s_][c])          # rank-uniform in [0,1]
                        Y[i, ci] = float(np.clip(_r * 2.0 * _prev, 0.0, 1.0))
                        W[i, ci] = a.override_w
                        _nov += 1
            print(f"OVERRIDE{' ALL (full self-distill)' if a.override_all else ''} ON: {_nov} cells replaced by student predictions "
                  f"(calibrated to spoken-study prevalence), weight {a.override_w}", flush=True)
        print(f"SILMASK ON: {_n_hit} studies matched, weight {a.silw} where the report is "
              f"silent and the teacher said negative; {_n_mask}/{_n_cell} "
              f"({_n_mask / max(_n_cell,1):.1%}) of compartment cells down-weighted",
              flush=True)

    if a.full:
        # every study with a label, nothing held out. The printed "gold" number below is
        # measured on studies the model trained on, so it is contaminated by construction
        # and is only a sanity check that training is progressing - never a result.
        tr = [i for i in keep if W[i].sum() > 0]
        va = [i for i in keep if studies[i] in gold.index]
        a.best = False
        print(f"FULL FIT: train {len(tr)} (no holdout); val {len(va)} is CONTAMINATED, "
              f"for progress only", flush=True)
    elif a.fold >= 0:
        import hashlib
        rep = train.set_index("StudyInstanceUID")["Report"].fillna("")
        fold_of = {s_: int(hashlib.md5(rep.get(s_, s_).encode()).hexdigest()[:8], 16)
                   % a.folds for s_ in studies}
        va = [i for i in keep if fold_of[studies[i]] == a.fold and W[i].sum() > 0]
        tr = [i for i in keep if fold_of[studies[i]] != a.fold and W[i].sum() > 0]
        gold_in_va = [i for i in va if studies[i] in gold.index]
        print(f"fold {a.fold}/{a.folds}: train {len(tr)}  val {len(va)} "
              f"(gold in val: {len(gold_in_va)})", flush=True)
    else:
        gi = [i for i in keep if studies[i] in gold.index]
        rng = np.random.default_rng(0)          # fixed split across all runs
        rng.shuffle(gi)
        va = sorted(gi[:len(gi) // 3])
        tr = [i for i in keep if i not in set(va) and W[i].sum() > 0]
        print(f"train {len(tr)}  gold-holdout {len(va)} (never trained on)", flush=True)

    dev = torch.device(os.environ.get("TC_DEVICE", "cuda"))   # TC_DEVICE=cpu: tests/test_report_clip.py only
    globals()['PAIRCH'] = a.pairch
    model = Net(a.arch, attn=a.attn, init=a.init, gray_fold=a.gray_fold,
                focal_k=a.focal_k, seq=a.seq, in25d=a.in25d, qhead=a.qhead,
                maxpool=a.maxpool, planepool=a.planepool,
                n_slots=int(mask.shape[1]), gru=a.gru, msc=a.msc, clsattn=a.clsattn, compool=a.compool, compool_soft=a.compool_soft).to(dev)
    # Net() does not take compsplit as a kwarg, so set it directly. The first attempt at
    # this flag failed TWICE over: the pooling patch went on a branch that focal_k==0 never
    # takes, AND the flag was never assigned to the model at all, so the run came back
    # byte-identical to its control and looked like a null result.
    model.compsplit = bool(a.compsplit)
    model.sideaux = float(a.sideaux)
    model.train_kper = _kper          # None when off: forward takes the untouched mean path
    model.corsplit = bool(a.corsplit)
    if model.corsplit:
        if not meta.get("lat_norm"):
            print("WARNING: --corsplit on a cache without lat_norm - right knees are not mirrored, "
                  "so the medial half is on the wrong side for them", flush=True)
        model.corsplit_slots = coronal_slots(meta, int(mask.shape[1]))
        model.corsplit_medial = a.corsplit_medial
        print(f"CORSPLIT ON: coronal slots {model.corsplit_slots}, medial = image-"
              f"{a.corsplit_medial} half; medial cols {MED_COLS}, lateral cols {LAT_COLS}", flush=True)
    _sag_slots = plane_slots(meta, int(mask.shape[1]), "Sagittal")
    if a.compswap > 0:
        if not meta.get("lat_norm"):
            print("WARNING: --compswap on a cache without lat_norm - right knees are not "
                  "mirrored, so the lateral half is on the wrong side for them", flush=True)
        CSW = {"sag_slots": _sag_slots,
               "cor_slots": plane_slots(meta, int(mask.shape[1]), "Coronal"),
               "ax_slots": plane_slots(meta, int(mask.shape[1]), "Axial"),
               "sag": a.compswap_sag, "ax": a.compswap_ax, "cor_medial": a.corsplit_medial}
        print(f"COMPSWAP ON: p={a.compswap} sag={a.compswap_sag} ax={a.compswap_ax} coronal "
              f"lateral = image-{'right' if a.corsplit_medial == 'left' else 'left'} half; slots "
              f"{CSW['sag_slots']}/{CSW['cor_slots']}/{CSW['ax_slots']}; zero-weight on mixed rows: "
              f"{COMPSWAP_ZERO}", flush=True)
    model.lm_slice_aux = float(a.lm_slice_aux)
    if model.lm_slice_aux > 0:
        if len(_sag_slots) != 2:
            raise SystemExit(f"--lm-slice-aux expects 2 sagittal slots, cache has {_sag_slots}")
        model.lm_head = nn.Linear(model.dim, 1).to(dev)
    if a.backbone_init:
        _bk = torch.load(a.backbone_init, map_location="cpu", weights_only=False)
        _sd = _bk.get("backbone", _bk.get("model", _bk))
        _sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in _sd.items()}
        _miss, _unexp = model.backbone.load_state_dict(_sd, strict=False)
        print(f"BACKBONE INIT from {a.backbone_init}: {len(_miss)} missing, {len(_unexp)} unexpected",
              flush=True)
    model.posw = bool(a.posw)
    if model.posw:
        per_ = max(int(mask.shape[2]) if mask.ndim > 2 else a.use, 1)
        _init = torch.zeros(len(LABELS), a.use)
        if a.posw_init:
            import json as _json
            _pi = _json.load(open(os.path.join(WORK, a.posw_init)))
            for _i, _l in enumerate(LABELS):
                _v = np.asarray(_pi[_l], dtype=np.float32)
                if len(_v) != a.use:
                    _v = np.interp(np.linspace(0, 1, a.use), np.linspace(0, 1, len(_v)), _v)
                _v = np.clip(_v, 1e-4, None); _v = _v / _v.sum()
                _init[_i] = torch.log(torch.from_numpy(_v))     # softmax(log p) == p
            print(f"POSW INIT from {a.posw_init}: anatomy-seeded, not uniform", flush=True)
        model.pos_table = nn.Parameter(_init.to(dev))
        model.register_parameter("pos_table", model.pos_table)
        print(f"POSW ON: learnable ({len(LABELS)} x {a.use}) position table on the sagittal slots",
              flush=True)
    if a.joint_oa > 0:
        model.joint_head = nn.Linear(model.dim, 4).to(dev)
        print(f"JOINT-OA ON: weight {a.joint_oa}, 4-way compartment head on pooled features", flush=True)
    if a.diff_oa > 0:
        model.diff_head = nn.Linear(model.dim, 1).to(dev)
        print(f"DIFF-OA ON: weight {a.diff_oa}, (Lateral OA - Medial OA) regression head", flush=True)
    if a.report_clip > 0:
        _RE, _RE_ok = load_report_emb(os.path.join(WORK, a.report_emb), studies, tr, a.report_key)
        RCLIP_E = torch.from_numpy(_RE).to(dev)                  # [len(tr), D], indexed by ridx
        RCLIP_OK = torch.from_numpy(_RE_ok).to(dev)
        RCLIP_Q = torch.zeros(0, _RE.shape[1], device=dev)        # FIFO text-negative queue
        model.rclip_proj = nn.Linear(model.dim, _RE.shape[1]).to(dev)
        print(f"REPORT-CLIP ON: weight {a.report_clip}, tau {a.report_tau}, {a.report_emb}[{a.report_key}] "
              f"covers {int(_RE_ok.sum())}/{len(tr)} training rows, dim {_RE.shape[1]}, "
              f"queue {RCLIP_QUEUE}", flush=True)
    if model.sideaux > 0:
        model.side_head = nn.Linear(model.dim, 1).to(dev)
        print(f"SIDEAUX ON: weight {a.sideaux}, per-slice M/L position on the sagittal slots",
              flush=True)
    if model.compsplit:
        print(f"COMPSPLIT ON: medial cols {MED_COLS}, lateral cols {LAT_COLS}, "
              f"n_slots={int(mask.shape[1])}", flush=True)
    if model.train_kper is not None:
        print(f"TRAIN-KPER ON: {model.train_kper}", flush=True)
    if a.fast:
        model = model.to(memory_format=torch.channels_last)
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    LW = (torch.tensor(LABEL_W, device=dev)[None] if a.label_weight
          else torch.ones(1, len(LABELS), device=dev))
    scaler = torch.amp.GradScaler("cuda")

    _drop = tuple(int(v) for v in a.drop_slots.split(",") if v.strip())
    _sampler = None
    if a.upsample:
        # Rare-positive upsampling (RSNA-2023 winners: bowel 0.833->0.881 from this
        # alone). "Label:mult" pairs; a study positive (soft y>0.5) for that label is
        # drawn mult x as often. Weights multiply across labels.
        sw = np.ones(len(tr), np.float64)
        for spec in a.upsample.split(","):
            lab, mult = spec.split(":")
            li = LABELS.index(lab.strip())
            sw *= np.where(Y[tr][:, li] > 0.5, float(mult), 1.0)
        from torch.utils.data import WeightedRandomSampler
        _sampler = WeightedRandomSampler(torch.from_numpy(sw), num_samples=len(tr),
                                         replacement=True)
        print(f"upsample: {a.upsample} -> {int((sw > 1).sum())} boosted studies",
              flush=True)
    _extra_kw = {}
    if a.compswap > 0:
        _extra_kw["ret_vflip"] = True
    if a.lm_slice_aux > 0:
        _LMT, _LMOK = load_lm_slice_targets(os.path.join(WORK, a.lm_slice_targets), studies, tr,
                                            int(arr.shape[2]))
        _extra_kw.update(lm_t=_LMT, lm_ok=_LMOK, lm_slots=_sag_slots)
        _lmv = _LMOK.sum(0)
        print(f"LM-SLICE-AUX ON: weight {a.lm_slice_aux}, {a.lm_slice_targets}: "
              f"{int((_LMOK.sum(1) > 0).sum())}/{len(tr)} training studies have a target "
              f"(per sagittal slot {_lmv.astype(int).tolist()}), positive slice rate "
              f"{float((_LMT * _LMOK[..., None]).sum() / max((_LMOK.sum() * _LMT.shape[-1]), 1)):.3f}",
              flush=True)
    dl = DataLoader(Cached(arr, mask, tr, Y[tr], W[tr], a.use, aug=a.aug,
                           aug_mult=a.aug_mult, drop_slots=_drop, in25d=a.in25d, dsample=a.dsample, pairch=a.pairch,
                           **_extra_kw),
                    batch_size=a.batch,
                    shuffle=_sampler is None, sampler=_sampler,
                    num_workers=a.workers, pin_memory=True, drop_last=True,
                    persistent_workers=a.workers > 0)
    vdl = DataLoader(Cached(arr, mask, va, Y[va], W[va], a.use, train=False,
                            drop_slots=_drop, in25d=a.in25d, dsample=a.dsample, pairch=a.pairch),
                     batch_size=a.batch, num_workers=2)
    # TRUTH ANCHOR. The val loop below scores against Y[va], which under --no-gold is the
    # --labels TEACHER csv for every study - so a model trained on teacher X was being graded
    # on how well it reproduces X. Swapping teacher swapped the yardstick, which is why 22
    # straight "CV wins" moved the LB by -0.001. These are the studies in this fold that have
    # REAL expert labels; they give an accuracy number that is comparable across teachers.
    _gpos = [k for k, i in enumerate(va) if studies[i] in gold.index]
    _gT = (np.stack([gold.loc[studies[va[k]]].values for k in _gpos]).astype(np.float32)
           if _gpos else None)
    print(f"truth anchor: {len(_gpos)} expert-labelled studies in this val fold", flush=True)

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * max(len(dl) // a.accum, 1), pct_start=0.2)

    # NVUM: per-sample EMA of class-prior-debiased logits, used as auxiliary soft
    # targets. Indexed by dataset row (position within tr), so it needs ridx from the
    # loader. Prior logit pi_c is computed from the training labels once.
    if a.nvum > 0:
        _pos = np.clip((Y[tr] > 0.5).mean(0), 1e-3, 1 - 1e-3)
        NV_PRIOR = torch.tensor(np.log(_pos / (1 - _pos)), dtype=torch.float32,
                                device=dev)[None]
        NV_MEM = torch.zeros(len(tr), len(LABELS), device=dev)

    def _strong_view(x, m):
        """Strong augmentation for the FixMatch student branch. Intensity gamma/noise plus
        SLICE DROPOUT via the mask - dropping slices is the natural strong perturbation for
        a slice-pooling model, and it is the one the weak view never sees."""
        g = torch.empty(x.shape[0], *([1] * (x.dim() - 1)), device=x.device).uniform_(0.7, 1.4)
        xs = x.clamp(min=0).pow(g)
        xs = xs + torch.randn_like(xs) * 0.03
        keep = (torch.rand_like(m) > 0.2).float()
        ms = m * keep
        ms = torch.where(ms.sum(1, keepdim=True) > 0, ms, m)   # never blank a whole study
        return xs, ms

    if a.fixmatch > 0:
        # labelled marginal for distribution alignment, from the confident cells only
        _cw = Y[tr]
        FM_PRIOR = torch.tensor(np.clip(_cw.mean(0), 1e-3, 1 - 1e-3),
                                dtype=torch.float32, device=dev)[None]
        FM_EMA = FM_PRIOR.clone()

    import time
    best_auc, best_state, best_ep = -1.0, None, 0
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        rc_tot, rc_n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, _batch in enumerate(dl):
            x, m, y, w, ridx = _batch[:5]
            x, m, y, w = x.to(dev, non_blocking=True), m.to(dev), y.to(dev), w.to(dev)
            _bx = list(_batch[5:])
            _vf = _bx.pop(0) if a.compswap > 0 else None
            if a.lm_slice_aux > 0:
                lm_t, lm_ok = _bx[0].to(dev), _bx[1].to(dev)
            if a.compswap > 0 and x.shape[0] >= 2 and float(torch.rand(1)) < a.compswap:
                # --compswap: on the device tensors, before the forward pass. The lm-slice
                # targets ride along with the sagittal tokens they describe.
                _ex = (lm_t, lm_ok) if a.lm_slice_aux > 0 else ()
                x, m, y, w, _ex, _mixed = compswap_apply(
                    x, m, y, w, compswap_partner(x.shape[0]).to(dev), model.n_slots,
                    CSW["sag_slots"], CSW["cor_slots"], CSW["ax_slots"], sag=CSW["sag"],
                    ax=CSW["ax"], cor_medial=CSW["cor_medial"], vflip=_vf, extra=_ex)
                if a.lm_slice_aux > 0:
                    lm_t, lm_ok = _ex
            if a.mlflip > 0:
                # Reverse the two SAGITTAL slots and swap the medial/lateral label pairs.
                # Coronal and axial encode side IN-PLANE, so their slice ORDER carries no
                # M/L meaning; they are left alone.
                #
                # Why p=0.5 costs nothing: the metric is per-label ROC AUC, which is
                # invariant to a constant offset on a label's score. So the marginal base
                # rate (medial OA really is commoner than lateral) contributes ZERO to the
                # score, and removing the model's access to it is free - while making the
                # "detect OA, split it across both outputs" shortcut strictly unusable.
                _per = max(x.shape[1] // model.n_slots, 1)
                _sel = torch.rand(x.shape[0], device=dev) < a.mlflip
                if _sel.any():
                    _i = _sel.nonzero(as_tuple=True)[0]
                    for _si in (0, 1):
                        _a, _b = _si * _per, (_si + 1) * _per
                        x[_i, _a:_b] = torch.flip(x[_i, _a:_b], dims=[1])
                        m[_i, _a:_b] = torch.flip(m[_i, _a:_b], dims=[1])
                    _sw = torch.arange(y.shape[1], device=dev)
                    _sw[MED_COLS], _sw[LAT_COLS] = _sw[LAT_COLS].clone(), _sw[MED_COLS].clone()
                    y[_i] = y[_i][:, _sw]
                    w[_i] = w[_i][:, _sw]
                    # MCL and Baker's are medial-only structures with NO lateral partner
                    # (there is no LCL label, and a popliteal cyst is posteromedial). A
                    # mirrored study puts them on the wrong side with the label unchanged,
                    # so drop them from the loss on mirrored samples rather than teach
                    # the model that an MCL can sit laterally.
                    w[_i[:, None], torch.tensor([1, 9], device=dev)[None]] = 0.0
            if a.fast and x.dim() == 4:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16 if a.bf16 else torch.float16):
                z = model(x, m)
                if a.usmooth > 0:
                    # STOCHASTIC uncertainty-targeted label smoothing. Distinct from
                    # --repgls, which deterministically interpolates the LOSS toward a
                    # uniform target: this redraws the TARGET each epoch, so an uncertain
                    # label becomes a distribution rather than a fixed softened value -
                    # the mechanism behind the CheXpert LSR result (+0.013/+0.014 mean AUC).
                    # Band width scales with (1 - confidence): a conf-1.0 label is left
                    # exactly alone, a conf-0 label is drawn from y +/- usmooth.
                    conf = ((w - 0.25) / 0.75).clamp(0, 1)
                    half = a.usmooth * (1.0 - conf)
                    y = (y + (torch.rand_like(y) * 2.0 - 1.0) * half).clamp(0.0, 1.0)
                bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
                if a.repgls:
                    # Rep-GLS: smoothing rate from extraction confidence, recovered from
                    # the loss weight (default weighting w = 0.25 + 0.75*conf). conf 1 ->
                    # r=-0.2 (sharpen away from 0.5); conf 0 -> r=+0.4 (smooth toward it).
                    conf = ((w - 0.25) / 0.75).clamp(0, 1)
                    r = (0.4 - 0.6 * conf).clamp(-0.2, 0.4)
                    uni = F.binary_cross_entropy_with_logits(
                        z, torch.full_like(y, 0.5), reduction="none")
                    bce = (1 - r) * bce + r * uni
                loss = (bce * w * LW).mean()
                if a.fixmatch > 0:
                    # Cells whose extraction confidence is low are UNLABELLED, not negative.
                    # The teacher CSV encodes "report never mentioned it" the same way it
                    # encodes "reported absent", which is exactly the supervision gap the
                    # forum thread measured at 85% of Synovitis targets.
                    conf_u = ((w - 0.25) / 0.75).clamp(0, 1)
                    unl = (conf_u < a.fm_conf).float()
                    with torch.no_grad():
                        p_w = torch.sigmoid(z.float())
                        if a.da:
                            # distribution alignment: rescale odds by prior/running-marginal
                            FM_EMA.mul_(0.999).add_(0.001 * p_w.mean(0, keepdim=True))
                            r = (FM_PRIOR / FM_EMA.clamp(1e-3, 1 - 1e-3))
                            odds = (p_w / (1 - p_w).clamp_min(1e-6)) * r
                            p_w = (odds / (1 + odds)).clamp(1e-4, 1 - 1e-4)
                        hard = (p_w > 0.5).float()
                        conf_mask = ((p_w > a.fixmatch) | (p_w < 1 - a.fixmatch)).float()
                    xs, ms = _strong_view(x, m)
                    z_s = model(xs, ms)
                    bce_u = F.binary_cross_entropy_with_logits(z_s, hard, reduction="none")
                    sel = unl * conf_mask
                    denom = sel.sum().clamp_min(1.0)
                    loss = loss + a.fm_lambda * (bce_u * sel * LW).sum() / denom
                if a.nvum > 0:
                    ri = ridx.to(dev)
                    with torch.no_grad():
                        NV_MEM[ri] = (a.nvum_beta * NV_MEM[ri]
                                      + (1 - a.nvum_beta) * (z.float() - NV_PRIOR))
                    nv_t = torch.sigmoid(NV_MEM[ri])
                    loss = loss + a.nvum * F.binary_cross_entropy_with_logits(
                        z, nv_t, reduction="none").mean()
                if (a.joint_oa > 0 or a.diff_oa > 0) and getattr(model, "_gfeat", None) is not None:
                    _iM, _iL = LABELS.index("Medial OA"), LABELS.index("Lateral OA")
                    _ym, _yl = y[:, _iM].float(), y[:, _iL].float()
                    _ok = ((w[:, _iM] > 0) & (w[:, _iL] > 0)).float()
                    if a.joint_oa > 0:
                        _t = torch.stack([(1 - _ym) * (1 - _yl), _ym * (1 - _yl), (1 - _ym) * _yl, _ym * _yl], 1)
                        _lp = F.log_softmax(model.joint_head(model._gfeat).float(), 1)
                        _ce = -(_t * _lp).sum(1)
                        loss = loss + a.joint_oa * (_ce * _ok).sum() / _ok.sum().clamp_min(1.0)
                    if a.diff_oa > 0:
                        _d = torch.tanh(model.diff_head(model._gfeat).float().squeeze(-1))
                        loss = loss + a.diff_oa * (((_d - (_yl - _ym)) ** 2) * _ok).sum() / _ok.sum().clamp_min(1.0)
                if a.report_clip > 0:
                    if getattr(model, "_gfeat", None) is None:
                        raise SystemExit("--report-clip needs the masked-mean pooling path (Net._gfeat); "
                                         "this arm's pooling never sets it")
                    _ri = ridx.to(dev)
                    _rl, _rt = report_clip_loss(model.rclip_proj(model._gfeat), RCLIP_E[_ri],
                                                RCLIP_OK[_ri], a.report_tau, RCLIP_Q)
                    if _rl is not None:
                        loss = loss + a.report_clip * _rl
                        RCLIP_Q = rclip_enqueue(RCLIP_Q, _rt)
                        rc_tot += float(_rl.detach()); rc_n += 1
                if a.sideaux > 0 and getattr(model, "_feat", None) is not None:
                    fsl = model._feat                              # [b, n, dim]
                    nn_ = fsl.shape[1]
                    st, sm = sideaux_target(nn_, model.n_slots, fsl.device)
                    pred = model.side_head(fsl).squeeze(-1)        # [b, n]
                    # only real (unmasked) sagittal positions contribute
                    sel = sm[None] * m
                    aux = (((pred - st[None]) ** 2) * sel).sum() / sel.sum().clamp_min(1.0)
                    loss = loss + a.sideaux * aux
                if a.lm_slice_aux > 0 and getattr(model, "_feat", None) is not None:
                    _lp = model.lm_head(model._feat).squeeze(-1).float()   # [b, n]
                    _ls = lm_ok * m
                    _lb = F.binary_cross_entropy_with_logits(_lp, lm_t, reduction="none")
                    loss = loss + a.lm_slice_aux * (_lb * _ls).sum() / _ls.sum().clamp_min(1.0)
                loss = loss / a.accum
            scaler.scale(loss).backward()
            if (i + 1) % a.accum == 0:
                if a.clip > 0:
                    scaler.unscale_(opt)          # clip true grads, not scaled ones
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            tot += loss.item() * a.accum
            n += 1

        model.eval()
        P, T = [], []
        with torch.no_grad():
            for x, m, y, *_ in vdl:
                with torch.autocast("cuda", dtype=torch.bfloat16 if a.bf16 else torch.float16):
                    P.append(torch.sigmoid(model(x.to(dev), m.to(dev))).float().cpu().numpy())
                T.append(y.numpy())
        P, T = np.concatenate(P), (np.concatenate(T) > 0.5).astype(int)
        auc = [roc_auc_score(T[:, j], P[:, j]) for j in range(len(LABELS))
               if len(np.unique(T[:, j])) > 1]
        cur = float(np.mean(auc))
        # accuracy against real expert labels (teacher-independent, but small n)
        if _gT is not None and len(_gpos) >= 8:
            Pg = P[_gpos]
            gauc = [roc_auc_score(_gT[:, j], Pg[:, j]) for j in range(len(LABELS))
                    if len(np.unique(_gT[:, j])) > 1]
            cur_truth = float(np.mean(gauc)) if gauc else float("nan")
        else:
            cur_truth = float("nan")
        # Keep the best epoch, not the last. Measured on a 30-epoch run: validation
        # peaks around epoch 22 (0.8748) and drifts down to 0.8726 by epoch 30, so
        # saving the final state silently discards ~0.002. Selecting on the validation
        # fold is ordinary early stopping - that fold is held out of training.
        if a.best and cur > best_auc:
            best_auc, best_state = cur, {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()}
            best_ep = ep + 1
        print(f"ep {ep+1}/{a.epochs} loss {tot/max(n,1):.4f} teach {cur:.4f} "
              f"truth {cur_truth:.4f} [{time.time()-t0:.0f}s]"
              + (f" rclip {rc_tot/max(rc_n,1):.4f}" if a.report_clip > 0 else ""), flush=True)

    if a.best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best epoch {best_ep} (teach {best_auc:.4f}) "
              f"vs final {float(np.mean(auc)):.4f} | final truth {cur_truth:.4f}", flush=True)

    if a.gold_ft > 0:
        # Short, low-LR pass on the gold studies that live in the TRAINING folds. The
        # validation fold's gold is untouched, so the reported metric stays honest.
        gtr = [i for i in tr if studies[i] in gold.index]
        if not gtr:
            print("gold-ft: no gold studies in the training folds, skipping", flush=True)
        else:
            Yg = np.zeros((len(studies), len(LABELS)), np.float32)
            Wg = np.zeros_like(Yg)
            for i in gtr:
                Yg[i] = gold.loc[studies[i]].values
                Wg[i] = 1.0
            gdl = DataLoader(Cached(arr, mask, np.array(gtr), Yg[gtr], Wg[gtr], a.use,
                                    aug=a.aug, aug_mult=a.aug_mult, drop_slots=_drop,
                                    in25d=a.in25d),
                             batch_size=max(a.batch // 2, 1), shuffle=True,
                             num_workers=a.workers)
            gopt = torch.optim.AdamW(model.parameters(), lr=a.lr / 20, weight_decay=0.01)
            model.train()
            for gep in range(a.gold_ft):
                gtot, gn = 0.0, 0
                for x, m, y, w, _ridx in gdl:
                    x, m, y, w = x.to(dev), m.to(dev), y.to(dev), w.to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16 if a.bf16
                                        else torch.float16):
                        gl = (F.binary_cross_entropy_with_logits(
                            model(x, m), y, reduction="none") * w * LW).mean()
                    gopt.zero_grad(set_to_none=True)
                    scaler.scale(gl).backward()
                    scaler.step(gopt)
                    scaler.update()
                    gtot += gl.item(); gn += 1
                print(f"gold-ft {gep+1}/{a.gold_ft} on {len(gtr)} studies "
                      f"loss {gtot/max(gn,1):.4f}", flush=True)
            # re-score so the saved gold_auc reflects the fine-tuned weights
            model.eval()
            P2, T2 = [], []
            with torch.no_grad():
                for x, m, y, *_ in vdl:
                    with torch.autocast("cuda", dtype=torch.bfloat16 if a.bf16
                                        else torch.float16):
                        P2.append(torch.sigmoid(model(x.to(dev), m.to(dev))).float()
                                  .cpu().numpy())
                    T2.append(y.numpy())
            P2, T2 = np.concatenate(P2), (np.concatenate(T2) > 0.5).astype(int)
            a2 = [roc_auc_score(T2[:, j], P2[:, j]) for j in range(len(LABELS))
                  if len(np.unique(T2[:, j])) > 1]
            best_auc = float(np.mean(a2))
            print(f"after gold-ft: {best_auc:.4f}", flush=True)
    _sd = (model._orig_mod.state_dict() if hasattr(model, "_orig_mod")
           else model.state_dict())
    if getattr(model, "posw", False):
        # READ OUT what the model learned. If the compartment story is true, Medial OA should
        # weight LOW indices (medial after --lat-norm) and Lateral OA HIGH ones. This is a
        # direct test of the r=0.747 vs r=0.317 diagnosis, independent of the score.
        _pt = model.pos_table.detach().softmax(1).cpu().numpy()
        print("POSW TABLE (per label, medial -> lateral across the sagittal slot):", flush=True)
        for _i, _l in enumerate(LABELS):
            _com = float((_pt[_i] * np.arange(_pt.shape[1])).sum() / max(_pt.shape[1] - 1, 1))
            print(f"  {_l:18s} CoM {_com:.3f}  " + " ".join(f"{v:.3f}" for v in _pt[_i]), flush=True)

    _rclip_sd = None
    if a.report_clip > 0:
        # training-only head: keep it out of ck["model"] so strict loaders (kernels) and
        # score_moaks.py (exits on unexpected keys without --lenient) see the usual key set
        _rclip_sd = {k[len("rclip_proj."):]: v.detach().cpu() for k, v in _sd.items()
                     if k.startswith("rclip_proj.")}
        _sd = {k: v for k, v in _sd.items() if not k.startswith("rclip_proj.")}
    _lm_sd = None
    if a.lm_slice_aux > 0:
        # training-only head, kept out of ck["model"] like rclip_proj (strict kernel loaders)
        _lm_sd = {k[len("lm_head."):]: v.detach().cpu() for k, v in _sd.items()
                  if k.startswith("lm_head.")}
        _sd = {k: v for k, v in _sd.items() if not k.startswith("lm_head.")}
    torch.save({"model": _sd, "arch": a.arch, "meta": meta,
                "compswap": float(a.compswap), "compswap_sag": a.compswap_sag,
                "compswap_ax": a.compswap_ax,
                "compswap_cor_medial": a.corsplit_medial if a.compswap > 0 else None,
                "lm_slice_aux": float(a.lm_slice_aux), "lm_slice_targets": a.lm_slice_targets,
                "lm_head": _lm_sd,
                "report_clip": float(a.report_clip), "report_emb": a.report_emb,
                "report_key": a.report_key, "report_tau": float(a.report_tau),
                "report_queue": RCLIP_QUEUE if a.report_clip > 0 else 0,
                "rclip_proj": _rclip_sd,
                "keras_v1_strides": bool(a.init), "gray_fold": a.gray_fold,
                "clip": a.clip, "bf16": a.bf16, "focal_k": a.focal_k, "full_fit": a.full,
                "conf_weight": a.conf_weight, "seq": a.seq, "in25d": a.in25d, "qhead": a.qhead, "gru": a.gru, "gold_ft": a.gold_ft, "maxpool": a.maxpool, "planepool": a.planepool,
                "use": a.use, "seed": a.seed, "labels": a.labels, "no_gold": a.no_gold,
                "train_kper": a.train_kper, "kper": _kper,
                "corsplit": bool(a.corsplit), "corsplit_medial": a.corsplit_medial,
                "corsplit_slots": list(getattr(model, "corsplit_slots", []) or []),
                "truth_auc": float(cur_truth), "teacher_auc": float(best_auc if (a.best and best_state is not None) else np.mean(auc)),
                "best_epoch": best_ep},
               os.path.join(WORK, out))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
