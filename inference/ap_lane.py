"""Submission kernel v8: cross-config mega-ensemble, rank-blended.

Four lineages x 5 folds = 20 checkpoints, each run under its own preprocessing
(read from its checkpoint meta), rank-averaged per label. All lineages share the
6-slot plane x Fluid_Sensitive scheme, so every DICOM decodes once and only the
crop/resize/flip differ per config. DINOv2 loads from a bundled torch-hub cache
because submission kernels have no internet. Gate OOF-58: 0.8090.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
          'Contusion', 'Fracture']
def _pick_device():
    """Fail in the first seconds, not 13 minutes in.

    Kaggle assigns either a T4 (sm_75) or a P100 (sm_60), and their current torch build
    has no sm_60 kernels - a P100 draw dies with 'no kernel image is available' only once
    the first conv runs, after the DICOM decode has already burned GPU quota. Probing a
    real conv up front turns a 13-minute loss into a 5-second one, and CPU is used only
    if it can plausibly finish (SLICE_BANDS is cut to 1 in that case).
    """
    if not torch.cuda.is_available():
        print("NO CUDA - falling back to CPU", flush=True)
        return torch.device("cpu"), True
    try:
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name()
        probe = torch.nn.Conv2d(1, 4, 3).cuda()
        probe(torch.zeros(1, 1, 8, 8, device="cuda")).sum().item()
        torch.cuda.synchronize()
        print(f"GPU OK: {name} sm_{cap[0]}{cap[1]}", flush=True)
        return torch.device("cuda"), False
    except Exception as e:
        print(f"GPU UNUSABLE ({torch.cuda.get_device_name()}): {e}", flush=True)
        print("falling back to CPU with a single slice band", flush=True)
        return torch.device("cpu"), True


DEV, _CPU_FALLBACK = _pick_device()
# Flip together with the cache used for training. False = the 0.935 lineage.
ANTIALIAS = True
# EFFICIENCY-TRACK LEAN CONFIG: 15 models + ft arm ONLY (1 of 3 arms - arms are the expensive part, 42 windows each)
# FIFTH LINEAGE: ConvNeXt-tiny @384 trained with --usmooth 0.25 (stochastic
# uncertainty-targeted label smoothing). A matched fold-0 A/B measured +0.0032 teacher
# / +0.0031 truth over an identical control - right at the 0.0037 seed-noise floor, so
# the leaderboard is the only instrument that can settle it. Per-fold teacher AUC:
# 0.8818 0.8741 0.8888 0.8800 0.8822.
SETS = [("wNG_f", "resnet"), ("wCNX_f", "convnext"), ("wAA_f", "convnext"), ("wUS5_f", "convnext")]

# HYBRID: override with the lineages this account can attach, e.g. AP_SETS="wCNX_f:convnext,wFDP_f:convnext"
if os.environ.get("AP_SETS"):
    SETS = [tuple(x.split(":")) for x in os.environ["AP_SETS"].split(",")]
# HYBRID per-label members (2026-09-26): lineages loaded and run in the SAME pass (no second decode of the study),
# kept OUT of the fused AP side, and written one file per lineage (member_<prefix>.csv, same rank units) for the
# orchestrator to fuse into single labels - e.g. the OAI-trained Lateral Meniscus / Synovitis columns. Unset -> no-op.
MEMBER_SETS = [tuple(x.split(":")) for x in os.environ.get("AP_MEMBER_SETS", "").split(",") if x]
_MEMBER = {p for p, _ in MEMBER_SETS} - {p for p, _ in SETS}
SETS = SETS + [s for s in MEMBER_SETS if s[0] in _MEMBER]
# HYBRID: input root is overridable for LOCAL pre-flight only. Unset on Kaggle -> /kaggle/input,
# so every path below resolves exactly as in his submitted kernel.
_KIN = os.environ.get("KAGGLE_INPUT", "/kaggle/input")


def _mount_glob(name):
    """Search dataset mounts only. NEVER recursive-glob the competition DICOM tree: one
    '/kaggle/input/**/x' walk costs ~4 min on the hidden test set (measured ~770 s for
    three of them). Dataset mounts are small, so a per-mount ** is cheap."""
    out = []
    for d in sorted(glob.glob(f"{_KIN}/*/")):
        if "rsna-knee-abnormality-detection" in d or "competition" in d.lower():
            continue
        out += glob.glob(os.path.join(d, "**", name), recursive=True)
    return out


def find(pattern):
    name = pattern.split("/**/")[-1]
    if name in ("test.csv", "test_series.csv", "sample_submission.csv"):
        for b in (f"{_KIN}/competitions/rsna-knee-abnormality-detection",
                  f"{_KIN}/rsna-knee-abnormality-detection"):
            if os.path.exists(b + "/" + name):
                return b + "/" + name
    hits = _mount_glob(name)
    if not hits:                                  # last resort only: the old full walk
        hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


COMP = os.path.dirname(find(f"{_KIN}/**/test.csv"))
DS = os.path.dirname(find(f"{_KIN}/**/{SETS[0][0]}0.pt"))



class ResNet_(nn.Module):
    def __init__(self, arch, keras_v1=False):
        super().__init__()
        import torchvision
        m = getattr(torchvision.models, arch)(weights=None)
        if keras_v1:
            # RadImageNet-initialised checkpoints were trained as Keras-style v1:
            # stride on the leading 1x1. Must match or the weights load cleanly
            # and predict from the wrong receptive field.
            for layer in (m.layer2, m.layer3, m.layer4):
                layer[0].conv1.stride, layer[0].conv2.stride = (2, 2), (1, 1)
        dim = m.fc.in_features
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.head = nn.Linear(dim, len(LABELS))
        self.dim = dim

    def forward(self, x, m):
        b, n = x.shape[:2]
        f = self.backbone(x.reshape(b * n, 1, *x.shape[2:])).reshape(b, n, self.dim)
        w = m[..., None]
        mean_p = self.head((f * w).sum(1) / w.sum(1).clamp(min=1))
        if not FOCAL_K:
            return mean_p
        # Baker's, Contusion and Fracture are focal: they occupy a few of the ~48 slices
        # and a flat mean dilutes them. Measured on two independent 5-fold lineages,
        # top-3 pooling on exactly these three gains +0.0036 and +0.0078 macro; the nine
        # diffuse targets lose badly under top-k so they keep the mean. The head is
        # linear, so mean-pooling logits here is identical to mean-pooling features.
        L = self.head(f).masked_fill(m[:, :, None] < 0.5, -1e4)
        # PER-LABEL POOLING WIDTH. The old test was binary: top-3 or mean-over-~48. Top-3 is far
        # too sharp for a compartment finding, which is why "the nine diffuse targets lose badly".
        # But OA IS focal in the slice dimension - medial OA lives only on medial slices, so a flat
        # mean dilutes it by roughly half. K_PER gives each label its own width; 0 keeps the mean.
        out = mean_p.clone()
        for kk in sorted({v for v in K_PER if v > 0}):
            cols = [i for i, v in enumerate(K_PER) if v == kk]
            tp = L.topk(min(kk, n), dim=1).values.mean(1)
            out[:, cols] = tp[:, cols]
        return out


class DinoNet(nn.Module):
    """Mirrors the training dinov2 branch: grey 336px slices expanded to 3ch, features
    are CLS concatenated with the patch-token mean (hence 2x embed_dim), then the same
    masked-mean + top-3-focal pooling every other set uses. The earlier version of this
    class pooled plain backbone() output into an embed_dim head, which matches neither
    the checkpoint shapes nor the training forward pass."""

    def __init__(self):
        super().__init__()
        # The dinov2 source tree ships as a dataset (kernels have no internet); load it
        # with source="local" and pretrained=False - the fine-tuned weights come from our
        # own checkpoint, so fetching Facebook's pretrained file would be both impossible
        # here and pointless.
        # Locate by hubconf.py: the dataset upload flattens the repo, so there is no
        # facebookresearch_dinov2_* directory to match on - only the tree's contents.
        conf = [p for p in _mount_glob("hubconf.py")
                if os.path.isdir(os.path.join(os.path.dirname(p), "dinov2"))]
        hits = ([os.path.dirname(p) for p in conf]
                or glob.glob("/root/.cache/torch/hub/facebookresearch_dinov2_*"))
        if not hits:
            raise RuntimeError(
                "dinov2 source tree not found; /kaggle/input has: "
                + str(sorted(glob.glob(f"{_KIN}/*"))))
        self.backbone = torch.hub.load(hits[0], "dinov2_vits14", source="local",
                                       pretrained=False)
        self.dim = self.backbone.embed_dim * 2
        self.head = nn.Linear(self.dim, len(LABELS))

    def forward(self, x, m):
        b, n = x.shape[:2]
        xin = x.reshape(b * n, 1, *x.shape[2:]).expand(-1, 3, -1, -1)
        o = self.backbone.forward_features(xin)
        f = torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], dim=-1)
        f = f.reshape(b, n, self.dim)
        w = m[..., None]
        mean_p = self.head((f * w).sum(1) / w.sum(1).clamp(min=1))
        if not FOCAL_K:
            return mean_p
        L = self.head(f).masked_fill(m[:, :, None] < 0.5, -1e4)
        # Same per-label width as ResNet_. 10 of the 15 checkpoints are CNXNet (wCNX, wAA),
        # so leaving this branch on the old binary rule tested K_PER on a THIRD of the ensemble.
        out = mean_p.clone()
        for kk in sorted({v for v in K_PER if v > 0}):
            cols = [i for i, v in enumerate(K_PER) if v == kk]
            tp = L.topk(min(kk, n), dim=1).values.mean(1)
            out[:, cols] = tp[:, cols]
        return out


class CNXNet(nn.Module):
    """Mirrors the training net's no-fc branch (convnext/efficientnet): keep the
    pretrained 3-channel stem, expand grey to 3ch at forward, last Linear ->
    Identity so backbone() returns flattened features. Pooling identical to
    ResNet_: masked mean + top-3 on the three focal labels (linear head, so
    mean-pooling logits == mean-pooling features)."""

    def __init__(self, arch="convnext_tiny"):
        super().__init__()
        import torchvision
        if arch.startswith("timm:"):
            # wUS5 was trained with --arch timm:convnext_tiny (the timm branch, which is what
            # enables gradient checkpointing). timm and torchvision ConvNeXt have DIFFERENT
            # state dicts - timm uses backbone.stem/stages, torchvision backbone.features - so
            # the checkpoint only loads under the implementation it was trained with.
            # Verified locally: timm build loads it with 0 missing / 0 unexpected keys.
            import timm
            m = timm.create_model(arch[5:], pretrained=False, num_classes=0)
            self.backbone = m
            self.head = nn.Linear(m.num_features, len(LABELS))
            self.dim = m.num_features
            return
        m = getattr(torchvision.models, arch)(weights=None)
        mods = dict(m.named_modules())
        lin = [k for k, v in mods.items() if isinstance(v, nn.Linear)][-1]
        dim = mods[lin].in_features
        parent = m if "." not in lin else mods[lin.rsplit(".", 1)[0]]
        setattr(parent, lin.rsplit(".", 1)[-1], nn.Identity())
        self.backbone = m
        self.head = nn.Linear(dim, len(LABELS))
        self.dim = dim

    def forward(self, x, m, lm=None):
        b, n = x.shape[:2]
        xin = x.reshape(b * n, 1, *x.shape[2:]).expand(-1, 3, -1, -1)
        f = self.backbone(xin).reshape(b, n, self.dim)
        w = m[..., None]
        if lm is not None:   # ATLAS (v12): per-label token weights; head is linear so mean of logits == head(mean f)
            W = w * lm
            Lr = self.head(f)
            mean_p = ((Lr * W).sum(1) / W.sum(1).clamp(min=1)).to(Lr.dtype)
            if not FOCAL_K:
                return mean_p
            L = Lr.masked_fill(W < 0.5, -1e4)
        else:
            mean_p = self.head((f * w).sum(1) / w.sum(1).clamp(min=1))
            if not FOCAL_K:
                return mean_p
            L = self.head(f).masked_fill(m[:, :, None] < 0.5, -1e4)
        # Same per-label width as ResNet_. 10 of the 15 checkpoints are CNXNet (wCNX, wAA),
        # so leaving this branch on the old binary rule tested K_PER on a THIRD of the ensemble.
        out = mean_p.clone()
        for kk in sorted({v for v in K_PER if v > 0}):
            cols = [i for i, v in enumerate(K_PER) if v == kk]
            tp = L.topk(min(kk, n), dim=1).values.mean(1)
            out[:, cols] = tp[:, cols]
        return out


def n_folds(pref):
    """However many folds actually exist. Test studies have no fold assignment - every
    checkpoint predicts every study and the ranks are averaged - so a 3-fold set is a
    valid, slightly smaller ensemble rather than a broken one."""
    return sorted(int(re.search(r"_f(\d+)\.pt$", p).group(1))
                  for p in glob.glob(f"{ds_for(pref)}/{pref}*.pt")
                  if re.search(r"_f(\d+)\.pt$", p))


_DSC = {}


def ds_for(pref):
    """Resolve each lineage's own dataset dir. The original code assumed every weight
    file sat in one directory (true while all lineages shipped in one dataset); wCSOFT
    is uploaded separately, so resolve per prefix instead of reusing SETS[0]'s dir."""
    if pref not in _DSC:
        hit = find(f"{_KIN}/**/{pref}0.pt")
        _DSC[pref] = os.path.dirname(hit) if hit else DS
    return _DSC[pref]

models, metas = {}, {}
for pref, kind in SETS:
    folds = n_folds(pref)
    print(f"{pref}: {len(folds)} folds {folds}", flush=True)
    for Fo in folds:
        ck = torch.load(f"{ds_for(pref)}/{pref}{Fo}.pt", map_location="cpu")
        m = (ResNet_(ck.get("arch", "resnet18"), ck.get("keras_v1_strides", False))
             if kind == "resnet"
             else CNXNet(ck.get("arch", "convnext_tiny")) if kind == "convnext"
             else DinoNet())
        # sideaux/posw checkpoints carry AUXILIARY parameters the inference model has no
        # slot for: side_head.* (the per-slice M/L regressor) and pos_table (the learned
        # position weights). Training-time only, so drop what the model does not declare.
        _sd = {k: v for k, v in ck["model"].items() if k in m.state_dict()}
        _drop = [k for k in ck["model"] if k not in _sd]
        if _drop:
            print(f"  dropping {len(_drop)} training-only keys: {_drop[:3]}", flush=True)
        m.load_state_dict(_sd, strict=True)
        m.eval().to(DEV)
        models[(pref, Fo)] = m
        metas[pref] = ck
print(f"{len(models)} checkpoints", flush=True)

SLOTS = [(p, f) for p in ["Sagittal", "Coronal", "Axial"] for f in (1, 0)]
# gate on 58 gold: single band 0.8823 -> 3 bands 0.8845
SLICE_BANDS = [(0, 15)]  # 3-band TTA LB-bracketed 2026-08-20: exact tie at 3x cost, dropped
if _CPU_FALLBACK:
    # 3x the passes is unaffordable on CPU; keep the full band, drop the TTA.
    SLICE_BANDS = [(0, 15)]
# Per-target pooling. Ours used top-3 for three labels and mean for the rest; the public
# 0.89-0.90 kernels converge independently on a finer split, justified by lesion extent:
# a fracture line or meniscal tear lives in ONE window and averaging dilutes it, whereas
# effusion and OA are diffuse and the mean is the better estimator.
#   max        Fracture, Contusion, Medial+Lateral Meniscus, Baker's
#   top-2 mean ACL, MCL
#   mean       Medial/Lateral/PF OA, Effusion, Synovitis
POOL_MAX = {"Fracture", "Contusion", "Medial Meniscus", "Lateral Meniscus", "Baker's"}
POOL_TOP2 = {"ACL", "MCL"}
POOL_K = [1 if c in POOL_MAX else (2 if c in POOL_TOP2 else 0) for c in LABELS]
FOCAL_ORIG = {"Baker's", "Contusion", "Fracture"}
FOCAL_MASK_ORIG = [c in FOCAL_ORIG for c in LABELS]
FOCAL = FOCAL_ORIG
FOCAL_MASK = FOCAL_MASK_ORIG
FOCAL_K = 3
# per-label pooling width: focal three keep top-3, the OA compartments get top-16
# of ~48 slices (between top-3 and the flat mean), everything else stays mean.
# POOLING WIDTH BY LESION TYPE. A tear, cyst, contusion or fracture is a DISCRETE lesion living
# in a few slices -> pool sharply. OA is a COMPARTMENT process on one side of the joint ->
# moderate. Effusion and Synovitis are joint-wide -> keep the flat mean. The old test only ever
# compared top-3 against mean-over-48, so everything between was unexamined.
# DIFFUSE SHARPENED: Effusion and Synovitis have only ever been flat-mean. top-24 of ~96 is a
# mild sharpening - far from the top-3 that 'loses badly' for diffuse targets, but no longer
# an unweighted average over every slot.
# DIFFUSE SHARPER STILL: top-24 beat the flat mean (+0.001) and top-48 was worse, so the
# gradient points toward sharper. 12 is the untested side of the winner.
K_PER = [6, 6, 6, 6, 16, 16, 16, 12, 12, 3, 3, 3]   # ACL:6, MCL:6, Medial Meniscus:6, Lateral Meniscus:6, Medial OA:16, Lateral OA:16, PF OA:16, Effusion:12, Synovitis:12, Baker's:3, Contusion:3, Fracture:3
# HYBRID (from AP @0644649): AP_KPER_MEAN=1 -> mean pooling for every label (K=0 keeps mean_p). MOAKS 4,336 knees,
# wCNX x5: mean 0.8306 vs this K_PER 0.7851; gold-58 plain mean +0.016. Default off = unchanged behaviour.
if os.environ.get("AP_KPER_MEAN") == "1":
    K_PER = [0] * 12


def geo_side(d, lim_mm=20.0):
    """'L'/'R'/None from scanner geometry, when the Laterality tag is absent.

    THIS WAS MISSING AND IT WAS A REAL BUG. build_cache.py falls back to this whenever the
    tag is empty (--geo-lat, recorded as '_gl' in the cache name), so TRAINING canonicalised
    those knees to one side. The kernel had no fallback and left them unmirrored - and the
    tag is absent on roughly 42% of studies. Nearly half the test set was therefore fed in
    the opposite orientation to what the model learned, wrecking the four medial/lateral
    labels (MCL, both menisci, medial and lateral OA). Invisible to every local metric,
    because they all read the cache, which is internally consistent.

    Patient +x is the patient's LEFT, so the FOV centre's x-coordinate gives the side when
    it sits clearly off the midline. Sagittal: slice position IS x. Coronal/axial: centre =
    origin + row_dir * cols/2, used only when rows are cleanly x-aligned. Ambiguous -> None,
    matching build_cache's 20mm dead zone exactly."""
    try:
        ipp = [float(v) for v in getattr(d, "ImagePositionPatient", []) or []]
        iop = [float(v) for v in getattr(d, "ImageOrientationPatient", []) or []]
        if len(ipp) < 1 or len(iop) < 6:
            return None
        rx_ = iop[0]
        if abs(rx_) < 0.2:
            cx = ipp[0]
        elif abs(rx_) > 0.9:
            cols = int(getattr(d, "Columns", 0) or 0)
            pxs = getattr(d, "PixelSpacing", None)
            if not cols or not pxs:
                return None
            cx = ipp[0] + rx_ * cols * float(pxs[0]) / 2.0
        else:
            return None
        if cx > lim_mm:
            return "L"
        if cx < -lim_mm:
            return "R"
        return None
    except Exception:
        return None


ATLAS = {x for x in os.environ.get("AP_ATLAS", "").split(",") if x}   # v12: prefixes whose compartment labels pool their own sagittal half
ATLAS_LAB = {"Medial Meniscus": "M", "Medial OA": "M", "Lateral Meniscus": "L", "Lateral OA": "L"}


def ipp_x(path):
    try:
        d = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        v = getattr(d, "ImagePositionPatient", None)
        return float(v[0]) if v is not None and len(v) >= 1 else None
    except Exception:
        return None


def hdr(path):
    try:
        d = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        lat = str(getattr(d, "Laterality", "") or "").strip().upper()[:1]
        side = lat if lat in ("L", "R") else geo_side(d)
        return int(getattr(d, "InstanceNumber", 0) or 0), side
    except Exception:
        return 0, None


def decode(path):
    try:
        ds = pydicom.dcmread(path, force=True)
        a = ds.pixel_array.astype(np.float32)
        a = a * float(getattr(ds, "RescaleSlope", 1) or 1) \
            + float(getattr(ds, "RescaleIntercept", 0) or 0)
        px = getattr(ds, "PixelSpacing", None)
        return a, (float(px[0]) if px else None)
    except Exception:
        return None, None


# Lineages differ in preprocessing, so the flag must be per-set, not global: wNG/wNGB/wCNX
# were trained on plain bilinear and wAA on antialiased. Feeding a lineage the wrong one is
# the same class of mismatch as the laterality bug. Folds built before build_cache.py
# recorded the flag fall back to this prefix list.
AA_PREFIXES = {"wAA_f"}


def transform(a, px, img, crop, off, antialias=False):
    if a is None:
        return torch.zeros(img, img)
    h, w = a.shape
    if px and crop:
        want = int(round(crop / px))
        if 16 < want < min(h, w):
            half = want // 2
            cy = min(max(h // 2 + int(round(off[0] / px)), half), h - half)
            cx = min(max(w // 2 + int(round(off[1] / px)), half), w - half)
            a = a[cy - half:cy + half, cx - half:cx + half]
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(a))[None, None]
    # must match the cache the weights were trained on - see ANTIALIAS in build_cache.py
    return F.interpolate(t, (img, img), mode="bilinear", align_corners=False,
                         antialias=antialias)[0, 0]


def centroid(a, px):
    if a is None or not px:
        return (0.0, 0.0)
    t = a > np.percentile(a, 60)
    ys, xs = np.nonzero(t)
    if len(ys) < 100:
        return (0.0, 0.0)
    h, w = a.shape
    dy, dx = (ys.mean() - h / 2) * px, (xs.mean() - w / 2) * px
    return (0.0, 0.0) if (abs(dy) > 35 or abs(dx) > 35) else (dy, dx)


test = pd.read_csv(f"{COMP}/test.csv")
ser = pd.read_csv(f"{COMP}/test_series.csv")
plane_of = dict(zip(ser.SeriesInstanceUID, ser.Anatomical_Plane))
def series_fluid(path):
    """Fluid-sensitivity from TR, matching build_cache.series_class().

    SECOND train/inference divergence, same class as the laterality bug. build_cache
    derives this from header physics (TR < 800 -> T1-weighted -> not fluid-sensitive)
    because the CSV's Fluid_Sensitive and Fat_Suppression columns are IDENTICAL - one bit
    where the physics needs two. The kernel was still reading that CSV column, so ~20% of
    series were assigned to a different slot at test time than in training: a series the
    model learned as sagittal-fluid arrives as sagittal-T1 and vice versa.
    """
    try:
        d = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        tr = float(getattr(d, "RepetitionTime", 0) or 0)
        if tr <= 0:
            return None
        return 0 if tr < 800 else 1          # T1-weighted is not fluid-sensitive
    except Exception:
        return None


fluid_of = dict(zip(ser.SeriesInstanceUID, ser.Fluid_Sensitive))
print(f"{len(test)} studies", flush=True)

# ---- v13: SKM-TEA segmentation-guided compartment crop member (mirrors skm_crops.py + crop_predict.py) ----
class _CropNet(nn.Module):
    def __init__(self, arch="convnext_tiny", nout=4):
        super().__init__()
        import timm
        self.b = timm.create_model(arch, pretrained=False, num_classes=0)
        self.h = nn.Linear(2 * self.b.num_features, nout)
    def forward(self, x):                                   # [B, S, 3, H, W]
        B, S_ = x.shape[:2]; f = self.b(x.flatten(0, 1)).view(B, S_, -1)
        return self.h(torch.cat([f.mean(1), f.amax(1)], 1))


def _crop_find(fname):
    d0 = os.environ.get("CROP_DIR")
    if d0 and os.path.exists(os.path.join(d0, fname)):
        return os.path.join(d0, fname)
    for d in sorted(glob.glob("/kaggle/input/*/")):
        if "competition" in d.lower() or "abnormality-detection" in d.lower():
            continue
        h = glob.glob(os.path.join(d, "**", fname), recursive=True)
        if h:
            return h[0]
    raise RuntimeError(fname + " not found")


class CropMember:
    R, SIDE, CP, NS, OUT = 384, 312, 165, 8, 224   # 130 mm / 384 px cache: U-Net at 0.4167 mm -> 312 px; 56 mm crop = 165 px
    def __init__(self, dev):
        self.dev = dev
        self.unet = torch.jit.load(_crop_find("unet_skm_ts16.pt"), map_location=dev).eval()
        def _load(tag, arch, nout):
            out = []
            for f in range(5):
                s_ = torch.load(_crop_find(f"w{tag}_f{f}_fp16.pt"), map_location="cpu", weights_only=False)
                m_ = _CropNet(s_.get("arch", arch), nout); m_.load_state_dict({k: v.float() for k, v in s_["model"].items()})
                out.append(m_.to(dev).eval())
            return out
        self.nets = _load("CROP", "convnext_tiny", 4)
        self.netsB = _load("CROPB", "convnext_small", 4) if os.environ.get("AP_CROPB") == "1" else []   # v15: crop-B averaged with v1
        self.nets2 = _load("CROP2", "convnext_tiny", 3) if os.environ.get("AP_CROP2") == "1" else []   # v15: PF + notch crops
        self.nets3 = _load("CROP3", "convnext_tiny", 3) if os.environ.get("AP_CROP3") == "1" else []   # v16: medial-edge + bone crops (Baker's used)
        print(f"CROP nets v1 {len(self.nets)} B {len(self.netsB)} v2 {len(self.nets2)} v3 {len(self.nets3)}", flush=True)
    @torch.no_grad()
    def __call__(self, vol):                                # vol [16, 384, 384] float in [0, 1], cache-equivalent slices
        v = (vol.clamp(0, 1) * 255).round().numpy().astype(np.float32) / 255.0   # uint8 cache quantisation
        t = torch.from_numpy(v)[:, None].to(self.dev)
        t = F.interpolate(t, size=(self.SIDE, self.SIDE), mode="bilinear", align_corners=False)
        p = (self.R - self.SIDE) // 2; t = F.pad(t, (p, self.R - self.SIDE - p, p, self.R - self.SIDE - p))
        lab = self.unet(t.repeat(1, 3, 1, 1).half()).argmax(1)
        lab = lab[:, p:p + self.SIDE, p:p + self.SIDE].float()[:, None]
        lab = F.interpolate(lab, size=v.shape[1:], mode="nearest")[:, 0].long().cpu().numpy()
        X_ = np.zeros((2, self.NS, self.OUT, self.OUT), np.float32)
        for c, m in enumerate((np.isin(lab, (3, 5)), np.isin(lab, (4, 6)))):
            a = m.sum((1, 2)); sl = np.sort(np.argsort(-a)[:self.NS])
            yy, xx = np.nonzero(m.any(0)) if m.any() else (np.array([v.shape[1] // 2]), np.array([v.shape[2] // 2]))
            cy0, cx0 = yy.mean(), xx.mean()
            for q, k in enumerate(sl):
                if a[k] > 0:
                    ys, xs = np.nonzero(m[k]); cy, cx = ys.mean(), xs.mean()
                else:
                    cy, cx = cy0, cx0
                y0, x0 = int(round(cy - self.CP / 2)), int(round(cx - self.CP / 2)); CP = self.CP
                img = np.pad(v[k], CP, mode="constant")[y0 + CP:y0 + 2 * CP, x0 + CP:x0 + 2 * CP]
                X_[c, q] = (F.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None], size=(self.OUT, self.OUT),
                                          mode="bilinear", align_corners=False)[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8) / 255.0
        self.last_X = X_   # v26: kept for the two-plane member
        x = torch.from_numpy(X_)[:, :, None].repeat(1, 1, 3, 1, 1).to(self.dev)
        def _run(nets, xx):
            o_ = 0
            for n_ in nets:
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev.type == "cuda"):
                    o_ = o_ + torch.sigmoid(n_(xx).float())
            return (o_ / len(nets)).cpu().numpy()
        o = _run(self.nets, x)
        if self.netsB:
            o = 0.5 * (o + _run(self.netsB, x))
        r = [o[0, 0], o[0, 1], o[1, 2], o[1, 3]]    # Medial Meniscus, Medial OA, Lateral Meniscus, Lateral OA
        if self.nets2:   # v15 crop v2 (mirrors skm_crops_v2.py): PF crop -> PF OA, Effusion; notch crop -> ACL
            CP2 = int(round(64.0 / (130.0 / 384))); up_pf, up_n = int(round(10.0 / (130.0 / 384))), int(round(8.0 / (130.0 / 384)))
            S_, H_, W_ = v.shape
            def cut2(k, cy, cx):
                y0, x0 = int(round(cy - CP2 / 2)), int(round(cx - CP2 / 2))
                img = np.pad(v[k], CP2, mode="constant")[y0 + CP2:y0 + 2 * CP2, x0 + CP2:x0 + 2 * CP2]
                return (F.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None], size=(self.OUT, self.OUT), mode="bilinear",
                                      align_corners=False)[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8) / 255.0
            X2 = np.zeros((2, self.NS, self.OUT, self.OUT), np.float32)
            P_ = lab == 1; aP = P_.sum((1, 2)); slP = np.sort(np.argsort(-aP)[:self.NS])
            if P_.any(): yy, xx = np.nonzero(P_.any(0)); cy0, cx0 = yy.mean(), xx.mean()
            else: cy0, cx0 = H_ * 0.4, W_ * 0.35
            for q, k in enumerate(slP):
                if aP[k] > 0: ys, xs = np.nonzero(P_[k]); cy, cx = ys.mean(), xs.mean()
                else: cy, cx = cy0, cx0
                X2[0, q] = cut2(k, cy - up_pf, cx)
            aM = np.isin(lab, (3, 5)).sum((1, 2)).astype(float); aL = np.isin(lab, (4, 6)).sum((1, 2)).astype(float); kk = np.arange(S_)
            mid = 0.5 * ((aM * kk).sum() / aM.sum() + (aL * kk).sum() / aL.sum()) if aM.sum() > 0 and aL.sum() > 0 else (S_ - 1) / 2
            slN = np.sort(np.argsort(np.abs(kk - mid), kind="stable")[:self.NS])
            T_ = np.isin(lab, (3, 4))
            if T_.any(): yy, xx = np.nonzero(T_.any(0)); ty, tx = yy.mean(), xx.mean()
            else: ty, tx = H_ * 0.6, W_ * 0.5
            for q, k in enumerate(slN):
                X2[1, q] = cut2(k, ty - up_n, tx)
            o2 = _run(self.nets2, torch.from_numpy(X2)[:, :, None].repeat(1, 1, 3, 1, 1).to(self.dev))
            r += [o2[0, 0], o2[0, 1], o2[1, 2]]    # PF OA, Effusion, ACL
        if self.nets3:   # v16 crop v3 (mirrors skm_crops_v3.py): medial-edge crop -> MCL, Baker's; bone crop -> Contusion
            cpx_ = 130.0 / 384; CPE, CPB, BACK = int(round(80.0 / cpx_)), int(round(96.0 / cpx_)), int(round(10.0 / cpx_))
            S_, H_, W_ = v.shape
            def cut3(k, cy, cx, CPx):
                y0, x0 = int(round(cy - CPx / 2)), int(round(cx - CPx / 2))
                img = np.pad(v[k], CPx, mode="constant")[y0 + CPx:y0 + 2 * CPx, x0 + CPx:x0 + 2 * CPx]
                return (F.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None], size=(self.OUT, self.OUT), mode="bilinear",
                                      align_corners=False)[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8) / 255.0
            aM = np.isin(lab, (3, 5)).sum((1, 2)).astype(float); aL = np.isin(lab, (4, 6)).sum((1, 2)).astype(float); kk = np.arange(S_)
            if aM.sum() > 0 and aL.sum() > 0: sM = (aM * kk).sum() / aM.sum(); sL = (aL * kk).sum() / aL.sum()
            else: sM, sL = 3.0, 12.0
            sg = 1.0 if sM >= sL else -1.0; d = (kk - sM) * sg
            cand = [k for k in np.argsort(np.where(d >= -0.5, d, 1e9), kind="stable") if d[k] >= -0.5][:self.NS]
            while len(cand) < self.NS: cand.append(cand[-1] if cand else int(round(sM)))
            slE = np.sort(np.array(cand)); TM = lab == 3; T_ = np.isin(lab, (3, 4))
            if T_.any(): yy, xx = np.nonzero(T_.any(0)); ty, tx = yy.mean(), xx.mean()
            else: ty, tx = H_ * 0.6, W_ * 0.5
            X3 = np.zeros((2, self.NS, self.OUT, self.OUT), np.float32)
            for q, k in enumerate(slE):
                if TM[k].any(): ys, xs = np.nonzero(TM[k]); cy, cx = ys.mean(), xs.mean()
                else: cy, cx = ty, tx
                X3[0, q] = cut3(k, cy, cx + BACK, CPE)
            slB = np.sort(np.concatenate([np.argsort(np.abs(kk - sM), kind="stable")[:4], np.argsort(np.abs(kk - sL), kind="stable")[:4]]))
            for q, k in enumerate(slB):
                if T_[k].any(): ys, xs = np.nonzero(T_[k]); cy, cx = ys.mean(), xs.mean()
                else: cy, cx = ty, tx
                X3[1, q] = cut3(k, cy, cx, CPB)
            o3 = _run(self.nets3, torch.from_numpy(X3)[:, :, None].repeat(1, 1, 3, 1, 1).to(self.dev))
            r += [o3[0, 0], o3[0, 1], o3[1, 2]]    # MCL, Baker's, Contusion
        if os.environ.get("AP_SKMEV") == "1":   # v16: SKM meniscus intensity evidence (mean normalised signal inside each predicted meniscus)
            r += [float(v[lab == 5].mean()) if (lab == 5).any() else np.nan, float(v[lab == 6].mean()) if (lab == 6).any() else np.nan]
        return r



class CorCropMember:   # v22: coronal compartment crops (mirrors skm_crops_cor.py; cache coronal orientation = identity on RSNA)
    R, SIDE, CP, NS, OUT = 384, 312, 165, 8, 224
    def __init__(self, dev):
        self.dev = dev
        self.unet = torch.jit.load(_crop_find("unet_skm_cor_ts16.pt"), map_location=dev).eval()
        self.nets = []
        for f in range(5):
            s_ = torch.load(_crop_find(f"wCROPC_f{f}_fp16.pt"), map_location="cpu", weights_only=False)
            m_ = _CropNet(s_.get("arch", "convnext_tiny"), 4); m_.load_state_dict({k: v.float() for k, v in s_["model"].items()})
            self.nets.append(m_.to(dev).eval())
        self.nets_m = []
        if os.environ.get("AP_CROPCM") == "1":   # coronal crops, classifier with an MCL head (cols MedMen, MedOA, MCL | LatMen, LatOA)
            for f in range(5):
                s_ = torch.load(_crop_find(f"wCROPCM_f{f}_fp16.pt"), map_location="cpu", weights_only=False)
                m_ = _CropNet(s_.get("arch", "convnext_tiny"), 5); m_.load_state_dict({k: v.float() for k, v in s_["model"].items()})
                self.nets_m.append(m_.to(dev).eval())
    @torch.no_grad()
    def __call__(self, vol):
        v = (vol.clamp(0, 1) * 255).round().numpy().astype(np.float32) / 255.0
        t = torch.from_numpy(v)[:, None].to(self.dev)
        t = F.interpolate(t, size=(self.SIDE, self.SIDE), mode="bilinear", align_corners=False)
        p = (self.R - self.SIDE) // 2; t = F.pad(t, (p, self.R - self.SIDE - p, p, self.R - self.SIDE - p))
        pr = self.unet(t.repeat(1, 3, 1, 1).half()).float().softmax(1)[:, :, p:p + self.SIDE, p:p + self.SIDE]
        lab = F.interpolate(pr.argmax(1).float()[:, None], size=v.shape[1:], mode="nearest")[:, 0].long().cpu().numpy()
        X_ = np.zeros((2, self.NS, self.OUT, self.OUT), np.float32); CP = self.CP
        for c, m in enumerate((np.isin(lab, (3, 5)), np.isin(lab, (4, 6)))):
            a = m.sum((1, 2)); sl = np.sort(np.argsort(-a)[:self.NS])
            yy, xx = np.nonzero(m.any(0)) if m.any() else (np.array([v.shape[1] // 2]), np.array([v.shape[2] // 2]))
            cy0, cx0 = yy.mean(), xx.mean()
            for q, k in enumerate(sl):
                if a[k] > 0:
                    ys, xs = np.nonzero(m[k]); cy, cx = ys.mean(), xs.mean()
                else:
                    cy, cx = cy0, cx0
                y0, x0 = int(round(cy - CP / 2)), int(round(cx - CP / 2))
                img = np.pad(v[k], CP, mode="constant")[y0 + CP:y0 + 2 * CP, x0 + CP:x0 + 2 * CP]
                X_[c, q] = (F.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None], size=(self.OUT, self.OUT), mode="bilinear",
                                          align_corners=False)[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8) / 255.0
        self.last_X = X_   # v26: kept for the two-plane member
        x = torch.from_numpy(X_)[:, :, None].repeat(1, 1, 3, 1, 1).to(self.dev); o = 0
        for n_ in self.nets:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev.type == "cuda"):
                o = o + torch.sigmoid(n_(x).float())
        o = (o / len(self.nets)).cpu().numpy()
        r = [o[0, 0], o[0, 1], o[1, 2], o[1, 3]]
        if self.nets_m:   # MCL from the medial coronal crop (head index 2)
            om = 0
            for n_ in self.nets_m:
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev.type == "cuda"):
                    om = om + torch.sigmoid(n_(x[:1]).float())
            r.append(float((om / len(self.nets_m)).cpu().numpy()[0, 2]))
        return r


class AxCropMember:   # v31: axial patellofemoral crops (mirrors skm_crops_ax.py; cache axial orientation = identity on RSNA, variant 0)
    R, SIDE, CP, NS, OUT = 384, 312, int(round(80.0 / (130.0 / 384))), 8, 224
    def __init__(self, dev):
        self.dev = dev
        self.unet = torch.jit.load(_crop_find("unet_skm_ax_ts16.pt"), map_location=dev).eval()
        self.nets = []
        for f in range(5):
            s_ = torch.load(_crop_find(f"wCROPAX_f{f}_fp16.pt"), map_location="cpu", weights_only=False)
            m_ = _CropNet(s_.get("arch", "convnext_tiny"), 3); m_.load_state_dict({k: v.float() for k, v in s_["model"].items()}); self.nets.append(m_.to(dev).eval())
    @torch.no_grad()
    def __call__(self, vol):
        v = (vol.clamp(0, 1) * 255).round().numpy().astype(np.float32) / 255.0
        t = torch.from_numpy(v)[:, None].to(self.dev)
        t = F.interpolate(t, size=(self.SIDE, self.SIDE), mode="bilinear", align_corners=False)
        p = (self.R - self.SIDE) // 2; t = F.pad(t, (p, self.R - self.SIDE - p, p, self.R - self.SIDE - p))
        pr = self.unet(t.repeat(1, 3, 1, 1).half()).float().softmax(1)[:, :, p:p + self.SIDE, p:p + self.SIDE]
        lab = F.interpolate(pr.argmax(1).float()[:, None], size=v.shape[1:], mode="nearest")[:, 0].long().cpu().numpy()
        m = lab == 1; a = m.sum((1, 2)); sl = np.sort(np.argsort(-a)[:self.NS]); CP = self.CP
        yy, xx = np.nonzero(m.any(0)) if m.any() else (np.array([v.shape[1] // 2]), np.array([v.shape[2] // 2])); cy0, cx0 = yy.mean(), xx.mean()
        X_ = np.zeros((1, self.NS, self.OUT, self.OUT), np.float32)
        for q, k in enumerate(sl):
            if a[k] > 0:
                ys, xs = np.nonzero(m[k]); cy, cx = ys.mean(), xs.mean()
            else:
                cy, cx = cy0, cx0
            y0, x0 = int(round(cy - CP / 2)), int(round(cx - CP / 2))
            img = np.pad(v[k], CP, mode="constant")[y0 + CP:y0 + 2 * CP, x0 + CP:x0 + 2 * CP]
            X_[0, q] = (F.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None], size=(self.OUT, self.OUT), mode="bilinear",
                                      align_corners=False)[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8) / 255.0
        x = torch.from_numpy(X_)[:, :, None].repeat(1, 1, 3, 1, 1).to(self.dev); o = 0
        for n_ in self.nets:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev.type == "cuda"):
                o = o + torch.sigmoid(n_(x).float())
        o = (o / len(self.nets)).cpu().numpy()
        return [o[0, 0], o[0, 1], o[0, 2]]    # PF OA, Effusion, Synovitis

CROP_COLS = ["Medial Meniscus", "Medial OA", "Lateral Meniscus", "Lateral OA"]
CROP_ON = os.environ.get("AP_CROP", "1") == "1"
crop_rows = []
if CROP_ON:
    try:
        _crop = CropMember(DEV); print("CROP member loaded (U-Net + 5 folds)", flush=True)
    except Exception as e:
        CROP_ON = False; print(f"CROP member DISABLED: {type(e).__name__}: {e}", flush=True)
AXC_ON = CROP_ON and os.environ.get("AP_CROPAX") == "1"; axc_rows = []
if AXC_ON:
    try:
        _axc = AxCropMember(DEV); print("AXIAL CROP member loaded (ax U-Net + 5 folds)", flush=True)
    except Exception as e:
        AXC_ON = False; print(f"AXIAL CROP member DISABLED: {type(e).__name__}: {e}", flush=True)
CORC_ON = CROP_ON and os.environ.get("AP_CROPC") == "1"; corc_rows = []
TP_ON = CROP_ON and os.environ.get("AP_TWOPLANE") == "1"; tp_rows = []
if TP_ON:
    try:
        _tp_nets = []
        for f in range(5):
            s_ = torch.load(_crop_find(f"wCROPSC_f{f}_fp16.pt"), map_location="cpu", weights_only=False)
            m_ = _CropNet(s_.get("arch", "convnext_tiny"), 4); m_.load_state_dict({k: v.float() for k, v in s_["model"].items()}); _tp_nets.append(m_.to(DEV).eval())
        print("TWO-PLANE member loaded (5 folds)", flush=True)
    except Exception as e:
        TP_ON = False; print(f"TWO-PLANE member DISABLED: {type(e).__name__}: {e}", flush=True)
if CORC_ON:
    try:
        _corc = CorCropMember(DEV); print("CORONAL CROP member loaded (cor U-Net + 5 folds)", flush=True)
    except Exception as e:
        CORC_ON = False; print(f"CORONAL CROP member DISABLED: {type(e).__name__}: {e}", flush=True)

@torch.no_grad()
def _tp3_sagcut(self, vol):                                # vol [16, 384, 384] float in [0, 1], cache-equivalent slices
    v = (vol.clamp(0, 1) * 255).round().numpy().astype(np.float32) / 255.0   # uint8 cache quantisation
    t = torch.from_numpy(v)[:, None].to(self.dev)
    t = F.interpolate(t, size=(self.SIDE, self.SIDE), mode="bilinear", align_corners=False)
    p = (self.R - self.SIDE) // 2; t = F.pad(t, (p, self.R - self.SIDE - p, p, self.R - self.SIDE - p))
    lab = self.unet(t.repeat(1, 3, 1, 1).half()).argmax(1)
    lab = lab[:, p:p + self.SIDE, p:p + self.SIDE].float()[:, None]
    lab = F.interpolate(lab, size=v.shape[1:], mode="nearest")[:, 0].long().cpu().numpy()
    X_ = np.zeros((2, self.NS, self.OUT, self.OUT), np.float32)
    for c, m in enumerate((np.isin(lab, (3, 5)), np.isin(lab, (4, 6)))):
        a = m.sum((1, 2)); sl = np.sort(np.argsort(-a)[:self.NS])
        yy, xx = np.nonzero(m.any(0)) if m.any() else (np.array([v.shape[1] // 2]), np.array([v.shape[2] // 2]))
        cy0, cx0 = yy.mean(), xx.mean()
        for q, k in enumerate(sl):
            if a[k] > 0:
                ys, xs = np.nonzero(m[k]); cy, cx = ys.mean(), xs.mean()
            else:
                cy, cx = cy0, cx0
            y0, x0 = int(round(cy - self.CP / 2)), int(round(cx - self.CP / 2)); CP = self.CP
            img = np.pad(v[k], CP, mode="constant")[y0 + CP:y0 + 2 * CP, x0 + CP:x0 + 2 * CP]
            X_[c, q] = (F.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None], size=(self.OUT, self.OUT),
                                      mode="bilinear", align_corners=False)[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8) / 255.0
    return X_   # (TP3: crops only, no classifier)

@torch.no_grad()
def _tp3_corcut(self, vol):
    v = (vol.clamp(0, 1) * 255).round().numpy().astype(np.float32) / 255.0
    t = torch.from_numpy(v)[:, None].to(self.dev)
    t = F.interpolate(t, size=(self.SIDE, self.SIDE), mode="bilinear", align_corners=False)
    p = (self.R - self.SIDE) // 2; t = F.pad(t, (p, self.R - self.SIDE - p, p, self.R - self.SIDE - p))
    pr = self.unet(t.repeat(1, 3, 1, 1).half()).float().softmax(1)[:, :, p:p + self.SIDE, p:p + self.SIDE]
    lab = F.interpolate(pr.argmax(1).float()[:, None], size=v.shape[1:], mode="nearest")[:, 0].long().cpu().numpy()
    X_ = np.zeros((2, self.NS, self.OUT, self.OUT), np.float32); CP = self.CP
    for c, m in enumerate((np.isin(lab, (3, 5)), np.isin(lab, (4, 6)))):
        a = m.sum((1, 2)); sl = np.sort(np.argsort(-a)[:self.NS])
        yy, xx = np.nonzero(m.any(0)) if m.any() else (np.array([v.shape[1] // 2]), np.array([v.shape[2] // 2]))
        cy0, cx0 = yy.mean(), xx.mean()
        for q, k in enumerate(sl):
            if a[k] > 0:
                ys, xs = np.nonzero(m[k]); cy, cx = ys.mean(), xs.mean()
            else:
                cy, cx = cy0, cx0
            y0, x0 = int(round(cy - CP / 2)), int(round(cx - CP / 2))
            img = np.pad(v[k], CP, mode="constant")[y0 + CP:y0 + 2 * CP, x0 + CP:x0 + 2 * CP]
            X_[c, q] = (F.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None], size=(self.OUT, self.OUT), mode="bilinear",
                                      align_corners=False)[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8) / 255.0
    return X_   # (TP3: crops only, no classifier)

TP3_GEO = os.environ.get("AP_TP3GEO", "cache")   # "cache" = the wCROPSC3 TRAINING crops' geometry (tprep build_cache: --pos-order sagittal, centroid of the
# middle FILE of the (re-ordered) series, lat-norm); "kernel" = reuse this lane's two-plane crops as they are (pos16 slice set, centroid of slice pos16[8])
def _tp3_recut(si, sag, X0):   # TP3: this slot's 8+8 crops in the training cache's geometry; X0 (the lane's own crops) is reused when the inputs agree
    sd = slot_data[si]; fl = sd["files"]; n = len(fl)
    if sag and n > 1:   # build_cache --pos-order: sagittal series in ascending patient-x order (only the slice SET matters to the crop cut)
        x0, x1 = ipp_x(fl[0]), ipp_x(fl[-1])
        if x0 is not None and x1 is not None and x0 > x1: fl = fl[::-1]
    rv = sag and sd["lat"] == "R"   # sagittal lat-norm: right knees reversed (order only decides ties in the crop cut's top-8 slice pick)
    paths = [fl[p] for p in (sd["pos16"][::-1] if rv else sd["pos16"])]; rp = {sd["files"][p]: v for p, v in sd["raws"].items()}
    own = [sd["files"][p] for p in (sd["pos16"][::-1] if rv else sd["pos16"])]   # the lane's own slice list (CROP / CORC)
    def raw(q):
        if q not in rp: rp[q] = decode(q)
        return rp[q]
    off = centroid(*raw(fl[n // 2]))   # build_cache centroid_off(files[n // 2])
    if X0 is not None and off == sd["off"] and paths == own: return X0, 0
    vol = torch.stack([transform(*raw(q), 384, 130.0, off, False) for q in paths])
    if not sag and sd["lat"] == "R": vol = torch.flip(vol, dims=[-1])   # coronal lat-norm (as CORC)
    return (_tp3_sagcut(_crop, vol) if sag else _tp3_corcut(_corc, vol)), 1
_TP3_RECUT = [0, 0]
TP3_SAVE = os.environ.get("AP_TP3SAVE") == "1"   # TP3 (10-09): keep each study's two-plane input (8 sag + 8 cor crops) for the parent's three-plane member
if TP3_SAVE:
    try:
        _TP3X = np.lib.format.open_memmap("tp_crops.npy", mode="w+", dtype=np.uint8, shape=(len(test), 2, 16, 224, 224)); _TP3F = np.zeros((len(test), 2), np.int8)
        print(f"TP3: saving sag+cor crops for {len(test)} studies (CROP_ON {CROP_ON}, CORC_ON {CORC_ON})", flush=True)
    except Exception as e:
        TP3_SAVE = False; print(f"TP3 save DISABLED: {type(e).__name__}: {e}", flush=True)
per_model_probs = {k: [] for k in models}
with torch.no_grad():
    for k_i, uid in enumerate(test.StudyInstanceUID):
        sdir = os.path.join(COMP, "test_series", uid)
        slot_files = {}
        if os.path.isdir(sdir):
            for se in os.listdir(sdir):
                pl = plane_of.get(se)
                if pl is None:
                    continue
                for si, (spl, sfl) in enumerate(SLOTS):
                    # REVERTED. I replaced this with a TR-derived classification on the
                    # theory that build_cache used header physics - it does NOT. Line 326
                    # of build_cache reads ser.Fluid_Sensitive for the contrast_slots path,
                    # which is the cache we train on (_cs). series_class() is used only by
                    # --physics-slots, a variant we tested and rejected. The two paths
                    # already agreed; my "fix" created a mismatch and cost 0.928 -> 0.924.
                    if pl != spl or int(fluid_of.get(se, -1)) != sfl:
                        continue
                    fl = [os.path.join(sdir, se, x)
                          for x in os.listdir(os.path.join(sdir, se)) if x.endswith(".dcm")]
                    if fl and len(fl) > len(slot_files.get(si, ())):
                        slot_files[si] = fl

        # one decode pass per slot: 16 canonical band positions shared by all configs
        slot_data = {}
        for si, files in slot_files.items():
            keyed = sorted((hdr(f)[0], f) for f in files)
            files = [f for _, f in keyed]
            n = len(files)
            lo, hi = int(0.2 * (n - 1)), int(0.8 * (n - 1))
            pos16 = ([n // 2] * 16 if hi <= lo
                     else [int(round(x)) for x in np.linspace(lo, hi, 16)])
            _, lat = hdr(files[n // 2])
            pos16_po = pos16   # v12: build_cache --pos-order positions (ascending patient x), atlas sets only
            if ATLAS and SLOTS[si][0] == "Sagittal" and n > 1:
                x0, x1 = ipp_x(files[0]), ipp_x(files[-1])
                if x0 is not None and x1 is not None and x0 > x1:
                    pos16_po = [n - 1 - p for p in pos16]
            raws = {p: decode(files[p]) for p in sorted(set(pos16) | set(pos16_po))}
            mid_a, mid_px = raws[pos16[8]]
            slot_data[si] = dict(pos16=pos16, pos16_po=pos16_po, raws=raws, lat=lat,
                                 off=centroid(mid_a, mid_px))
            if TP3_SAVE: slot_data[si]["files"] = files   # TP3: InstanceNumber-sorted files of the slot (training-cache geometry recut)

        if TP3_SAVE and CORC_ON: _corc.last_X = None   # TP3: never reuse the previous study's coronal crops
        if CROP_ON:
            _crop.last_X = None
            if CORC_ON: _corc.last_X = None
        if CROP_ON:   # v13: first sagittal slot (fluid first) through the training-cache geometry: 384 px, 130 mm, centroid crop, lat-norm
            _cr = [np.nan] * (4 + 3 * (os.environ.get('AP_CROP2') == '1') + 3 * (os.environ.get('AP_CROP3') == '1') + 2 * (os.environ.get('AP_SKMEV') == '1'))
            try:
                _si = next((i_ for i_, (pl_, _) in enumerate(SLOTS) if pl_ == "Sagittal" and i_ in slot_data), None)
                if _si is not None:
                    _sd = slot_data[_si]; _ps = _sd["pos16"][::-1] if _sd["lat"] == "R" else _sd["pos16"]
                    _vol = torch.stack([transform(*_sd["raws"][p_], 384, 130.0, _sd["off"], False) for p_ in _ps])
                    _cr = _crop(_vol)
            except Exception as e:
                print(f"CROP study {uid[-8:]} failed: {type(e).__name__}: {e}", flush=True)
            crop_rows.append(_cr)
        if AXC_ON:   # v31: axial fluid slot (f1 first) through the cache geometry; lat-norm flips axial for right knees
            _ax = [np.nan] * 3
            try:
                _ai = next((i_ for i_, (pl_, _) in enumerate(SLOTS) if pl_ == "Axial" and i_ in slot_data), None)
                if _ai is not None:
                    _sd = slot_data[_ai]
                    _vol = torch.stack([transform(*_sd["raws"][p_], 384, 130.0, _sd["off"], False) for p_ in _sd["pos16"]])
                    if _sd["lat"] == "R":
                        _vol = torch.flip(_vol, dims=[-1])
                    _ax = _axc(_vol)
            except Exception as e:
                print(f"AXC study {uid[-8:]} failed: {type(e).__name__}: {e}", flush=True)
            axc_rows.append(_ax)
        if CORC_ON:   # v22: coronal fluid slot (f1 first) through the cache geometry; lat-norm flips coronal for right knees
            _cc = [np.nan] * (5 if os.environ.get('AP_CROPCM') == '1' else 4)
            try:
                _ci = next((i_ for i_, (pl_, _) in enumerate(SLOTS) if pl_ == "Coronal" and i_ in slot_data), None)
                if _ci is not None:
                    _sd = slot_data[_ci]
                    _vol = torch.stack([transform(*_sd["raws"][p_], 384, 130.0, _sd["off"], False) for p_ in _sd["pos16"]])
                    if _sd["lat"] == "R":
                        _vol = torch.flip(_vol, dims=[-1])
                    _cc = _corc(_vol)
            except Exception as e:
                print(f"CORC study {uid[-8:]} failed: {type(e).__name__}: {e}", flush=True)
            corc_rows.append(_cc)
        if TP_ON:   # v26: two-plane member on the crops both members just cut
            _tq = [np.nan] * 4
            try:
                _sx = getattr(_crop, "last_X", None); _cx = getattr(_corc, "last_X", None) if CORC_ON else None
                if _sx is not None or _cx is not None:
                    _z = np.zeros((2, 8, 224, 224), np.float32)
                    _x2 = np.concatenate([_sx if _sx is not None else _z, _cx if _cx is not None else _z], axis=1)
                    _xt = torch.from_numpy(_x2)[:, :, None].repeat(1, 1, 3, 1, 1).to(DEV); _o = 0
                    with torch.no_grad():
                        for n_ in _tp_nets:
                            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                                _o = _o + torch.sigmoid(n_(_xt).float())
                    _o = (_o / len(_tp_nets)).cpu().numpy(); _tq = [_o[0, 0], _o[0, 1], _o[1, 2], _o[1, 3]]
            except Exception as e:
                print(f"TP study {uid[-8:]} failed: {type(e).__name__}: {e}", flush=True)
            tp_rows.append(_tq)
        if TP3_SAVE:   # TP3: the two-plane input of this study as uint8 (crop values are k/255 -> lossless)
            try:
                _sx = _crop.last_X if CROP_ON else None; _cx = _corc.last_X if CORC_ON else None
                if TP3_GEO == "cache":
                    _si3 = next((i_ for i_, (pl_, _) in enumerate(SLOTS) if pl_ == "Sagittal" and i_ in slot_data), None)
                    _ci3 = next((i_ for i_, (pl_, _) in enumerate(SLOTS) if pl_ == "Coronal" and i_ in slot_data), None)
                    if CROP_ON and _si3 is not None: _sx, _r = _tp3_recut(_si3, True, _sx); _TP3_RECUT[0] += _r
                    if CORC_ON and _ci3 is not None: _cx, _r = _tp3_recut(_ci3, False, _cx); _TP3_RECUT[1] += _r
                if _sx is not None or _cx is not None:
                    _z = np.zeros((2, 8, 224, 224), np.float32)
                    _x2 = np.concatenate([_sx if _sx is not None else _z, _cx if _cx is not None else _z], axis=1)
                    _TP3X[k_i] = np.clip(np.round(_x2 * 255.0), 0, 255).astype(np.uint8); _TP3F[k_i] = (int(_sx is not None), int(_cx is not None))
            except Exception as e:
                print(f"TP3 save {uid[-8:]} failed: {type(e).__name__}: {e}", flush=True)
        for pref, kind in SETS:
            ck = metas[pref]
            meta = ck["meta"]
            IMG, CROP = meta["img"], meta.get("crop_mm", 160.0)
            LAT, CC = bool(meta.get("lat_norm")), bool(meta.get("centroid_crop"))
            USE = meta["slices"]   # native read: all 16 band slices
            AA = bool(meta.get("antialias", pref in AA_PREFIXES))
            # Every kind now reads the same way: grey slices at that set's own meta["img"]
            # (336 for the dinov2 sets, 384 for the CNNs), 3-channel expansion happens
            # inside the model. The old dino branch built 3-slice RGB groups for an
            # earlier checkpoint design and does not match anything we train now.
            imgs, msk = [], []
            for si in range(len(SLOTS)):
                sd = slot_data.get(si)
                if sd is None:
                    # one zero block per TTA band, so every slot contributes the same
                    # count and the (slots, bands, USE) reshape below stays aligned
                    imgs += [torch.zeros((IMG, IMG))] * (USE * len(SLICE_BANDS))
                    msk += [0.0] * USE
                    continue
                pos16 = sd["pos16_po"] if pref in ATLAS else sd["pos16"]   # v12: true patient-x order for atlas sets (then lat-norm below)
                raws = sd["raws"]
                off = sd["off"] if CC else (0.0, 0.0)
                flip = LAT and sd["lat"] == "R" and SLOTS[si][0] in ("Coronal", "Axial")
                rev = LAT and sd["lat"] == "R" and SLOTS[si][0] == "Sagittal"
                pseq = pos16[::-1] if rev else pos16

                def img_at(j):
                    a, px = raws[pseq[j]]
                    t = transform(a, px, IMG, CROP, off, AA)
                    return torch.flip(t, dims=[-1]) if flip else t

                for lo_b, hi_b in SLICE_BANDS:
                    for s in np.linspace(lo_b, hi_b, USE).round().astype(int):
                        imgs.append(img_at(s))
                msk += [1.0] * USE
            # slice-band TTA: the model samples a random USE-of-16 each epoch during
            # training but evaluates on one fixed draw, so it is NOT invariant to which
            # slices it gets - averaging over draws marginalises that choice. Rigid-view
            # TTA gains nothing here (+0.0003) because the training augmentation already
            # makes the model invariant to rotation and scale; this gains +0.0022.
            nb = len(SLICE_BANDS)
            stacked = torch.stack(imgs)
            per_slot = stacked.reshape(len(SLOTS), nb, USE, *stacked.shape[1:])
            mt = torch.tensor(msk)[None].to(DEV)
            lmb = None
            if pref in ATLAS:   # v12 atlas: tokens are slot-major, USE per slot; slice index s<8 = medial half after pos-order + lat-norm
                sidx = np.linspace(0, 15, USE).round().astype(int)
                lm = np.ones((len(SLOTS), USE, len(LABELS)), np.float32)
                for si_, (pl_, _) in enumerate(SLOTS):
                    if pl_ != "Sagittal":
                        continue
                    for lab_, side_ in ATLAS_LAB.items():
                        j_ = LABELS.index(lab_)
                        lm[si_, :, j_] = (sidx < 8) if side_ == "M" else (sidx >= 8)
                lmb = torch.tensor(lm.reshape(1, len(SLOTS) * USE, len(LABELS))).to(DEV)
            band_probs = {Fo: [] for Fo in n_folds(pref)}
            for b in range(nb):
                xb = per_slot[:, b].reshape(-1, *stacked.shape[1:])[None].to(DEV)
                for Fo in n_folds(pref):
                    with torch.autocast("cuda", enabled=DEV.type == "cuda"):
                        p = torch.sigmoid((models[(pref, Fo)](xb, mt, lmb) if lmb is not None else models[(pref, Fo)](xb, mt)).float())
                    band_probs[Fo].append(p[0].cpu().numpy())
            for Fo in n_folds(pref):
                per_model_probs[(pref, Fo)].append(np.mean(band_probs[Fo], axis=0))
        if (k_i + 1) % 50 == 0:
            print(f"  {k_i+1}/{len(test)}", flush=True)

from scipy.stats import rankdata
# Per-lineage blend weight. Equal weighting is the default because we cannot rank
# lineages locally (gold-58 has mispredicted the LB repeatedly), but LB-measured solo
# strength is trustworthy: ConvNeXt 0.931 as five models vs resnet 0.928. SET_W lets one
# submission test whether upweighting the strong lineage beats the flat average.
SET_W = {"wBCNX_f": 2.0, "wAA_f": 2.0}   # ConvNeXt 2x carried over; wBCNX inherits wCNX weight
ranked, wts = [], []
for k in models:
    if k[0] in _MEMBER:
        continue
    P = np.stack(per_model_probs[k])
    ranked.append(np.stack([rankdata(P[:, j]) for j in range(len(LABELS))], axis=1))
    wts.append(SET_W.get(k[0], 1.0))
w = np.asarray(wts, dtype=np.float64)
E = np.tensordot(w / w.sum(), np.stack(ranked), axes=(0, 0)) / (len(test) + 1)
OURS_E = E   # blended with the CoAtNet arm below
for _mp in sorted(_MEMBER):
    _ks = [k for k in models if k[0] == _mp]
    if not _ks:
        print(f"HYBRID_MEMBER {_mp}: no folds found - skipped", flush=True)
        continue
    _Em = np.mean([np.stack([rankdata(np.stack(per_model_probs[k])[:, j]) for j in range(len(LABELS))], axis=1)
                   for k in _ks], axis=0) / (len(test) + 1)
    _dm = pd.DataFrame(_Em, columns=LABELS)
    _dm.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values)
    _dm.to_csv(f"member_{_mp}.csv", index=False)
    print("HYBRID_MEMBER wrote", _mp, _dm.shape, "folds", [k[1] for k in _ks], flush=True)
if CROP_ON and len(crop_rows) == len(test):
    _C = np.array(crop_rows, dtype=np.float64)
    _dc = pd.DataFrame({c: pd.Series(_C[:, j]).rank(pct=True).fillna(0.5).values for j, c in enumerate(CROP_COLS)})
    _dc.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values)
    _dc.to_csv("member_wCROP.csv", index=False)
    _j = 4
    def _wm(tag, cols, rank=True):
        global _j
        d_ = pd.DataFrame({c: (pd.Series(_C[:, _j + q]).rank(pct=True).fillna(0.5) if rank else pd.Series(_C[:, _j + q])).values for q, c in enumerate(cols)})
        d_.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values); d_.to_csv(f"member_{tag}.csv", index=False)
        print(f"HYBRID_MEMBER wrote {tag}", d_.shape, "missing", int(np.isnan(_C[:, _j]).sum()), flush=True); _j += len(cols)
    if os.environ.get("AP_CROP2") == "1": _wm("wCROP2", ["PF OA", "Effusion", "ACL"])        # v15 crop v2 member
    if os.environ.get("AP_CROP3") == "1": _wm("wCROP3", ["MCL", "Baker's", "Contusion"])     # v16 crop v3 member
    if os.environ.get("AP_SKMEV") == "1": _wm("wSKMEV", ["Men_M", "Men_L"], rank=False)      # v16 referee evidence (raw)
if CORC_ON and len(corc_rows) == len(test):   # v22 coronal crop member
    _CC = np.array(corc_rows, dtype=np.float64)
    _dcc = pd.DataFrame({c: pd.Series(_CC[:, j]).rank(pct=True).fillna(0.5).values for j, c in enumerate(CROP_COLS)})
    _dcc.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values); _dcc.to_csv("member_wCROPC.csv", index=False)
    print("HYBRID_MEMBER wrote wCROPC", _dcc.shape, "missing", int(np.isnan(_CC[:, 0]).sum()), flush=True)
if AXC_ON and len(axc_rows) == len(test):   # v31 axial crop member
    _AX = np.array(axc_rows, dtype=np.float64)
    _dax = pd.DataFrame({c: pd.Series(_AX[:, j]).rank(pct=True).fillna(0.5).values for j, c in enumerate(["PF OA", "Effusion", "Synovitis"])})
    _dax.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values); _dax.to_csv("member_wCROPAX.csv", index=False)
    print("HYBRID_MEMBER wrote wCROPAX", _dax.shape, "missing", int(np.isnan(_AX[:, 0]).sum()), flush=True)
    if _CC.shape[1] == 5:   # coronal MCL member
        _dm5 = pd.DataFrame({"MCL": pd.Series(_CC[:, 4]).rank(pct=True).fillna(0.5).values}); _dm5.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values)
        _dm5.to_csv("member_wCROPCM.csv", index=False); print("HYBRID_MEMBER wrote wCROPCM", _dm5.shape, flush=True)
if TP_ON and len(tp_rows) == len(test):   # v26 two-plane member
    _TP = np.array(tp_rows, dtype=np.float64)
    _dtp = pd.DataFrame({c: pd.Series(_TP[:, j]).rank(pct=True).fillna(0.5).values for j, c in enumerate(CROP_COLS)})
    _dtp.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values); _dtp.to_csv("member_wCROPSC.csv", index=False)
    print("HYBRID_MEMBER wrote wCROPSC", _dtp.shape, "missing", int(np.isnan(_TP[:, 0]).sum()), flush=True)
    print("HYBRID_MEMBER wrote wCROP", _dc.shape, "missing", int(np.isnan(_C[:, 0]).sum()), flush=True)
if TP3_SAVE:
    _TP3X.flush(); del _TP3X
    pd.DataFrame({"StudyInstanceUID": test.StudyInstanceUID.astype(str).values, "sag_ok": _TP3F[:, 0], "cor_ok": _TP3F[:, 1]}).to_csv("tp_crops_meta.csv", index=False)
    print("TP3 crops saved: sag", int(_TP3F[:, 0].sum()), "cor", int(_TP3F[:, 1].sum()), "of", len(test), "| geometry", TP3_GEO, "recut sag", _TP3_RECUT[0], "cor", _TP3_RECUT[1], flush=True)
if os.environ.get("AP_CNN_ONLY") == "1":
    # HYBRID: write his CNN ensemble (rank units, [0,1]) and stop before the raptor section.
    _sub = pd.DataFrame(OURS_E, columns=LABELS)
    _sub.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values)
    _sub.to_csv("submission.csv", index=False)
    print("HYBRID_AP_CNN_ONLY wrote", _sub.shape, "lineages", [p for p, _ in SETS if p not in _MEMBER], flush=True)
    raise SystemExit(0)




# ============================================================================
# Fifth lineage: public CoAtNet (raptor, CC0). Verified solo on the LB at 0.924,
# matching its author's claim, so the integration is known-good. It is a genuinely
# different architecture AND a different preprocessing pipeline (140mm crop, 336px,
# 64-slice stack, 3-slice windows, per-label attention pooling) - which is exactly
# the kind of decorrelation an ensemble stuck at 0.935 needs. Kept as its own arm
# rather than merged into SETS because none of our cache assumptions apply to it.
# ============================================================================
# PER-LABEL CoAtNet weight. Flat 0.25 and flat 0.40 both scored 0.939 - a plateau, which
# says the flat knob is saturated. But CoAtNet is not uniformly better or worse: measured on
# the 58 gold studies it beats our OOF by +0.093 on MCL, +0.078 on Lateral Meniscus and
# +0.051 on Fracture, while LOSING by 0.027 on Synovitis and 0.016 on PF OA. A single scalar
# throws that structure away.
#
# Weights are the standardised per-label skill gap, scaled by ALPHA=0.30 about W0=0.32 and
# clipped to [0.15, 0.50]. Deliberately the SHAPE of the gap, not its magnitude: our OOF file
# is an older/smaller model set than the current 20-model blend, so the raw gaps overstate
# CoAtNet and must not set the level. The level stays anchored where the LB validated it.
# Mean weight is 0.320, inside the 0.25-0.40 band that scored 0.939.
# RE-DERIVED FOR A SINGLE ft ARM. gg1 (0.942) inherited weights calibrated for a
# 3-arm mixture; with ft as the only arm the right profile is ft-vs-ours alone.
# Biggest shift: Synovitis (ft beats us by .102) and MCL (.061) up; PF OA (.001) down.
RAPTOR_W_PER = [0.4843, 0.5010, 0.3337, 0.4360, 0.2667, 0.3695, 0.2225, 0.3551, 0.5500, 0.3007, 0.3797, 0.4620]
RAPTOR_W = 0.32   # unused when RAPTOR_W_PER is set; kept so the flat path still works

#!/usr/bin/env python3
"""Knee MRI: twelve findings from a single model

This notebook takes a knee MRI study and scores twelve findings at once: ACL tear, MCL tear,
medial and lateral meniscus tears, osteoarthritis in the medial, lateral and patellofemoral
compartments, joint effusion, synovitis, a Baker's cyst, bone contusion and fracture. It scores
0.924 on the public leaderboard using one model, with no ensembling and no test-time augmentation.

This is the inference half of the work. The model was trained separately and its weights are
attached as a dataset, so this notebook only loads them and predicts:
https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense

Where the training labels came from

Worth saying up front, because it shapes everything else. The competition gives you 4,407 studies
but structured labels for only 58 of them. Every other study arrives with a free-text radiology
report and nothing more, so there is very little to train against out of the box.

The labels behind these weights were made by reading those reports with a language model and
turning each into twelve probabilities rather than twelve yes or no answers. A report that says a
tear is suspected becomes a number near 0.8, not a 1, which is a fairer target than forcing every
hedged sentence into a hard label. That yields 4,349 studies to train on. The 58 studies that came
with real labels were never trained on and are used to check the result honestly; the model reaches
0.9167 macro-AUC on them.

Building a fixed input from studies that are all shaped differently

The hard part of this competition is not the network, it is that no two studies look alike. A
study holds several DICOM series shot in different planes, the number of series varies, and the
number of slices in a series varies more. Anything that expects a fixed-size input has to be given
one.

The approach here is to fill five fixed slots per study, always in the same order, for a stack of
64 images:

  18 slices from a sagittal series, preferring a fluid-sensitive one
  14 slices from a second sagittal series, preferring one that is not fluid-sensitive
  12 slices from a coronal series, preferring a fluid-sensitive one
   8 slices from a second coronal series
  12 slices from an axial series

Preferring a fluid-sensitive series for some slots and not for others is deliberate. Fluid-
sensitive sequences show swelling, effusion and acute injury clearly, while the other sequences
show anatomy and cartilage better, and the twelve findings are split across both. If a study has
no series for a slot, the slot is left as zeros and the model is told to skip it rather than being
fed something misleading.

Within a series, slices are taken evenly across 6 to 94 percent of the stack rather than from the
middle. The outer slices are where the collateral ligaments and the lateral meniscus sit, and
cutting them was measurably costing accuracy on exactly those findings.

Every slice is cropped to a 140 mm box around the centre of the image using the pixel spacing from
the DICOM header, then resized to 336 pixels. Cropping by millimetres rather than by pixel count
matters: it means a knee occupies the same fraction of the frame whether the scan was acquired at
0.3 or 0.5 mm per pixel, so the model is not asked to learn scale differences that carry no medical
information.

How the model reads the stack

Three neighbouring slices are stacked into the three channels of one image. The network then sees
a little of what lies above and below the slice in the middle, which is most of the benefit of a 3D
model at the cost of a 2D one. Each of these three-slice windows is passed through a CoAtNet
backbone at 384 pixels.

The windows are combined with an attention layer that has separate weights for each of the twelve
findings. This is the part that matters most. A cruciate tear may be visible on two sagittal slices
while osteoarthritis is spread across many coronal ones, and a single pooled score forces those two
to share one notion of which slices are important. Giving each finding its own attention weights
lets each one draw on the slices that actually show it.

Running it

Scoring uses 42 windows per study. Inference runs in half precision and automatically retries a
study in full precision if it fails, so no study is ever dropped from the submission. The notebook
needs no internet: the backbone is loaded from the attached weights rather than downloaded.
"""
import os, sys, glob, time, json, gc
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
# T4 (Turing) cuDNN v9 has fp16/fp32 conv engines but NOT bf16 for these shapes
# ("GET was unable to find an engine..."); benchmark lets it pick a valid algo for
# the fixed (1,24,3,res,res) input.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---- fixed config (must match training exactly) -----------------------------
IMG = 336
CROP_MM = 140.0
# 64 slices per study instead of 44, same proportions. Must match the corpus the weights
# were trained on (knee_corpus_v4.py).
R_SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(s[2] for s in R_SLOTS)                     # 64
K_EVAL = 62   # full coverage of the 64-slice stack (author: +0.003 vs 42)
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Three arms: (weights filename, fallback arch, fallback res). ck carries arch+res too.
# Selected 2026-08-19 by greedy forward selection AND exhaustive subset search over a 7-arm
# panel on the 45-study gold set (phase2/blend_panel.py); both agree on this exact set.
# Singles: coatnet384 0.9025 | swinbase384 0.8825 | effv2l480 0.8716.
# Blend {coatnet+swin+effv2l} = 0.9068 (2-arm {coatnet+swin} = 0.9059, coatnet alone 0.9025).
# Dropped as redundant: cnn336 (0.8833, the former champion), cnbase384 (0.8754),
# cnlarge384 (0.8752), maxvit384 (0.8438).
#
# SINGLE ARM: coatnet_rmlp_2_rw_384 retrained on the EXPANDED 4,349-study corpus.
#
# Why one arm and not the 3-arm blend: on the live leaderboard CoAtNet alone scored 0.914 while
# every blend scored 0.914-0.915, so ensembling is worth ~+0.001 there -- the ~+0.010 it showed
# on the old 45-study gold set was gold-set noise. One arm is also 1/3 the kernel runtime.
#
# Corpus expansion: the corpus previously held 3,200 of the 4,349 labelled studies and only 45
# of the 58 gold studies. Rebuilt to 4,407 studies (+37.8% training data, 58-study gate).
#
# Measured on the 58-study gate (the incumbent re-scored on the SAME gate for a fair compare):
#   incumbent CoAtNet (3,155-study corpus) 0.8923
#   this model       (4,349-study corpus) 0.9054   (+0.0131, better in 92.7% of 2000 bootstraps)
# Biggest gains land on the findings that were capping us: Lateral Meniscus +0.071,
# Fracture +0.057, Lateral OA +0.048, Medial Meniscus +0.035, ACL +0.028.
# ft-ENSEMBLE: SOUP6 (5 seeds + ft, gold58 .9284) REPLACES ft - 1 arm, same cost
# TIER-2 ARM: soup3 arm + 62 WINDOWS - stacks the two biggest pending levers
ARMS = [
    {"file": "raptor_v5soup.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},
]


# ============================================================================
# Model -- verbatim from finetune_raptor.py
# ============================================================================
def build_backbone(arch, pretrained=False):
    # maxvit/maxxvit/coatnet are conv-attention hybrids: NO CLS token, NO interpolatable
    # pos-embed -> avg pool. The "vit" substring in "coatnet"/"maxvit" must NOT route them
    # down the ViT path (mirrors finetune_raptor.py exactly).
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", arch_default)
    ck_res = int(ck.get("res", res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    # NOTE: DataParallel removed on purpose. On the full hidden test it drove a system-RAM OOM
    # (per-forward module replication over many studies); a single T4 handles K_EVAL=24 windows
    # fine. Arms are also run SEQUENTIALLY (see main) so peak RAM == one model, not two.
    del ck
    gc.collect()
    return model, ck_res


# ============================================================================
# Eval windowing -- verbatim from finetune_raptor.py StudyWindows (train=False)
# ============================================================================
def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x


@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != "cpu" and str(device).startswith("cuda")
    if use_cuda:
        # fp16 conv on T4 is fully cuDNN-supported (bf16 is NOT -> "no engine").
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
        except RuntimeError:
            # fp32 always has a Turing conv engine; slower but never drops a study.
            torch.cuda.empty_cache()
            o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
    o = torch.sigmoid(model(x).float())
    return o[0].cpu().numpy()


def rankpct(x):                                   # per-column percentile rank in [0,1]
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))


# ============================================================================
# Preprocessing -- verbatim from kprep2/dino_preprocess.py, retargeted to TEST
# ============================================================================
def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                    n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps); recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in R_SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        # wide span: the collateral ligaments and lateral meniscus live in the
        # peripheral slices the old 0.15-0.85 crop threw away. Must match the corpus
        # the weights were trained on (knee_corpus_v2.py, SPAN_LO/SPAN_HI).
        # WIDER SPAN: v5 was trained on the "Max-Span" corpus; 0.06-0.94 is the
        # 0.924-era geometry. The outer slices hold the far medial/lateral anatomy,
        # where our three weakest labels live.
        n = len(files); lo, hi = int(n * 0.02), int(n * 0.98) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []; pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


# ============================================================================
# Test-root discovery + weights + main
# ============================================================================
def find_test_root():
    cands = [f"{_KIN}/competitions/rsna-knee-abnormality-detection",
             f"{_KIN}/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk(f"{_KIN}"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk(f"{_KIN}"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")


def find_weight_file(fname):
    # direct dataset mounts first; NEVER recursive-glob the competitions DICOM tree.
    direct = [f"{_KIN}/raptor-knee-arms/{fname}",
              f"{_KIN}/raptor-knee-arms/1/{fname}",
              f"{_KIN}/raptor-cnn336/{fname}"]
    for p in direct:
        if os.path.exists(p):
            return p
    for d in sorted(glob.glob(f"{_KIN}/*/")):
        if "competition" in d.lower():
            continue
        hits = glob.glob(os.path.join(d, "**", fname), recursive=True)
        if hits:
            return hits[0]
    raise RuntimeError(f"{fname} not found under /kaggle/input")


def raptor_main():
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device {dev} | gpus {ngpu} | torch {torch.__version__}", flush=True)

    ROOT = find_test_root()
    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"
    print("test root:", ROOT, "| series dir:", tsdir, flush=True)

    test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(ROOT + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)

    sub_cols = ["StudyInstanceUID"] + LAB
    ssub = os.path.join(ROOT, "sample_submission.csv")
    if os.path.exists(ssub):
        sub_cols = list(pd.read_csv(ssub, nrows=1).columns)

    reader = _make_reader()
    N = len(test_ids); A = len(ARMS)
    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]

    # SEQUENTIAL ARMS (the OOM fix): only ONE model is resident at a time, so peak system RAM ==
    # one model == the single-arm champion's footprint (which graded fine at 0.879). Holding both
    # arms simultaneously OOM'd system RAM on the full hidden test. Each study is re-preprocessed
    # per arm (build_study is cheap vs inference) and every per-study buffer is freed. Same models,
    # same windowing, same rank-mean blend -> identical 0.8893 result, just serialized.
    for a, arm in enumerate(ARMS):
        wp = find_weight_file(arm["file"])
        model, res = load_model(wp, arm["arch"], arm["res"], dev)
        print(f"[arm {a}] loaded {arm['file']} | res {res} | {time.time()-t0:.0f}s", flush=True)
        for i, sid in enumerate(test_ids):
            try:
                vol, mask = build_study(sid, SER, tsdir, reader)
                xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)
                arm_probs[a][i] = infer_probs(model, xw, dev)
                del vol, mask, xw
            except Exception as e:
                print(f"  [arm {a}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
            if (i + 1) % 100 == 0 or i + 1 == N:
                print(f"  [arm {a}] {i+1}/{N} | {time.time()-t0:.0f}s", flush=True)
        del model
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[arm {a}] done + freed | {time.time()-t0:.0f}s", flush=True)

    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe).
    # Weights come from ARMS[*]["w"] and are normalised here, so dropping/adding an arm can
    # never silently change the scale. Falls back to equal weights if none are declared.
    _w = np.array([float(a.get("w", 1.0)) for a in ARMS], dtype=np.float64)
    _w = _w / _w.sum()
    print(f"[blend] weighted rank-mean w={dict(zip([a['file'] for a in ARMS], _w.round(4)))}", flush=True)
    ranks = np.tensordot(_w, np.stack([rankpct(np.clip(p, 0, 1)) for p in arm_probs]),
                         axes=(0, 0))                                          # (N,12) in [0,1]
    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5

    return ranks, test_ids


_rap_ranks, _rap_ids = raptor_main()
_pos = {s: i for i, s in enumerate(_rap_ids)}
_idx = [_pos[s] for s in test.StudyInstanceUID.values]      # align to our row order
R = _rap_ranks[_idx]
assert list(LAB) == list(LABELS), "label order mismatch between arms"

_w = np.asarray(RAPTOR_W_PER, dtype=np.float64)[None, :] if RAPTOR_W_PER else RAPTOR_W
E_FINAL = (1.0 - _w) * OURS_E + _w * R
sub = pd.DataFrame(E_FINAL, columns=LABELS)
sub.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values)
sub.to_csv("submission.csv", index=False)
print("per-label coatnet weights:", dict(zip(LABELS, RAPTOR_W_PER)), flush=True)
print("wrote submission.csv", sub.shape, "nulls", int(sub[LABELS].isna().sum().sum()), flush=True)
