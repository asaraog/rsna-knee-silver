#!/usr/bin/env python3
"""skm_train_ax.py - 2D AXIAL U-Net on SKM-TEA (reformatted from the sagittal DESS; LR flips so medial/lateral is learned from anatomy).
Based on skm_train.py (155 DESS knees, 6 labels) for per-compartment evidence on RSNA.

Labels: 1 patellar cart, 2 femoral cart, 3 MEDIAL tibial cart, 4 LATERAL tibial cart, 5 MEDIAL meniscus, 6 LATERAL meniscus.
The net must tell medial from lateral from one sagittal slice (fibula, meniscus shape): that per-slice side call is the point.

Geometry (verified on MTR_001): dicom-track mask array is [DICOM col, DICOM row, slice], axis 2 = slices in DESCENDING
LPS x (origin = first slice IPP). So training pair k = (pixel_array of k-th slice by descending x, mask[:, :, k].T).
Training orientation = DICOM native for SKM-TEA: rows run superior->inferior, cols run anterior->posterior.
RSNA inference must canonicalise to the same (see skm_infer.py).

Inputs: /workspace/skmtea/{dicom/MTR_*.tar.gz, masks/dicom-track__MTR_*.nii.gz}. Cache: /root/skc (uint8, 384 px, both echoes).
Split: 125 train / 30 val by sorted case id (every 5th case is val). Output: /workspace/skmtea/out/unet_skm.pt + val Dice.
"""
import glob, os, re, shutil, sys, tarfile, time
import numpy as np, nibabel as nib, pydicom, torch, torch.nn as nn, torch.nn.functional as F

S = "/workspace/skmtea"; C = "/root/skc_ax"; O = f"{S}/out_ax"; R = 384
os.makedirs(C, exist_ok=True); os.makedirs(O, exist_ok=True)


def prep(case):
    out = f"{C}/{case}.npz"
    if os.path.exists(out): return out
    tmp = f"/root/x_{case}"; shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(f"{S}/dicom/{case}.tar.gz") as t: t.extractall(tmp)
    ds = [pydicom.dcmread(f) for f in glob.glob(f"{tmp}/**/*.dcm", recursive=True)]
    ech = {e: sorted([d for d in ds if str(d.get("EchoNumbers")) == e and "T2 map" not in str(d.SeriesDescription)],
                     key=lambda d: -float(d.ImagePositionPatient[0])) for e in ("1", "2")}
    m = np.asarray(nib.load(f"{S}/masks/dicom-track__{case}.nii.gz").dataobj)
    n = m.shape[2]; assert len(ech["1"]) == n == len(ech["2"]), (case, len(ech["1"]), len(ech["2"]), m.shape)
    assert abs(float(ech["1"][0].ImagePositionPatient[0]) - float(ech["1"][1].ImagePositionPatient[0])) > 0.1
    vols = []
    for e in ("1", "2"):
        v = np.stack([d.pixel_array.astype(np.float32) for d in ech[e]])
        lo, hi = np.percentile(v, [1, 99.5]); v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
        v = F.interpolate(torch.from_numpy(v)[:, None], size=(R, R), mode="bilinear", align_corners=False)[:, 0]
        vols.append((v.numpy() * 255).astype(np.uint8))
    ym = m.transpose(2, 1, 0)   # [slice, row, col], aligned with pixel_array
    iop = np.array(ech["1"][0].ImageOrientationPatient, float)
    assert abs(iop[1]) > 0.7 and abs(iop[5]) > 0.7, (case, iop)
    if iop[1] < 0: vols = [v[:, :, ::-1] for v in vols]; ym = ym[:, :, ::-1]   # cols -> posterior
    if iop[5] > 0: vols = [v[:, ::-1, :] for v in vols]; ym = ym[:, ::-1, :]   # rows -> inferior
    vols = [np.ascontiguousarray(v) for v in vols]
    y = torch.from_numpy(np.ascontiguousarray(ym).astype(np.int64))
    y = F.interpolate(y[:, None].float(), size=(R, R), mode="nearest")[:, 0].numpy().astype(np.uint8)
    # coronal reformat: [LR slice, SI row, AP col] at 384 -> coronal image at AP index j = [SI, LR]; LR resampled to 0.4167 mm
    sp = float(getattr(ech["1"][0], "SpacingBetweenSlices", 0) or abs(float(ech["1"][0].ImagePositionPatient[0]) - float(ech["1"][1].ImagePositionPatient[0])))
    nlr = int(round(n * sp / (160.0 / R))); pl = (R - nlr) // 2
    keep = [j for j in range(R) if (y[:, j, :] >= 1).any()][::3]   # axial: rows of the sagittal image = SI
    keep += list(np.random.default_rng(0).choice([j for j in range(R) if j not in keep], size=min(8, R - len(keep)), replace=False))
    def cor(a, mode):
        t = torch.from_numpy(np.ascontiguousarray(a[:, keep, :].transpose(1, 2, 0))).float()[:, None]   # [n_keep, 1, AP, LR]
        t = F.interpolate(t, size=(R, nlr), mode=mode, **({} if mode == "nearest" else {"align_corners": False}))[:, 0]
        return F.pad(t, (pl, R - nlr - pl)) if nlr < R else t[:, :, (nlr - R) // 2:(nlr - R) // 2 + R]
    np.savez(out, e1=cor(vols[0], "bilinear").numpy().astype(np.uint8), e2=cor(vols[1], "bilinear").numpy().astype(np.uint8),
             y=cor(y, "nearest").numpy().astype(np.uint8))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def _prep(c):
    torch.set_num_threads(1)
    try: return c, prep(c)
    except Exception as e: print("SKIP", c, type(e).__name__, str(e)[:120], flush=True); return c, None


class DS(torch.utils.data.Dataset):
    def __init__(self, files, aug):
        self.d = [dict(np.load(f)) for f in files]
        self.idx = [(i, s) for i, d in enumerate(self.d) for s in range(d["y"].shape[0])]
        self.aug = aug
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i, s = self.idx[k]; d = self.d[i]
        e1, e2 = d["e1"][s].astype(np.float32) / 255, d["e2"][s].astype(np.float32) / 255
        if self.aug:   # contrast family: echo1 (PD/T1-ish), echo2 (fluid-bright, closest to RSNA FS), mixes; gamma; noise
            a = np.random.rand(); x = a * e1 + (1 - a) * e2 if np.random.rand() < 0.5 else (e2 if np.random.rand() < 0.6 else e1)
            x = np.clip(x, 0, 1) ** np.exp(np.random.uniform(-0.5, 0.5))
            x = x * np.random.uniform(0.7, 1.3) + np.random.uniform(-0.1, 0.1) + np.random.randn(*x.shape) * np.random.uniform(0, 0.04)
        else:
            x = e2
        x = torch.from_numpy(np.asarray(x, np.float32))[None]; y = torch.from_numpy(d["y"][s].astype(np.int64))
        if self.aug:   # scale / shift / small rotation (no flips: rows/cols are canonical)
            th = np.deg2rad(np.random.uniform(-12, 12)); sc = np.random.uniform(0.75, 1.3); tx, ty = np.random.uniform(-0.12, 0.12, 2)
            A = torch.tensor([[np.cos(th) / sc, -np.sin(th) / sc, tx], [np.sin(th) / sc, np.cos(th) / sc, ty]], dtype=torch.float32)[None]
            g = F.affine_grid(A, (1, 1, R, R), align_corners=False)
            x = F.grid_sample(x[None], g, align_corners=False)[0]
            y = F.grid_sample(y[None, None].float(), g, mode="nearest", align_corners=False)[0, 0].long()
            if np.random.rand() < 0.5:   # LR flip: medial/lateral must come from anatomy, not image side
                x, y = x.flip(-1), y.flip(-1)
            if np.random.rand() < 0.3:   # thick-slice look: blur
                x = F.avg_pool2d(x[None], 3, 1, 1)[0]
        return x.repeat(3, 1, 1), y


def main():
    import segmentation_models_pytorch as smp
    cases = sorted(re.findall(r"(MTR_\d+)", " ".join(glob.glob(f"{S}/masks/dicom-track__MTR_*.nii.gz"))))
    cases = [c for c in cases if os.path.exists(f"{S}/dicom/{c}.tar.gz")]
    t0 = time.time(); files = []
    from multiprocessing import Pool
    with Pool(int(os.environ.get("NPREP", 6))) as pool:
        for i, (c, f) in enumerate(pool.imap(_prep, cases)):
            if f: files.append((c, f))
            if i % 20 == 0: print(f"prep {i}/{len(cases)} [{time.time()-t0:.0f}s]", flush=True)
    va = [f for j, (c, f) in enumerate(files) if j % 5 == 0]; tr = [f for j, (c, f) in enumerate(files) if j % 5 != 0]
    print(f"cases {len(files)}: train {len(tr)} val {len(va)}", flush=True)
    tds, vds = DS(tr, True), DS(va, False)
    w = np.array([1.0 + 4.0 * (tds.d[i]["y"][s] >= 3).any() for i, s in tds.idx])
    tl = torch.utils.data.DataLoader(tds, 16, sampler=torch.utils.data.WeightedRandomSampler(w, 6000, True), num_workers=8, drop_last=True)
    vl = torch.utils.data.DataLoader(vds, 32, num_workers=8)
    net = smp.Unet("resnet34", encoder_weights="imagenet", classes=7).cuda()
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4); EP = int(os.environ.get("EP", 25))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * len(tl))
    dl = smp.losses.DiceLoss("multiclass", from_logits=True); ce = nn.CrossEntropyLoss(); scaler = torch.amp.GradScaler()
    best = -1
    for ep in range(EP):
        net.train(); t1 = time.time()
        for x, y in tl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                o = net(x); loss = ce(o, y) + dl(o, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        net.eval(); I = np.zeros(7); P = np.zeros(7); T = np.zeros(7)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for x, y in vl:
                p = net(x.cuda()).argmax(1); y = y.cuda()
                for c in range(1, 7):
                    I[c] += ((p == c) & (y == c)).sum().item(); P[c] += (p == c).sum().item(); T[c] += (y == c).sum().item()
        d = 2 * I / np.maximum(P + T, 1); m = float(d[1:].mean())
        # medial/lateral confusion: of true medial tibia+meniscus pixels, share predicted as the LATERAL twin (and vice versa)
        print(f"ep {ep} loss {loss.item():.3f} dice pat {d[1]:.3f} fem {d[2]:.3f} tibM {d[3]:.3f} tibL {d[4]:.3f} "
              f"menM {d[5]:.3f} menL {d[6]:.3f} mean {m:.3f} [{time.time()-t1:.0f}s]", flush=True)
        if m > best:
            best = m; torch.save({"model": net.state_dict(), "dice": d.tolist(), "epoch": ep, "R": R,
                                  "orient": "AXIAL rows ant->post, cols LR (either side; trained with LR flips), 0.4167 mm"}, f"{O}/unet_skm_ax.pt")
    print(f"TRAIN DONE best mean dice {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
