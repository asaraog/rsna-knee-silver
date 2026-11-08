#!/usr/bin/env python3
"""skm_crops_v2.py <unet_skm.pt|ts> <cache_base> <out_dir> [ids.txt] - crop v2: two NEW segmentation-guided crops per knee.

  crop 0 = PATELLOFEMORAL: the 8 sagittal slices with the most patellar cartilage (label 1), 64 mm squares centred on that
           slice's patellar-cartilage centroid, shifted 10 mm superior so the suprapatellar recess is in view (PF OA, Effusion).
  crop 1 = NOTCH: the 8 slices nearest the midpoint between the medial (3+5) and lateral (4+6) compartment centres, 64 mm
           squares centred 8 mm superior of the study's tibial-cartilage centroid (intercondylar notch, cruciates: ACL).
Same cache geometry / U-Net handling as skm_crops.py (cache in-plane orientation = native; rows run superior->inferior).
Output crops.npy uint8 [N, 2, 8, 224, 224] + meta.csv; incremental and resumable.
"""
import json, os, sys, numpy as np, pandas as pd, torch, torch.nn.functional as F

mp, base, out = sys.argv[1:4]; ids_f = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(out, exist_ok=True)
if mp.endswith("_ts16.pt"):
    net = torch.jit.load(mp, map_location="cuda").eval(); R = 384
else:
    import segmentation_models_pytorch as smp
    ck = torch.load(mp, map_location="cpu", weights_only=False); R = ck.get("R", 384)
    net = smp.Unet("resnet34", encoder_weights=None, classes=7); net.load_state_dict(ck["model"]); net = net.cuda().eval().half()
PX = 160.0 / R
meta = json.load(open(base + ".json")); st = meta["studies"]; slots = meta["slots"]
X = np.load(base + ".npy", mmap_mode="r"); Mk = np.load(base + ".mask.npy")
cpx = meta["crop_mm"] / meta["img"]; side = int(round(meta["img"] * cpx / PX)); pad = (R - side) // 2
sag = [j for j, s in enumerate(slots) if s.startswith("Sagittal")]
ids = [l.strip() for l in open(ids_f) if l.strip()] if ids_f else list(st)
pos = {s: i for i, s in enumerate(st)}; ids = [i for i in ids if i in pos]
NS, CP, OUTPX = 8, int(round(float(os.environ.get('PF_MM', '64')) / cpx)), 224; UP_PF, UP_N = int(round(float(os.environ.get('PF_UP_MM', '10')) / cpx)), int(round(8.0 / cpx))   # PF_MM=96 PF_UP_MM=35: suprapatellar pouch crop
C = np.lib.format.open_memmap(f"{out}/crops.npy", mode="r+" if os.path.exists(f"{out}/crops.npy") else "w+",
                              dtype=np.uint8, shape=(len(ids), 2, NS, OUTPX, OUTPX))
done = pd.read_csv(f"{out}/meta.csv") if os.path.exists(f"{out}/meta.csv") else pd.DataFrame()
rows = done.to_dict("records"); have = set(done.StudyInstanceUID) if len(done) else set()
print(f"{len(ids)} studies, crop {CP}px ({CP*cpx:.0f} mm) -> {OUTPX}; resume {len(have)}", flush=True)


def cut(v, k, cy, cx):
    y0, x0 = int(round(cy - CP / 2)), int(round(cx - CP / 2))
    img = np.pad(v[k], CP, mode="constant")[y0 + CP:y0 + 2 * CP, x0 + CP:x0 + 2 * CP]
    return (F.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None], size=(OUTPX, OUTPX), mode="bilinear",
                          align_corners=False)[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)


for n, sid in enumerate(ids):
    if sid in have: continue
    i = pos[sid]; j = next((j for j in sag if Mk[i, j]), None)
    if j is None: rows.append({"StudyInstanceUID": sid, "idx": n, "ok": 0}); continue
    v = np.asarray(X[i, j], dtype=np.float32) / 255.0
    t = torch.from_numpy(v)[:, None].cuda()
    t = F.interpolate(t, size=(side, side), mode="bilinear", align_corners=False); t = F.pad(t, (pad, R - side - pad, pad, R - side - pad))
    with torch.no_grad():
        lab = net(t.repeat(1, 3, 1, 1).half()).argmax(1)
    lab = lab[:, pad:pad + side, pad:pad + side].float()[:, None]
    lab = F.interpolate(lab, size=v.shape[1:], mode="nearest")[:, 0].long().cpu().numpy()
    S, H, W = v.shape; r = {"StudyInstanceUID": sid, "idx": n, "ok": 1}
    # PF crop
    P = lab == 1; aP = P.sum((1, 2)); slP = np.sort(np.argsort(-aP)[:NS])
    if P.any(): yy, xx = np.nonzero(P.any(0)); cy0, cx0 = yy.mean(), xx.mean()
    else: cy0, cx0 = H * 0.4, W * 0.35
    for q, k in enumerate(slP):
        if aP[k] > 0: ys, xs = np.nonzero(P[k]); cy, cx = ys.mean(), xs.mean()
        else: cy, cx = cy0, cx0
        C[n, 0, q] = cut(v, k, cy - UP_PF, cx)
    # notch crop
    aM = np.isin(lab, (3, 5)).sum((1, 2)).astype(float); aL = np.isin(lab, (4, 6)).sum((1, 2)).astype(float); kk = np.arange(S)
    mid = 0.5 * ((aM * kk).sum() / aM.sum() + (aL * kk).sum() / aL.sum()) if aM.sum() > 0 and aL.sum() > 0 else (S - 1) / 2
    slN = np.sort(np.argsort(np.abs(kk - mid), kind="stable")[:NS])
    T = np.isin(lab, (3, 4))
    if T.any(): yy, xx = np.nonzero(T.any(0)); ty, tx = yy.mean(), xx.mean()
    else: ty, tx = H * 0.6, W * 0.5
    for q, k in enumerate(slN):
        C[n, 1, q] = cut(v, k, ty - UP_N, tx)
    r.update({"areaP": int(aP.sum()), "slicesP": " ".join(map(str, slP)), "mid": round(float(mid), 2), "slicesN": " ".join(map(str, slN))})
    rows.append(r)
    if n % 200 == 0:
        C.flush(); pd.DataFrame(rows).to_csv(f"{out}/meta.csv", index=False); print(f"{n}/{len(ids)}", flush=True)
C.flush(); pd.DataFrame(rows).to_csv(f"{out}/meta.csv", index=False)
m = pd.DataFrame(rows); print("CROPS2 DONE", int(m.ok.sum()), "patellar area median", float(m.areaP.median()), "mid slice median", float(m.mid.median()))
