#!/usr/bin/env python3
"""skm_crops_ax.py <unet_skm_ax.pt> <cache_base> <out_dir> [ids.txt] - AXIAL patellofemoral crops (PF OA, effusion and synovitis are read on axial).

The coronal SKM-TEA U-Net (skm_train_cor.py, trained with LR flips) segments the cache's coronal fluid slot (Coronal|f1, else f0;
16 slices, 384 px over 130 mm). In-plane orientation of the cache is not assumed: identity / vertical flip / transpose /
transpose+flip are scored on 40 knees (mean max-softmax on foreground x log foreground area) and the best is used for all.
Per compartment (M = 3+5, L = 4+6): the 8 coronal slices with the most compartment area, 56 mm squares on the slice centroid,
224 px. Output crops.npy uint8 [N, 2, 8, 224, 224] (0 = medial, 1 = lateral) + meta.csv - same layout as skm_crops.py,
so crop_train.py's default spec (Medial Meniscus, Medial OA | Lateral Meniscus, Lateral OA) applies unchanged.
"""
import json, os, sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp

mp, base, out = sys.argv[1:4]; ids_f = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(out, exist_ok=True)
ck = torch.load(mp, map_location="cpu", weights_only=False); R = ck.get("R", 384); PX = 160.0 / R
net = smp.Unet("resnet34", encoder_weights=None, classes=7); net.load_state_dict(ck["model"]); net = net.cuda().eval().half()
meta = json.load(open(base + ".json")); st = meta["studies"]; slots = meta["slots"]
X = np.load(base + ".npy", mmap_mode="r"); Mk = np.load(base + ".mask.npy")
cpx = meta["crop_mm"] / meta["img"]; side = int(round(meta["img"] * cpx / PX)); pad = (R - side) // 2
cor = [j for j, s in enumerate(slots) if s.startswith("Axial")]
ids = [l.strip() for l in open(ids_f) if l.strip()] if ids_f else list(st)
pos = {s: i for i, s in enumerate(st)}; ids = [i for i in ids if i in pos]
NS, CP, OUTPX = 8, int(round(float(os.environ.get('CROP_MM', '80')) / cpx)), 224


def vol(i):
    j = next((j for j in cor if Mk[i, j]), None)
    return None if j is None else np.asarray(X[i, j], dtype=np.float32) / 255.0


def variant(v, k):
    if k & 2: v = v.transpose(0, 2, 1)
    if k & 1: v = v[:, ::-1, :]
    return np.ascontiguousarray(v)


def seg(v):
    t = torch.from_numpy(v)[:, None].cuda()
    t = F.interpolate(t, size=(side, side), mode="bilinear", align_corners=False); t = F.pad(t, (pad, R - side - pad, pad, R - side - pad))
    with torch.no_grad():
        pr = net(t.repeat(1, 3, 1, 1).half()).float().softmax(1)
    return pr[:, :, pad:pad + side, pad:pad + side]


probe = [i for i in ids if vol(pos[i]) is not None][:40]; sc = {}
for k in range(4):
    cs, fs = [], []
    for i in probe:
        pr = seg(variant(vol(pos[i]), k)); lab = pr.argmax(1); fg = (lab == 1) | (lab == 2)
        cs.append(float(pr.max(1).values[fg].mean()) if fg.any() else 0.0); fs.append(float(fg.sum()))
    sc[k] = np.mean(cs) * np.log1p(np.mean(fs)); print(f"variant {k}: conf {np.mean(cs):.3f} fg {np.mean(fs):.0f} score {sc[k]:.3f}", flush=True)
best = max(sc, key=sc.get); print("chosen variant", best, flush=True)

C = np.lib.format.open_memmap(f"{out}/crops.npy", mode="r+" if os.path.exists(f"{out}/crops.npy") else "w+",
                              dtype=np.uint8, shape=(len(ids), 1, NS, OUTPX, OUTPX))
done = pd.read_csv(f"{out}/meta.csv") if os.path.exists(f"{out}/meta.csv") else pd.DataFrame()
rows = done.to_dict("records"); have = set(done.StudyInstanceUID) if len(done) else set()
for n, sid in enumerate(ids):
    if sid in have: continue
    v = vol(pos[sid])
    if v is None: rows.append({"StudyInstanceUID": sid, "idx": n, "ok": 0}); continue
    v = variant(v, best); pr = seg(v)
    lab = F.interpolate(pr.argmax(1).float()[:, None], size=v.shape[1:], mode="nearest")[:, 0].long().cpu().numpy()
    r = {"StudyInstanceUID": sid, "idx": n, "ok": 1}
    m = lab == 1; a = m.sum((1, 2)); sl = np.sort(np.argsort(-a)[:NS])   # 8 axial slices with the most patellar cartilage
    if os.environ.get("AX_SUP") == "1" and a.sum() > 0:   # suprapatellar pouch: the window from 2 patellar slices up to 6 slices SUPERIOR of the patella
        kk = np.arange(len(a)); tib = np.isin(lab, (3, 4)).sum((1, 2)).astype(float)
        d = -1 if (tib.sum() == 0 or (tib * kk).sum() / tib.sum() > (a * kk).sum() / a.sum()) else 1   # superior = away from the tibial cartilage
        top = int(kk[a > 0].min()) if d == -1 else int(kk[a > 0].max())
        sl = np.array([int(np.clip(top + d * (i - 1), 0, len(a) - 1)) for i in range(NS)])
    yy, xx = np.nonzero(m.any(0)) if m.any() else (np.array([v.shape[1] // 2]), np.array([v.shape[2] // 2])); cy0, cx0 = yy.mean(), xx.mean()
    for q, k in enumerate(sl):
        if a[k] > 0: ys, xs = np.nonzero(m[k]); cy, cx = ys.mean(), xs.mean()
        else: cy, cx = cy0, cx0
        y0, x0 = int(round(cy - CP / 2)), int(round(cx - CP / 2))
        img = np.pad(v[k], CP, mode="constant")[y0 + CP:y0 + 2 * CP, x0 + CP:x0 + 2 * CP]
        C[n, 0, q] = (F.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None], size=(OUTPX, OUTPX), mode="bilinear", align_corners=False)[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    r["areaP"] = int(a.sum())
    rows.append(r)
    if n % 200 == 0:
        C.flush(); pd.DataFrame(rows).to_csv(f"{out}/meta.csv", index=False); print(f"{n}/{len(ids)}", flush=True)
C.flush(); pd.DataFrame(rows).to_csv(f"{out}/meta.csv", index=False)
m = pd.DataFrame(rows); print("CROPSAX DONE", int(m.ok.sum()), "median patellar px", float(m.areaP.median()))
