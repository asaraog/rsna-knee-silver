#!/usr/bin/env python3
"""skm_crops.py <unet_skm.pt> <cache_base> <out_dir> [ids.txt] - segmentation-guided compartment crops from an AP slice cache.

For every study: the sagittal fluid slot (16 slices, 384 px over 130 mm, lat-normed, pos-ordered caches put medial at index 0)
goes through the SKM-TEA U-Net (cache in-plane orientation = native, verified on OAI: rho -0.86, the other 7 variants <= 0.39).
Per compartment (M = labels 3+5 tibial cartilage + meniscus, L = 4+6): the 8 slices with the largest compartment area
(kept in slice order), each cropped as a 56 mm square centred on that slice's compartment centroid (study mean centroid when
the slice has none), resized to 224 px. Output (memmap-friendly):
  crops.npy  uint8 [N, 2, 8, 224, 224]   (comp 0 = medial, 1 = lateral)
  meta.csv   StudyInstanceUID, idx, rho (side-split check), areaM, areaL, slicesM, slicesL
Writes incrementally (memmap + meta every 200 studies), resumable.
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
sag = [j for j, s in enumerate(slots) if s.startswith("Sagittal")]
ids = [l.strip() for l in open(ids_f) if l.strip()] if ids_f else list(st)
pos = {s: i for i, s in enumerate(st)}; ids = [i for i in ids if i in pos]
NS, CP, OUTPX = int(os.environ.get('NS', '8')), int(round(float(os.environ.get('CROP_MM', '56')) / cpx)), int(os.environ.get('OUT_PX', '224'))   # CROP_MM=40: zoomed meniscus crops
C = np.lib.format.open_memmap(f"{out}/crops.npy", mode="r+" if os.path.exists(f"{out}/crops.npy") else "w+",
                              dtype=np.uint8, shape=(len(ids), 2, NS, OUTPX, OUTPX))
done = pd.read_csv(f"{out}/meta.csv") if os.path.exists(f"{out}/meta.csv") else pd.DataFrame()
rows = done.to_dict("records"); have = set(done.StudyInstanceUID) if len(done) else set()
print(f"{len(ids)} studies, crop {CP}px of cache ({CP*cpx:.0f} mm) -> {OUTPX}; resume {len(have)}", flush=True)
for n, sid in enumerate(ids):
    if sid in have: continue
    i = pos[sid]; _js = [j for j in sag if Mk[i, j]]
    j = (_js[1] if len(_js) > 1 else None) if os.environ.get("SAG_SLOT") == "second" else (_js[0] if _js else None)   # SAG_SLOT=second: the non-fluid sagittal series
    if j is None: rows.append({"StudyInstanceUID": sid, "idx": n, "ok": 0}); continue
    v = np.asarray(X[i, j], dtype=np.float32) / 255.0                       # [16, 384, 384]
    t = torch.from_numpy(v)[:, None].cuda()
    t = F.interpolate(t, size=(side, side), mode="bilinear", align_corners=False); t = F.pad(t, (pad, R - side - pad, pad, R - side - pad))
    with torch.no_grad():
        lab = net(t.repeat(1, 3, 1, 1).half()).argmax(1)                   # model space
    lab = lab[:, pad:pad + side, pad:pad + side].float()[:, None]
    lab = F.interpolate(lab, size=v.shape[1:], mode="nearest")[:, 0].long().cpu().numpy()   # back to cache pixels
    if os.environ.get("POSE") == "1":   # pose normalisation (pose_norm.py): plateau horizontal, fixed plateau scale
        if "_POSE_STD" not in globals():
            import pose_norm as _pn; _sds = [p_[1] for p_ in (_pn.plateau(lab),) if p_ is not None]
            globals()["_pn"] = _pn; globals()["_POSE_STD"] = float(os.environ.get("POSE_STD", "0")) or None; globals()["_sd_probe"] = []
        if _POSE_STD is None:
            _p = _pn.plateau(lab)
            if _p is not None: _sd_probe.append(_p[1])
            if len(_sd_probe) >= 60: globals()["_POSE_STD"] = float(np.median(_sd_probe)); print(f"POSE_STD probe median {_POSE_STD:.1f} px", flush=True)
        if _POSE_STD is not None:
            v, lab, _pi = _pn.normalise(v, lab, _POSE_STD)
    r = {"StudyInstanceUID": sid, "idx": n, "ok": 1}
    _mc = ((5,), (6,)) if os.environ.get('MEN_ONLY') == '1' else ((3, 5), (4, 6))   # MEN_ONLY=1: centre on the meniscus labels only
    M = np.isin(lab, _mc[0]); L = np.isin(lab, _mc[1]); aM, aL = M.sum((1, 2)), L.sum((1, 2))
    s = (aM - aL) / np.maximum(aM + aL, 1); ok = (aM + aL) > 30
    r["rho"] = pd.Series(s[ok]).corr(pd.Series(np.arange(len(s))[ok]), method="spearman") if ok.sum() > 3 else np.nan
    for c, (m, a) in enumerate(((M, aM), (L, aL))):
        sl = np.sort(np.argsort(-a)[:NS]); yy, xx = np.nonzero(m.any(0)) if m.any() else (np.array([v.shape[1] // 2]), np.array([v.shape[2] // 2]))
        cy0, cx0 = yy.mean(), xx.mean()
        for q, k in enumerate(sl):
            if a[k] > 0: ys, xs = np.nonzero(m[k]); cy, cx = ys.mean(), xs.mean()
            else: cy, cx = cy0, cx0
            y0, x0 = int(round(cy - CP / 2)), int(round(cx - CP / 2))
            img = np.pad(v[k], CP, mode="constant")[y0 + CP:y0 + 2 * CP, x0 + CP:x0 + 2 * CP]
            C[n, c, q] = (F.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None], size=(OUTPX, OUTPX), mode="bilinear",
                                        align_corners=False)[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
        r[f"area{'ML'[c]}"] = int(a.sum()); r[f"slices{'ML'[c]}"] = " ".join(map(str, sl))
    rows.append(r)
    if n % 200 == 0:
        C.flush(); pd.DataFrame(rows).to_csv(f"{out}/meta.csv", index=False); print(f"{n}/{len(ids)}", flush=True)
C.flush(); pd.DataFrame(rows).to_csv(f"{out}/meta.csv", index=False)
m = pd.DataFrame(rows); print("CROPS DONE", int(m.ok.sum()), "median rho", round(float(m.rho.median()), 3), "share rho<-0.5", round(float((m.rho < -0.5).mean()), 3))
