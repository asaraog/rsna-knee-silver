#!/usr/bin/env python3
"""skm_crops_v3.py <unet_skm.pt|ts> <cache_base> <out_dir> [ids.txt] - crop v3: two more segmentation-guided crops per knee.

  crop 0 = MEDIAL EDGE: the 8 slices at or beyond the medial compartment centre, moving away from the lateral side
           (collateral ligament, semimembranosus / popliteal region), 80 mm squares centred 10 mm posterior of that slice's
           medial tibial-cartilage centroid (study centroid when absent): MCL, Baker's.
  crop 1 = BONE: the 4 slices nearest the medial centre + the 4 nearest the lateral centre, 96 mm squares on the tibial
           cartilage centroid so both femoral condyle and tibial plateau marrow are in view: Contusion.
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
NS, OUTPX = 8, 224; CPE, CPB = int(round(80.0 / cpx)), int(round(96.0 / cpx)); BACK = int(round(10.0 / cpx))
C = np.lib.format.open_memmap(f"{out}/crops.npy", mode="r+" if os.path.exists(f"{out}/crops.npy") else "w+",
                              dtype=np.uint8, shape=(len(ids), 2, NS, OUTPX, OUTPX))
done = pd.read_csv(f"{out}/meta.csv") if os.path.exists(f"{out}/meta.csv") else pd.DataFrame()
rows = done.to_dict("records"); have = set(done.StudyInstanceUID) if len(done) else set()
print(f"{len(ids)} studies, edge {CPE}px bone {CPB}px -> {OUTPX}; resume {len(have)}", flush=True)


def cut(v, k, cy, cx, CP):
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
    aM = np.isin(lab, (3, 5)).sum((1, 2)).astype(float); aL = np.isin(lab, (4, 6)).sum((1, 2)).astype(float); kk = np.arange(S)
    if aM.sum() > 0 and aL.sum() > 0:
        sM = (aM * kk).sum() / aM.sum(); sL = (aL * kk).sum() / aL.sum()
    else:
        sM, sL = 3.0, 12.0
    sg = 1.0 if sM >= sL else -1.0
    d = (kk - sM) * sg                                  # >= 0 : at or beyond the medial centre, away from lateral
    cand = [k for k in np.argsort(np.where(d >= -0.5, d, 1e9), kind="stable") if d[k] >= -0.5][:NS]
    while len(cand) < NS: cand.append(cand[-1] if cand else int(round(sM)))
    slE = np.sort(np.array(cand))
    TM = lab == 3; T = np.isin(lab, (3, 4))
    if T.any(): yy, xx = np.nonzero(T.any(0)); ty, tx = yy.mean(), xx.mean()
    else: ty, tx = H * 0.6, W * 0.5
    for q, k in enumerate(slE):
        if TM[k].any(): ys, xs = np.nonzero(TM[k]); cy, cx = ys.mean(), xs.mean()
        else: cy, cx = ty, tx
        C[n, 0, q] = cut(v, k, cy, cx + BACK, CPE)        # cols run anterior -> posterior: + = posterior
    slB = np.sort(np.concatenate([np.argsort(np.abs(kk - sM), kind="stable")[:4], np.argsort(np.abs(kk - sL), kind="stable")[:4]]))
    for q, k in enumerate(slB):
        if T[k].any(): ys, xs = np.nonzero(T[k]); cy, cx = ys.mean(), xs.mean()
        else: cy, cx = ty, tx
        C[n, 1, q] = cut(v, k, cy, cx, CPB)
    r.update({"sM": round(float(sM), 2), "sL": round(float(sL), 2), "slicesE": " ".join(map(str, slE)), "slicesB": " ".join(map(str, slB))})
    rows.append(r)
    if n % 200 == 0:
        C.flush(); pd.DataFrame(rows).to_csv(f"{out}/meta.csv", index=False); print(f"{n}/{len(ids)}", flush=True)
C.flush(); pd.DataFrame(rows).to_csv(f"{out}/meta.csv", index=False)
m = pd.DataFrame(rows); print("CROPS3 DONE", int(m.ok.sum()), "medial centre median", float(m.sM.median()), "lateral centre median", float(m.sL.median()))
