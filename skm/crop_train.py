#!/usr/bin/env python3
"""crop_train.py - compartment crop classifier on SKM-TEA-guided crops (skm_crops.py output).

Sample = (study, compartment). Input: 8 sagittal 224 px crops of that compartment. ConvNeXt-tiny (ImageNet) per slice ->
mean+max over slices -> 4 logits [Medial Meniscus, Medial OA, Lateral Meniscus, Lateral OA]; the loss uses only the two that
belong to the crop's compartment. Labels: pubv4_sharp.csv soft targets (BCE). Folds: same md5(Report) % 5 rule as
train_cached.py. Val metric `teach`: AUC vs pubv4 > 0.5 on the 4 compartment columns (macro), per label too.
Usage: crop_train.py --crops DIR --labels pubv4_sharp.csv --train-csv train.csv --fold 0 --out DIR
Writes best checkpoint + per-epoch log + val predictions (one row per study, 4 columns) incrementally.
"""
import argparse, hashlib, os, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--crops", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--train-csv", required=True)
ap.add_argument("--fold", type=int, default=0); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--out", required=True)
ap.add_argument("--arch", default="convnext_tiny.fb_in22k_ft_in1k")
ap.add_argument("--spec", default="Medial Meniscus,Medial OA|Lateral Meniscus,Lateral OA", help="labels read by crop 0 | crop 1")
ap.add_argument("--tag", default="CROP"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pu", default="", help="positive-unlabelled columns with class priors, e.g. 'Synovitis:0.45,Effusion:0.6': nnPU loss (Kiryo 2017) on those columns, report-negatives treated as unlabelled")
ap.add_argument("--init", default="", help="checkpoint to fine-tune from (its 'model' state dict), e.g. /workspace/tp_out/wCROPSC_f{fold}.pt")
ap.add_argument("--resid", action="store_true", help="channel 2 := crops_resid.npy (aligned crop minus atlas template, 128 offset)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
SPEC = [x.split(",") for x in a.spec.split("|")]; COLS = list(dict.fromkeys(c for g in SPEC for c in g))
PU = {COLS.index(k): float(v) for k, v in (x.split(":") for x in a.pu.split(",") if x)} if a.pu else {}   # column index -> class prior   # a column may be read by several crops
OWN = [[COLS.index(c) for c in g] for g in SPEC]   # head indices each crop is trained on
C = np.load(f"{a.crops}/crops.npy", mmap_mode="r"); meta = pd.read_csv(f"{a.crops}/meta.csv")
RS = np.load(f"{a.crops}/crops_resid.npy", mmap_mode="r") if a.resid else None
meta = meta[meta.ok == 1].drop_duplicates("StudyInstanceUID")
lab = pd.read_csv(a.labels).set_index("StudyInstanceUID"); tr = pd.read_csv(a.train_csv).set_index("StudyInstanceUID")
rep = tr["Report"].fillna("") if "Report" in tr.columns else pd.Series(dtype=str)
meta = meta[meta.StudyInstanceUID.isin(lab.index)]
fold = {s: int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % a.folds for s in meta.StudyInstanceUID}
va_m = meta[[fold[s] == a.fold for s in meta.StudyInstanceUID]]; tr_m = meta[[fold[s] != a.fold for s in meta.StudyInstanceUID]]
Y = lab.loc[meta.StudyInstanceUID, COLS].astype(np.float32)
YNAN = Y.isna(); Y = Y.fillna(0.0)   # NaN label = unknown: masked out of the loss (consensus-masked side labels)
print(f"fold {a.fold}: train {len(tr_m)} studies, val {len(va_m)}", flush=True)


class DS(torch.utils.data.Dataset):
    def __init__(self, m, aug):
        self.s = [(r.StudyInstanceUID, int(r.idx), c) for r in m.itertuples() for c in range(len(SPEC))]; self.aug = aug
    def __len__(self): return len(self.s)
    def __getitem__(self, k):
        sid, n, c = self.s[k]
        x = torch.from_numpy(np.asarray(C[n, c], dtype=np.float32) / 255.0)[:, None]      # [8, 1, 224, 224]
        if self.aug:
            th = np.deg2rad(np.random.uniform(-10, 10)); sc = np.random.uniform(0.85, 1.15); t = np.random.uniform(-0.08, 0.08, 2)
            A = torch.tensor([[np.cos(th) / sc, -np.sin(th) / sc, t[0]], [np.sin(th) / sc, np.cos(th) / sc, t[1]]], dtype=torch.float32)
            x = F.grid_sample(x, F.affine_grid(A[None].expand(len(x), 2, 3), x.shape, align_corners=False), align_corners=False)
            x = (x.clamp(0, 1) ** np.exp(np.random.uniform(-0.3, 0.3))) * np.random.uniform(0.85, 1.15)
        y = torch.tensor(Y.loc[sid].values); m = torch.zeros(len(COLS)); m[OWN[c]] = 1.0; m = m * (~torch.tensor(YNAN.loc[sid].values)).float()
        if RS is not None:   # --resid: [img, residual-to-template, img]
            rr = torch.from_numpy((np.asarray(RS[n, c], dtype=np.float32) - 128.0) / 127.5)[:, None]
            return torch.cat([x, rr, x], 1), y, m, c, sid
        return x.repeat(1, 3, 1, 1), y, m, c, sid


class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.b = timm.create_model(a.arch, pretrained=True, num_classes=0); d = self.b.num_features
        self.h = nn.Linear(2 * d, len(COLS))
    def forward(self, x):                                   # x [B, S, 3, H, W]
        B, S = x.shape[:2]; f = self.b(x.flatten(0, 1)).view(B, S, -1)
        return self.h(torch.cat([f.mean(1), f.amax(1)], 1))


tl = torch.utils.data.DataLoader(DS(tr_m, True), a.batch, shuffle=True, num_workers=8, drop_last=True)
vl = torch.utils.data.DataLoader(DS(va_m, False), a.batch * 2, num_workers=8)
net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=0.05)
if a.init:
    _ck = torch.load(a.init.format(fold=a.fold), map_location="cpu", weights_only=False); print("init from", a.init.format(fold=a.fold), "missing", net.load_state_dict({k: v.float() for k, v in _ck["model"].items()}, strict=False).missing_keys, flush=True)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl), pct_start=0.1); scaler = torch.amp.GradScaler()
best = -1
for ep in range(a.epochs):
    net.train(); t0 = time.time()
    for x, y, m, c, _ in tl:
        x, y, m = x.cuda(non_blocking=True), y.cuda(), m.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            o = net(x)
        bce = F.binary_cross_entropy_with_logits(o.float(), y, reduction="none")
        if PU:
            npu = torch.ones(len(COLS), device=m.device); npu[list(PU)] = 0
            loss = (bce * m * npu).sum() / torch.clamp((m * npu).sum(), min=1.0)
            for j, pi in PU.items():   # nnPU: pi*R_p+ + max(0, R_u- - pi*R_p-)
                g = o.float()[:, j]; sel = m[:, j] > 0; pos = sel & (y[:, j] > 0.5); unl = sel & (y[:, j] <= 0.5)
                rpp = F.softplus(-g[pos]).mean() if pos.any() else g.sum() * 0; rpn = F.softplus(g[pos]).mean() if pos.any() else g.sum() * 0
                run = F.softplus(g[unl]).mean() if unl.any() else g.sum() * 0
                loss = loss + pi * rpp + torch.clamp(run - pi * rpn, min=0.0)
        else:
            loss = (bce * m).sum() / m.sum()
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    net.eval(); P = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y, m, c, sid in vl:
            o = torch.sigmoid(net(x.cuda()).float()).cpu().numpy()
            for k in range(len(sid)):
                for q in OWN[int(c[k])]:
                    acc = P.setdefault(sid[k], [[0.0, 0] for _ in COLS]); acc[q][0] += o[k, q]; acc[q][1] += 1
    pv = pd.DataFrame.from_dict({k: [s_ / n if n else np.nan for s_, n in v] for k, v in P.items()}, orient="index", columns=COLS); yv = (Y.loc[pv.index] > 0.5)
    au = {c_: roc_auc_score(yv[c_], pv[c_]) for c_ in COLS if yv[c_].nunique() == 2}; mac = float(np.mean(list(au.values())))
    print(f"ep {ep+1}/{a.epochs} loss {loss.item():.4f} teach4 {mac:.4f} " + " ".join(f"{k.replace(chr(32), chr(95))} {v:.4f}" for k, v in au.items()) + f" [{time.time()-t0:.0f}s]", flush=True)
    if mac > best:
        best = mac; torch.save({"model": net.state_dict(), "arch": a.arch, "epoch": ep, "teach4": mac, "cols": COLS, "spec": a.spec}, f"{a.out}/w{a.tag}_f{a.fold}.pt")
        pv.rename_axis("StudyInstanceUID").to_csv(f"{a.out}/val_{a.tag}_f{a.fold}.csv")
print(f"CROP TRAIN DONE best teach4 {best:.4f}", flush=True)
