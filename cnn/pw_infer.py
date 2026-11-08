"""pw_infer.py - OOF + gold-58 inference for the PW16 (_po, --posw) lineage, on the training cache.

Loads one or more train_cached.py checkpoints, captures the PER-TOKEN logits (forward hook on the
linear head, which the --posw path applies per slice), and pools them three ways:
  posw : the trainer's own pooling (per-label softmax position weights over the sagittal slots) = the logged `teach`
  mean : masked mean of per-token logits == head(mean f) (linear head) = how the kernel's CNXNet pools with K_PER off
  kper : the hybrid AP lane's rule (AP_SRC CNXNet.forward): masked tokens -> -1e4, per-label top-k over tokens, mean
Scores every pooling against pubv4_sharp.csv (the recipe's --eval-labels), labels_v4fb_sharp.csv (the training labels)
and train.csv gold (expert, 58 studies), per column. Writes per-study probabilities to --out.

  --fold k     : the fold's validation rows (same report-hash split as train_cached.py) -> OOF; gold rows in that fold too
  --fold -1    : the 58 gold studies only (5-fold mean over --weights; NB with --no-gold the gold studies sit in the
                 TRAINING folds of 4 of the 5 checkpoints, so this is the kernel's in-sample situation, not OOF)
"""
import argparse, hashlib, io, json, os, sys, zipfile
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_cached as tc

K_PER = [6, 6, 6, 6, 16, 16, 16, 12, 12, 3, 3, 3]   # AP_SRC K_PER verbatim
LABELS = tc.LABELS


def pool_all(L, m, pos_table, n_slots):
    """L [b,n,12] per-token logits, m [b,n] mask, pos_table [12, per] -> dict of pooled logits."""
    b, n, _ = L.shape
    w = m[..., None]
    out = {"mean": (L * w).sum(1) / w.sum(1).clamp(min=1)}
    per = max(n // n_slots, 1)
    wcol = torch.ones(n, len(LABELS), device=L.device)
    pt = pos_table.softmax(1) * per
    for si in (0, 1):
        wcol[si * per:(si + 1) * per] = pt.T
    lw = w * wcol[None]
    out["posw"] = (L * lw).sum(1) / lw.sum(1).clamp(min=1)
    Lm = L.masked_fill(m[:, :, None] < 0.5, -1e4)
    kp = out["mean"].clone()
    for kk in sorted({v for v in K_PER if v > 0}):
        cols = [i for i, v in enumerate(K_PER) if v == kk]
        tp = Lm.topk(min(kk, n), dim=1).values.mean(1)
        kp[:, cols] = tp[:, cols]
    out["kper"] = kp
    return out


def aucs(T, P):
    """per-column AUC (nan where one class); T binarised at 0.5."""
    T = (np.asarray(T) > 0.5).astype(int)
    return [roc_auc_score(T[:, j], P[:, j]) if len(np.unique(T[:, j])) > 1 else float("nan")
            for j in range(len(LABELS))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--use", type=int, default=16)
    p.add_argument("--tag", default="wPWV4")
    p.add_argument("--out", default="/root/out")
    p.add_argument("--labels", default="labels_v4fb_sharp.csv")
    p.add_argument("--eval-labels", dest="eval_labels", default="pubv4_sharp.csv")
    a = p.parse_args()
    WORK = os.environ.get("WORK", os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(tc.CACHE_DIR, a.cache)
    meta = json.load(open(f"{base}.json"))
    arr = np.load(f"{base}.npy", mmap_mode="r")
    mask = np.load(f"{base}.mask.npy")
    studies = meta["studies"]
    n_slots = int(mask.shape[1])
    z = zipfile.ZipFile(tc.ZIP)
    train = pd.read_csv(io.BytesIO(z.read("train.csv")))
    gold = train[train[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS]
    rep = train.set_index("StudyInstanceUID")["Report"].fillna("")
    fold_of = {s: int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % a.folds for s in studies}
    lab = pd.read_csv(os.path.join(WORK, a.labels)).set_index("StudyInstanceUID")
    ev = pd.read_csv(os.path.join(WORK, a.eval_labels)).set_index("StudyInstanceUID")
    keep = [i for i, s in enumerate(studies) if mask[i].sum() > 0]
    if a.fold >= 0:
        rows = [i for i in keep if fold_of[studies[i]] == a.fold and studies[i] in lab.index]
        print(f"fold {a.fold}: {len(rows)} validation rows, gold in fold: "
              f"{sum(studies[i] in gold.index for i in rows)}", flush=True)
    else:
        rows = [i for i in keep if studies[i] in gold.index]
        print(f"gold-58 run: {len(rows)} gold studies cached, {len(a.weights)} checkpoints", flush=True)
    rows = np.array(rows)
    dev = torch.device("cuda")
    acc = {k: np.zeros((len(rows), len(LABELS)), np.float64) for k in ("posw", "mean", "kper")}
    for wpath in a.weights:
        ck = torch.load(wpath, map_location="cpu", weights_only=False)
        model = tc.Net(ck.get("arch", "convnext_tiny"), n_slots=n_slots)
        model.posw = True
        per = max(n_slots * a.use // n_slots, 1)
        model.pos_table = nn.Parameter(torch.zeros(len(LABELS), ck["model"]["pos_table"].shape[1]))
        miss, unexp = model.load_state_dict(ck["model"], strict=False)
        assert not unexp, unexp[:5]
        miss = [k for k in miss if not k.startswith(("rclip_proj", "lm_head"))]
        assert not miss, miss[:5]
        model = model.eval().to(dev)
        cap = {}
        h = model.head.register_forward_hook(lambda mod, inp, out: cap.__setitem__("L", out) if out.dim() == 3 else None)
        dl = DataLoader(tc.Cached(arr, mask, rows, np.zeros((len(studies), len(LABELS)), np.float32)[rows],
                                  np.ones((len(studies), len(LABELS)), np.float32)[rows], a.use, train=False),
                        batch_size=1, num_workers=2)
        P = {k: [] for k in acc}
        with torch.no_grad():
            for x, m, *_ in dl:
                x, m = x.to(dev), m.to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    o = model(x, m).float()
                L = cap["L"].float()
                pooled = pool_all(L, m.float(), model.pos_table.float(), n_slots)
                assert torch.allclose(pooled["posw"], o, atol=2e-2), (pooled["posw"], o)   # hook path == model path
                for k in acc:
                    P[k].append(torch.sigmoid(pooled[k]).cpu().numpy())
        h.remove()
        for k in acc:
            acc[k] += np.concatenate(P[k]) / len(a.weights)
        print(f"  done {os.path.basename(wpath)} (best_epoch {ck.get('best_epoch')}, saved teach {ck.get('teacher_auc'):.4f}, "
              f"truth {ck.get('truth_auc')})", flush=True)
        del model; torch.cuda.empty_cache()
    sid = [studies[i] for i in rows]
    name = f"{a.tag}_f{a.fold}" if a.fold >= 0 else f"{a.tag}_gold58"
    for k in acc:
        pd.DataFrame(acc[k], columns=LABELS).assign(StudyInstanceUID=sid)[["StudyInstanceUID"] + LABELS].to_csv(
            os.path.join(a.out, f"{name}_{k}.csv"), index=False)
    # ---- scoring
    def table(title, T, idx):
        print(f"\n== {title} (n={len(idx)})", flush=True)
        print("pool  " + " ".join(f"{l[:7]:>7s}" for l in LABELS) + "    mean", flush=True)
        for k in ("posw", "mean", "kper"):
            au = aucs(T, acc[k][idx])
            print(f"{k:5s} " + " ".join(f"{v:7.3f}" for v in au) + f"  {np.nanmean(au):.4f}", flush=True)
    if a.fold >= 0:
        ie = [j for j, s in enumerate(sid) if s in ev.index]
        table(f"OOF vs {a.eval_labels} (= logged teach yardstick)", ev.loc[[sid[j] for j in ie], LABELS].values, ie)
        il = [j for j, s in enumerate(sid) if s in lab.index]
        table(f"OOF vs {a.labels} (training labels)", lab.loc[[sid[j] for j in il], LABELS].values, il)
    ig = [j for j, s in enumerate(sid) if s in gold.index]
    if len(ig) >= 8:
        table("vs train.csv GOLD (expert)", gold.loc[[sid[j] for j in ig], LABELS].values, ig)
    else:
        print(f"\n== gold: only {len(ig)} gold studies here, no AUC", flush=True)
    print("PW_INFER DONE", flush=True)


if __name__ == "__main__":
    main()
