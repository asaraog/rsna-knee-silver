#!/usr/bin/env python3
"""crop_predict.py <crops_dir> <out.csv> <ckpt...> - fold-averaged crop-classifier probabilities per study.
Medial crop -> Medial Meniscus / Medial OA, lateral crop -> Lateral Meniscus / Lateral OA (as trained)."""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn, timm

d, outf, cks = sys.argv[1], sys.argv[2], sys.argv[3:]
C = np.load(f"{d}/crops.npy", mmap_mode="r"); meta = pd.read_csv(f"{d}/meta.csv"); meta = meta[meta.ok == 1].drop_duplicates("StudyInstanceUID")
s0 = torch.load(cks[0], map_location="cpu", weights_only=False)
SPEC = [x.split(",") for x in s0.get("spec", "Medial Meniscus,Medial OA|Lateral Meniscus,Lateral OA").split("|")]
COLS = list(dict.fromkeys(c for g in SPEC for c in g)); OWN = [[COLS.index(c) for c in g] for g in SPEC]


class Net(nn.Module):
    def __init__(self, arch):
        super().__init__(); self.b = timm.create_model(arch, pretrained=False, num_classes=0); self.h = nn.Linear(2 * self.b.num_features, len(COLS))
    def forward(self, x):
        B, S = x.shape[:2]; f = self.b(x.flatten(0, 1)).view(B, S, -1)
        return self.h(torch.cat([f.mean(1), f.amax(1)], 1))


P = np.zeros((len(meta), len(COLS))); NC = np.zeros(len(COLS))
for g in OWN:
    for q in g: NC[q] += 1
for ck in cks:
    s = torch.load(ck, map_location="cpu", weights_only=False); net = Net(s["arch"]); net.load_state_dict(s["model"]); net = net.cuda().eval()
    for b0 in range(0, len(meta), 8):
        rows = meta.iloc[b0:b0 + 8]
        for c in range(len(SPEC)):
            x = torch.from_numpy(np.stack([np.asarray(C[int(n), c], dtype=np.float32) / 255.0 for n in rows.idx]))[:, :, None].repeat(1, 1, 3, 1, 1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                o = torch.sigmoid(net(x.cuda()).float()).cpu().numpy()
            for q in OWN[c]: P[b0:b0 + len(rows), q] += o[:, q]
    print("done", ck, flush=True)
pd.DataFrame(P / len(cks) / NC, index=meta.StudyInstanceUID.values, columns=COLS).rename_axis("StudyInstanceUID").to_csv(outf)
print("PREDICT DONE", len(meta))
