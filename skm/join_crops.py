#!/usr/bin/env python3
"""join_crops.py <sag_dir> <cor_dir> <out_dir> - two-plane compartment crops: per study and compartment, the 8 sagittal crops (skm_crops.py)
followed by the 8 coronal crops (skm_crops_cor.py) -> crops.npy [N, 2, 16, 224, 224]. Rows aligned by StudyInstanceUID; studies missing a plane
get that plane's 8 slots zero-filled. crop_train.py reads it unchanged (default spec; mean+max over 16 slices)."""
import os, sys, numpy as np, pandas as pd
sd, cd, od = sys.argv[1:4]; os.makedirs(od, exist_ok=True)
S = np.load(f"{sd}/crops.npy", mmap_mode="r"); Cc = np.load(f"{cd}/crops.npy", mmap_mode="r")
ms = pd.read_csv(f"{sd}/meta.csv").drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
mc = pd.read_csv(f"{cd}/meta.csv").drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
ids = sorted(set(ms.index[ms.ok == 1]) | set(mc.index[mc.ok == 1]))
O = np.lib.format.open_memmap(f"{od}/crops.npy", mode="w+", dtype=np.uint8, shape=(len(ids), 2, S.shape[2] + Cc.shape[2], S.shape[3], S.shape[4]))
rows = []
for n, sid in enumerate(ids):
    if sid in ms.index and ms.at[sid, "ok"] == 1: O[n, :, :S.shape[2]] = S[int(ms.at[sid, "idx"])]
    if sid in mc.index and mc.at[sid, "ok"] == 1: O[n, :, S.shape[2]:] = Cc[int(mc.at[sid, "idx"])]
    rows.append({"StudyInstanceUID": sid, "idx": n, "ok": 1})
    if n % 500 == 0: print(n, len(ids), flush=True)
O.flush(); pd.DataFrame(rows).to_csv(f"{od}/meta.csv", index=False)
print("JOINED", len(ids), "both planes:", len(set(ms.index[ms.ok == 1]) & set(mc.index[mc.ok == 1])))
