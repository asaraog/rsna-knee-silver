"""Score training-label sets against the 58 expert-labelled studies -> data/label_auc.json (Figure 2).

usage: python make_data.py --train train.csv --set "Ours (report-only, LLM + Claude)=labels_v4fb_sharp.csv" \
           --set "Public frontier blend=blend_top3_argon_astra_opus_all4314_labels.csv" \
           --set "1st place, image + report=labels_astra_trained_guide.csv"
Each label CSV needs StudyInstanceUID plus the 12 finding columns (any monotonic scale)."""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc

LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--set", action="append", required=True)
a = ap.parse_args()
T = pd.read_csv(a.train, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
G = T[T[LAB].notna().all(axis=1)][LAB]
out = {}
for spec in a.set:
    name, path = spec.split("=", 1)
    X = pd.read_csv(path, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    r = {c: round(float(auc(G[c], X[c].reindex(G.index))), 3) for c in LAB}
    r["macro"] = round(float(np.mean([r[c] for c in LAB])), 3); out[name] = r
    print(f"{name}: macro {r['macro']}")
here = os.path.dirname(os.path.abspath(__file__))
json.dump({"labels": out}, open(os.path.join(here, "data", "label_auc.json"), "w"), indent=1)
