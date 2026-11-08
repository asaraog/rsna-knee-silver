"""Score a candidate label file against the 58 expert-labeled training studies.

usage: python score_labels.py report_labels_v1.csv [more.csv ...]

The candidate CSV needs StudyInstanceUID plus the 12 label columns, holding
scores in any monotonic scale. AUC only cares about ranking.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
          'Contusion', 'Fracture']

N_BOOT = 2000
RNG = np.random.default_rng(0)


def gold():
    t = pd.read_csv('train.csv')
    g = t[t[LABELS].notna().all(axis=1)]
    return g[['StudyInstanceUID', 'Report'] + LABELS].reset_index(drop=True)


def boot_ci(y, s):
    """Percentile CI on AUC. At n=58 the interval is wide and that is the point."""
    stats = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        stats.append(roc_auc_score(y[i], s[i]))
    return np.percentile(stats, [2.5, 97.5])


def score(path, g):
    p = pd.read_csv(path)
    missing = [c for c in ['StudyInstanceUID'] + LABELS if c not in p.columns]
    if missing:
        sys.exit(f'{path}: missing columns {missing}')

    m = g.merge(p[['StudyInstanceUID'] + LABELS], on='StudyInstanceUID',
                suffixes=('_gold', '_pred'))
    if len(m) < len(g):
        print(f'  warning: only {len(m)}/{len(g)} gold studies present')

    print(f'\n{path}   n={len(m)}')
    print(f'  {"label":18s} {"AUC":>6s} {"95% CI":>16s} {"flat":>6s}')
    aucs = []
    for c in LABELS:
        y = m[c + '_gold'].values.astype(int)
        s = m[c + '_pred'].values.astype(float)
        if len(np.unique(y)) < 2:
            print(f'  {c:18s}    n/a  (one class only)')
            continue
        a = roc_auc_score(y, s)
        lo, hi = boot_ci(y, s)
        # share of the full candidate file where this column is a single constant,
        # i.e. where the labeler found nothing and contributes no ranking signal
        flat = (p[c].value_counts(normalize=True).iloc[0]) if p[c].nunique() > 0 else 1.0
        aucs.append(a)
        print(f'  {c:18s} {a:6.3f}  [{lo:.3f}, {hi:.3f}] {flat:6.1%}')
    print(f'  {"MACRO":18s} {np.mean(aucs):6.3f}   over {len(aucs)} labels')
    return np.mean(aucs)


if __name__ == '__main__':
    files = sys.argv[1:] or ['report_labels_v1.csv']
    g = gold()
    results = {f: score(f, g) for f in files}
    if len(results) > 1:
        print('\nsummary')
        for f, a in sorted(results.items(), key=lambda kv: -kv[1]):
            print(f'  {a:.4f}  {f}')
