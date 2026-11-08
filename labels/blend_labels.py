"""Rank-average two or more label files into one.

usage: python blend_labels.py report_labels_v1.csv llm_labels.csv -o blend.csv

Why rank-average rather than weighted or per-label selection: AUC only sees
ranking, so averaging ranks is the natural combination, and it uses none of the
58 gold labels. Picking a per-label winner on n=58 would be fitting noise.

Measured on the 52 gold studies labeled so far (2026-08-06):

  regex alone   0.8081
  qwen 7b alone 0.8037
  rank blend    0.8530

The two are complementary because they fail in different places. The regex
system is strong on ligaments and menisci, where the whole task is parsing
negation ("ACL is intact"), and it gives up entirely on the compartment
osteoarthritis labels, defaulting on 70% of Medial OA. The LLM is the reverse:
weaker on MCL and medial meniscus, far stronger on Medial OA, 0.948 against
0.759. The blend beats both on 7 of the 12 labels.
"""
import argparse

import pandas as pd
from scipy.stats import rankdata

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
          'Contusion', 'Fracture']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('-o', '--out', default='blend.csv')
    a = ap.parse_args()

    frames = []
    for f in a.files:
        d = pd.read_csv(f)[['StudyInstanceUID'] + LABELS].set_index('StudyInstanceUID')
        frames.append(d)
        print(f'{f}: {len(d)} studies')

    # inner join, so the blend covers only studies every input scored
    idx = frames[0].index
    for d in frames[1:]:
        idx = idx.intersection(d.index)
    print(f'common to all inputs: {len(idx)}')

    out = pd.DataFrame(index=idx)
    for c in LABELS:
        total = sum(rankdata(d.loc[idx, c].values) for d in frames)
        out[c] = total / (len(frames) * len(idx))   # back into [0, 1], ranking preserved

    out.reset_index().to_csv(a.out, index=False)
    print(f'wrote {a.out}')
    print(f'score it: python score_labels.py {a.out}')


if __name__ == '__main__':
    main()
