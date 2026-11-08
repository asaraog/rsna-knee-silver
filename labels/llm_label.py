"""Label RSNA knee reports with an LLM, emitting graded scores for ranking.

usage:
  python llm_label.py --gold-only                    # 58 studies, fast sanity check
  python llm_label.py --model qwen2.5:7b-instruct    # whole corpus
  python llm_label.py --limit 200 --workers 4

Writes a JSONL cache keyed by StudyInstanceUID so runs are resumable, then a
labels CSV that score_labels.py can read directly.

Design notes, all of them from measurements on the 58 gold studies:

* Emit 0-100 confidences, not 0/1. The metric is AUC, which only sees ranking,
  and binary output collapses the ranking into two ties.
* Word presence carries almost no signal on its own. On the gold set, reports
  mentioning the ACL are positive 44% of the time versus 38% for reports that
  never mention it. Radiologists comment on every major structure whether or not
  it is torn, so the entire signal lives in assertion polarity. That is the part
  regex gets wrong and an LLM gets right.
* Effusion and Synovitis are entangled in the annotations. Reports describing
  effusion alongside synovial thickening were sometimes coded as Synovitis with
  Effusion left at 0. That noise is irreducible from text, so expect these two
  to stay the weakest labels regardless of prompt quality.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
          'Contusion', 'Fracture']

KEYS = ['acl', 'mcl', 'medial_meniscus', 'lateral_meniscus', 'medial_oa',
        'lateral_oa', 'pf_oa', 'effusion', 'synovitis', 'bakers',
        'contusion', 'fracture']

DEFINITIONS = """acl: tear, rupture or injury of the anterior cruciate ligament
mcl: tear, rupture or injury of the medial collateral ligament
medial_meniscus: tear of the medial meniscus (degeneration alone does not count)
lateral_meniscus: tear of the lateral meniscus (degeneration alone does not count)
medial_oa: osteoarthritis of the medial compartment, including cartilage loss,
  chondral defect, joint space narrowing or osteophytes there
lateral_oa: the same, in the lateral compartment
pf_oa: the same, in the patellofemoral compartment (patella, trochlea)
effusion: fluid in the joint
synovitis: synovial thickening, synovial hypertrophy or enhancement
bakers: Baker's cyst, also called a popliteal cyst
contusion: bone contusion, bone bruise or bone marrow edema
fracture: any fracture, including avulsion and osteochondral fracture"""

PROMPT = """You are reading a knee MRI radiology report. It may be in any language.

For each of the 12 findings below, give a confidence from 0 to 100 that the
finding is PRESENT in this knee.

{definitions}

Rules that matter:
- Radiologists describe every major structure whether or not it is abnormal.
  "ACL is intact", "menisco normal", "Kreuzband unauffaellig" all mean ABSENT.
  Score those near 0. Do not score a finding high just because it is mentioned.
- If a structure is never mentioned at all, it is usually normal, but you are
  less certain than an explicit negation. Score it low, not zero.
- Grade your confidence. Use the middle of the range when the report is
  equivocal ("possible", "cannot exclude", "minimal", "probable").
- Judge effusion and synovitis independently. A report can describe both.

Report:
\"\"\"{report}\"\"\"

Reply with only a JSON object, no other text, with exactly these keys:
{keys}"""


def build_prompt(report):
    return PROMPT.format(definitions=DEFINITIONS, report=report.strip()[:6000],
                         keys=', '.join(KEYS))


def call_ollama(prompt, model, host):
    r = requests.post(f'{host}/api/chat', timeout=600, json={
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0, 'num_ctx': 4096},
    })
    r.raise_for_status()
    return r.json()['message']['content']


def parse(text):
    """Pull the JSON object out and coerce to floats in [0, 1]."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    out = {}
    for k in KEYS:
        v = obj.get(k)
        if isinstance(v, str):
            v = re.sub(r'[^0-9.\-]', '', v) or None
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 50.0          # unparseable: sit in the middle rather than bias the ranking
        out[k] = min(max(v, 0.0), 100.0) / 100.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen2.5:7b-instruct')
    ap.add_argument('--host', default=os.environ.get('OLLAMA_HOST', 'http://localhost:11434'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--gold-only', action='store_true')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()

    tag = a.model.replace(':', '_').replace('/', '_')
    cache_path = a.cache or f'cache_{tag}.jsonl'
    out_path = a.out or f'llm_labels_{tag}.csv'

    t = pd.read_csv('train.csv')
    if a.gold_only:
        t = t[t[LABELS].notna().all(axis=1)]
    t = t[['StudyInstanceUID', 'Report']].dropna()
    if a.limit:
        t = t.head(a.limit)

    done = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done[d['StudyInstanceUID']] = d['scores']
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = t[~t.StudyInstanceUID.isin(done)]
    print(f'{len(t)} studies, {len(done)} cached, {len(todo)} to do -> {out_path}')

    if len(todo):
        lock_out = open(cache_path, 'a')
        n = [0]

        def work(row):
            try:
                scores = parse(call_ollama(build_prompt(row.Report), a.model, a.host))
            except Exception as e:                      # noqa: BLE001 - keep the run alive
                print(f'  {row.StudyInstanceUID[:24]}: {type(e).__name__} {e}', file=sys.stderr)
                return
            if scores is None:
                print(f'  {row.StudyInstanceUID[:24]}: unparseable reply', file=sys.stderr)
                return
            done[row.StudyInstanceUID] = scores
            lock_out.write(json.dumps({'StudyInstanceUID': row.StudyInstanceUID,
                                       'scores': scores}) + '\n')
            lock_out.flush()
            n[0] += 1
            if n[0] % 25 == 0:
                print(f'  {n[0]}/{len(todo)}')

        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(work, [r for _, r in todo.iterrows()]))
        lock_out.close()

    rows = []
    for uid in t.StudyInstanceUID:
        s = done.get(uid)
        if s:
            rows.append({'StudyInstanceUID': uid, **{L: s[k] for L, k in zip(LABELS, KEYS)}})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'wrote {len(rows)} rows to {out_path}')
    print(f'score it: python score_labels.py {out_path}')


if __name__ == '__main__':
    main()
