import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.hsrag.metrics import score_gate

R = Path("experiments/results")
df = pd.read_parquet("data/processed/en_hatecheck_eval.parquet")
gold = {str(r.id): r for r in df.itertuples(index=False)}
tmpl = {str(r.id): json.loads(r.raw)["templ_id"] for r in df.itertuples(index=False)}
N_BOOT, SEED = 3000, 42


def load(stem, arm):
    out = {}
    for line in (R / f"{stem}_live.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" not in r and r["arm"] == arm and r["result"] is not None:
            out[r["item_id"]] = r
    return out


ARMS = {
    a: {rep: load(f"hatecheck_A_r{rep}", a) for rep in (1, 2, 3)}
    for a in ("zero_shot", "few_shot", "rag")
}

print("SELF-CHECK, gate macro on the common set")
common = set(gold)
for a in ARMS:
    for rep in (1, 2, 3):
        common &= set(ARMS[a][rep])
common = sorted(common)
print(f"  common {len(common)}")
for a in ARMS:
    v = [
        score_gate([ARMS[a][rep][i] for i in common], gold, "strict").macro_f1
        for rep in (1, 2, 3)
    ]
    print(
        f"  {a:10s} {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}  mean {np.mean(v):.4f}"
        f"  spread {max(v)-min(v):.4f}"
    )

by_t = defaultdict(list)
for i in common:
    by_t[tmpl[i]].append(i)
tids = sorted(by_t)
print(f"  {len(tids)} templates\n")

rng = np.random.default_rng(SEED)
for a, b in [("rag", "few_shot"), ("rag", "zero_shot"), ("few_shot", "zero_shot")]:
    print(f"{'=' * 66}\n{a} - {b}   {N_BOOT} draws, clustered on templ_id")
    for rep in (1, 2, 3):
        A, B = ARMS[a][rep], ARMS[b][rep]
        obs = (
            score_gate([A[i] for i in common], gold, "strict").macro_f1
            - score_gate([B[i] for i in common], gold, "strict").macro_f1
        )
        draws = np.empty(N_BOOT)
        for k in range(N_BOOT):
            samp = [
                i for t in rng.choice(tids, len(tids), replace=True) for i in by_t[t]
            ]
            draws[k] = (
                score_gate([A[i] for i in samp], gold, "strict").macro_f1
                - score_gate([B[i] for i in samp], gold, "strict").macro_f1
            )
        lo, hi = np.percentile(draws, [2.5, 97.5])
        p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
        print(
            f"  r{rep}  obs {obs:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  p {p:.3f}"
            f"  {'excludes 0' if lo * hi > 0 else 'CONTAINS ZERO'}"
        )
