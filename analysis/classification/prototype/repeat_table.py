import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.hsrag.metrics import score_multilabel

R = Path("experiments/results")
df = pd.read_parquet("data/processed/en_hatecheck_eval.parquet")
gold = {str(r.id): r for r in df.itertuples(index=False)}
meta = {str(r.id): json.loads(r.raw) for r in df.itertuples(index=False)}
gate = {k: bool(v.gate) for k, v in gold.items()}
blk = {k: m["block"] for k, m in meta.items()}
func = {k: m["functionality"] for k, m in meta.items()}

SRC = {
    "zero_shot": ("hatecheck_A_r{}", "zero_shot"),
    "few_shot": ("hatecheck_A_r{}", "few_shot"),
    "rag": ("hatecheck_A_r{}", "rag"),
    "guidelines_only": ("hatecheck_G_r{}", "rag"),
    "definitions_only": ("hatecheck_D_r{}", "rag"),
    "examples_only": ("hatecheck_E_r{}", "rag"),
    "abstract_only": ("hatecheck_AB_r{}", "rag"),
}
ARMS = list(SRC)
REPS = (1, 2, 3)


def load(stem, arm):
    out = {}
    for line in (R / f"{stem}_live.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" not in r and r["arm"] == arm:
            out[r["item_id"]] = r
    return out


P = {rep: {a: load(SRC[a][0].format(rep), SRC[a][1]) for a in ARMS} for rep in REPS}
common = set(gold)
for rep in REPS:
    for a in ARMS:
        common &= {i for i, r in P[rep][a].items() if r["result"] is not None}
nh = sorted(i for i in common if not gate[i])
h = sorted(i for i in common if gate[i])
print(f"common set {len(common)}  non-hateful {len(nh)}  hateful {len(h)}\n")

print("=" * 78)
print("FALSE-POSITIVE COUNTS, all non-hateful, per replicate")
print(f"  {'arm':18s} {'r1':>6} {'r2':>6} {'r3':>6}  of {len(nh)}")
for a in ARMS:
    c = [sum(1 for i in nh if P[rep][a][i]["result"]["hate"]) for rep in REPS]
    print(f"  {a:18s} {c[0]:6d} {c[1]:6d} {c[2]:6d}")

print("\nFALSE-NEGATIVE COUNTS, hateful side, per replicate")
print(f"  {'arm':18s} {'r1':>6} {'r2':>6} {'r3':>6}  of {len(h)}")
for a in ARMS:
    c = [sum(1 for i in h if not P[rep][a][i]["result"]["hate"]) for rep in REPS]
    print(f"  {a:18s} {c[0]:6d} {c[1]:6d} {c[2]:6d}")

print("\n" + "=" * 78)
print("GATE MACRO-F1 over both classes, per replicate, common set")
from src.hsrag.metrics import score_gate

print(f"  {'arm':18s} {'r1':>8} {'r2':>8} {'r3':>8} {'mean':>8} {'spread':>8}")
for a in ARMS:
    v = []
    for rep in REPS:
        s = score_gate([P[rep][a][i] for i in common], gold, "strict")
        v.append(s.macro_f1)
    print(
        f"  {a:18s} {v[0]:8.4f} {v[1]:8.4f} {v[2]:8.4f} "
        f"{np.mean(v):8.4f} {max(v)-min(v):8.4f}"
    )

print("\n" + "=" * 78)
print("FP RATE PER FUNCTIONALITY, mean over replicates, seven arms")
fns = sorted({func[i] for i in nh}, key=lambda f: -sum(1 for i in nh if func[i] == f))
print(f"  {'functionality':20s} {'n':>4} " + " ".join(f"{a[:9]:>9s}" for a in ARMS))
for f in fns:
    ids = [i for i in nh if func[i] == f]
    cells = [
        np.mean(
            [
                sum(1 for i in ids if P[rep][a][i]["result"]["hate"]) / len(ids)
                for rep in REPS
            ]
        )
        for a in ARMS
    ]
    print(f"  {f:20s} {len(ids):4d} " + " ".join(f"{c:9.3f}" for c in cells))

print("\n" + "=" * 78)
print("TARGET_GROUP per-label F1, per replicate, for the log table")
labs = [
    "disability",
    "gender",
    "national_origin",
    "race",
    "religion",
    "sexual_orientation",
]
for a in ARMS:
    print(f"\n  {a}")
    for lab in labs:
        v = []
        for rep in REPS:
            s = score_multilabel(
                [P[rep][a][i] for i in h], gold, "target_groups", "target_group"
            )
            v.append(s.per_label.get(lab, {}).get("f1", float("nan")))
        print(
            f"    {lab:20s} {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}  "
            f"mean {np.mean(v):.4f}  spread {max(v)-min(v):.4f}"
        )

print("\n" + "=" * 78)
print("STRATEGY SWEEP, hate_type macro, en_dev_eval_sq3_types")
sdf = pd.read_parquet("data/processed/en_dev_eval_sq3_types.parquet")
sgold = {str(r.id): r for r in sdf.itertuples(index=False)}
for strat in ("dense", "hybrid", "bm25"):
    v, card = [], []
    for rep in (1, 2, 3):
        rows = list(load(f"strat_{strat}_r{rep}", "rag").values())
        s = score_multilabel(rows, sgold, "hate_types", "hate_type")
        v.append(s.macro_f1)
        card.append(s.extra["mean_pred_labels"])
    print(
        f"  {strat:8s} {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}  mean {np.mean(v):.4f}"
        f"  spread {max(v)-min(v):.4f}  pred/item {np.mean(card):.3f}"
    )
print("  historical hybrid, sq3_round0_r1..r4, for the reproduction claim:")
hv = []
for rep in (1, 2, 3, 4):
    rows = list(load(f"sq3_round0_r{rep}", "rag").values())
    s = score_multilabel(rows, sgold, "hate_types", "hate_type")
    hv.append(s.macro_f1)
print(
    f"  {'sq3_round0':8s} "
    + " ".join(f"{x:.4f}" for x in hv)
    + f"  mean {np.mean(hv):.4f}  spread {max(hv)-min(hv):.4f}"
)
