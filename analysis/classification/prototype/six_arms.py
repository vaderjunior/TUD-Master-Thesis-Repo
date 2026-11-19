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
tmpl = {k: m["templ_id"] for k, m in meta.items()}

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
BLOCKS = ["guideline_covered", "definition_covered", "uncovered"]


def load(stem, arm):
    out = {}
    for line in (R / f"{stem}_live.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" in r or r["arm"] != arm:
            continue
        out[r["item_id"]] = r
    return out


P = {rep: {a: load(SRC[a][0].format(rep), SRC[a][1]) for a in ARMS} for rep in REPS}

common = set(gold)
for rep in REPS:
    for a in ARMS:
        common &= {i for i, r in P[rep][a].items() if r["result"] is not None}
nh = sorted(i for i in common if not gate[i])
h = sorted(i for i in common if gate[i])
print(f"common across 6 arms x 3 reps: {len(common)} of {len(gold)}")
print(f"  non-hateful {len(nh)}   hateful {len(h)}\n")


def fp(rep, arm, ids):
    return sum(1 for i in ids if P[rep][arm][i]["result"]["hate"]) / len(ids)


print("=" * 78)
print("GATE FALSE-POSITIVE RATE")
FPM = {}
for scope in ["ALL_NON_HATEFUL"] + BLOCKS:
    ids = nh if scope == "ALL_NON_HATEFUL" else [i for i in nh if blk[i] == scope]
    print(f"\n  {scope}  n={len(ids)}")
    st = {}
    for a in ARMS:
        v = [fp(rep, a, ids) for rep in REPS]
        st[a] = (float(np.mean(v)), max(v) - min(v), v)
        print(
            f"    {a:18s} {' '.join(f'{x:.4f}' for x in v)}"
            f"   mean {np.mean(v):.4f}  spread {max(v)-min(v):.4f}"
        )
    FPM[scope] = st
    for a, b in [
        ("rag", "zero_shot"),
        ("examples_only", "zero_shot"),
        ("definitions_only", "zero_shot"),
        ("guidelines_only", "zero_shot"),
        ("few_shot", "zero_shot"),
        ("abstract_only", "zero_shot"),
        ("examples_only", "rag"),
        ("examples_only", "few_shot"),
        ("definitions_only", "rag"),
        ("abstract_only", "definitions_only"),
        ("abstract_only", "rag"),
    ]:
        d = st[a][0] - st[b][0]
        fl = max(st[a][1], st[b][1])
        sep = max(st[a][2]) < min(st[b][2]) or min(st[a][2]) > max(st[b][2])
        print(
            f"      {a} - {b:18s} {d:+.4f}  floor {fl:.4f}  "
            f"{abs(d)/fl if fl else 0:5.2f}x  "
            f"{'CLEARS' if fl and abs(d) > fl else 'inside floor'}"
            f"{'  complete separation' if sep else ''}"
        )
    s = st
    resid = (
        (s["definitions_only"][0] - s["zero_shot"][0])
        + (s["guidelines_only"][0] - s["zero_shot"][0])
        + (s["examples_only"][0] - s["zero_shot"][0])
        - (s["rag"][0] - s["zero_shot"][0])
    )
    fl = max(v[1] for v in s.values())
    print(
        f"      ADDITIVITY residual {resid:+.4f}  widest floor {fl:.4f}  "
        f"{abs(resid)/fl:5.2f}x  "
        f"{'HOLDS' if abs(resid) <= fl else 'FAILS'}"
    )

    pair = (
        (s["definitions_only"][0] - s["zero_shot"][0])
        + (s["guidelines_only"][0] - s["zero_shot"][0])
        - (s["abstract_only"][0] - s["zero_shot"][0])
    )
    pfl = max(
        s[k][1]
        for k in ("definitions_only", "guidelines_only", "abstract_only", "zero_shot")
    )
    print(
        f"      D+G vs AB residual  {pair:+.4f}  widest floor {pfl:.4f}  "
        f"{abs(pair)/pfl:5.2f}x  "
        f"{'HOLDS' if abs(pair) <= pfl else 'FAILS'}"
    )

print("\n" + "=" * 78)
print("CLUSTERED PAIRED BOOTSTRAP on templ_id, pooled over replicates")
rng = np.random.default_rng(42)
for scope in ["ALL_NON_HATEFUL"] + BLOCKS:
    ids = nh if scope == "ALL_NON_HATEFUL" else [i for i in nh if blk[i] == scope]
    by_t = defaultdict(list)
    for i in ids:
        by_t[tmpl[i]].append(i)
    tids = sorted(by_t)
    print(f"\n  {scope}  n={len(ids)}  templates={len(tids)}")
    for a, b in [
        ("rag", "zero_shot"),
        ("examples_only", "zero_shot"),
        ("definitions_only", "zero_shot"),
        ("abstract_only", "zero_shot"),
        ("examples_only", "rag"),
        ("examples_only", "few_shot"),
    ]:

        def rate(arm, s):
            return np.mean(
                [
                    sum(1 for i in s if P[rep][arm][i]["result"]["hate"]) / len(s)
                    for rep in REPS
                ]
            )

        obs = rate(a, ids) - rate(b, ids)
        dr = []
        for _ in range(3000):
            s = [i for t in rng.choice(tids, len(tids), replace=True) for i in by_t[t]]
            dr.append(rate(a, s) - rate(b, s))
        lo, hi = np.percentile(dr, [2.5, 97.5])
        print(
            f"    {a} - {b:18s} obs {obs:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
            f"{'excludes 0' if lo * hi > 0 else 'CONTAINS ZERO'}"
        )

print("\n" + "=" * 78)
print("TARGET_GROUP, hateful side, common items only (gold cardinality 1.00)")
TG = {}
for a in ARMS:
    ms, cs = [], []
    for rep in REPS:
        rows = [P[rep][a][i] for i in h]
        s = score_multilabel(rows, gold, "target_groups", "target_group")
        ms.append(s.macro_f1)
        cs.append(s.extra["mean_pred_labels"])
    TG[a] = (float(np.mean(ms)), max(ms) - min(ms), ms, float(np.mean(cs)))
    print(
        f"  {a:18s} {' '.join(f'{x:.4f}' for x in ms)}  mean {np.mean(ms):.4f}"
        f"  spread {max(ms)-min(ms):.4f}  pred/item {np.mean(cs):.3f}"
    )
for a, b in [
    ("definitions_only", "zero_shot"),
    ("rag", "zero_shot"),
    ("abstract_only", "zero_shot"),
    ("examples_only", "zero_shot"),
    ("guidelines_only", "zero_shot"),
    ("few_shot", "zero_shot"),
    ("definitions_only", "rag"),
    ("abstract_only", "definitions_only"),
    ("abstract_only", "rag"),
]:
    d = TG[a][0] - TG[b][0]
    fl = max(TG[a][1], TG[b][1])
    sep = max(TG[a][2]) < min(TG[b][2]) or min(TG[a][2]) > max(TG[b][2])
    print(
        f"    {a} - {b:18s} {d:+.4f}  floor {fl:.4f}  {abs(d)/fl:5.2f}x  "
        f"{'CLEARS' if abs(d) > fl else 'inside floor'}"
        f"{'  complete separation' if sep else ''}"
    )
resid = sum(
    TG[k][0] - TG["zero_shot"][0]
    for k in ("definitions_only", "guidelines_only", "examples_only")
) - (TG["rag"][0] - TG["zero_shot"][0])
fl = max(v[1] for v in TG.values())
print(
    f"    ADDITIVITY residual {resid:+.4f}  widest floor {fl:.4f}  "
    f"{abs(resid)/fl:5.2f}x  {'HOLDS' if abs(resid) <= fl else 'FAILS'}"
)

print("\n  per-label F1, mean over replicates")
labs = [
    "disability",
    "gender",
    "national_origin",
    "race",
    "religion",
    "sexual_orientation",
]
print(f"  {'arm':18s} " + " ".join(f"{l[:11]:>12s}" for l in labs))
for a in ARMS:
    cells = []
    for lab in labs:
        vals = []
        for rep in REPS:
            s = score_multilabel(
                [P[rep][a][i] for i in h], gold, "target_groups", "target_group"
            )
            vals.append(s.per_label.get(lab, {}).get("f1", float("nan")))
        cells.append(float(np.mean(vals)))
    print(f"  {a:18s} " + " ".join(f"{c:12.4f}" for c in cells))

print("\n" + "=" * 78)
print("SCHEMA REPAIRS and PARSE FAILURES per item, common set")
for a in ARMS:
    rp = [np.mean([P[rep][a][i]["repairs"] for i in common]) for rep in REPS]
    pf = [
        np.mean(
            [P[rep][a][i]["parse_failures"] / P[rep][a][i]["n_runs"] for i in common]
        )
        for rep in REPS
    ]
    print(
        f"  {a:18s} repairs {np.mean(rp):.4f} (spread {max(rp)-min(rp):.4f})"
        f"   parse-fail {np.mean(pf):.4f}"
    )
