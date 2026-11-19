import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from analysis.feedback.relevance.prototype_coverage import EN_DEV_SUBSETS
from analysis.feedback.relevance.prototype_coverage import PROCESSED
from analysis.feedback.relevance.prototype_coverage import local_gold_sets
from scripts.make_eval_subsets import _largest_remainder, clean, label_stratified
from src.hsrag.metrics import MIN_SUPPORT

PARENT = "en_dev"
NAME = "en_dev_eval_sq3_types"
GOLD_COL = "hate_types"
SEED = 42


HELD_OUT = 120
BATCHES = 4 * 30
SENTINEL_TARGET = 80
R_CONTAINS = 0.427


def partition_by_label(df: pd.DataFrame, sizes: dict, seed: int) -> dict:

    assert sum(sizes.values()) == len(
        df
    ), f"sizes sum to {sum(sizes.values())}, subset has {len(df)}"
    names = list(sizes)
    total = sum(sizes.values())
    buckets = {n: [] for n in names}

    keyed = df.assign(_lbl=[",".join(sorted(clean(v) or [])) for v in df[GOLD_COL]])
    for _, group in keyed.groupby("_lbl"):
        group = group.sample(frac=1.0, random_state=seed)
        exact = {n: len(group) * sizes[n] / total for n in names}
        take = _largest_remainder(exact, len(group))
        i = 0
        for n in names:
            if take[n] > 0:
                buckets[n].append(group.iloc[i : i + take[n]])
            i += take[n]

    out = {}
    for n, parts in buckets.items():
        d = pd.concat(parts) if parts else keyed.iloc[0:0]
        out[n] = (
            d.drop(columns=["_lbl"])
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
    return out


def label_counts(df: pd.DataFrame) -> Counter:
    return Counter(l for v in df[GOLD_COL] for l in (clean(v) or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-label",
        type=int,
        default=68,
        help="items per hate_type label; the rarest label caps it",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force", action="store_true", help="overwrite (frozen means frozen)"
    )
    args = ap.parse_args()

    out_path = PROCESSED / f"{NAME}.parquet"
    if out_path.exists() and not args.force and not args.dry_run:
        print(f"SKIP: {out_path} exists (use --force to overwrite)")
        return

    parent = pd.read_parquet(PROCESSED / f"{PARENT}.parquet")
    used = set()
    for name in EN_DEV_SUBSETS:
        p = PROCESSED / f"{name}.parquet"
        if p.exists():
            used |= set(pd.read_parquet(p, columns=["id"])["id"])
    print(f"{PARENT}: {len(parent)} items, {len(used)} already in a subset")

    avail = label_counts(
        parent[~parent["id"].isin(used)][lambda d: d["gate"] == True]
    )  # noqa: E712
    rarest, rn = min(avail.items(), key=lambda kv: kv[1])
    print(
        f"unused pool, per label: "
        + ", ".join(f"{k}={v}" for k, v in sorted(avail.items(), key=lambda kv: kv[1]))
    )
    print(f"rarest '{rarest}' at {rn} -> per_label caps at {rn}")
    assert args.per_label <= rn, (
        f"--per-label {args.per_label} exceeds the {rn} available for "
        f"'{rarest}'. A balanced draw cannot be filled."
    )

    sub = label_stratified(parent, GOLD_COL, args.per_label, SEED, used)
    assert not sub.empty, "draw came back empty"

    print(f"\n{'=' * 70}\n{NAME}: {len(sub)} items")
    counts = label_counts(sub)
    for label, n in sorted(counts.items(), key=lambda kv: kv[1]):
        print(f"    {label:20} {n:5}")
    card = Counter(len(clean(v) or []) for v in sub[GOLD_COL])
    print(
        f"  cardinality: "
        + ", ".join(f"{k} label(s)={v}" for k, v in sorted(card.items()))
    )
    print(f"  sources: {dict(Counter(sub['source']))}")
    print(f"  gate: {dict(Counter(bool(g) for g in sub['gate']))}")

    pool_n = len(sub) - HELD_OUT - BATCHES
    need = round(SENTINEL_TARGET / R_CONTAINS)
    print(f"\n  partition: pool {pool_n} / held_out {HELD_OUT} / " f"batches {BATCHES}")
    print(
        f"  sentinel needs a pool of {need} at r={R_CONTAINS} "
        f"(contains, measured on 3 replicates)"
    )
    assert pool_n >= need, (
        f"pool {pool_n} < {need} needed for a {SENTINEL_TARGET}-item "
        f"sentinel. Raise --per-label, or lower HELD_OUT / BATCHES / "
        f"SENTINEL_TARGET - but change one deliberately, not to make an "
        f"assert pass."
    )

    parts = partition_by_label(
        sub, {"pool": pool_n, "held_out": HELD_OUT, "batches": BATCHES}, SEED
    )

    print(f"\n  {'role':12} {'n':>5}  per-label")
    for role, d in parts.items():
        c = label_counts(d)
        thin = [k for k, v in c.items() if v < MIN_SUPPORT]
        print(
            f"  {role:12} {len(d):5}  "
            + ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
            + (f"   << below MIN_SUPPORT: {thin}" if thin else "")
        )

    thin = [k for k, v in label_counts(parts["held_out"]).items() if v < MIN_SUPPORT]
    assert not thin, (
        f"held_out labels below MIN_SUPPORT: {thin}. The learning curve would "
        f"average noise on these. Raise --per-label or HELD_OUT."
    )

    ids = {r: set(d["id"]) for r, d in parts.items()}
    roles = list(ids)
    for i, a in enumerate(roles):
        for b in roles[i + 1 :]:
            assert not ids[a] & ids[b], f"{a} and {b} overlap"
    assert sum(len(v) for v in ids.values()) == len(sub), "roles lost items"
    assert not set(sub["id"]) & used, "overlaps an existing subset"

    scorable = local_gold_sets(sub, GOLD_COL)
    assert len(scorable) == len(
        sub
    ), f"{len(sub) - len(scorable)} drawn items are not hate_type-scorable"
    print(f"\n  disjoint from all {len(EN_DEV_SUBSETS)} existing subsets: OK")
    print(f"  roles mutually disjoint and exhaustive: OK")
    print(f"  all {len(sub)} items hate_type-scorable: OK")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return

    tagged = pd.concat([d.assign(sq3_role=role) for role, d in parts.items()])
    tagged = tagged.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    tagged.to_parquet(out_path, index=False)
    print(f"\n  wrote {out_path}  ({len(tagged)} items, sq3_role column)")
    print(
        f"  ADD TO EN_DEV_SUBSETS in check_sq3_coverage.py so future "
        f"disjointness checks see it."
    )


if __name__ == "__main__":
    main()
