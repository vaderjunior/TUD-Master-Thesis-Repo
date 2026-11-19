import argparse
from collections import Counter
from pathlib import Path
from src.hsrag.paths import project_path

import numpy as np
import pandas as pd

PROCESSED = project_path("data/processed")
SEED = 42

SUBSETS = {
    ("en", "dev"): {"sq1_tune": 250, "main": 300, "sq3_feedback": 350},
    ("en", "test"): {None: 500},
    ("de", "dev"): {None: 150},
    ("de", "test"): {None: 300},
    ("de_legal", "dev"): {None: 175},
    ("de_legal", "test"): {None: 177},
}

LIST_DIMS = ["target_groups", "hate_types"]

DIM_SUBSETS = {
    ("en", "dev"): 15,
    ("en", "test"): 25,
}
DIM_COLS = {"targets": "target_groups", "types": "hate_types", "severity": "severity"}
MIN_SUPPORT = 10


def label_stratified(df, col, per_label, seed, exclude_ids):

    pool = df[~df["id"].isin(exclude_ids)].reset_index(drop=True)

    labels_by_idx = {}
    for i, v in enumerate(pool[col]):
        v = clean(v)
        if not v:
            continue
        labels_by_idx[i] = [v] if isinstance(v, str) else list(v)

    counts = Counter(l for ls in labels_by_idx.values() for l in ls)
    if not counts:
        return pool.iloc[0:0]

    rng = np.random.default_rng(seed)
    chosen = set()
    for label in sorted(counts, key=lambda l: counts[l]):
        have = sum(1 for i in chosen if label in labels_by_idx[i])
        need = per_label - have
        if need <= 0:
            continue
        cands = [
            i for i, ls in labels_by_idx.items() if label in ls and i not in chosen
        ]
        rng.shuffle(cands)
        chosen.update(cands[:need])

    out = pool.iloc[sorted(chosen)]
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def clean(val):

    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def _largest_remainder(exact: dict, total: int) -> dict:

    take = {k: int(np.floor(v)) for k, v in exact.items()}
    short = total - sum(take.values())
    order = sorted(exact, key=lambda k: (exact[k] - take[k], str(k)), reverse=True)
    for k in order[:short]:
        take[k] += 1
    return take


def draw_pool(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:

    strata = list(df.groupby(["source", "gate"], dropna=False))
    total = len(df)
    exact = {key: len(g) / total * n for key, g in strata}
    take = _largest_remainder(exact, n)

    parts = []
    for key, group in strata:
        k = min(take[key], len(group))
        if k > 0:
            parts.append(group.sample(n=k, random_state=seed))

    out = pd.concat(parts)

    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def partition(pool: pd.DataFrame, slices: dict, seed: int) -> dict:

    names = list(slices)
    total = sum(slices.values())
    buckets = {n: [] for n in names}

    for _, group in pool.groupby(["source", "gate"], dropna=False):
        group = group.sample(frac=1.0, random_state=seed)
        exact = {n: len(group) * slices[n] / total for n in names}
        take = _largest_remainder(exact, len(group))
        i = 0
        for n in names:
            if take[n] > 0:
                buckets[n].append(group.iloc[i : i + take[n]])
            i += take[n]

    out = {}
    for n, parts in buckets.items():
        df = pd.concat(parts) if parts else pool.iloc[0:0]
        out[n] = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def report(df: pd.DataFrame, name: str, parent: pd.DataFrame) -> None:

    print(f"\n{'=' * 68}\n{name}: {len(df)} items (parent split: {len(parent)})")

    print("\n  composition (subset % vs parent %):")
    for src in sorted(parent["source"].unique()):
        sub = (df["source"] == src).mean() * 100
        par = (parent["source"] == src).mean() * 100
        print(f"    {src:15} {sub:5.1f}%  vs {par:5.1f}%")

    g_sub, g_par = df["gate"].mean(), parent["gate"].mean()
    if pd.isna(g_sub):
        print(f"    {'gate=True':15} not annotated by this dataset")
    else:
        print(f"    {'gate=True':15} {g_sub * 100:5.1f}%  vs {g_par * 100:5.1f}%")

    print("\n  evaluable items per dimension:")
    gate_ann = int(df["gate"].notna().sum())
    print(f"    {'gate':16} {gate_ann:4}  annotated")
    for col in LIST_DIMS + ["legal", "severity"]:
        if col not in df.columns:
            continue
        vals = [clean(v) for v in df[col]]
        annotated = sum(1 for v in vals if v is not None)

        hateful = sum(
            1 for v, g in zip(vals, df["gate"]) if v is not None and g is True
        )
        print(
            f"    {col:16} {annotated:4}  annotated, " f"{hateful:4} of those hateful"
        )

    print(
        "\n  positive items per label "
        "(macro-F1 is only as good as its thinnest label):"
    )
    for col in LIST_DIMS + ["legal"]:
        if col not in df.columns:
            continue
        counter = Counter()
        for v in df[col]:

            v = clean(v)
            if v:
                counter.update(v)
        line = (
            ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
            if counter
            else "none"
        )
        print(f"    {col:16} {line}")

    sev = Counter(v for v in (clean(x) for x in df["severity"]) if v)
    line = ", ".join(f"{k}={v}" for k, v in sorted(sev.items())) if sev else "none"
    print(f"    {'severity':16} {line}")


def out_path(lang: str, split: str, slice_name, processed_dir=None) -> Path:
    stem = f"{lang}_{split}_eval"
    if slice_name is not None:
        stem += f"_{slice_name}"
    return (processed_dir or PROCESSED) / f"{stem}.parquet"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing subsets (frozen means frozen)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the diagnostics, write nothing"
    )
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED)
    args = ap.parse_args(argv)

    for (lang, split), slices in SUBSETS.items():
        paths = {s: out_path(lang, split, s, args.processed_dir) for s in slices}
        dim_paths = (
            [args.processed_dir / f"{lang}_{split}_eval_{n}.parquet" for n in DIM_COLS]
            if DIM_SUBSETS.get((lang, split))
            else []
        )

        exists = any(p.exists() for p in list(paths.values()) + dim_paths)
        if exists and not args.force and not args.dry_run:
            print(
                f"SKIP {lang}_{split}: subsets already exist "
                f"(use --force to overwrite)"
            )
            continue

        parent = pd.read_parquet(args.processed_dir / f"{lang}_{split}.parquet")
        pool = draw_pool(parent, sum(slices.values()), SEED)

        if list(slices) == [None]:
            written = {None: pool}
        else:
            written = partition(pool, slices, SEED)

            ids = [set(df["id"]) for df in written.values()]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    assert not ids[i] & ids[j], "slices overlap"
            assert sum(len(s) for s in ids) == len(pool), "slice sizes drifted"

        for slice_name, df in written.items():
            label = f"{lang}_{split}_eval" + (f"_{slice_name}" if slice_name else "")
            report(df, label, parent)
            if args.dry_run:
                continue
            df.to_parquet(paths[slice_name], index=False)
            print(f"\n  wrote {paths[slice_name]}")

        per_label = DIM_SUBSETS.get((lang, split))
        if not per_label:
            continue

        used = set()
        for df in written.values():
            used |= set(df["id"])

        for name, col in DIM_COLS.items():
            sub = label_stratified(parent, col, per_label, SEED, used)
            if sub.empty:
                print(
                    f"\n{'=' * 68}\n{lang}_{split}_eval_{name}: EMPTY - "
                    f"no positive labels available for {col}"
                )
                continue
            used |= set(sub["id"])
            report(sub, f"{lang}_{split}_eval_{name}", parent)
            if args.dry_run:
                continue
            path = args.processed_dir / f"{lang}_{split}_eval_{name}.parquet"
            sub.to_parquet(path, index=False)
            print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
