import argparse
import json
from collections import Counter
from pathlib import Path
from src.hsrag.paths import project_path

import pandas as pd
import yaml

PROCESSED = project_path("data/processed")
OUT = project_path("data/encoder")
EXCL = project_path("experiments/leakage_train_exclusions.json")
KB_RECORDS = project_path("kb/records.jsonl")
TAXONOMY = project_path("config/taxonomy.yaml")

HEADS = {
    "hate": ("gate", "binary", False),
    "target_group": ("target_groups", "multilabel", True),
    "hate_type": ("hate_types", "multilabel", True),
    "severity": ("severity", "ordinal", True),
    "legal": ("legal", "multilabel", False),
}

ARMS = {
    "full": {
        "lang": "en",
        "source": ["en_train"],
        "heads": ["hate", "target_group", "hate_type", "severity"],
    },
    "de": {
        "lang": "de",
        "source": ["de_train", "de_legal_train"],
        "heads": ["hate", "legal"],
    },
    "kb": {
        "lang": "en",
        "source": ["__kb__"],
        "heads": ["hate", "target_group", "hate_type", "severity"],
    },
}

VAL_FRACTION = 0.1
SEED = 42
MIN_SUPPORT = 10


def clean(val):

    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def taxonomy_labels(dim: str) -> list:
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    d = tax["dimensions"][dim]
    if d["type"] == "binary":
        return ["false", "true"]
    labels = d["labels"]
    return list(labels) if isinstance(labels, dict) else list(labels)


def load_kb_examples() -> pd.DataFrame:

    rows = []
    for line in KB_RECORDS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "example":
            continue
        m = r.get("meta", {})
        rows.append(
            {
                "id": r["id"],
                "text": r["text"],
                "lang": r["lang"],
                "source": r.get("source", "kb"),
                "gate": m.get("gate"),
                "target_groups": m.get("target_groups"),
                "hate_types": m.get("hate_types"),
                "severity": m.get("severity"),
                "legal": None,
            }
        )
    return pd.DataFrame(rows)


def load_pool(arm: str) -> pd.DataFrame:

    spec = ARMS[arm]
    if spec["source"] == ["__kb__"]:
        return _by_lang(load_kb_examples(), spec["lang"], "kb examples")
    frames = []
    for name in spec["source"]:
        p = PROCESSED / f"{name}.parquet"
        if not p.exists():
            print(f"  MISSING {p}")
            continue
        df = pd.read_parquet(p)
        df["_split"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return _by_lang(
        pd.concat(frames, ignore_index=True), spec["lang"], "+".join(spec["source"])
    )


def _by_lang(df: pd.DataFrame, lang: str, label: str) -> pd.DataFrame:
    if df.empty or "lang" not in df.columns:
        return df
    before = len(df)
    out = df[df["lang"].astype(str) == lang].reset_index(drop=True)
    if len(out) != before:
        print(
            f"  language filter: {label} {before} -> {len(out)} rows "
            f"(lang == {lang!r}); {before - len(out)} dropped"
        )
    return out


def support_of(rows: list, kind: str, labels: list) -> dict:
    c = Counter()
    for v in rows:
        if kind == "binary":
            c[str(bool(v)).lower()] += 1
        elif kind == "ordinal":
            if v is not None:
                c[str(v)] += 1
        else:
            for l in v or []:
                c[str(l)] += 1
    return {l: c.get(l, 0) for l in labels}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report every count and write nothing. Run this "
        "first: a head that turns out unbuildable is much "
        "cheaper to find here than after training.",
    )
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--head",
        default=None,
        help="build one head only; default is every head of the " "arm",
    )
    ap.add_argument(
        "--exclude-labels",
        nargs="*",
        default=[],
        help="remove these labels from the head's output space "
        "AND drop every row whose gold contains one. For "
        "7.10: --exclude-labels grievance.",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="write to data/encoder/{arm}_{tag}/ so a restricted "
        "build never overwrites the full one",
    )
    args = ap.parse_args(argv)

    spec = ARMS[args.arm]
    print(
        f"\n{'=' * 78}\nENCODER DATA PREP  arm={args.arm}  "
        f"lang={spec['lang']}\n{'=' * 78}"
    )

    if not EXCL.exists():
        print(
            f"\nFAIL: {EXCL} missing. Run `python -m scripts.check_leakage` "
            f"(full scan, no --smoke and no --only) first. Training without "
            f"the filter would inflate Arm A, which is the ceiling the "
            f"whole adaptability comparison is read against."
        )
        raise SystemExit(1)
    exc = json.loads(EXCL.read_text(encoding="utf-8"))
    excluded_ids = set(exc["ids"].get(spec["lang"], []))
    print(
        f"\nleakage filter: {len(excluded_ids)} {spec['lang']} ids at "
        f">= {exc['threshold']} cosine against ANY eval subset"
    )

    pool = load_pool(args.arm)
    if pool.empty:
        print("no source rows; nothing to build.")
        raise SystemExit(1)
    print(f"pool: {len(pool)} rows from {', '.join(spec['source'])}")

    out_dir = OUT / (f"{args.arm}_{args.tag}" if args.tag else args.arm)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    heads_to_build = [args.head] if args.head else spec["heads"]
    for head in heads_to_build:
        gold_col, kind, gate_required = HEADS[head]
        labels = taxonomy_labels(head)

        excl = [l for l in args.exclude_labels if l in labels]
        if args.exclude_labels and not excl:
            print(
                f"  --exclude-labels {args.exclude_labels} matches nothing "
                f"in {head}'s labels {labels}; nothing removed"
            )
        labels = [l for l in labels if l not in excl]
        print(
            f"\n{'-' * 78}\nhead: {head}  ({kind}, {len(labels)} labels)"
            + ("  [gold-hateful only]" if gate_required else "")
        )

        keep, dropped_none, dropped_gate, dropped_excl = [], 0, 0, 0
        stripped_excl = 0
        for r in pool.itertuples(index=False):
            v = clean(getattr(r, gold_col, None))
            if v is None:
                dropped_none += 1
                continue
            if gate_required and not bool(getattr(r, "gate", False)):
                dropped_gate += 1
                continue

            if excl and kind != "binary":
                vv = [x for x in (v if isinstance(v, list) else [v]) if x not in excl]
                if not vv:
                    dropped_excl += 1
                    continue
                if len(vv) != len(v if isinstance(v, list) else [v]):
                    stripped_excl += 1
                v = vv
            keep.append(
                (
                    str(r.id),
                    str(r.text),
                    str(getattr(r, "lang", "")),
                    str(getattr(r, "source", "")),
                    getattr(r, "_split", args.arm),
                    v,
                )
            )
        print(
            f"  mask         {len(keep)} rows with non-None `{gold_col}` "
            f"({dropped_none} masked out)"
            + (
                f", {dropped_gate} gold-not-hateful dropped to match the "
                f"scored population"
                if gate_required
                else ""
            )
        )
        if excl:
            print(
                f"  EXCLUDED     {excl} -> {len(labels)}-label output "
                f"space, {dropped_excl} rows dropped, {stripped_excl} rows "
                f"kept with the label stripped"
            )
            print(
                f"               ^ this count is the '{','.join(excl)} "
                f"labelled examples required' cell of the 7.10 cost table"
            )
        if not keep:
            print(f"  UNBUILDABLE: no row in this pool annotates `{head}`.")
            summary[head] = {
                "buildable": False,
                "reason": f"no non-None {gold_col} in pool",
            }
            continue

        before = len(keep)
        keep = [k for k in keep if k[0] not in excluded_ids]
        n_excluded = before - len(keep)
        print(f"  leakage      {n_excluded} row(s) excluded, {len(keep)} " f"remain")

        sup = support_of([k[5] for k in keep], kind, labels)
        thin = [l for l, n in sup.items() if n < MIN_SUPPORT]
        print(f"  support      " + "  ".join(f"{l}={n}" for l, n in sup.items()))
        if thin:

            print(f"  THIN         below MIN_SUPPORT={MIN_SUPPORT}: {thin}")

        df = pd.DataFrame(
            keep, columns=["id", "text", "lang", "source", "split", gold_col]
        )
        n_val = max(1, int(round(len(df) * args.val_fraction)))

        shuffled = df.sample(frac=1.0, random_state=args.seed)
        val = shuffled.head(n_val).reset_index(drop=True)
        train = shuffled.tail(len(df) - n_val).reset_index(drop=True)
        assert (
            set(train["id"]) & set(val["id"]) == set()
        ), "train and val overlap - the shuffle-then-slice is wrong"
        print(
            f"  split        train {len(train)}  val {len(val)}  "
            f"(seed {args.seed}, {args.val_fraction:.0%} held out FROM "
            f"TRAIN, not from dev)"
        )

        meta = {
            "head": head,
            "dimension": head,
            "kind": kind,
            "labels": labels,
            "arm": args.arm,
            "lang": spec["lang"],
            "source_splits": spec["source"],
            "n_pool": len(pool),
            "n_masked_out": dropped_none,
            "gate_required": gate_required,
            "n_dropped_not_hateful": dropped_gate,
            "leakage_threshold": exc["threshold"],
            "leakage_excluded": n_excluded,
            "n_train": len(train),
            "n_val": len(val),
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "support_train_plus_val": sup,
            "labels_below_min_support": thin,
            "min_support": MIN_SUPPORT,
            "source_mix": dict(Counter(df["source"])),
            "labels_excluded_from_output": excl,
            "n_dropped_excluded_label": dropped_excl,
            "n_stripped_excluded_label": stripped_excl,
            "tag": args.tag,
        }
        summary[head] = {
            "buildable": True,
            **{k: meta[k] for k in ("n_train", "n_val", "leakage_excluded")},
            "thin": thin,
        }

        if args.dry_run:
            print("  (dry run - nothing written)")
            continue
        train.to_parquet(out_dir / f"{head}_train.parquet", index=False)
        val.to_parquet(out_dir / f"{head}_val.parquet", index=False)
        (out_dir / f"{head}_meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"  wrote        {out_dir / f'{head}_train.parquet'}, "
            f"{head}_val.parquet, {head}_meta.json"
        )

    print(f"\n{'=' * 78}\nSUMMARY  arm={args.arm}")
    for head, s in summary.items():
        if not s.get("buildable"):
            print(f"  {head:16} UNBUILDABLE - {s['reason']}")
        else:
            print(
                f"  {head:16} train {s['n_train']:>6}  val {s['n_val']:>5}  "
                f"leakage-excluded {s['leakage_excluded']:>3}"
                + (f"  THIN {s['thin']}" if s["thin"] else "")
            )
    if args.arm == "kb":
        print(
            "\n  Arm B has no `legal` head by construction: KB records "
            "carry meta.stgb, a paragraph number, not a taxonomy label."
        )
    print(
        f"\n  Next: python -m scripts.check_encoder_reachability " f"--arm {args.arm}"
    )
    print(
        "  Hop 1 should turn from NOT BUILT to ok; hops 2-8 stay NOT BUILT "
        f"until 7.6 trains a checkpoint.\n{'=' * 78}\n"
    )


if __name__ == "__main__":
    main()
