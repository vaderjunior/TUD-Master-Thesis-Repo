import json
from collections import Counter, defaultdict
from pathlib import Path
from src.hsrag.paths import project_path

import pandas as pd
import yaml

TAXONOMY = project_path("config/taxonomy.yaml")
GUIDELINES = project_path("config/guidelines.yaml")
GUIDELINES_DE = project_path("config/guidelines_de.yaml")
PROCESSED = project_path("data/processed")
OUT = project_path("kb/records.jsonl")

EXAMPLES_PER_LABEL = 10

LEGAL_EXAMPLES_PER_LABEL = 15
LEGAL_NEGATIVES = 20


def gen_definitions() -> list[dict]:

    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    records = []

    for dim_name, dim in tax["dimensions"].items():
        dtype = dim["type"]

        if dtype == "binary":

            records.append(
                {
                    "id": f"def-{dim_name}",
                    "kind": "definition",
                    "dimension": dim_name,
                    "label": None,
                    "lang": "en",
                    "text": " ".join(dim["definition"].split()),
                    "source": "taxonomy_v1",
                    "meta": {},
                }
            )

        elif dtype in ("multilabel", "ordinal"):
            labels = dim["labels"]

            if isinstance(labels, list):
                labels = {l: {} for l in labels}
            for label_name, label in labels.items():
                definition = (label or {}).get("definition", "")
                if not definition:
                    continue
                records.append(
                    {
                        "id": f"def-{dim_name}-{label_name}",
                        "kind": "definition",
                        "dimension": dim_name,
                        "label": label_name,
                        "lang": "en",
                        "text": " ".join(definition.split()),
                        "source": "taxonomy_v1",
                        "meta": {},
                    }
                )

    return records


def gen_guidelines() -> list[dict]:

    records = []
    for path, source, lang in (
        (GUIDELINES, "guidelines_v1", "en"),
        (GUIDELINES_DE, "guidelines_de_botox", "de"),
    ):
        if not path.exists():
            continue
        g = yaml.safe_load(path.read_text(encoding="utf-8"))
        for rule in g["guidelines"]:
            records.append(
                {
                    "id": f"guide-{rule['id']}",
                    "kind": "guideline",
                    "dimension": rule.get("dimension"),
                    "label": None,
                    "lang": rule.get("lang", lang),
                    "text": " ".join(rule["text"].split()),
                    "source": source,
                    "meta": {},
                }
            )
    return records


def clean(val):

    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def gen_examples(processed_dir: Path | None = None) -> list[dict]:

    records = []
    seen_ids = set()
    coverage = defaultdict(int)

    for lang in ["en", "de"]:
        path = (processed_dir or PROCESSED) / f"{lang}_train.parquet"
        df = pd.read_parquet(path)

        def richness(row):
            r = 0
            if row.target_groups is not None:
                r += 1
            if row.hate_types is not None:
                r += 1
            if row.severity is not None:
                r += 1
            return r

        df = df.assign(rich=[richness(r) for r in df.itertuples(index=False)])

        def add_row(row, label_key):

            rec_id = f"ex-{row.id}"
            coverage[(lang, label_key)] += 1
            if rec_id in seen_ids:
                return
            seen_ids.add(rec_id)
            records.append(
                {
                    "id": rec_id,
                    "kind": "example",
                    "dimension": None,
                    "label": None,
                    "lang": lang,
                    "text": row.text,
                    "source": f"{row.source}-train",
                    "meta": {
                        "gate": bool(row.gate) if row.gate is not None else None,
                        "target_groups": clean(row.target_groups),
                        "hate_types": clean(row.hate_types),
                        "severity": clean(row.severity),
                        "illustrative_only": False,
                    },
                }
            )

        for col, is_list in [
            ("target_groups", True),
            ("hate_types", True),
            ("severity", False),
        ]:

            labels = set()
            for row in df.itertuples(index=False):
                val = getattr(row, col)
                if val is None:
                    continue
                if is_list:

                    if isinstance(val, (list, tuple)) or hasattr(val, "__iter__"):
                        labels.update(v for v in val if isinstance(v, str))
                else:
                    if isinstance(val, str):
                        labels.add(val)

            for label in sorted(labels):

                def has_label(row):
                    val = getattr(row, col)
                    if val is None:
                        return False
                    if is_list:
                        if not (
                            isinstance(val, (list, tuple)) or hasattr(val, "__iter__")
                        ):
                            return False
                        return label in val
                    return val == label

                candidates = [r for r in df.itertuples(index=False) if has_label(r)]
                candidates.sort(key=lambda r: r.rich, reverse=True)
                for row in candidates[:EXAMPLES_PER_LABEL]:
                    add_row(row, f"{col}:{label}")

        negatives = [r for r in df.itertuples(index=False) if r.gate is False]

        negatives.sort(key=lambda r: r.rich, reverse=True)
        for row in negatives[: EXAMPLES_PER_LABEL * 2]:
            add_row(row, "gate:false")

    print("\n  example coverage (lang, label -> count):")
    for (lang, label), n in sorted(coverage.items()):
        print(f"    {lang}  {label:30} {n}")

    return records


def gen_legal_illustrations(
    existing: list[dict], processed_dir: Path | None = None
) -> list[dict]:

    path = (processed_dir or PROCESSED) / "de_train.parquet"
    df = pd.read_parquet(path)
    df = df[df["source"] == "detox"]

    by_id = {r["id"]: r for r in existing}

    records = []
    merged = 0
    for row in df.itertuples(index=False):
        raw = json.loads(row.raw)
        paragraphs = raw.get("legal_paragraphs", {})

        flagged = [p.replace("p_", "") for p, v in paragraphs.items() if v > 0.5]
        if not flagged:
            continue
        stgb = ",".join(flagged)

        base_id = f"ex-{row.id}"
        if base_id in by_id:
            by_id[base_id]["meta"]["stgb"] = stgb
            merged += 1
            continue

        records.append(
            {
                "id": f"ex-detox-legal-{row.id.split('-', 1)[1]}",
                "kind": "example",
                "dimension": None,
                "label": None,
                "lang": "de",
                "text": row.text,
                "source": "detox-train-legal",
                "meta": {
                    "gate": bool(row.gate),
                    "target_groups": clean(row.target_groups),
                    "hate_types": None,
                    "severity": None,
                    "illustrative_only": True,
                    "stgb": stgb,
                },
            }
        )

    print(
        f"\n  legal: {len(records)} standalone illustrations, "
        f"{merged} merged into existing balanced examples"
    )
    return records


def gen_botox_examples(
    existing: list[dict], processed_dir: Path | None = None
) -> list[dict]:

    path = (processed_dir or PROCESSED) / "de_legal_train.parquet"
    if not path.exists():
        print("\n  botox: de_legal_train.parquet not found, skipping")
        return []

    df = pd.read_parquet(path)
    seen = {r["id"] for r in existing}
    records = []
    coverage = defaultdict(int)

    def add(row, key):
        rec_id = f"ex-{row.id}"
        coverage[key] += 1
        if rec_id in seen:
            return
        seen.add(rec_id)
        records.append(
            {
                "id": rec_id,
                "kind": "example",
                "dimension": None,
                "label": None,
                "lang": "de",
                "text": row.text,
                "source": "botox-train",
                "meta": {
                    "gate": None,
                    "target_groups": None,
                    "hate_types": None,
                    "severity": None,
                    "legal": clean(row.legal),
                    "illustrative_only": False,
                },
            }
        )

    rows = list(df.itertuples(index=False))

    labels = sorted({l for r in rows for l in (clean(r.legal) or [])})
    by_freq = sorted(
        labels, key=lambda l: sum(1 for r in rows if l in (clean(r.legal) or []))
    )

    def covered(label):
        return sum(1 for r in records if label in (r["meta"]["legal"] or []))

    for label in by_freq:

        cands = [r for r in rows if label in (clean(r.legal) or [])]
        cands.sort(key=lambda r: len(clean(r.legal) or []), reverse=True)
        for row in cands:
            if covered(label) >= LEGAL_EXAMPLES_PER_LABEL:
                break
            add(row, f"legal:{label}")

    negatives = [r for r in rows if not (clean(r.legal) or [])]
    for row in negatives[:LEGAL_NEGATIVES]:
        add(row, "legal:none")

    print("\n  botox example coverage:")
    for key, n in sorted(coverage.items()):
        print(f"    {key:30} {n}")
    print(f"  botox: {len(records)} records added")
    return records


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Prepare prototype knowledge records from guidance and TRAIN files."
    )
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument(
        "--guidance-only",
        action="store_true",
        help="write definitions/guidelines without dataset examples",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.output.exists() and not args.force:
        ap.error(
            "Knowledge records already exist. Choose another --output or use --force."
        )
    all_records = gen_definitions() + gen_guidelines()
    if not args.guidance_only:
        missing = [
            str(args.processed_dir / f"{lang}_{split}.parquet")
            for lang in ("en", "de")
            for split in ("train", "dev", "test")
            if not (args.processed_dir / f"{lang}_{split}.parquet").is_file()
        ]
        if missing:
            ap.error("Prepare general partitions first. Missing: " + ", ".join(missing))
        examples = gen_examples(args.processed_dir)
        all_records += examples + gen_legal_illustrations(examples, args.processed_dir)
        all_records += gen_botox_examples(all_records, args.processed_dir)
        held_out = set()
        for split in (
            "en_dev",
            "en_test",
            "de_dev",
            "de_test",
            "de_legal_dev",
            "de_legal_test",
        ):
            p = args.processed_dir / f"{split}.parquet"
            if p.exists():
                held_out |= {
                    " ".join(str(t).lower().split()) for t in pd.read_parquet(p)["text"]
                }
        all_records = [
            r
            for r in all_records
            if r["kind"] != "example"
            or " ".join(str(r["text"]).lower().split()) not in held_out
        ]
    duplicates = [
        rid for rid, count in Counter(r["id"] for r in all_records).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate knowledge IDs: {duplicates[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(
        "w" if args.force else "x", encoding="utf-8", newline="\n"
    ) as output:
        for record in all_records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_records)} records to {args.output}")
    print("Record kinds:", dict(Counter(r["kind"] for r in all_records)))


if __name__ == "__main__":
    main()
