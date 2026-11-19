import argparse
import json
from collections import Counter
from pathlib import Path
from src.hsrag.paths import project_path

import pandas as pd

PROCESSED = project_path("data/processed")
OUT = PROCESSED / "en_hatecheck_eval.parquet"

IDENT_MAP = {
    "women": "gender",
    "trans people": "gender",
    "gay people": "sexual_orientation",
    "black people": "race",
    "disabled people": "disability",
    "Muslims": "religion",
    "immigrants": "national_origin",
}

GUIDELINE_COVERED = {
    "counter_quote_nh",
    "counter_ref_nh",
    "slur_reclaimed_nh",
    "profanity_nh",
    "negate_neg_nh",
    "target_indiv_nh",
}
DEFINITION_COVERED = {"target_group_nh", "target_obj_nh"}
UNCOVERED = {"ident_neutral_nh", "ident_pos_nh", "slur_homonym_nh"}


def block(functionality: str, gold: str) -> str:
    if gold == "hateful":
        return "hateful"
    if functionality in GUIDELINE_COVERED:
        return "guideline_covered"
    if functionality in DEFINITION_COVERED:
        return "definition_covered"
    if functionality in UNCOVERED:
        return "uncovered"
    raise AssertionError(f"unmapped non-hateful functionality: {functionality}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--revision")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.output.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"{args.output} exists. --force to overwrite.")

    source = args.input or project_path("data/raw/hatecheck/hatecheck.csv")
    if source.is_file():
        hc = (
            pd.read_parquet(source)
            if source.suffix == ".parquet"
            else pd.read_csv(source)
        )
    elif args.download:
        from datasets import load_dataset

        options = {"revision": args.revision} if args.revision else {}
        hc = load_dataset("Paul/hatecheck", **options)["test"].to_pandas()
    else:
        ap.error(
            f"Source not found: {source}. Supply --input or explicitly use --download."
        )
    rows = []
    for r in hc.itertuples(index=False):
        gate = r.label_gold == "hateful"
        ident = r.target_ident
        if gate:
            assert ident in IDENT_MAP, f"hateful case {r.case_id} ident={ident!r}"
            tg = [IDENT_MAP[ident]]
        else:
            tg = None
        rows.append(
            {
                "id": f"hatecheck-{r.case_id}",
                "text": str(r.test_case).strip(),
                "lang": "en",
                "source": "hatecheck",
                "gate": gate,
                "target_groups": tg,
                "hate_types": None,
                "severity": None,
                "raw": json.dumps(
                    {
                        "functionality": r.functionality,
                        "block": block(r.functionality, r.label_gold),
                        "templ_id": int(r.templ_id),
                        "case_id": int(r.case_id),
                        "ref_case_id": (
                            None if pd.isna(r.ref_case_id) else int(r.ref_case_id)
                        ),
                        "target_ident": ident,
                        "direction": r.direction,
                        "test_case_verbatim": str(r.test_case),
                    }
                ),
            }
        )

    df = pd.DataFrame(rows)
    assert len(df) == 3728, len(df)
    assert df["gate"].sum() == 2563, int(df["gate"].sum())

    print(f"\nen_hatecheck_eval: {len(df)} items")
    print("  gate:", dict(Counter(df["gate"])))
    blocks = Counter(json.loads(x)["block"] for x in df["raw"])
    print("  blocks:", dict(blocks))
    assert sum(v for k, v in blocks.items() if k != "hateful") == 1165
    tg = Counter(t[0] for t in df["target_groups"] if t is not None)
    print("  target_groups (hateful only):", dict(tg))
    print("  distinct templ_id:", len({json.loads(x)["templ_id"] for x in df["raw"]}))
    print("  target_groups None count:", int(df["target_groups"].isna().sum()))

    if args.dry_run:
        print("\n  DRY RUN, nothing written.\n")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\n  wrote {OUT}\n")


if __name__ == "__main__":
    main()
