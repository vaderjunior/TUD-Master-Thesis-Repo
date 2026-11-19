"""Implicit Hate Corpus loader (ElSherief et al., 2021)."""

import csv
from collections import Counter
from pathlib import Path
from src.hsrag.paths import project_path

import pandas as pd

from src.hsrag.data.record import Record

RAW = project_path("data/raw/implicit_hate")
STG1 = RAW / "implicit_hate_v1_stg1_posts.tsv"
STG2 = RAW / "implicit_hate_v1_stg2_posts.tsv"
MAPPING_PATH = project_path("data/mappings/implicit_hate_types.csv")


def load_type_mapping(path: Path | None = None) -> dict[str, str]:
    mapping = {}
    with open(path or MAPPING_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["source_label"]] = row["unified_label"]
    return mapping


def map_types(raw_types: list[str], mapping: dict[str, str]) -> list[str]:
    """Map recorded hate types, rejecting unknown source labels."""
    out = []
    for t in raw_types:
        if t not in mapping:
            raise KeyError(
                f"Type '{t}' is in the data but missing from {MAPPING_PATH}. "
                f"Add it (or mark EXCLUDE) and re-run."
            )
        unified = mapping[t]
        if unified != "EXCLUDE" and unified not in out:
            out.append(unified)
    return out


def load_implicit_hate(stg1_path: Path | None = None, stg2_path: Path | None = None, mapping_path: Path | None = None) -> tuple[list[Record], Counter]:
    mapping = load_type_mapping(mapping_path)

    stg1 = pd.read_csv(stg1_path or STG1, sep="\t", encoding="utf-8")
    stg2 = pd.read_csv(stg2_path or STG2, sep="\t", encoding="utf-8")

    stg2_types: dict[str, list[str]] = {}
    for row in stg2.itertuples(index=False):
        types = [row.implicit_class]
        extra = row.extra_implicit_class
        if isinstance(extra, str) and extra.strip():
            types.append(extra)
        stg2_types[row.post] = types

    records = []
    stats = Counter()
    matched_posts = set()

    for _, row in stg1.iterrows():
        cls = row["class"]
        post = row["post"]

        if cls == "not_hate":
            gate, hate_types = False, None
        elif cls == "explicit_hate":
            gate, hate_types = True, ["explicit"]
        elif cls == "implicit_hate":
            gate = True
            if post in stg2_types:
                matched_posts.add(post)
                mapped = map_types(stg2_types[post], mapping)
                hate_types = mapped or None
            else:
                hate_types = None
                stats["implicit_no_type"] += 1
        else:
            raise ValueError(f"Unexpected class '{cls}'")

        stats[f"gate_{gate}"] += 1
        records.append(
            Record(
                id=f"implicit_hate-{len(records)}",
                text=post,
                lang="en",
                source="implicit_hate",
                gate=gate,
                target_groups=None,
                hate_types=hate_types,
                severity=None,
                raw={"stage1_class": cls},
            )
        )

    for post, types in stg2_types.items():
        if post in matched_posts:
            continue
        stats["stg2_unmatched"] += 1
        mapped = map_types(types, mapping)
        records.append(
            Record(
                id=f"implicit_hate-{len(records)}",
                text=post,
                lang="en",
                source="implicit_hate",
                gate=True,
                target_groups=None,
                hate_types=mapped or None,
                severity=None,
                raw={"stage1_class": "implicit_hate (stg2-only)"},
            )
        )

    return records, stats
