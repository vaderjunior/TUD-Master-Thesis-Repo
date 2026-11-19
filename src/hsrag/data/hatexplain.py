"""HateXplain loader: hate votes and targets (Mathew et al., 2021)."""

import csv
import json
from collections import Counter
from pathlib import Path
from src.hsrag.paths import project_path

from src.hsrag.data.record import Record

DATA_PATH = project_path("data/raw/hatexplain/dataset.json")
SPLITS_PATH = project_path("data/raw/hatexplain/post_id_divisions.json")
MAPPING_PATH = project_path("data/mappings/hatexplain_targets.csv")


def load_target_mapping(path: Path | None = None) -> dict[str, str]:
    """Read source target labels and explicit exclusions."""
    mapping = {}
    with open(path or MAPPING_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["source_label"]] = row["unified_label"]
    return mapping


def map_targets(raw_targets: set[str], mapping: dict[str, str]) -> list[str]:
    """Map targets, rejecting labels absent from the mapping."""
    out = set()
    for t in raw_targets:
        if t not in mapping:
            raise KeyError(
                f"Target label '{t}' is in the data but missing from "
                f"{MAPPING_PATH}. Add it (or mark it EXCLUDE) and re-run."
            )
        unified = mapping[t]
        if unified != "EXCLUDE":
            out.add(unified)
    return sorted(out)


def load_official_splits(path: Path | None = None) -> dict[str, str]:
    """{post_id: 'train'|'val'|'test'} from the authors' division. We don't use this for our splits, but we keep it in `raw`."""
    divisions = json.loads((path or SPLITS_PATH).read_text(encoding="utf-8"))
    return {
        post_id: split
        for split, post_ids in divisions.items()
        for post_id in post_ids
    }


def load_hatexplain(path: Path | None = None, splits_path: Path | None = None, mapping_path: Path | None = None) -> tuple[list[Record], Counter]:
    """Returns (records, stats)."""
    data = json.loads((path or DATA_PATH).read_text(encoding="utf-8"))
    mapping = load_target_mapping(mapping_path)
    official_splits = load_official_splits(splits_path)

    records = []
    stats = Counter()

    for post_id, post in data.items():
        text = " ".join(post["post_tokens"])
        votes = [a["label"] for a in post["annotators"]]

        vote_counts = Counter(votes)
        majority_label, majority_count = vote_counts.most_common(1)[0]

        if majority_count == 1:
            stats["dropped_tie"] += 1
            continue

        gate = majority_label == "hatespeech"

        if gate:
            all_targets = set()
            for a in post["annotators"]:
                all_targets.update(a["target"])
            target_groups = map_targets(all_targets, mapping)
        else:
            target_groups = None

        records.append(
            Record(
                id=f"hatexplain-{post_id}",
                text=text,
                lang="en",
                source="hatexplain",
                gate=gate,
                target_groups=target_groups,
                hate_types=None,
                severity=None,
                raw={
                    "votes": votes,
                    "official_split": official_splits.get(post_id),
                    "rationales": post["rationales"],
                },
            )
        )

        stats[f"label_{majority_label}"] += 1

    return records, stats
