"""DeTox loader: hate and target annotations (Demus et al., 2022)."""

import csv
import sqlite3
from collections import Counter
from pathlib import Path
from src.hsrag.paths import project_path

import pandas as pd

from src.hsrag.data.record import Record

DB_PATH = project_path("data/raw/detox/DeTox-Dataset_complete.sqlite3")
MAPPING_PATH = project_path("data/mappings/detox_targets.csv")

GATE_THRESHOLD = 0.5
TARGET_THRESHOLD = 0.5

P_COLS = [
    "p_86", "p_86a", "p_111", "p_126", "p_130", "p_131", "p_140",
    "p_166", "p_185", "p_186", "p_187", "p_189", "p_240", "p_241",
]


def load_target_mapping(path: Path | None = None) -> dict[str, str]:
    """Read source target labels and explicit exclusions."""
    mapping = {}
    with open(path or MAPPING_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["source_column"]] = row["unified_label"]
    return mapping


def load_detox(path: Path | None = None, mapping_path: Path | None = None) -> tuple[list[Record], Counter]:
    mapping = load_target_mapping(mapping_path)
    discrim_cols = list(mapping.keys())

    path = Path(path or DB_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"DeTox source not found: {path}")
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    gs = pd.read_sql_query("SELECT * FROM Goldstandard;", con)
    con.close()

    db_discrim = [c for c in gs.columns if c.startswith("discrim_")]
    missing = set(db_discrim) - set(discrim_cols)
    if missing:
        raise KeyError(
            f"discrim_ columns in DB but missing from {MAPPING_PATH}: {missing}"
        )

    records = []
    stats = Counter()

    for row in gs.itertuples(index=False):
        hate_score = getattr(row, "hate_speech")

        if pd.isna(hate_score):
            stats["dropped_nan_score"] += 1
            continue

        gate = bool(hate_score > GATE_THRESHOLD)

        if gate:
            targets = set()
            for col in discrim_cols:
                unified = mapping[col]
                if unified == "EXCLUDE":
                    continue
                if getattr(row, col) > TARGET_THRESHOLD:
                    targets.add(unified)
            target_groups = sorted(targets)
        else:
            target_groups = None

        stats[f"gate_{gate}"] += 1

        records.append(
            Record(
                id=f"detox-{row.c_id}",
                text=row.c_text,
                lang="de",
                source="detox",
                gate=gate,
                target_groups=target_groups,
                hate_types=None,
                severity=None,
                raw={
                    "hate_speech_score": round(float(hate_score), 3),
                    "criminal_rel": float(row.criminal_rel),
                    "legal_paragraphs": {
                        p: float(getattr(row, p)) for p in P_COLS
                    },
                    "threat": float(row.threat),
                    "extrem": float(row.extrem),
                    "toxi": float(row.toxi),
                    "n_annotators": int(row.nb_annotators),
                },
            )
        )

    return records, stats
