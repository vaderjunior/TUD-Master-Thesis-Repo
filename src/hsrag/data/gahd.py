"""GAHD loader (Goldzycher et al., 2024). Dataset license: CC BY 4.0."""

from collections import Counter
from pathlib import Path
from src.hsrag.paths import project_path

import pandas as pd

from src.hsrag.data.record import Record

CSV_PATH = project_path("data/raw/gahd/gahd.csv")


def load_gahd(path: Path | None = None) -> tuple[list[Record], Counter]:
    df = pd.read_csv(path or CSV_PATH, encoding="utf-8")

    records = []
    stats = Counter()

    for row in df.itertuples(index=False):
        gate = bool(row.label == 1)
        stats[f"gate_{gate}"] += 1
        stats[f"round_{row.round}"] += 1

        records.append(
            Record(
                id=f"gahd-{row.gahd_id}",
                text=row.text,
                lang="de",
                source="gahd",
                gate=gate,
                target_groups=None,
                hate_types=None,
                severity=None,
                raw={
                    "round": row.round,
                    "official_split": row.split,
                    "contrastive_id": (
                        None if pd.isna(row.contrastive_gahd_id)
                        else str(row.contrastive_gahd_id)
                    ),
                },
            )
        )

    return records, stats
