"""Measuring Hate Speech loader (Sachdeva et al., 2022)."""

from collections import Counter
from pathlib import Path

import pandas as pd

from src.hsrag.data.record import Record

GATE_THRESHOLD = 0.5

SEV_LOW_MAX = 1.00
SEV_MED_MAX = 1.63

TARGET_MAP = {
    "target_race": "race",
    "target_religion": "religion",
    "target_origin": "national_origin",
    "target_gender": "gender",
    "target_sexuality": "sexual_orientation",
    "target_disability": "disability",
}


def severity_band(score: float) -> str:
    if score <= SEV_LOW_MAX:
        return "low"
    if score <= SEV_MED_MAX:
        return "medium"
    return "high"


def load_mhs(path: Path | None = None, revision: str | None = None) -> tuple[list[Record], Counter]:
    if path is None:
        from datasets import load_dataset

        options = {"revision": revision} if revision else {}
        df = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train", **options).to_pandas()
    else:
        path = Path(path)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    agg = df.groupby("comment_id").agg(
        text=("text", "first"),
        score=("hate_speech_score", "mean"),
        n_annotators=("annotator_id", "count"),
        **{col: (col, "any") for col in TARGET_MAP},
    ).reset_index()

    records = []
    stats = Counter()

    for row in agg.itertuples(index=False):
        gate = row.score > GATE_THRESHOLD

        if gate:
            targets = sorted(
                {
                    unified
                    for col, unified in TARGET_MAP.items()
                    if getattr(row, col)
                }
            )
            severity = severity_band(row.score)
            stats[f"severity_{severity}"] += 1
        else:
            targets = None
            severity = None

        stats[f"gate_{gate}"] += 1

        records.append(
            Record(
                id=f"mhs-{row.comment_id}",
                text=row.text,
                lang="en",
                source="mhs",
                gate=bool(gate),
                target_groups=targets,
                hate_types=None,
                severity=severity,
                raw={
                    "score": round(float(row.score), 4),
                    "n_annotators": int(row.n_annotators),
                },
            )
        )

    return records, stats
