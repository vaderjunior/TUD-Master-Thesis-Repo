from analysis.common import micro_f1_files as c
from pathlib import Path
from datetime import datetime, timezone
import json
import zipfile

BASE = c.ROOT / "experiments/astra/revision5_reporting_20261113_v1"
READER = c.ROOT / "Thesis_Log/reader_package_20261113_jaccard"
PERFORMANCE = c.ROOT / "experiments/astra/sq2_combined_performance_20261104_v1"
QUALITY = c.ROOT / "experiments/astra/sq2_combined_quality_mechanisms_20261104_v1"
BROAD = c.ROOT / "experiments/astra/sq2_broad_quality_review_20261104_v1"


def save(name, value):
    path = (BASE / name).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Write outside new review evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write(name, text):
    path = (BASE / name).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Write outside new review evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        f.write(text)


def freeze():
    c.require(not BASE.exists(), "Preserve previous reporting results")
    BASE.mkdir(parents=True)
    c.bind(READER / "evidence/jaccard/display_values.json")
    save(
        "preservation_before.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=c.BINDINGS
        ),
    )
    save(
        "scope.json",
        dict(
            task="Recalculate descriptive label, overlap and severity scores.",
            primary="Existing macro scores remain primary; no new selection or hypothesis tests.",
            no_qwen=True,
            no_training=True,
        ),
    )


if __name__ == "__main__":
    freeze()
