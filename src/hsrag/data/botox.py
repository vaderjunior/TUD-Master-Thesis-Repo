"""BoTox loader: grouped legal annotations (Kums et al., 2025)."""

import json
from pathlib import Path
from src.hsrag.paths import project_path
import ftfy
import pandas as pd

from src.hsrag.data.record import Record

RAW = project_path("data/raw/botox/botox.csv")
PROCESSED = project_path("data/processed")

CLASS_MAP = {
    "class_1": "propaganda",
    "class_2": "incitement_public_peace",
    "class_3": "insult_defamation",
}

HELD_OUT = ("en_dev", "en_test", "de_dev", "de_test")

MOJIBAKE = ("Ã", "ðŸ", "Â", "â€")


def _fix_encoding(s: str) -> str:
    """Repair mixed encoding damage with ftfy."""
    return ftfy.fix_text(s) if isinstance(s, str) else s


def _norm(s) -> str:
    return " ".join(str(s).lower().split())


def load(path: Path = RAW, drop_mojibake: bool = False, processed_dir: Path | None = None) -> list[Record]:
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    n_raw = len(df)
    df["text"] = df["text"].map(_fix_encoding)

    held = set()
    for split in HELD_OUT:
        p = (processed_dir or PROCESSED) / f"{split}.parquet"
        if p.exists():
            held |= {_norm(t) for t in pd.read_parquet(p)["text"]}
    mask = df["text"].map(lambda t: _norm(t) in held)
    n_leak = int(mask.sum())
    df = df[~mask]

    bad = df["text"].map(lambda t: any(m in str(t) for m in MOJIBAKE))
    n_bad = int(bad.sum())
    if drop_mojibake:
        df = df[~bad]

    records = []
    for row in df.itertuples(index=False):
        labels = sorted(name for col, name in CLASS_MAP.items()
                        if int(getattr(row, col)) == 1)
        records.append(Record(
            id=f"botox-{row.index}",
            text=str(row.text),
            lang="de",
            source="botox",

            gate=None,
            target_groups=None,
            hate_types=None,
            severity=None,

            legal=labels,
            raw=json.dumps({"output": str(getattr(row, "output", "")),
                            "botox_index": int(row.index)}),
        ))

    print(f"BoTox: {len(records)} records from {n_raw} rows")
    print(f"  dropped {n_leak} that appear in a held-out split "
          f"({', '.join(HELD_OUT)})")
    print(f"  {'dropped' if drop_mojibake else 'kept'} {n_bad} with "
          f"unrepairable encoding damage")
    counts = {name: sum(1 for r in records if name in r.legal)
              for name in CLASS_MAP.values()}
    counts["(none / class 0)"] = sum(1 for r in records if not r.legal)
    print(f"  class distribution: {counts}")
    print(f"  multi-class items: "
          f"{sum(1 for r in records if len(r.legal) > 1)}")
    return records

if __name__ == "__main__":
    load()
