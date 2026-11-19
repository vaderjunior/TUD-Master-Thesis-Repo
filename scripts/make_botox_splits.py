import argparse
from collections import Counter
from pathlib import Path
from src.hsrag.paths import project_path

import numpy as np
import pandas as pd

from src.hsrag.data.botox import load

PROCESSED = project_path("data/processed")
SEED = 42
RATIOS = (0.70, 0.15, 0.15)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prepare separate BoTox legal partitions.")
    ap.add_argument("--raw-dir", type=Path, default=project_path("data/raw"))
    ap.add_argument("--output-dir", type=Path, default=PROCESSED)
    ap.add_argument(
        "--allow-missing-heldout",
        action="store_true",
        help="prepare a standalone dataset without all general DEV/TEST files",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    paths = {
        s: args.output_dir / f"de_legal_{s}.parquet" for s in ("train", "dev", "test")
    }
    if any(p.exists() for p in paths.values()) and not args.force:
        ap.error(
            "Legal partitions already exist. Choose another --output-dir or use --force."
        )
    missing = [
        str(args.output_dir / f"{s}.parquet")
        for s in ("en_dev", "en_test", "de_dev", "de_test")
        if not (args.output_dir / f"{s}.parquet").exists()
    ]
    if missing and not args.allow_missing_heldout:
        ap.error(
            "Prepare general partitions first, or explicitly use --allow-missing-heldout. Missing: "
            + ", ".join(missing)
        )
    records = load(args.raw_dir / "botox/botox.csv", processed_dir=args.output_dir)
    df = pd.DataFrame(
        [
            {
                "id": r.id,
                "text": r.text,
                "lang": r.lang,
                "source": r.source,
                "gate": r.gate,
                "target_groups": r.target_groups,
                "hate_types": r.hate_types,
                "severity": r.severity,
                "legal": r.legal,
                "raw": r.raw,
            }
            for r in records
        ]
    )
    df["_norm"] = df["text"].map(lambda t: " ".join(str(t).lower().split()))
    df = df.drop_duplicates("_norm", keep="first").drop(columns="_norm")
    df["_stratum"] = df["legal"].map(bool)
    parts = {"train": [], "dev": [], "test": []}
    for _, group in df.groupby("_stratum"):
        group = group.sample(frac=1.0, random_state=SEED)
        n = len(group)
        n_train, n_dev = int(n * RATIOS[0]), int(n * RATIOS[1])
        parts["train"].append(group.iloc[:n_train])
        parts["dev"].append(group.iloc[n_train : n_train + n_dev])
        parts["test"].append(group.iloc[n_train + n_dev :])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, chunks in parts.items():
        frame = (
            pd.concat(chunks)
            .drop(columns="_stratum")
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )
        frame.to_parquet(paths[split], index=False)
        print(f"Wrote {len(frame)} legal records to {paths[split]}")


if __name__ == "__main__":
    main()
