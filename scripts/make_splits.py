import json
from collections import Counter
from pathlib import Path
from src.hsrag.paths import project_path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.hsrag.data.hatexplain import load_hatexplain
from src.hsrag.data.mhs import load_mhs
from src.hsrag.data.implicit_hate import load_implicit_hate
from src.hsrag.data.gahd import load_gahd
from src.hsrag.data.detox import load_detox

SEED = 42
OUT_DIR = project_path("data/processed")
FIG_DIR = project_path("experiments/results/figures")

LOADERS = [
    load_hatexplain,
    load_mhs,
    load_implicit_hate,
    load_gahd,
    load_detox,
]


def record_to_dict(r) -> dict:

    return {
        "id": r.id,
        "text": r.text,
        "lang": r.lang,
        "source": r.source,
        "gate": r.gate,
        "target_groups": r.target_groups,
        "hate_types": r.hate_types,
        "severity": r.severity,
        "raw": json.dumps(r.raw, ensure_ascii=False),
    }


def load_all(loaders=None) -> pd.DataFrame:
    rows = []
    for loader in (loaders if loaders is not None else LOADERS):
        records, _ = loader()
        rows.extend(record_to_dict(r) for r in records)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} records total")

    import re

    df["_norm"] = df["text"].map(lambda t: re.sub(r"\s+", " ", str(t).strip().lower()))
    before = len(df)
    df = df.drop_duplicates(subset="_norm", keep="first").drop(columns="_norm")
    dropped = before - len(df)
    print(f"Dropped {dropped} duplicate texts (kept first occurrence)")

    print(f"  by language:\n{df['lang'].value_counts().to_string()}")
    return df


def split_one_language(df_lang: pd.DataFrame) -> dict[str, pd.DataFrame]:

    temp, test = train_test_split(
        df_lang, test_size=0.15, random_state=SEED, stratify=df_lang["gate"]
    )

    train, dev = train_test_split(
        temp, test_size=0.15 / 0.85, random_state=SEED, stratify=temp["gate"]
    )
    return {"train": train, "dev": dev, "test": test}


def main(argv=None):
    import argparse
    import hashlib
    from functools import partial

    ap = argparse.ArgumentParser(
        description="Prepare seeded corpus partitions from local source files."
    )
    ap.add_argument(
        "--sources",
        nargs="+",
        choices=("hatexplain", "mhs", "implicit_hate", "gahd", "detox"),
        default=["hatexplain", "mhs", "implicit_hate", "gahd", "detox"],
    )
    ap.add_argument("--raw-dir", type=Path, default=project_path("data/raw"))
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--mhs-file", type=Path)
    ap.add_argument(
        "--download-mhs",
        action="store_true",
        help="allow MHS acquisition when no local export is supplied",
    )
    ap.add_argument(
        "--revision",
        help="Hugging Face revision for an explicitly requested MHS download",
    )
    ap.add_argument("--language", choices=("en", "de"))
    ap.add_argument(
        "--force", action="store_true", help="replace existing prepared partitions"
    )
    args = ap.parse_args(argv)
    raw = args.raw_dir
    mhs = args.mhs_file or raw / "mhs/mhs.parquet"
    if not mhs.exists() and not args.mhs_file and (raw / "mhs/mhs.csv").exists():
        mhs = raw / "mhs/mhs.csv"
    required = {
        "hatexplain": [
            raw / "hatexplain/dataset.json",
            raw / "hatexplain/post_id_divisions.json",
        ],
        "mhs": [] if args.download_mhs and not mhs.exists() else [mhs],
        "implicit_hate": [
            raw / "implicit_hate/implicit_hate_v1_stg1_posts.tsv",
            raw / "implicit_hate/implicit_hate_v1_stg2_posts.tsv",
        ],
        "gahd": [raw / "gahd/gahd.csv"],
        "detox": [raw / "detox/DeTox-Dataset_complete.sqlite3"],
    }
    missing = [
        str(p) for source in args.sources for p in required[source] if not p.is_file()
    ]
    if missing:
        ap.error("Missing source files: " + ", ".join(missing))
    selected_languages = {
        "de" if source in ("gahd", "detox") else "en" for source in args.sources
    }
    if args.language:
        selected_languages &= {args.language}
    if not selected_languages:
        ap.error("No selected sources contain the requested language.")
    targets = [
        args.output_dir / f"{lang}_{split}.parquet"
        for lang in selected_languages
        for split in ("train", "dev", "test")
    ]
    if any(p.exists() for p in targets) and not args.force:
        ap.error(
            "Prepared partitions already exist. Choose another --output-dir or use --force."
        )
    loaders = {
        "hatexplain": partial(
            load_hatexplain, required["hatexplain"][0], required["hatexplain"][1]
        ),
        "mhs": partial(
            load_mhs, path=mhs if mhs.is_file() else None, revision=args.revision
        ),
        "implicit_hate": partial(
            load_implicit_hate,
            required["implicit_hate"][0],
            required["implicit_hate"][1],
        ),
        "gahd": partial(load_gahd, required["gahd"][0]),
        "detox": partial(load_detox, required["detox"][0]),
    }
    df = load_all([loaders[source] for source in args.sources])
    df = df[df["lang"].isin(selected_languages)]
    if df.empty:
        ap.error("The selected sources contain no usable records.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": SEED,
        "ratios": [0.70, 0.15, 0.15],
        "sources": args.sources,
        "source_files": {
            str(p.relative_to(raw)) if p.is_relative_to(raw) else p.name: {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
            for source in args.sources
            for p in required[source]
        },
        "partitions": {},
    }
    for lang in sorted(df["lang"].unique()):
        for split, frame in split_one_language(
            df[df["lang"] == lang].reset_index(drop=True)
        ).items():
            path = args.output_dir / f"{lang}_{split}.parquet"
            frame.to_parquet(path, index=False)
            summary["partitions"][path.name] = {
                "rows": len(frame),
                "hateful": int(frame["gate"].sum()),
                "sources": dict(Counter(frame["source"])),
            }
            print(f"Wrote {len(frame)} records to {path}")
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
