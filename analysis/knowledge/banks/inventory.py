"""Banks inventory."""

from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = PROJECT_ROOT
OUT = ROOT / "Thesis_Log/astra_result/evidence/kb_inventory_20260912_v1.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    inputs = [
        ROOT / p
        for p in (
            "kb/records.jsonl",
            "data/processed/en_train.parquet",
            "data/processed/de_train.parquet",
            "data/processed/de_legal_train.parquet",
            "scripts/build_kb_records.py",
            "config/config.yaml",
        )
    ]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs}
    records = [
        json.loads(v) for v in inputs[0].read_text(encoding="utf-8").splitlines() if v.strip()
    ]
    kb = {}
    for lang in ("en", "de"):
        rows = [r for r in records if r["kind"] == "example" and r["lang"] == lang]
        kb[lang] = {
            "examples": len(rows),
            "sources": dict(Counter((r["source"] for r in rows))),
            "gate": dict(Counter((str(r["meta"].get("gate")) for r in rows))),
            "illustrative_only": sum((bool(r["meta"].get("illustrative_only")) for r in rows)),
        }
    pools, ids = ({}, {})
    for name in ("en_train", "de_train", "de_legal_train"):
        df = pd.read_parquet(ROOT / f"data/processed/{name}.parquet")
        ids[name] = set(df.id)
        pools[name] = {
            "rows": len(df),
            "unique_ids": len(ids[name]),
            "sources": dict(Counter(df.source)),
            "gate": dict(Counter((str(v) for v in df.gate))),
        }
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kb": kb,
        "raw_train_pools": pools,
        "german_train_union_ids": len(ids["de_train"] | ids["de_legal_train"]),
        "german_train_shared_ids": len(ids["de_train"] & ids["de_legal_train"]),
        "input_hashes": hashes,
        "script_sha256": digest(Path(__file__)),
        "limits": "Inventory only. Raw counts are not final overlap-cleaned KB sizes. No embeddings, retrieval, predictions, KB generation or new semantic-overlap scan.",
        "original_selection": "Builder caps up to 10 rows per target-group/type/severity label, rich annotations first, adds up to 20 hate negatives and deduplicates IDs. English 149 is the resulting unique count, not a fixed quota or uniform gate balance. German additionally has DeTox illustrations and BoTox legal examples.",
    }
    assert all((digest(ROOT / p) == h for p, h in hashes.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("EVIDENCE_SHA256", digest(OUT), flush=True)


if __name__ == "__main__":
    main()
