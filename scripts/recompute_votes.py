import argparse
import json
from collections import Counter
from pathlib import Path

from src.hsrag.parse import RunResult
from src.hsrag.schema import Result
from src.hsrag.vote import aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.path)
    out = Path(args.out) if args.out else src.with_suffix(".revoted.jsonl")

    lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    changed = 0
    written = []

    for line in lines:
        r = json.loads(line)
        if "_manifest" in r:
            written.append(line)
            continue

        runs = [
            RunResult(result=Result(**p) if p else None) for p in r["run_predictions"]
        ]
        v = aggregate(runs)

        before = r["result"]
        after = v.result.model_dump() if v.result is not None else None
        if before != after:
            changed += 1
        r["result"] = after
        r["uncertain"] = v.uncertain
        r["n_valid"] = v.n_valid
        r["agreement"] = v.agreement
        written.append(json.dumps(r, ensure_ascii=False))

    out.write_text("\n".join(written) + "\n", encoding="utf-8")
    print(f"{len(written) - 1} results revoted, {changed} changed")
    print(f"wrote {out}")

    res = [json.loads(l) for l in written if "_manifest" not in l]
    for field in ("target_group", "hate_type", "legal"):
        c = Counter(len(r["result"][field]) if r["result"] else None for r in res)
        print(f"  {field:14} labels per item: {dict(sorted(c.items(), key=str))}")


if __name__ == "__main__":
    main()
