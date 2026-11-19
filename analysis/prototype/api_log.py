import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

LOG = Path("experiments/results/api_log.jsonl")


PHASES = [
    ("Phase 8 (SQ3, KB re-runs, new label)", "2026-08-06", "2026-08-13"),
    ("Phase 7 (encoder baselines)", "2026-08-14", "2099-01-01"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-bad", type=int, default=5)
    args = ap.parse_args()

    ok, bad = [], []
    total = 0
    for i, line in enumerate(
        LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if not line.strip():
            continue
        total += 1
        try:
            ok.append(json.loads(line))
        except (ValueError, TypeError):
            bad.append((i + 1, line))

    print(f"\n{'=' * 74}\nAPI LOG INTEGRITY   {LOG}\n{'=' * 74}")
    print(f"  non-empty lines   {total}")
    print(f"  parsed            {len(ok)}")
    print(f"  CORRUPT           {len(bad)}")
    if bad:
        print("\n  Torn or interleaved writes. The log is appended from 8")
        print("  worker threads; if that append is not lock-protected, two")
        print("  threads can write into the same buffer flush. No result is")
        print("  affected - attribution lives on the ItemResult stamps - but")
        print("  any call total from this file is a LOWER BOUND.")
        for ln, txt in bad[: args.show_bad]:
            print(f"    line {ln}: {txt[:160]!r}")

    dated = [r for r in ok if isinstance(r, dict) and "timestamp" in r]
    print(f"\n  with timestamp    {len(dated)}")
    if not dated:
        print("  no timestamps; per-phase attribution not possible")
        return

    days = Counter(
        datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d") for r in dated
    )
    first = min(r["timestamp"] for r in dated)
    last = max(r["timestamp"] for r in dated)
    print(
        f"  first call        "
        f"{datetime.fromtimestamp(first).strftime('%Y-%m-%d %H:%M')}"
    )
    print(
        f"  last call         "
        f"{datetime.fromtimestamp(last).strftime('%Y-%m-%d %H:%M')}"
    )
    print(
        f"  span              {(last - first) / 86400:.1f} days over "
        f"{len(days)} active day(s)"
    )

    print(f"\n{'-' * 74}\nCALLS PER DAY")
    for d in sorted(days):
        print(f"  {d}   {days[d]:>7}   {'#' * min(60, days[d] // 400)}")

    print(f"\n{'-' * 74}\nPER PHASE (by the dates recorded in the sign-offs)")
    for label, lo, hi in PHASES:
        n = sum(v for k, v in days.items() if lo <= k <= hi)
        print(f"  {label:34} {n:>8}   ({lo} to {hi})")
    earliest = min(days)
    pre = sum(v for k, v in days.items() if k < PHASES[0][1])
    print(
        f"  {'everything before Phase 8':34} {pre:>8}   "
        f"({earliest} to {PHASES[0][1]})"
    )
    print(f"  {'PROJECT TOTAL (lower bound)':34} {len(ok):>8}")

    print(f"\n{'-' * 74}\nBY MODEL")
    for m, n in Counter(r.get("model", "?") for r in ok).most_common():
        print(f"  {str(m):44} {n:>8}")

    prov = Counter(r.get("provider", "(unstamped, pre-schema)") for r in ok)
    print(f"\n{'-' * 74}\nBY PROVIDER")
    for p, n in prov.most_common():
        print(f"  {str(p):44} {n:>8}")

    lat = [r["latency_s"] for r in ok if isinstance(r.get("latency_s"), (int, float))]
    if lat:
        lat_s = sorted(lat)
        print(f"\n{'-' * 74}\nLATENCY over {len(lat)} calls")
        print(
            f"  median {lat_s[len(lat_s) // 2]:.2f}s   "
            f"p90 {lat_s[int(len(lat_s) * 0.9)]:.2f}s   "
            f"p99 {lat_s[int(len(lat_s) * 0.99)]:.2f}s   "
            f"max {lat_s[-1]:.2f}s"
        )

        print(
            f"  cumulative call latency {sum(lat) / 3600:.1f} h "
            f"(~{sum(lat) / 3600 / 8:.1f} h wall clock at 8 workers)"
        )
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
