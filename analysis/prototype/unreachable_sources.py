import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.hsrag.metrics import clean

RESULTS = Path("experiments/results")
PROCESSED = Path("data/processed")
SUBSET = "en_dev_eval_sq3_types"
A_RUNS = ["oracle_A", "oracle_A_rep2", "oracle_A_rep3"]
B_RUNS = ["oracle_B", "oracle_B_rep2", "oracle_B_rep3"]


def load(stem: str) -> dict:

    p = RESULTS / f"{stem}_live.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" not in r:
            out[r["item_id"]] = r
    return out


def predicted(rec) -> set | None:

    if rec is None or rec.get("result") is None:
        return None
    return set(rec["result"]["hate_type"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump", action="store_true", help="print the text of every unreachable item"
    )
    ap.add_argument("--max-dump", type=int, default=44)
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    gold = {str(r.id): r for r in df.itertuples(index=False)}
    held = {i for i, g in gold.items() if str(getattr(g, "sq3_role", "")) == "held_out"}

    print(f"\n{'=' * 78}")
    print("WHAT MAKES THE ORACLE-UNREACHABLE ITEMS UNREACHABLE")
    print(f"{'=' * 78}")
    print(f"\n  subset {SUBSET}, held_out stratum: {len(held)} items")

    runs = {}
    for stem in A_RUNS + B_RUNS:
        recs = load(stem)
        if not recs:
            print(f"  MISSING: {stem}_live.jsonl")
            continue
        runs[stem] = recs
        arms = Counter(r.get("arm") for r in recs.values())
        print(f"  {stem:16} {len(recs):>4} records   arms {dict(arms)}")
    if len(runs) != 6:
        raise SystemExit(
            "\n  Need all six oracle files. Stopping rather than "
            "reporting a bound computed from fewer."
        )

    rows = {}
    for item in sorted(held):
        g = gold[item]
        truth = clean(getattr(g, "hate_types", None))
        if truth is None:
            continue
        truth = set(truth)
        cell = {"gold": truth, "preds": [], "gate_false": 0, "unparsed": 0}
        for cond, stems in (("A", A_RUNS), ("B", B_RUNS)):
            n_ok = 0
            for stem in stems:
                rec = runs[stem].get(item)
                p = predicted(rec)
                if p is None:
                    cell["unparsed"] += 1
                    continue
                cell["preds"].append((cond, stem, frozenset(p)))
                if rec["result"]["hate"] is False:
                    cell["gate_false"] += 1
                if truth <= p:
                    n_ok += 1
            cell[cond] = n_ok
        rows[item] = cell

    buckets = Counter((c["A"], c["B"]) for c in rows.values())
    unreachable = [i for i, c in rows.items() if c["A"] == 0 and c["B"] == 0]
    always_ok = [i for i, c in rows.items() if c["A"] == 3 and c["B"] == 3]

    print(f"\n{'-' * 78}\n1. THE SPLIT, recomputed")
    print(f"  scored items          {len(rows)}")
    print(
        f"  correct regardless    {len(always_ok):>4} "
        f"({len(always_ok) / max(len(rows), 1):.1%})"
    )
    print(
        f"  WRONG regardless      {len(unreachable):>4} "
        f"({len(unreachable) / max(len(rows), 1):.1%})"
    )
    print(
        f"  responsive            "
        f"{len(rows) - len(always_ok) - len(unreachable):>4}"
    )
    print(f"\n  full (A,B) distribution: " f"{dict(sorted(buckets.items()))}")
    print(
        f"\n  sq3_log 4.3 records 41 / 44 / 1 partial / 36 discordant on the "
        f"86 TIED\n  items. This recomputes over the whole stratum, so the "
        f"two counts are\n  reconcilable but not identical quantities. If "
        f"the WRONG-regardless count\n  is not 44, say so rather than "
        f"quoting 44."
    )

    if not unreachable:
        raise SystemExit("\n  No unreachable items found. Stopping.")

    print(f"\n{'-' * 78}\n2. CONSISTENT DISAGREEMENT, OR SCATTERED?")
    print(
        "  Distinct predicted label sets across the six runs, per item.\n"
        "  1 means the model says the same wrong thing every time.\n"
    )
    modes = Counter()
    detail = {}
    for item in unreachable:
        sets = [p for _, _, p in rows[item]["preds"]]
        distinct = Counter(sets)
        modes[len(distinct)] += 1
        detail[item] = distinct
    for k in sorted(modes):
        bar = "#" * modes[k]
        print(f"    {k} distinct set(s)  {modes[k]:>3}  {bar}")
    consistent = modes.get(1, 0)
    print(
        f"\n  CONSISTENT (one set in all six runs): {consistent} of "
        f"{len(unreachable)} = "
        f"{consistent / len(unreachable):.0%}"
    )
    print(
        f"  The model is not guessing on these. It has a settled answer and\n"
        f"  the answer disagrees with gold."
    )

    print(f"\n{'-' * 78}\n3. WHERE THE DISAGREEMENT GOES")
    print("  gold label -> what the model says instead, over all six runs.\n")
    conf = defaultdict(Counter)
    for item in unreachable:
        for lab in rows[item]["gold"]:
            for _, _, p in rows[item]["preds"]:
                for q in p or {"(nothing predicted)"}:
                    conf[lab][q] += 1
    for lab in sorted(conf):
        top = ", ".join(f"{q} {n}" for q, n in conf[lab].most_common(4))
        n_items = sum(1 for i in unreachable if lab in rows[i]["gold"])
        print(f"    {lab:14} (n={n_items:>2})  {top}")

    print(f"\n{'-' * 78}\n4. IS IT THE GATE?")
    gated = sum(1 for i in unreachable if rows[i]["gate_false"] > 0)
    all_gated = sum(1 for i in unreachable if rows[i]["gate_false"] == 6)
    print(
        f"  items with at least one gate=False run   {gated:>3} of "
        f"{len(unreachable)}"
    )
    print(f"  items gated in ALL six runs              {all_gated:>3}")
    print(
        "\n  Under gate consistency a gate=False prediction clears hate_type\n"
        "  to [], so it is automatically a `contains` failure. An item that\n"
        "  is gated in all six runs is unreachable for a GATE reason, not a\n"
        "  hate_type reason, and a hate_type correction was never going to\n"
        "  reach it."
    )

    print(f"\n{'-' * 78}\n5. LENGTH, against the rest of the stratum")

    def chars(ids):
        v = [len(str(gold[i].text)) for i in ids]
        return (sum(v) / len(v), sorted(v)[len(v) // 2]) if v else (0, 0)

    u_mean, u_med = chars(unreachable)
    o_mean, o_med = chars(always_ok)
    print(f"    unreachable        mean {u_mean:>6.1f}  median {u_med:>4}")
    print(f"    correct regardless mean {o_mean:>6.1f}  median {o_med:>4}")
    print(
        "  Reported because it is cheap, not because length was expected to\n"
        "  explain anything. A difference here would be a confound worth\n"
        "  knowing about."
    )

    print(f"\n{'-' * 78}\n6. GOLD LABEL DISTRIBUTION")
    print(
        "  sq3_log 4.3: 'the always-wrong 44 spread across all seven labels\n"
        "  (explicit 9, incitement 10, irony 8, threatening 7, inferiority "
        "5,\n  stereotyping 4, grievance 3), so this is not one broken "
        "label.'\n"
    )
    got = Counter(l for i in unreachable for l in rows[i]["gold"])
    base = Counter(l for i in rows for l in rows[i]["gold"])
    print(f"    {'label':14} {'unreachable':>12} {'stratum':>9} {'rate':>7}")
    for lab in sorted(base):
        rate = got[lab] / base[lab] if base[lab] else 0
        print(f"    {lab:14} {got[lab]:>12} {base[lab]:>9} {rate:>6.0%}")
    print(
        "\n  n = 44 over seven labels is 3 to 10 per cell, below anything "
        "this\n  project reports as a finding. Read the spread, do not test "
        "it."
    )

    if args.dump:
        print(f"\n{'=' * 78}\n7. THE ITEMS\n{'=' * 78}")
        for item in unreachable[: args.max_dump]:
            c = rows[item]
            sets = detail[item]
            top = sets.most_common(1)[0]
            print(f"\n  {item}   gold {sorted(c['gold'])}")
            print(
                f"    model says {sorted(top[0]) if top[0] else '[]'} "
                f"in {top[1]}/6 runs"
                + (
                    "   [CONSISTENT]"
                    if len(sets) == 1
                    else f"   [{len(sets)} distinct sets]"
                )
            )
            if c["gate_false"]:
                print(f"    gate=False in {c['gate_false']}/6 runs")
            print(f"    {str(gold[item].text)[:300]}")

    print(f"\n{'=' * 78}")
    print(
        "  WHAT TO WRITE FROM THIS. If most unreachable items are "
        "CONSISTENT,\n  the ceiling bound is really a statement about the "
        "taxonomy boundary\n  rather than about retrieval, and that is "
        "stronger and more\n  transferable. If they are scattered, the bound "
        "stands as a statement\n  about item difficulty and nothing more. "
        "Either way it is a\n  characterisation at n = 44, not a measured "
        "effect."
    )
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
