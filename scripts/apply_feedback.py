import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from analysis.feedback.relevance.prototype_coverage import PROCESSED
from analysis.feedback.relevance.prototype_coverage import local_gold_sets
from scripts.make_sq3_subset import partition_by_label

RESULTS = Path("experiments/results")
KB = Path("kb")
BASE_RECORDS = KB / "records.jsonl"
FEEDBACK_LOG = KB / "feedback_log.jsonl"
SUBSET = "en_dev_eval_sq3_types"
GOLD_COL, DIM = "hate_types", "hate_type"
CRITERION = "contains"
N_BATCHES = 4
SEED = 42
ARM_TAG = {"feedback": "fb", "control": "ctl"}


def batches(df: pd.DataFrame) -> dict:

    role = df[df["sq3_role"] == "batches"].reset_index(drop=True)
    n = len(role)
    sizes = {
        f"b{i + 1}": n // N_BATCHES + (1 if i < n % N_BATCHES else 0)
        for i in range(N_BATCHES)
    }
    return partition_by_label(role, sizes, SEED)


def draw_train_topup(n: int, round_n: int, exclude: set) -> list:

    df = pd.read_parquet(PROCESSED / "en_train.parquet")
    g = local_gold_sets(df, GOLD_COL)
    cands = [
        r
        for r in df.itertuples(index=False)
        if str(r.id) in g and " ".join(str(r.text).lower().split()) not in exclude
    ]
    assert len(cands) >= n, f"only {len(cands)} eligible train items for {n}"

    return [(r, g[str(r.id)]) for r in random.Random(SEED + round_n).sample(cands, n)]


def load_results(stem: str, arm: str = "rag") -> dict:
    path = RESULTS / f"{stem}_live.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" in r or r.get("arm") != arm:
            continue
        out[r["item_id"]] = r
    return out


def make_record(row, gold: set, round_n: int, arm: str) -> dict:

    text = str(row.text)
    return {
        "id": "fb-" + hashlib.sha1(" ".join(text.split()).encode()).hexdigest()[:12],
        "kind": "feedback",
        "dimension": None,
        "label": None,
        "lang": str(row.lang),
        "text": text,
        "source": f"sq3-{arm}-r{round_n}",
        "meta": {
            "gate": bool(row.gate),
            "target_groups": None,
            "hate_types": sorted(gold),
            "severity": None,
            "illustrative_only": False,
            "round": round_n,
            "origin_item_id": str(row.id),
            "criterion": CRITERION,
            "feedback_arm": arm,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument(
        "--results",
        default=None,
        help="results stem for BOTH arms. Correct at round 1, "
        "where both arms share the round-0 KB.",
    )
    ap.add_argument(
        "--results-fb",
        default=None,
        help="from round 2 on, the feedback arm's own previous "
        "run - its errors are errors of ITS KB",
    )
    ap.add_argument(
        "--results-ctl", default=None, help="the control arm's own previous run"
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    assert 1 <= args.round <= N_BATCHES, f"round must be 1..{N_BATCHES}"

    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    batch = batches(df)[f"b{args.round}"]
    gold = local_gold_sets(batch, GOLD_COL)

    stem_fb = args.results_fb or args.results
    stem_ctl = args.results_ctl or args.results
    assert stem_fb and stem_ctl, "give --results, or --results-fb and --results-ctl"
    res_fb, res_ctl = load_results(stem_fb), load_results(stem_ctl)

    allowed = set(df.loc[df["sq3_role"] == "batches", "id"].astype(str))
    assert set(gold) <= allowed, (
        "batch contains items outside the batches role - a correction from "
        "held_out or pool would turn retrieval into a lookup of the answer"
    )

    for stem, res in ((stem_fb, res_fb), (stem_ctl, res_ctl)):
        missing = set(gold) - set(res)
        assert not missing, (
            f"{stem} is missing {len(missing)} of this batch's items; the "
            f"batch was never classified under the KB this round corrects"
        )

    def ok(res, iid) -> bool:
        r = res[iid].get("result")
        if r is None:
            return False
        pred = set(r.get(DIM) or [])
        return gold[iid] <= pred if CRITERION == "contains" else (pred == gold[iid])

    errors = sorted(i for i in gold if not ok(res_fb, i))
    correct = sorted(i for i in gold if ok(res_ctl, i))
    k = len(errors)
    assert k > 0, "no errors in this batch; the round has nothing to correct"

    base = [
        json.loads(l)
        for l in BASE_RECORDS.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    base_texts = {" ".join(r["text"].lower().split()) for r in base}
    batch_texts = {
        " ".join(str(r.text).lower().split()) for r in batch.itertuples(index=False)
    }

    rows = {str(r.id): r for r in batch.itertuples(index=False)}
    fb_items = [(rows[i], gold[i]) for i in errors]
    ctl_items = [(rows[i], gold[i]) for i in correct[:k]]
    n_topup = k - len(ctl_items)
    if n_topup > 0:
        ctl_items += draw_train_topup(n_topup, args.round, base_texts | batch_texts)
    picks = {"feedback": fb_items, "control": ctl_items}

    print(
        f"\n{'=' * 74}\nround {args.round}  batch b{args.round}  " f"{len(batch)} items"
    )
    print(f"  errors from  {stem_fb}\n  correct from {stem_ctl}")
    print(f"  criterion={CRITERION}: {len(errors)} wrong, {len(correct)} right")
    print(f"  k = {k} records per arm (all errors corrected)")
    if n_topup:
        print(
            f"  control: {len(correct[:k])} from this batch + {n_topup} "
            f"correctly-labelled TRAIN items, so both arms add {k}"
        )

    for arm, items in picks.items():
        new = [make_record(row, g, args.round, arm) for row, g in items]
        tag = ARM_TAG[arm]

        prior = []
        if FEEDBACK_LOG.exists():
            prior = [
                json.loads(l)
                for l in FEEDBACK_LOG.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
        prior = [
            r
            for r in prior
            if r["meta"]["feedback_arm"] == arm and r["meta"]["round"] < args.round
        ]

        by_id = {r["id"]: r for r in prior}
        by_id.update({r["id"]: r for r in new})
        feedback = list(by_id.values())

        clash = [
            r for r in feedback if " ".join(r["text"].lower().split()) in base_texts
        ]
        assert not clash, (
            f"{len(clash)} corrections duplicate a base record's text; "
            f"check_kb_schema asserts KB-wide text uniqueness and the build "
            f"would fail after spending a minute on embeddings"
        )

        out = KB / f"records_sq3_{tag}_r{args.round}.jsonl"
        labels = Counter(l for r in new for l in r["meta"]["hate_types"])
        print(
            f"\n  {arm:9} +{len(new)} this round, {len(feedback)} "
            f"cumulative -> {len(base) + len(feedback)} records"
        )
        print(
            f"    labels: " + ", ".join(f"{k_}={v}" for k_, v in sorted(labels.items()))
        )
        print(f"    {out}")
        if args.dry_run:
            continue
        with open(out, "w", encoding="utf-8") as f:
            for r in base + feedback:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.dry_run:
        print(f"\n  --dry-run: nothing written\n{'=' * 74}\n")
        return

    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        for arm, items in picks.items():
            for row, g in items:
                f.write(
                    json.dumps(make_record(row, g, args.round, arm), ensure_ascii=False)
                    + "\n"
                )
    print(
        f"\n  appended {sum(len(v) for v in picks.values())} entries to "
        f"{FEEDBACK_LOG}\n{'=' * 74}\n"
    )


if __name__ == "__main__":
    main()
