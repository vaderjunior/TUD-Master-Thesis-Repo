"""Banks english policy."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
from analysis.knowledge.banks import english_clean as c
from analysis.knowledge.banks import build as u

ROOT, BASE = (c.ROOT, c.BASE)
OLD = ROOT / "experiments/astra/sq1_kb_size_20260911_v1"


def closure(train, initial):
    """Propagate exact normalized body equivalence, never fuzzy similarity."""
    excluded = set(initial)
    links = {}
    iterations = 0
    while True:
        keys = defaultdict(list)
        for t in train:
            if t["id"] in excluded:
                for kind, key in (
                    ("username_body", c.body_norm(t["text"])),
                    ("wrapper_body", c.wrapper_key(t["text"])),
                ):
                    if key:
                        keys[kind, key].append(t["id"])
        add = {}
        for t in train:
            if t["id"] not in excluded:
                hits = []
                for kind, key in (
                    ("username_body", c.body_norm(t["text"])),
                    ("wrapper_body", c.wrapper_key(t["text"])),
                ):
                    if key and (kind, key) in keys:
                        hits.append(
                            {
                                "rule": kind,
                                "key": key,
                                "already_excluded_ids": sorted(keys[kind, key]),
                            }
                        )
                if hits:
                    add[t["id"]] = hits
        if not add:
            break
        excluded.update(add)
        links.update(add)
        iterations += 1
        assert iterations <= len(train)
    return (excluded, links, iterations)


def annotation_counts(records):
    return {
        "rows": len(records),
        "sources": dict(Counter((r["source"] for r in records))),
        "gate": dict(Counter((str(r["meta"]["gate"]) for r in records))),
        "annotated": {
            g: sum((r["meta"].get(g) is not None for r in records)) for g in u.GOLD
        },
        "labels": {
            g: dict(Counter((x for r in records for x in r["meta"].get(g) or [])))
            for g in ("target_groups", "hate_types")
        },
    }


def write_lines(name, rows):
    p = BASE / name
    assert p.resolve().is_relative_to(BASE.resolve()) and (not p.exists())
    with p.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )


def main():
    assert not (BASE / "complete.json").exists() and (
        not (BASE / "policy.json").exists()
    ), "Preserve frozen policy"
    survey = u.read_json(BASE / "survey.json")
    review = u.read_json(c.REVIEW)
    result = u.read_json(c.RESULT)
    for name, h in {
        **survey["input_hashes"],
        **result["input_hashes"],
        **review["input_hashes"],
    }.items():
        assert u.digest(ROOT / name) == h, name
    train = u.read_lines(c.AUDIT / "train_texts.jsonl")
    held = u.read_lines(c.AUDIT / "heldout_texts.jsonl")
    scan = set(u.read_json(c.AUDIT / "report.json")["proposed_excluded_ids"])
    nearest = {x["train_id"]: x for x in u.read_lines(c.AUDIT / "nearest_train.jsonl")}
    oldsmall = set(u.read_json(c.AUDIT / "preparation.json")["previous_small_ids"])
    keep_orders = {1, 11, 32, 40, 55, 72, 83}
    manual = []
    for x in review["manual_review"]:
        keep = x["review_order"] in keep_orders
        if x["review_order"] == 19:
            reason = "Conservative exclusion: same short pronoun/evaluation with contraction and intensifier; source identity unknown."
        elif x["review_order"] == 58:
            reason = "Conservative exclusion: same distinctive violent throat-kicking wording, changed intent framing; source identity unknown."
        else:
            reason = x["reason"]
        manual.append(
            {
                **x,
                "decision": "retain" if keep else "exclude",
                "policy_reason": (
                    reason
                    if keep
                    else "Conservative text-family/close-wording exclusion. " + reason
                ),
                "policy_state": "Frozen additive policy; original review categories retained as provenance.",
            }
        )
    manual_ids = {x["train_id"] for x in manual if x["decision"] == "exclude"}
    body_ids = {x["train_id"] for x in review["username_body_candidates"]}
    wrapper_ids = {x["train_id"] for x in survey["wrapper_candidates"]}
    seeds = scan | manual_ids | body_ids | wrapper_ids
    excluded, propagation, iterations = closure(train, seeds)
    retained_exceptions = [x for x in manual if x["decision"] == "retain"]
    assert all(
        (x["train_id"] not in excluded for x in retained_exceptions)
    ), "Explicit retain collided with exact-body rule; review first"
    for t in train:
        if t["id"] not in excluded:
            assert nearest[t["id"]]["max_cosine"] < 0.95
    heldbody = {c.body_norm(h["text"]) for h in held}
    heldwrap = {key for h in held if (key := c.wrapper_key(h["text"]))}
    assert not [
        t
        for t in train
        if t["id"] not in excluded and c.body_norm(t["text"]) in heldbody
    ]
    assert not [
        t
        for t in train
        if t["id"] not in excluded and c.wrapper_key(t["text"]) in heldwrap
    ]
    raw = pd.read_parquet(ROOT / "data/processed/en_train.parquet").to_dict("records")
    assert [{k: t[k] for k in ("id", "text", "lang", "source")} for t in raw] == train
    assert all((any((u.clean(t.get(g)) is not None for g in u.GOLD)) for t in raw))
    assert all(
        (u.clean(t.get("legal")) is None for t in raw)
    ), "English legal unexpectedly annotated"
    full = [u.example(t) for t in raw if t["id"] not in excluded]
    byid = {x["id"]: x for x in full}
    oldrecords = [
        x
        for x in u.read_lines(OLD / "kb/small/records.jsonl")
        if x["kind"] == "example" and x["lang"] == "en"
    ]
    small = [x for x in oldrecords if x["id"].removeprefix("ex-") not in excluded]
    assert all((byid[x["id"]] == x for x in small))
    assert len({u.canonical_text(x["text"]) for x in full}) == len(full) == len(byid)
    encoder_paths = sorted(
        (
            p
            for family in ("full", "full_nogrievance", "kb")
            for p in (ROOT / "data/encoder" / family).glob("*.parquet")
            if p.stem.endswith(("_train", "_val"))
        )
    )
    encoders = {}
    for p in encoder_paths:
        df = pd.read_parquet(p, columns=["id"])
        ids = set(df.id.astype(str))
        assert len(ids) == len(df)
        encoders[u.rel(p)] = {
            "rows": len(df),
            "excluded_ids": sorted(ids & excluded),
            "excluded_count": len(ids & excluded),
            "additional_beyond_scan_ids": sorted((ids & excluded) - scan),
            "note": "Membership only: train is gradient-data candidate, val is model-selection data. No score effect or retrain decision inferred.",
        }
    paths = [
        Path(__file__),
        ROOT / "tests/knowledge/banks/test_english_policy.py",
        BASE / "survey.json",
        c.RESULT,
        c.REVIEW,
        ROOT / "data/processed/en_train.parquet",
        c.AUDIT / "report.json",
        c.AUDIT / "preparation.json",
        OLD / "kb/small/records.jsonl",
        OLD / "kb/all/records.jsonl",
        *encoder_paths,
        ROOT / "kb/records.jsonl",
        ROOT / "config/config.yaml",
        ROOT / "src/hsrag/prompt.py",
        ROOT / "experiments/leakage_train_exclusions.json",
    ]
    hashes = {u.rel(p): u.digest(p) for p in paths}
    added = excluded - scan
    counts = {
        "raw_train": len(train),
        "frozen_scan_proposal": len(scan),
        "manual_excluded": len(manual_ids),
        "manual_retained": len(retained_exceptions),
        "username_candidates": len(body_ids),
        "wrapper_candidates": len(wrapper_ids),
        "seed_union": len(seeds),
        "propagation_added": len(excluded - seeds),
        "total_additional_beyond_scan": len(added),
        "total_excluded": len(excluded),
        "full_english": len(full),
        "small_english": len(small),
        "small_dropped_from_previous_148": sorted(oldsmall & excluded),
        "additional_small_beyond_scan": sorted(oldsmall & added),
    }
    policy = {
        "id": "ASTRA-ENGLISH-TEXT-POLICY-20260914-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "frozen_scan_ids": sorted(scan),
        "manual_review_decisions": manual,
        "username_body_ids": sorted(body_ids),
        "wrapper_ids": sorted(wrapper_ids),
        "wrapper_review": "All 41 saved survey candidates and their heldout text/reference matches inspected; accepted for conservative text-body filtering.",
        "propagation": propagation,
        "closure_iterations": iterations,
        "final_excluded_ids": sorted(excluded),
        "rule_order": [
            "preserve 320-ID scan/carry-forward union",
            "six handle-normalized bodies",
            "41 inspected exact wrapper-body matches across full rosters",
            "76 individually reviewed close-body/formulation exclusions, seven explicit retains",
            "fixed-point exact username/wrapper equivalence among TRAIN, without fuzzy matching or annotation merging",
        ],
        "source_policy": "Only eligible TRAIN annotations form pools. No heldout labels used. Original example metadata/schema preserved, missing vs empty retained. No within-TRAIN canonical duplicates here; do not merge merely wrapper-equivalent or semantically similar labels.",
        "evaluation_policy": "This is post-hoc conservative text-policy preparation, informed by previously seen dev/test texts and results. It does not create untouched evaluation or prove absence of semantic overlap. Report retained exceptions and regressions.",
        "next": "English eligible records only. Final bilingual index, prompt/protocol/reuse audit, concurrent API logging fix and classifier batch are separate.",
        "input_hashes": hashes,
    }
    inventory = {
        "pool_counts": {
            "full": annotation_counts(full),
            "small": annotation_counts(small),
        },
        "encoder_split_membership": encoders,
        "encoder_action": "Existing encoder files/checkpoints unchanged. Decide per-head reruns only after final protocol/alignment review.",
        "heldout_checks": {
            "retained_at_or_above_095": 0,
            "retained_username_body_matches": 0,
            "retained_wrapper_key_matches": 0,
            "retained_manual_exceptions": len(retained_exceptions),
        },
        "scope": "Pool files contain English examples only, no German examples or abstracts, no Chroma/index. Small is an identical record subset of full.",
    }
    assert all((u.digest(ROOT / n) == h for n, h in hashes.items()))
    c.put("policy.json", policy)
    write_lines("full_english_records.jsonl", full)
    write_lines("small_english_records.jsonl", small)
    c.put("inventory.json", inventory)
    outputs = {
        u.rel(BASE / n): u.digest(BASE / n)
        for n in (
            "policy.json",
            "full_english_records.jsonl",
            "small_english_records.jsonl",
            "inventory.json",
        )
    }
    assert all((u.digest(ROOT / n) == h for n, h in hashes.items()))
    c.put(
        "complete.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": outputs,
            "status": "english_policy_and_record_pools_frozen_no_classifier_or_index",
            "input_hashes": hashes,
        },
    )
    print(
        json.dumps(
            {
                "counts": counts,
                "outputs": outputs,
                "encoders": {
                    p: {k: v for k, v in x.items() if k in ("rows", "excluded_count")}
                    for p, x in encoders.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
