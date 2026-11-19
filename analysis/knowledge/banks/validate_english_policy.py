from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from analysis.knowledge.banks import english_clean as c
from analysis.knowledge.banks import english_policy as p
from analysis.knowledge.banks import build as u

ROOT, BASE = (c.ROOT, c.BASE)
OUT = (
    ROOT / "Thesis_Log/astra_result/evidence/english_kb_cleanup_check_20260914_v1.json"
)


def main():
    assert not OUT.exists(), "Preserve existing final receipt"
    receipt = u.read_json(BASE / "complete.json")
    policy = u.read_json(BASE / "policy.json")
    survey = u.read_json(BASE / "survey.json")
    audit = u.read_json(c.AUDIT / "report.json")
    prior = u.read_json(c.RESULT)
    sup = u.read_json(c.REVIEW)
    hashes = {
        **receipt["input_hashes"],
        **receipt["outputs"],
        **survey["input_hashes"],
        **prior["input_hashes"],
        **sup["input_hashes"],
    }
    extra = [
        Path(__file__),
        BASE / "complete.json",
        c.RESULT,
        c.REVIEW,
        ROOT / "Thesis_Log/astra_result/astra_english_kb_review_20260914_v1.md",
    ]
    hashes.update({u.rel(x): u.digest(x) for x in extra})
    assert all((u.digest(ROOT / n) == h for n, h in hashes.items()))
    full = u.read_lines(BASE / "full_english_records.jsonl")
    small = u.read_lines(BASE / "small_english_records.jsonl")
    raw = pd.read_parquet(ROOT / "data/processed/en_train.parquet").to_dict("records")
    byid = {x["id"]: x for x in raw}
    excluded = set(policy["final_excluded_ids"])
    assert len(excluded) == 437
    assert {x["id"].removeprefix("ex-") for x in full} == set(byid) - excluded
    assert len(full) == 55753 and len(small) == 147
    for x in full:
        src = byid[x["id"].removeprefix("ex-")]
        assert (
            x["kind"] == "example" and x["lang"] == "en" and (x["text"] == src["text"])
        )
        assert x["source"] == src["source"] + "-train"
        expected = {
            k: u.clean(src.get(k))
            for k in ("gate", "target_groups", "hate_types", "severity")
        }
        assert x["meta"] == {**expected, "illustrative_only": False}
        assert x["dimension"] is None and x["label"] is None
    current = {x["id"]: x for x in full}
    assert all((current[x["id"]] == x for x in small))
    old = u.read_lines(p.OLD / "kb/small/records.jsonl")
    expected_small = [
        x
        for x in old
        if x["kind"] == "example"
        and x["lang"] == "en"
        and (x["id"].removeprefix("ex-") not in excluded)
    ]
    assert small == expected_small
    held = u.read_lines(c.AUDIT / "heldout_texts.jsonl")
    hb = {c.body_norm(x["text"]) for x in held}
    hw = {k for x in held if (k := c.wrapper_key(x["text"]))}
    assert not any(
        (c.body_norm(x["text"]) in hb or c.wrapper_key(x["text"]) in hw for x in full)
    )
    assert p.closure(raw, excluded)[0] == excluded and (not p.closure(raw, excluded)[1])
    scanset = set(audit["proposed_excluded_ids"])
    assert scanset <= excluded
    assert all(
        (
            x["train_id"] not in excluded
            for x in policy["manual_review_decisions"]
            if x["decision"] == "retain"
        )
    )
    propagation_review = [
        {
            "train_id": rid,
            "train_text": byid[rid]["text"],
            "links": links,
            "assessment": "Inspected: exact punctuation/retweet/handle/body variant of an excluded TRAIN text; conservative family exclusion.",
        }
        for rid, links in policy["propagation"].items()
    ]
    assert len(propagation_review) == 13
    encoders = {}
    for name in u.read_json(BASE / "inventory.json")["encoder_split_membership"]:
        df = pd.read_parquet(ROOT / name, columns=["id"])
        raw_ids = df.id.astype(str).tolist()
        canonical = {x.removeprefix("ex-") for x in raw_ids}
        assert len(canonical) == len(raw_ids) and canonical <= set(byid)
        hit = canonical & excluded
        encoders[name] = {
            "rows": len(raw_ids),
            "ids_with_ex_prefix": sum((x.startswith("ex-") for x in raw_ids)),
            "excluded_count": len(hit),
            "excluded_ids": sorted(hit),
            "additional_beyond_scan_ids": sorted(hit - scanset),
            "remaining_under_policy": len(raw_ids) - len(hit),
        }
    historical = {}
    h_by_id = {ref["id"]: h for h in held for ref in h["references"]}
    for pool in ("small", "all"):
        direct = []
        for key, ctx in u.read_json(p.OLD / f"retrieval_{pool}.json").items():
            subset, qid = key.split("/", 1)
            text = h_by_id[qid]["text"]
            for ex in ctx["Labelled examples"]:
                rid = ex.removeprefix("ex-")
                rt = byid[rid]["text"]
                body = c.body_norm(rt) == c.body_norm(text)
                wrapper = c.wrapper_key(rt) is not None and c.wrapper_key(
                    rt
                ) == c.wrapper_key(text)
                if body or wrapper:
                    assert rid in excluded
                    direct.append(
                        {
                            "subset": subset,
                            "query_id": qid,
                            "train_id": rid,
                            "username_body_equal": body,
                            "wrapper_body_equal": wrapper,
                        }
                    )
        historical[pool] = direct
    result = {
        "id": "ASTRA-ENGLISH-CLEANUP-CHECK-20260914-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_policy_sha256": u.digest(BASE / "policy.json"),
        "counts": policy["counts"],
        "verified": {
            "source_annotations_for_all_full_records": len(full),
            "identical_small_subset": len(small),
            "no_retained_declared_direct_overlap": True,
            "exact_family_closure_stable": True,
            "manual_retain_exceptions_preserved": 7,
            "old_inputs_unchanged": True,
        },
        "propagation_text_review": propagation_review,
        "encoder_split_membership_corrected": encoders,
        "amendment": "Supersedes encoder_split_membership in cleanup inventory.json: initial membership used raw IDs, so ex-prefixed compact-KB encoder IDs falsely appeared unaffected. Canonical remove-ex-prefix joins are now checked against original TRAIN IDs for every split. Full encoder counts are unchanged. Frozen exclusions and English pool files are unaffected; original inventory is preserved.",
        "historical_direct_body_matches": historical,
        "interpretation": "Final English pool policy for next preparation, not universal semantic independence. No encoder training or score effect calculated. Encoder overlap with this broader policy requires protocol/per-head source review before reusing a baseline as clean/matched; do not silently drop validation rows from an already selected checkpoint and call it retrained.",
        "input_hashes": hashes,
    }
    assert all((u.digest(ROOT / n) == h for n, h in hashes.items()))
    with OUT.open("x", encoding="utf-8") as f:
        import json

        json.dump(
            result, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        f.write("\n")
    print(
        u.json_bytes(
            {
                "path": u.rel(OUT),
                "sha256": u.digest(OUT),
                "input_hashes": len(hashes),
                "encoder_counts": {n: v["excluded_count"] for n, v in encoders.items()},
                "historical": historical,
            }
        ).decode(),
        flush=True,
    )


if __name__ == "__main__":
    main()
