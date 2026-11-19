from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
from analysis.knowledge.banks import german_audit as audit
from analysis.knowledge.banks.german_results import username_norm

ROOT, BASE = (audit.ROOT, audit.BASE)
EVIDENCE = ROOT / "Thesis_Log/astra_result/evidence/german_kb_results_20260913_v1.json"
OUT = (
    ROOT
    / "Thesis_Log/astra_result/evidence/german_kb_overlap_followup_20260913_v1.json"
)


def main():
    assert not OUT.exists(), "Preserve existing evidence"
    evidence = audit.util.read_json(EVIDENCE)
    paths = [
        EVIDENCE,
        BASE / "heldout_texts.jsonl",
        BASE / "train_texts.jsonl",
        ROOT / "kb/records.jsonl",
        ROOT / "analysis/knowledge/banks/german_audit.py",
        ROOT / "analysis/knowledge/banks/german_results.py",
        Path(__file__),
    ]
    paths += [ROOT / f"data/encoder/de/hate_{s}.parquet" for s in ("train", "val")]
    historical = sorted(
        (ROOT / "experiments/results").glob("de_legal_kbv3*_live.jsonl")
    )
    paths += historical
    hashes = {audit.util.rel(p): audit.util.digest(p) for p in paths}
    tr = {r["id"]: r for r in audit.util.read_lines(BASE / "train_texts.jsonl")}
    held = audit.util.read_lines(BASE / "heldout_texts.jsonl")
    original = {
        audit.canonical_id(r["id"]): r
        for r in audit.util.read_lines(ROOT / "kb/records.jsonl")
        if r["kind"] == "example" and r["lang"] == "de"
    }
    enc = {
        s: set(
            pd.read_parquet(
                ROOT / f"data/encoder/de/hate_{s}.parquet", columns=["id"]
            ).id.astype(str)
        )
        for s in ("train", "val")
    }
    history = {
        p.name: [r for r in audit.util.read_lines(p) if r.get("arm") == "rag"]
        for p in historical
    }
    pairs, coverage, small, encoder_dev = ([], defaultdict(set), [], [])
    for r in evidence["retained_username_normalized_candidates"]:
        rid = r["train_id"]
        references = []
        for ix in r["heldout_text_indices"]:
            h = held[ix]
            assert username_norm(tr[rid]["text"]) == username_norm(h["text"])
            references += h["references"]
            pairs.append(
                {
                    "train_id": rid,
                    "source": tr[rid]["source"],
                    "train_text": tr[rid]["text"],
                    "heldout_text": h["text"],
                    "heldout_references": h["references"],
                    "global_max_cosine": r["max_cosine"],
                    "in_original_small_kb": rid in original,
                }
            )
            for ref in h["references"]:
                coverage[ref["file"]].add(rid)
        if rid in original:
            item = {
                "train_id": rid,
                "record_id": original[rid]["id"],
                "source": original[rid]["source"],
                "illustrative_only": original[rid]["meta"].get("illustrative_only"),
                "heldout_references": references,
                "historical_legal_dev_rag_exposure": [],
            }
            for ref in references:
                if not ref["file"].endswith("de_legal_dev_eval.parquet"):
                    continue
                for name, rr in history.items():
                    row = next((z for z in rr if z["item_id"] == ref["id"]))
                    item["historical_legal_dev_rag_exposure"].append(
                        {
                            "file": name,
                            "item_id": ref["id"],
                            "matching_record_delivered": original[rid]["id"]
                            in [
                                i
                                for ids in row.get("retrieved", {}).values()
                                for i in ids
                            ],
                        }
                    )
            small.append(item)
        if any((ref["file"].endswith("de_dev_eval.parquet") for ref in references)):
            encoder_dev.append(
                {
                    "train_id": rid,
                    "source": tr[rid]["source"],
                    "in_encoder_train": rid in enc["train"],
                    "in_encoder_internal_val": rid in enc["val"],
                    "heldout_references": [
                        ref
                        for ref in references
                        if ref["file"].endswith("de_dev_eval.parquet")
                    ],
                }
            )
    selected = []
    by_id = {r["train_id"]: r for r in evidence["retained_094_to_095"]}
    judgments = {
        "gahd-ae7de5df": "Same short sentence and exclusionary meaning with a synonym substitution. Recommend treating as a near-copy in an explicit future manual amendment; below frozen .95 cutoff.",
        "botox-658": "Same comment body with two original handles replaced by @user. Supports a systematic username-normalized text exclusion rule across both pools.",
        "detox-1406639349408616454": "Nearest texts differ by emoji content and URL. High similarity alone does not establish an equivalent duplicate. Preserve as a review candidate.",
    }
    for rid, note in judgments.items():
        row = by_id[rid]
        selected.append(
            {
                "train_id": rid,
                "cosine": row["max_cosine"],
                "train_text": tr[rid]["text"],
                "heldout": held[row["heldout_text_index"]],
                "reviewer_interpretation": note,
            }
        )
    candidates = evidence["retained_username_normalized_candidates"]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "screening_rule": "Replace Twitter-style ASCII @handles with @user, preserving handle count/order and all other text; then apply existing NFKC/casefold/whitespace/edge-punctuation normalization. Diagnostic only, not a change to the frozen exclusion policy. No URL/emoji/hashtag removal.",
        "username_candidate_train_rows": len(candidates),
        "username_candidate_pairs": len(pairs),
        "candidate_sources": dict(
            Counter((tr[r["train_id"]]["source"] for r in candidates))
        ),
        "small_candidate_records": small,
        "candidate_coverage_by_file": {
            f: sorted(v) for f, v in sorted(coverage.items())
        },
        "username_pairs": pairs,
        "retained_boundary_candidates": len(evidence["retained_094_to_095"]),
        "boundary_username_intersection": len(
            set(by_id) & {r["train_id"] for r in candidates}
        ),
        "selected_pair_review": selected,
        "german_gate_dev_encoder_membership": encoder_dev,
        "interpretation": "The completed scan is numerically valid. Its .95 threshold is insufficient for certifying an overlap-free pool. Identical normalized bodies can be anonymized copies or repeated statements addressed to different users; text matching alone does not establish identical source events or labels. Review/apply a conservative declared policy before final pool construction, consistently for small and full pools. Six original small records are affected by this additional screen. Do not infer score inflation or retraining needs solely from candidate counts.",
        "next_step": "Freeze an additive text-exclusion/annotation policy, retain original 167-ID scan results, review the 47 near-boundary candidates without gold-score optimization, and prepare the isolated corrected renderer. Check any candidate affecting the existing encoder evaluation before claiming complete overlap clearance. No classifier run or historical rescoring in this follow-up.",
        "input_hashes": hashes,
    }
    assert all((audit.util.digest(ROOT / p) == h for p, h in hashes.items()))
    with OUT.open("xb") as f:
        f.write(audit.js(result))
    print(
        json.dumps(
            {
                "candidate_rows": len(candidates),
                "candidate_pairs": len(pairs),
                "candidate_sources": result["candidate_sources"],
                "small_candidates": len(small),
                "coverage": {f: len(v) for f, v in coverage.items()},
                "encoder_gate_dev": encoder_dev,
                "verified_hashes": len(hashes),
                "evidence": audit.util.rel(OUT),
                "sha256": audit.util.digest(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
