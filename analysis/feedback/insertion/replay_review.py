import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
)
import argparse
from collections import Counter
from pathlib import Path
import unittest
import numpy as np
from analysis.feedback.insertion import common as c
from analysis.feedback.insertion import prepare as prep
from analysis.feedback.insertion import validate as check

REVIEW = c.BASE / "cold_replay_review_v1"
REPORT = REVIEW / "inspection.json"
SENSITIVITY = REVIEW / "sensitivity_plan.json"
DECISION = REVIEW / "qualification.json"
TESTS = REVIEW / "tests.json"
AMENDMENT = REVIEW / "PROTOCOL_AMENDMENT.md"
EXPECTED_KEYS = sorted(
    [
        "en/hatexplain-1080863043356160000_twitter",
        "en/implicit_hate-15658",
        "en/implicit_hate-20128",
        "en/implicit_hate-20606",
        "en/implicit_hate-20720",
        "en/implicit_hate-4207",
    ]
)
ALLOWED_FIELDS = {
    "channels",
    "context",
    "context_sha256",
    "dense_candidates",
    "messages",
    "messages_sha256",
    "reference_input_tokens",
    "retrieved",
}


def sources():
    return [
        Path(__file__),
        c.ROOT / "tests/feedback/insertion/test_replay_review.py",
        c.ROOT / "analysis/feedback/insertion/scores.py",
        c.ROOT / "scripts/run_correction_insertion.ps1",
        AMENDMENT,
    ]


def assess_change(old, row):

    new = row.get("actual_entry", old)
    fields = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    c.require(fields == row["changed_fields"], "Incorrect changed-field inventory")
    c.require(
        row["same_messages"] == (old["messages"] == new["messages"])
        and row["same_retrieved_ids"] == (old["retrieved"] == new["retrieved"]),
        "Incorrect replay flags",
    )
    c.require(
        set(fields) <= ALLOWED_FIELDS, "Unreviewed query/condition/insertion change"
    )
    if not fields:
        return None
    c.require(
        old["channels"]["bm25"] == new["channels"]["bm25"], "Lexical channel changed"
    )
    left = {h["id"]: h["score"] for h in old["dense_candidates"]}
    right = {h["id"]: h["score"] for h in new["dense_candidates"]}
    c.require(
        len(left) == len(right) == 30 and left.keys() != right.keys(),
        "Unreviewed dense change",
    )
    c.require(
        all(left[k] == right[k] for k in left.keys() & right.keys()),
        "Shared candidate score changed",
    )
    c.require(
        old["inserted_ids_delivered"] == new["inserted_ids_delivered"] == [],
        "Correction delivery changed",
    )
    c.require(
        old["delivered_gates"] == new["delivered_gates"] == [],
        "Correction polarity changed",
    )
    return {
        "changed_fields": fields,
        "same_messages": row["same_messages"],
        "same_retrieved_ids": row["same_retrieved_ids"],
        "bm25_exact": True,
        "shared_candidate_scores_exact": True,
        "correction_delivery_changed": False,
        "removed_candidates": [
            {"id": k, "score": left[k]} for k in sorted(left.keys() - right.keys())
        ],
        "added_candidates": [
            {"id": k, "score": right[k]} for k in sorted(right.keys() - left.keys())
        ],
    }


def check_chunks(folder, roster, values, stamp, receipt):
    chunks = sorted(folder.glob("chunk_*.json"))
    c.require(
        [p.name for p in chunks]
        == [f"chunk_{i:06d}.json" for i in range(0, len(roster), 25)],
        "Incomplete chunk roster",
    )
    c.require(
        {p.name for p in folder.iterdir()} == {p.name for p in chunks},
        "Unexpected chunk files",
    )
    combined = []
    for start, path in zip(range(0, len(roster), 25), chunks):
        chunk = c.read(path)
        keys = [r["key"] for r in roster[start : start + 25]]
        c.require(
            chunk["stamp"] == stamp and chunk["keys"] == keys,
            "Chunk attribution differs",
        )
        c.require(
            [r["key"] for r in chunk["rows"]] == keys
            and c.sig(chunk["rows"]) == chunk["rows_sha256"],
            "Chunk content differs",
        )
        combined.extend(chunk["rows"])
    c.require(
        combined == values and receipt["checkpoint"]["chunks"] == len(chunks),
        "Combined file differs from durable chunks",
    )
    c.require(
        receipt["checkpoint"]["hashes"] == c.bindings(chunks), "Chunk inventory differs"
    )


def inspect():
    c.require(
        not (c.BASE / "execution/live").exists(), "Review must precede live execution"
    )
    plan = prep.validate()
    strict_path = c.BASE / "independent_verification.json"
    strict = c.read(strict_path)
    c.require(
        strict["status"] == "cold_replay_requires_review" and strict["pilot"] is False,
        "Expected preserved strict stop",
    )
    c.require(
        strict["plan_sha256"] == c.sha(prep.PLAN), "Strict report has different plan"
    )
    c.require(
        strict["all_prompt_rrf_checks"] == 12825
        and strict["dense_cosine_checks"] == 384750
        and strict["sentinel_bm25_checks"] == 228,
        "Strict content checks incomplete",
    )
    c.verify(strict["source_hashes"])
    c.verify(strict["inputs"])
    science = c.read(prep.SCIENCE)
    roster = c.rows(c.BASE / "roster.jsonl")
    keys = [r["key"] for r in roster]
    c.require(
        len(keys) == len(set(keys)) == 675 and len(science["conditions"]) == 19,
        "Wrong matrix",
    )
    by_key = {r["key"]: r for r in roster}
    bases = {
        r["key"]: r
        for r in c.rows(c.TAXONOMY / "prompts.jsonl")
        if r["condition"] == "rag_full"
    }
    full = c.rows(c.CORE / "kb/full/records.jsonl")
    fullv = np.load(
        c.BASE / "index_vectors/full.npy", mmap_mode="r", allow_pickle=False
    )
    sourcev = np.load(
        c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    c.require(
        fullv.shape == sourcev.shape == (len(full), 1024), "Full-vector shape differs"
    )
    max_delta = 0.0
    for start in range(0, len(full), 512):
        delta = float(
            np.max(np.abs(fullv[start : start + 512] - sourcev[start : start + 512]))
        )
        max_delta = max(max_delta, delta)
        c.require(delta < 2e-7, "Indexed/cache vectors disagree")
    query_rows = c.rows(c.CORE / "dev_inputs.jsonl")
    qi = {r["key"]: i for i, r in enumerate(query_rows)}
    queryv = np.load(
        c.CORE / "retrieval_v1/query_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    selection = c.read(c.SELECTION / "gate_only_records.json")
    pools = {
        p["condition"]: p["ids"]
        for p in c.read(c.SELECTION / "actual_pools.json")["pools"]
    }
    source_rows = c.rows(c.CANDIDATES / "roster.jsonl")
    si = {r["id"]: i for i, r in enumerate(source_rows)}
    insertion_sourcev = np.load(
        c.CANDIDATES / "query_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    token = c.old.old.tokenizer()
    changes = []
    totals = Counter()
    replay_totals = {}
    outputs = {
        **strict["inputs"],
        **c.bindings([strict_path, prep.PLAN, prep.SCIENCE, c.BASE / "roster.jsonl"]),
    }
    for condition in science["conditions"]:
        root = c.BASE / "retrieval" / condition
        records_path = root / "records.jsonl"
        records = c.rows(records_path)
        ids = [] if condition == "reference" else pools[condition]
        added = [c.ordinary(selection[i]) for i in ids]
        c.require(records == full + added, "Source pool/text/annotation changed")
        original_prefix = c.rel(c.CORE / "retrieval_v1/full/chroma") + "/"
        copy_receipt = c.read(root / "copy.json")
        c.require(
            copy_receipt["source"]
            == {
                n[len(original_prefix) :]: h
                for n, h in plan["original_index_hashes"].items()
            }
            and copy_receipt["plan_sha256"] == c.sha(prep.PLAN)
            and copy_receipt["originals_opened_with_chroma"] is False,
            "Index-copy provenance changed",
        )
        identity = c.read(root / "index_identity.json")
        c.require(
            identity["metadata"]
            == prep.metadata_identity(
                c.read(c.CORE / "retrieval_v1/full/origin.json"),
                c.sha(records_path),
                c.sha(prep.PLAN),
                condition,
            )
            and identity["condition"] == condition
            and identity["records_sha256"] == c.sha(records_path)
            and identity["schema_sha256"] == c.sig(prep.persisted_schema(root)),
            "Owned index identity/schema changed",
        )
        c.require(
            not {r["text"] for r in added}.intersection(r["text"] for r in roster),
            "Insertion/evaluation leakage",
        )
        inserted = {r["id"] for r in added}
        known = {r["id"]: r for r in records}
        iv = np.load(
            root / "inserted_index_vectors.npy", mmap_mode="r", allow_pickle=False
        )
        c.require(iv.shape == (len(ids), 1024), "Insertion-vector shape differs")
        if ids:
            delta = float(
                np.max(np.abs(iv - np.asarray([insertion_sourcev[si[i]] for i in ids])))
            )
            c.require(delta < 2e-7, "Inserted source vectors disagree")
        vectors = {r["id"]: fullv[i] for i, r in enumerate(full)}
        vectors.update({r["id"]: iv[i] for i, r in enumerate(added)})
        prompts = c.rows(root / "prompts.jsonl")
        c.require([r["key"] for r in prompts] == keys, "First-pass order differs")
        first = {r["key"]: r for r in prompts}
        for row in prompts:
            c.require(
                row["condition"] == condition
                and row["scientific_plan_sha256"] == c.sha(prep.SCIENCE),
                "Prompt condition changed",
            )
            c.require(
                query_rows[qi[row["key"]]]["text"] == row["text"],
                "Query vector attribution differs",
            )
            check.check_prompt(
                row, by_key[row["key"]], bases[row["key"]], known, inserted, token
            )
        replay_totals[condition] = {}
        for mode in ["freeze", "replay1", "replay2"]:
            receipt_path = root / mode / "complete.json"
            receipt = c.read(receipt_path)
            stamp = {
                "condition": condition,
                "mode": mode,
                "pilot": False,
                "plan_sha256": c.sha(prep.PLAN),
            }
            c.require(
                all(receipt[k] == v for k, v in stamp.items())
                and receipt["rows"] == 675,
                "Wrong pass receipt",
            )
            c.require(
                receipt["Qwen_calls"] == receipt["new_embeddings"] == 0
                and receipt["vector_max_abs_difference"] < 2e-7,
                "Pass contract differs",
            )
            c.verify(receipt["outputs"])
            outputs.update(receipt["outputs"])
            outputs.update(
                c.bindings(
                    [
                        receipt_path,
                        records_path,
                        root / "copy.json",
                        root / "index_identity.json",
                    ]
                )
            )
            values = (
                prompts
                if mode == "freeze"
                else c.rows(root / mode / "comparisons.jsonl")
            )
            c.require([r["key"] for r in values] == keys, "Pass roster differs")
            check_chunks(root / mode / "chunks", roster, values, stamp, receipt)
            if mode == "freeze":
                continue
            counts = Counter(entry=0, messages=0, retrieved_ids=0)
            for row in values:
                old = first[row["key"]]
                details = assess_change(old, row)
                counts["entry"] += details is not None
                counts["messages"] += not row["same_messages"]
                counts["retrieved_ids"] += not row["same_retrieved_ids"]
                if details is None:
                    continue
                new = row["actual_entry"]
                check.check_prompt(
                    new, by_key[row["key"]], bases[row["key"]], known, inserted, token
                )
                q = np.asarray(queryv[qi[row["key"]]], dtype=np.float64)
                for entry in [old, new]:
                    candidates = entry["dense_candidates"]
                    c.require(
                        [h["id"] for h in entry["channels"]["dense"]]
                        == check.mmr_ids(q, candidates, vectors),
                        "MMR reconstruction changed",
                    )
                    c.require(
                        all(
                            candidates[i]["score"] >= candidates[i + 1]["score"] - 1e-7
                            for i in range(29)
                        ),
                        "Unordered candidates",
                    )
                    for hit in candidates:
                        c.require(
                            known[hit["id"]]["kind"] == "example"
                            and known[hit["id"]]["lang"] == "en",
                            "Dense source filter differs",
                        )
                        v = np.asarray(vectors[hit["id"]], dtype=np.float64)
                        cosine = float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v)))
                        c.require(
                            abs(cosine - hit["score"]) < 1e-4,
                            "Dense score/source vector differs",
                        )
                changes.append(
                    {
                        "condition": condition,
                        "mode": mode,
                        "key": row["key"],
                        "population": by_key[row["key"]]["population"],
                        **details,
                    }
                )
            c.require(
                dict(counts)
                == receipt["differences"]
                == strict["cold_replays"][condition][mode],
                "Replay counts changed",
            )
            replay_totals[condition][mode] = dict(counts)
            totals.update(counts)
        print(
            f"{condition}: first prompts, durable chunks and changed retrieval traces verified.",
            flush=True,
        )
    c.require(
        dict(totals) == {"entry": 81, "messages": 46, "retrieved_ids": 46},
        "Unreviewed discrepancy totals",
    )
    changed_keys = sorted(
        {
            r["key"]
            for r in changes
            if not r["same_messages"] or not r["same_retrieved_ids"]
        }
    )
    c.require(
        changed_keys == strict["changed_identity_union"] == EXPECTED_KEYS,
        "Unreviewed discrepancy identities",
    )
    c.verify(plan["original_index_hashes"])
    c.jsonlines(REVIEW / "changes.jsonl", changes)
    sensitivity = {
        "status": "frozen_before_A14_model_outcomes",
        "excluded_keys": changed_keys,
        "policy": "All 675 inputs remain in primary results. Also exclude the same six input identities from every condition/repeat, never condition-specific deletions.",
        "remaining_by_population": dict(
            Counter(r["population"] for r in roster if r["key"] not in changed_keys)
        ),
        "retain": "Original endpoint label universes, common-valid masks, cluster bootstrap, seeds, intervals and null bounds.",
        "scope": "Repeat both co-primary contrasts and supporting gate/type contrasts; not a bound on unseen retrieval changes or a replay-outcome estimate.",
        "scientific_plan_sha256": c.sha(prep.SCIENCE),
        "strict_verification_sha256": c.sha(strict_path),
    }
    c.save(SENSITIVITY, sensitivity)
    report = {
        "status": "inspected_for_explicit_qualification",
        "cold_replay_exact": False,
        "first_pass_authoritative": True,
        "plan_sha256": c.sha(prep.PLAN),
        "conditions": 19,
        "first_pass_prompts": 12825,
        "replay_comparisons": 25650,
        "durable_chunks": 1539,
        "totals": dict(totals),
        "entry_changed_identity_union": sorted({r["key"] for r in changes}),
        "changed_identity_union": changed_keys,
        "changed_population_counts": dict(
            Counter(by_key[k]["population"] for k in changed_keys)
        ),
        "changed_condition_key_pairs": len(
            {(r["condition"], r["key"]) for r in changes if not r["same_messages"]}
        ),
        "replays": replay_totals,
        "indexed_vs_cached_max_abs_difference": max_delta,
        "observed_mechanism": "Every changed entry has a different dense candidate set, identical scores for shared candidates, and identical BM25 traces. Independent MMR, RRF, source-label, prompt and token checks pass. Correction delivery is empty in all changed entries. Unique internal Chroma cause is not established.",
        "Qwen_calls": 0,
        "CUDA_used": False,
        "original_index_hashes": plan["original_index_hashes"],
        "source_hashes": c.bindings(sources()),
        "inputs": outputs,
        "outputs": c.bindings([REVIEW / "changes.jsonl", SENSITIVITY]),
    }
    c.save(REPORT, report)
    print(
        "Offline discrepancy inspection complete. Explicit qualification/test publication still required.",
        flush=True,
    )


def test():
    from tests.feedback.insertion import test_replay_review as tests

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    c.require(result.wasSuccessful(), "Replay recovery tests failed")
    c.save(
        TESTS,
        {
            "passed": True,
            "tests": result.testsRun,
            "source_hashes": c.bindings(sources()),
            "Qwen_calls": 0,
        },
    )


def verify_review_files():
    report = c.read(REPORT)
    tests = c.read(TESTS)
    c.require(
        report["status"] == "inspected_for_explicit_qualification"
        and report["cold_replay_exact"] is False,
        "Wrong review status",
    )
    c.require(
        report["changed_identity_union"] == EXPECTED_KEYS and tests["passed"],
        "Review/tests incomplete",
    )
    for hashes in [
        report["inputs"],
        report["outputs"],
        report["source_hashes"],
        tests["source_hashes"],
        report["original_index_hashes"],
    ]:
        c.verify(hashes)
    c.require(report["plan_sha256"] == c.sha(prep.PLAN), "Different retrieval plan")
    return report


def publish():
    c.require(
        not (c.BASE / "execution/live").exists(), "Cannot amend a live experiment"
    )
    prep.validate()
    report = verify_review_files()
    c.save(
        DECISION,
        {
            "status": "accepted_frozen_first_pass_with_retrieval_qualification",
            "cold_replay_exact": False,
            "reason": "All content/ranking/source checks passed; use the originally prespecified first snapshot and report observed non-exact cold retrieval. Do not choose a replay.",
            "mandatory_sensitivity": c.rel(SENSITIVITY),
            "sensitivity_sha256": c.sha(SENSITIVITY),
            "inspection_sha256": c.sha(REPORT),
            "tests_sha256": c.sha(TESTS),
            "amendment_sha256": c.sha(AMENDMENT),
            "outcomes_available_at_decision": False,
            "original_strict_checker_modified": False,
            "Qwen_calls": 0,
        },
    )
    prompts = [
        row
        for condition in c.read(prep.SCIENCE)["conditions"]
        for row in c.rows(c.BASE / "retrieval" / condition / "prompts.jsonl")
    ]
    c.require(len(prompts) == 12825, "Prompt matrix incomplete")
    c.jsonlines(c.BASE / "prompts.jsonl", prompts)
    outputs = {
        **report["inputs"],
        **report["outputs"],
        **report["source_hashes"],
        **c.bindings(
            [
                REPORT,
                TESTS,
                DECISION,
                c.BASE / "roster.jsonl",
                c.BASE / "prompts.jsonl",
                prep.SCIENCE,
                c.BASE / "STUDY_PROTOCOL.md",
            ]
        ),
    }
    c.save(
        c.BASE / "prepared.json",
        {
            "status": "independently_checked_with_reviewed_cold_replay_differences",
            "cold_replay_exact": False,
            "qualification": c.rel(DECISION),
            "outputs": outputs,
        },
    )
    c.save(
        c.BASE / "complete.json",
        {
            "status": "offline_preparation_independently_verified",
            "qualification": "Non-exact cold retrieval explicitly reviewed; original first-pass prompts retained.",
            "cold_replay_exact": False,
            "prepared_sha256": c.sha(c.BASE / "prepared.json"),
            "verification": c.rel(DECISION),
            "verification_sha256": c.sha(DECISION),
            "new_model_calls": 0,
        },
    )
    verify()
    print(
        "Qualified first-pass preparation published. Original strict stop and every replay preserved.",
        flush=True,
    )


def verify():
    prep.validate()
    verify_review_files()
    d = c.read(DECISION)
    prepared = c.read(c.BASE / "prepared.json")
    done = c.read(c.BASE / "complete.json")
    c.require(
        d["status"] == "accepted_frozen_first_pass_with_retrieval_qualification"
        and d["cold_replay_exact"] is False,
        "Qualification required",
    )
    for path, field in [
        (REPORT, "inspection_sha256"),
        (TESTS, "tests_sha256"),
        (AMENDMENT, "amendment_sha256"),
        (SENSITIVITY, "sensitivity_sha256"),
    ]:
        c.require(c.sha(path) == d[field], "Qualification evidence changed")
    c.require(
        done["prepared_sha256"] == c.sha(c.BASE / "prepared.json")
        and done["verification_sha256"] == c.sha(DECISION),
        "Published receipts changed",
    )
    c.require(
        prepared["cold_replay_exact"] is False and done["cold_replay_exact"] is False,
        "Qualification lost",
    )
    c.verify(prepared["outputs"])
    print(
        "Reviewed first-pass prompts and required common-exclusion sensitivity verified.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inspect", "test", "publish", "verify"])
    args = parser.parse_args()
    os.chdir(c.ROOT)
    with c.offline():
        with c.a.item_lock(c.BASE / "preparation.lock"):
            {"inspect": inspect, "test": test, "publish": publish, "verify": verify}[
                args.mode
            ]()


if __name__ == "__main__":
    main()
