from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import io
import unittest
import numpy as np
from analysis.feedback.rounds import robustness_inputs as a
from analysis.feedback.rounds import robustness_metrics as k


def main():
    a.need(not a.OUT.exists(), "Preserve existing Part 7 work")
    latest = a.read(a.PART6 / "final_verification.json")
    a.need(
        latest["status"] == "iterative_part6_complete_with_final_readback"
        and latest["complete_sha256"] == a.sha(a.PART6 / "complete.json"),
        "Part 6 checkpoint required",
    )
    reader = a.read(a.READER / "complete.json")
    a.check(reader["artifacts"])
    proof = a.read(a.READER / "independent_verification.json")
    a.need(
        proof["reader_complete_sha256"] == a.sha(a.READER / "complete.json"),
        "Reader proof changed",
    )
    inputs = [
        a.STUDY / "plan.json",
        a.STUDY / "STUDY_PROTOCOL.md",
        a.STUDY / "complete.json",
        a.STUDY / "evaluation.jsonl",
        a.REVIEW / "review_plan.json",
        a.REVIEW / "part1/readiness.json",
        a.READER / "complete.json",
        a.READER / "independent_verification.json",
        a.READER / "matrix.jsonl",
        a.READER / "duplicate_vote_labels.json",
        a.PART2 / "analysis_contract_v3.json",
        a.PART2 / "errors_v1/results.json",
        a.PART2 / "errors_v1/independent_verification.json",
        a.PART2 / "tables_v1/control_diagnostics.json",
        a.PART3 / "source_cluster_inventory.json",
        a.PART3 / "analysis.json",
        a.PART3 / "selections.json",
        a.PART3 / "candidate_selection_ledger.jsonl",
        a.PART3 / "final_verification.json",
        a.REVIEW / "part4/analysis.json",
        a.REVIEW / "part4/final_verification.json",
        a.REVIEW / "part5/annotation_boundary_review.json",
        a.REVIEW / "part5/final_verification.json",
        a.PART6 / "analysis.json",
        a.PART6 / "analysis_contract.json",
        a.PART6 / "complete.json",
        a.PART6 / "final_verification.json",
    ]
    execution = a.STUDY / "execution"
    for folder, final in [
        ("concurrency_20261024_v1", "live_runtime_complete.json"),
        ("recovery_20261024_v1", "live_recovery_complete.json"),
        ("path_20261024_v2", "live_path_runtime_complete.json"),
        ("ranking_20261025_v3", "live_ranking_runtime_complete.json"),
    ]:
        inputs.extend(
            [execution / folder / "manifest.json", execution / folder / final]
        )
    sources = [
        Path(__file__),
        Path(k.__file__),
        Path(a.__file__),
        a.ROOT / "analysis/feedback/rounds/robustness_summary.py",
        a.ROOT / "tests/feedback/rounds/test_robustness.py",
    ]
    bindings = a.bindings(inputs)
    source_hashes = a.bindings(sources)
    a.write(
        a.OUT / "analysis_contract.json",
        dict(
            status="part7_rules_recorded_before_added_sensitivity_scoring",
            created_utc=a.pure.utc(),
            timing="After generation and Parts 2–6. The original protocol already requested missingness, strict votes, replay, representative and dependence checks. Detailed added views are exploratory, not new confirmatory hypotheses.",
            primary="Reuse all four original final English contrasts, original 98.75% intervals/seeds and original common mask. No new primary family, no tuning or reruns.",
            matrix="2077 texts, four rounds, ten conditions, three decoder repeats; each final output is a three-vote aggregate. Reference has three real outputs, shared across stream contrasts.",
            masks=[
                "final common",
                "all-round common",
                "strict final",
                "strict all-round",
                "replay excluded",
                "representative before validity",
                "no final ties",
                "no final repairs",
                "no final transport retry",
                "three-policy-only common",
                "adaptive versus each control pair only",
            ],
            exclusions="All final masks require common non-null outputs unless explicitly policy-only/pair-only. No-repair and strict views select easier/different texts and are diagnostic, not alternate classifiers.",
            representative="One hash-first key per original population/gold/normalized-text cluster before validity; A17-part7-representative|key. Keep opposite-gold members separate. Do not substitute a valid member.",
            inference="10,000 PCG64 draws, nominal 95% supporting intervals; preserve original inference components. Text-weighted baseline; cluster-equal is a different estimand; row bootstrap intentionally ignores dependence and is a diagnostic only.",
            sampling="Average decisions within text first. New views have deterministic SHA256 first-eight little-endian seeds A17-part7-robustness-v1|record. Store every draw. Report undefined cohorts explicitly.",
            influence="Every leave-one-inference-cluster, whole-source, HateCheck-function and execution-batch omission on final/strict common masks, descriptive points only. Every stream/repeat omission on fixed texts; no independent-trial sign test.",
            bounds="Full-roster null-answer opposing error bounds reproduce Part 2. Also hypothetically fill only invalid individual vote slots with arbitrary binary values, keeping surviving votes fixed, and bound a complete three-vote majority. Neither is a confidence interval, a rerun or a replacement estimator.",
            chronology="Check actual effective workers on all 259224 bound compact outputs, completion timing, repair/failure composition, all phase/stage counts and saved model aliases. These are observational; no remote checkpoint or concurrency-causal claim.",
            references="Keep fresh within-round references. Compare absolute, contemporaneous-adjusted and fixed-first-reference changes on the same all-round texts. Round-one current/initial prompts coincide; retain this variability control.",
            detailed_quality="Reuse the independently verified Part 6 sensitivity estimates, uncertainty and known-gold limits; never impute fine labels or use unknown fields as errors.",
            preservation="New directory and code only. Reuse prior raw audits without repeating HTTP replay. No API, embeddings, fits or writer-owned edits.",
            input_bindings=bindings,
            source_hashes=source_hashes,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.feedback.rounds.test_robustness"
    )
    log = io.StringIO()
    test = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    a.write(
        a.OUT / "tests.json",
        dict(tests_run=test.testsRun, passed=test.wasSuccessful(), log=log.getvalue()),
    )
    a.need(test.wasSuccessful(), "Focused fixtures failed")
    matrix = list(a.rows(a.READER / "matrix.jsonl"))
    index = {m["key"]: i for i, m in enumerate(matrix)}
    N = len(matrix)
    a.need(N == 2077, "Matrix size")
    arrays = {
        name: np.zeros((N, 4, 10, 3), dtype=dtype)
        for name, dtype in [
            ("seen", "uint8"),
            ("repairs", "int16"),
            ("retries", "int16"),
            ("parse_failures", "int16"),
            ("ties", "bool"),
            ("workers", "int16"),
            ("completed_unix", "float64"),
            ("stage", "int8"),
        ]
    }
    arrays["vote_gates"] = np.full((N, 4, 10, 3, 3), -1, dtype=np.int8)
    totals = Counter()
    workers = Counter()
    models = Counter()
    finishes = Counter()
    classes = Counter()
    accounting = Counter()
    nulls = []
    partials = []
    groups = {}
    stages = []

    def accumulate(name, r):
        if name not in groups:
            groups[name] = dict(
                outputs=0,
                votes=Counter(),
                repairs=0,
                parse_failures=0,
                ties=0,
                gate_normalised=0,
                retries=0,
                reported_models=Counter(),
                finish_reasons=Counter(),
                response_classes=Counter(),
                first_unix=None,
                last_unix=None,
            )
        g = groups[name]
        g["outputs"] += 1
        g["votes"][str(r["n_valid"])] += 1
        for field in ["repairs", "parse_failures", "gate_normalised"]:
            g[field] += r[field]
        g["ties"] += int(r["uncertain"])
        g["retries"] += r["accounting"]["retry_attempts"]
        for f in ["reported_models", "finish_reasons", "response_classes"]:
            g[f].update(r[f])
        t = datetime.fromisoformat(r["item_completed_utc"]).timestamp()
        g["first_unix"] = t if g["first_unix"] is None else min(g["first_unix"], t)
        g["last_unix"] = t if g["last_unix"] is None else max(g["last_unix"], t)

    for st in reader["stage_receipts"]:
        receipt = a.read(a.ROOT / st["receipt"])
        a.need(
            a.sha(a.ROOT / st["receipt"]) == st["receipt_sha256"],
            "Stage receipt changed",
        )
        path = a.ROOT / st["predictions"]
        a.need(a.sha(path) == receipt["predictions_sha256"], "Compact output changed")
        bindings.update(a.bindings([a.ROOT / st["receipt"], path]))
        local = Counter({key: 0 for key in st["counts"]})
        local_workers = Counter()
        first = None
        last = None
        for r in a.rows(path):
            local["predictions"] += 1
            local["nulls"] += int(r["result"] is None)
            local["valid_votes_" + str(r["n_valid"])] += 1
            for f in ["repairs", "parse_failures", "normalisations", "gate_normalised"]:
                local[f] += r[f]
            local["gate_ties"] += int(r["uncertain"])
            local_workers[str(r["effective_worker_limit"])] += 1
            workers[str(r["effective_worker_limit"])] += 1
            models.update(r["reported_models"])
            finishes.update(r["finish_reasons"])
            classes.update(r["response_classes"])
            accounting.update(r["accounting"])
            t = datetime.fromisoformat(r["item_completed_utc"]).timestamp()
            first = t if first is None else min(first, t)
            last = t if last is None else max(last, t)
            a.need(
                len(r["votes"]) == 3
                and sum(v is not None for v in r["votes"]) == r["n_valid"],
                "Saved vote validity differs",
            )
            compact = {
                f: r[f]
                for f in [
                    "key",
                    "phase",
                    "stage",
                    "role",
                    "round",
                    "condition",
                    "repeat",
                    "n_valid",
                    "uncertain",
                    "item_completed_utc",
                    "item_receipt",
                    "item_receipt_sha256",
                    "result_sha256",
                    "repairs",
                    "parse_failures",
                    "finish_reasons",
                    "response_classes",
                ]
            }
            if r["n_valid"] == 0:
                nulls.append(compact)
            elif r["n_valid"] < 3:
                partials.append(compact)
            groupnames = [
                "all",
                "role/" + r["role"],
                "phase/" + r["phase"],
                "stage/" + r["phase"] + "/" + r["stage"],
            ]
            if r["role"] == "evaluation":
                i = index[r["key"]]
                rr = r["round"] - 1
                c = a.CONDITIONS.index(r["condition"])
                d = r["repeat"]
                pos = (i, rr, c, d)
                a.need(arrays["seen"][pos] == 0, "Duplicate execution slot")
                arrays["seen"][pos] = 1
                original = matrix[i]["predictions"][rr][c][d]
                a.need(
                    all(
                        original[f] == r[f] for f in ["result", "n_valid", "uncertain"]
                    ),
                    "Execution-matrix join differs",
                )
                for f in ["repairs", "parse_failures"]:
                    arrays[f][pos] = r[f]
                arrays["retries"][pos] = r["accounting"]["retry_attempts"]
                arrays["ties"][pos] = r["uncertain"]
                arrays["workers"][pos] = r["effective_worker_limit"]
                arrays["completed_unix"][pos] = t
                arrays["stage"][pos] = int(r["stage"].rsplit("_", 1)[1])
                arrays["vote_gates"][pos] = [
                    -1 if v is None else int(v["hate"]) for v in r["votes"]
                ]
                m = matrix[i]
                groupnames += [
                    "population/" + m["population"],
                    "round/" + str(r["round"]) + "/" + m["population"],
                    "source/" + m["source"] + "/gold_" + str(int(m["gold_hate"])),
                    "condition/" + r["condition"],
                    "repeat/" + str(d),
                ]
            for name in groupnames:
                accumulate(name, r)
        a.need(
            dict(local) == st["counts"],
            "Stage totals differ: " + st["phase"] + "/" + st["stage"],
        )
        totals.update(local)
        stages.append(
            dict(
                phase=st["phase"],
                stage=st["stage"],
                counts=dict(local),
                worker_counts=dict(local_workers),
                first_completion_unix=first,
                last_completion_unix=last,
            )
        )
        print("Execution checked: " + st["phase"] + "/" + st["stage"], flush=True)
    a.need(
        dict(totals) == reader["counts"] and dict(accounting) == reader["accounting"],
        "Whole-study count/accounting mismatch",
    )
    a.need(
        (arrays["seen"] == 1).all() and dict(workers) == {"128": 259224},
        "Missing slot or different actual concurrency",
    )
    a.npz(a.OUT / "execution_arrays.npz", **arrays)
    a.lines(a.OUT / "null_outputs.jsonl", nulls)
    a.lines(a.OUT / "partial_vote_outputs.jsonl", partials)
    a.lines(
        a.OUT / "metadata.jsonl",
        ({k: v for k, v in m.items() if k != "predictions"} for m in matrix),
    )
    runtime = a.read(execution / "concurrency_20261024_v1/live_runtime_complete.json")
    a.check(runtime["marker_hashes"])
    bindings.update(runtime["marker_hashes"])
    a.write(
        a.OUT / "execution.json",
        dict(
            status="all_bound_output_execution_checked",
            created_utc=a.pure.utc(),
            totals=totals,
            worker_counts=workers,
            reported_models=models,
            finish_reasons=finishes,
            response_classes=classes,
            accounting=accounting,
            groups=groups,
            stages=stages,
            actual_concurrency_eras=1,
            raw_audits_reused=True,
            unknown_remote_work="Preserved initial outage failures do not reveal whether timed-out requests ran remotely. The same model alias does not prove a constant backend checkpoint.",
            new_Qwen_calls=0,
        ),
    )
    a.write(
        a.OUT / "assembled.json",
        dict(
            status="part7_execution_assembled",
            created_utc=a.pure.utc(),
            input_bindings=bindings,
            source_hashes=source_hashes,
            artifacts=a.bindings(
                [
                    a.OUT / f
                    for f in [
                        "analysis_contract.json",
                        "tests.json",
                        "metadata.jsonl",
                        "execution_arrays.npz",
                        "execution.json",
                        "null_outputs.jsonl",
                        "partial_vote_outputs.jsonl",
                    ]
                ]
            ),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print("Execution assembly saved; no live or original evidence changed.", flush=True)


if __name__ == "__main__":
    with a.pure.offline():
        main()
