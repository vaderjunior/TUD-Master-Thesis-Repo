from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
from contextlib import redirect_stdout
import ctypes
import gc
import hashlib
import io
import itertools
import json
import msvcrt
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = PROJECT_ROOT
OUT = ROOT / "experiments/astra/sq3_transfer_launch_check_20261020_v1"
BASE = ROOT / "experiments/astra/sq3_transfer_20261012_v1"
EXPECTED = {"batches": 22, "ordinary_calls": 567288, "predictions": 189096}


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def physical(path):
    path = Path(str(path).removeprefix("\\\\?\\")).resolve()
    return Path("\\\\?\\" + str(path)) if os.name == "nt" else path


def rel(path):
    return (
        Path(str(path).removeprefix("\\\\?\\")).resolve().relative_to(ROOT).as_posix()
    )


def read(path):
    return json.loads(physical(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with physical(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def new(name, value):
    path = OUT / name
    need(path.resolve().is_relative_to(OUT.resolve()), "Output outside this check")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        value
        if isinstance(value, str)
        else json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def bind(target, mapping):
    for name, expected in mapping.items():
        need(
            name not in target or target[name] == expected,
            "Conflicting binding: " + name,
        )
        target[name] = expected


def verify(mapping):
    for name, expected in mapping.items():
        need(sha(ROOT / name) == expected, "Frozen evidence changed: " + name)


def inventory(path):
    return {rel(p): sha(p) for p in physical(path).rglob("*") if p.is_file()}


def tests_and_guards(c, q, api, b):

    from tests.feedback.common import test_queue as queue_tests
    from tests.feedback.summary import test_balanced as engine_tests

    selected = [
        "test_full_schema_exactly_matches_production",
        "test_completed_item_restart_zero_requests",
        "test_ambiguous_interrupted_intent_never_resends",
        "test_saved_response_before_parser_resumes_two_remaining_calls",
        "test_retry_and_repair_budgets_preserved",
        "test_exhausted_retry_does_not_get_new_budget",
        "test_invalid_outputs_are_completed_nulls",
        "test_truncation_and_wrong_model_stop",
        "test_changed_payload_is_rejected",
        "test_live_transport_cannot_be_marked_mock",
        "test_completed_stage_restart_never_constructs_client",
        "test_stage_restart_blocks_ambiguous_request_before_client_creation",
        "test_clean_stage_interruption_resumes_only_two_unsent_calls",
        "test_completed_stage_summary_change_stops_before_client",
        "test_stage_audit_rejects_duplicate_response_ids",
        "test_failed_audit_prevents_next_stage",
    ]
    suite = unittest.defaultTestLoader.loadTestsFromModule(queue_tests)
    for name in selected:
        suite.addTest(engine_tests.CombinedTests(name))
    stdout, stream = (io.StringIO(), io.StringIO())
    with patch.object(c, "SHARED", OUT / "fixtures"):
        fixture = q.queue(
            c.SHARED / "engine",
            "fixture",
            [],
            {"inputs": 1, "conditions": 1, "batches": 1, "predictions": 3},
        )
        with patch.object(engine_tests, "b", fixture), patch.object(
            engine_tests, "audit", fixture.audit
        ), patch.object(
            engine_tests, "TEST_ROOT", fixture.BASE / "offline_check" / "behavior"
        ), redirect_stdout(
            stdout
        ):
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    new("focused_tests.txt", stream.getvalue())
    new("synthetic_execution.txt", stdout.getvalue())
    need(
        result.wasSuccessful() and result.testsRun == 25, "Focused offline test failure"
    )
    with patch.object(sys, "argv", ["transfer", "run-all"]), patch.object(
        q, "require_a15_finished", side_effect=RuntimeError("predecessor incomplete")
    ), patch.object(
        b.a, "item_lock", side_effect=AssertionError("lock reached")
    ) as lock, patch.object(
        q, "execute", side_effect=AssertionError("execution reached")
    ) as execution:
        try:
            api.main()
        except RuntimeError as exc:
            need(str(exc) == "predecessor incomplete", "Wrong prerequisite failure")
        else:
            raise AssertionError("Incomplete predecessor accepted")
        need(
            lock.call_count == execution.call_count == 0,
            "Prerequisite did not stop before locks",
        )
    with patch.object(q, "require_a15_finished", return_value=None), patch.object(
        b, "validate", side_effect=RuntimeError("frozen mismatch")
    ), patch.object(
        c, "resources", side_effect=AssertionError("resources reached")
    ) as resources, patch.object(
        b, "execute_stage", side_effect=AssertionError("stage reached")
    ) as stage:
        try:
            q.execute(b)
        except RuntimeError as exc:
            need(str(exc) == "frozen mismatch", "Wrong validation failure")
        else:
            raise AssertionError("Invalid frozen state accepted")
        need(resources.call_count == stage.call_count == 0, "Validation gate too late")
    return {
        "focused_tests_passed": result.testsRun,
        "entrypoint_guard_checks": 2,
        "fixtures": rel(OUT / "fixtures"),
        "actual_API_calls": 0,
    }


def prompt_and_scope_review(c, science):
    roster = list(c.rows(BASE / "roster.jsonl"))
    scope = list(c.rows(c.SHARED / "scope/evaluation.jsonl"))
    need(
        roster == [dict(r, population=r["phase"]) for r in scope],
        "Roster differs from scope",
    )
    bykey = {r["key"]: r for r in roster}
    need(len(bykey) == len(roster) == 15758, "Wrong input roster")
    counts = Counter(((r["population"], r["gate"]) for r in roster))
    need(
        counts
        == {
            ("en_test", False): 8356,
            ("en_test", True): 3674,
            ("en_hatecheck_eval", False): 1165,
            ("en_hatecheck_eval", True): 2563,
        },
        "Population/endpoint denominator changed",
    )
    cluster_counts = {
        phase: len({r["inference_cluster"] for r in roster if r["phase"] == phase})
        for phase in ("en_test", "en_hatecheck_eval")
    }
    need(
        cluster_counts == {"en_test": 12017, "en_hatecheck_eval": 556},
        "Inference clusters changed",
    )
    support = {}
    for phase in ("en_test", "en_hatecheck_eval"):
        support[phase] = {}
        for field in ("hate_types", "target_groups", "severity", "legal"):
            rr = [
                r
                for r in roster
                if r["phase"] == phase
                and r.get(field) is not None
                and (field == "legal" or r["gate"] is True)
            ]
            cnt = Counter()
            for r in rr:
                value = r[field]
                cnt.update(value if isinstance(value, list) else [value])
            support[phase][field] = {
                "rows": len(rr),
                "positive_support": dict(cnt),
                "support_ge_10": sorted((k for k, n in cnt.items() if n >= 10)),
            }
    need(
        support == science["fixed_support"], "Frozen supporting endpoint masks changed"
    )
    sensitivity = c.read(c.SHARED / "scope/sensitivity.json")
    need(
        science["mandatory_replay_keys"] == sensitivity["replay_keys"]
        and len(science["mandatory_replay_keys"]) == 20
        and (set(science["mandatory_replay_keys"]) <= set(bykey)),
        "Replay exclusion changed",
    )
    pools = [
        p
        for p in c.read(c.SELECTION / "actual_pools.json")["pools"]
        if p["selection"] == "error" and p["polarity"] == "balanced"
    ]
    need(
        [p["seed"] for p in pools]
        == science["pool_seeds"]
        == list(range(2026100441, 2026100444)),
        "Fixed source pools changed",
    )
    sources = c.read(c.SELECTION / "gate_only_records.json")
    pool_ids = {p["condition"]: set(p["ids"]) for p in pools}
    for pool in pools:
        need(
            len(pool["ids"]) == len(pool_ids[pool["condition"]]) == 40,
            "Source pool size",
        )
        need(
            Counter((sources[s]["meta"]["gate"] for s in pool["ids"]))
            == {False: 20, True: 20},
            "Balanced source labels changed",
        )
    choices = {(p["key"], p["condition"]): p for p in c.rows(BASE / "choices.jsonl")}
    need(len(choices) == 47274, "Correction choice count")
    base_hashes = {
        p["key"]: (c.sig(p["context"]), c.sig(p["messages"]))
        for p in c.rows(c.LATER / "retrieval_v3/full/prompts.jsonl")
        if p["key"] in bykey
    }
    conditions = ["reference"] + [p["condition"] for p in pools]
    need(conditions == science["conditions"], "Fixed comparison arms changed")
    seen, tokens, delivery = (set(), [], Counter())
    for key, rows in itertools.groupby(
        c.rows(BASE / "prompts.jsonl"), key=lambda p: p["key"]
    ):
        pp = list(rows)
        need(key not in seen and key in bykey, "Duplicate/unknown prompt input")
        seen.add(key)
        need([p["condition"] for p in pp] == conditions, "Prompt arm order/membership")
        reference = pp[0]
        ctx = reference["context"]
        need(
            (c.sig(ctx), c.sig(reference["messages"])) == base_hashes[key],
            "Qualified first-pass reference changed",
        )
        need(
            reference["source_id"] is None
            and len(ctx["definitions"]) == len(ctx["guidelines"]) == 2,
            "Reference abstract/source structure changed",
        )
        need(
            len(ctx["example_groups"]) == 1
            and ctx["example_groups"][0][0] == "Labelled examples"
            and (len(ctx["example_groups"][0][1]) == 5),
            "Reference examples changed",
        )
        for p in pp:
            need(
                p["text"] == bykey[key]["text"]
                and p["item_id"] == bykey[key]["id"]
                and (p["population"] == bykey[key]["population"]),
                "Input/prompt attribution",
            )
            tokens.append(p["reference_input_tokens"])
            if p["condition"] == "reference":
                continue
            cc = p["context"]
            need(
                cc["definitions"] == ctx["definitions"]
                and cc["guidelines"] == ctx["guidelines"]
                and (len(cc["example_groups"]) == 1)
                and (cc["example_groups"][0][0] == "Labelled examples")
                and (len(cc["example_groups"][0][1]) == 5)
                and (cc["example_groups"][0][1][:4] == ctx["example_groups"][0][1][:4]),
                "Correction altered abstract context, example count or ordinary first four",
            )
            sid = p["source_id"]
            need(sid in pool_ids[p["condition"]], "Source outside fixed pool")
            source = sources[sid]
            hit = cc["example_groups"][0][1][4]
            need(
                hit
                == {
                    "id": source["id"],
                    "text": source["text"],
                    "meta": source["meta"],
                    "via": "rrf",
                },
                "Correction label/rendered source differs",
            )
            need(
                all((v is None for k, v in hit["meta"].items() if k != "gate")),
                "Invented fine labels",
            )
            choice = choices[key, p["condition"]]
            need(
                choice["source_id"] == sid
                and choice["source_gate"] is hit["meta"]["gate"]
                and (choice["cosine"] < c.NEAR),
                "Source choice/label/near exclusion",
            )
            delivery[p["condition"] + "/" + str(hit["meta"]["gate"])] += 1
        if len(seen) % 4000 == 0:
            print(
                f"Fresh prompt/context readback: {len(seen)}/15758 texts.", flush=True
            )
    need(seen == set(bykey), "Incomplete prompt/input coverage")
    prior = read(BASE / "independent_verification.json")
    need(
        dict(delivery) == prior["delivery"]
        and min(tokens) == 622
        and (max(tokens) == 1367),
        "Delivery/token summary changed",
    )
    return {
        "inputs": len(roster),
        "prompts": len(tokens),
        "correction_deliveries": len(choices),
        "population_gate_counts": {f"{p}/{g}": n for (p, g), n in counts.items()},
        "inference_clusters": cluster_counts,
        "mandatory_replay_exclusions": 20,
        "fixed_endpoint_support_reconstructed": True,
        "source_pool_seeds": science["pool_seeds"],
        "source_records_per_pool": 40,
        "source_positive_negative_per_pool": [20, 20],
        "first_four_and_abstract_contexts_preserved": True,
        "delivery": dict(delivery),
        "prompt_token_range": [min(tokens), max(tokens)],
        "source_counts": dict(Counter((r["source"] for r in roster))),
        "new_retrieval_or_tokenization_performed": False,
    }


def main():
    need(not (OUT / "before.json").exists(), "Preserve previous launch review")
    from analysis.common import runtime as c

    with c.offline():
        from analysis.feedback.transfer import run as api
        from analysis.feedback.common import queue as q
        import requests

        with patch.object(q.template.core, "live_client", c.forbidden), patch.object(
            q.template.a.RequestsTransport, "__call__", c.forbidden
        ), patch.object(requests.sessions.Session, "request", c.forbidden):
            need(
                not (BASE / "execution/live").exists(),
                "Existing live work: inspect instead of new-launch review",
            )
            need(
                not (BASE / "execution/launcher.lock").exists(),
                "Unexpected transfer launcher state",
            )
            resources_before = c.resources(2.5, 20.0)
            plan = read(BASE / "execution/plan.json")
            prepared = read(BASE / "prepared.json")
            prep = read(BASE / "preparation_plan.json")
            scope_plan = read(c.SHARED / "scope/plan.json")
            scope_done = read(c.SHARED / "scope/complete.json")
            ready = read(BASE / "execution/ready.json")
            rehearsal = read(ROOT / ready["evidence"])
            old_mock_root = ROOT / rehearsal["rehearsal_root"]
            bindings = {}
            for mapping in [
                plan["input_hashes"],
                plan["source_hashes"],
                prep["inputs"],
                prepared["outputs"],
                scope_plan["inputs"],
                scope_plan["prepared"],
                scope_done["outputs"],
            ]:
                bind(bindings, mapping)
            verify(bindings)
            need(
                scope_done["status"] == "independent_exact_scope_verified",
                "Scope verification missing",
            )
            receipt_paths = [
                BASE / n
                for n in [
                    "STUDY_PROTOCOL.md",
                    "scientific_plan_v1.json",
                    "preparation_plan.json",
                    "prepared.json",
                    "complete.json",
                    "independent_verification.json",
                    "execution/plan.json",
                    "execution/ready.json",
                    "execution/development/readiness.json",
                ]
            ] + [
                c.SHARED / "SCOPE_PROTOCOL.md",
                c.SHARED / "scope/plan.json",
                c.SHARED / "scope/complete.json",
                old_mock_root / "complete.json",
                c.A15 / "execution/live/complete.json",
                ROOT
                / "experiments/astra/sq3_reserved_part10_results_20261020_v1/complete.json",
                ROOT
                / "experiments/astra/sq3_human_feedback_20261015_v1/execution_v2/live/complete.json",
                ROOT
                / "experiments/astra/sq3_human_feedback_results_20261020_v1/complete.json",
            ]
            receipts = {rel(p): sha(p) for p in receipt_paths}
            original_study = inventory(BASE)
            new(
                "before.json",
                {
                    "created_utc": c.utc(),
                    "frozen_bindings": bindings,
                    "receipt_bindings": receipts,
                    "original_transfer_files": original_study,
                    "transfer_live_exists": False,
                    "resource_snapshot": resources_before,
                },
            )
            print(
                f"Frozen bindings passed: {len(bindings)} files; saved study inventory: {len(original_study)} files.",
                flush=True,
            )
            b = api.queue()
            q.require_a15_finished()
            predecessor = read(receipt_paths[-3])
            need(
                predecessor["completed_parts"] == list(range(1, 11))
                and predecessor["unresolved_synthesis_discrepancies"] == [],
                "Earlier scientific review incomplete",
            )
            human_live, human_review = (
                read(receipt_paths[-2]),
                read(receipt_paths[-1]),
            )
            human_plan = (
                ROOT
                / "experiments/astra/sq3_human_feedback_20261015_v1/execution_v2/plan.json"
            )
            need(
                human_live["status"] == "all_studies_executed_and_audited"
                and human_live["mode"] == "live"
                and (human_live["plan_sha256"] == sha(human_plan))
                and (human_live["counts"]["predictions"] == 216),
                "Human execution prerequisite incomplete",
            )
            need(
                human_review["status"]
                == "human_feedback_study_complete_independently_reviewed"
                and human_review["unresolved_scientific_discrepancies"] == [],
                "Human review prerequisite incomplete",
            )
            verify(predecessor["artifacts"])
            verify(human_review["artifacts"])
            validated = io.StringIO()
            with redirect_stdout(validated), patch.object(
                sys, "argv", ["transfer", "validate"]
            ):
                api.main()
            new("launcher_validate.txt", validated.getvalue())
            new(
                "launcher_validate_exit.json",
                {"exit_code": 0, "mode": "validate", "actual_API_calls": 0},
            )
            plan_sha = sha(b.PLAN)
            need(
                ready["plan_sha256"] == plan_sha
                and ready["evidence_sha256"] == sha(ROOT / ready["evidence"])
                and (
                    rehearsal["completion_sha256"]
                    == sha(old_mock_root / "complete.json")
                ),
                "Readiness/rehearsal chain changed",
            )
            need(
                rehearsal["matrix"] == EXPECTED
                and rehearsal["mock_predictions"] == 528
                and (rehearsal["mock_replies"] == 1584)
                and (rehearsal["new_calls_on_restart"] == 0)
                and (rehearsal["actual_API_calls"] == 0),
                "Original rehearsal does not match",
            )
            mock = read(old_mock_root / "complete.json")
            need(
                mock["mode"] == "mock"
                and mock["plan_sha256"] == plan_sha
                and (mock["counts"]["predictions"] == 528)
                and (len(mock["stages"]) == 22),
                "Synthetic completion attribution",
            )
            verify(mock["review_hashes"])
            synthetic_bindings = {}
            for stage in mock["stages"]:
                review_path = ROOT / stage["review"]
                rv = read(review_path)
                b.verify_review(
                    {"id": stage["stage"]}, old_mock_root / "transfer", plan_sha, "mock"
                )
                rr = read(review_path.parent / rv["report"])
                for mapping in [
                    read(review_path.parent / rv["raw_hashes"]),
                    rr["attempt_hashes"],
                ]:
                    bind(
                        synthetic_bindings,
                        {
                            rel(old_mock_root / "transfer" / p): h
                            for p, h in mapping.items()
                        },
                    )
            verify(synthetic_bindings)
            print(
                f"Preserved all 22 original rehearsal audits and {len(synthetic_bindings)} raw/history bindings.",
                flush=True,
            )
            tested = tests_and_guards(c, q, api, b)
            print("25 focused tests and two launch-gate probes passed.", flush=True)
            science = read(BASE / "scientific_plan_v1.json")
            need(
                science["primary"]
                == {
                    "bootstrap_draws": 10000,
                    "bootstrap_seeds": list(range(2026101216, 2026101220)),
                    "confidence": 0.9875,
                    "endpoints": [
                        "en_test_FPR",
                        "en_test_FNR",
                        "en_hatecheck_eval_FPR",
                        "en_hatecheck_eval_FNR",
                    ],
                },
                "Frozen four-endpoint inference changed",
            )
            scope = prompt_and_scope_review(c, science)
            cat = b.load_catalog("transfer")
            schedule = read(b.BASE / "schedule.json")
            matrix = q.validate_schedule(b, schedule, {"transfer": cat})
            need(
                matrix == EXPECTED
                and all((plan["budget"][k] == v for k, v in matrix.items())),
                "Execution matrix/budget mismatch",
            )
            need(
                {s["study"] for s in schedule["stages"]} == {"transfer"},
                "Another study in this queue",
            )
            new(
                "batch_inventory.json",
                [
                    {
                        "id": s["id"],
                        "inputs": len(s["input_keys"]),
                        "predictions": s["items"],
                        "ordinary_calls": s["ordinary_calls"],
                    }
                    for s in schedule["stages"]
                ],
            )
            del cat, schedule
            gc.collect()
            lock = c.SHARED / "qwen_queue.lock"
            need(
                lock.is_file(),
                "Expected existing shared queue lock absent; inspect state",
            )
            lock_sha = sha(lock)
            with lock.open("rb") as stream:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            need(sha(lock) == lock_sha, "Shared lock bytes changed")

            class Power(ctypes.Structure):
                _fields_ = [
                    ("ac", ctypes.c_ubyte),
                    ("flag", ctypes.c_ubyte),
                    ("percent", ctypes.c_ubyte),
                    ("reserved", ctypes.c_ubyte),
                    ("life", ctypes.c_ulong),
                    ("full", ctypes.c_ulong),
                ]

            power = Power()
            power_ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power))
            verify(bindings)
            verify(receipts)
            need(
                inventory(BASE) == original_study,
                "Original transfer evidence changed or new files appeared",
            )
            need(
                not (BASE / "execution/live").exists()
                and (not (BASE / "execution/launcher.lock").exists()),
                "Unexpected live/launcher state",
            )
            summary = {
                "status": "offline_launch_checks_passed",
                "checked_utc": c.utc(),
                "execution_plan_sha256": plan_sha,
                "scientific_plan_sha256": sha(BASE / "scientific_plan_v1.json"),
                "frozen_bindings_checked": len(bindings),
                "receipt_bindings_checked": len(receipts),
                "original_transfer_files_preserved": len(original_study),
                "execution_source_files": len(plan["source_hashes"]),
                "execution_input_files": len(plan["input_hashes"]),
                "reused_synthetic_raw_history_bindings": len(synthetic_bindings),
                "reused_mock_predictions": 528,
                "reused_mock_replies": 1584,
                "reused_rehearsal_batches": 22,
                "reused_rehearsal_new_calls_on_restart": 0,
                "tests": tested,
                "both_predecessor_studies_complete_and_reviewed": True,
                "matrix": matrix,
                "scope": scope,
                "primary": science["primary"],
                "shared_lock": {
                    "OS_lock_available_at_check": True,
                    "bytes_unchanged": True,
                    "sha256": lock_sha,
                },
                "resource_snapshot_before": resources_before,
                "resource_snapshot_after": c.resources(2.5, 20.0),
                "power": {
                    "AC_status": int(power.ac) if power_ok else None,
                    "battery_percent": int(power.percent) if power_ok else None,
                },
                "scientific_execution_started": False,
                "new_model_POSTs": 0,
                "new_model_list_GETs": 0,
                "new_training_fits": 0,
                "new_embeddings": 0,
                "new_human_judgments": 0,
                "writes_to_frozen_transfer_study": 0,
                "new_source_policy_selection": False,
                "original_transfer_evidence_unchanged": True,
                "checker_sha256": sha(Path(__file__)),
                "failure_boundary": "Clean saved completion/response resumes without resampling. Unknown intents, terminal stops and exhausted budgets require separate review; do not delete evidence or blindly retry.",
                "writing_preference": "Ajay will request SQ3 writing only after final expanded SQ3 synthesis.",
            }
            new("offline_checks.json", summary)
            print(
                json.dumps(
                    {k: v for k, v in summary.items() if k not in ("scope", "primary")},
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
