import os

for _n in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_n] = "1"
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import time
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.common import runtime as c
from analysis.feedback.common import queue as q
from analysis.feedback.rounds import policy as policy
from analysis.common import prompt as renderer
from analysis.common import feedback_inputs as tokens
from analysis.feedback.reserved.common import intervene
from analysis.feedback.source.retrieval import FeedbackBucket
from analysis.feedback.explanations.calibration import LocalBucket

BASE = c.ITERATIVE
SCOPE = c.SHARED / "scope"
SOURCES = [
    "analysis/feedback/rounds/run.py",
    "analysis/feedback/rounds/policy.py",
    "tests/feedback/rounds/test_policy.py",
    "scripts/run_feedback_rounds.ps1",
    "tests/feedback/rounds/test_sequence.py",
    "analysis/feedback/reserved/common.py",
    "analysis/feedback/source/retrieval.py",
    "analysis/feedback/explanations/calibration.py",
    "analysis/common/prompt.py",
    "analysis/common/feedback_inputs.py",
    "src/hsrag/prompt.py",
    "prompts/classify_v1.txt",
]


def validate():
    plan = c.read(BASE / "plan.json")
    c.verify(plan["inputs"])
    c.verify(plan["outputs"])
    c.verify(plan["source_hashes"])
    c.require(
        plan["environment"] == q.template.core.environment(), "A17 environment changed"
    )
    c.active_guard()
    return plan


def prepare():
    if (BASE / "plan.json").exists():
        validate()
        return
    c.resources(2.0, 3.0)
    c.active_guard()
    scope = c.read(SCOPE / "complete.json")
    c.verify(scope["outputs"])
    parent = c.read(c.CANDIDATES / "complete.json")
    c.require(
        parent["prepared_sha256"] == c.sha(c.CANDIDATES / "prepared.json"),
        "Candidate parent changed",
    )
    old = c.read(c.CANDIDATES / "prepared.json")["outputs"]
    for name in ("roster.jsonl", "prompts.jsonl", "query_vectors.npy"):
        c.require(
            c.sha(c.CANDIDATES / name) == old[c.rel(c.CANDIDATES / name)],
            "Candidate source differs from earlier audited version",
        )
    rows = list(c.rows(c.CANDIDATES / "roster.jsonl"))
    by = {r["id"]: r for r in rows}
    c.require(
        len(rows) == 1024
        and Counter((r["gold"]["hate"] for r in rows))
        == Counter({False: 512, True: 512}),
        "Wrong candidate scope",
    )
    keys = set(c.read(SCOPE / "iteration_evaluation.json")["keys"])
    evalrows = [r for r in c.rows(SCOPE / "evaluation.jsonl") if r["key"] in keys]
    c.require(len(evalrows) == len(keys) == 2077, "Wrong fixed transfer sample")
    blocks = {str(seed): policy.partition(rows, seed) for seed in c.STREAM_SEEDS}
    for seed, parts in blocks.items():
        c.require(
            len(parts) == 4
            and all(
                (
                    len(v) == 256 and sum((by[i]["gold"]["hate"] for i in v)) == 128
                    for v in parts
                )
            ),
            "Wrong source block balance",
        )
    components = c.read(SCOPE / "components.json")
    source_clusters = [
        r for r in components if sum((x.startswith("source|") for x in r["nodes"])) > 1
    ]
    records = {
        r["id"]: {
            "id": "sq3iter-" + r["id"],
            "text": r["text"],
            "lang": "en",
            "kind": "feedback",
            "source": r["source"],
            "meta": {
                "gate": r["gold"]["hate"],
                "hate_types": None,
                "target_groups": None,
                "severity": None,
                "legal": None,
            },
            "provenance": {
                "source_id": r["id"],
                "annotation_scope": "binary gate only; all other heads unannotated",
            },
        }
        for r in rows
    }
    c.save(
        BASE / "source_blocks.json",
        {
            "seeds": list(c.STREAM_SEEDS),
            "blocks": blocks,
            "source_clusters_with_multiple_candidates": source_clusters,
        },
    )
    c.save(BASE / "gate_only_records.json", records)
    c.jsonlines(
        BASE / "evaluation.jsonl", (dict(r, population=r["phase"]) for r in evalrows)
    )
    inputs = c.bindings(
        [
            BASE / "STUDY_PROTOCOL.md",
            SCOPE / "complete.json",
            SCOPE / "iteration_evaluation.json",
            SCOPE / "evaluation.jsonl",
            SCOPE / "vectors.npy",
            SCOPE / "sensitivity.json",
            SCOPE / "components.json",
            c.CANDIDATES / "complete.json",
            c.CANDIDATES / "prepared.json",
            c.CANDIDATES / "roster.jsonl",
            c.CANDIDATES / "prompts.jsonl",
            c.CANDIDATES / "query_vectors.npy",
            c.LATER / "retrieval_v3/full/prompts.jsonl",
            tokens.TOK / "tokenizer.json",
            tokens.TOK / "complete.json",
        ]
    )
    from tests.feedback.rounds import test_policy as tests
    from tests.feedback.rounds import test_sequence as sequence_tests

    suite = unittest.TestSuite(
        [
            unittest.defaultTestLoader.loadTestsFromModule(tests),
            unittest.defaultTestLoader.loadTestsFromModule(sequence_tests),
        ]
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    c.require(result.wasSuccessful(), "A17 policy/sequence tests failed")
    source_hashes = {
        n: c.sha(c.ROOT / n) for n in dict.fromkeys([*q.BASE_SOURCES, *SOURCES])
    }
    c.save(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "inputs": inputs,
            "source_hashes": source_hashes,
            "environment": q.template.core.environment(),
            "outputs": c.bindings(
                (
                    BASE / n
                    for n in [
                        "source_blocks.json",
                        "gate_only_records.json",
                        "evaluation.jsonl",
                    ]
                )
            ),
            "protocol_sha256": c.sha(BASE / "STUDY_PROTOCOL.md"),
            "evaluation_inputs": 2077,
            "candidate_inputs": 1024,
            "rounds": 4,
            "source_seeds": list(c.STREAM_SEEDS),
            "policies": list(policy.POLICIES),
            "primary_confidence": 0.9875,
            "bootstrap_draws": 10000,
            "bootstrap_seeds": list(range(2026101220, 2026101224)),
            "ordinary_calls": 777672,
            "predictions": 259224,
            "policy_tests": result.testsRun,
            "A15_outcomes_read": False,
            "new_model_calls": 0,
            "status": "prospective_dynamic_protocol_frozen",
        },
    )
    validate()
    print(
        "A17 source streams, evaluation sample and update rules frozen; no calls.",
        flush=True,
    )


def load_data():
    candidates = list(c.rows(c.CANDIDATES / "roster.jsonl"))
    indices = {r["id"]: i for i, r in enumerate(candidates)}
    evalrows = list(c.rows(BASE / "evaluation.jsonl"))
    want = {r["key"] for r in evalrows}
    bases = {r["key"]: r for r in c.rows(c.CANDIDATES / "prompts.jsonl")}
    bases.update(
        {
            r["key"]: r
            for r in c.rows(c.LATER / "retrieval_v3/full/prompts.jsonl")
            if r["key"] in want
        }
    )
    return (
        candidates,
        indices,
        evalrows,
        bases,
        c.read(BASE / "gate_only_records.json"),
        np.load(c.CANDIDATES / "query_vectors.npy", mmap_mode="r"),
        np.load(SCOPE / "vectors.npy", mmap_mode="r"),
    )


def make_phase(root, study, rows, pools, dependencies, data):

    c.own(root / "complete.json")
    validate()
    science = {
        "inputs": len(rows),
        "conditions": list(pools),
        "pools": pools,
        "parent_plan_sha256": c.sha(BASE / "plan.json"),
        "source_dependencies": dependencies,
    }
    if (root / "complete.json").exists():
        c.require(
            c.read(root / "scientific_plan_v1.json") == science
            and list(c.rows(root / "roster.jsonl")) == rows,
            "Completed phase does not match requested adaptive state",
        )
        b = q.queue(root, study, SOURCES)
        b.study_inputs(study)
        return b
    c.require(not (root / "execution/live").exists(), "Do not rebuild a live phase")
    _, indices, _, bases, records, sv, ev = data
    conditions = list(pools)
    inputs = {c.rel(BASE / "plan.json"): c.sha(BASE / "plan.json"), **dependencies}
    c.verify(inputs)
    c.write_bytes(root / "STUDY_PROTOCOL.md", (BASE / "STUDY_PROTOCOL.md").read_bytes())
    c.save(
        root / "preparation_plan.json",
        {"inputs": inputs, "parent_plan_sha256": c.sha(BASE / "plan.json")},
    )
    c.save(root / "scientific_plan_v1.json", science)
    science_sha = c.sha(root / "scientific_plan_v1.json")
    c.jsonlines(root / "roster.jsonl", rows)
    buckets = {
        cond: FeedbackBucket([records[i] for i in ids], sv[[indices[i] for i in ids]])
        for cond, ids in pools.items()
        if ids
    }
    independent = {
        cond: LocalBucket([records[i] for i in ids], sv[[indices[i] for i in ids]])
        for cond, ids in pools.items()
        if ids
    }
    tok = tokens.tokenizer()
    checked = 0
    delivery = Counter()

    def generate():
        nonlocal checked
        for row in rows:
            old = bases[row["key"]]
            source = row["id"] in indices
            query = sv[indices[row["id"]]] if source else ev[row["scope_node_index"]]
            c.require(
                old["messages"]
                == renderer.render_messages(
                    row["text"], "en", old["context"], "corrected"
                ),
                "A17 old prompt changed",
            )
            for cond, ids in pools.items():
                ctx = deepcopy(old["context"])
                expected = deepcopy(old["context"])
                chosen_id = None
                trace = None
                c.require(
                    row["id"] not in ids, "Self-correction cannot be a prediction input"
                )
                if ids:
                    record, trace = buckets[cond].retrieve(row["text"], query)
                    other, similarity, other_trace = independent[cond].retrieve(
                        row["text"], query
                    )
                    c.require(
                        record == other and record["provenance"]["source_id"] in ids,
                        "Independent adaptive retrieval selection differs",
                    )
                    for channel in ("dense", "bm25", "rrf"):
                        c.require(
                            [t["id"] for t in trace[channel]]
                            == [t["id"] for t in other_trace[channel]],
                            "Adaptive ranking differs",
                        )
                        c.require(
                            all(
                                (
                                    abs(x["score"] - y["score"]) < 1e-10
                                    for x, y in zip(
                                        trace[channel], other_trace[channel]
                                    )
                                )
                            ),
                            "Adaptive scores differ",
                        )
                    if not source:
                        c.require(similarity < c.NEAR, "Transfer correction overlap")
                    c.require(
                        all(
                            (
                                record["meta"][k] is None
                                for k in (
                                    "hate_types",
                                    "target_groups",
                                    "severity",
                                    "legal",
                                )
                            )
                        ),
                        "Fine labels must remain unknown",
                    )
                    hit = {
                        "id": record["id"],
                        "text": record["text"],
                        "meta": record["meta"],
                        "via": "rrf",
                    }
                    ctx = intervene(ctx, "reserved5", hit)
                    expected["example_groups"][0][1].pop()
                    expected["example_groups"][0][1].append(deepcopy(hit))
                    chosen_id = record["provenance"]["source_id"]
                    delivery[cond + "/" + str(record["meta"]["gate"])] += 1
                messages = renderer.render_messages(row["text"], "en", ctx, "corrected")
                independent_messages = renderer.render_messages(
                    row["text"], "en", expected, "corrected"
                )
                count = tokens.token_count(tok, messages)
                c.require(
                    ctx == expected
                    and messages == independent_messages
                    and (messages[0] == old["messages"][0])
                    and (0 < count <= 8192),
                    "Adaptive rendering/token check failed",
                )
                checked += 1
                yield {
                    "key": row["key"],
                    "item_id": row["id"],
                    "condition": cond,
                    "population": row["population"],
                    "lang": "en",
                    "text": row["text"],
                    "context": ctx,
                    "context_sha256": c.sig(ctx),
                    "messages": messages,
                    "messages_sha256": c.sig(messages),
                    "source_id": chosen_id,
                    "retrieval_trace": trace,
                    "reference_input_tokens": count,
                    "scientific_plan_sha256": science_sha,
                }

    c.jsonlines(root / "prompts.jsonl", generate())
    c.require(
        checked == len(rows) * len(conditions), "Adaptive prompt matrix incomplete"
    )
    outputs = c.bindings(
        (
            root / n
            for n in [
                "scientific_plan_v1.json",
                "roster.jsonl",
                "prompts.jsonl",
                "STUDY_PROTOCOL.md",
            ]
        )
    )
    c.save(
        root / "prepared.json",
        {"status": "prepared_requires_independent_check", "outputs": outputs},
    )
    c.verify(inputs)
    validate()
    v = root / "independent_verification.json"
    c.save(
        v,
        {
            "status": "all_prompts_independently_retrieved_and_rendered",
            "prompts": checked,
            "delivery": dict(delivery),
            "new_model_calls": 0,
            "prepared_sha256": c.sha(root / "prepared.json"),
            "parent_plan_sha256": c.sha(BASE / "plan.json"),
        },
    )
    c.save(
        root / "complete.json",
        {
            "status": "offline_preparation_independently_verified",
            "prepared_sha256": c.sha(root / "prepared.json"),
            "verification": c.rel(v),
            "verification_sha256": c.sha(v),
        },
    )
    return q.queue(root, study, SOURCES)


def ready_phase(b):
    if not b.PLAN.exists():
        plan = validate()
        q.prepare(
            b,
            {
                "passed": True,
                "tests": plan["policy_tests"],
                "source_hashes": {n: c.sha(c.ROOT / n) for n in b.SOURCE_FILES},
                "environment": b.core.environment(),
                "actual_API_calls": 0,
                "policy_receipt": c.rel(BASE / "plan.json"),
                "queue_receipt": c.rel(c.TRANSFER / "execution/ready.json"),
            },
        )
    q.rehearse(b)


def extract(b):

    c.require(
        b.STUDIES.keys() == {"source"}, "Selection cannot read evaluation outcomes"
    )
    path = b.BASE / "source_gates.json"
    if path.exists():
        receipt = c.read(b.BASE / "source_gates_receipt.json")
        c.verify(receipt["bindings"])
        return (c.read(path), receipt["bindings"])
    done = c.read(b.BASE / "live/complete.json")
    c.require(
        done["mode"] == "live" and done["plan_sha256"] == c.sha(b.PLAN),
        "Audited live screening required",
    )
    c.verify(done["review_hashes"])
    schedule = c.read(b.BASE / "schedule.json")
    values = {}
    bindings = {
        c.rel(b.BASE / "live/complete.json"): c.sha(b.BASE / "live/complete.json")
    }
    roster = {r["key"]: r for r in c.rows(b.STUDIES["source"]["root"] / "roster.jsonl")}
    for stage in schedule["stages"]:
        for job in stage["jobs"]:
            c.require(
                job["condition"] == "reference" or job["condition"] == "adaptive",
                "Wrong screening condition",
            )
            p = b.item_dir(b.BASE / "live/source", job, c.sha(b.PLAN)) / "result.json"
            v = c.read(p)
            bindings[c.rel(p)] = c.sha(p)
            i = roster[job["key"]]["id"]
            out = values.setdefault(i, [None] * 3)
            out[job["repeat"]] = None if v["result"] is None else v["result"]["hate"]
    c.require(len(values) == len(roster), "Missing screening inputs")
    c.save(path, values)
    bindings[c.rel(path)] = c.sha(path)
    c.save(
        b.BASE / "source_gates_receipt.json",
        {"bindings": bindings, "gold_evaluation_read": False},
    )
    return (values, bindings)


def prepare_initial():
    prepare()
    data = load_data()
    b = make_phase(
        BASE / "phases/source_initial", "source", data[0], {"reference": []}, {}, data
    )
    ready_phase(b)
    candidates = data[0]
    grouped = {
        g: [r["id"] for r in candidates if r["gold"]["hate"] is g]
        for g in (False, True)
    }
    small = sorted(grouped[False][:2] + grouped[True][:2])
    large = sorted(grouped[False][:20] + grouped[True][:20])
    examples = [
        next((r for r in data[2] if r["phase"] == phase))
        for phase in ("en_test", "en_hatecheck_eval")
    ]
    probe = make_phase(
        BASE / "offline_check/dynamic_prompt_probe",
        "probe",
        examples,
        {
            "reference": [],
            "empty_cumulative": [],
            "four_records": small,
            "forty_records": large,
        },
        {},
        data,
    )
    ready_phase(probe)
    c.save(
        BASE / "ready.json",
        {
            "status": "prospective_sequence_and_initial_queue_verified",
            "plan_sha256": c.sha(BASE / "plan.json"),
            "initial_ready_sha256": c.sha(b.BASE / "ready.json"),
            "dynamic_probe_ready_sha256": c.sha(probe.BASE / "ready.json"),
            "actual_API_calls": 0,
            "later_pools_require_audited_source_outcomes": True,
        },
    )
    return b


def require_a16_finished():
    path = c.TRANSFER / "execution/live/complete.json"
    c.require(
        path.exists(), "Run and review A16 before A17; do not start another Qwen queue"
    )
    d = c.read(path)
    c.require(
        d["mode"] == "live"
        and d["status"] == "all_studies_executed_and_audited"
        and (d["plan_sha256"] == c.sha(c.TRANSFER / "execution/plan.json"))
        and (d["counts"]["predictions"] == 189096),
        "A16 final audit/identity differs",
    )


def run():
    q.require_a15_finished()
    require_a16_finished()
    validate()
    data = load_data()
    candidates, _, evaluation, _, _, _, _ = data
    by = {r["id"]: r for r in candidates}
    blocks = c.read(BASE / "source_blocks.json")["blocks"]
    with q.template.a.item_lock(
        q.template.a.storage_path(c.SHARED / "qwen_queue.lock")
    ), q.template.a.item_lock(q.template.a.storage_path(BASE / "launcher.lock")):
        initial = make_phase(
            BASE / "phases/source_initial",
            "source",
            candidates,
            {"reference": []},
            {},
            data,
        )
        ready_phase(initial)
        q.execute(initial)
        baseline, base_deps = extract(initial)
        previous = {str(s): {p: [] for p in policy.POLICIES} for s in c.STREAM_SEEDS}
        phase_receipts = []
        phase_receipts.append(initial.BASE / "live/complete.json")
        for round_no in range(1, 5):
            combined = {"reference": []}
            selection_paths = []
            for seed in c.STREAM_SEEDS:
                ids = blocks[str(seed)][round_no - 1]
                rows = [by[i] for i in ids]
                old = {i: baseline[i] for i in ids}
                deps = dict(base_deps)
                if round_no == 1:
                    current = old
                else:
                    prior = BASE / f"rounds/r{round_no - 1}/selection_s{seed}.json"
                    deps[c.rel(prior)] = c.sha(prior)
                    b = make_phase(
                        BASE / f"phases/r{round_no}_source_s{seed}",
                        "source",
                        rows,
                        {"adaptive": previous[str(seed)]["adaptive"]},
                        deps,
                        data,
                    )
                    ready_phase(b)
                    q.execute(b)
                    current, current_deps = extract(b)
                    deps.update(current_deps)
                    phase_receipts.append(b.BASE / "live/complete.json")
                chosen = policy.verified_selection(
                    rows, old, current, previous[str(seed)], seed, round_no
                )
                chosen.update(
                    parent_plan_sha256=c.sha(BASE / "plan.json"),
                    source_bindings=deps,
                    evaluation_outcomes_read=False,
                )
                path = BASE / f"rounds/r{round_no}/selection_s{seed}.json"
                c.save(path, chosen)
                selection_paths.append(path)
                previous[str(seed)] = chosen["cumulative"]
                for pol, values in chosen["cumulative"].items():
                    combined[f"{pol}_s{seed}"] = values
                print(
                    f"A17 round {round_no}, stream {seed}: {2 * chosen['per_class_budget']} additions per policy; {len(chosen['cumulative']['adaptive'])} cumulative.",
                    flush=True,
                )
            b = make_phase(
                BASE / f"phases/r{round_no}_evaluation",
                "evaluation",
                evaluation,
                combined,
                c.bindings(selection_paths),
                data,
            )
            ready_phase(b)
            q.execute(b)
            phase_receipts.append(b.BASE / "live/complete.json")
        totals = Counter()
        for path in phase_receipts:
            d = c.read(path)
            totals.update(
                {
                    "predictions": d["counts"]["predictions"],
                    "ordinary_calls": d["accounting"]["ordinary_logical_calls"],
                }
            )
        c.require(
            dict(totals) == {"predictions": 259224, "ordinary_calls": 777672},
            "Iterative execution matrix incomplete",
        )
        c.save(
            BASE / "complete.json",
            {
                "status": "all_four_simulated_rounds_audited_scientific_review_pending",
                "plan_sha256": c.sha(BASE / "plan.json"),
                "phase_receipts": c.bindings(phase_receipts),
                "totals": dict(totals),
                "real_expert_feedback": False,
                "evaluation_scoring_performed": False,
            },
        )
    print(
        "A17 FOUR ROUNDS COMPLETE. Tell Astra to inspect. SQ2 and other studies were not started.",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "initial", "validate", "run-all"])
    args = ap.parse_args()
    with threadpool_limits(limits=1):
        if args.mode == "run-all":
            run()
            return
        with c.offline():
            if args.mode == "prepare":
                prepare()
            elif args.mode == "initial":
                prepare_initial()
            else:
                validate()
                print("A17 frozen protocol and inputs verified.")


if __name__ == "__main__":
    main()
