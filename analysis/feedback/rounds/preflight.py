import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
import gc
import hashlib
import json
import math
import shutil
import time
import uuid
import xml.etree.ElementTree as ET
from unittest.mock import patch
from analysis.feedback.rounds import runtime as m

c = m.c
a = m.a
old = m.original
BASE = m.BASE


def tests():
    import pytest

    attempt = BASE / "development" / ("tests_" + uuid.uuid4().hex[:10])
    attempt.mkdir(parents=True)
    paths = [
        c.ROOT / "scratch/analysis" / n
        for n in (
            "test_astra_sq3_iterative_runtime_v1.py",
            "test_astra_sq3_iterative_policy_v1.py",
            "test_astra_sq3_iterative_sequence_v1.py",
            "test_astra_sq3_followup_queue_v1.py",
        )
    ]
    sources = c.bindings([*m.sources(), *paths])
    for path in [*m.sources(), *paths]:
        target = attempt / "sources" / path.name
        if not target.exists():
            c.write_bytes(target, path.read_bytes())
    code = pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            *[str(p) for p in paths],
            "--basetemp",
            str(attempt / "fixtures"),
            "--junitxml",
            str(attempt / "results.xml"),
        ]
    )
    suites = ET.parse(attempt / "results.xml").getroot().findall("testsuite")
    count = sum(int(s.attrib["tests"]) for s in suites)
    receipt = dict(
        passed=code == 0,
        tests=count,
        actual_Qwen_calls=0,
        actual_model_list_GETs=0,
        environment=m.q.template.core.environment(),
        sources=sources,
        artifacts=c.bindings([attempt / "results.xml"]),
        attempt=c.rel(attempt),
    )
    c.save(attempt / "result.json", receipt)
    c.require(code == 0, "Offline runtime tests failed; attempt preserved")
    c.verify(sources)
    c.save(BASE / "tests.json", receipt)
    print(f"{count} offline tests passed. No Qwen or model-list requests.", flush=True)


def independent_scope(candidates, evaluation):
    by = {r["id"]: r for r in candidates}
    saved = c.read(c.ITERATIVE / "source_blocks.json")
    c.require(
        len(by) == 1024
        and Counter(r["gold"]["hate"] for r in candidates) == {False: 512, True: 512},
        "Candidate scope",
    )
    for seed in c.STREAM_SEEDS:
        blocks = [[] for _ in range(4)]
        for gate in (False, True):
            ranked = sorted(
                [i for i, r in by.items() if r["gold"]["hate"] is gate],
                key=lambda i: hashlib.sha256(
                    f"A17|{seed}|0|source|{int(gate)}|{i}".encode()
                ).hexdigest(),
            )
            for j in range(4):
                blocks[j].extend(ranked[j * 128 : (j + 1) * 128])
        c.require(
            [sorted(v) for v in blocks] == saved["blocks"][str(seed)],
            "Independent source partition",
        )
    full = list(c.rows(old.SCOPE / "evaluation.jsonl"))
    selected = []
    for phase in ("en_test", "en_hatecheck_eval"):
        groups = defaultdict(list)
        for r in full:
            if r["phase"] == phase:
                groups[r["inference_cluster"]].append(r["key"])
        chosen = []
        for cluster in sorted(
            groups,
            key=lambda x: hashlib.sha256(
                ("A17-evaluation-20261012|" + x).encode()
            ).hexdigest(),
        ):
            if len(chosen) >= 1024:
                break
            chosen.extend(groups[cluster])
        selected.extend(chosen)
    fixed = c.read(old.SCOPE / "iteration_evaluation.json")
    c.require(
        not fixed["outcome_selected"]
        and sorted(selected) == fixed["keys"] == sorted(r["key"] for r in evaluation),
        "Independent complete-cluster sample",
    )
    c.require(
        Counter(r["phase"] for r in evaluation)
        == {"en_test": 1024, "en_hatecheck_eval": 1053},
        "Evaluation populations",
    )
    c.require(
        not set(by).intersection(r["id"] for r in evaluation),
        "No direct source/evaluation identity overlap",
    )
    return dict(
        candidates=1024,
        blocks_per_stream=4,
        block_size=256,
        per_block_per_class=128,
        evaluation_populations=dict(Counter(r["phase"] for r in evaluation)),
        evaluation_polarities=dict(
            Counter(r["phase"] + "/" + str(r["gate"]) for r in evaluation)
        ),
        evaluation_clusters={
            phase: len(
                {r["inference_cluster"] for r in evaluation if r["phase"] == phase}
            )
            for phase in ("en_test", "en_hatecheck_eval")
        },
        source_components_with_multiple_candidates=len(
            saved["source_clusters_with_multiple_candidates"]
        ),
        independent_partition_and_cluster_sample=True,
        evaluation_outcomes_used_for_selection=False,
    )


def layout(evaluation):
    root = BASE / "offline_check/full_evaluation_schedule"
    n = len(evaluation)
    conditions = ["reference"] + [
        f"{p}_s{s}" for s in c.STREAM_SEEDS for p in old.policy.POLICIES
    ]
    b = m.q.queue(
        root,
        "evaluation",
        old.SOURCES,
        dict(inputs=n, conditions=10, batches=7, predictions=n * 30),
    )
    schema = "config/taxonomy.yaml"
    sha = c.sha(c.ROOT / schema)
    cat = {}
    for row in evaluation:
        message = [
            dict(role="system", text="OFFLINE SCHEDULE FIXTURE ONLY"),
            dict(role="user", text=row["key"]),
        ]
        for cond in conditions:
            cat[row["key"], cond] = dict(
                messages=message,
                messages_sha256=a.signature(message),
                schema_path=schema,
                schema_sha256=sha,
                population=row["population"],
                entry_sha256="excluded-layout-fixture",
            )
    schedule = b.make_schedule({"evaluation": cat})
    checked = m.q.validate_schedule(b, schedule, {"evaluation": cat})
    c.require(
        checked == {"predictions": 62310, "ordinary_calls": 186930, "batches": 7},
        "Each round evaluation schedule",
    )
    c.save(
        root / "checked.json",
        dict(
            status="full_evaluation_identity_schedule_verified_without_future_pools",
            conditions=conditions,
            checked=checked,
            batch_predictions=[s["items"] for s in schedule["stages"]],
            schedule_sha256=c.sig(schedule),
            actual_Qwen_calls=0,
            scientific_future_prompts_created=False,
        ),
    )
    return checked


def dynamic_probes(data):
    candidates, _, evaluation, _, _, _, _ = data
    grouped = {
        g: sorted((r["id"] for r in candidates if r["gold"]["hate"] is g))
        for g in (False, True)
    }
    pool_max = set(grouped[False][:20] + grouped[True][:20])
    rows = [
        next(r for r in evaluation if r["phase"] == phase)
        for phase in ("en_test", "en_hatecheck_eval")
    ]
    rows.append(next(r for r in candidates if r["id"] not in pool_max))
    results = []
    for size in (0, 2, 10, 20, 30, 40):
        ids = sorted(grouped[False][: size // 2] + grouped[True][: size // 2])
        pools = {
            "reference": [],
            **{
                f"{pol}_s{s}": ids
                for s in c.STREAM_SEEDS
                for pol in old.policy.POLICIES
            },
        }
        root = BASE / f"offline_check/dynamic_{size:02d}"
        b = old.make_phase(root, "probe", rows, pools, {}, data)
        old.ready_phase(b)
        cats = b.load_catalog("probe")
        schedule = c.read(b.BASE / "schedule.json")
        plan_sha = c.sha(b.PLAN)
        m.bind_stage(b, dict(excluded_fixture=True, protocol_sha256=c.sha(m.PROTOCOL)))
        live_mock = b.BASE / "offline_check/rehearsal_128"
        transport = m.q.SyntheticTransport()
        for stage in schedule["stages"]:
            b.execute_stage(
                stage,
                cats,
                live_mock / "probe",
                plan_sha,
                mode="mock",
                client_factory=lambda: transport,
                sleep=lambda _: None,
            )
            b.audit.audit_stage(stage, cats, live_mock / "probe", plan_sha, mode="mock")
        first = b.audit.finalize_queue(schedule, live_mock, plan_sha, mode="mock")
        before = transport.count
        for stage in schedule["stages"]:
            b.execute_stage(
                stage,
                cats,
                live_mock / "probe",
                plan_sha,
                mode="mock",
                client_factory=c.forbidden,
            )
            b.verify_review(stage, live_mock / "probe", plan_sha, "mock")
        c.require(
            transport.count == before == 270 and first["counts"]["predictions"] == 90,
            "Dynamic mock response budget and restart",
        )
        prompt_rows = list(c.rows(root / "prompts.jsonl"))
        refs = {r["key"]: r for r in prompt_rows if r["condition"] == "reference"}
        for row in prompt_rows:
            n = len(row["context"]["example_groups"][0][1])
            c.require(n == 5, "Exactly five delivered examples")
            if size == 0:
                c.require(
                    row["messages"] == refs[row["key"]]["messages"]
                    and row["source_id"] is None,
                    "Empty pool uses full unchanged reference",
                )
            elif row["condition"] != "reference":
                c.require(
                    row["source_id"] in ids,
                    "Correction delivered from declared cumulative pool",
                )
        results.append(
            dict(
                pool_size=size,
                inputs=3,
                conditions=10,
                prompts=len(prompt_rows),
                mock_predictions=90,
                mock_replies=before,
                new_replies_on_restart=0,
                source_and_both_evaluation_paths=True,
                root=c.rel(root),
                ready_sha256=c.sha(b.BASE / "ready.json"),
                mock_complete=c.rel(live_mock / "complete.json"),
                mock_complete_sha256=c.sha(live_mock / "complete.json"),
            )
        )
        del cats, prompt_rows, b
        gc.collect()
    return results


def storage_estimate():

    matrix = c.ROOT / "experiments/astra/sq3_transfer_part1_20261023_v1/matrix.jsonl"
    groups = defaultdict(list)
    for row in c.rows(matrix):
        groups[row["population"], row["gold_hate"]].append(row["key"])
    conditions = c.read(c.TRANSFER / "scientific_plan_v1.json")["conditions"]
    plan = c.sha(c.TRANSFER / "execution/plan.json")
    records = []
    for group, keys in sorted(groups.items()):
        ordered = sorted(keys, key=lambda x: c.text_sha("iterative-storage|" + x))[:64]
        for key in ordered:
            number = int(c.text_sha("iterative-storage-condition|" + key), 16)
            cond = conditions[number % 4]
            rep = number % 3
            identity = dict(
                experiment_sha256=plan,
                condition="transfer::" + cond,
                repeat=rep,
                key=key,
            )
            folder = a.storage_path(
                c.TRANSFER / "execution/live/transfer/items" / a.signature(identity)
            )
            c.require(
                (folder / "complete.json").is_file(), "Storage sample completed source"
            )
            sizes = [p.stat().st_size for p in folder.rglob("*") if p.is_file()]
            records.append(
                dict(
                    population=group[0],
                    gold_hate=group[1],
                    key=key,
                    condition=cond,
                    repeat=rep,
                    files=len(sizes),
                    logical_bytes=sum(sizes),
                    estimated_4KiB_allocation=sum(
                        math.ceil(n / 4096) * 4096 for n in sizes
                    ),
                )
            )
    mean = sum(r["estimated_4KiB_allocation"] for r in records) / len(records)
    raw = mean * 259224 / 2**30
    planning = raw * 1.3 + 5
    return dict(
        sample_predictions=len(records),
        sample_records=records,
        assumed_allocation_unit_bytes=4096,
        projected_item_allocation_GiB=raw,
        planning_GiB_with_30percent_and_5GiB_allowance=planning,
        available_disk_GiB=shutil.disk_usage(c.ROOT).free / 2**30,
        boundary="Rough stratified file-size extrapolation, not a worst-case bound. 4 KiB allocation is an assumption; directory/MFT, shared payloads, prompts and audits are covered only by a planning allowance. Future repairs and outputs can differ.",
        source=c.rel(matrix),
        source_sha256=c.sha(matrix),
    )


def preflight():
    c.require(
        not (BASE / "preflight.json").exists(), "Preserve completed readiness check"
    )
    started = time.perf_counter()
    old.validate()
    m.prerequisites()
    resources = c.resources(2.5, 20.0)
    c.require(
        not list((c.ITERATIVE / "phases").glob("*/execution/live"))
        and not (c.ITERATIVE / "rounds").exists()
        and not (c.ITERATIVE / "complete.json").exists(),
        "Unstarted study required for this initial amendment",
    )
    plan = c.read(c.ITERATIVE / "plan.json")
    preserved = {**plan["source_hashes"], **plan["inputs"], **plan["outputs"]}
    preserved.update(
        c.bindings(
            [
                c.ITERATIVE / "plan.json",
                c.ITERATIVE / "ready.json",
                c.ROOT
                / "Thesis_Log/astra_result/astra_remaining_work_before_writing_20261023_v85.md",
            ]
        )
    )
    initial = m.q.queue(c.ITERATIVE / "phases/source_initial", "source", old.SOURCES)
    initial.validate(ready=True)
    initial_plan = c.read(initial.PLAN)
    preserved.update(initial_plan["input_hashes"])
    preserved.update(c.bindings([initial.PLAN, initial.BASE / "ready.json"]))
    lock = c.SHARED / "qwen_queue.lock"
    before = c.sha(lock)
    with a.item_lock(a.storage_path(lock)):
        lock_checked = True
    c.require(c.sha(lock) == before, "Server lock content unchanged")
    cats = initial.load_catalog("source")
    initial_layout = m.q.validate_schedule(
        initial, c.read(initial.BASE / "schedule.json"), {"source": cats}
    )
    c.require(
        initial_layout == {"predictions": 3072, "ordinary_calls": 9216, "batches": 1},
        "Fresh initial screening schedule",
    )
    del cats
    gc.collect()
    data = old.load_data()
    scope = independent_scope(data[0], data[2])
    shape = layout(data[2])
    probes = dynamic_probes(data)
    del data
    gc.collect()
    storage = storage_estimate()
    c.save(BASE / "storage_estimate.json", storage)
    c.require(
        storage["available_disk_GiB"]
        >= storage["planning_GiB_with_30percent_and_5GiB_allowance"] + 20,
        "Storage planning allowance plus 20 GiB reserve exceeds free disk; review before launch",
    )
    c.verify(preserved)
    c.require(
        not list((c.ITERATIVE / "phases").glob("*/execution/live")),
        "No live phase created",
    )
    support = {**preserved, **c.read(m.CAPACITY)["bindings"]}
    support.update(
        c.bindings(
            [
                BASE / "storage_estimate.json",
                m.TRANSFER_REVIEW / "verification.json",
                c.ROOT
                / "experiments/astra/sq3_transfer_part9_20261023_v1/resource_summary.json",
                c.ROOT
                / "experiments/astra/sq3_transfer_part9_20261023_v1/timeline.json",
            ]
        )
    )
    for row in probes:
        support.update(
            c.bindings(
                [
                    c.ROOT / row["mock_complete"],
                    c.ROOT / row["root"] / "execution/ready.json",
                ]
            )
        )
    c.verify(support)
    c.save(
        BASE / "preflight.json",
        dict(
            passed=True,
            created_utc=c.utc(),
            seconds=time.perf_counter() - started,
            actual_Qwen_calls=0,
            actual_model_list_GETs=0,
            actual_fits=0,
            new_embeddings=0,
            resources=resources,
            shared_OS_queue_lock_available=lock_checked,
            shared_lock_bytes_unchanged=True,
            original_environment=plan["environment"],
            initial_schedule=initial_layout,
            each_evaluation_round=shape,
            phase_count=14,
            automatic_batches=38,
            predictions=259224,
            ordinary_calls=777672,
            scope=scope,
            dynamic_prompt_and_128_worker_probes=probes,
            storage_estimate=c.rel(BASE / "storage_estimate.json"),
            preserved_bindings=preserved,
            supporting_bindings=support,
            unstarted_live_study_preserved=True,
            protocol_sha256=c.sha(m.PROTOCOL),
            source_sha256=c.sha(Path(__file__)),
        ),
    )
    print(
        "Offline preflight passed: original scope, all phase layouts, six cumulative-pool sizes, 128-worker mock audits and preserved evidence.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "preflight"])
    args = parser.parse_args()
    with c.offline(), patch.object(
        m.q.template.core, "live_client", c.forbidden
    ), old.threadpool_limits(limits=1):
        globals()[args.mode]()
