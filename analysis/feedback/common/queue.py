import ast
from collections import Counter
import gc
import math
from pathlib import Path
import time
from types import SimpleNamespace
from analysis.common import runtime as c
from analysis.common import vote_api as template
from analysis.feedback.insertion import run as binding
from analysis.feedback.balanced.validate import SyntheticTransport

BASE_SOURCES = list(
    dict.fromkeys(
        [
            *binding.SOURCE_FILES,
            "analysis/common/runtime.py",
            "analysis/feedback/common/queue.py",
        ]
    )
)


def queue(root, study, source_files, info=None):
    root = Path(root)
    c.own(root / "execution")
    if info is None:
        science = c.read(root / "scientific_plan_v1.json")
        n = science["inputs"]
        conditions = len(science["conditions"])
        info = {
            "inputs": n,
            "conditions": conditions,
            "batches": max(1, math.ceil(n * conditions * 3 / 9000)),
            "predictions": n * conditions * 3,
        }
    sources = list(dict.fromkeys([*BASE_SOURCES, *source_files]))
    base = root / "execution"
    ns = {
        **vars(template),
        "BASE": base,
        "PLAN": base / "plan.json",
        "WORKERS": 8,
        "SOURCE_FILES": sources,
        "STUDIES": {study: dict(info, root=root)},
    }
    tree = ast.parse(Path(template.__file__).read_text(encoding="utf8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in binding.COPIED
    ]
    c.require(
        {n.name for n in functions} == set(binding.COPIED),
        "Reviewed engine functions missing",
    )
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), template.__file__, "exec"),
        ns,
    )
    b = SimpleNamespace(
        **{
            k: ns[k]
            for k in [
                "ROOT",
                "BASE",
                "PLAN",
                "WORKERS",
                "SOURCE_FILES",
                "STUDIES",
                "a",
                "core",
                "isolated",
                *binding.COPIED,
            ]
        }
    )
    b.audit = binding.rebound_module(
        c.ROOT / "analysis/common/vote_audit.py", {"b": b}, {"b"}
    )
    return b


def validate_schedule(b, schedule, catalogs):
    seen = set()
    owners = {}
    counts = Counter()
    stageids = set()
    for stage in schedule["stages"]:
        study = stage["study"]
        cat = catalogs[study]
        info = b.STUDIES[study]
        keys = stage["input_keys"]
        c.require(
            stage["id"] not in stageids and keys and (len(keys) == len(set(keys))),
            "Invalid batch identity",
        )
        stageids.add(stage["id"])
        c.require(
            stage["items"] == len(stage["jobs"]) == len(keys) * info["conditions"] * 3
            and stage["ordinary_calls"] == stage["items"] * 3,
            "Invalid batch budget",
        )
        c.require(stage["items"] <= 9000, "Automatic batch exceeds declared ceiling")
        for job in stage["jobs"]:
            ident = (study, job["key"], job["condition"], job["repeat"])
            c.require(
                ident not in seen and job["repeat"] in (0, 1, 2),
                "Duplicate or invalid scheduled job",
            )
            seen.add(ident)
            c.require(
                job["key"] in keys
                and owners.setdefault((study, job["key"]), stage["id"]) == stage["id"],
                "Input split across batches",
            )
            entry = cat[job["key"], job["condition"]]
            c.require(
                all(
                    (
                        job[k] == entry[k]
                        for k in (
                            "messages_sha256",
                            "schema_path",
                            "schema_sha256",
                            "population",
                        )
                    )
                ),
                "Schedule/catalog attribution differs",
            )
            c.require(
                b.a.signature(entry["messages"]) == job["messages_sha256"],
                "Message checksum changed",
            )
            c.require(
                [m["content"] for m in b.a.body(entry["messages"])["messages"]]
                == [m["text"] for m in entry["messages"]],
                "Wire payload conversion changed",
            )
            counts[study] += 1
    expected = {
        (study, key, condition, rep)
        for study, cat in catalogs.items()
        for key, condition in cat
        for rep in range(3)
    }
    c.require(seen == expected, "Scheduled comparison matrix incomplete")
    c.require(
        all(
            (counts[study] == info["predictions"] for study, info in b.STUDIES.items())
        ),
        "Prediction count differs",
    )
    c.require(
        len(stageids) == sum((info["batches"] for info in b.STUDIES.values())),
        "Batch count differs",
    )
    return {
        "predictions": len(seen),
        "ordinary_calls": len(seen) * 3,
        "batches": len(stageids),
    }


def prepare(b, test_record):
    if b.PLAN.exists():
        b.validate()
        print("Existing execution plan verified.", flush=True)
        return
    c.require(not (b.BASE / "live").exists(), "Preserve live work")
    c.require(
        test_record["passed"] and test_record["actual_API_calls"] == 0,
        "Passing offline tests required",
    )
    c.verify(test_record["source_hashes"])
    c.require(
        test_record["environment"] == b.core.environment(), "Test environment differs"
    )
    b.save(b.test_receipt(), test_record)
    inputs = {}
    catalogs = {}
    for study in b.STUDIES:
        inputs.update(b.study_inputs(study))
        catalogs[study] = b.load_catalog(study)
    schedule = b.make_schedule(catalogs)
    schedule["ordering"] = (
        "Stable hashed input order; rotated conditions/repeats; every comparison for one input in one batch; at most 9000 predictions per automatic batch."
    )
    budget = validate_schedule(b, schedule, catalogs)
    budget.update(
        max_logical_calls=budget["ordinary_calls"] * 3,
        max_transport_POST_attempts=budget["ordinary_calls"] * 15,
    )
    b.save(b.BASE / "schedule.json", schedule)
    for path in (b.BASE / "schedule.json", b.test_receipt()):
        inputs[c.rel(path)] = c.sha(path)
    b.save(
        b.PLAN,
        {
            "created_utc": c.utc(),
            "input_hashes": inputs,
            "source_hashes": {n: c.sha(c.ROOT / n) for n in b.SOURCE_FILES},
            "environment": b.core.environment(),
            "api_contract": b.a.CONTRACT,
            "workers": 8,
            "budget": budget,
            "status": "frozen_requires_mock_rehearsal",
            "no_accuracy_based_stopping": True,
            "model_calls_before_readiness": 0,
            "scope": "Only this frozen follow-up queue; no original source edits or old response reuse.",
        },
    )
    print("Execution schedule frozen:", budget, flush=True)


def rehearse(b):
    if (b.BASE / "ready.json").exists():
        b.validate(ready=True)
        print("Existing readiness verified.")
        return
    b.validate()
    schedule = c.read(b.BASE / "schedule.json")
    cats = {st: b.load_catalog(st) for st in b.STUDIES}
    plan_sha = c.sha(b.PLAN)
    matrix = validate_schedule(b, schedule, cats)
    selected = []
    for stage in schedule["stages"]:
        cat = cats[stage["study"]]
        by_pop = {}
        condition = next(iter(cat))[1]
        for key in stage["input_keys"]:
            by_pop.setdefault(cat[key, condition]["population"], key)
        keys = set(by_pop.values())
        jobs = [j for j in stage["jobs"] if j["key"] in keys]
        selected.append(
            dict(
                stage,
                input_keys=sorted(keys),
                jobs=jobs,
                items=len(jobs),
                ordinary_calls=len(jobs) * 3,
            )
        )
    root = b.BASE / "offline_check" / f"rehearsal_{time.time_ns()}"
    transport = SyntheticTransport()
    for stage in selected:
        st = stage["study"]
        b.execute_stage(
            stage,
            cats[st],
            root / st,
            plan_sha,
            mode="mock",
            client_factory=lambda: transport,
            sleep=lambda _: None,
        )
        b.audit.audit_stage(stage, cats[st], root / st, plan_sha, mode="mock")
    reduced = dict(schedule, stages=selected)
    first = b.audit.finalize_queue(reduced, root, plan_sha, mode="mock")
    before = transport.count
    for stage in selected:
        st = stage["study"]
        b.execute_stage(
            stage,
            cats[st],
            root / st,
            plan_sha,
            mode="mock",
            client_factory=c.forbidden,
        )
        b.verify_review(stage, root / st, plan_sha, "mock")
    second = b.audit.finalize_queue(reduced, root, plan_sha, mode="mock")
    expected = sum((s["items"] for s in selected))
    c.require(
        first == second
        and transport.count == before == expected * 3
        and (first["counts"]["predictions"] == expected),
        "Mock restart duplicated or lost calls",
    )
    path = b.BASE / "development/readiness.json"
    b.save(
        path,
        {
            "status": "full_matrix_and_automatic_audits_rehearsed",
            "matrix": matrix,
            "mock_predictions": expected,
            "mock_replies": before,
            "new_calls_on_restart": 0,
            "actual_API_calls": 0,
            "rehearsal_root": c.rel(root),
            "completion_sha256": c.sha(root / "complete.json"),
            "plan_sha256": plan_sha,
            "environment": b.core.environment(),
        },
    )
    b.save(
        b.BASE / "ready.json",
        {
            "status": "offline_verified_ready_for_user",
            "plan_sha256": plan_sha,
            "evidence": c.rel(path),
            "evidence_sha256": c.sha(path),
            "actual_API_calls": 0,
        },
    )
    b.validate(ready=True)
    print("Follow-up queue offline ready; no real API calls.", flush=True)


def require_a15_finished():
    path = c.A15 / "execution/live/complete.json"
    c.require(
        path.exists(),
        "A15 is still running or unaudited. Keep one Qwen queue; wait for A15 review before this launch.",
    )
    d = c.read(path)
    p = c.read(c.A15 / "execution/plan.json")
    c.require(
        d["status"] == "all_studies_executed_and_audited"
        and d["mode"] == "live"
        and (d["plan_sha256"] == c.sha(c.A15 / "execution/plan.json"))
        and (d["counts"]["predictions"] == p["budget"]["predictions"]),
        "A15 completion differs; review required",
    )


def execute(b):

    require_a15_finished()
    b.validate(ready=True)
    c.resources(2.5, 20.0)
    plan_sha = c.sha(b.PLAN)
    schedule = c.read(b.BASE / "schedule.json")
    cats = {st: b.load_catalog(st) for st in b.STUDIES}

    def run(stage):
        b.validate(ready=True)
        c.resources(0.4, 4.0)
        b.execute_stage(
            stage,
            cats[stage["study"]],
            b.BASE / "live" / stage["study"],
            plan_sha,
            mode="live",
            client_factory=b.core.live_client,
        )

    b.run_sequence(
        schedule["stages"],
        run,
        lambda s: b.audit.audit_stage(
            s, cats[s["study"]], b.BASE / "live" / s["study"], plan_sha, mode="live"
        ),
        lambda s: (
            b.BASE / "live" / s["study"] / "stages" / s["id"] / "review.json"
        ).exists(),
        lambda s: b.verify_review(s, b.BASE / "live" / s["study"], plan_sha, "live"),
    )
    del cats
    gc.collect()
    b.validate(ready=True)
    return b.audit.finalize_queue(schedule, b.BASE / "live", plan_sha, mode="live")
