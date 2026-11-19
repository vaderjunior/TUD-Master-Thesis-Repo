import argparse
from collections import Counter
from pathlib import Path
from types import FunctionType, SimpleNamespace
from analysis.feedback.rounds import run as original

c = original.c
q = original.q
a = q.template.a
BASE = c.ITERATIVE / "execution/concurrency_20261024_v1"
MANIFEST = BASE / "manifest.json"
PROTOCOL = BASE / "EXECUTION_AMENDMENT.md"
LAUNCHER = c.ROOT / "scripts/run_feedback_rounds_parallel.ps1"
WORKERS = 128
CAPACITY = c.TRANSFER / "execution/concurrency_20261021_v3/manifest.json"
TRANSFER_REVIEW = c.ROOT / "experiments/astra/sq3_transfer_part10_20261023_v1"
RESERVED_REVIEW = c.ROOT / "experiments/astra/sq3_reserved_part10_results_20261020_v1"


def sources():
    return [
        Path(__file__),
        PROTOCOL,
        LAUNCHER,
        c.ROOT / "analysis/feedback/rounds/preflight.py",
        c.ROOT / "tests/feedback/rounds/test_runtime.py",
    ]


def prerequisites():
    q.require_a15_finished()
    original.require_a16_finished()
    x = c.read(TRANSFER_REVIEW / "complete.json")
    check = c.read(TRANSFER_REVIEW / "publication/final_check.json")
    c.require(
        x["broader_transfer_complete"]
        and x["principal_static_classifier_comparison_complete"]
        and x["completed_review_parts"] == list(range(1, 11))
        and not x["remaining_review_parts"],
        "Complete broader-transfer review required",
    )
    c.require(
        check["status"] == "passed"
        and check["complete_sha256"] == c.sha(TRANSFER_REVIEW / "complete.json"),
        "Broader-transfer publication verification changed",
    )
    c.require(
        c.read(RESERVED_REVIEW / "publication/final_validation.json")["status"]
        == "final_publication_readback_passed",
        "Earlier reserved-correction review required",
    )
    capacity = c.read(CAPACITY)
    c.require(
        capacity["status"] == "tested_ready"
        and capacity["workers"] == WORKERS
        and capacity["api_contract"] == a.CONTRACT,
        "Existing tested 128-worker evidence differs",
    )


def authorize(record):
    c.require(
        record["status"] == "offline_verified_ready_for_user",
        "Runtime amendment not ready",
    )
    c.require(
        type(record["effective_workers"]) is int
        and record["effective_workers"] == WORKERS,
        "Only the reviewed 128-worker setting is allowed",
    )
    c.require(
        record["original_workers"] == 8 and record["api_contract"] == a.CONTRACT,
        "Frozen votes, parsing, transport or scientific contract changed",
    )
    c.require(
        record["parent_plan_sha256"] == c.sha(c.ITERATIVE / "plan.json"),
        "Iterative plan changed",
    )
    c.require(
        record["predictions"] == 259224
        and record["ordinary_calls"] == 777672
        and record["rounds"] == 4
        and record["streams"] == 3,
        "Frozen scope changed",
    )


def validate():
    record = c.read(MANIFEST)
    authorize(record)
    c.verify(record["bindings"])
    required = [
        *sources(),
        BASE / "tests.json",
        BASE / "preflight.json",
        BASE / "launcher_check.json",
        c.ITERATIVE / "plan.json",
        c.ITERATIVE / "ready.json",
        CAPACITY,
        TRANSFER_REVIEW / "complete.json",
        TRANSFER_REVIEW / "publication/final_check.json",
        RESERVED_REVIEW / "complete.json",
        RESERVED_REVIEW / "publication/final_validation.json",
    ]
    c.require(
        all(record["bindings"].get(c.rel(path)) == c.sha(path) for path in required),
        "Missing runtime provenance",
    )
    tests = c.read(BASE / "tests.json")
    pre = c.read(BASE / "preflight.json")
    launch = c.read(BASE / "launcher_check.json")
    c.require(
        tests["passed"]
        and tests["actual_Qwen_calls"] == 0
        and pre["passed"]
        and pre["actual_Qwen_calls"] == 0,
        "Offline checks required",
    )
    c.require(
        launch["parsed"]
        and launch["set_succeeded"]
        and launch["restored"]
        and launch["launcher_sha256"] == c.sha(LAUNCHER),
        "Launcher check changed",
    )
    c.require(
        record["environment"] == q.template.core.environment(),
        "Frozen environment changed",
    )
    prerequisites()
    return record


def provenance_path(b, attempt):
    return (
        b.BASE
        / "concurrency_20261024_v1/provenance"
        / attempt.parent.name
        / (attempt.name + ".json")
    )


def bind_stage(b, amendment):

    fn = b.execute_stage
    c.require(
        b.WORKERS == 8 and fn.__globals__["WORKERS"] == 8,
        "Expected unchanged eight-worker engine",
    )
    save_original = fn.__globals__["save"]
    review_original = b.verify_review

    def save(path, value):
        save_original(path, value)
        path = Path(str(path).removeprefix("\\\\?\\"))
        if path.name == "start.json" and path.parent.name.startswith("attempt_"):
            save_original(
                provenance_path(b, path.parent),
                dict(
                    original_attempt=c.rel(path.parent),
                    original_plan_sha256=value["plan_sha256"],
                    start_sha256=c.sha(path),
                    stage=value["stage"],
                    mode=value["mode"],
                    original_workers=8,
                    effective_workers=WORKERS,
                    amendment=amendment,
                ),
            )

    namespace = dict(fn.__globals__, WORKERS=WORKERS, save=save)
    rebound = FunctionType(
        fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__
    )
    rebound.__kwdefaults__ = dict(fn.__kwdefaults__ or {})
    c.require(
        {k for k in namespace if namespace[k] is not fn.__globals__[k]}
        == {"WORKERS", "save"},
        "Unintended stage transformation",
    )

    def verify_review(stage, root, plan_sha, mode):
        reviewed = review_original(stage, root, plan_sha, mode)
        target = a.storage_path(root) / "stages" / stage["id"]
        expected = set()
        for attempt in target.glob("attempt_*"):
            ordinary = Path(str(attempt).removeprefix("\\\\?\\"))
            path = provenance_path(b, ordinary)
            expected.add(path.name)
            marker = c.read(path)
            c.require(
                marker
                == dict(
                    original_attempt=c.rel(ordinary),
                    original_plan_sha256=plan_sha,
                    start_sha256=c.sha(ordinary / "start.json"),
                    stage=stage["id"],
                    mode=mode,
                    original_workers=8,
                    effective_workers=WORKERS,
                    amendment=amendment,
                ),
                "Runtime attempt provenance differs",
            )
        folder = b.BASE / "concurrency_20261024_v1/provenance" / stage["id"]
        c.require(
            {p.name for p in folder.glob("*.json")} == expected,
            "Unattributed runtime marker",
        )
        return reviewed

    b.execute_stage = rebound
    b.verify_review = verify_review
    b.WORKERS = WORKERS
    return b


def execute(b):
    validate()
    bind_stage(b, dict(path=c.rel(MANIFEST), sha256=c.sha(MANIFEST)))
    print(
        "Checked 128-request execution for "
        + b.BASE.parent.name
        + ". Source and evaluation audits stay unchanged.",
        flush=True,
    )
    return q.execute(b)


def bound_controller():

    fn = original.run
    proxy = SimpleNamespace(**dict(vars(q), execute=execute))
    return FunctionType(
        fn.__code__,
        dict(fn.__globals__, q=proxy),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


def final_runtime_check():

    done = c.read(c.ITERATIVE / "complete.json")
    c.verify(done["phase_receipts"])
    c.require(
        done["status"] == "all_four_simulated_rounds_audited_scientific_review_pending"
        and done["plan_sha256"] == c.sha(c.ITERATIVE / "plan.json")
        and done["totals"] == {"predictions": 259224, "ordinary_calls": 777672},
        "Iterative completion differs",
    )
    c.require(len(done["phase_receipts"]) == 14, "Fourteen required phases")
    remote = set()
    markers = {}
    counts = Counter()
    accounting = Counter()
    phase_details = []
    amendment = dict(path=c.rel(MANIFEST), sha256=c.sha(MANIFEST))
    for name in done["phase_receipts"]:
        complete = c.ROOT / name
        root = complete.parent.parent.parent
        study = c.read(root / "scientific_plan_v1.json")["conditions"]
        study_name = (
            "source"
            if root.name == "source_initial" or "_source_" in root.name
            else "evaluation"
        )
        b = q.queue(root, study_name, original.SOURCES)
        b.validate(ready=True)
        bind_stage(b, amendment)
        phase = c.read(complete)
        plan_sha = c.sha(b.PLAN)
        schedule = c.read(b.BASE / "schedule.json")
        c.require(
            phase["status"] == "all_studies_executed_and_audited"
            and phase["mode"] == "live"
            and phase["plan_sha256"] == plan_sha,
            "Phase identity differs",
        )
        c.verify(phase["review_hashes"])
        counts.update(phase["counts"])
        accounting.update(phase["accounting"])
        n = 0
        for stage in schedule["stages"]:
            target = b.BASE / "live" / stage["study"] / "stages" / stage["id"]
            review = b.verify_review(
                stage, b.BASE / "live" / stage["study"], plan_sha, "live"
            )
            ids = c.read(target / review["response_ids"])
            c.require(
                len(ids) == len(set(ids)) and not remote.intersection(ids),
                "Remote response identity reused across phases",
            )
            remote.update(ids)
            n += len(ids)
        c.require(n == phase["distinct_remote_ids"], "Phase response count differs")
        for path in (b.BASE / "concurrency_20261024_v1/provenance").rglob("*.json"):
            markers[c.rel(path)] = c.sha(path)
        phase_details.append(
            dict(
                phase=c.rel(root),
                conditions=len(study),
                predictions=phase["counts"]["predictions"],
                distinct_responses=n,
            )
        )
    c.require(
        counts["predictions"] == 259224
        and accounting["ordinary_logical_calls"] == 777672,
        "Total phase budget differs",
    )
    c.require(
        accounting["http_success_with_text"] == len(remote),
        "Global response count differs",
    )
    receipt = dict(
        status="all_four_rounds_and_runtime_provenance_verified_scientific_review_pending",
        original_complete_sha256=c.sha(c.ITERATIVE / "complete.json"),
        manifest_sha256=c.sha(MANIFEST),
        phases=phase_details,
        marker_hashes=markers,
        phase_receipts=done["phase_receipts"],
        counts=dict(counts),
        accounting=dict(accounting),
        global_distinct_response_ids=len(remote),
        actual_extra_API_calls=0,
        scientific_scoring=False,
        cumulative_classifier_started=False,
        next_queue_started=False,
    )
    c.save(BASE / "live_runtime_complete.json", receipt)
    print(
        "FOUR FEEDBACK ROUNDS AND ALL AUDITS COMPLETE. Ask Astra to analyse them. No classifier or next Qwen study started.",
        flush=True,
    )


def run():
    validate()
    original.validate()
    c.resources(2.5, 20.0)
    with a.item_lock(a.storage_path(BASE / "runtime_launcher.lock")):

        bound_controller()()
        final_runtime_check()


def seal():
    c.require(not MANIFEST.exists(), "Preserve the sealed runtime amendment")
    original.validate()
    prerequisites()
    pre = c.read(BASE / "preflight.json")
    tests = c.read(BASE / "tests.json")
    launch = c.read(BASE / "launcher_check.json")
    c.require(
        pre["passed"] and tests["passed"] and launch["parsed"] and launch["restored"],
        "Checks incomplete",
    )
    c.verify(pre["preserved_bindings"])
    c.verify(tests["sources"])
    c.verify(tests["artifacts"])
    paths = [
        *sources(),
        BASE / "tests.json",
        BASE / "preflight.json",
        BASE / "launcher_check.json",
        c.ITERATIVE / "plan.json",
        c.ITERATIVE / "ready.json",
        CAPACITY,
        TRANSFER_REVIEW / "complete.json",
        TRANSFER_REVIEW / "publication/final_check.json",
        RESERVED_REVIEW / "complete.json",
        RESERVED_REVIEW / "publication/final_validation.json",
    ]
    paths.extend(c.ROOT / name for name in pre["supporting_bindings"])
    record = dict(
        status="offline_verified_ready_for_user",
        created_utc=c.utc(),
        original_workers=8,
        effective_workers=WORKERS,
        parent_plan_sha256=c.sha(c.ITERATIVE / "plan.json"),
        api_contract=a.CONTRACT,
        environment=q.template.core.environment(),
        predictions=259224,
        ordinary_calls=777672,
        rounds=4,
        streams=3,
        automatic_batches=38,
        bindings=c.bindings(dict.fromkeys(paths)),
        science_unchanged=True,
        actual_Qwen_calls=0,
        actual_model_list_GETs=0,
        runtime="Original controller with only q.execute rebound; original stage bytecode with WORKERS=128 and separate provenance save.",
        future_phases="Original eight-worker plans and mock audits remain base provenance; every live phase uses this supplemental manifest.",
        restart="Completed raw work is checked and reused. Unknown intents, exhausted retries or preserved stops require review; no silent retry reset.",
        writing="User requests SQ3 writing after all remaining SQ3 experiments and final combined synthesis.",
        required_final_runtime_receipt=c.rel(BASE / "live_runtime_complete.json"),
    )
    c.save(MANIFEST, record)
    validate()
    print(
        "128-request four-round launcher sealed offline. No Qwen call or live phase started.",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "seal", "run"])
    args = ap.parse_args()
    with original.threadpool_limits(limits=1):
        if args.mode == "run":
            run()
        else:
            with c.offline():
                if args.mode == "seal":
                    seal()
                else:
                    validate()
                    original.validate()
                    c.resources(2.5, 20.0)
                    print(
                        "Offline readiness passed: four rounds, 259,224 predictions, 128-request limit. Zero Qwen calls.",
                        flush=True,
                    )
