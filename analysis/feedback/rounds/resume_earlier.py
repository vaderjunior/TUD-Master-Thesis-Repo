from analysis.common.source import replace_source, insert_before_assignment

import argparse
import inspect
from pathlib import Path
from types import FunctionType, SimpleNamespace
from analysis.feedback.rounds import outage_inspect as inspection

runtime = inspection.runtime
c, q, a = (runtime.c, runtime.q, runtime.a)
BASE, PHASE, STORE, STAGE = (
    inspection.BASE,
    inspection.PHASE,
    inspection.STORE,
    inspection.STAGE,
)
MANIFEST, READY = (BASE / "manifest.json", BASE / "ready.json")
SUPPORT = c.ROOT / "analysis/feedback/reserved/transport.py"
LAUNCHER = c.ROOT / "scripts/resume_feedback_rounds_initial.ps1"


def replace(text, before, after, count=1):
    try:
        return replace_source(text, before, after, count)
    except ValueError as error:
        a.require(False, str(error))


def transport_namespace():
    ns = dict(
        __name__=__name__ + "_transport", __package__=__package__, __file__=str(SUPPORT)
    )
    exec(compile(SUPPORT.read_text(encoding="utf8"), str(SUPPORT), "exec"), ns)
    ns.update(BASE=PHASE / "execution", HERE=BASE, MANIFEST=MANIFEST, READY=READY)
    return SimpleNamespace(**ns)


transport = transport_namespace()


def provenance():
    ctx = transport.context()
    return dict(
        manifest=ctx.get("manifest_path", inspection.rel(MANIFEST)),
        manifest_sha256=ctx["manifest_sha256"],
        supplemental_source_hashes=ctx["source_hashes"],
        reviewed_calls=len(ctx["calls"]),
        extra_attempts_per_call=5,
        scope="Only manifest-listed store/item/vote/turn tuples may use attempts 5..9; all other calls retain 0..4.",
        timing="All old failures remain counted. Pause time is not request latency; remote work after transport errors is unknown.",
        original_stamp_qualification="Original stamps and 128-worker markers identify the unchanged base; this manifest and recovery markers identify the added execution and audit rules.",
    )


class Scope:

    def verify_records(self):
        for entry in transport.context()["calls"]:
            transport.verify_files(entry["prior_hashes"])

    def verify_stops(self, target):
        transport.verify_stops(target.parent.parent, target)

    def history(self, target):
        self.verify_stops(target)
        return transport.audit_history(target.parent.parent, target)

    def enrich_report(self, report, stage, hashes):
        self.verify_records()
        a.require(
            stage["id"] == transport.context()["stage"],
            "Recovery applied outside reviewed batch",
        )
        report["reviewed_transport_amendment"] = provenance()
        root = a.storage_path(c.ROOT / transport.context()["calls"][0]["store"])
        for entry in transport.context()["calls"]:
            for name, sha in entry["prior_hashes"].items():
                key = a.storage_path(c.ROOT / name).relative_to(root).as_posix()
                a.require(
                    hashes.get(key) == sha,
                    "Preserved outage prefix absent from final audit",
                )
        report["limits"].append(
            "Only the manifest-listed outage calls have a ten-attempt total cap; no retry budget is reset."
        )

    def enrich_queue(self, result):
        self.verify_records()
        result["reviewed_transport_amendment"] = provenance()


def marker_path(b, attempt):
    return (
        b.BASE
        / "recovery_20261024_v1/provenance"
        / attempt.parent.name
        / (attempt.name + ".json")
    )


def apply_recovery(b):

    a.require(b.WORKERS == 128, "Reviewed concurrency required")
    scope = Scope()
    ns = dict(b.execute_stage.__globals__)
    ns.update(transport=transport, recovery=scope)
    old_save = ns["save"]

    def save(path, value):
        old_save(path, value)
        path = Path(str(path).removeprefix("\\\\?\\"))
        if path.name == "start.json" and path.parent.name.startswith("attempt_"):
            old_save(
                marker_path(b, path.parent),
                dict(
                    start_sha256=a.digest(path),
                    original_attempt=inspection.rel(path.parent),
                    stage=value["stage"],
                    mode=value["mode"],
                    plan_sha256=value["plan_sha256"],
                    recovery=provenance(),
                ),
            )

    ns["save"] = save
    for name in ("engines", "scan_jobs", "execute_stage"):
        source = inspect.getsource(getattr(b, name))
        if name == "engines":
            source = replace(
                source,
                "isolated.adapter(ROOT/v['schema_path'])",
                "transport.adapter(ROOT/v['schema_path'])",
            )
        elif name == "scan_jobs":
            source = replace(
                source,
                "action, reason, _ = a.interpret(out)\n                if action == 'stop' or action == 'retry' and intent['attempt'] == 4: raise a.CallStopped('Preserved terminal request: '+reason)",
                "transport.partial_outcome(root,d,intent,out)",
            )
        else:
            source = replace(
                source,
                "a.require(not list(target.glob('stop_*.json')), 'Preserved execution stop requires review')",
                "recovery.verify_stops(target)",
                2,
            )
        exec(compile(source, __file__ + "[" + name + "]", "exec"), ns)
    for name in ("engines", "scan_jobs", "execute_stage"):
        setattr(b, name, ns[name])
    original_review = b.verify_review

    def verify_review(stage, root, plan_sha, mode):
        target = a.storage_path(root) / "stages" / stage["id"]
        scope.verify_stops(target)
        record = original_review(stage, root, plan_sha, mode)
        report = a.read(target / record["report"])
        a.require(
            report["reviewed_transport_amendment"] == provenance(),
            "Recovery audit provenance differs",
        )
        expected = set()
        for attempt in target.glob("attempt_*"):
            if (
                inspection.rel(attempt / "summary.json")
                in transport.context()["reviewed_failed_summaries"]
            ):
                continue
            marker = marker_path(b, attempt)
            expected.add(marker.name)
            a.require(
                a.read(marker)
                == dict(
                    start_sha256=a.digest(attempt / "start.json"),
                    original_attempt=inspection.rel(attempt),
                    stage=stage["id"],
                    mode=mode,
                    plan_sha256=plan_sha,
                    recovery=provenance(),
                ),
                "Recovery attempt marker differs",
            )
        a.require(
            {
                p.name
                for p in (
                    b.BASE / "recovery_20261024_v1/provenance" / stage["id"]
                ).glob("*.json")
            }
            == expected,
            "Unattributed recovery marker",
        )
        return record

    b.verify_review = verify_review
    source = (c.ROOT / "analysis/common/vote_audit.py").read_text(encoding="utf8")
    source = replace(source, "from . import astra_sq3_balanced_api_v1 as b", "")
    source = replace(
        source,
        "0 <= attempt <= 4, 'Retry limit differs'",
        "0 <= attempt <= 9, 'Retry limit differs'",
    )
    source = replace(
        source,
        "a.CONTRACT['backoff_seconds'][attempt-1]",
        "transport.reviewed_backoff(attempt)",
    )
    source = replace(
        source,
        "a.require(not list(target.glob('stop_*.json')), 'Preserved execution stop blocks audit')",
        "recovery.verify_stops(target)",
    )
    source = replace(
        source,
        "starts=[]; history={}; model_checks=0",
        "starts=[]; history=recovery.history(target); model_checks=0",
    )
    source = replace(
        source,
        "a.require(not s['errors'] and not s['stopped'],'Failed attempt preserved; requires review')",
        "transport.verify_attempt_summary(attempt,s)",
    )
    source = replace(
        source,
        "previous=timing(intent,outcome,start,end,previous,live=mode=='live')",
        "a.require(intent['attempt'] in transport.reviewed_attempts(root,folder,intent['vote'],intent['turn']), 'Attempt outside reviewed scope')\n                previous=timing(intent,outcome,start,end,previous,live=mode=='live')",
    )
    source = replace(
        source,
        "    # Failed or interrupted audit attempts are preserved",
        "    recovery.enrich_report(report,stage,hashes)\n    # Failed or interrupted audit attempts are preserved",
    )
    source = replace(
        source,
        "    b.save(live_root/'complete.json',result)",
        "    recovery.enrich_queue(result)\n    b.save(live_root/'complete.json',result)",
    )
    audit_ns = dict(
        __name__=__name__ + "_audit",
        __file__=__file__,
        __package__=__package__,
        b=b,
        transport=transport,
        recovery=scope,
    )
    exec(compile(source, __file__ + "[audit]", "exec"), audit_ns)
    b.audit = SimpleNamespace(**audit_ns)
    return b


def original_amendment():
    return dict(path=c.rel(runtime.MANIFEST), sha256=c.sha(runtime.MANIFEST))


def bind_for_phase(b, amendment):
    runtime.bind_stage(b, amendment)
    if (
        a.storage_path(b.BASE).resolve()
        == a.storage_path(PHASE / "execution").resolve()
    ):
        apply_recovery(b)
    return b


def validate(*, ready=True):
    runtime.validate()
    runtime.original.validate()
    m = a.read(MANIFEST)
    a.require(
        m["status"] == "reviewed_iterative_connection_outage_amendment"
        and m["phase"] == c.rel(PHASE)
        and (m["stage"] == STAGE),
        "Wrong recovery scope",
    )
    a.require(
        m["parent_runtime_sha256"] == a.digest(runtime.MANIFEST)
        and m["parent_scientific_plan_sha256"] == a.digest(c.ITERATIVE / "plan.json")
        and (m["original_plan_sha256"] == a.digest(PHASE / "execution/plan.json")),
        "Original plan changed",
    )
    a.require(
        len(m["calls"]) == 128
        and len({(e["store"], e["item"], e["vote"], e["turn"]) for e in m["calls"]})
        == 128
        and (m["extra_attempts_per_call"] == 5)
        and (m["maximum_extra_attempts"] == 640)
        and (m["effective_workers"] == 128)
        and (m["api_contract"] == a.CONTRACT),
        "Recovery size or contract differs",
    )
    transport.verify_files(m["source_hashes"])
    transport.verify_files(m["preserved_hashes"])
    transport.verify_files(m["evidence_hashes"])
    a.require(
        a.read(BASE / "tests_v2.json")["passed"]
        and a.read(BASE / "real_prefix_rehearsal.json")["passed"],
        "Offline recovery checks incomplete",
    )
    diag = a.read(BASE / "diagnosis.json")
    for key in (
        "calls",
        "preserved_hashes",
        "reviewed_stop_files",
        "reviewed_failed_summaries",
        "original_plan_sha256",
    ):
        a.require(m[key] == diag[key], "Recovery differs from reviewed outage")
    transport.activate(m, a.digest(MANIFEST))
    Scope().verify_stops(a.storage_path(STORE / "stages" / STAGE))
    if ready:
        r = a.read(READY)
        a.require(
            r["status"] == "offline_verified_ready_for_user"
            and r["manifest_sha256"] == a.digest(MANIFEST),
            "Recovery not sealed",
        )
        transport.verify_files(r["verification_hashes"])
    return m


def execute(b):
    runtime.validate()
    transport.verify_files(transport.context()["source_hashes"])
    bind_for_phase(b, original_amendment())
    return q.execute(b)


def bound_controller():
    fn = runtime.original.run
    proxy = SimpleNamespace(**dict(vars(q), execute=execute))
    return FunctionType(
        fn.__code__,
        dict(fn.__globals__, q=proxy),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


def final_check():
    fn = runtime.final_runtime_check
    rebound = FunctionType(
        fn.__code__,
        dict(fn.__globals__, bind_stage=bind_for_phase),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    rebound()
    Scope().verify_records()
    phase_complete = a.read(PHASE / "execution/live/complete.json")
    a.require(
        phase_complete["reviewed_transport_amendment"] == provenance(),
        "Recovered phase final provenance differs",
    )
    marker_dir = PHASE / "execution/recovery_20261024_v1/provenance"
    c.save(
        BASE / "live_recovery_complete.json",
        dict(
            status="all_four_rounds_and_reviewed_outage_audited_scientific_review_pending",
            original_complete_sha256=a.digest(c.ITERATIVE / "complete.json"),
            runtime_complete_sha256=a.digest(
                runtime.BASE / "live_runtime_complete.json"
            ),
            recovery=provenance(),
            recovered_phase_complete_sha256=a.digest(
                PHASE / "execution/live/complete.json"
            ),
            recovery_marker_hashes=inspection.inventory(marker_dir),
            actual_finalization_API_calls=0,
        ),
    )
    print(
        "Feedback rounds and recovery audits complete. Ask Astra to analyse the saved results.",
        flush=True,
    )


def run():
    with a.item_lock(a.storage_path(runtime.BASE / "runtime_launcher.lock")):
        validate()
        c.resources(2.5, 20.0)
        bound_controller()()
        final_check()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "run"], nargs="?", default="validate")
    args = ap.parse_args()
    with runtime.original.threadpool_limits(limits=1):
        if args.mode == "run":
            run()
        else:
            with inspection.offline():
                validate()
                c.resources(2.5, 20.0)
                print(
                    "Recovery verified offline: 505 completed predictions preserved; only 128 reviewed calls gain bounded continuation. Zero Qwen calls.",
                    flush=True,
                )
