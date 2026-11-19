from analysis.common.source import replace_source, insert_before_assignment

import argparse
import ast
import gc
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from analysis.feedback.reserved import recovery as shutdown
from analysis.feedback.reserved import transport as transport

original = shutdown.original
a, c, ROOT, BASE = (shutdown.a, shutdown.c, shutdown.ROOT, shutdown.BASE)
HERE = BASE / "recovery_20261016_v2"
MANIFEST, READY = (HERE / "manifest.json", HERE / "ready.json")
STAGE = "reserved_slot_09"


def replace(text, before, after, count=1):
    try:
        return replace_source(text, before, after, count)
    except ValueError as error:
        a.require(False, str(error))


class CombinedScope:

    def __init__(self, previous):
        self.previous = previous
        self.applied = (
            previous.applied
            if previous is not None
            else {"administrative_outcomes": {}}
        )

    def verify_records(self):
        if self.previous is not None:
            self.previous.verify_records()
        for entry in transport.context()["calls"]:
            transport.verify_files(entry["prior_hashes"])

    def verify_stops(self, target):
        transport.verify_stops(target.parent.parent, target)

    def history(self, target):
        self.verify_stops(target)
        return transport.audit_history(target.parent.parent, target)

    def enrich_report(self, report, stage, hashes):
        self.verify_records()
        if self.previous is not None:
            self.previous.enrich_report(report, stage, hashes)
        else:
            report["administrative_unknown_requests"] = 0
        if stage["id"] == transport.context()["stage"]:
            report["reviewed_transport_amendment"] = transport.provenance()
            root = a.storage_path(ROOT / transport.context()["calls"][0]["store"])
            for entry in transport.context()["calls"]:
                for name, digest in entry["prior_hashes"].items():
                    key = a.storage_path(ROOT / name).relative_to(root).as_posix()
                    a.require(
                        hashes.get(key) == digest,
                        "Outage prefix omitted from final raw audit",
                    )
            report["limits"].append(
                "Eight reviewed outage calls alone have a ten-attempt total cap. Old failures remain counted; restart downtime is not request latency."
            )

    def enrich_queue(self, result):
        self.verify_records()
        if self.previous is not None:
            self.previous.enrich_queue(result)
        else:
            result["administrative_unknown_requests"] = 0
        result["reviewed_transport_amendment"] = transport.provenance()


def wrapped_queue(previous):

    b = original.queue()
    scope = CombinedScope(previous)
    ns = dict(b.execute_stage.__globals__)
    ns.update(transport=transport, recovery=scope)
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
            source = replace(
                source,
                "existing.append((j, a.read(d/'result.json'))); continue",
                "existing.append((j, a.read(d/'result.json')))\n            if len(existing)%250 == 0: print(f'Verified {len(existing)}/{len(jobs)} already saved predictions; no new request for these.', flush=True)\n            continue",
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
    unmodified_review = b.verify_review

    def verify_review(stage, root, plan_sha, mode):
        target = a.storage_path(root) / "stages" / stage["id"]
        scope.verify_stops(target)
        record = unmodified_review(stage, root, plan_sha, mode)
        report = a.read(target / record["report"])
        if previous is not None and stage["id"] == previous.m["stage"]:
            a.require(
                report["reviewed_interruption_recovery"] == previous.provenance(),
                "Earlier shutdown provenance changed",
            )
            previous.verify_records()
        if stage["id"] == transport.context()["stage"]:
            a.require(
                report["reviewed_transport_amendment"] == transport.provenance(),
                "Transport audit provenance changed",
            )
        return record

    b.verify_review = verify_review
    source = (ROOT / "analysis/common/vote_audit.py").read_text(encoding="utf8")
    source = replace(source, "from analysis.common import vote_api as b", "")
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
    source = insert_before_assignment(
        source, "audit_folder", "recovery.enrich_report(report,stage,hashes)"
    )
    source = replace(
        source,
        "    b.save(live_root/'complete.json',result)",
        "    recovery.enrich_queue(result)\n    b.save(live_root/'complete.json',result)",
    )
    audit_ns = {
        "__name__": __name__ + "_audit",
        "__file__": __file__,
        "__package__": __package__,
        "b": b,
        "transport": transport,
        "recovery": scope,
    }
    exec(compile(source, __file__ + "[combined audit]", "exec"), audit_ns)
    b.audit = SimpleNamespace(**audit_ns)
    b.recovery = scope
    return b


def verify_manifest(m):
    a.require(
        m["status"] == "reviewed_reserved_connection_outage_amendment"
        and m["stage"] == STAGE,
        "Wrong outage recovery",
    )
    a.require(
        m["original_plan_sha256"] == a.digest(BASE / "plan.json"),
        "Original execution plan changed",
    )
    a.require(
        len(m["calls"]) == 8
        and m["extra_attempts_per_call"] == 5
        and (m["maximum_extra_attempts"] == 40),
        "Outage amendment size changed",
    )
    a.require(
        len({(e["store"], e["item"], e["vote"], e["turn"]) for e in m["calls"]}) == 8,
        "Duplicate outage call",
    )
    transport.verify_files(m["source_hashes"])
    transport.verify_files(m["preserved_hashes"])
    transport.verify_files(m["previous_recovery_hashes"])
    a.require(
        m["tests_sha256"] == a.digest(HERE / "tests.json")
        and a.read(HERE / "tests.json")["passed"],
        "Recovery tests changed",
    )
    a.require(
        m["rehearsal_sha256"] == a.digest(HERE / "rehearsal.json")
        and a.read(HERE / "rehearsal.json")["passed"],
        "Recovery rehearsal changed",
    )


def scope_from_disk(*, ready=True):
    previous = shutdown.scope_from_disk()
    m = a.read(MANIFEST)
    verify_manifest(m)
    a.require(
        m["previous_recovery_provenance"] == previous.provenance(),
        "Earlier recovery binding changed",
    )
    transport.activate(m, a.digest(MANIFEST))
    scope = CombinedScope(previous)
    scope.verify_records()
    for target in a.storage_path(BASE / "live/reserved_slot/stages").iterdir():
        a.require(target.is_dir(), "Unreviewed stage entry")
        scope.verify_stops(target)
    if ready:
        r = a.read(READY)
        a.require(
            r["status"] == "offline_verified_ready_for_user"
            and r["manifest_sha256"] == a.digest(MANIFEST),
            "Recovery not ready",
        )
        for name, sha in r["verification_hashes"].items():
            a.require(a.digest(ROOT / name) == sha, "Recovery verification changed")
        transport.verify_files(r["analysis_source_hashes"])
    return scope


def run_all():
    with a.item_lock(a.storage_path(BASE / "combined_launcher.lock")):
        original.resource_guard(live=True)
        scope = scope_from_disk()
        b = wrapped_queue(scope.previous)
        schedule = a.read(BASE / "schedule.json")
        plan_sha = a.digest(b.PLAN)
        cat = b.load_catalog(original.STUDY)
        root = BASE / "live" / original.STUDY

        def run(stage):
            b.validate(ready=True)
            transport.verify_files(transport.context()["source_hashes"])
            a.require(
                original.shutil.disk_usage(ROOT).free / 2**30 >= 4,
                "Disk below 4 GiB; preserve all outputs",
            )
            b.execute_stage(
                stage,
                cat,
                root,
                plan_sha,
                mode="live",
                client_factory=original.core.live_client,
            )

        b.run_sequence(
            schedule["stages"],
            run,
            lambda stage: b.audit.audit_stage(stage, cat, root, plan_sha, mode="live"),
            lambda stage: (
                a.storage_path(root) / "stages" / stage["id"] / "review.json"
            ).exists(),
            lambda stage: b.verify_review(stage, root, plan_sha, "live"),
        )
        del cat
        gc.collect()
        b.validate(ready=True)
        b.audit.finalize_queue(schedule, BASE / "live", plan_sha, mode="live")
    print(
        "ALL SIXTEEN BATCHES AND AUTOMATIC AUDITS COMPLETE. Both recovery histories are included. No next study started.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["validate", "run-all"], nargs="?", default="validate"
    )
    args = parser.parse_args()
    if args.mode == "run-all":
        run_all()
        return
    with c.offline(), patch.object(
        original.core, "live_client", c.forbidden
    ), patch.object(a.RequestsTransport, "__call__", c.forbidden):
        scope_from_disk()
        print(
            "Both recoveries, frozen sources, preserved evidence and offline readiness verified. No network calls.",
            flush=True,
        )


if __name__ == "__main__":
    main()
