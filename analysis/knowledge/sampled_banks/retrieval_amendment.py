import argparse
import io
from pathlib import Path
import socket
import time
import unittest
from unittest.mock import patch
from analysis.knowledge.sampled_banks import retrieval as p

r = p.r
AMENDMENT = p.BASE / "audit_amendment_v2.json"


def sources():
    return [
        Path(__file__),
        p.ROOT / "analysis/knowledge/sampled_banks/storage.py",
        p.ROOT / "analysis/knowledge/sampled_banks/validate_retrieval.py",
        p.ROOT / "tests/knowledge/sampled_banks/test_storage.py",
        p.ROOT / "scripts/prepare_sampled_retrieval_amendment.ps1",
    ]


def validate():
    plan = p.validate()
    amendment = r.read(AMENDMENT)
    r.require(
        amendment["retrieval_plan_sha256"] == r.sha(p.PLAN),
        "Amendment retrieval plan changed",
    )
    r.verify_hashes(amendment["hashes"])
    r.require(
        all(
            (amendment["hashes"].get(r.rel(path)) == r.sha(path) for path in sources())
        ),
        "Unbound amendment source",
    )
    return (plan, amendment)


def freeze():
    p.validate()
    r.require(not AMENDMENT.exists(), "Amendment already frozen")
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        [
            "tests.knowledge.sampled_banks.test_retrieval",
            "tests.knowledge.sampled_banks.test_storage",
        ]
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    test = p.BASE / "development" / ("amendment_tests_" + str(time.time_ns()) + ".json")
    p.save(
        test,
        {
            "status": "passed" if result.wasSuccessful() else "failed",
            "tests_run": result.testsRun,
            "output": stream.getvalue(),
            "source_hashes": {r.rel(path): r.sha(path) for path in sources()},
        },
    )
    print(stream.getvalue(), flush=True)
    r.require(
        result.wasSuccessful() and result.testsRun == 23, "Amendment tests failed"
    )
    failures = sorted((p.BASE / "logs").glob("smoke_audit_*.txt"))
    r.require(
        len(failures) == 1
        and "Copied vector graph changed after queries"
        in failures[0].read_text(encoding="utf8"),
        "Missing original failed audit log",
    )
    evidence = p.BASE / "development/storage_diagnosis_v2.json"
    from analysis.knowledge.sampled_banks.storage import amended_graph_check

    details = []
    hashes = {
        r.rel(path): r.sha(path) for path in sources() + [test, failures[0], p.PLAN]
    }
    for job in r.read(p.PLAN)["jobs"]:
        a = p.receipt(job["pool_id"], "smoke")
        b = p.receipt(job["pool_id"], "smoke_replay")
        r.require(a is not None and b is not None, "Missing preserved smoke pair")
        src = p.ROOT / a["attempt"] / "chroma"
        dst = p.ROOT / b["attempt"] / "chroma"
        old = {
            str((p.ROOT / path).relative_to(src)): val
            for path, val in a["index_hashes"].items()
        }
        new = {
            str((p.ROOT / path).relative_to(dst)): val
            for path, val in b["index_hashes"].items()
        }
        r.require(old.keys() == new.keys(), "Copied index file roster changed")
        changed = [name for name in old if old[name] != new[name]]
        note = amended_graph_check(
            b, r.rows(p.ROOT / job["records"]), src, dst, changed
        )
        details.append(
            {
                "pool_id": job["pool_id"],
                "changed_files": changed,
                "persistence_check": note,
                "replay_differences": b["replay_differences"],
            }
        )
        for receipt in (a, b):
            path = p.receipt_path(job["pool_id"], receipt["kind"])
            hashes[r.rel(path)] = r.sha(path)
            hashes.update(receipt["hashes"])
            hashes.update(receipt["index_hashes"])
    p.save(
        evidence,
        {
            "status": "preserved_smoke_storage_diagnosed",
            "by_pool": details,
            "sources_consulted": [
                "https://github.com/chroma-core/chroma/blob/1.5.9/rust/segment/src/local_hnsw.rs",
                "https://github.com/chroma-core/hnswlib/blob/master/hnswlib/hnswalg.h",
            ],
            "source_qualification": "Chroma tag matches installed 1.5.9. HNSW master is format corroboration, not a claim that it is the exact wheel build. All decisions additionally require exact observed local header, WAL, SQL, vectors and physical hashes.",
        },
    )
    hashes[r.rel(evidence)] = r.sha(evidence)
    p.save(
        AMENDMENT,
        {
            "status": "audit_only_amendment_frozen_preflight_pending",
            "created_utc": p.utc(),
            "retrieval_plan_sha256": r.sha(p.PLAN),
            "tests": r.rel(test),
            "hashes": hashes,
            "reason": "v1 audit incorrectly required byte-identical empty graph capacity files for 340-record pools below sync_threshold 1000. No metadata pickle or persisted vector sequence exists; all 340 ordered inserts remain in WAL. v2 validates this narrow state, complete WAL vectors/text, exact logical SQL except lock bookkeeping, identical stored-vector exports, unchanged header/link files and all original ranking/prompt checks. Noncompact graph equality remains strict.",
            "preserved": "Original frozen retrieval/checker/launcher, smoke attempts, all first/replay prompts and failure log. No new retrieval, altered settings, package upgrade, outcome access or API calls.",
            "execution": "Use v2 launcher only; it calls unchanged v1 workers and uses v2 audit destinations preflight_v2.json/full_audit_v2.json. Scientific design and v1 retrieval plan remain authoritative.",
        },
    )
    validate()
    print(
        "AUDIT AMENDMENT FROZEN. Original retrieval sources and smoke evidence unchanged.",
        flush=True,
    )


def run(mode):
    with p.lock():
        plan, amendment = validate()
        if mode == "run":
            ready = r.read(p.BASE / "preflight_v2.json")
            r.require(
                ready["status"] == "ready_for_user_full_retrieval_not_API"
                and ready["plan_sha256"] == r.sha(p.PLAN)
                and (ready["audit_amendment_sha256"] == r.sha(AMENDMENT)),
                "Reviewed v2 preflight required",
            )
            r.verify_hashes(ready["hashes"])
            for job in plan["jobs"]:
                for kind in ("prepare", "replay"):
                    if p.receipt(job["pool_id"], kind):
                        print(
                            kind + " " + job["condition"] + ": verified and skipped",
                            flush=True,
                        )
                        continue
                    r.require(
                        not any(
                            p.receipt_path(job["pool_id"], kind).parent.glob(
                                "attempt_*"
                            )
                        ),
                        "Incomplete attempt preserved: ask Astra to inspect",
                    )
                    log, seconds = p.child(
                        [p.MODULE, "worker", "--pool", job["pool_id"], "--kind", kind],
                        job["pool_id"] + "_" + kind,
                    )
                    p.seal(job["pool_id"], kind, log, seconds)
            p.assemble(plan)
        else:
            for job in plan["jobs"]:
                r.require(
                    p.receipt(job["pool_id"], "smoke")
                    and p.receipt(job["pool_id"], "smoke_replay"),
                    "Preserved smoke missing",
                )
        p.child(
            [
                "analysis.knowledge.sampled_banks.validate_retrieval",
                "full" if mode == "run" else "smoke",
            ],
            mode + "_audit_v2",
        )
        validate()
        print(
            (
                "ALL POOL PROMPTS, COLD REPLAYS AND OFFLINE CHECKS COMPLETE. Tell Astra to inspect. No Qwen calls were made."
                if mode == "run"
                else "PRESERVED SMOKE EVIDENCE CHECKED WITH V2 AUDIT; NO RETRIEVAL RERUN."
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "validate", "smoke", "run"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", p.no_network), patch.object(
        socket, "create_connection", p.no_network
    ):
        if args.mode == "freeze":
            freeze()
        elif args.mode == "validate":
            validate()
            print(
                "Frozen original plan and v2 audit amendment verified. No retrieval started."
            )
        else:
            run(args.mode)


if __name__ == "__main__":
    main()
