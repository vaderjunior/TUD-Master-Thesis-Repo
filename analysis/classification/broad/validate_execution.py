from collections import Counter
import gc
from pathlib import Path
import time

from analysis.classification.broad import run as r

a, e, c = r.a, r.e, r.c


def main():
    with r.original.offline():
        manifest, plan = r.validate()
        preflight = a.read(r.LOCAL / "preflight.json")
        proof = preflight["rehearsal"]
        root = r.resolve_reference(proof["root"])
        stages = proof["stages"]
        final = a.read(root / "complete.json")
        counts = Counter()
        responses = set()
        for stage in stages:
            store = root / "broad"
            target = store / "stages" / stage["id"]
            review = a.read(target / "review.json")
            complete = a.read(target / "complete.json")
            a.require(
                c.sha(target / "complete.json") == review["complete_sha256"],
                "Stage binding differs",
            )
            for name, digest in review["artifacts"].items():
                a.require(c.sha(target / name) == digest, "Audit artifact differs")
            report = a.read(target / review["report"])
            ids = a.read(target / review["response_ids"])
            a.require(
                not responses.intersection(ids) and len(set(ids)) == len(ids),
                "Duplicate synthetic response",
            )
            responses.update(ids)
            for name, digest in a.read(target / review["raw_hashes"]).items():
                a.require(c.sha(store / name) == digest, "Raw D: evidence changed")
                counts["raw_files"] += 1
            final_attempt = r.resolve_reference(complete["final_attempt"])
            a.require(
                c.sha(final_attempt / "summary.json") == complete["summary_sha256"],
                "Cross-drive attempt cannot resolve",
            )
            a.require(
                r.normal(final_attempt).is_relative_to(r.BASE), "Attempt is not on D:"
            )
            a.require(report["actual_audit_API_calls"] == 0, "Audit made network calls")
            counts["predictions"] += report["counts"]["predictions"]
            counts["ordinary_calls"] += report["accounting"]["ordinary_logical_calls"]
        a.require(
            counts["predictions"]
            == proof["predictions"]
            == final["counts"]["predictions"],
            "Synthetic count differs",
        )
        a.require(
            counts["ordinary_calls"] == len(responses) == proof["synthetic_responses"],
            "Response accounting differs",
        )
        for name, digest in final["review_hashes"].items():
            a.require(
                c.sha(r.resolve_reference(name)) == digest,
                "Final cross-drive review reference differs",
            )

        marker = dict(
            path=r.reference(r.PROTOCOL), sha256=c.sha(r.PROTOCOL), synthetic=True
        )
        b = r.queue(marker, base=r.resolve_reference(proof["queue_base"]))
        cat = r.original.load_catalog("broad")
        started = time.perf_counter()
        restarted = r.runtime().execute_schedule(
            b,
            {"stages": stages},
            cat,
            root,
            c.sha(r.PLAN),
            mode="mock",
            factory=c.forbidden,
        )
        restart_seconds = time.perf_counter() - started
        a.require(restarted == final, "Independent completed restart changed evidence")
        del cat
        gc.collect()
        c.verify(a.read(r.LOCAL / "preservation_before.json"))
        host = r.check()
        a.require(
            not (r.BASE / "live").exists() and not (r.SOURCE_BASE / "live").exists(),
            "Scientific queue was started",
        )
        value = dict(
            status="independent_D_readiness_review_passed",
            created_utc=a.utc(),
            manifest_sha256=c.sha(r.MANIFEST),
            raw_files_rechecked=counts["raw_files"],
            synthetic_predictions=counts["predictions"],
            synthetic_responses=len(responses),
            extra_requests_on_independent_restart=0,
            independent_restart_seconds=restart_seconds,
            preserved_bindings=len(a.read(r.LOCAL / "preservation_before.json")),
            final_offline_check=host,
            source_sha256=c.sha(Path(__file__)),
            actual_Qwen_calls=0,
            actual_model_list_GETs=0,
            GPU_accesses=0,
            old_outputs_moved=False,
            scientific_live_directories_absent=True,
        )
        r.save(r.LOCAL / "independent_final_review.json", value)
        print(
            "FINAL D: REVIEW PASSED. All scientific work remains unstarted; command may be handed over.",
            flush=True,
        )


if __name__ == "__main__":
    main()
