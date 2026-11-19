from collections import defaultdict

from analysis.classification.broad import run as r

a, c, e = r.a, r.c, r.e


def read_matrix(b, schedule, root, plan_sha, *, mode, expected_count):
    root = a.storage_path(root).resolve()
    done = a.read(root / "complete.json")
    a.require(
        done["mode"] == mode
        and done["plan_sha256"] == plan_sha
        and done["status"] == "all_studies_executed_and_audited",
        "Complete audit required before reading",
    )
    a.require(
        done["counts"]["predictions"] == expected_count, "Completion count differs"
    )
    expected_reviews = {
        r.reference(root / s["study"] / "stages" / s["id"] / "review.json")
        for s in schedule["stages"]
    }
    a.require(
        set(done["review_hashes"]) == expected_reviews, "Final review roster differs"
    )
    r.verify(done["review_hashes"])
    output = defaultdict(dict)
    seen = set()
    result_bindings = {}
    nulls = 0
    for stage in schedule["stages"]:
        store = root / stage["study"]
        target = store / "stages" / stage["id"]
        a.require(
            not list(target.glob("stop_*.json")), "Unreviewed stop blocks analysis"
        )
        b.verify_review(stage, store, plan_sha, mode)
        receipt = a.read(target / "complete.json")
        expected_items = {
            b.item_dir(store, j, plan_sha).relative_to(store).as_posix()
            + "/complete.json"
            for j in stage["jobs"]
        }
        a.require(
            set(receipt["item_receipt_hashes"]) == expected_items,
            "Stage item roster differs",
        )
        for job in stage["jobs"]:
            ident = job["key"], job["condition"], job["repeat"]
            a.require(ident not in seen, "Repeated analysis output")
            seen.add(ident)
            folder = b.item_dir(store, job, plan_sha)
            cp, rp = folder / "complete.json", folder / "result.json"
            receipt_sha, result_sha = c.sha(cp), c.sha(rp)
            a.require(
                receipt_sha
                == receipt["item_receipt_hashes"][cp.relative_to(store).as_posix()],
                "Item receipt changed",
            )
            item = a.read(cp)
            a.require(
                item["status"] == "complete"
                and item["hashes"][rp.relative_to(store).as_posix()] == result_sha,
                "Saved result changed after audit",
            )
            value = a.read(rp)
            a.require(
                value["identity"] == b.identity(job, plan_sha)
                and value["execution_mode"] == mode
                and value["messages_sha256"] == job["messages_sha256"],
                "Scored identity/mode/prompt differs",
            )
            nulls += value["result"] is None
            output[job["condition"], job["repeat"]][job["key"]] = value["result"]
            result_bindings[r.reference(cp)] = receipt_sha
            result_bindings[r.reference(rp)] = result_sha
    a.require(
        len(seen) == expected_count and nulls == done["counts"]["nulls"],
        "Incomplete matrix or changed null count",
    )
    return dict(output), dict(
        predictions=len(seen),
        nulls=nulls,
        result_bindings=result_bindings,
        completion=r.reference(root / "complete.json"),
        completion_sha256=c.sha(root / "complete.json"),
        new_Qwen_calls=0,
    )


def load_predictions():
    "Production entry: requires the full live completion and storage seals."
    with r.original.offline():
        _, plan = r.validate()
        seal = a.read(r.BASE / "storage_complete.json")
        a.require(
            seal["status"] == "broad_D_execution_and_all_audits_complete"
            and seal["source_plan_sha256"] == r.original.PLANS["broad"]
            and seal["amendment"] == r.amendment()
            and seal["workers"] == 128,
            "Storage completion differs",
        )
        done_path = r.BASE / "live/complete.json"
        concurrency_path = r.BASE / "concurrency_20261029_v1/live_complete.json"
        a.require(
            c.sha(done_path) == seal["original_completion_sha256"]
            and c.sha(concurrency_path) == seal["concurrency_completion_sha256"],
            "Final completion binding differs",
        )
        concurrency = a.read(concurrency_path)
        a.require(
            concurrency["amendment"] == r.amendment()
            and concurrency["original_completion_sha256"] == c.sha(done_path)
            and concurrency["workers"] == 128,
            "Runtime completion differs",
        )
        r.verify(concurrency["provenance_hashes"])
        b = r.queue(r.amendment())
        output, provenance = read_matrix(
            b,
            a.read(r.SCHEDULE),
            r.BASE / "live",
            c.sha(r.PLAN),
            mode="live",
            expected_count=plan["budget"]["predictions"],
        )
        provenance.update(
            execution_plan=r.reference(r.PLAN),
            plan_sha256=c.sha(r.PLAN),
            storage_manifest_sha256=c.sha(r.MANIFEST),
            reader="D: audited reader; original scientific scorer unchanged",
        )
        return output, provenance
