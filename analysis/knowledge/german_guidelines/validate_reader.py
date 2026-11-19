from collections import Counter, defaultdict
import json
from pathlib import Path
from analysis.knowledge.german_guidelines import common as c
from analysis.knowledge.german_guidelines import reader as r
from analysis.common import schema as isolated


def verify(manifest_path):
    with r.offline():
        manifest = c.read(manifest_path)
        r.require(
            manifest["status"] == "reader_cache_ready_for_independent_check",
            "Reader build incomplete",
        )
        c.verify(manifest["source_hashes"])
        c.verify(manifest["outputs"])
        run = c.ROOT / manifest["run"]
        plan_sha = c.sha(c.PLAN)
        schedule = c.read(c.BASE / "schedule.json")
        done = c.read(r.LIVE / "complete.json")
        c.verify(done["review_hashes"])
        jobs = {}
        hashes = {}
        for stage in schedule["stages"]:
            review = c.read(r.LIVE / "batches" / stage["id"] / "review.json")
            hashes.update(review["items"])
            for job in stage["jobs"]:
                key = (job["key"], job["condition"], job["repeat"])
                r.require(key not in jobs, "Duplicate schedule identity")
                jobs[key] = (job, stage["id"])
        backend = isolated.adapter(c.ROOT / "config/taxonomy.yaml")
        bundle = backend.bundle
        counts, cells = (Counter(), defaultdict(Counter))
        seen, null_ids, duplicate_ids, validity = (set(), [], [], defaultdict(dict))
        for row in c.rows(run / "predictions.jsonl"):
            key = (row["key"], row["condition"], row["repeat"])
            r.require(
                key in jobs and key not in seen, "Unexpected/duplicate cached row"
            )
            seen.add(key)
            job, stage = jobs[key]
            ident = {
                "experiment_sha256": plan_sha,
                "key": key[0],
                "condition": key[1],
                "repeat": key[2],
            }
            stamp = c.signature(ident)
            r.require(
                row["identity"] == ident
                and row["phase"] == job["population"]
                and (row["stage"] == stage),
                "Cache identity differs",
            )
            original_value = {name: row[name] for name in r.VALUE_FIELDS}
            r.require(
                c.signature(original_value)
                == hashes[stamp]["result.json"]
                == row["evidence"]["result_sha256"],
                "Derived cache does not reconstruct audited result bytes",
            )
            r.require(
                row["evidence"]["receipt_sha256"] == hashes[stamp]["complete.json"],
                "Wrong item receipt",
            )
            expected_path = (r.LIVE / "items" / stamp / "result.json").as_posix()
            r.require(
                row["evidence"]["result_path"] == expected_path, "Wrong raw result path"
            )
            expected_job = {
                "identity": ident,
                "messages_sha256": job["messages_sha256"],
                "contract": c.a.CONTRACT,
                "implementation": backend.implementation(),
                "execution_mode": "live",
            }
            r.require(
                c.signature(expected_job) == row["evidence"]["job_sha256"],
                "Job-source binding differs",
            )
            votes = [
                bundle.parser.RunResult(
                    result=None if v is None else bundle.schema.Result(**v)
                )
                for v in row["run_predictions"]
            ]
            output = bundle.voter.aggregate(votes)
            r.require(
                row["result"]
                == (output.result.model_dump() if output.result is not None else None)
                and row["n_valid"] == output.n_valid
                and (row["uncertain"] == output.uncertain)
                and (row["agreement"] == output.agreement),
                "Frozen voter/cache mismatch",
            )
            counts["predictions"] += 1
            counts["null_results"] += row["result"] is None
            counts.update(row["accounting"])
            cell = (row["phase"], row["condition"], row["repeat"])
            validity[cell][row["key"]] = row["result"] is not None
            cells[cell]["predictions"] += 1
            cells[cell]["valid"] += row["result"] is not None
            if row["result"] is None:
                null_ids.append(key)
            duplicate = any(
                (
                    len(v[field]) != len(set(v[field]))
                    for v in row["run_predictions"]
                    if v is not None
                    for field in r.LIST_FIELDS
                )
            )
            if duplicate:
                duplicate_ids.append(key)
        r.require(
            seen == set(jobs) and counts == done["counts"] == manifest["counts"],
            "Cache completion/accounting differs",
        )
        reported_nulls = c.read(run / "null_outputs.json")
        r.require(
            set(null_ids)
            == {(v["key"], v["condition"], v["repeat"]) for v in reported_nulls},
            "Null roster differs",
        )
        endpoints = c.read(run / "endpoint_setup.json")
        frozen = c.read(c.BASE / "endpoints.json")
        common_counts = {}
        for phase in c.COUNTS:
            keys = frozen[phase]["scorable_ids"]
            common = [
                key
                for key in keys
                if all(
                    (
                        validity[phase, cond, rep][key]
                        for cond in c.CONDITIONS
                        for rep in range(3)
                    )
                )
            ]
            r.require(
                endpoints[phase]["common_valid_keys"] == common
                and endpoints[phase]["common_excluded_keys"]
                == [key for key in keys if key not in set(common)],
                "Independent common mask differs",
            )
            for name in ("all_labels", "macro_labels", "gold_support", "scorable_ids"):
                r.require(
                    endpoints[phase][name] == frozen[phase][name],
                    "Frozen gold definition changed",
                )
            for cond in c.CONDITIONS:
                for rep in range(3):
                    reported = endpoints[phase]["cells"][f"{cond}/{rep}"]
                    r.require(
                        reported["valid"] == cells[phase, cond, rep]["valid"]
                        and reported["invalid_keys"]
                        == [key for key in keys if not validity[phase, cond, rep][key]],
                        "Own-valid mask differs",
                    )
            common_counts[phase] = len(common)
        samples = c.read(run / "raw_spot_checks.json")
        sampled = {
            (x["identity"]["key"], x["identity"]["condition"], x["identity"]["repeat"])
            for x in samples
        }
        r.require(
            set(null_ids) <= sampled and len(sampled) == len(samples),
            "Missing null-case raw replay or duplicate sample",
        )
        tests = sorted((r.BASE / "tests").glob("tests_*/test_report.json"))
        r.require(bool(tests), "Reader tests required")
        tested = c.read(tests[-1])
        r.require(
            tested["passed"] and tested["source_hashes"] == manifest["source_hashes"],
            "Current reader tests did not pass",
        )
        c.validate()
        c.verify(manifest["outputs"])
        report = {
            "status": "independent_cache_hash_vote_and_mask_check_pass",
            "predictions": len(seen),
            "canonical_result_hashes_reconstructed": len(seen),
            "production_votes_crosschecked": len(seen),
            "common_valid_counts": common_counts,
            "null_predictions": len(null_ids),
            "duplicate_label_predictions": len(duplicate_ids),
            "targeted_raw_items": len(samples),
            "reader_tests": tested["tests"],
            "test_report": c.ref(tests[-1]),
            "full_raw_tree_reaudited": False,
            "actual_Qwen_calls": 0,
            "performance_scores_computed": False,
        }
        report_path = run / "independent_check.json"
        r.save(report_path, report)
        r.save(
            r.BASE / "complete.json",
            {
                "status": "part1_complete_reader_ready",
                "created_utc": c.a.utc(),
                "manifest": c.ref(manifest_path),
                "bindings": c.bindings(
                    [
                        manifest_path,
                        report_path,
                        tests[-1],
                        r.BASE / "analysis_contract.json",
                        *r.own_sources(),
                    ]
                ),
                "plan_sha256": plan_sha,
                "predictions": len(seen),
                "performance_scores_computed": False,
                "new_Qwen_calls": 0,
            },
        )
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    verify(Path(args.manifest))
