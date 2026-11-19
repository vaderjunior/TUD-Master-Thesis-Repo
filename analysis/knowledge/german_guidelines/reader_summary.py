from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import yaml
from analysis.knowledge.german_guidelines import common as c
from analysis.knowledge.german_guidelines import reader as r
from analysis.common import schema as isolated


def main():
    with r.offline():
        c.validate()
        c.verify(c.read(r.BASE / "analysis_contract.json")["source_bindings"])
        manifest, base = r.read_verified()
        coverage = c.read(base / "coverage.json")
        totals, population, condition = (
            Counter(),
            defaultdict(Counter),
            defaultdict(Counter),
        )
        for cell, stats in coverage.items():
            phase, cond, repeat = cell.split("/")
            totals.update(stats)
            population[phase].update(stats)
            condition[cond].update(stats)
        samples = c.read(base / "raw_spot_checks.json")
        sampled = {
            (x["identity"]["key"], x["identity"]["condition"], x["identity"]["repeat"])
            for x in samples
        }
        cached_counts, retry_keys, null_keys = (Counter(), set(), set())
        free_normalisations = 0
        for row in c.rows(base / "predictions.jsonl"):
            key = (row["key"], row["condition"], row["repeat"])
            cached_counts["predictions"] += 1
            cached_counts["valid"] += row["result"] is not None
            cached_counts["valid_votes_" + str(row["n_valid"])] += 1
            cached_counts["invalid_votes"] += row["parse_failures"]
            cached_counts["repair_calls"] += row["repairs"]
            cached_counts["gate_normalized_votes"] += row["gate_normalised"]
            free_normalisations += row["normalisations"]
            if row["accounting"]["retry_attempts"]:
                retry_keys.add(key)
            if row["result"] is None:
                null_keys.add(key)
        r.require(
            all((totals[k] == n for k, n in cached_counts.items())),
            "Coverage totals differ from cache",
        )
        r.require(
            retry_keys <= sampled and null_keys <= sampled,
            "Missing raw replay for a null or retry item",
        )
        r.require(
            totals["valid_votes_1"]
            + 2 * totals["valid_votes_2"]
            + 3 * totals["valid_votes_3"]
            == 3 * totals["predictions"] - totals["invalid_votes"],
            "Inner-vote totals differ",
        )
        endpoints = c.read(base / "endpoint_setup.json")
        cohort_rows = []
        for phase, value in endpoints.items():
            clusters, labels = (Counter(value["clusters"]), defaultdict(set))
            for cluster, gold in zip(value["clusters"], value["gold"]):
                labels[cluster].add(json.dumps(gold, sort_keys=True))
            cohort_rows.append(
                {
                    "population": phase,
                    "inputs": len(value["keys"]),
                    "common_valid": len(value["common_valid_keys"]),
                    "support": value["gold_support"],
                    "empty_legal_gold": value["observed_empty_gold_sets"],
                    "source_counts": value["source_counts"],
                    "clusters": len(clusters),
                    "repeated_clusters": sum((n > 1 for n in clusters.values())),
                    "cross_source_clusters": sum(
                        (
                            len(v) > 1
                            for v in value["source_memberships_by_cluster"].values()
                        )
                    ),
                    "discordant_gold_clusters": sum(
                        (len(v) > 1 for v in labels.values())
                    ),
                    "cell_valid": {k: v["valid"] for k, v in value["cells"].items()},
                    "excluded_keys": value["common_excluded_keys"],
                }
            )
        backend = isolated.adapter(c.ROOT / "config/taxonomy.yaml")
        taxonomy = yaml.safe_load(
            (c.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
        )
        nulls = c.read(base / "null_outputs.json")
        causes, final_causes, unknown, null_details = (
            Counter(),
            Counter(),
            Counter(),
            [],
        )
        for item in nulls:
            raw = r.checked_json(
                Path(item["evidence"]["result_path"]), item["evidence"]["result_sha256"]
            )
            r.require(
                raw["result"] is None and raw["n_valid"] == 0, "Null item changed"
            )
            final_votes = []
            for vote, chain in enumerate(raw["raw_runs"]):
                r.require(
                    len(chain) == 3, "All-null vote did not exhaust its two repairs"
                )
                for turn, text in enumerate(chain):
                    value, _, error = backend.bundle.parser.parse_response(text)
                    r.require(
                        value is None and error is not None,
                        "Null vote contains a valid answer",
                    )
                    causes[error] += 1
                    if turn == 2:
                        final_causes[error] += 1
                        obj = json.loads(backend.bundle.parser.extract_json(text))
                        normalized, _ = backend.bundle.schema.normalise_raw(obj)
                        unrecognised = {
                            field: [
                                x
                                for x in normalized.get(field, [])
                                if x not in taxonomy["dimensions"][field]["labels"]
                            ]
                            for field in r.LIST_FIELDS
                        }
                        unrecognised = {
                            field: values
                            for field, values in unrecognised.items()
                            if values
                        }
                        for field, values in unrecognised.items():
                            unknown.update((field + "/" + str(x) for x in values))
                        final_votes.append(
                            {
                                "vote": vote,
                                "final_error": error,
                                "unrecognised_labels": unrecognised,
                            }
                        )
            null_details.append({**item, "final_votes": final_votes})
        timeline = c.read(base / "execution_timeline.json")
        attempts = [attempt for stage in timeline for attempt in stage["attempts"]]
        start = min((a["start"]["started_utc"] for a in attempts))
        finish = max((a["finished_utc"] for a in attempts))
        execution = {
            "start": start,
            "last_execution_finish": finish,
            "elapsed_hours_including_between_batch_checks": (
                datetime.fromisoformat(finish) - datetime.fromisoformat(start)
            ).total_seconds()
            / 3600,
            "stopped_attempts": sum((a["summary"]["stopped"] for a in attempts)),
            "errors": sum((len(a["summary"]["errors"]) for a in attempts)),
            "unsubmitted": sum((a["summary"]["unsubmitted"] for a in attempts)),
            "submitted": sum((a["summary"]["submitted"] for a in attempts)),
            "peak_submitted_values": sorted(
                {a["summary"]["peak_submitted"] for a in attempts}
            ),
            "note": "Chronology only. This is not pure API time or a resource-efficiency comparison.",
        }
        r.require(
            execution["stopped_attempts"]
            == execution["errors"]
            == execution["unsubmitted"]
            == 0
            and execution["submitted"] == 33093,
            "Execution has an unresolved stop or submission gap",
        )
        prompts = c.read(base / "prompt_setup.json")
        common = set(endpoints["de_test"]["common_valid_keys"])
        sensitivity = {
            name: {
                "declared_keys": keys,
                "present_in_main_common_mask": [k for k in keys if k in common],
            }
            for name, keys in prompts["sensitivity_rosters"].items()
        }
        out = r.BASE / "report"
        summary = {
            "status": "coverage_and_failure_summary_verified",
            "totals": dict(totals),
            "free_normalisation_operations": free_normalisations,
            "populations": cohort_rows,
            "by_population": dict(population),
            "by_condition": dict(condition),
            "null_final_parse_errors": dict(final_causes),
            "null_all_parse_errors": dict(causes),
            "unrecognised_final_null_labels": dict(unknown),
            "execution": execution,
            "all_retry_items_replayed": len(retry_keys),
            "all_null_items_replayed": len(null_keys),
            "sensitivity_setup": sensitivity,
            "performance_scores_computed": False,
            "actual_Qwen_calls": 0,
            "D_writes": 0,
        }
        r.save(out / "summary.json", summary)
        r.save(out / "null_failure_details.json", null_details)
        r.save(
            out / "manifest.json",
            {
                "status": "part1_summary_verified",
                "bindings": c.bindings(
                    [
                        Path(__file__),
                        r.BASE / "complete.json",
                        base / "manifest.json",
                        out / "summary.json",
                        out / "null_failure_details.json",
                    ]
                ),
                "performance_scores_computed": False,
                "actual_Qwen_calls": 0,
            },
        )
        print(
            json.dumps(
                {
                    "summary": c.ref(out / "summary.json"),
                    "unknown_final_labels": dict(unknown),
                    "free_normalisation_operations": free_normalisations,
                    "all_retry_items_replayed": len(retry_keys),
                    "sensitivity": sensitivity,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
