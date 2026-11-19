from src.hsrag.paths import PROJECT_ROOT
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_full_kb_relevance_20260930_v1"
DEST = BASE / "results_v1"
OLD = ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1"


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def rows(p):
    return [json.loads(x) for x in p.read_text(encoding="utf8").splitlines() if x]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    with p.open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            value, f, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        )
        f.write("\n")


def main():
    assert not (DEST / "independent_verification.json").exists()
    completed = read(DEST / "complete.json")
    for name, expected in {**completed["inputs"], **completed["outputs"]}.items():
        assert sha(ROOT / name) == expected, name
    science = read(BASE / "scientific_plan_v1.json")
    labels = science["labels"]
    roster = {r["id"]: r for r in rows(OLD / "feasibility_v1/roster.jsonl")}
    gold = {i: set(r["gold"]["hate_type"]) for i, r in roster.items()}
    exported = rows(BASE / "review_v1/predictions.jsonl")
    runs = defaultdict(dict)
    duplicate_labels = Counter()
    deduplicated_vote_differences = []
    for r in exported:
        assert (
            r["gold"] == roster[r["id"]]["gold"]
            and r["cluster"] == roster[r["id"]]["cluster"]
        )
        assert r["id"] not in runs[r["condition"], r["repeat"]]
        runs[r["condition"], r["repeat"]][r["id"]] = r
        valid = [v for v in r["run_predictions"] if v is not None]
        if valid:
            gate = sum(v["hate"] for v in valid) * 2 > len(valid)
            for field in ["hate_type", "target_group", "legal"]:
                for v in valid:
                    duplicate_labels[field] += len(v[field]) != len(set(v[field]))
                supported = Counter(l for v in valid for l in set(v[field]))
                kept = sorted(l for l, n in supported.items() if n * 2 > len(valid))
                if not gate and field != "legal":
                    kept = []
                if kept != r["result"][field]:
                    deduplicated_vote_differences.append(
                        {
                            "condition": r["condition"],
                            "repeat": r["repeat"],
                            "id": r["id"],
                            "field": field,
                        }
                    )

    def counts(ids, run, missing=None):
        out = {l: Counter() for l in labels}
        n = 0
        contains = exact = misses = length = 0
        for i in ids:
            p = run[i]["result"]
            y = gold[i]
            if p is None and missing is None:
                continue
            x = (
                set(p["hate_type"])
                if p
                else (set(labels) - y if missing == "worst" else y)
            )
            n += 1
            contains += y <= x
            exact += y == x
            length += len(x)
            if p:
                misses += p["hate"] is False
            for l in labels:
                out[l]["tp"] += l in x and l in y
                out[l]["fp"] += l in x and l not in y
                out[l]["fn"] += l not in x and l in y
                out[l]["support"] += l in y
        for l, c in out.items():
            den = 2 * c["tp"] + c["fp"] + c["fn"]
            c["f1"] = 2 * c["tp"] / den if den else 0.0
            c["precision"] = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
            c["recall"] = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        return {
            "valid": n,
            "macro_f1": statistics.mean(c["f1"] for c in out.values()),
            "contains": contains / n,
            "exact": exact / n,
            "predicted_labels": length / n,
            "missed_hate_rate": misses / n,
        }, {l: dict(c) for l, c in out.items()}

    def equal(a, b):
        assert abs(a - b) < 1e-12, (a, b)

    checked = 0
    detailed = {}
    scopes = {
        "primary": "primary_full_cover",
        "primary_controls": "primary_full_cover",
        "all_transfer": "evaluation",
        "selection_holdout": "holdout_from_original_selection",
        "historical_sentinels": "historical_sentinels",
    }
    for ep, roster_key in scopes.items():
        value = read(DEST / f"{ep}.json")
        ids = science["rosters"][roster_key]
        policies = list(value["groups"])
        groups = {
            p: (
                ["reference"]
                if p == "reference"
                else [f"{p}_s{s}" for s in science["pool_seeds"]]
            )
            for p in policies
        }
        keys = [
            (c, r) for members in groups.values() for c in members for r in range(3)
        ]
        common = [i for i in ids if all(runs[k][i]["result"] is not None for k in keys)]
        assert (
            common == value["common_valid_ids"]
            and len(common) == value["common_valid_n"]
        )
        assert sorted(set(ids) - set(common)) == value["excluded_ids"]
        per_run = {}
        for c, rep in keys:
            name = f"{c}_rep{rep}"
            run = runs[c, rep]
            for which, subset in [("own_valid", ids), ("common_valid", common)]:
                m, pl = counts(subset, run)
                expected = value["settings"][name][which]
                for k, v in m.items():
                    equal(v, expected[k])
                for l in labels:
                    equal(pl[l]["f1"], expected["per_label_f1"][l])
                for j, missing in enumerate(["worst", "best"]):
                    equal(
                        counts(subset, run, missing)[0]["macro_f1"],
                        expected["full_roster_missing_prediction_bounds"][j],
                    )
                assert expected["null_ids"] == [
                    i for i in subset if run[i]["result"] is None
                ]
                checked += 1
                if which == "common_valid":
                    per_run[name] = pl
        for p, members in groups.items():
            points = [
                value["settings"][f"{c}_rep{r}"]["common_valid"]
                for c in members
                for r in range(3)
            ]
            for metric in [
                "macro_f1",
                "contains",
                "exact",
                "predicted_labels",
                "missed_hate_rate",
            ]:
                equal(
                    statistics.mean(v[metric] for v in points),
                    value["groups"][p][metric],
                )
            if p == "reference":
                assert len(members) == 1 and len(points) == 3
            assert len(value["groups"][p]["pool_means"]) == len(members)
        for name, v in value["contrasts"].items():
            if name == "text_by_label_overlap_interaction":
                expected = sum(
                    sign * value["groups"][p]["macro_f1"]
                    for sign, p in [
                        (1, "relevant_full"),
                        (-1, "relevant_labels"),
                        (-1, "disjoint_full"),
                        (1, "disjoint_labels"),
                    ]
                )
            else:
                l, r = name.split("_minus_")
                expected = (
                    value["groups"][l]["macro_f1"] - value["groups"][r]["macro_f1"]
                )
            equal(expected, v["difference"])
            assert v["ci95"][0] <= v["ci95"][1]
        detailed[ep] = {
            "support": {l: sum(l in gold[i] for i in common) for l in labels},
            "per_run_label_counts": per_run,
        }
    partial = read(DEST / "partial_cover_seven.json")
    for (c, rep), run in runs.items():
        expected = partial[f"{c}_rep{rep}"]
        m, pl = counts(science["rosters"]["secondary_only_partial_cover"], run)
        for k, v in m.items():
            equal(v, expected[k])
        for l in labels:
            equal(pl[l]["f1"], expected["per_label_f1"][l])
        checked += 1
    pools = {
        p["seed"]: p
        for p in read(OLD / "update_payloads_v1/pools.json")["pools"]
        if p["size"] == 40 and p["arm"] == "error"
    }
    natural = read(OLD / "results_v1/retrieval_diagnostics.json")["per_item"]
    coverage = Counter()
    source_occurrences = Counter()
    primary = set(science["rosters"]["primary_full_cover"])
    for c in rows(BASE / "choices.jsonl"):
        i = c["query_id"]
        seed = c["pool_seed"]
        pool = pools[seed]["ids"]
        q = gold[i]
        assert (
            i not in pool
            and c["natural_source_id"]
            == natural[c["old_pool_condition"]][i]["source_id"]
        )
        candidates = [j for j in pool if q <= gold[j]]
        assert c["full_cover_available"] == bool(candidates)
        if not candidates:
            best = max(len(q & gold[j]) for j in pool)
            candidates = [j for j in pool if len(q & gold[j]) == best]
        disjoint = [j for j in pool if not q & gold[j]]
        for purpose, eligible in [("relevant", candidates), ("disjoint", disjoint)]:
            chosen = min(
                eligible,
                key=lambda j: hashlib.sha256(
                    f"{science['selection']['seed']}|{seed}|{i}|{purpose}|{j}".encode()
                ).hexdigest(),
            )
            assert chosen == c[purpose + "_source_id"]
        a = gold[c["natural_source_id"]]
        h = gold[c["relevant_source_id"]]
        coverage["query_pool_pairs"] += 1
        coverage["natural_covers_gold"] += q <= a
        coverage["natural_disjoint"] += not (q & a)
        coverage["natural_partial_overlap"] += bool(q & a) and not q <= a
        coverage["same_natural_relevant_source"] += (
            c["natural_source_id"] == c["relevant_source_id"]
        )
        coverage["relevant_covers_gold"] += q <= h
        coverage["relevant_overlaps_gold"] += bool(q & h)
        coverage["disjoint_overlaps_gold"] += bool(q & gold[c["disjoint_source_id"]])
        if i in primary:
            assert q <= h
        for p in ["natural", "relevant", "disjoint"]:
            source_occurrences[p + ":" + c[p + "_source_id"]] += 1
    operational = read(BASE / "review_v1/summary.json")
    costs = defaultdict(Counter)
    for r in exported:
        costs[r["policy"]].update(r["costs"])
        costs[r["policy"]]["predictions"] += 1
        costs[r["policy"]]["nulls"] += r["result"] is None
        costs[r["policy"]]["invalid_votes"] += 3 - r["n_valid"]
    for p, c in costs.items():
        for k, v in c.items():
            assert abs(v - operational["by_policy_costs"][p][k]) < (
                1e-7 if k == "vote_seconds" else 1e-12
            )
    save(DEST / "label_count_diagnostics.json", detailed)
    save(
        DEST / "source_selection_checks.json",
        {
            "coverage": dict(coverage),
            "source_occurrences": dict(source_occurrences),
            "choices_independently_recomputed": 1050,
            "gold_used_only_for_declared_diagnostic": True,
        },
    )
    assert (
        not deduplicated_vote_differences
    ), "Duplicate labels change vote; preserve and investigate before publication"
    save(
        DEST / "independent_verification.json",
        {
            "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scored_completion_sha256": sha(DEST / "complete.json"),
            "checked_setting_metrics": checked,
            "source_choices": 1050,
            "exported_rows": len(exported),
            "within_vote_duplicate_label_fields": dict(duplicate_labels),
            "deduplicated_field_vote_differences": len(deduplicated_vote_differences),
            "method": "Independent Python set/count F1, missing-value bounds, masks, run/pool averaging, contrast arithmetic, label counts, frozen source hash choices and cost aggregation. Bootstrap already checked by expanded draws in scorer.",
            "source_sha256": sha(Path(__file__)),
            "additional_outputs": {
                str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                for p in [
                    DEST / "label_count_diagnostics.json",
                    DEST / "source_selection_checks.json",
                ]
            },
            "new_model_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "checked_setting_metrics": checked,
                "source_choices": dict(coverage),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
