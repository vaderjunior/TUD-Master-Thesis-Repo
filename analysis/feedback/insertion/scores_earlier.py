from collections import defaultdict
import hashlib
import numpy as np
from analysis.feedback.insertion import common as c


def primary(
    roster,
    results,
    truth,
    seed,
    strict=False,
    representatives=False,
    excluded=(),
    draws=10000,
):
    conditions = [f"error_balanced_s{s}" for s in c.SEEDS]
    keys = [
        r["key"]
        for r in roster
        if r["gold"]["hate"] == truth and r["key"] not in excluded
    ]
    by_key = {r["key"]: r for r in roster}
    if representatives:
        groups = defaultdict(list)
        for key in keys:
            groups[by_key[key]["cluster"]].append(key)
        chosen = {
            min(
                group,
                key=lambda key: hashlib.sha256(
                    ("A14-representative|" + key).encode()
                ).hexdigest(),
            )
            for group in groups.values()
        }
        keys = [key for key in keys if key in chosen]
    common = []
    errors = {}
    bounds = {}
    for key in keys:
        arms = [
            [
                results[key, condition, repeat]
                for condition in members
                for repeat in range(3)
            ]
            for members in [conditions, ["reference"]]
        ]
        valid = [
            [r["result"] is not None and (not strict or r["n_valid"] == 3) for r in arm]
            for arm in arms
        ]
        if all(all(v) for v in valid):
            common.append(key)
            errors[key] = [
                float(np.mean([r["result"]["hate"] != truth for r in arm]))
                for arm in arms
            ]
        bounds[key] = [
            [
                float(
                    np.mean(
                        [
                            (r["result"]["hate"] != truth) if ok else missing
                            for r, ok in zip(arm, mask)
                        ]
                    )
                )
                for missing in [0, 1]
            ]
            for arm, mask in zip(arms, valid)
        ]
    if not common:
        return {
            "status": "not_estimable_no_common_valid_inputs",
            "original_n": len(keys),
            "common_n": 0,
            "excluded_ids": keys,
        }
    clusters = sorted({by_key[key]["cluster"] for key in common})
    ix = {label: i for i, label in enumerate(clusters)}
    totals = np.zeros((len(clusters), 2))
    sizes = np.zeros(len(clusters))
    for key in common:
        totals[ix[by_key[key]["cluster"]]] += errors[key]
        sizes[ix[by_key[key]["cluster"]]] += 1
    rng = np.random.Generator(np.random.PCG64(seed))
    values = []
    for start in range(0, draws, 250):
        chosen = rng.integers(
            0, len(clusters), size=(min(250, draws - start), len(clusters))
        )
        means = totals[chosen].sum(axis=1) / sizes[chosen].sum(axis=1)[:, None]
        values.extend(means[:, 0] - means[:, 1])
    point = np.mean([errors[key] for key in common], axis=0)
    interval = np.quantile(values, [0.0125, 0.9875]).tolist()
    b = np.mean([bounds[key] for key in keys], axis=0)
    return {
        "status": "scored",
        "gold_hate": truth,
        "original_n": len(keys),
        "common_n": len(common),
        "common_ids": common,
        "excluded_ids": sorted(set(keys) - set(common)),
        "clusters": len(clusters),
        "balanced_error_rate": float(point[0]),
        "reference_error_rate": float(point[1]),
        "difference": float(point[0] - point[1]),
        "ci975": interval,
        "full_roster_difference_missing_bounds": [
            float(b[0, 0] - b[1, 1]),
            float(b[0, 1] - b[1, 0]),
        ],
        "bootstrap": {
            "seed": seed,
            "draws": draws,
            "unit": "original whole text cluster",
            "generator": "PCG64",
        },
        "strict_all_three_votes": strict,
        "one_representative": representatives,
        "shared_reference_repeats": 3,
    }


def main():
    from analysis.feedback.insertion import run as api

    b = api.queue()
    b.validate(ready=True)
    done = c.read(api.BASE / "live/complete.json")
    c.require(
        done["status"] == "all_studies_executed_and_audited"
        and done["mode"] == "live"
        and done["plan_sha256"] == c.sha(b.PLAN),
        "New raw audit required",
    )
    c.verify(done["review_hashes"])
    schedule = c.read(api.BASE / "schedule.json")
    result = {}
    runs = []
    for stage in schedule["stages"]:
        root = api.BASE / "live" / api.STUDY
        review = b.verify_review(stage, root, c.sha(b.PLAN), "live")
        target = root / "stages" / stage["id"]
        for name, h in c.read(target / review["raw_hashes"]).items():
            c.require(c.sha(root / name) == h, "Raw evidence changed after audit")
        for job in stage["jobs"]:
            value = c.read(b.item_dir(root, job, c.sha(b.PLAN)) / "result.json")
            ident = job["key"], job["condition"], job["repeat"]
            c.require(ident not in result, "Repeated prediction")
            result[ident] = value
    roster = c.rows(c.BASE / "roster.jsonl")
    science = c.read(c.BASE / "scientific_plan_v1.json")
    c.require(
        set(result)
        == {
            (r["key"], condition, rep)
            for r in roster
            for condition in science["conditions"]
            for rep in range(3)
        },
        "Incomplete scored matrix",
    )
    scores = {}
    for endpoint in science["primary"]:
        for name, strict, reps in [
            ("primary", False, False),
            ("all_three_votes", True, False),
            ("one_per_cluster", False, True),
        ]:
            scores[endpoint["name"] + "/" + name] = primary(
                roster,
                result,
                endpoint["gold_hate"],
                endpoint["bootstrap_seed"],
                strict,
                reps,
            )
    for condition in science["conditions"]:
        for repeat in range(3):
            counts = defaultdict(int)
            for row in roster:
                v = result[row["key"], condition, repeat]["result"]
                truth = row["gold"]["hate"]
                counts["inputs"] += 1
                if v is None:
                    counts["nulls"] += 1
                    continue
                counts[
                    ("t" if v["hate"] == truth else "f") + ("p" if v["hate"] else "n")
                ] += 1
            tp, fp, fn, tn = (counts[k] for k in ["tp", "fp", "fn", "tn"])
            f1 = lambda a, b, d: 2 * a / (2 * a + b + d) if 2 * a + b + d else 0.0
            runs.append(
                {
                    "condition": condition,
                    "repeat": repeat,
                    **dict(counts),
                    "macro_f1_own_valid": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
                    "false_positive_rate_own_valid": (
                        fp / (fp + tn) if fp + tn else None
                    ),
                    "false_negative_rate_own_valid": (
                        fn / (fn + tp) if fn + tp else None
                    ),
                }
            )
    c.save(
        c.BASE / "analysis/primary_and_runs.json",
        {
            "status": "scored_requires_scientific_review",
            "plan_sha256": c.sha(b.PLAN),
            "scientific_plan_sha256": c.sha(c.BASE / "scientific_plan_v1.json"),
            "primary": scores,
            "all_runs": runs,
            "scope": "Primary family and basic all-run gate metrics only. Additional prespecified supporting/type/delivery/deletion analyses remain part of the post-run review.",
            "new_model_calls": 0,
        },
    )
    print("Primary endpoints and all 57 gate runs saved; supporting review remains.")


if __name__ == "__main__":
    with c.offline():
        main()
