from collections import Counter
from analysis.feedback.insertion import common as c
from analysis.feedback.insertion import scores_earlier as old
from analysis.feedback.insertion import replay_review as review


def sensitivity_scores(roster, results, science, excluded, draws=10000):
    c.require(
        len(excluded) == len(set(excluded))
        and set(excluded) <= {r["key"] for r in roster},
        "Invalid exclusion identities",
    )
    primary = {}
    for endpoint in science["primary"]:
        for name, strict, representatives in [
            ("primary", False, False),
            ("all_three_votes", True, False),
            ("one_per_cluster", False, True),
        ]:
            primary[endpoint["name"] + "/" + name] = old.primary(
                roster,
                results,
                endpoint["gold_hate"],
                endpoint["bootstrap_seed"],
                strict=strict,
                representatives=representatives,
                excluded=excluded,
                draws=draws,
            )
    remaining = [r for r in roster if r["key"] not in excluded]
    runs = []
    for condition in science["conditions"]:
        for repeat in range(3):
            counts = Counter(inputs=len(remaining), nulls=0, tp=0, fp=0, fn=0, tn=0)
            for row in remaining:
                result = results[row["key"], condition, repeat]["result"]
                if result is None:
                    counts["nulls"] += 1
                    continue
                truth = row["gold"]["hate"]
                pred = result["hate"]
                counts[("t" if pred == truth else "f") + ("p" if pred else "n")] += 1
            tp, fp, fn, tn = [counts[k] for k in ["tp", "fp", "fn", "tn"]]

            def f1(a, b, d):
                return 2 * a / (2 * a + b + d) if 2 * a + b + d else 0.0

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
    return {
        "excluded_keys": sorted(excluded),
        "remaining_by_population": dict(Counter(r["population"] for r in remaining)),
        "primary": primary,
        "all_runs": runs,
        "scope": "Common-exclusion co-primary and gate all-run sensitivity. Supporting hate-type, source and delivery contrasts still require the post-run review on both rosters.",
    }


def main():
    review.verify()

    old.main()
    from analysis.feedback.insertion import run as api

    b = api.queue()
    b.validate(ready=True)
    results = {}
    for stage in c.read(api.BASE / "schedule.json")["stages"]:
        for job in stage["jobs"]:
            ident = job["key"], job["condition"], job["repeat"]
            c.require(ident not in results, "Repeated result identity")
            results[ident] = c.read(
                b.item_dir(api.BASE / "live" / api.STUDY, job, c.sha(b.PLAN))
                / "result.json"
            )
    roster = c.rows(c.BASE / "roster.jsonl")
    science = c.read(c.BASE / "scientific_plan_v1.json")
    c.require(
        set(results)
        == {
            (r["key"], condition, rep)
            for r in roster
            for condition in science["conditions"]
            for rep in range(3)
        },
        "Incomplete matrix",
    )
    plan = c.read(review.SENSITIVITY)
    scores = sensitivity_scores(
        roster, results, science, plan["excluded_keys"], science["bootstrap_draws"]
    )
    c.require(
        scores["remaining_by_population"] == plan["remaining_by_population"],
        "Sensitivity roster differs",
    )
    c.save(
        c.BASE / "analysis/primary_and_runs_replay_exclusion.json",
        {
            "status": "scored_requires_scientific_review",
            "plan_sha256": c.sha(b.PLAN),
            "sensitivity_plan_sha256": c.sha(review.SENSITIVITY),
            "original_primary_scores_sha256": c.sha(
                c.BASE / "analysis/primary_and_runs.json"
            ),
            **scores,
            "new_model_calls": 0,
        },
    )
    print(
        "Full-roster and mandatory common-exclusion gate analyses saved; supporting scientific review remains.",
        flush=True,
    )


if __name__ == "__main__":
    with c.offline():
        main()
