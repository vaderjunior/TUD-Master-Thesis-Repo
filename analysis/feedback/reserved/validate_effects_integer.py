from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json
import math
import numpy as np
from analysis.feedback.reserved import effects as a


def main():
    a.io.verify(a.io.ROOT, a.io.read(a.OUT / "complete.json")["artifacts"])
    roster = list(a.io.rows(a.io.STUDY / "roster.jsonl"))
    bykey = {r["key"]: r for r in roster}
    science = a.io.read(a.io.STUDY / "scientific_plan_v1.json")
    specs = {
        s["endpoint"]: s for s in a.rates.contrasts(science) if s["family"] == "primary"
    }
    predictions = {}
    for path in (a.reader.OUT / "assembly").glob("*.jsonl"):
        for row in a.io.rows(path):
            if row["condition"] in specs["FPR"]["treatment_conditions"] + [
                "reference5"
            ]:
                predictions[row["key"], row["condition"], row["repeat"]] = row
    checked = {}
    results = a.io.read(a.OUT / "primary_and_sensitivities.json")
    for name, result in results.items():
        spec = specs[result["endpoint"]]
        groups = defaultdict(list)
        for key in result["common_ids"]:
            groups[bykey[key]["cluster"]].append(key)
        numerator = []
        size = []
        for label in sorted(groups):
            total = 0
            for k in groups[label]:
                gold = bykey[k]["gold"]["hate"]
                treatment = sum(
                    predictions[k, c, r]["result"]["hate"] != gold
                    for c in spec["treatment_conditions"]
                    for r in range(3)
                )
                reference = sum(
                    predictions[k, "reference5", r]["result"]["hate"] != gold
                    for r in range(3)
                )
                total += treatment - 3 * reference
            numerator.append(total)
            size.append(len(groups[label]))
        numerator = np.asarray(numerator, dtype=np.int64)
        size = np.asarray(size, dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(spec["seed"]))
        nums = []
        dens = []
        for start in range(0, 10000, 250):
            chosen = rng.integers(
                0, len(groups), size=(min(250, 10000 - start), len(groups))
            )
            nums.extend(numerator[chosen].sum(axis=1).tolist())
            dens.extend((9 * size[chosen].sum(axis=1)).tolist())
        exact_values = [Fraction(n, d) for n, d in zip(nums, dens)]
        direct = np.asarray([float(x) for x in exact_values])
        saved = np.load(
            a.OUT / "draws" / (name.replace("/", "__") + ".npy"), allow_pickle=False
        )
        largest = float(np.max(np.abs(direct - saved)))
        a.need(largest < 1e-12, "Integer draw reconstruction differs")
        ordered = sorted(exact_values)
        endpoints = []
        tails = []

        for q in (Fraction(1, 80), Fraction(79, 80)):
            pos = (len(ordered) - 1) * q
            lo = pos.numerator // pos.denominator
            hi = math.ceil(pos)
            alpha = pos - lo
            value = (1 - alpha) * ordered[lo] + alpha * ordered[hi]
            endpoints.append(float(value))
            tails.append(
                {
                    "lower_rank_zero_based": lo,
                    "upper_rank_zero_based": hi,
                    "rank_fraction": str(alpha),
                    "lower_value_exact": str(ordered[lo]),
                    "upper_value_exact": str(ordered[hi]),
                    "quantile_exact": str(value),
                }
            )
        a.need(
            all(
                abs(x - y) < 1e-12 for x, y in zip(endpoints, result["family_interval"])
            ),
            "Exact percentile differs",
        )
        checked[name] = {
            "draws_checked": 10000,
            "max_absolute_error": largest,
            "interval_from_exact_integer_errors": endpoints,
            "tails": tails,
            "exact_zero_draws": nums.count(0),
        }
    special = checked["same_six_excluded/all_three_votes/primary/reference5/FNR"]
    a.need(
        special["tails"][1]["quantile_exact"] == "0",
        "Reinspect strict/exclusion zero qualification",
    )
    a.write(
        a.OUT / "integer_draw_verification.json",
        {
            "status": "all_primary_integer_draws_and_exact_percentile_tails_passed",
            "created_utc": a.io.utc(),
            "source_script_sha256": a.io.sha(Path(__file__)),
            "draws_checked": sum(v["draws_checked"] for v in checked.values()),
            "results": checked,
            "qualification": "Strict all-three-valid-vote plus same-six exclusion FNR upper 97.5% endpoint is exactly zero. Preserve raw floating output but do not call its rounding residue evidence of an interval below zero.",
            "new_model_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "draws_checked": 120000,
                "strict_exclusion_fnr_exact_interval": special[
                    "interval_from_exact_integer_errors"
                ],
                "strict_exclusion_fnr_upper_tail": special["tails"][1],
                "new_model_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with a.io.offline():
        main()
