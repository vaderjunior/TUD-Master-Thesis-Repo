import os

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import numpy as np
from analysis.feedback.transfer import archive as a

OUT = a.ROOT / "experiments/astra/sq3_transfer_part2_20261023_v1"
VIEWS = {
    "main": (False, False, False),
    "strict": (True, False, False),
    "replay_excluded": (False, False, True),
    "strict_replay_excluded": (True, False, True),
    "representative": (False, True, False),
    "representative_replay_excluded": (False, True, True),
}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def save(path, value):
    check(path.resolve().is_relative_to(OUT), "Output left new analysis")
    with path.open("xb") as stream:
        stream.write(a.encoded(value, pretty=True))


def near(actual, expected, message):
    if expected is None:
        check(actual is None, message)
    else:
        check(
            actual is not None and abs(float(actual) - float(expected)) < 1e-13, message
        )


def support_seed(text):
    return int.from_bytes(
        hashlib.sha256(("A16-part2-support-v1|" + text).encode()).digest()[:8], "little"
    )


def percentile_linear(values, q):
    ordered = sorted(float(v) for v in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main():
    check(
        not (OUT / "independent_check.json").exists(), "Keep prior independent evidence"
    )
    calculated = a.read(OUT / "calculated.json")
    check(
        calculated["status"] == "part2_calculated_independent_verification_pending",
        "Wrong calculation checkpoint",
    )
    for name, expected in {**calculated["bindings"], **calculated["artifacts"]}.items():
        check(
            a.sha(a.inside(a.ROOT, name)) == expected,
            "Changed calculation input/output: " + name,
        )
    science = a.read(a.STUDY / "scientific_plan_v1.json")
    contract = a.read(OUT / "analysis_contract.json")
    check(
        contract["outcomes_used_to_choose_conventions"] is False
        and contract["representatives"]["prefix"] == "A16-representative|",
        "Wrong pre-scoring convention contract",
    )
    meta = []
    errors = np.full((15758, 4, 3), -1, dtype=np.int64)
    valid_votes = np.empty((15758, 4, 3), dtype=np.int8)
    gates = np.full((15758, 4, 3), -1, dtype=np.int8)
    for row in a.rows(a.OUT / "matrix.jsonl"):
        i = len(meta)
        meta.append(
            {
                key: row[key]
                for key in (
                    "key",
                    "population",
                    "source",
                    "gold_hate",
                    "inference_cluster",
                    "stage",
                    "execution_period",
                )
            }
        )
        for c, repetitions in enumerate(row["predictions"]):
            for repeat, p in enumerate(repetitions):
                valid_votes[i, c, repeat] = p["n_valid"]
                if p["result"] is not None:
                    gates[i, c, repeat] = int(p["result"]["hate"])
                    errors[i, c, repeat] = int(p["result"]["hate"] != row["gold_hate"])
    check(len(meta) == 15758, "Wrong source matrix length")
    global_common = (errors >= 0).all(axis=(1, 2))
    check(
        int(global_common.sum()) == 15608
        and int((valid_votes == 3).all(axis=(1, 2)).sum()) == 14299,
        "Known availability counts differ",
    )

    def select(pop, truth, view, source=None):
        strict, reps, exclude = VIEWS[view]
        omitted = set(science["mandatory_replay_keys"]) if exclude else set()
        eligible = [
            i
            for i, r in enumerate(meta)
            if r["population"] == pop
            and r["gold_hate"] == truth
            and r["key"] not in omitted
            and (source is None or source == r["source"])
        ]
        if reps:
            clusters = defaultdict(list)
            for i in eligible:
                clusters[meta[i]["inference_cluster"]].append(i)
            representative = {
                min(
                    values,
                    key=lambda i: hashlib.sha256(
                        ("A16-representative|" + meta[i]["key"]).encode()
                    ).hexdigest(),
                )
                for values in clusters.values()
            }
            eligible = [i for i in eligible if i in representative]
        return eligible, strict

    def point(record, eligible, strict, arms=(1, 2, 3)):
        eligible = list(eligible)
        used = [0, *arms]
        valid = errors >= 0
        if strict:
            valid = valid & (valid_votes == 3)
        common = [i for i in eligible if valid[i, used, :].all()]
        check(
            record["eligible_n"] == len(eligible)
            and record["common_n"] == len(common)
            and record["excluded_by_availability"] == len(eligible) - len(common),
            "Wrong point denominator",
        )
        control_errors = sum(int(errors[i, 0, :].sum()) for i in common)
        treatment_errors = sum(int(errors[i, list(arms), :].sum()) for i in common)
        control = Fraction(control_errors, len(common) * 3) if common else None
        treatment = (
            Fraction(treatment_errors, len(common) * len(arms) * 3) if common else None
        )
        for name, value in [
            ("reference_rate", control),
            ("treatment_rate", treatment),
            ("difference", treatment - control if common else None),
        ]:
            near(record[name], value, "Wrong exact point: " + name)
        check(
            record["reference_error_count"] == control_errors
            and record["treatment_error_count"] == treatment_errors
            and record["reference_output_count"] == len(common) * 3
            and record["treatment_output_count"] == len(common) * len(arms) * 3,
            "Wrong real-output counts",
        )
        for j, arm in enumerate(arms):
            value = (
                Fraction(
                    sum(int(errors[i, arm].sum()) for i in common), 3 * len(common)
                )
                if common
                else None
            )
            if common:
                near(record["pool_rates"][j], value, "Wrong pool mean")
            else:
                check(record["pool_rates"] is None, "Empty pool rate defined")
        observed = [
            sum(
                int(errors[i, j, r])
                for i in eligible
                for r in range(3)
                if valid[i, j, r]
            )
            for j in range(4)
        ]
        missing = [
            sum(not bool(valid[i, j, r]) for i in eligible for r in range(3))
            for j in range(4)
        ]
        check(
            record["observed_errors_by_condition_full_eligible"] == observed
            and record["missing_outputs_by_condition_full_eligible"] == missing,
            "Wrong missingness/error count",
        )
        if eligible:
            lo_c = Fraction(observed[0], len(eligible) * 3)
            hi_c = Fraction(observed[0] + missing[0], len(eligible) * 3)
            lo_t = Fraction(
                sum(observed[j] for j in arms), len(eligible) * len(arms) * 3
            )
            hi_t = Fraction(
                sum(observed[j] + missing[j] for j in arms),
                len(eligible) * len(arms) * 3,
            )
            for name, bounds in [
                ("reference_missing_bounds", (lo_c, hi_c)),
                ("treatment_missing_bounds", (lo_t, hi_t)),
                ("difference_missing_bounds", (lo_t - hi_c, hi_t - lo_c)),
            ]:
                for actual, expected in zip(record[name], bounds):
                    near(actual, expected, "Incorrect opposing missing-output bound")
        else:
            check(
                record["difference_missing_bounds"] is None,
                "Empty-cohort bound defined",
            )
        return common, {
            "reference": str(control),
            "treatment": str(treatment),
            "difference": str(treatment - control) if common else None,
        }

    all_rates = a.read(OUT / "rates.json")
    expected_names = {
        view + "/" + ep for view in VIEWS for ep in science["primary"]["endpoints"]
    }
    for ep in science["primary"]["endpoints"]:
        pop, _ = ep.rsplit("_", 1)
        mode = "source_stratified" if pop == "en_test" else "item"
        expected_names.update(
            mode + "/" + view + "/" + ep
            for view in ("main", "strict", "replay_excluded")
        )
        if pop == "en_test":
            expected_names.update(
                "source/" + src + "/" + ep
                for src in ("hatexplain", "implicit_hate", "mhs")
            )
    check(
        set(all_rates) == expected_names and len(all_rates) == 42,
        "Incomplete comparison inventory",
    )
    evidence = {}
    for name, record in all_rates.items():
        s = record["selection"]
        view = s["view"]
        strict, reps, excluded = VIEWS[view]
        check(
            s["strict"] == strict
            and s["representatives"] == reps
            and s["excluded_keys"]
            == (science["mandatory_replay_keys"] if excluded else []),
            "Selection contract changed",
        )
        eligible, strict = select(
            s["population"], s["gold_hate"], view, s["source_filter"]
        )
        common, exact = point(record, eligible, strict)
        check(
            eligible == record["eligible_indices"]
            and common == record["common_indices"],
            "Wrong saved mask",
        )
        mode = record["resampling_mode"]
        endpoint_index = science["primary"]["endpoints"].index(record["endpoint"])
        seed = (
            science["primary"]["bootstrap_seeds"][endpoint_index]
            if name.split("/")[0] in VIEWS
            else support_seed(name)
        )
        check(seed == record["bootstrap"]["seed"], "Bootstrap seed differs")
        group_labels = sorted(
            {
                meta[i]["key"] if mode == "item" else meta[i]["inference_cluster"]
                for i in common
            }
        )
        group_pos = {label: j for j, label in enumerate(group_labels)}
        group_of_row = np.asarray(
            [
                group_pos[
                    meta[i]["key"] if mode == "item" else meta[i]["inference_cluster"]
                ]
                for i in common
            ],
            dtype=np.int64,
        )
        per_row_counts = errors[common].sum(axis=2)
        streams = []
        stratum_record = []
        if mode == "source_stratified":
            source_membership = defaultdict(set)
            for i in eligible:
                source_membership[meta[i]["inference_cluster"]].add(meta[i]["source"])
            strata = defaultdict(list)
            for j, label in enumerate(group_labels):
                strata[tuple(sorted(source_membership[label]))].append(j)
            for signature, positions in sorted(strata.items()):
                sub_seed = support_seed(
                    "stratum|" + str(seed) + "|" + "|".join(signature)
                )
                streams.append(
                    (
                        np.asarray(positions),
                        np.random.Generator(np.random.PCG64(sub_seed)),
                    )
                )
                stratum_record.append(
                    {
                        "source_signature": list(signature),
                        "units": len(positions),
                        "seed": sub_seed,
                    }
                )
        check(
            record["bootstrap"]["strata"] == stratum_record
            and record["bootstrap"]["units"] == len(group_labels),
            "Unit/stratum inventory differs",
        )
        saved = np.load(a.inside(a.ROOT, record["draw_file"]), allow_pickle=False)
        check(saved.shape == (10000, 6), "Wrong bootstrap shape")
        rng = np.random.Generator(np.random.PCG64(seed))
        largest = 0.0
        for draw in range(10000):
            if streams:
                selection = np.concatenate(
                    [
                        indices[
                            generator.integers(
                                0, len(indices), len(indices), dtype=np.int64
                            )
                        ]
                        for indices, generator in streams
                    ]
                )
            else:
                selection = rng.integers(
                    0, len(group_labels), len(group_labels), dtype=np.int64
                )
            cluster_weight = np.bincount(selection, minlength=len(group_labels))
            row_weight = cluster_weight[group_of_row]
            n = int(row_weight.sum())
            totals = row_weight @ per_row_counts
            independently = [int(total) / (3 * n) for total in totals]
            independently += [
                int(totals[1:].sum()) / (9 * n),
                int(totals[1:].sum() - 3 * totals[0]) / (9 * n),
            ]
            largest = max(largest, float(np.max(np.abs(saved[draw] - independently))))
        check(largest < 1e-13, "Original-text multiplicity bootstrap differs: " + name)
        for field, qs in [
            ("interval98_75", (0.00625, 0.99375)),
            ("interval95", (0.025, 0.975)),
        ]:
            for got, q in zip(record[field], qs):
                near(
                    got,
                    percentile_linear(saved[:, 5], q),
                    "Independent percentile differs",
                )
        for j, interval in enumerate(record["pool_difference_intervals95"]):
            for got, q in zip(interval, (0.025, 0.975)):
                near(
                    got,
                    percentile_linear(saved[:, j + 1] - saved[:, 0], q),
                    "Pool percentile differs",
                )
        evidence[name] = {
            "eligible_n": len(eligible),
            "common_n": len(common),
            "exact_rates": exact,
            "units": len(group_labels),
            "draws_checked": 10000,
            "scalars_checked": 60000,
            "max_absolute_error": largest,
        }
        print("Independently checked " + name, flush=True)

    descriptions = a.read(OUT / "descriptive_errors.json")
    groups = a.read(OUT / "group_rates.json")
    group_checks = 0
    for endpoint, d in descriptions.items():
        pop, kind = endpoint.rsplit("_", 1)
        eligible, strict = select(pop, kind == "FNR", "main")
        common, _ = point(all_rates["main/" + endpoint], eligible, False)
        ref = errors[common, 0, :]
        for row in d["runs"]:
            j = science["conditions"].index(row["condition"])
            counts = int(errors[common, j, row["repeat"]].sum())
            check(
                row["errors"] == counts and row["n"] == len(common),
                "Descriptive run count differs",
            )
            near(
                row["rate"],
                Fraction(counts, len(common)),
                "Descriptive run rate differs",
            )
        for j, row in enumerate(d["pools"], 1):
            treatment = errors[common, j, :]
            check(
                row["matched_repeat_fixes"]
                == int(((ref == 1) & (treatment == 0)).sum())
                and row["matched_repeat_new_errors"]
                == int(((ref == 0) & (treatment == 1)).sum())
                and row["matched_repeat_persistent_errors"]
                == int(((ref == 1) & (treatment == 1)).sum())
                and row["matched_repeat_stable_correct"]
                == int(((ref == 0) & (treatment == 0)).sum()),
                "Transition count differs",
            )
            check(
                row["errors"] == int(treatment.sum())
                and row["outputs"] == treatment.size,
                "Pool count differs",
            )
            near(
                row["rate"],
                Fraction(int(treatment.sum()), treatment.size),
                "Pool rate differs",
            )
            near(
                row["difference"],
                Fraction(int(treatment.sum() - ref.sum()), treatment.size),
                "Pool difference differs",
            )
        direction_counts = Counter()
        variation = Counter({key: 0 for key in d["variation"]})
        for i in common:
            e = errors[i]
            total = e.sum(axis=1)
            delta = int(total[1:].sum() - total[0] * 3)
            direction_counts[
                (
                    "fewer_errors"
                    if delta < 0
                    else "more_errors" if delta > 0 else "same_average_errors"
                )
            ] += 1
            variation["reference_disagreement_inputs"] += int(len(set(e[0])) > 1)
            variation["any_treatment_repeat_disagreement_inputs"] += int(
                any(len(set(x)) > 1 for x in e[1:])
            )
            variation["pool_directions_disagree_inputs"] += int(
                min(total[1:]) < total[0] < max(total[1:])
            )
            variation["all_twelve_correct"] += int(np.all(e == 0))
            variation["all_twelve_wrong"] += int(np.all(e == 1))
            variation["equal_mean_with_offsetting_repeat_changes"] += int(
                delta == 0 and any(not np.array_equal(row, e[0]) for row in e[1:])
            )
        check(
            dict(direction_counts) == d["per_text_average_directions"]
            and dict(variation) == d["variation"],
            "Text-direction/variation counts differ",
        )
        for field, category in [
            ("source", "by_source"),
            ("stage", "by_batch"),
            ("execution_period", "by_execution_period"),
        ]:
            check(
                {r["group"] for r in groups[endpoint][category]}
                == {meta[i][field] for i in eligible},
                "Missing descriptive group",
            )
            for row in groups[endpoint][category]:
                chosen = [i for i in eligible if meta[i][field] == row["group"]]
                point(row, chosen, False)
                group_checks += 1
        for row in groups[endpoint]["own_valid_runs"]:
            j = science["conditions"].index(row["condition"])
            repeat = row["repeat"]
            e = [int(errors[i, j, repeat]) for i in eligible]
            valid = [v for v in e if v >= 0]
            check(
                row["eligible"] == len(e)
                and row["valid"] == len(valid)
                and row["null"] == e.count(-1)
                and row["errors"] == sum(valid),
                "Own-valid diagnostic counts differ",
            )
            near(
                row["rate"],
                Fraction(sum(valid), len(valid)) if valid else None,
                "Own-valid rate differs",
            )

    omission_checks = 0
    for name, omission in a.read(OUT / "omissions.json").items():
        record = all_rates[name]
        s = record["selection"]
        eligible, strict = select(s["population"], s["gold_hate"], s["view"])
        common = record["common_indices"]
        for field in ("pools_fixed_mask", "pools_recomputed_mask"):
            check(len(omission[field]) == 3, "Missing omitted-pool result")
            for row in omission[field]:
                omitted = science["conditions"].index(row["omitted_condition"])
                arms = tuple(j for j in (1, 2, 3) if j != omitted)
                point(
                    row,
                    common if field == "pools_fixed_mask" else eligible,
                    strict,
                    arms=arms,
                )
                omission_checks += 1
        check(
            {row["omitted_batch"] for row in omission["execution_batches"]}
            == {meta[i]["stage"] for i in eligible},
            "Missing omitted batch",
        )
        for row in omission["execution_batches"]:
            point(
                row,
                [i for i in eligible if meta[i]["stage"] != row["omitted_batch"]],
                strict,
            )
            omission_checks += 1

    ledger_count = 0
    for i, row in enumerate(a.rows(OUT / "per_input_error_ledger.jsonl")):
        check(
            row["input_index"] == i
            and all(row[key] == meta[i][key] for key in meta[i]),
            "Ledger identity/metadata differs",
        )
        check(
            row["errors"] == errors[i].tolist()
            and row["predicted_hate"] == gates[i].tolist()
            and row["valid_votes"] == valid_votes[i].tolist()
            and row["missing_sentinel"] == -1,
            "Ledger decisions differ",
        )
        common = bool(global_common[i])
        sums = errors[i].sum(axis=1)
        delta = int(sums[1:].sum() - 3 * sums[0]) if common else None
        direction = (
            "missing_output"
            if delta is None
            else (
                "fewer_errors"
                if delta < 0
                else "more_errors" if delta > 0 else "same_average_errors"
            )
        )
        check(
            row["difference_ninths"] == delta
            and row["direction"] == direction
            and row["in_primary_common_mask"] == common
            and row["strict_all_votes"] == bool((valid_votes[i] == 3).all()),
            "Ledger derived field differs",
        )
        ledger_count += 1
    check(ledger_count == 15758, "Incomplete per-input ledger")
    for name, expected in {**calculated["bindings"], **calculated["artifacts"]}.items():
        check(
            a.sha(a.inside(a.ROOT, name)) == expected,
            "Evidence changed during independent verification",
        )
    save(
        OUT / "independent_check.json",
        {
            "status": "all_part2_numerical_checks_pass",
            "checked_utc": a.utc(),
            "verifier_sha256": a.sha(Path(__file__)),
            "calculated_sha256": a.sha(OUT / "calculated.json"),
            "rate_checks": evidence,
            "bootstrap_sets": 42,
            "bootstrap_draws_checked": 420000,
            "bootstrap_scalars_checked": 2520000,
            "omission_point_checks": omission_checks,
            "descriptive_group_checks": group_checks,
            "ledger_inputs_checked": ledger_count,
            "method": "Independent original-matrix masks and integer/Fraction arithmetic; original-text multiplicity-weighted replay of every PCG64 draw; independent sorted linear percentiles; all transitions, run/group/omission counts and ledger rows.",
            "new_Qwen_calls": 0,
            "new_fits": 0,
            "new_raw_response_replay": False,
            "remaining": "Interpretation, figures, report and navigation publication.",
        },
    )
    save(
        OUT / "verified.json",
        {
            "status": "part2_numerically_verified_interpretation_pending",
            "completed_utc": a.utc(),
            "calculated_sha256": a.sha(OUT / "calculated.json"),
            "independent_check_sha256": a.sha(OUT / "independent_check.json"),
            "verifier_sha256": a.sha(Path(__file__)),
            "new_Qwen_calls": 0,
        },
    )
    print(
        "Every Part 2 numerical check passed; interpretation/publication are next.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()
