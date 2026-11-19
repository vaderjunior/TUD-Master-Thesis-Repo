from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import math
import hashlib
import numpy as np
from analysis.feedback.rounds import resources_inputs as a


def close(left, right, message, atol=1e-7):
    a.need(np.allclose(left, right, rtol=2e-13, atol=atol), message)


def main():
    proof = a.read(a.OUT / "assembled.json")
    a.check(proof["input_bindings"])
    a.check(proof["source_hashes"])
    a.check(proof["artifacts"])
    scored = a.read(a.OUT / "scored.json")
    a.check(scored["inputs"])
    a.check(scored["artifacts"])
    a.check(scored["source_hashes"])
    reader = a.read(a.READER / "complete.json")
    meta = list(a.rows(a.REVIEW / "part7/metadata.jsonl"))
    index = {r["key"]: i for i, r in enumerate(meta)}
    function = {
        r["key"]: r.get("functionality")
        for r in a.rows(a.REVIEW / "part5/metadata.jsonl")
    }
    prompts = {
        (r["phase"], r["key"], r["condition"]): r
        for r in a.rows(a.REVIEW / "part3/prompt_bindings.jsonl")
    }
    with np.load(a.OUT / "arrays.npz", allow_pickle=False) as arrays:
        x = arrays["resources"]
        local = arrays["local_prompt_tokens"]
    groups = a.read(a.OUT / "resource_groups.json")
    numeric = {
        name: dict(
            accounting=Counter(),
            observed_usage=Counter(),
            ordinary_usage=Counter(),
            repair_usage=Counter(),
            counts=Counter(),
            durations=[],
            backoffs=[],
        )
        for name in groups
    }
    seen = set()
    rows = 0
    source_rows = []
    targeted = 0
    for st in reader["stage_receipts"]:
        receipt = a.read(a.ROOT / st["receipt"])
        a.need(
            a.sha(a.ROOT / st["predictions"]) == receipt["predictions_sha256"],
            "Compact scientific evidence differs",
        )
        for r in a.rows(a.ROOT / st["predictions"]):
            rows += 1
            ac = r["accounting"]
            u = r["observed_usage"]
            ordinary = r["ordinary_usage"]
            repair = {f: r["repair_usage"].get(f, 0) for f in u}
            a.need(
                ac["ordinary_logical_calls"] == 3
                and ac["post_attempts"]
                == 3 + ac["repair_logical_calls"] + ac["retry_attempts"]
                and ac["http_success_with_text"] == 3 + ac["repair_logical_calls"],
                "Ordinary/repair/retry boundary differs",
            )
            a.need(
                u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
                and all(u[f] == ordinary[f] + repair[f] for f in u),
                "Token partition differs",
            )
            a.need(
                len(r["known_request_durations_s"])
                + r["administrative_unknown_requests"]
                == ac["post_attempts"],
                "Attempt clock coverage differs",
            )
            names = [
                "all",
                "role/" + r["role"],
                "phase/" + r["phase"],
                "stage/" + r["phase"] + "/" + r["stage"],
                "valid_votes/" + str(r["n_valid"]),
                "repair_used/" + str(bool(r["repairs"])),
                "retry_used/" + str(bool(ac["retry_attempts"])),
                "population/" + r["population"],
                "repeat/" + str(r["repeat"]),
            ]
            if r["role"] == "evaluation":
                i = index[r["key"]]
                rr = r["round"] - 1
                c = a.CONDITIONS.index(r["condition"])
                d = r["repeat"]
                m = meta[i]
                slot = (i, rr, c, d)
                a.need(slot not in seen, "Duplicate evaluation cost slot")
                seen.add(slot)
                expected = [
                    u["total_tokens"],
                    ordinary["total_tokens"],
                    repair["total_tokens"],
                    ac["repair_logical_calls"],
                    ac["retry_attempts"],
                    ac["post_attempts"],
                    math.fsum(r["known_request_durations_s"]),
                    r["retry_backoff_seconds_sum"],
                ]
                close(x[slot], expected, "Resource array differs", 1e-10)
                a.need(
                    local[i, rr, c]
                    == prompts[r["phase"], r["key"], r["condition"]][
                        "local_token_estimate"
                    ],
                    "Local token count differs",
                )
                names.extend(
                    [
                        "policy/" + r["policy"],
                        "round/" + str(r["round"]),
                        "round_population_policy/"
                        + str(r["round"])
                        + "/"
                        + r["population"]
                        + "/"
                        + r["policy"],
                        "source_gold/" + m["source"] + "/" + str(int(m["gold_hate"])),
                        "population_policy/" + r["population"] + "/" + r["policy"],
                    ]
                )
                if function[r["key"]] is not None:
                    names.append("function/" + function[r["key"]])
                names.extend(
                    [
                        "mask/round_common/" + str(m["masks"]["round_common"][rr]),
                        "mask/round_strict/" + str(m["masks"]["round_strict"][rr]),
                        "mask/all_round_common/" + str(m["masks"]["all_round_common"]),
                    ]
                )
            else:
                source_rows.append(r)
            if r["repairs"] or ac["retry_attempts"]:
                targeted += 1
            counts = dict(
                predictions=1,
                nulls=int(r["result"] is None),
                administrative_unknown_requests=r["administrative_unknown_requests"],
                parse_failures=r["parse_failures"],
                repairs=r["repairs"],
                repair_used=int(bool(r["repairs"])),
                retry_used=int(bool(ac["retry_attempts"])),
                **{"valid_votes_" + str(r["n_valid"]): 1},
            )
            for name in names:
                n = numeric[name]
                for field, value in [
                    ("accounting", ac),
                    ("observed_usage", u),
                    ("ordinary_usage", ordinary),
                    ("repair_usage", repair),
                    ("counts", counts),
                ]:
                    n[field].update(value)
                n["durations"].extend(r["known_request_durations_s"])
                n["backoffs"].append(r["retry_backoff_seconds_sum"])
        print(
            "Independent resource census " + st["phase"] + "/" + st["stage"], flush=True
        )
    a.need(
        rows == 259224
        and len(seen) == 249240
        and len(source_rows) == 9984
        and targeted == proof["targeted_results"],
        "Resource census coverage differs",
    )
    for name, n in numeric.items():
        saved = groups[name]
        for field in [
            "counts",
            "accounting",
            "observed_usage",
            "ordinary_usage",
            "repair_usage",
        ]:
            a.need(
                n[field] == Counter(saved[field]),
                "Grouped resource mismatch: " + name + "/" + field,
            )
        ds = np.asarray(n["durations"])
        total = math.fsum(n["durations"])
        noutputs = n["counts"]["predictions"]
        known = len(ds)
        close(
            [
                total,
                known,
                total / known,
                float(ds.min()),
                float(ds.max()),
                *np.quantile(ds, [0.5, 0.95, 0.99]),
                math.fsum(n["backoffs"]),
            ],
            [
                saved[f]
                for f in [
                    "known_request_seconds_sum",
                    "known_request_duration_count",
                    "known_request_seconds_mean",
                    "known_request_seconds_min",
                    "known_request_seconds_max",
                    "known_request_seconds_p50",
                    "known_request_seconds_p95",
                    "known_request_seconds_p99",
                    "retry_backoff_seconds_sum",
                ]
            ],
            "Clock summaries differ: " + name,
        )
        for field, num in [
            ("observed_tokens", n["observed_usage"]["total_tokens"]),
            ("prompt_tokens", n["observed_usage"]["prompt_tokens"]),
            ("completion_tokens", n["observed_usage"]["completion_tokens"]),
            ("repair_calls", n["accounting"]["repair_logical_calls"]),
            ("attempt_slots", n["accounting"]["post_attempts"]),
            ("known_request_seconds", total),
        ]:
            close(
                saved["per_planned_prediction"][field],
                num / noutputs,
                "Per-output denominator differs",
            )
    del numeric, source_rows
    analysis = a.read(a.OUT / "analysis.json")
    points = 0
    scalars = 0
    largest = 0
    for name, r in analysis["records"].items():
        rr = r["round"] - 1
        ids = [
            i
            for i, m in enumerate(meta)
            if m["population"] == r["population"]
            and (
                r["mask"] == "all_roster"
                or (
                    m["masks"]["round_common"][rr]
                    if r["mask"] == "round_common"
                    else m["masks"]["all_round_common"]
                )
            )
        ]
        a.need(ids == r["indices"], "Cost cohort differs")
        xx = x[ids, rr]
        features = np.empty((len(ids), 8, 7))
        for j in range(len(ids)):
            for metric in range(8):
                rates = [
                    math.fsum(float(xx[j, c, d, metric]) for c in cs for d in range(3))
                    / (3 * len(cs))
                    for cs in [[0], [1, 4, 7], [2, 5, 8], [3, 6, 9]]
                ]
                features[j, metric] = rates + [
                    rates[1] - rates[2],
                    rates[1] - rates[3],
                    rates[1] - rates[0],
                ]
        expected = np.mean(features, axis=0)
        close(expected, r["points"], "Independent cost mean differs", 2e-10)
        points += 56
        for metric in range(6):
            rate = [
                Fraction(
                    sum(
                        int(xx[j, c, d, metric])
                        for j in range(len(ids))
                        for c in cs
                        for d in range(3)
                    ),
                    len(ids) * len(cs) * 3,
                )
                for cs in [[0], [1, 4, 7], [2, 5, 8], [3, 6, 9]]
            ]
            exact = [
                float(v)
                for v in rate
                + [rate[1] - rate[2], rate[1] - rate[3], rate[1] - rate[0]]
            ]
            close(
                exact,
                r["points"][metric],
                "Integer/Fraction resource points differ",
                2e-10,
            )
        labels = sorted({meta[i]["inference_cluster"] for i in ids})
        a.need(labels == r["cluster_labels"], "Bootstrap cluster membership differs")
        members = [
            np.flatnonzero(np.array([meta[i]["inference_cluster"] for i in ids]) == lab)
            for lab in labels
        ]
        sums = np.array([features[z].sum(axis=0) for z in members])
        sizes = np.array([len(z) for z in members])
        seed = int.from_bytes(
            hashlib.sha256(("A17-part8-resources-v1|" + name).encode()).digest()[:8],
            "little",
        )
        a.need(seed == r["seed"], "Bootstrap seed differs")
        with np.load(a.ROOT / r["draw_file"], allow_pickle=False) as file:
            draw = file["draws"]
        a.need(
            a.sha(a.ROOT / r["draw_file"]) == r["draw_sha256"],
            "Saved cost draws changed",
        )
        regen = np.empty_like(draw)
        rng = np.random.Generator(np.random.PCG64(seed))
        for j in range(len(draw)):
            chosen = rng.integers(0, len(labels), size=len(labels))
            weights = np.bincount(chosen, minlength=len(labels))
            denom = int(weights @ sizes)
            regen[j] = np.tensordot(weights, sums, axes=(0, 0)) / denom
        delta = float(np.max(abs(regen - draw)))
        largest = max(largest, delta)
        close(regen, draw, "All independent bootstrap draws differ", 2e-9)
        close(
            np.quantile(regen, [0.025, 0.975], axis=0).transpose(1, 2, 0),
            r["intervals"],
            "Cost interval differs",
            2e-9,
        )
        scalars += draw.size
        print("Independent paired costs " + name, flush=True)
    a.need(points == 1344 and scalars == 13440000, "Cost numerical coverage differs")
    total = analysis["round_increments"][-1]
    a.need(
        total["cumulative_tokens"] == 737894387
        and total["cumulative_ordinary_calls"] == 777672
        and total["cumulative_outputs"] == 259224,
        "Marginal totals differ",
    )
    science = a.read(a.REVIEW / "part2/errors_v1/results.json")
    quality = a.read(a.REVIEW / "part6/analysis.json")
    for r in analysis["fixed_error_curves"]:
        original = science["records"][r["source_record"]]
        a.need(r["n"] == original["inputs"], "Error-curve cohort changed")
        for col, point in r["points"].items():
            a.need(
                point == original["points"][col]
                and r["intervals"][col] == original["intervals"][col],
                "Inherited error-curve value changed",
            )
    for r in analysis["fixed_quality_curves"]:
        original = quality["records"][
            f'population/{r["population"]}|known_answer|fixed'
        ]
        met = original["metrics"].index("known_answer_exact")
        a.need(r["n"] == original["n"], "Quality-curve cohort changed")
        for col, point in r["points"].items():
            j = original["column_order"].index(col)
            a.need(
                point == original["estimates"][j][met]
                and r["intervals"][col] == original["intervals"][j][met],
                "Inherited quality changed",
            )
    for primary in analysis["original_primary"]:
        original = science["records"][primary["name"]]
        a.need(
            all(original[f] == v for f, v in primary.items()),
            "Original primary family changed",
        )
    a.write(
        a.OUT / "independent_verification.json",
        dict(
            status="part8_independent_resource_and_cost_verification_pass",
            created_utc=a.pure.utc(),
            source_sha256=a.sha(Path(__file__)),
            inputs=a.bindings(
                [
                    a.OUT / f
                    for f in [
                        "assembled.json",
                        "scored.json",
                        "analysis.json",
                        "resource_groups.json",
                        "arrays.npz",
                        "chronology.json",
                    ]
                ]
                + [
                    a.READER / "complete.json",
                    a.REVIEW / "part2/errors_v1/results.json",
                    a.REVIEW / "part6/analysis.json",
                ]
            ),
            compact_outputs=rows,
            evaluation_slots=len(seen),
            resource_groups=len(groups),
            exact_resource_points=points,
            bootstrap_sets=24,
            bootstrap_scalars=scalars,
            max_bootstrap_difference=largest,
            inherited_error_curve_rows=16,
            inherited_quality_curve_rows=8,
            primary_contrasts_preserved=4,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "Independent resources and 13.44 million bootstrap scalar values PASS.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()
