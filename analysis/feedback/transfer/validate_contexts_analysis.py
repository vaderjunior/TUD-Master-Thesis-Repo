import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import numpy as np
from analysis.feedback.transfer import validate_contexts_context as v

a = v.a
OUT = v.OUT


def percentile(values, q):
    x = sorted(map(float, values))
    p = (len(x) - 1) * q
    lo = math.floor(p)
    hi = math.ceil(p)
    return x[lo] * (1 - (p - lo)) + x[hi] * (p - lo)


def distribution(values):
    return dict(
        n=len(values),
        mean=math.fsum(values) / len(values) if values else None,
        quantiles=(
            [percentile(values, q) for q in (0, 0.25, 0.5, 0.75, 1)] if values else None
        ),
    )


def concentration(values):
    vals = sorted(values, reverse=True)
    n = sum(vals)
    return dict(
        total=n,
        used=sum(x > 0 for x in vals),
        max_share=Fraction(vals[0], n),
        top5_share=Fraction(sum(vals[:5]), n),
        top10_share=Fraction(sum(vals[:10]), n),
        inverse_simpson=Fraction(n * n, sum(x * x for x in vals)),
    )


def classify(row, group):
    bits = group.split("/")
    name = bits[0]
    if name == "all":
        return True
    value = bits[1]
    if name == "correction":
        return row["source_gate"] == bool(int(value))
    if name == "swap":
        return f"{int(row['removed_gate'])}-{int(row['source_gate'])}" == value
    if name == "cosine":
        return sum(row["cosine"] >= x for x in (0.4, 0.5, 0.6, 0.7)) == int(value)
    if name == "cosine_vs_fifth":
        return (
            "higher"
            if row["cosine"] - row["removed_cosine"] > 1e-12
            else "lower" if row["cosine"] - row["removed_cosine"] < -1e-12 else "tie"
        ) == value
    if name == "removed_fine":
        return any(row["removed_known_fine"].values()) == bool(int(value))
    v.check(name == "baseline_hateful", "Known stratum")
    return row["reference_hateful_examples"] == int(value)


def totals(n, ref, trt):
    return dict(
        assignments=n,
        reference_errors=sum(ref),
        treatment_errors=sum(trt),
        reference=Fraction(sum(ref), 3 * n) if n else None,
        treatment=Fraction(sum(trt), 3 * n) if n else None,
        difference=Fraction(sum(trt) - sum(ref), 3 * n) if n else None,
    )


def rate(ids, group, rows, error, meta):
    pools = []
    selected_texts = set()
    features = {}
    for c in (1, 2, 3):
        chosen = [i for i in ids if classify(rows[3 * i + c - 1], group)]
        r = [sum(int(error[i, 0, t]) for i in chosen) for t in range(3)]
        t = [sum(int(error[i, c, j]) for i in chosen) for j in range(3)]
        pools.append(
            dict(
                totals(len(chosen), r, t),
                reference_errors_by_repeat=r,
                treatment_errors_by_repeat=t,
            )
        )
        selected_texts.update(chosen)
        for i in chosen:
            f = features.setdefault(i, [0, 0, 0])
            f[0] += 1
            f[1] += int(error[i, 0].sum())
            f[2] += int(error[i, c].sum())
    n = sum(z["assignments"] for z in pools)
    ref = [sum(z["reference_errors_by_repeat"][j] for z in pools) for j in range(3)]
    trt = [sum(z["treatment_errors_by_repeat"][j] for z in pools) for j in range(3)]
    result = dict(
        totals(n, ref, trt),
        reference_errors_by_repeat=ref,
        treatment_errors_by_repeat=trt,
        per_pool=pools,
        unique_texts=len(selected_texts),
        clusters=len({meta[i]["inference_cluster"] for i in selected_texts}),
    )
    texts = sorted(selected_texts)
    return result, texts, [features[i] for i in texts]


def replay(name, record, texts, features, meta):
    seed = int.from_bytes(
        hashlib.sha256(("A16-part5-context-v1|" + name).encode()).digest()[:8], "little"
    )
    v.check(seed == record["bootstrap"]["seed"], "Bootstrap seed")
    units = sorted({meta[i]["inference_cluster"] for i in texts})
    index = {u: i for i, u in enumerate(units)}
    text_unit = np.asarray([index[meta[i]["inference_cluster"]] for i in texts])
    values = np.asarray(features, dtype=np.int64)
    v.check(
        record["bootstrap"]["units"] == len(units)
        and record["text_indices"] == texts
        and record["per_text_counts"] == features,
        "Bootstrap text/cluster inputs",
    )
    saved = np.load(a.ROOT / record["draw_file"], allow_pickle=False)
    v.check(saved.shape == (10000, 3), "Draw dimensions")
    rng = np.random.Generator(np.random.PCG64(seed))
    maxerr = 0.0
    for start in range(0, 10000, 32):
        size = min(32, 10000 - start)
        picked = rng.integers(0, len(units), (size, len(units)), dtype=np.int64)
        weights = np.asarray([np.bincount(x, minlength=len(units)) for x in picked])[
            :, text_unit
        ]
        counts = weights @ values
        expected = np.stack(
            [
                counts[:, 1] / (counts[:, 0] * 3),
                counts[:, 2] / (counts[:, 0] * 3),
                (counts[:, 2] - counts[:, 1]) / (counts[:, 0] * 3),
            ],
            axis=1,
        )
        v.check(
            np.array_equal(saved[start : start + size], expected),
            "Every independent bootstrap scalar " + name,
        )
        maxerr = max(maxerr, float(abs(saved[start : start + size] - expected).max()))
    v.numeric(
        record["interval95"],
        [percentile(saved[:, 2], 0.025), percentile(saved[:, 2], 0.975)],
        "Independent interval interpolation",
    )
    return saved.size, maxerr


def main():
    calc = a.read(OUT / "calculated.json")
    context_check = a.read(OUT / "independent_context_check.json")
    assembled = a.read(OUT / "assembled.json")
    v.check(
        context_check["status"] == "part5_original_contexts_and_delivery_verified",
        "Context verification required",
    )
    for name, digest in {**calc["artifacts"], **assembled["artifacts"]}.items():
        v.check(a.sha(a.ROOT / name) == digest, "Changed analysis data " + name)
    rows = list(a.rows(OUT / "delivery.jsonl"))
    meta = list(a.rows(v.P4 / "metadata.jsonl"))
    matrix = list(a.rows(a.OUT / "matrix.jsonl"))
    N = len(meta)
    inv = a.read(OUT / "pool_inventory.json")
    pools = inv["pools"]
    exposure = a.read(OUT / "exposure.json")
    collected = defaultdict(list)
    for row in rows:
        i = row["input_index"]
        keys = [
            f"population/{row['population']}",
            f"source/{row['source']}",
            f"gold/{row['population']}/{int(row['gold_hate'])}",
        ]
        if row["functionality"] is not None:
            keys.append("function/" + row["functionality"])
        for view in ("full", "common", "strict"):
            allow = view == "full" or all(
                slot["result"] is not None
                and (view != "strict" or slot["n_valid"] == 3)
                for arm in matrix[i]["predictions"]
                for slot in arm
            )
            if allow:
                for key in keys:
                    collected[f'{view}/{key}/{row["pool_index"]}'].append(row)
    v.check(
        set(collected) == set(exposure) and len(exposure) == 351,
        "Complete exposure group inventory",
    )
    for name, rr in collected.items():
        ids = pools[rr[0]["pool_index"] - 1]["ids"]
        count = Counter(r["source_id"] for r in rr)
        expected = dict(
            assignments=len(rr),
            unique_texts=len({r["key"] for r in rr}),
            inference_clusters=len({r["inference_cluster"] for r in rr}),
            pool=rr[0]["condition"],
            source_counts={s: count[s] for s in ids},
            source_concentration=concentration([count[s] for s in ids]),
            inserted_hateful=sum(r["source_gate"] for r in rr),
            removed_hateful=sum(r["removed_gate"] for r in rr),
            swap_counts=dict(
                Counter(
                    str(int(r["removed_gate"])) + "-" + str(int(r["source_gate"]))
                    for r in rr
                )
            ),
            hateful_example_change=dict(
                Counter(str(int(r["source_gate"]) - int(r["removed_gate"])) for r in rr)
            ),
            cosine=distribution([r["cosine"] for r in rr]),
            fifth_cosine=distribution([r["removed_cosine"] for r in rr]),
            cosine_difference=distribution(
                [r["cosine"] - r["removed_cosine"] for r in rr]
            ),
            similarity_relation=dict(Counter(r["similarity_relation"] for r in rr)),
            cosine_bins={
                str(j): sum(r["cosine_bin"] == j for r in rr) for j in range(5)
            },
            dense_winner_selected=sum(r["dense_winner_same"] for r in rr),
            retrieval_channels=dict(Counter(r["retrieval_channels"] for r in rr)),
            dense_cosine_regret=distribution([r["dense_cosine_regret"] for r in rr]),
            token_difference=distribution(
                [r["feedback_input_tokens"] - r["reference_input_tokens"] for r in rr]
            ),
            removed_source=dict(Counter(r["removed_source"] for r in rr)),
            removed_known_fields={
                f: sum(r["removed_known_fine"][f] for r in rr) for f in v.FIELDS
            },
            removed_last_known_field={
                f: sum(
                    r["removed_known_fine"][f]
                    and r["remaining_known_field_counts"][f] == 0
                    for r in rr
                )
                for f in v.FIELDS
            },
        )
        v.numeric(exposure[name], expected, "Exposure counts/distributions")
    shared = a.read(OUT / "shared_exposure.json")
    expected_shared = {}
    for pop in ("en_test", "en_hatecheck_eval"):
        for view in ("full", "common", "strict"):
            triples = []
            for i, m in enumerate(meta):
                if m["population"] != pop:
                    continue
                ok = view == "full" or all(
                    slot["result"] is not None
                    and (view != "strict" or slot["n_valid"] == 3)
                    for arm in matrix[i]["predictions"]
                    for slot in arm
                )
                if ok:
                    triples.append(rows[3 * i : 3 * i + 3])
            expected_shared[pop + "/" + view] = dict(
                texts=len(triples),
                unique_source_choices_histogram=dict(
                    Counter(str(len(set(r["source_id"] for r in rr))) for rr in triples)
                ),
                unique_feedback_message_histogram=dict(
                    Counter(
                        str(len(set(r["messages_sha256"] for r in rr)))
                        for rr in triples
                    )
                ),
                same_source_pairs={
                    f"{i+1}-{j+1}": sum(
                        rr[i]["source_id"] == rr[j]["source_id"] for rr in triples
                    )
                    for i, j in [(0, 1), (0, 2), (1, 2)]
                },
                same_message_pairs={
                    f"{i+1}-{j+1}": sum(
                        rr[i]["messages_sha256"] == rr[j]["messages_sha256"]
                        for rr in triples
                    )
                    for i, j in [(0, 1), (0, 2), (1, 2)]
                },
                correction_labels_agree=sum(
                    len(set(r["source_gate"] for r in rr)) == 1 for rr in triples
                ),
                unique_feedback_messages=len(
                    set(r["messages_sha256"] for rr in triples for r in rr)
                ),
            )
    v.numeric(shared, expected_shared, "Every overlap/exposure table")
    errors = np.zeros((N, 4, 3), dtype=np.int8)
    for i, m in enumerate(matrix):
        for c, arm in enumerate(m["predictions"]):
            for t, slot in enumerate(arm):
                errors[i, c, t] = (
                    -1
                    if slot["result"] is None
                    else slot["result"]["hate"] != m["gold_hate"]
                )
    rates = a.read(OUT / "conditional_rates.json")
    prior = a.read(
        a.ROOT / "experiments/astra/sq3_transfer_part2_20261023_v1/rates.json"
    )
    strata = [
        "all",
        *[f"correction/{x}" for x in (0, 1)],
        *[f"swap/{x}-{y}" for x in (0, 1) for y in (0, 1)],
        *[f"cosine/{j}" for j in range(5)],
        *[f"cosine_vs_fifth/{x}" for x in ("lower", "tie", "higher")],
        *[f"removed_fine/{x}" for x in (0, 1)],
        *[f"baseline_hateful/{j}" for j in range(6)],
    ]
    views = (
        "main",
        "strict",
        "replay_excluded",
        "strict_replay_excluded",
        "representative",
        "representative_replay_excluded",
    )
    expected_names = {
        f"population/{p}_{ep}/{view}/{s}"
        for p in ("en_test", "en_hatecheck_eval")
        for ep in ("FPR", "FNR")
        for view in views
        for s in strata
    }
    expected_names |= {
        f"source/{source}/{ep}/{view}/{s}"
        for source in ("hatexplain", "implicit_hate", "mhs")
        for ep in ("FPR", "FNR")
        for view in ("main", "strict")
        for s in strata
    }
    fns = {m["functionality"] for m in meta if m["functionality"] is not None}
    expected_names |= {
        f"function/{fn}/{ep}/main/{s}"
        for fn in fns
        for ep in ("FPR", "FNR")
        for s in strata
        if s == "all" or s.startswith(("correction/", "swap/"))
    }
    v.check(
        set(rates) == expected_names and len(rates) == 1234,
        "Complete conditional-rate catalogue",
    )
    draw_count = scalars = 0
    maxerr = 0.0
    for name, record in rates.items():
        bits = name.split("/")
        kind = bits[0]
        if kind == "population":
            ep = bits[1]
            view = bits[2]
            group = "/".join(bits[3:])
            ids = prior[view + "/" + ep]["common_indices"]
            prior_id = view + "/" + ep
        else:
            view = bits[3]
            group = "/".join(bits[4:])
            ep = ("en_test_" if kind == "source" else "en_hatecheck_eval_") + bits[2]
            ids = [
                i
                for i in prior[view + "/" + ep]["common_indices"]
                if meta[i]["source" if kind == "source" else "functionality"] == bits[1]
            ]
        v.check(
            all(errors[i, c, t] >= 0 for i in ids for c in range(4) for t in range(3)),
            "No missing predictions counted as errors",
        )
        point, texts, features = rate(ids, group, rows, errors, meta)
        v.numeric(
            {key: record[key] for key in point},
            point,
            "Exact conditional rates " + name,
        )
        v.check(record["base_common_texts"] == len(ids), "Conditional base denominator")
        v.numeric(
            record["contribution_to_group_difference"],
            (
                Fraction(
                    point["treatment_errors"] - point["reference_errors"], 9 * len(ids)
                )
                if ids
                else None
            ),
            "Additive rate contribution",
        )
        if kind == "population" and group == "all":
            v.numeric(
                record["reference"],
                prior[prior_id]["reference_rate"],
                "Part 2 reference",
            )
            v.numeric(
                record["treatment"],
                prior[prior_id]["treatment_rate"],
                "Part 2 treatment",
            )
            v.numeric(
                record["interval95"],
                prior[prior_id]["interval95"],
                "Reused Part 2 interval",
            )
        if "bootstrap" in record:
            count, error = replay(name, record, texts, features, meta)
            draw_count += 1
            scalars += count
            maxerr = max(maxerr, error)
    for name, z in rates.items():
        if not name.endswith("/all"):
            continue
        prefix = name[:-3]
        for family in (
            "correction/",
            "swap/",
            "cosine/",
            "cosine_vs_fifth/",
            "removed_fine/",
            "baseline_hateful/",
        ):
            members = [r for k, r in rates.items() if k.startswith(prefix + family)]
            if not members:
                continue
            for field in ("assignments", "reference_errors", "treatment_errors"):
                v.check(
                    sum(r[field] for r in members) == z[field],
                    "Stratum counts partition the whole group",
                )
    quality = a.read(v.P4 / "quality_results.json")
    fine = a.read(OUT / "fine_information_associations.json")
    fine_count = 0
    field = {
        "hate_type": "hate_types",
        "target_group": "target_groups",
        "severity": "severity",
    }
    expected_fine = {
        f"{pop}/{head}/{view}/{measure}/{kind}/{flag}"
        for pop in ("en_test", "en_hatecheck_eval")
        for head in field
        for view in views
        if quality[f"population/{pop}/{head}/{view}"]["common_n"]
        for measure in ("exact", "gated_exact")
        for kind in ("removed_known", "removed_last_known")
        for flag in (0, 1)
    }
    v.check(
        set(fine) == expected_fine and len(fine) == 192, "All fine-information strata"
    )
    failures = {}
    for head in field:
        for measure in ("exact", "gated_exact"):
            array = np.zeros((N, 4, 3), dtype=np.int8)
            for i, m in enumerate(matrix):
                gold = m["gold_" + field[head]]
                if gold is None:
                    continue
                for c, arm in enumerate(m["predictions"]):
                    for t, slot in enumerate(arm):
                        pred = slot["result"]
                        if pred is None:
                            array[i, c, t] = -1
                            continue
                        same = (
                            pred[head] == gold
                            if head == "severity"
                            else set(pred[head]) == set(gold)
                        )
                        array[i, c, t] = not (
                            same and (measure != "gated_exact" or pred["hate"])
                        )
            failures[head, measure] = array
    for name, z in fine.items():
        pop, head, view, measure, kind, flag = name.split("/")
        ids = quality[f"population/{pop}/{head}/{view}"]["common_indices"]
        eligible = []
        for i in ids:
            row = rows[i * 3]
            yes = row["removed_known_fine"][field[head]] and (
                kind != "removed_last_known"
                or row["remaining_known_field_counts"][field[head]] == 0
            )
            if yes == bool(int(flag)):
                eligible.append(i)
        point, _, _ = rate(eligible, "all", rows, failures[head, measure], meta)
        expected = dict(
            point,
            id=name,
            reference_exact=(
                1 - point["reference"] if point["reference"] is not None else None
            ),
            treatment_exact=(
                1 - point["treatment"] if point["treatment"] is not None else None
            ),
            exact_difference=(
                -point["difference"] if point["difference"] is not None else None
            ),
        )
        v.numeric(z, expected, "Original-output fine information association")
        fine_count += 1
    v.check(draw_count == 24 and scalars == 720000, "Draw inventory")
    receipt = dict(
        status="part5_contexts_exposure_and_outcomes_independently_verified_interpretation_pending",
        checked_utc=a.utc(),
        exposure_views=len(exposure),
        shared_exposure_views=len(shared),
        conditional_rate_views=len(rates),
        fine_information_views=fine_count,
        new_bootstrap_sets=draw_count,
        new_scalar_values=scalars,
        max_bootstrap_difference=maxerr,
        all_integer_fraction_counts_and_contributions_checked=True,
        all_source_memberships_including_zero_use_retained=True,
        calculated_sha256=a.sha(OUT / "calculated.json"),
        independent_context_sha256=a.sha(OUT / "independent_context_check.json"),
        verifier_sha256=a.sha(Path(__file__)),
        new_Qwen_calls=0,
        new_fits=0,
        new_embeddings=0,
    )
    with (OUT / "verified.json").open("xb") as f:
        f.write(a.encoded(receipt, pretty=True))
    print(
        "All exposure, conditional rates, fine-information strata and 24 draw sets independently verified.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()
