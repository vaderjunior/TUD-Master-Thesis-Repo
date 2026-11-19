from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import numpy as np
from analysis.feedback.transfer import execution as p
from analysis.feedback.transfer import validate_quality as iv

a = p.a
OUT = p.OUT


def near(got, want, label):
    iv.near(got, want, label)


def ratio(x, y):
    return Fraction(int(x), int(y)) if y else None


def point(error, nv, eligible, common, strict):
    n = len(eligible)
    cn = len(common)
    av = (nv[eligible] == 3) if strict else (nv[eligible] > 0)
    obs = np.where(av, error[eligible], 0).sum(axis=(0, 2), dtype=np.int64)
    miss = (~av).sum(axis=(0, 2), dtype=np.int64)
    actual = error[common].sum(axis=(0, 2), dtype=np.int64)
    rl, rh = ratio(obs[0], 3 * n), ratio(obs[0] + miss[0], 3 * n)
    tl, th = ratio(obs[1:].sum(), 9 * n), ratio((obs + miss)[1:].sum(), 9 * n)
    return dict(
        eligible_n=n,
        common_n=cn,
        excluded_by_availability=n - cn,
        reference_rate=ratio(actual[0], 3 * cn),
        treatment_rate=ratio(actual[1:].sum(), 9 * cn),
        difference=ratio(actual[1:].sum() - 3 * actual[0], 9 * cn),
        reference_error_count=int(actual[0]),
        treatment_error_count=int(actual[1:].sum()),
        reference_output_count=3 * cn,
        treatment_output_count=9 * cn,
        pool_rates=[ratio(actual[c], 3 * cn) for c in (1, 2, 3)] if cn else None,
        observed_errors_by_condition_full_eligible=obs,
        missing_outputs_by_condition_full_eligible=miss,
        reference_missing_bounds=[rl, rh] if n else None,
        treatment_missing_bounds=[tl, th] if n else None,
        difference_missing_bounds=[tl - rh, th - rl] if n else None,
    )


def endpoint_ids(meta, codes, endpoint, view, excluded):
    pop, ep = endpoint.rsplit("_", 1)
    members = [
        i
        for i, m in enumerate(meta)
        if m["population"] == pop and m["gold_hate"] == (ep == "FNR")
    ]
    return iv.select(members, meta, codes, "binary", view, excluded)


def bounds(lo, hi):
    n = len(lo)
    lc = lo.sum(axis=(0, 2), dtype=np.int64)
    hc = hi.sum(axis=(0, 2), dtype=np.int64)
    return dict(
        inputs=n,
        lower_error_counts=lc,
        upper_error_counts=hc,
        variable_output_counts=(lo != hi).sum(axis=(0, 2), dtype=np.int64),
        reference=[ratio(lc[0], 3 * n), ratio(hc[0], 3 * n)] if n else None,
        treatment=(
            [ratio(lc[1:].sum(), 9 * n), ratio(hc[1:].sum(), 9 * n)] if n else None
        ),
        difference=(
            [
                ratio(lc[1:].sum() - 3 * hc[0], 9 * n),
                ratio(hc[1:].sum() - 3 * lc[0], 9 * n),
            ]
            if n
            else None
        ),
    )


def main():
    a.need(
        not (OUT / "verified.json").exists(),
        "Preserve previous independent verification",
    )
    assembled = a.read(OUT / "assembled.json")
    calc = a.read(OUT / "calculated.json")
    bound = {
        **assembled["input_bindings"],
        **assembled["analysis_sources"],
        **calc["artifacts"],
    }
    for name, digest in bound.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed bound evidence " + name)
    a.need(
        calc["assembled_sha256"] == a.sha(OUT / "assembled.json"),
        "Current assembly binding",
    )
    contract = a.read(p.P4 / "analysis_contract.json")
    schema = a.read(p.P4 / "feature_schema.json")
    meta, data, codes = iv.rebuild(contract, schema)
    nv = codes["votes"]
    N = len(meta)
    gold = np.array([m["gold_hate"] for m in meta])
    error = np.where(nv > 0, codes["gates"] != gold[:, None, None], -1).astype(np.int8)
    with np.load(p.P4 / "features.npz", allow_pickle=False) as f:
        for head in data:
            a.need(
                np.array_equal(data[head], f[head]),
                "Original sufficient statistics " + head,
            )
    print(
        "Reconstructed original gold, masks and output scores independently.",
        flush=True,
    )

    boundary = a.read(p.B / "boundary.json")
    old = set(boundary["completed_prediction_identity_hashes"])
    before = a.read(a.ROOT / boundary["checkpoint"])
    new = {}
    for path in sorted((p.B / "provenance").glob("transfer_*/*.json")):
        q = a.read(path)
        a.need(
            q["effective_workers"] == 128 and q["amendment"]["sha256"] == a.sha(p.M),
            "Actual 128-worker provenance",
        )
        inv = a.read(a.ROOT / q["original_attempt"] / "inventory.json")
        for item in inv["pending_identity_hashes"]:
            a.need(item not in new, "Duplicated resumed item")
            new[item] = q["stage"]
    a.need(
        not old & set(new)
        and len(old) == 31565
        and len(new) == 157531
        and not boundary["partial_item_identity_hashes"],
        "Disjoint actual runtime inventories",
    )
    with np.load(OUT / "workers.npz", allow_pickle=False) as f:
        workers = f["workers"]
    records = iter(p.gzrows(OUT / "output_execution.jsonl.gz"))
    seen = set()
    nulls = []
    failure = defaultdict(
        lambda: dict(
            outputs=0,
            valid_vote_counts=Counter(),
            repairs=0,
            parse_failures=0,
            finish_reasons=Counter(),
            response_classes=Counter(),
            reported_models=Counter(),
            gate_ties=0,
        )
    )
    for path in sorted((a.OUT / "assembly").glob("transfer_*.jsonl")):
        for original in a.rows(path):
            row = next(records)
            i, c, t = (
                original["input_index"],
                p.CONDITIONS.index(original["condition"]),
                original["repeat"],
            )
            item = Path(original["item_receipt"]).parent.name
            w = 8 if item in old else 128
            a.need((i, c, t) not in seen, "Duplicate actual slot")
            seen.add((i, c, t))
            a.need(
                workers[i, c, t] == w and (item in old or item in new),
                "Runtime policy assignment",
            )
            for key, value in row.items():
                if key in original:
                    a.need(value == original[key], "Original output metadata " + key)
            a.need(
                row["identity_hash"] == item
                and row["effective_workers"] == w
                and row["prior_checkpoint_member"] == (w == 8),
                "Output provenance identity",
            )
            if w == 8:
                a.need(
                    before[original["item_receipt"]] == original["item_receipt_sha256"]
                    and before[
                        original["item_receipt"].replace(
                            "/complete.json", "/result.json"
                        )
                    ]
                    == original["result_sha256"],
                    "Old result/hash preservation",
                )
            else:
                a.need(new[item] == original["stage"], "Pending item stage")
            if row["n_valid"] == 0:
                nulls.append(row)
            for group in (
                f"all/workers_{w}",
                f"{row['population']}/workers_{w}",
                f"{meta[i]['source']}/gold_{int(meta[i]['gold_hate'])}/workers_{w}",
                f"batch/{row['stage']}",
            ):
                f = failure[group]
                f["outputs"] += 1
                f["valid_vote_counts"][str(row["n_valid"])] += 1
                f["repairs"] += row["repairs"]
                f["parse_failures"] += row["parse_failures"]
                f["gate_ties"] += int(row["uncertain"])
                for name in ("finish_reasons", "response_classes", "reported_models"):
                    f[name].update(row[name])
    a.need(
        next(records, None) is None and len(seen) == 189096,
        "Runtime ledger exact coverage",
    )
    near(
        a.read(OUT / "execution_failure_groups.json"),
        failure,
        "All parser/failure group descriptions",
    )
    a.need(a.read(OUT / "null_outputs.json") == nulls, "Every null output retained")
    era = [
        (
            "all_8"
            if (workers[i] == 8).all()
            else "all_128" if (workers[i] == 128).all() else "cross_boundary"
        )
        for i in range(N)
    ]
    inputs = list(p.gzrows(OUT / "input_execution.jsonl.gz"))
    a.need(
        Counter(era) == Counter(all_8=2630, all_128=13127, cross_boundary=1),
        "Whole-input runtime eras",
    )
    for i, row in enumerate(inputs):
        a.need(
            row["key"] == meta[i]["key"]
            and row["actual_execution_era"] == era[i]
            and row["effective_workers"] == workers[i].tolist()
            and row["valid_votes"] == nv[i].tolist(),
            "Input runtime ledger",
        )
    del before
    print(
        "Verified all actual runtime assignments and preserved old result identities.",
        flush=True,
    )
    exclusions = a.read(a.STUDY / "scientific_plan_v1.json")["mandatory_replay_keys"]
    rates = a.read(OUT / "era_rates.json")
    scalar_draws = 0
    max_error = 0.0
    for name, q in rates.items():
        parts = name.split("/")
        e, c = endpoint_ids(meta, codes, q["endpoint"], q["view"], exclusions)

        def member(i):
            if parts[0] == "era":
                return era[i] == parts[1]
            if parts[0] == "source_era":
                return meta[i]["source"] == parts[1] and era[i] == parts[2]
            if parts[0] == "coarse_period":
                return meta[i]["execution_period"] == parts[1]
            return (
                era[i] == "cross_boundary"
                if parts[0] == "cross_boundary"
                else era[i] != "cross_boundary"
            )

        e = [i for i in e if member(i)]
        c = [i for i in c if member(i)]
        a.need(
            e == q["eligible_indices"] and c == q["common_indices"], "Era masks " + name
        )
        expected = point(
            error, nv, e, c, q["view"] in ("strict", "strict_replay_excluded")
        )
        near({k: q[k] for k in expected}, expected, "Exact era point/bounds " + name)
        near(
            q["per_run_errors"],
            error[c].sum(axis=0, dtype=np.int64),
            "All era actual-run errors",
        )
        a.need(
            q["clusters"] == len({meta[i]["inference_cluster"] for i in c})
            and q["source_counts"] == dict(Counter(meta[i]["source"] for i in c)),
            "Era clustering/composition",
        )
        if "draw_file" not in q:
            continue
        seed = int.from_bytes(
            hashlib.sha256(("A16-part7-support-v1|" + name).encode()).digest()[:8],
            "little",
        )
        a.need(seed == q["bootstrap"]["seed"], "Independent seed convention")
        units = sorted({meta[i]["inference_cluster"] for i in c})
        pos = {u: j for j, u in enumerate(units)}
        text_unit = np.array([pos[meta[i]["inference_cluster"]] for i in c])
        rng = np.random.Generator(np.random.PCG64(seed))
        original = error[c].sum(axis=2, dtype=np.int64)
        saved = np.load(a.ROOT / q["draw_file"], allow_pickle=False)
        a.need(saved.shape == (10000, 6), "Six columns per new draw")
        for start in range(0, 10000, 32):
            n = min(32, 10000 - start)
            chosen = rng.integers(0, len(units), (n, len(units)), dtype=np.int64)
            weights = np.array(
                [np.bincount(row, minlength=len(units)) for row in chosen]
            )[:, text_unit]
            total = weights @ original
            size = weights.sum(axis=1)
            control = total[:, 0] / (3 * size)
            treat = total[:, 1:].sum(axis=1) / (9 * size)
            expected = np.column_stack(
                [
                    total / (3 * size[:, None]),
                    treat,
                    (total[:, 1:].sum(axis=1) - 3 * total[:, 0]) / (9 * size),
                ]
            )
            actual = saved[start : start + n]
            a.need(
                np.allclose(expected, actual, rtol=0, atol=1e-15),
                "Independent original-text bootstrap " + name,
            )
            max_error = max(max_error, float(np.max(np.abs(expected - actual))))
        near(
            q["interval95"],
            [iv.percentile(saved[:, 5], 0.025), iv.percentile(saved[:, 5], 0.975)],
            "Independent era percentile",
        )
        pd = saved[:, 1:4] - saved[:, :1]
        near(
            q["pool_difference_intervals95"],
            np.stack([iv.percentile(pd, 0.025), iv.percentile(pd, 0.975)], axis=1),
            "Independent pool percentile",
        )
        scalar_draws += saved.size
    a.need(
        len(rates) == 124 and scalar_draws == 960000, "Complete new rate/draw inventory"
    )
    print(
        "Verified 124 rate views and all 960,000 new bootstrap scalar values.",
        flush=True,
    )
    qresults = a.read(OUT / "era_quality.json")
    defined = 0
    for name, q in qresults.items():
        members = [
            i
            for i, m in enumerate(meta)
            if m["population"] == q["population"]
            and (
                era[i] != "cross_boundary"
                if q["group"] == "boundary_excluded"
                else era[i] == q["group"]
            )
        ]
        head = q["head"]
        e, c = iv.select(members, meta, codes, head, q["view"], exclusions)
        a.need(
            e == q["eligible_indices"] and c == q["common_indices"], "Quality era masks"
        )
        if not c:
            a.need(not q["metrics"], "Unscorable quality stays empty")
            continue
        counts = data[head][c].sum(axis=0, dtype=np.int64)
        cols = schema["features"][head]
        metrics = schema["metrics"][head]
        labels = contract["labels"][head]
        near(q["counts"], counts, "Original quality-era counts")
        rr, mm, pp = iv.fractional_summary(
            counts, head, q["macro_indices"], labels, cols, metrics
        )
        near(q["run_scores"], rr, "Fractional actual-run quality")
        near(q["condition_means"], mm, "Fractional condition mean")
        near(q["metrics"], pp, "Fractional quality differences")
        iv.check_bounds(q, meta, data[head], codes, labels, cols, metrics)
        defined += 1
    a.need(len(qresults) == 72 and defined == 48, "Quality-era inventory")

    with np.load(p.P6 / "arrays.npz", allow_pickle=False) as f:
        raw = f["votes"][..., 0]
    lo = np.zeros((N, 4, 3), dtype=np.int8)
    hi = np.zeros_like(lo)
    for i in range(N):
        for c in range(4):
            for t in range(3):
                observed = [int(x) for x in raw[i, c, t] if x >= 0]
                outcomes = [
                    int((sum(observed) + sum(fill) >= 2) != bool(gold[i]))
                    for fill in product((0, 1), repeat=3 - len(observed))
                ]
                lo[i, c, t] = min(outcomes)
                hi[i, c, t] = max(outcomes)
    with np.load(OUT / "vote_completion_arrays.npz", allow_pickle=False) as f:
        a.need(
            np.array_equal(lo, f["lower_error"])
            and np.array_equal(hi, f["upper_error"]),
            "All enumerated vote-completion possibilities",
        )
    bresults = a.read(OUT / "vote_completion_bounds.json")
    for name, q in bresults.items():
        pop, ep = q["endpoint"].rsplit("_", 1)
        ids = [
            i
            for i, m in enumerate(meta)
            if m["population"] == pop and m["gold_hate"] == (ep == "FNR")
        ]
        group = q["group"]
        if group.startswith("source/"):
            ids = [i for i in ids if meta[i]["source"] == group.split("/")[1]]
        elif group != "full_roster":
            ids = [i for i in ids if era[i] == group]
        a.need(ids == q["indices"], "Vote-bound cohort")
        ee = error[ids]
        vv = nv[ids]
        for key, x, y in [
            ("null_only", np.where(vv > 0, ee, 0), np.where(vv > 0, ee, 1)),
            (
                "all_under_three_unknown",
                np.where(vv == 3, ee, 0),
                np.where(vv == 3, ee, 1),
            ),
            ("invalid_vote_completion", lo[ids], hi[ids]),
        ]:
            near(q[key], bounds(x, y), "Independent Fraction completion bounds")
        u = lo[ids] != hi[ids]
        under = vv < 3
        for key, want in dict(
            fewer_than_three_outputs=int(under.sum()),
            already_determined_despite_invalid_votes=int((under & ~u).sum()),
            binary_undetermined_outputs=int(u.sum()),
            inputs_with_any_under_three=int(under.any(axis=(1, 2)).sum()),
            inputs_with_any_undetermined=int(u.any(axis=(1, 2)).sum()),
        ).items():
            a.need(q[key] == want, "Vote determination count")
        if ids:
            null, logical, strict = [
                q[key]["difference"]
                for key in (
                    "null_only",
                    "invalid_vote_completion",
                    "all_under_three_unknown",
                )
            ]
            a.need(
                strict[0] - 1e-14 <= logical[0] <= null[0] + 1e-14
                and null[1] - 1e-14 <= logical[1] <= strict[1] + 1e-14,
                "Nested missing-vote bounds",
            )
    print(
        "Verified 72 quality views and every hypothetical missing-vote completion bound.",
        flush=True,
    )

    availability = a.read(OUT / "availability.json")
    for name, q in availability.items():
        parts = name.split("/")

        def keep(i):
            if name == "all":
                return True
            if parts[0] in ("population", "source", "stage"):
                return meta[i][parts[0]] == parts[1] and (
                    len(parts) == 2 or era[i] == parts[2]
                )
            if parts[0] == "era":
                return era[i] == parts[1]
            return (
                meta[i]["population"] == parts[0]
                and int(meta[i]["gold_hate"]) == int(parts[1][-1])
                and era[i] == parts[2]
            )

        ids = [i for i in range(N) if keep(i)]
        v = nv[ids]
        null = v == 0
        under = v < 3
        want = dict(
            inputs=len(ids),
            slots=len(ids) * 12,
            inputs_any_null=int(null.any(axis=(1, 2)).sum()),
            inputs_any_under_three=int(under.any(axis=(1, 2)).sum()),
            common_inputs=int((~null.any(axis=(1, 2))).sum()),
            strict_inputs=int((~under.any(axis=(1, 2))).sum()),
            valid_vote_histogram=np.bincount(v.reshape(-1), minlength=4),
            per_condition_repeat_vote_counts=np.stack(
                [(v == n).sum(axis=0, dtype=np.int64) for n in range(4)]
            ),
            affected_input_source_gold=Counter(
                f"{meta[i]['source']}/gold_{int(meta[i]['gold_hate'])}"
                for i in ids
                if (nv[i] == 0).any()
            ),
            null_outputs_per_input_histogram=Counter(map(int, null.sum(axis=(1, 2)))),
            under_three_per_input_histogram=Counter(map(int, under.sum(axis=(1, 2)))),
        )
        near(q, p.native(want), "Availability view " + name)

    influence = a.read(OUT / "group_influence_summary.json")
    rows = 0
    byview = defaultdict(list)
    cache = {}
    for q in p.gzrows(OUT / "group_omissions.jsonl.gz"):
        endpoint, view, field = q["view_id"].split("/")
        if q["view_id"] not in cache:
            e, c = endpoint_ids(meta, codes, endpoint, view, exclusions)
            groups = defaultdict(list)
            for i in c:
                groups[meta[i][field]].append(i)
            cache[q["view_id"]] = (c, groups, error[c].sum(axis=(0, 2), dtype=np.int64))
        common, groups, total = cache[q["view_id"]]
        ids = groups[q["omitted"]]
        a.need(ids == q["omitted_indices"], "Entire influence group retained")
        dropped = error[ids].sum(axis=(0, 2), dtype=np.int64)
        left = total - dropped
        n = len(common) - len(ids)
        num = int(left[1:].sum() - 3 * left[0])
        near(q["omitted_errors"], dropped, "Dropped original errors")
        a.need(
            q["remaining_inputs"] == n and q["difference_numerator"] == num,
            "Exact omission numerator",
        )
        near(q["difference"], ratio(num, 9 * n), "Exact omission point")
        byview[q["view_id"]].append(q)
        rows += 1
    for name, q in influence.items():
        rr = byview[name]
        finite = [r for r in rr if r["difference"] is not None]
        base = q["base_difference"]
        a.need(
            len(rr) == q["groups"] == len(cache[name][1])
            and len({v["omitted"] for v in rr}) == len(rr),
            "All groups omitted exactly once",
        )
        near(q["minimum"], min(v["difference"] for v in finite), "Influence minimum")
        near(q["maximum"], max(v["difference"] for v in finite), "Influence maximum")
        a.need(
            q["opposite_sign_omissions"]
            == sum(v["difference"] * base < 0 for v in finite)
            and q["exact_zero_omissions"] == sum(v["difference"] == 0 for v in finite),
            "Omission sign count",
        )
        a.need(
            q["largest_effect_change"]
            == sorted(
                finite, key=lambda v: (-abs(v["difference"] - base), v["omitted"])
            )[:10],
            "Influence ranked display",
        )
    a.need(
        rows == 23899
        and len(influence) == 12
        and len(availability) == 56
        and len(bresults) == 22,
        "Complete diagnostics inventory",
    )
    reused = a.read(OUT / "reused_robustness.json")
    oldrates = a.read(p.P2 / "rates.json")
    oldquality = a.read(p.P4 / "quality_results.json")
    for name, row in reused["primary_rates"].items():
        for key, v in row.items():
            if key != "interval98_75_direction":
                a.need(v == oldrates[name][key], "Prior rate evidence preserved")
    for name, row in reused["quality"].items():
        for metric, v in row["metrics"].items():
            for key, value in v.items():
                if key != "interval95_direction":
                    a.need(
                        value == oldquality[name]["metrics"][metric].get(key),
                        "Prior quality evidence preserved",
                    )
    a.need(
        reused["error_omissions"] == a.read(p.P2 / "omissions.json")
        and reused["quality_omissions"] == a.read(p.P4 / "omissions.json"),
        "All original pool/batch omissions retained",
    )
    standard = a.read(OUT / "source_standardization.json")
    for endpoint, q in standard.items():
        _, c = endpoint_ids(meta, codes, endpoint, "main", exclusions)
        cc = Counter(meta[i]["source"] for i in c)
        near(
            q["weights"],
            {s: ratio(n, len(c)) for s, n in cc.items()},
            "Fixed source weights",
        )
        for label, v in q["eras"].items():
            for metric, value in v.items():
                near(
                    value,
                    sum(
                        q["weights"][s]
                        * rates[f"source_era/{s}/{label}/{endpoint}/main"][metric]
                        for s in cc
                    ),
                    "Descriptive source-standardized rate",
                )
    for name, digest in assembled["input_bindings"].items():
        a.need(
            a.sha(a.ROOT / name) == digest, "Original evidence changed during analysis"
        )
    p.save(
        "verified.json",
        dict(
            status="part7_runtime_missingness_rates_quality_bootstrap_and_influence_independently_verified",
            checked_utc=a.utc(),
            assembled_sha256=a.sha(OUT / "assembled.json"),
            calculated_sha256=a.sha(OUT / "calculated.json"),
            verifier_sha256=a.sha(Path(__file__)),
            reused_independent_arithmetic_source=a.rel(Path(iv.__file__)),
            reused_independent_arithmetic_sha256=a.sha(Path(iv.__file__)),
            runtime_slots=189096,
            old_receipt_result_hash_pairs=31565,
            original_feature_cells=sum(v.size for v in data.values()),
            rate_views=124,
            quality_views=72,
            defined_quality_views=48,
            new_bootstrap_sets=16,
            verified_bootstrap_scalars=scalar_draws,
            max_bootstrap_discrepancy=max_error,
            vote_bound_views=22,
            enumerated_output_vote_completions=189096,
            availability_views=56,
            group_omission_rows=rows,
            reused_rate_views=42,
            reused_quality_views=188,
            prior_pool_batch_omissions_preserved=True,
            original_evidence_unchanged=True,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_primary_families=0,
        ),
    )
    print(
        "All independent Part 7 numerical, runtime and missingness checks passed.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()
