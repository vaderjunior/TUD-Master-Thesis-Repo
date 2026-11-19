"""German guidelines verify robustness."""

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from analysis.knowledge.german_guidelines import robustness as r
from analysis.knowledge.german_guidelines import verify_hate as hc
from analysis.knowledge.german_guidelines import verify_legal as lc
from analysis.knowledge.german_guidelines.verify_context import same

c = r.c


def point_check(saved, y, p, checks):
    legal = np.asarray(y).ndim == 2
    expected = [lc.reference(y, row, [0, 1, 2])[0] if legal else hc.rates(y, row) for row in p]
    for i, v in enumerate(expected):
        same(saved["repeats"][i], r.h.safe(v))
    for name in expected[0]:
        values = [v[name] for v in expected]
        if any((v is None for v in values)):
            mean = None
            sd = None
            ran = [None, None]
        else:
            z = np.array(values)
            mean = r.h.safe(z.mean())
            sd = r.h.safe(z.std(ddof=1))
            ran = r.h.safe([z.min(), z.max()])
        same(saved["mean"][name], mean)
        same(saved["sample_sd"][name], sd)
        same(saved["range"][name], ran)
    if legal:
        for j, label in enumerate(r.l.LABELS):
            values = [lc.reference(y, row, [0, 1, 2])[1] for row in p]
            for i, val in enumerate(values):
                same(
                    saved["per_label"][label]["repeats"][i],
                    {k: r.h.safe(z[j]) for k, z in val.items()},
                )
    checks["condition_summaries"] += 1


def direct_draws(y, p, clusters, memberships, seed, saved, design):
    unique = sorted(set(clusters))
    strata = defaultdict(list)
    for i, cl in enumerate(unique):
        strata[tuple(sorted(memberships[cl]))].append(i)
    require = r.require
    require(
        design["clusters"] == unique
        and design["strata"]
        == [{"sources": list(k), "cluster_indices": v} for k, v in sorted(strata.items())],
        "Independent strata differ",
    )
    grouped = np.zeros((9, len(unique), 4), np.int64)
    lookup = {cl: i for i, cl in enumerate(unique)}
    for j, cl in enumerate(clusters):
        for setting, answer in enumerate(p.reshape(9, len(y))[:, j]):
            grouped[setting, lookup[cl], 2 * int(y[j]) + int(answer)] += 1
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    actual = saved.reshape(-1, 9, 4)
    for start in range(0, len(actual), 64):
        size = min(64, len(actual) - start)
        picks = np.concatenate(
            [rng.choice(v, (size, len(v)), replace=True) for _, v in sorted(strata.items())], axis=1
        )
        digest.update(picks.astype("<i4").tobytes())
        direct = np.stack([grouped[:, row].sum(axis=1) for row in picks])
        require(
            np.array_equal(direct, actual[start : start + size]), "Independent draw count differs"
        )
    require(
        digest.hexdigest() == design["pick_stream_sha256"], "Independent resampling stream differs"
    )


def verify(path):
    with r.offline():
        manifest = c.read(path)
        r.require(
            manifest["status"] == "part6_pending_independent_check", "Incomplete Part 6 build"
        )
        c.verify(manifest["outputs"])
        c.verify(manifest["sources"])
        contract = c.read(r.BASE / "analysis_contract.json")
        c.verify(contract["inputs"])
        out = c.ROOT / manifest["run"]
        rp, hp, lp, _ = r.cache_checks()
        eps = c.read(rp / "endpoint_setup.json")
        checks = Counter()
        compact = list(c.rows(out / "prediction_metadata.jsonl"))
        byid = {(x["key"], x["condition"], x["repeat"]): x for x in compact}
        r.require(len(byid) == len(compact) == 33093, "Metadata identities differ")
        meta = c.read(out / "parser_metadata.json")
        expected_cells = defaultdict(Counter)
        models = Counter()
        finishes = Counter()
        retry = Counter()
        inner = defaultdict(Counter)
        fresh = {}
        for row in c.rows(rp / "predictions.jsonl"):
            ident = (row["key"], row["condition"], row["repeat"])
            saved = byid[ident]
            name = "/".join([row["phase"], row["condition"], str(row["repeat"])])
            same(
                {
                    k: saved[k]
                    for k in [
                        "key",
                        "condition",
                        "repeat",
                        "phase",
                        "stage",
                        "messages_sha256",
                        "n_valid",
                        "repairs",
                    ]
                },
                {
                    k: row[k]
                    for k in [
                        "key",
                        "condition",
                        "repeat",
                        "phase",
                        "stage",
                        "messages_sha256",
                        "n_valid",
                        "repairs",
                    ]
                },
            )
            same(saved["created_utc"], row["evidence"]["created_utc"])
            same(saved["tie"], row["uncertain"])
            same(saved["evidence"], row["evidence"])
            t = Counter(
                predictions=1,
                valid=int(row["result"] is not None),
                nulls=int(row["result"] is None),
                partial_predictions=int(0 < row["n_valid"] < 3),
                repaired_predictions=int(row["repairs"] > 0),
                repair_calls=row["repairs"],
                invalid_votes=3 - row["n_valid"],
                hate_ties=int(row["uncertain"]),
                gate_normalized_votes=row["gate_normalised"],
                gate_normalized_predictions=int(row["gate_normalised"] > 0),
                free_normalisations=row["normalisations"],
                duplicate_label_predictions=int(
                    any(
                        (
                            len(v[f]) != len(set(v[f]))
                            for v in row["run_predictions"]
                            if v
                            for f in r.reader.LIST_FIELDS
                        )
                    )
                ),
                duplicate_sensitive_predictions=int(
                    bool(row["reader_flags"]["duplicate_sensitive_fields"])
                ),
                legal_with_nonhate=int(
                    row["result"] is not None
                    and (not row["result"]["hate"])
                    and bool(row["result"]["legal"])
                ),
            )
            t["valid_votes_" + str(row["n_valid"])] = 1
            t["raw_texts"] = sum((len(v) for v in row["raw_runs"]))
            initial_valid = sum((len(chain) == 1 for chain in row["raw_runs"]))
            for vote, chain in zip(row["run_predictions"], row["raw_runs"]):
                t["inner_success_turn_" + ("failed" if vote is None else str(len(chain) - 1))] += 1
                t["initially_invalid_votes"] += len(chain) > 1
                t["repaired_valid_votes"] += vote is not None and len(chain) > 1
            t["all_initial_votes_invalid"] = int(initial_valid == 0)
            t["valid_after_all_initial_invalid"] = int(
                initial_valid == 0 and row["result"] is not None
            )
            same(saved["counts"], dict(t))
            expected_cells[name].update(t)
            for field in ["hate", "legal"]:
                values = [
                    json.dumps(sorted(set(v[field]))) if field == "legal" else str(v[field])
                    for v in row["run_predictions"]
                    if v is not None
                ]
                if len(values) >= 2:
                    inner[name + "/" + field].update(
                        informative_predictions=1,
                        unanimous_predictions=int(len(set(values)) == 1),
                        agreeing_pairs=sum((n * (n - 1) // 2 for n in Counter(values).values())),
                        pairs=len(values) * (len(values) - 1) // 2,
                    )
            for ev in row["events"]:
                if ev["action"] == "ok":
                    models[str(ev["model_response_metadata"]["reported_model"])] += 1
                    finishes[str(ev["model_response_metadata"]["finish_reason"])] += 1
                else:
                    retry[ev["response_class"]] += 1
            fresh[ident] = row["result"]
            checks["cached_predictions"] += 1
        same(meta["cells"], dict(expected_cells))
        same(meta["inner_agreement"], dict(inner))
        same(meta["reported_models"], dict(models))
        same(meta["finish_reasons"], dict(finishes))
        same(meta["retry_classes"], dict(retry))
        total = Counter()
        for name, cell in expected_cells.items():
            total.update(cell)
            for kind, expected in [
                ("initial", cell["initially_invalid_votes"]),
                ("final", cell["invalid_votes"]),
                ("all", cell["repair_calls"] + cell["invalid_votes"]),
            ]:
                same(sum(meta["errors"].get(name + "/" + kind, {}).values()), expected)
        same(meta["totals"], dict(total))
        allqualities = c.read(out / "quality_subset_scores.json")
        consolidated = c.read(out / "original_coverage_bounds_repeats.json")
        matrices = {}
        for phase in r.PHASES:
            prior = lp if "legal" in phase else hp
            ep = eps[phase]
            with np.load(prior / phase / "arrays.npz", allow_pickle=False) as f:
                y, p, common = (f["gold"], f["predictions"], f["common_mask"])
            matrices[phase] = (y, p, common)
            with np.load(out / phase / "metadata.npz", allow_pickle=False) as f:
                nv, repairs, ties = (f["n_valid"], f["repairs"], f["ties"])
            for i, key in enumerate(ep["keys"]):
                for ci, cond in enumerate(r.CONDITIONS):
                    for repeat in range(3):
                        value = byid[key, cond, repeat]
                        same(
                            [nv[ci, repeat, i], repairs[ci, repeat, i], bool(ties[ci, repeat, i])],
                            [value["n_valid"], value["repairs"], value["tie"]],
                        )
            original = c.read(prior / phase / "results.json")
            for field in [
                "full_inputs",
                "common_inputs",
                "full_gold_support",
                "common_gold_support",
                "common",
                "own_valid",
                "full_roster_bounds",
                "contrasts",
            ]:
                same(consolidated[phase][field], original[field])
            for pair, metrics in consolidated[phase]["repeat_diagnostics"].items():
                left, right = pair.split("_minus_")
                for name, v in metrics.items():
                    a = np.array([z[name] for z in original["common"][left]["repeats"]])
                    b = np.array([z[name] for z in original["common"][right]["repeats"]])
                    z = a - b
                    same(
                        v,
                        {
                            "difference": z.mean(),
                            "by_repeat": z,
                            "leave_one_repeat_out": [(z.sum() - z[i]) / 2 for i in range(3)],
                            "all_nine_cross_repeat_differences": a[:, None] - b[None, :],
                            "cross_repeat_range": [
                                (a[:, None] - b[None, :]).min(),
                                (a[:, None] - b[None, :]).max(),
                            ],
                        },
                    )
            for kind, q in allqualities[phase].items():
                good = []
                for i, key in enumerate(ep["keys"]):
                    records = [byid[key, co, rr] for co in r.CONDITIONS for rr in range(3)]
                    ok = (
                        all((x["n_valid"] == 3 for x in records))
                        if kind == "three_valid_inner_votes"
                        else (
                            all((x["n_valid"] == 3 and x["repairs"] == 0 for x in records))
                            if kind == "no_repairs"
                            else all((not x["tie"] for x in records))
                        )
                    )
                    good.append(bool(common[i] and ok))
                mask = np.array(good)
                same(q["n"], int(mask.sum()))
                same(
                    q["excluded_from_common"],
                    [key for key, on, orig in zip(ep["keys"], mask, common) if orig and (not on)],
                )
                for ci, co in enumerate(r.CONDITIONS):
                    point_check(q["scores"][co], y[mask], p[ci][:, mask], checks)
                checks["quality_subsets"] += 1
        exc = c.read(out / "exclusion_scores.json")["de_test"]
        ep = eps["de_test"]
        y, p, common = matrices["de_test"]
        declared = contract["declared_exclusions"]
        declared["combined_declared"] = declared["whitespace_only"] + declared["prior_ANN_replay"]
        for name, v in exc.items():
            same(v["excluded"], declared[name])
            mask = np.array(
                [bool(ok and key not in declared[name]) for key, ok in zip(ep["keys"], common)]
            )
            same(v["remaining_keys"], [key for key, on in zip(ep["keys"], mask) if on])
            same(v["fixed_labels"], ["not_hate", "hate"])
            same(v["common_n"], mask.sum())
            for ci, co in enumerate(r.CONDITIONS):
                point_check(v["scores"][co], y[mask], p[ci][:, mask], checks)
            design = c.read(out / "exclusions" / name / "bootstrap_design.json")
            with np.load(out / "exclusions" / name / "bootstrap.npz", allow_pickle=False) as f:
                z = f["counts"]
            direct_draws(
                y[mask],
                p[:, :, mask],
                [cl for cl, on in zip(ep["clusters"], mask) if on],
                ep["source_memberships_by_cluster"],
                r.SEEDS[name],
                z,
                design,
            )
            bm = hc.bootstrap_metrics(z)
            for pair in v["contrasts"]:
                li, ri = [r.CONDITIONS.index(pair[k]) for k in ["left", "right"]]
                for metric, ci in pair["metrics"].items():
                    hc.check_interval(
                        ci,
                        bm[metric][:, li] - bm[metric][:, ri],
                        4 if pair["right"] == "mixed_2" and metric == "macro_f1" else None,
                    )
                    same(
                        ci["difference"],
                        v["scores"][pair["left"]]["mean"][metric]
                        - v["scores"][pair["right"]]["mean"][metric],
                    )
                    checks["intervals"] += 1
            checks["independent_bootstrap_draws"] += len(z)
            print(name + ": all draws and fixed-label contrasts independently checked.", flush=True)
        hist = c.read(out / "historical_comparison.json")
        cachedhist = list(c.rows(out / "historical_matched_rows.jsonl"))
        histbyid = {(x["key"], x["repeat"]): x for x in cachedhist}
        found = 0
        for old in c.rows(r.OLD / "predictions.jsonl"):
            if old["condition"] == "rag_full" and (old["key"], old["repeat"]) in histbyid:
                same(histbyid[old["key"], old["repeat"]], old)
                found += 1
        same(found, 10056)
        for phase, v in hist["populations"].items():
            ep = eps[phase]
            with np.load(out / "historical" / phase / "arrays.npz", allow_pickle=False) as f:
                y, p, mask = (f["gold"], f["predictions"], f["common_mask"])
            for i, key in enumerate(ep["keys"]):
                for rep in range(3):
                    for ci, record in enumerate(
                        [histbyid[key, rep]["result"], fresh[key, "mixed_2", rep]]
                    ):
                        expected = (
                            -1
                            if record is None
                            else (
                                [int(label in record["legal"]) for label in r.l.LABELS]
                                if "legal" in phase
                                else int(record["hate"])
                            )
                        )
                        if record is None and "legal" in phase:
                            expected = [-1] * 3
                        same(p[ci, rep, i].tolist(), expected)
            expected = (p >= 0).all(axis=(0, 1, 3) if y.ndim == 2 else (0, 1))
            same(mask, expected)
            same(v["common_six_valid_inputs"], mask.sum())
            for ci, co in enumerate(["historical_full", "fresh_shared"]):
                point_check(v["scores"][co], y[mask], p[ci][:, mask], checks)
            for m, delta in v["fresh_minus_historical"].items():
                same(
                    delta,
                    v["scores"]["fresh_shared"]["mean"][m]
                    - v["scores"]["historical_full"]["mean"][m],
                )
            checks["historical_populations"] += 1
        for sample in hist["raw_samples"]:
            c.verify(sample["bindings"])
        timeline = c.read(out / "timeline.json")
        stages = {x["stage"]: x for x in timeline["stages"]}
        for stage in stages.values():
            rows = [x for x in compact if x["stage"] == stage["stage"]]
            same(len(rows), 9 * stage["inputs"])
            same(
                Counter(((x["condition"], x["repeat"]) for x in rows)),
                {(co, rr): stage["inputs"] for co in r.CONDITIONS for rr in range(3)},
            )
        same(len(stages), 27)
        same(len(timeline["input_completion_spreads"]), 3677)
        c.verify(contract["inputs"])
        c.verify(manifest["outputs"])
        r.save(
            out / "independent_check.json",
            {
                "status": "part6_independent_checks_pass",
                "checks": dict(checks),
                "actual_Qwen_calls": 0,
                "D_writes": 0,
            },
        )
        r.save(
            r.BASE / "complete.json",
            {
                "status": "part6_numerically_verified",
                "manifest": c.ref(path),
                "created_utc": c.a.utc(),
                "bindings": c.bindings(
                    [
                        path,
                        out / "independent_check.json",
                        r.BASE / "analysis_contract.json",
                        *r.sources(),
                    ]
                ),
                "actual_Qwen_calls": 0,
            },
        )
        print(
            json.dumps({"status": "part6_independent_checks_pass", "checks": dict(checks)}),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    verify(Path(p.parse_args().manifest))
