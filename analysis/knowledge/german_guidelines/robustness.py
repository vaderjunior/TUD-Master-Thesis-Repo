"""German guidelines robustness."""

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from analysis.knowledge.german_guidelines import datasets as d

c, h, l, reader = (d.c, d.h, d.l, d.reader)
offline, require = (d.offline, d.require)
BASE = c.ROOT / "experiments/astra/german_guidelines_robustness_analysis_20261113_v1"
OLD = c.ROOT / "experiments/astra/sq2_broad_reader_review_20261104_v1"
PHASES, CONDITIONS, PAIRS = (d.PHASES, d.CONDITIONS, d.PAIRS)
SEEDS = {
    "whitespace_only": 2026111361,
    "prior_ANN_replay": 2026111362,
    "combined_declared": 2026111363,
}
QUALITY = ("three_valid_inner_votes", "no_repairs", "no_hate_ties")
METRICS = {
    "hate": ("macro_f1", "fp", "fn", "accuracy"),
    "legal": ("macro_f1", "exact_set_accuracy", "extra_labels", "missed_labels"),
}


def save(path, value):
    require(c.normal(path).is_relative_to(BASE), "Part 6 write outside new folder")
    c.save(path, h.safe(value))


def rowsave(path, values):
    require(c.normal(path).is_relative_to(BASE), "Part 6 rows outside new folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        for v in values:
            f.write(
                json.dumps(h.safe(v), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            )


def arrays(path, **values):
    require(c.normal(path).is_relative_to(BASE), "Part 6 arrays outside new folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def score(y, p, mask, names=CONDITIONS):
    y, p, mask = (np.asarray(y), np.asarray(p), np.asarray(mask, dtype=bool))
    require(
        mask.shape == (len(y),) and mask.any() and (p.shape[:2] == (len(names), 3)),
        "Invalid score population",
    )
    require(np.isin(p[:, :, mask], [0, 1]).all(), "Missing answers must not be scored")
    yy, pp = (y[mask], p[:, :, mask])
    return {
        cond: (
            l.summarize(*l.totals(yy, pp[i]), [0, 1, 2])
            if yy.ndim == 2
            else h.summarize(h.contributions(yy, pp[i]).sum(axis=-2))
        )
        for i, cond in enumerate(names)
    }


def exclude_mask(keys, common, excluded):
    require(len(set(keys)) == len(keys) and len(keys) == len(common), "Invalid exclusion roster")
    require(set(excluded) <= set(keys), "Declared key absent")
    return np.asarray(common, dtype=bool) & np.array([key not in set(excluded) for key in keys])


def quality_mask(common, n_valid, repairs, ties, kind):
    require(
        n_valid.shape == repairs.shape == ties.shape == (3, 3, len(common)),
        "Quality metadata axes differ",
    )
    if kind == "three_valid_inner_votes":
        qualified = (n_valid == 3).all(axis=(0, 1))
    elif kind == "no_repairs":
        qualified = ((n_valid == 3) & (repairs == 0)).all(axis=(0, 1))
    elif kind == "no_hate_ties":
        qualified = (~ties.astype(bool)).all(axis=(0, 1))
    else:
        raise ValueError("Unknown quality diagnostic")
    return np.asarray(common, dtype=bool) & qualified


def repeat_summary(scores, kind):
    result = {}
    for left, right in PAIRS:
        metrics = {}
        for metric in METRICS[kind]:
            a = np.array([v[metric] for v in scores[left]["repeats"]])
            b = np.array([v[metric] for v in scores[right]["repeats"]])
            delta = a - b
            metrics[metric] = {
                "difference": float(delta.mean()),
                "by_repeat": delta.tolist(),
                "leave_one_repeat_out": [float(np.delete(delta, i).mean()) for i in range(3)],
                "all_nine_cross_repeat_differences": (a[:, None] - b[None, :]).tolist(),
                "cross_repeat_range": [
                    float((a[:, None] - b[None, :]).min()),
                    float((a[:, None] - b[None, :]).max()),
                ],
            }
        result[left + "_minus_" + right] = metrics
    return result


def cache_checks():
    c.validate()
    _, rp = reader.read_verified()
    _, hp = h.verified()
    _, lp = l.verified()
    _, dp = d.verified()
    assembly = c.read(OLD / "assembly.json")
    require(
        assembly["status"] == "verified_D_reader_cache_complete", "Historical cache not verified"
    )
    for name in ["predictions.jsonl", "item_bindings.json", "reader_review.json"]:
        require(
            c.sha(OLD / name) == assembly["outputs"][c.ref(OLD / name)], "Historical cache changed"
        )
    return (rp, hp, lp, dp)


def sources():
    return [
        Path(__file__),
        c.ROOT / "analysis/knowledge/german_guidelines/verify_robustness.py",
        c.ROOT / "tests/knowledge/german_guidelines/test_robustness.py",
        Path(h.__file__),
        Path(l.__file__),
        Path(d.__file__),
        Path(reader.__file__),
    ]


def prepare():
    rp, hp, lp, dp = cache_checks()
    tests = sorted((BASE / "tests").glob("*/report.json"))
    require(tests and c.read(tests[-1])["passed"], "Passing synthetic checks required")
    require(c.read(tests[-1])["sources"] == c.bindings(sources()), "Code changed after tests")
    rosters = c.read(rp / "prompt_setup.json")["sensitivity_rosters"]
    require(
        len(rosters["prior_ANN_replay"]) == 7
        and rosters["whitespace_only"] == ["de/gahd-f4c7eaf2"],
        "Declared exclusions differ",
    )
    paths = [
        c.PLAN,
        c.PROTOCOL,
        reader.ANALYSIS_PLAN,
        reader.BASE / "complete.json",
        rp / "manifest.json",
        h.BASE / "complete.json",
        hp / "manifest.json",
        l.BASE / "complete.json",
        lp / "manifest.json",
        d.BASE / "complete.json",
        dp / "manifest.json",
        OLD / "assembly.json",
        OLD / "part1_complete.json",
        OLD / "predictions.jsonl",
        OLD / "item_bindings.json",
        c.PREP / "broad_roster.jsonl",
        tests[-1],
        *sources(),
    ]
    save(
        BASE / "analysis_contract.json",
        {
            "status": "part6_scope_before_new_sensitivity_and_historical_scores",
            "created_utc": c.a.utc(),
            "inputs": c.bindings(paths),
            "prior_exposure": "Parts 1–5 and their outcomes are known; this is a scoped analysis, not a new preregistration.",
            "declared_exclusions": rosters,
            "combined": "Also report the union of the two declared exclusions as an explicit additional diagnostic, never replacing either separate result.",
            "sensitivity_method": {
                "draws": 10000,
                "seeds": SEEDS,
                "fixed_labels": "Keep original full-roster labels after all exclusions and bootstrap draws.",
                "mask": "Original all-nine-valid mask minus the same declared keys for all conditions/repeats. Unaffected endpoints reuse the original result.",
                "uncertainty": "Whole normalized-text clusters, original full-roster source memberships. Preserve main 98.75% family roles within each sensitivity; not joint correction across exploratory sensitivities.",
            },
            "quality_diagnostics": {
                "subsets": QUALITY,
                "scope": "Additional descriptive point scores only, applied symmetrically over all nine predictions. These post-generation subsets can select easier/different texts; not a causal estimate of repairs or an alternative primary result.",
            },
            "missing": "Consolidate verified common/own-valid and exact full-roster missing bounds from Parts 2–3; no imputation or new main mask.",
            "cache_replay": "Reparse saved cached response strings with frozen parser for complete error/repair accounting; no HTTP or full raw tree audit. Compare final parsed votes and counts to the verified cache.",
            "votes": "Retain frozen tie-to-nonhate gate and independent legal labels. Check duplicates, gate normalization and at-least-two-valid inner agreement. Do not reinterpret least-supported label occurrence as agreement.",
            "repeat": "Mean of three separate scores, same-ordinal and all-nine descriptive differences, leave-one-repeat-out means. No outer vote or independent-text inflation.",
            "historical": "TEST only: original broad rag_full versus fresh mixed_2 on exact matching key/text/gold/prompt hashes, common all six baseline predictions. Descriptive point scores and disagreements only; never substitute for concurrent controls. Six deterministic raw items (minimum key hash per endpoint/repeat) check saved contract/model metadata; sample times are not full collection duration.",
            "temporal": "Use saved stage windows and per-prediction completion timestamps to check balanced nine-setting batches and overlap. Completion spread is not start-time synchrony, server execution overlap or proof of immutable weights.",
            "limits": [
                "late exposed data",
                "English general vs German legal content",
                "extra rule length and order",
                "bucket-dependent BM25 statistics",
                "small fixed rule bank",
                "one unpinned served model revision",
                "small legal population",
                "near-duplicate/context/annotation boundaries",
                "no complete five-dimension German accuracy",
            ],
            "deferred": "Part 7 resources/practical synthesis and Part 8 writer handover.",
            "actual_Qwen_calls": 0,
            "training_runs": 0,
            "D_writes": 0,
        },
    )
    print("Part 6 contract frozen; no new sensitivity scores yet.", flush=True)


def metadata(rp, out, data):
    from analysis.common import schema as isolated

    backend = isolated.adapter(c.ROOT / "config/taxonomy.yaml")
    parser = backend.bundle.parser
    cells = defaultdict(Counter)
    errors = defaultdict(Counter)
    agreement = defaultdict(Counter)
    compact = []
    event_models = Counter()
    event_finishes = Counter()
    retry = Counter()
    all_norm = 0
    for row in c.rows(rp / "predictions.jsonl"):
        phase, cond, rep = (row["phase"], row["condition"], row["repeat"])
        name = "/".join([phase, cond, str(rep)])
        counts = Counter(
            predictions=1,
            valid=int(row["result"] is not None),
            nulls=int(row["result"] is None),
            partial_predictions=int(row["n_valid"] in (1, 2)),
            repaired_predictions=int(row["repairs"] > 0),
            repair_calls=row["repairs"],
            invalid_votes=row["parse_failures"],
            hate_ties=int(row["uncertain"]),
            gate_normalized_votes=row["gate_normalised"],
            gate_normalized_predictions=int(row["gate_normalised"] > 0),
            free_normalisations=row["normalisations"],
            duplicate_label_predictions=int(bool(row["reader_flags"]["duplicate_labels"])),
            duplicate_sensitive_predictions=int(
                bool(row["reader_flags"]["duplicate_sensitive_fields"])
            ),
            legal_with_nonhate=int(
                row["result"] is not None
                and (not row["result"]["hate"])
                and bool(row["result"]["legal"])
            ),
        )
        counts["valid_votes_" + str(row["n_valid"])] = 1
        initial_valid = 0
        normal = 0
        gate = 0
        for vi, chain in enumerate(row["raw_runs"]):
            last = None
            for turn, raw in enumerate(chain):
                parsed, n, error = parser.parse_response(raw)
                normal += n
                counts["raw_texts"] += 1
                require((parsed is None) == bool(error), "Parser returned inconsistent validity")
                require(
                    parsed is None or turn == len(chain) - 1,
                    "Saved chain continues after valid response",
                )
                if turn == 0:
                    initial_valid += int(parsed is not None)
                if error:
                    errors[name + "/all"][error] += 1
                    if turn == 0:
                        errors[name + "/initial"][error] += 1
                    if turn == len(chain) - 1:
                        errors[name + "/final"][error] += 1
                last = parsed
            require(
                (None if last is None else last.model_dump()) == row["run_predictions"][vi],
                "Cached repair replay changed final vote",
            )
            gate += int(last is not None and last.gate_normalised)
            counts["inner_success_turn_" + ("failed" if last is None else str(len(chain) - 1))] += 1
            counts["initially_invalid_votes"] += int(len(chain) > 1)
            counts["repaired_valid_votes"] += int(len(chain) > 1 and last is not None)
        require(
            normal == row["normalisations"] and gate == row["gate_normalised"],
            "Parser normalization counts differ",
        )
        counts["all_initial_votes_invalid"] = int(initial_valid == 0)
        counts["valid_after_all_initial_invalid"] = int(
            initial_valid == 0 and row["result"] is not None
        )
        all_norm += normal
        valid = [v for v in row["run_predictions"] if v is not None]
        for field in ("hate", "legal"):
            vals = [tuple(sorted(set(v[field]))) if field == "legal" else v[field] for v in valid]
            if len(vals) >= 2:
                agreement[name + "/" + field].update(
                    informative_predictions=1,
                    unanimous_predictions=int(len(set(vals)) == 1),
                    agreeing_pairs=sum(
                        (vals[i] == vals[j] for i in range(len(vals)) for j in range(i))
                    ),
                    pairs=len(vals) * (len(vals) - 1) // 2,
                )
        require(
            not row["uncertain"] or (row["n_valid"] == 2 and row["result"]["hate"] is False),
            "Gate tie semantics differ",
        )
        for event in row["events"]:
            if event["action"] == "ok":
                event_models[str(event["model_response_metadata"]["reported_model"])] += 1
                event_finishes[str(event["model_response_metadata"]["finish_reason"])] += 1
            else:
                retry[event["response_class"]] += 1
        cells[name].update(counts)
        i = data[phase]["index"][row["key"]]
        ci = CONDITIONS.index(cond)
        for target, value in [
            ("n_valid", row["n_valid"]),
            ("repairs", row["repairs"]),
            ("ties", row["uncertain"]),
        ]:
            data[phase][target][ci, rep, i] = value
        compact.append(
            {
                "key": row["key"],
                "phase": phase,
                "condition": cond,
                "repeat": rep,
                "stage": row["stage"],
                "messages_sha256": row["messages_sha256"],
                "created_utc": row["evidence"]["created_utc"],
                "evidence": row["evidence"],
                "n_valid": row["n_valid"],
                "repairs": row["repairs"],
                "tie": row["uncertain"],
                "counts": dict(counts),
            }
        )
        if len(compact) % 5000 == 0:
            print(f"Cached parser/metadata review: {len(compact)}/33093 predictions", flush=True)
    require(len(compact) == 33093, "Incomplete metadata")
    total = Counter()
    for v in cells.values():
        total.update(v)
    require(
        total["raw_texts"] == 104220
        and total["nulls"] == 12
        and (total["repair_calls"] == 4941)
        and (total["invalid_votes"] == 579),
        "Reader aggregate differs",
    )
    rowsave(out / "prediction_metadata.jsonl", compact)
    save(
        out / "parser_metadata.json",
        {
            "cells": dict(cells),
            "errors": dict(errors),
            "inner_agreement": dict(agreement),
            "totals": dict(total),
            "reported_models": dict(event_models),
            "finish_reasons": dict(event_finishes),
            "retry_classes": dict(retry),
            "interpretation": "Cache strings replayed, not new responses. Repairs recover format validity, not necessarily correctness; no causal repair accuracy claim.",
        },
    )
    return compact


def timeline(rp, compact, out):
    saved = c.read(rp / "execution_timeline.json")
    by_stage = defaultdict(list)
    by_key = defaultdict(list)
    for row in compact:
        by_stage[row["stage"]].append(row)
        by_key[row["key"]].append(row)
    stages = []
    for record in saved:
        rows = by_stage[record["stage"]]
        times = [datetime.fromisoformat(x["created_utc"]).timestamp() for x in rows]
        first = min(
            (
                datetime.fromisoformat(x["start"]["started_utc"]).timestamp()
                for x in record["attempts"]
            )
        )
        last = max(
            (datetime.fromisoformat(x["finished_utc"]).timestamp() for x in record["attempts"])
        )
        require(first <= min(times) <= max(times) <= last, "Receipt time outside saved execution")
        counts = Counter(((r["condition"], r["repeat"]) for r in rows))
        require(
            len(counts) == 9 and len(set(counts.values())) == 1, "Unbalanced nine-setting batch"
        )
        require(
            all(
                (
                    x["pinned_model_available"]
                    and x["stop"] is None
                    and (x["start"]["workers"] == 128)
                    and (not x["summary"]["errors"])
                    and (not x["summary"]["stopped"])
                    and (x["summary"]["unsubmitted"] == 0)
                    for x in record["attempts"]
                )
            ),
            "Unexpected execution interruption",
        )
        by_condition = {
            cond: {
                "first_completion": min((x["created_utc"] for x in rows if x["condition"] == cond)),
                "last_completion": max((x["created_utc"] for x in rows if x["condition"] == cond)),
            }
            for cond in CONDITIONS
        }
        stages.append(
            {
                "stage": record["stage"],
                "phase": record["population"],
                "inputs": len(rows) // 9,
                "rows_per_setting": next(iter(counts.values())),
                "start_utc": min((x["start"]["started_utc"] for x in record["attempts"])),
                "finish_utc": max((x["finished_utc"] for x in record["attempts"])),
                "by_condition": by_condition,
                "condition_completion_ranges_overlap": max(
                    (v["first_completion"] for v in by_condition.values())
                )
                <= min((v["last_completion"] for v in by_condition.values())),
            }
        )
    spreads = []
    for key, rows in by_key.items():
        require(
            len(rows) == 9 and len({r["stage"] for r in rows}) == 1,
            "Input split across collection batches",
        )
        t = [datetime.fromisoformat(r["created_utc"]).timestamp() for r in rows]
        spreads.append(
            {"key": key, "phase": rows[0]["phase"], "completion_spread_seconds": max(t) - min(t)}
        )
    save(
        out / "timeline.json",
        {
            "stages": stages,
            "input_completion_spreads": spreads,
            "spread_quantiles_seconds": dict(
                zip(
                    ["min", "median", "p95", "max"],
                    np.quantile(
                        [r["completion_spread_seconds"] for r in spreads], [0, 0.5, 0.95, 1]
                    ),
                )
            ),
            "limits": "Completion timing is not simultaneous start/execution timing. Balanced batches reduce whole-condition/time separation but cannot rule out time-varying service behaviour.",
        },
    )


def historical(data, compact, out):
    fresh = {
        (r["key"], r["repeat"]): r
        for r in compact
        if r["condition"] == "mixed_2" and r["phase"] in ("de_test", "de_legal_test")
    }
    roster = {r["key"]: r for r in c.rows(c.BASE / "roster.jsonl")}
    oldroster = {r["key"]: r for r in c.rows(c.PREP / "broad_roster.jsonl") if r["key"] in roster}
    chosen = []
    coverage = Counter()
    by_phase = defaultdict(list)
    for row in c.rows(OLD / "predictions.jsonl"):
        if row["condition"] != "rag_full" or (row["key"], row["repeat"]) not in fresh:
            continue
        f = fresh[row["key"], row["repeat"]]
        key = row["key"]
        require(
            row["phase"] == f["phase"]
            and row["messages_sha256"]
            == f["messages_sha256"]
            == roster[key]["original_messages_sha256"],
            "Historical baseline prompt/key alignment differs",
        )
        require(oldroster[key]["text"] == roster[key]["text"], "Historical query text differs")
        field = "legal" if "legal" in row["phase"] else "gate"
        require(oldroster[key].get(field) == roster[key].get(field), "Historical gold differs")
        chosen.append(row)
        by_phase[row["phase"]].append(row)
        coverage[row["phase"]] += 1
    require(
        dict(coverage) == {"de_test": 9525, "de_legal_test": 531},
        "Historical matched roster incomplete",
    )
    rowsave(out / "historical_matched_rows.jsonl", chosen)
    outputs = {}
    for phase, rows in by_phase.items():
        v = data[phase]
        p = np.full_like(v["p"][0], -9)
        for row in rows:
            p[row["repeat"], v["index"][row["key"]]] = (
                -1
                if row["result"] is None
                else (
                    l.bits(row["result"]["legal"])
                    if "legal" in phase
                    else int(row["result"]["hate"])
                )
            )
        require(np.isin(p, [-1, 0, 1]).all(), "Historical matrix incomplete")
        paired = np.stack([p, v["p"][0]])
        mask = (paired >= 0).all(axis=tuple((i for i in range(paired.ndim) if i != 2)))
        scores = score(v["y"], paired, mask, ("historical_full", "fresh_shared"))
        left, right = (paired[1][:, mask], paired[0][:, mask])
        disagreements = left[:, None] != right[None, :]
        if "legal" in phase:
            disagreements = disagreements.any(axis=-1)
        outputs[phase] = {
            "full_inputs": len(v["y"]),
            "common_six_valid_inputs": int(mask.sum()),
            "keys": [k for k, on in zip(v["index"], mask) if on],
            "scores": scores,
            "fresh_minus_historical": {
                m: scores["fresh_shared"]["mean"][m] - scores["historical_full"]["mean"][m]
                for m in METRICS["legal" if "legal" in phase else "hate"]
            },
            "all_nine_cross_period_disagreements": disagreements.sum(axis=-1).tolist(),
            "mean_disagreement_fraction": float(disagreements.mean()),
            "limits": "Same inputs/gold/initial prompt; different collection period. This comparison cannot identify stochastic, server, revision or load causes and is not a guideline treatment contrast.",
        }
        arrays(
            out / "historical" / phase / "arrays.npz",
            gold=v["y"],
            predictions=paired,
            common_mask=mask,
        )
    bindings = c.read(OLD / "item_bindings.json")
    samples = []
    for phase in ("de_test", "de_legal_test"):
        for rep in range(3):
            row = min(
                (r for r in by_phase[phase] if r["repeat"] == rep),
                key=lambda r: hashlib.sha256(("part6-history|" + r["key"]).encode()).hexdigest(),
            )
            p = Path(row["result_path"])
            cp = p.with_name("complete.json")
            raw = reader.checked_json(p, bindings[c.ref(p)])
            receipt = reader.checked_json(cp, bindings[c.ref(cp)])
            require(
                raw["result"] == row["result"] and raw["messages_sha256"] == row["messages_sha256"],
                "Historical raw sample differs",
            )
            jp = p.with_name("job.json")
            digest = next((v for k, v in receipt["hashes"].items() if k.endswith("/job.json")))
            job = reader.checked_json(jp, digest)
            samples.append(
                {
                    "phase": phase,
                    "repeat": rep,
                    "key": row["key"],
                    "created_utc": receipt["created_utc"],
                    "recorded_contract": job["contract"],
                    "same_request_contract_as_supplement": job["contract"] == c.a.CONTRACT,
                    "requested_model": raw["requested_model"],
                    "active_model": raw["active_model"],
                    "reported_models": dict(
                        Counter(
                            (
                                str(e["model_response_metadata"]["reported_model"])
                                for e in raw["events"]
                                if e["action"] == "ok"
                            )
                        )
                    ),
                    "bindings": c.bindings([p, cp, jp]),
                }
            )
    save(
        out / "historical_comparison.json",
        {
            "populations": outputs,
            "raw_samples": samples,
            "DEV_not_compared": "Only the exact original broader TEST prompt/input pairs are used. No inferred DEV equivalence.",
        },
    )


def run():
    contract = c.read(BASE / "analysis_contract.json")
    c.verify(contract["inputs"])
    rp, hp, lp, dp = cache_checks()
    endpoints = c.read(rp / "endpoint_setup.json")
    out = BASE / "runs" / f"run_{time.time_ns()}"
    out.mkdir(parents=True)
    save(
        out / "start.json",
        {"contract_sha256": c.sha(BASE / "analysis_contract.json"), "created_utc": c.a.utc()},
    )
    data = {}
    consolidated = {}
    for phase in PHASES:
        prior = lp if "legal" in phase else hp
        with np.load(prior / phase / "arrays.npz", allow_pickle=False) as f:
            y, p, mask = (f["gold"], f["predictions"], f["common_mask"])
        ep = endpoints[phase]
        old = c.read(prior / phase / "results.json")
        data[phase] = {
            "y": y,
            "p": p,
            "mask": mask,
            "ep": ep,
            "index": {k: i for i, k in enumerate(ep["keys"])},
            "n_valid": np.full((3, 3, len(y)), -1, np.int8),
            "repairs": np.full((3, 3, len(y)), -1, np.int16),
            "ties": np.zeros((3, 3, len(y)), bool),
        }
        consolidated[phase] = {
            k: old[k]
            for k in [
                "full_inputs",
                "common_inputs",
                "full_gold_support",
                "common_gold_support",
                "common",
                "own_valid",
                "full_roster_bounds",
                "contrasts",
            ]
        }
        consolidated[phase]["repeat_diagnostics"] = repeat_summary(
            old["common"], "legal" if "legal" in phase else "hate"
        )
        consolidated[phase]["prior_result"] = c.ref(prior / phase / "results.json")
    save(out / "original_coverage_bounds_repeats.json", consolidated)
    compact = metadata(rp, out, data)
    timeline(rp, compact, out)
    quality = {}
    for phase, v in data.items():
        require((v["n_valid"] >= 0).all() and (v["repairs"] >= 0).all(), "Missing metadata slot")
        require(
            np.array_equal((v["n_valid"] > 0).all(axis=(0, 1)), v["mask"]),
            "Original common mask differs",
        )
        arrays(
            out / phase / "metadata.npz", n_valid=v["n_valid"], repairs=v["repairs"], ties=v["ties"]
        )
        quality[phase] = {}
        for kind in QUALITY:
            selected = quality_mask(v["mask"], v["n_valid"], v["repairs"], v["ties"], kind)
            scores = score(v["y"], v["p"], selected)
            quality[phase][kind] = {
                "n": int(selected.sum()),
                "excluded_from_common": [
                    k
                    for k, on, orig in zip(v["ep"]["keys"], selected, v["mask"])
                    if orig and (not on)
                ],
                "scores": scores,
                "contrasts": repeat_summary(scores, "legal" if "legal" in phase else "hate"),
                "source_counts": dict(
                    Counter((s for s, on in zip(v["ep"]["sources"], selected) if on))
                ),
                "gold_support": v["y"][selected].sum(axis=0).tolist(),
                "role": "Additional descriptive, post-generation selected subset; not a causal or confirmatory result.",
            }
    save(out / "quality_subset_scores.json", quality)
    sensitivities = {}
    rosters = contract["declared_exclusions"]
    rosters["combined_declared"] = rosters["whitespace_only"] + rosters["prior_ANN_replay"]
    v = data["de_test"]
    ep = v["ep"]
    for name, excluded in rosters.items():
        mask = exclude_mask(ep["keys"], v["mask"], excluded)
        scores = score(v["y"], v["p"], mask)
        selected_clusters = [cl for cl, on in zip(ep["clusters"], mask) if on]
        boot, design = h.bootstrap(
            v["y"][mask],
            v["p"][:, :, mask],
            selected_clusters,
            ep["source_memberships_by_cluster"],
            SEEDS[name],
        )
        bm = {k: z.mean(axis=2) for k, z in h.metrics(boot).items()}
        contrasts = []
        for left, right in PAIRS:
            li, ri = (CONDITIONS.index(left), CONDITIONS.index(right))
            metrics = {}
            for m in METRICS["hate"]:
                metrics[m] = h.interval(
                    bm[m][:, li] - bm[m][:, ri],
                    4 if right == "mixed_2" and m == "macro_f1" else None,
                )
                metrics[m].update(difference=scores[left]["mean"][m] - scores[right]["mean"][m])
            contrasts.append({"left": left, "right": right, "metrics": metrics})
        sensitivities[name] = {
            "excluded": excluded,
            "removed_from_common": len(v["y"][v["mask"]]) - int(mask.sum()),
            "common_n": int(mask.sum()),
            "remaining_keys": [k for k, on in zip(ep["keys"], mask) if on],
            "scores": scores,
            "contrasts": contrasts,
            "fixed_labels": ep["macro_labels"],
        }
        save(out / "exclusions" / name / "bootstrap_design.json", design)
        arrays(out / "exclusions" / name / "bootstrap.npz", counts=boot)
        print(name + ": 10000 paired source-stratified exclusion draws complete.", flush=True)
    save(
        out / "exclusion_scores.json",
        {
            "de_test": sensitivities,
            "unaffected_populations": {
                p: "No declared exclusion keys in this population; original scores and intervals retained without recomputation."
                for p in PHASES
                if p != "de_test"
            },
        },
    )
    rowsave(
        out / "excluded_inputs.jsonl",
        [r for r in c.rows(c.BASE / "roster.jsonl") if r["key"] in rosters["combined_declared"]],
    )
    historical(data, compact, out)
    c.verify(contract["inputs"])
    c.validate()
    save(
        out / "manifest.json",
        {
            "status": "part6_pending_independent_check",
            "run": c.ref(out),
            "created_utc": c.a.utc(),
            "outputs": c.bindings((p for p in out.rglob("*") if p.is_file())),
            "sources": c.bindings(sources()),
            "actual_Qwen_calls": 0,
            "D_writes": 0,
        },
    )
    print(json.dumps({"manifest": c.ref(out / "manifest.json")}), flush=True)


def verified():
    done = c.read(BASE / "complete.json")
    require(done["status"] == "part6_numerically_verified", "Part 6 independent check missing")
    c.verify(done["bindings"])
    manifest = c.read(c.ROOT / done["manifest"])
    c.verify(manifest["outputs"])
    return (manifest, c.ROOT / manifest["run"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "run"])
    args = p.parse_args()
    with offline():
        prepare() if args.mode == "prepare" else run()
