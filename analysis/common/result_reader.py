"""Read and score saved later-study predictions."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from analysis.common import study as c
from analysis.common import scoring as original

PRIMARY_BROAD = {
    ("larger_dev", "target_group"),
    ("larger_dev", "severity"),
    ("en_test", "hate"),
    ("de_legal_test", "legal"),
}
MATCHED_DIMS = {
    "en_dev_eval_main": "hate",
    "en_dev_eval_sq3_types": "hate_type",
    "de_legal_dev_eval": "legal",
}


def contract():
    return {
        "minimum_support": 10,
        "paired_mask": "Intersection of valid predictions across every compared fixed condition/repeat; own-valid scores and full-roster failure bounds accompany contrasts.",
        "labels": "Freeze labels from full-roster support, never from a bootstrap sample or valid-only subset. Missing annotations are excluded, observed empty lists retained. Gold-hate gate for fine heads, legal independent.",
        "replicates": "Three fixed decoder repeats; three fitted encoder seeds; selection/permutation variants separate from decoder repeats. Difference of means of individual F1 scores, never pool their predictions as extra observations.",
        "resampling": "10000 paired draws of whole normalized-text clusters; HateCheck uses connected template/reference/text components. Same cluster weights for every compared setting.",
        "sensitivity": "Natural data source-stratified cluster draws. A cluster spanning sources uses the sorted tuple of its source memberships as a stratum, so a cluster is never split. HateCheck also gets item-level sensitivity. German TEST additionally reports exclusion of its prospectively flagged whitespace-only text.",
        "primary_families": {
            "broad": {"size": 4, "confidence": 0.9875},
            "matched": {"size": 6, "confidence": 1 - 0.05 / 6},
        },
        "limits": "Exposed data, fixed models/pools/repeats, conditional resampling intervals. No untouched-test, universal-winning-system or causal-mechanism claim.",
    }


def endpoint(records, dimension, taxonomy):
    labels = (
        ["not_hate", "hate"]
        if dimension == "hate"
        else list(taxonomy["dimensions"][dimension]["labels"])
    )
    ids = []
    support = Counter()
    for row in records:
        value = original.truth(row, dimension)
        if value is None:
            continue
        ids.append(str(row["id"]))
        support.update(
            ["hate" if value else "not_hate"]
            if dimension == "hate"
            else [value] if dimension == "severity" else value
        )
    c.require(set(support) <= set(labels), "Unknown gold label")
    fixed = [lab for lab in labels if support[lab] >= 10]
    return {
        "dimension": dimension,
        "scorable_ids": ids,
        "all_labels": labels,
        "macro_labels": fixed,
        "gold_support": {lab: support[lab] for lab in labels},
        "rare_labels": [lab for lab in labels if 0 < support[lab] < 10],
        "absent_labels": [lab for lab in labels if not support[lab]],
    }


def contributions(records, outputs, ep):
    for value in outputs.values():
        if value is None:
            continue
        c.require(type(value["hate"]) is bool, "Malformed predicted gate")
        if not value["hate"]:
            c.require(
                value["hate_type"] == []
                and value["target_group"] == []
                and (value["severity"] is None),
                "Ungated fine prediction",
            )
    result = original.contributions(
        records, {k: {"result": v, "n_valid": int(v is not None)} for k, v in outputs.items()}, ep
    )
    result["pred"] = np.zeros_like(result["gold"])
    for i, key in enumerate(result["ids"]):
        if outputs[key] is not None:
            result["pred"][i] = original.bits(
                outputs[key][ep["dimension"]], ep["dimension"], ep["all_labels"]
            )
    return result


def summarize(values, records, dimension):
    result = original.score(values)
    ok = values["valid"]
    y = values["gold"]
    x = values["pred"]
    result["null_ids"] = [rid for rid, valid in zip(values["ids"], ok) if not valid]
    result["exact_set_accuracy"] = float((x[ok] == y[ok]).all(axis=1).mean()) if ok.any() else None
    result["per_label"] = {
        lab: {
            "support": int(y[:, i].sum()),
            "valid_support": int(y[ok, i].sum()),
            "tp": int(values["counts"][:, i, 0].sum()),
            "fp": int(values["counts"][:, i, 1].sum()),
            "fn": int(values["counts"][:, i, 2].sum()),
        }
        for i, lab in enumerate(values["labels"])
    }
    if dimension == "hate":
        lookup = {str(r["id"]): r for r in records}
        group = defaultdict(Counter)
        index = values["labels"].index("hate")
        for i, key in enumerate(values["ids"]):
            g = group[lookup[key]["source"]]
            g["eligible"] += 1
            if not ok[i]:
                g["invalid"] += 1
                continue
            truth = bool(y[i, index])
            pred = bool(x[i, index])
            g["hateful" if truth else "benign"] += 1
            g["fn"] += truth and (not pred)
            g["fp"] += not truth and pred
        result["gate_by_source"] = {
            key: dict(v)
            | {
                "false_positive_rate": v["fp"] / v["benign"] if v["benign"] else None,
                "false_negative_rate": v["fn"] / v["hateful"] if v["hateful"] else None,
            }
            for key, v in group.items()
        }
    return result


def contrast(
    left, right, clusters, *, sources=None, stratify=False, draws=10000, seed=20261008, family=None
):
    c.require(left and right and (draws > 0), "Both comparison sides and draws required")
    allc = left + right
    first = allc[0]
    c.require(
        all(
            (
                v["ids"] == first["ids"]
                and v["labels"] == first["labels"]
                and (v["macro_indices"] == first["macro_indices"])
                for v in allc
            )
        ),
        "Unpaired contributions",
    )
    c.require(len(clusters) == len(first["ids"]), "Cluster roster mismatch")
    if stratify:
        c.require(sources is not None and len(sources) == len(clusters), "Source strata missing")
    common = np.logical_and.reduce([x["valid"] for x in allc])
    idx = first["macro_indices"]
    own = [original.score(x) for x in allc]
    bounds = [None, None]
    if idx and len(common):
        bounds = [
            float(
                np.mean([x["invalid_inclusive_bounds"][0] for x in own[: len(left)]])
                - np.mean([x["invalid_inclusive_bounds"][1] for x in own[len(left) :]])
            ),
            float(
                np.mean([x["invalid_inclusive_bounds"][1] for x in own[: len(left)]])
                - np.mean([x["invalid_inclusive_bounds"][0] for x in own[len(left) :]])
            ),
        ]
    out = {
        "common_valid_ids": [key for key, v in zip(first["ids"], common) if v],
        "excluded_ids": [key for key, v in zip(first["ids"], common) if not v],
        "n_common_valid": int(common.sum()),
        "family_size": family,
        "full_roster_failure_difference_bounds": bounds,
        "draws": draws,
        "seed": seed,
        "source_stratified": stratify,
    }
    if not common.any() or not idx:
        return out | {
            "mean_difference": None,
            "ci95": None,
            "family_interval": None,
            "n_clusters": 0,
            "reason": "No common-valid inputs or no full-roster supported macro labels",
        }
    memberships = defaultdict(set)
    if stratify:
        for cluster, source in zip(clusters, sources):
            memberships[cluster].add(source)
    kept = [clusters[i] for i in np.flatnonzero(common)]
    unique = sorted(set(kept))
    mapping = {v: i for i, v in enumerate(unique)}
    grouped = np.zeros((len(allc), len(unique), len(first["labels"]), 3), dtype=np.int64)
    counts = np.stack([x["counts"][common] for x in allc])
    for i, key in enumerate(kept):
        grouped[:, mapping[key]] += counts[:, i]
    strata = defaultdict(list)
    for j, key in enumerate(unique):
        strata[tuple(sorted(memberships[key])) if stratify else ("all",)].append(j)
    delta = lambda score: (score[..., : len(left), None] - score[..., None, len(left) :]).mean(
        axis=(-2, -1)
    )
    scores = original.macro(counts.sum(axis=1), idx)
    point = float(delta(scores))
    values = []
    rng = np.random.default_rng(seed)
    checks = 0
    flat = grouped.reshape(len(allc), len(unique), -1)
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        picks = np.concatenate(
            [
                rng.choice(group, (size, len(group)), replace=True)
                for _, group in sorted(strata.items())
            ],
            axis=1,
        )
        weights = np.stack([np.bincount(x, minlength=len(unique)) for x in picks])
        total = np.einsum("bg,sgk->bsk", weights, flat, optimize=True).reshape(
            size, len(allc), len(first["labels"]), 3
        )
        c.require(
            np.array_equal(total[0], grouped[:, picks[0]].sum(axis=1)),
            "Cluster aggregation check failed",
        )
        checks += 1
        values.extend(delta(original.macro(total, idx)).tolist())
    return out | {
        "mean_difference": point,
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "family_interval": (
            np.quantile(values, [0.05 / (2 * family), 1 - 0.05 / (2 * family)]).tolist()
            if family
            else None
        ),
        "family_confidence": 1 - 0.05 / family if family else None,
        "n_clusters": len(unique),
        "common_f1_by_setting": scores.tolist(),
        "left_common_mean": float(scores[: len(left)].mean()),
        "right_common_mean": float(scores[len(left) :].mean()),
        "cluster_strata": {"|".join(k): len(v) for k, v in strata.items()},
        "direct_cluster_checks": checks,
    }


def scientific_rosters(study):
    if study == "broad":
        grouped = defaultdict(list)
        for row in c.rows(c.PREP / "broad_roster.jsonl"):
            grouped[row["phase"]].append(row | {"id": row["key"]})
        return (
            dict(grouped),
            {
                phase: sorted({d for r in rr for d in r["endpoints"]})
                for phase, rr in grouped.items()
            },
        )
    from analysis.common import replay as gold

    grouped = {}
    for phase in MATCHED_DIMS:
        grouped[phase] = [
            r
            | {
                "id": r["lang"] + "/" + str(r["id"]),
                "key": r["lang"] + "/" + str(r["id"]),
                "text_cluster": hashlib.sha256(gold.normalized(r["text"]).encode()).hexdigest(),
            }
            for r in gold.gold_rows(phase)
        ]
    return (grouped, {phase: [dim] for phase, dim in MATCHED_DIMS.items()})


def load_predictions(study):
    from analysis.common import execution as e

    b = e.queue(study)
    e.validate(study, True)
    plan_sha = c.sha(b.PLAN)
    root = a_root = b.BASE / "live"
    done = c.read(root / "complete.json")
    c.require(
        done["mode"] == "live"
        and done["plan_sha256"] == plan_sha
        and (done["status"] == "all_studies_executed_and_audited"),
        "Complete live audit required before scoring",
    )
    c.verify(done["review_hashes"])
    schedule = c.read(b.BASE / "schedule.json")
    output = defaultdict(dict)
    seen = set()
    for stage in schedule["stages"]:
        receipt = c.read(root / study / "stages" / stage["id"] / "complete.json")
        b.verify_review(stage, root / study, plan_sha, "live")
        for j in stage["jobs"]:
            ident = (j["key"], j["condition"], j["repeat"])
            c.require(ident not in seen, "Repeated scored job")
            seen.add(ident)
            folder = b.item_dir(root / study, j, plan_sha)
            rp = folder / "result.json"
            cp = folder / "complete.json"
            name = cp.relative_to(b.a.storage_path(root / study)).as_posix()
            c.require(
                c.sha(cp) == receipt["item_receipt_hashes"][name],
                "Item receipt changed since audit",
            )
            completed = c.read(cp)
            c.require(
                completed["hashes"][rp.relative_to(b.a.storage_path(root / study)).as_posix()]
                == c.sha(rp),
                "Result changed since raw audit",
            )
            value = c.read(rp)
            c.require(
                value["identity"] == b.identity(j, plan_sha)
                and value["execution_mode"] == "live"
                and (value["messages_sha256"] == j["messages_sha256"]),
                "Result input/mode mismatch",
            )
            output[j["condition"], j["repeat"]][j["key"]] = value["result"]
    c.require(len(seen) == e.SIZES[study][2], "Transport-incomplete matrix")
    return (
        output,
        {
            "execution_plan": c.rel(b.PLAN),
            "plan_sha256": plan_sha,
            "completion": c.rel(root / "complete.json"),
            "completion_sha256": c.sha(root / "complete.json"),
        },
    )


def main(study):
    import yaml

    predictions, provenance = load_predictions(study)
    groups, dims = scientific_rosters(study)
    available = {}
    if study == "broad":
        path = c.BASE / "encoders/assembled/complete.json"
        enc = c.read(path)
        c.require(
            enc["status"] == "all_logits_verified_and_gated_predictions_assembled",
            "Fixed encoder verification required for complete broader scoring",
        )
        c.verify(enc["inputs"])
        c.verify(enc["outputs"])
        c.verify(enc["source_sha256"])
        available = enc["available_dimensions"]
        provenance["encoder_completion"] = {"path": c.rel(path), "sha256": c.sha(path)}
        for name in enc["outputs"]:
            for row in c.rows(c.ROOT / name):
                ident = (row["condition"], row["seed"])
                predictions.setdefault(ident, {})
                c.require(
                    row["key"] not in predictions[ident], "Duplicate assembled encoder output"
                )
                predictions[ident][row["key"]] = row["result"]
    taxonomy = yaml.safe_load((c.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
    results = {}
    for phase, rr in groups.items():
        keys = {r["id"] for r in rr}
        for dim in dims[phase]:
            ep = endpoint(rr, dim, taxonomy)
            runs = {}
            for (cond, repeat), output in predictions.items():
                if cond in available and (dim not in available[cond] or not keys <= output.keys()):
                    continue
                c.require(keys <= output.keys(), "Missing population result")
                runs[cond, repeat] = contributions(rr, {k: output[k] for k in keys}, ep)
            reference = {r["id"]: r for r in rr}
            ids = next(iter(runs.values()))["ids"]
            clusters = [
                (
                    reference[k]["functional_cluster"]
                    if phase == "en_hatecheck_eval"
                    else reference[k]["text_cluster"]
                )
                for k in ids
            ]
            sources = [reference[k]["source"] for k in ids]
            pairs = []
            if study == "broad":
                pairs = [
                    ("rag_full", "rag_small", 4 if (phase, dim) in PRIMARY_BROAD else None),
                    ("rag_full", "zero_shot", None),
                ]
                pairs += [
                    ("rag_full", name, None) for name in sorted(available) if (name, 0) in runs
                ]
            else:
                pairs = [
                    ("retrieved_complete", "coverage_matched_", 6),
                    ("retrieved_complete", "permuted_", 6),
                    ("retrieved_complete", "fixed_", None),
                    ("retrieved_complete", "retrieved_text_only", None),
                    ("retrieved_complete", "retrieved_labels_only", None),
                    ("retrieved_complete", "original_bridge", None),
                ]
            contrasts = {}
            for left_name, right_name, family in pairs:
                left = [runs[k] for k in sorted(runs) if k[0] == left_name]
                right = [
                    runs[k]
                    for k in sorted(runs)
                    if k[0].startswith(right_name)
                    if right_name.endswith("_") or k[0] == right_name
                ]
                value = contrast(left, right, clusters, sources=sources, family=family)
                value["sensitivity"] = contrast(
                    left,
                    right,
                    ids if phase == "en_hatecheck_eval" else clusters,
                    sources=sources,
                    stratify=phase != "en_hatecheck_eval",
                )
                if phase == "de_test" and dim == "hate":
                    keep = np.array([k != c.EMPTY_KEY for k in ids])
                    filtered = []
                    for v in left + right:
                        filtered.append(
                            v
                            | {
                                "ids": [k for k in ids if k != c.EMPTY_KEY],
                                **{n: v[n][keep] for n in ["gold", "pred", "counts", "valid"]},
                            }
                        )
                    value["empty_input_exclusion_sensitivity"] = contrast(
                        filtered[: len(left)],
                        filtered[len(left) :],
                        [k for k, on in zip(clusters, keep) if on],
                    )
                contrasts[left_name + " minus " + right_name] = value
            summaries = {
                cond + "/repeat" + str(rep): summarize(v, rr, dim)
                for (cond, rep), v in runs.items()
            }
            replicate = {}
            for cond in sorted({k[0] for k in runs}):
                values = [
                    summaries[cond + "/repeat" + str(rep)]["macro_f1_valid_only"]
                    for rep in range(3)
                ]
                replicate[cond] = {
                    "values": values,
                    "mean": (
                        float(np.mean(values)) if all((v is not None for v in values)) else None
                    ),
                    "sample_sd": (
                        float(np.std(values, ddof=1))
                        if all((v is not None for v in values))
                        else None
                    ),
                    "kind": (
                        "fixed fitted seeds"
                        if cond in available
                        else "decoder repeats; selection variant held fixed"
                    ),
                }
            results[phase + "/" + dim] = {
                "endpoint": ep,
                "per_condition_repeat": summaries,
                "replicate_summary": replicate,
                "contrasts": contrasts,
            }
            print("Scored", phase, dim, "with paired clusters.", flush=True)
    dest = c.BASE / "scores" / study / f"results_{__import__('time').time_ns()}.json"
    c.save(
        dest,
        {
            "created_utc": c.utc(),
            "status": "completed_live_matrix_scored",
            "study": study,
            "provenance": provenance,
            "contract": contract(),
            "results": results,
            "source_sha256": {c.rel(Path(__file__)): c.sha(Path(__file__))},
            "new_Qwen_calls": 0,
        },
    )
    print("Offline scores saved:", c.rel(dest), "; human interpretation/review still required.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=["broad", "matched"], required=True)
    args = p.parse_args()
    with c.offline():
        main(args.study)
