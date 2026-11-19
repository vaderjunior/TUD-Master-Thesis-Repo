from collections import Counter, defaultdict
import json
import socket
from unittest.mock import patch
import numpy as np
from analysis.feedback.supplements import scores as s
from analysis.feedback.supplements.archive import ROOT
from analysis.feedback.supplements.archive import DEST
from analysis.feedback.supplements.archive import REVIEW
from analysis.feedback.supplements.archive import STUDIES
from analysis.feedback.supplements.archive import LABELS
from analysis.feedback.supplements.archive import SEEDS
from analysis.feedback.supplements.archive import read
from analysis.feedback.supplements.archive import rows
from analysis.feedback.supplements.archive import sha
from analysis.feedback.supplements.archive import rel
from analysis.feedback.supplements.archive import save
from analysis.feedback.supplements.archive import verify
from analysis.feedback.supplements.archive import require
from analysis.feedback.supplements.archive import utc
from analysis.feedback.supplements.archive import forbid


def coverage(data, availability, ids, criterion, seed):
    policy = ["plain", "human", "generic"]
    by_pool = {}
    all_keys = {}
    for pool in SEEDS:
        eligible = [i for i in ids if criterion(availability[i, pool])]
        keys = [data.kpos[f"{p}_s{pool}", r] for p in policy for r in range(3)]
        common = data.common(eligible, keys)
        by_pool[pool] = {
            "original_ids": eligible,
            "common_ids": common,
            "excluded_ids": sorted(set(eligible) - set(common)),
        }
        all_keys[pool] = keys
    if any((not x["common_ids"] for x in by_pool.values())):
        return {"status": "unestimable_empty_pool", "by_pool": by_pool}
    union = sorted({i for v in by_pool.values() for i in v["common_ids"]})
    w, member, picks = s.weights(union, data.roster, seed)
    clusters = sorted({data.roster[i]["cluster"] for i in union})
    ci = {v: n for n, v in enumerate(clusters)}
    hateful = all((data.roster[i]["gold"]["hate"] for i in union))
    require(
        hateful or all((not data.roster[i]["gold"]["hate"] for i in union)),
        "Mixed coverage population",
    )
    boots = defaultdict(list)
    points = defaultdict(list)
    settings = {}
    for pool, v in by_pool.items():
        ids = v["common_ids"]
        pos = data.indices(ids)
        mm = np.array([ci[data.roster[i]["cluster"]] for i in ids])
        for p in policy:
            for rep in range(3):
                k = data.kpos[f"{p}_s{pool}", rep]
                if hateful:
                    raw = s.counts(data.y[pos], data.x[k, pos])
                    z = np.zeros((len(clusters), 7, 3), dtype=int)
                    np.add.at(z, mm, raw)
                    boot = s.f1(
                        (w @ z.reshape(len(clusters), -1)).reshape(len(w), 7, 3)
                    ).mean(axis=1)
                    metric = s.run_metric(data, ids, k)
                    point = metric["macro_f1"]
                    for b in [0, 1, 7, 9999]:
                        expanded = np.concatenate(
                            [np.flatnonzero(mm == c) for c in picks[b]]
                        )
                        require(
                            abs(boot[b] - s.f1(raw[expanded].sum(axis=0)).mean())
                            < 1e-12,
                            "Coverage bootstrap expansion differs",
                        )
                else:
                    error = data.gate[k, pos].astype(int)
                    num = np.zeros(len(clusters), dtype=int)
                    np.add.at(num, mm, error)
                    den = w @ np.bincount(mm, minlength=len(clusters))
                    require(
                        np.all(den > 0),
                        "Empty bootstrap pool requires explicit handling",
                    )
                    boot = w @ num / den
                    point = float(error.mean())
                    metric = {
                        "false_positive_rate": point,
                        "false_positives": int(error.sum()),
                        "n": len(ids),
                    }
                settings[f"{p}_s{pool}_rep{rep}"] = metric
                boots[p].append(boot)
                points[p].append(point)
    mean = {p: float(np.mean(points[p])) for p in policy}
    curves = {p: np.mean(boots[p], axis=0) for p in policy}
    contrasts = {
        l
        + "_minus_"
        + r: {
            "difference": mean[l] - mean[r],
            "ci95": s.intervals(curves[l] - curves[r]),
        }
        for l, r in [("human", "plain"), ("human", "generic"), ("generic", "plain")]
    }
    return {
        "status": "supporting_conditional_coverage_diagnostic",
        "endpoint": "seven_type_macro_f1" if hateful else "false_positive_rate",
        "by_pool": by_pool,
        "union_inputs": len(union),
        "cluster_count": len(clusters),
        "settings": settings,
        "groups": {
            p: {
                "mean": mean[p],
                "pool_means": np.array(points[p]).reshape(3, 3).mean(axis=1).tolist(),
            }
            for p in policy
        },
        "contrasts": contrasts,
        "bootstrap": {
            "seed": seed,
            "draws": 10000,
            "unit": "shared original-text clusters, all pool/condition/repeat memberships together",
        },
        "interpretation": "Source-card stratum fixed before generation; conditional on pool coverage. Not a randomized authorship comparison or new primary.",
    }


def factorial_detail(data):
    primary = read(DEST / "factorial/primary.json")
    ids = primary["common_ids"]
    p = data.indices(ids)
    n = len(p)
    kk = len(data.keys)
    cat = np.array([LABELS.index(data.roster[i]["target_label"]) for i in ids])
    w, member, picks = s.weights(ids, data.roster, 202610041)
    clustered = np.zeros((w.shape[1], kk, 7, 7, 3), dtype=np.int64)
    for i, position in enumerate(p):
        clustered[member[i], :, cat[i]] += s.counts(
            data.y[position][None, :], data.x[:, position, :]
        )
    bs = s.f1((w @ clustered.reshape(w.shape[1], -1)).reshape(len(w), kk, 7, 7, 3))
    target = np.diagonal(bs, axis1=2, axis2=3)
    other = (bs.sum(axis=3) - target) / 6
    boots = {
        c: target[:, [data.kpos[c, r] for r in range(3)]].mean(axis=1)
        for c in data.conditions
    }
    other_boot = {
        c: other[:, [data.kpos[c, r] for r in range(3)]].mean(axis=(1, 2))
        for c in data.conditions
    }
    per_category = {}
    for li, label in enumerate(LABELS):
        per_category[label] = {
            "pairs": int((cat == li).sum()),
            "positive_support": primary["common_positive_support_by_category"][li],
            "cell_f1": {
                c: primary["groups"][c]["per_category_f1"][li] for c in data.conditions
            },
            "effects": {
                name: {
                    "difference": sum(
                        (
                            weight * primary["groups"][c]["per_category_f1"][li]
                            for c, weight in terms.items()
                        )
                    ),
                    "ci95": s.intervals(
                        sum((weight * boots[c][:, li] for c, weight in terms.items()))
                    ),
                }
                for name, terms in s.EFFECTS.items()
            },
        }
    other_effects = {
        name: {
            "difference": sum(
                (
                    weight * primary["groups"][c]["other_six_macro_f1"]
                    for c, weight in terms.items()
                )
            ),
            "ci95": s.intervals(
                sum((weight * other_boot[c] for c, weight in terms.items()))
            ),
        }
        for name, terms in s.EFFECTS.items()
    }
    flips = {}
    for left, right in [
        ("E1D1", "E0D0"),
        ("E1D0", "E0D0"),
        ("E1D1", "E0D1"),
        ("E0D1", "E0D0"),
        ("E1D1", "E1D0"),
    ]:
        by_label = {}
        total = Counter()
        for li, label in enumerate(LABELS):
            v = Counter()
            for i, position in enumerate(p):
                if cat[i] != li:
                    continue
                for rep in range(3):
                    a = data.kpos[left, rep]
                    b = data.kpos[right, rep]
                    x = bool(data.x[a, position, li])
                    old = bool(data.x[b, position, li])
                    gold = bool(data.y[position, li])
                    if x == old:
                        continue
                    event = ("tp" if gold else "fp") + ("_gained" if x else "_lost")
                    v[event] += 1
                    same_gate = bool(data.gate[a, position] and data.gate[b, position])
                    v[
                        event + ("_both_gates_hate" if same_gate else "_gate_changed")
                    ] += 1
            by_label[label] = dict(v)
            total.update(v)
        flips[left + "_minus_" + right] = {
            "summed_over_three_repeats": dict(total),
            "by_label": by_label,
            "qualification": "Descriptive paired target-label transitions, not an additive decomposition of macro F1.",
        }
    prompt_counts = Counter(
        (
            p["messages_sha256"]
            for p in rows(STUDIES["factorial"] / "prompts.jsonl")
            if p["condition"] == "E1D1"
        )
    )
    return {
        "per_category": per_category,
        "other_six_label_effects": other_effects,
        "paired_flips": flips,
        "full_cell_distinct_prompts": len(prompt_counts),
        "full_cell_prompt_rows": sum(prompt_counts.values()),
        "full_cell_reused_message_multiplicities": dict(
            Counter(prompt_counts.values())
        ),
        "qualification": "All category-level and other-label intervals are unadjusted 95% supporting diagnostics; only the two aggregate planned effects use 97.5%.",
    }


def operational_detail():
    summary = read(REVIEW / "summary.json")
    export = list(rows(REVIEW / "predictions.jsonl"))
    bad = Counter()
    categories = Counter()
    from analysis.common import schema as isolated

    parser = isolated.adapter(ROOT / "config/taxonomy.yaml").bundle.parser
    allowed = {
        "hate_type": set(LABELS),
        "target_group": {
            "race",
            "religion",
            "gender",
            "sexual_orientation",
            "disability",
            "national_origin",
            "other",
        },
    }
    for r in export:
        if r["n_valid"] == 3:
            continue
        saved = read(ROOT / r["source"])
        for raw, pred in zip(saved["raw_runs"], saved["run_predictions"]):
            if pred is not None:
                continue
            try:
                obj = json.loads(parser.extract_json(raw[-1]))
            except (ValueError, TypeError):
                categories["non_json"] += 1
                continue
            if not isinstance(obj, dict):
                categories["not_object"] += 1
                continue
            for field, vocab in allowed.items():
                if isinstance(obj.get(field), list):
                    for label in obj[field]:
                        if isinstance(label, str) and label not in vocab:
                            bad[field + ":" + label] += 1
    late = []
    from analysis.feedback.supplements import recovery_support as recovery

    manifest = read(recovery.MANIFEST)
    for entry in manifest["calls"]:
        saved = read(ROOT / entry["item"] / "result.json")
        events = [
            e
            for e in saved["events"]
            if e["vote"] == entry["vote"] and e["turn"] == entry["turn"]
        ]
        require(
            len([e for e in events if e["action"] == "ok"]) == 1
            and events[-1]["action"] == "ok",
            "Outage call did not recover once",
        )
        require(
            read(ROOT / entry["item"] / "complete.json")[
                "reviewed_transport_amendment_sha256"
            ]
            == sha(recovery.MANIFEST),
            "Recovery receipt differs",
        )
        late.append(
            {
                "key": entry["job"]["key"],
                "condition": entry["job"]["condition"],
                "repeat": entry["job"]["repeat"],
                "vote": entry["vote"],
                "successful_attempt": events[-1]["attempt"],
                "saved_old_failures": sum((e["attempt"] < 5 for e in events)),
                "item_null": saved["result"] is None,
            }
        )
    return {
        "terminal_unknown_labels": dict(bad),
        "other_terminal_categories": dict(categories),
        "outage_recovered_calls": late,
        "null_unique_texts": len({r["key"] for r in summary["null_jobs"]}),
        "nulls_by_population": dict(
            Counter((r["population"] for r in summary["null_jobs"]))
        ),
        "qualification": "These 11 schema-null predictions remain in reporting. Model-list/name metadata does not establish hidden checkpoint identity. Timeout remote cost is unknown.",
    }


def main():
    done = read(DEST / "scoring_complete.json")
    verify(done["inputs"])
    verify(done["outputs"])
    verified = read(DEST / "independent_verification.json")
    require(verified["status"] == "passed", "Independent primary checks required")
    dest = DEST / "diagnostics"
    outputs = {}

    def put(name, value):
        p = dest / (name + ".json")
        save(p, value)
        outputs[rel(p)] = sha(p)
        print("Diagnostic saved: " + name, flush=True)

    save(
        dest / "plan.json",
        {
            "created_utc": utc(),
            "scope": "Declared category/other-label and source-card coverage diagnostics. Additional gate strict-valid sensitivities and recovered-null explanation are supporting post-hoc checks, not new primaries.",
            "source_sha256": sha(__file__),
            "primary_results_already_seen": True,
            "no_tuning_or_new_calls": True,
        },
    )
    export = list(rows(REVIEW / "predictions.jsonl"))
    human = s.Data("human_meaning", export)
    factor = s.Data("factorial", export)
    put("factorial_categories", factorial_detail(factor))
    put("operational", operational_detail())
    availability = {
        (r["key"], r["pool_seed"]): r
        for r in rows(STUDIES["human_meaning"] / "availability.jsonl")
    }
    strata = [
        ("available", lambda v: v["human_eligible"], 202610064),
        (
            "old_assistant_and_human",
            lambda v: v["human_eligible"] and v["old_assistant_available"],
            202610065,
        ),
        ("flagged", lambda v: v["human_eligible"] and v["source_flagged"], 202610066),
        ("fallback", lambda v: not v["human_eligible"], 202610067),
    ]
    for name, rule, seed in strata:
        for hate, pop in [(True, "hateful"), (False, "benign")]:
            put(
                "coverage_" + name + "_" + pop,
                coverage(human, availability, human.population(hate), rule, seed),
            )
    for hate, pop, seed in [(True, "hateful", 202610062), (False, "benign", 202610063)]:
        put(
            "human_gate_all_three_valid_" + pop,
            s.gate_compare(
                human, human.population(hate), s.groups_human(), seed, strict=True
            ),
        )
    inputs = {
        rel(DEST / "scoring_complete.json"): sha(DEST / "scoring_complete.json"),
        rel(DEST / "independent_verification.json"): sha(
            DEST / "independent_verification.json"
        ),
        rel(REVIEW / "predictions.jsonl"): sha(REVIEW / "predictions.jsonl"),
        rel(__file__): sha(__file__),
        rel(STUDIES["human_meaning"] / "availability.jsonl"): sha(
            STUDIES["human_meaning"] / "availability.jsonl"
        ),
    }
    save(
        dest / "complete.json",
        {
            "status": "supporting_diagnostics_complete",
            "created_utc": utc(),
            "inputs": inputs,
            "outputs": outputs,
            "new_classifier_calls": 0,
        },
    )


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ), threadpool_limits(limits=2):
        main()
