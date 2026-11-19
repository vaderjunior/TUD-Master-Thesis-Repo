from analysis.common import micro_f1_files as c
from analysis.common import micro_f1_reader as r
from analysis.common import micro_f1 as metric
import numpy as np
import traceback
from datetime import datetime, timezone

B = c.ROOT / "experiments/astra"
TYPE = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
TARGET = [
    "race",
    "religion",
    "gender",
    "sexual_orientation",
    "disability",
    "national_origin",
    "other",
]


def from_labels(per, labels):
    return np.asarray([[per[x][k] for k in ("tp", "fp", "fn")] for x in labels])


def relevance():
    base = B / "sq3_full_kb_relevance_20260930_v1/results_v1"
    diagnostics = c.read(base / "label_count_diagnostics.json")
    for name in ("primary", "primary_controls", "all_transfer"):
        d = c.read(base / (name + ".json"))
        values = {}
        for key, per in diagnostics[name]["per_run_label_counts"].items():
            z = from_labels(per, TYPE)
            score = metric.macro(z, range(7))
            values[key] = score
            r.equal(
                score,
                d["settings"][key]["common_valid"]["macro_f1"],
                "relevance count macro",
            )
            r.equal(
                metric.label_f1(z),
                [d["settings"][key]["common_valid"]["per_label_f1"][x] for x in TYPE],
                "relevance per label",
            )
        for key, old in d["contrasts"].items():
            left, right = key.split("_minus_") if "_minus_" in key else (None, None)
            if left is None:
                continue
            l = [v for k, v in values.items() if k.startswith(left + "_")]
            rr = [v for k, v in values.items() if k.startswith(right + "_")]
            if l and rr and ("mean_difference" in old):
                r.equal(
                    float(np.mean(l) - np.mean(rr)),
                    old["mean_difference"],
                    "relevance contrast",
                )
    return "common-valid label counts for primary and both main control rosters; partial-cover cases remain descriptive"


def reserved():
    base = B / "sq3_reserved_part5_results_20261020_v1"
    d = c.read(base / "paired_run_details.json")
    saved = {}
    for row in c.lines(base / "quality_outcomes.jsonl"):
        if row.get("gold_hate") is not True or not row.get("known_type_gold"):
            continue
        key = (row["condition"], row["repeat"], row["key"])
        c.require(key not in saved, "Duplicate reserved outcome")
        saved[key] = (row["gold_types"], row["predicted_types"], row["n_valid"])
    for key, item in d.items():
        if not key.endswith("/types"):
            continue
        common = item["point"]["common_ids"]
        means = []
        for arm in item["arms"]:
            vals = []
            for run in arm:
                z = from_labels(run["per_label"], TYPE)
                r.equal(
                    metric.macro(z, range(7)),
                    run["metrics"]["type_macro_f1_fixed7"],
                    "reserved count macro",
                )
                raw = [
                    saved[run["condition"], run["repeat"], identifier]
                    for identifier in common
                ]
                c.require(
                    all((p is not None and votes > 0 for _, p, votes in raw)),
                    "Invalid reserved common answer",
                )
                z = r.counts(
                    [r.bits(g, TYPE) for g, _, _ in raw],
                    [r.bits(p, TYPE) for _, p, _ in raw],
                )
                vals.append(float(metric.macro(z, range(7))))
            means.append(float(np.mean(vals)))
        old = item["point"]["metrics"]["type_macro_f1_fixed7"]
        r.equal(means[0], old["treatment"], "reserved paired treatment")
        r.equal(means[1], old["control"], "reserved paired control")
        r.equal(means[0] - means[1], old["difference"], "reserved paired contrast")
    return "all recorded type arm counts and per-text outcomes on the original pair-specific common IDs, including their treatment/control means"


def human():
    d = c.read(
        B / "sq3_human_feedback_results_20261020_v1/frozen_descriptive_analysis.json"
    )
    for cohort, targets in d["transfer"].items():
        for target, record in targets.items():
            for name, old in record["labels"]["arms"].items():
                vals = []
                for run in old["per_repeat"]:
                    z = from_labels(run["labels"], TYPE)
                    value = metric.macro(z, range(7))
                    vals.append(value)
                    r.equal(value, run["macro_f1_fixed_seven"], "human count macro")
                r.equal(float(np.mean(vals)), old["mean_macro_f1"], "human repeat mean")
    return "original seven-label counts for all eight descriptive cohorts, three distinct gold targets and all feedback arms"


def quality(study):
    if study == "transfer":
        base = B / "sq3_transfer_part4_20261023_v1"
        records = c.read(base / "quality_results.json")
    else:
        base = B / "sq3_iterative_review_20261026_v1/part6"
        records = c.read(base / "analysis.json")["records"]
    schema = c.read(base / "feature_schema.json")
    meta = list(c.lines(base / "metadata.jsonl"))
    pred = r.arrays(base / "prediction_codes.npz")
    features = r.arrays(base / "features.npz")
    registry = c.read(c.BASE / "comparison_registry_v2.json")
    for u in [x for x in registry["units"] if x["study"] == study]:
        rec = records[u["endpoint"]]
        dim = rec["head"]
        labels = TYPE if dim == "hate_type" else TARGET
        idx = rec["common_indices"] if study == "transfer" else rec["indices"]
        field = "gold_hate_types" if dim == "hate_type" else "gold_target_groups"
        c.require(
            all(
                (
                    meta[i][field] is not None and meta[i]["gold_hate"] is True
                    for i in idx
                )
            ),
            "Quality gold eligibility differs",
        )
        g = np.asarray([r.bits(meta[i][field], labels) for i in idx], bool)
        codes = pred[dim][idx]
        valid = codes >= 0
        bitpred = codes[..., None] & 1 << np.arange(7) != 0
        shape = bitpred.shape[1:-1]
        flatpred = bitpred.reshape(len(idx), -1, 7).transpose(1, 0, 2)
        flatvalid = valid.reshape(len(idx), -1).T
        raw = np.stack(
            [r.counts(g, p, v) for p, v in zip(flatpred, flatvalid)]
        ).reshape(*shape, 7, 3)
        order = schema["features"][dim]
        positions = [
            order.index(x + ":" + k) for x in labels for k in ("tp", "fp", "fn")
        ]
        summed = features[dim][idx].sum(0)
        saved = np.asarray(rec["counts"])
        r.equal(summed, saved, study + " feature sums")
        r.equal(
            raw,
            saved[..., positions].reshape(*shape, 7, 3),
            study + " raw label counts",
        )
        vals = np.asarray(
            [metric.macro(x, rec["macro_indices"]) for x in raw.reshape(-1, 7, 3)]
        ).reshape(shape)
        metric_names = (
            rec["metrics"]
            if isinstance(rec["metrics"], list)
            else schema["metrics"][dim]
        )
        r.equal(
            vals,
            np.asarray(rec["run_scores"])[..., metric_names.index("macro_f1")],
            study + " original macro",
        )
        if study == "transfer":
            old = rec["metrics"]["macro_f1"]
            r.equal(float(vals[0].mean()), old["reference"], "transfer reference mean")
            r.equal(
                float(vals[1:].mean(axis=-1).mean()),
                old["treatment"],
                "transfer pool mean",
            )
            r.equal(
                float(vals[1:].mean() - vals[0].mean()),
                old["difference"],
                "transfer contrast",
            )
        else:
            means = vals.mean(-1)

            def policy(round_no, name):
                selected = [
                    i
                    for i, k in enumerate(schema["conditions"])
                    if k == name or k.startswith(name + "_")
                ]
                c.require(selected, "No policy members")
                return float(means[round_no, selected].mean())

            for j, key in enumerate(rec["column_order"]):
                point = sum(
                    (
                        weight * policy(rd, name)
                        for rd, name, weight in rec["columns"][key]
                    )
                )
                r.equal(point, rec["estimates"][j][0], "rounds policy contrast")
    return "all registered global population/view arrays: gold eligibility, prediction bit codes, saved feature counts, macro and policy means"


def main():
    results = []
    for key, fn in [
        ("relevance", relevance),
        ("reserved", reserved),
        ("human", human),
        ("transfer", lambda: quality("transfer")),
        ("rounds", lambda: quality("rounds")),
    ]:
        before = sum(r.CHECKS.values())
        try:
            detail = fn()
            results.append(
                dict(
                    study=key,
                    status="passed",
                    checks=sum(r.CHECKS.values()) - before,
                    depth=detail,
                )
            )
            print(key, "passed", sum(r.CHECKS.values()) - before, flush=True)
        except Exception as e:
            results.append(
                dict(
                    study=key,
                    status="failed",
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
            )
            print(key, "FAILED", str(e), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    name = "reader_sq3_" + stamp + ".json"
    c.save(
        name,
        dict(
            created_utc=stamp,
            results=results,
            checks=dict(r.CHECKS),
            max_absolute_error=r.MAX_ERROR,
            sources=c.BINDINGS,
            real_micro_calculations=0,
        ),
    )
    print("REPORT", name, flush=True)
    if any((x["status"] == "failed" for x in results)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
