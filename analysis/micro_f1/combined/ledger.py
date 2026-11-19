from analysis.micro_f1.combined import prepare as p
from collections import Counter
from pathlib import Path
import csv
import json
import math
import hashlib

c = p.c
ROWS = []
COHORTS = {}
METRICS = ["macro", "micro", "micro_macro_labels"]


def cls(interval):
    if interval is None:
        return "not_tested"
    c.require(len(interval) == 2 and interval[0] <= interval[1], "Invalid interval")
    return (
        "positive"
        if interval[0] > 0
        else "negative" if interval[1] < 0 else "includes_zero"
    )


def point(x):
    return 0 if x == 0 else 1 if x > 0 else -1


def difference_kind(a, b):
    ma, mi = cls(a["ci95"]), cls(b["ci95"])
    opposite = point(a["value"]) * point(b["value"]) < 0
    if ma == "not_tested" or mi == "not_tested":
        return (
            "descriptive_point_reversal" if opposite else "descriptive_same_direction"
        )
    if ma in ["positive", "negative"] and mi in ["positive", "negative"]:
        return (
            "supported_opposite_directions" if ma != mi else "same_supported_direction"
        )
    if ma != mi:
        return ("micro_only_" + mi) if ma == "includes_zero" else ("macro_only_" + ma)
    return "uncertain_point_reversal" if opposite else "both_uncertain_same_direction"


def metric(v):
    c.require(
        v.get("value", v.get("difference")) is not None, "Missing comparison value"
    )
    adj = v.get("original_adjusted_interval", v.get("adjusted_interval"))
    conf = v.get("original_adjusted_confidence", v.get("adjusted_confidence"))
    return dict(
        value=v.get("value", v.get("difference")),
        ci95=v.get("ci95"),
        adjusted_interval=adj,
        adjusted_confidence=conf,
        same_original_confidence_diagnostic=v.get(
            "original_confidence_diagnostic",
            v.get("original_family_confidence_diagnostic"),
        ),
    )


def add(
    part,
    study,
    endpoint,
    view,
    key,
    metrics,
    source,
    pointer,
    n,
    labels,
    macro_labels,
    qid=None,
    ids=None,
    original=None,
    extra=None,
):
    values = {m: metric(metrics[m]) for m in METRICS}
    uid = f"{part}|{study}|{endpoint}|{view}|{key}"
    if original is not None:
        values["macro"]["adjusted_interval"] = original.get("interval")
        values["macro"]["adjusted_confidence"] = original.get("confidence")
    for m, v in values.items():
        c.require(math.isfinite(v["value"]), "Nonfinite value")
        v["class95"] = cls(v["ci95"])
        v["adjusted_class"] = cls(v["adjusted_interval"])
        if v["adjusted_interval"] is not None:
            c.require(v["adjusted_confidence"] is not None, "Missing confidence")
            c.require(
                v["adjusted_interval"][0] <= v["ci95"][0] + 1e-12
                and v["adjusted_interval"][1] >= v["ci95"][1] - 1e-12,
                "Adjusted interval narrower than 95%",
            )
    w = values["micro_macro_labels"]["value"] - values["macro"]["value"]
    s = values["micro"]["value"] - values["micro_macro_labels"]["value"]
    c.require(
        abs((values["micro"]["value"] - values["macro"]["value"]) - (w + s)) < 1e-12,
        "Decomposition mismatch",
    )
    if labels == macro_labels:
        c.require(abs(s) < 1e-12, "Different full/restricted point with same label set")
    cohort = None
    if ids is not None:
        c.require(len(ids) == n and len(set(ids)) == n, "Cohort ID mismatch")
        ordered = sorted(ids)
        cohort = hashlib.sha256(
            json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        if cohort not in COHORTS:
            COHORTS[cohort] = dict(n=n, ids=ordered, rows=[])
        COHORTS[cohort]["rows"].append(uid)
    ROWS.append(
        dict(
            id=uid,
            part=part,
            study=study,
            endpoint=endpoint,
            view=view,
            contrast=key,
            n=n,
            labels=labels,
            macro_labels=macro_labels,
            metrics=values,
            question_id=qid,
            cohort_sha256=cohort,
            source=c.rel(source),
            source_pointer=pointer,
            matched95_interpretation=difference_kind(values["macro"], values["micro"]),
            point_direction_reversed=point(values["macro"]["value"])
            * point(values["micro"]["value"])
            < 0,
            full_vs_restricted_class95_changed=values["micro"]["class95"]
            != values["micro_macro_labels"]["class95"],
            averaging_component=w,
            label_scope_component=s,
            extra=extra or {},
        )
    )


def load_retrieval():
    index = c.read(p.ROOTS[2] / "accepted_analysis_index.json")
    for branch, entry in index.items():
        if branch == "synthetic":
            continue
        for f in entry["outputs"]:
            path = c.ROOT / f
            r = c.read(path)
            view = r.get("pool", r.get("condition", "main"))
            for j, x in enumerate(r["contrasts"]):

                key = (
                    "+".join(sorted({str(k[0]) for k in x["left_settings"]}))
                    + " minus "
                    + "+".join(sorted({str(k[0]) for k in x["right_settings"]}))
                )
                add(
                    2,
                    r["study"],
                    r["endpoint"],
                    view,
                    key,
                    x["metrics"],
                    path,
                    f"/contrasts/{j}",
                    x["n_common"],
                    r["labels"],
                    r["macro_labels"],
                    x.get("question_id"),
                    x["common_ids"],
                    original=dict(
                        interval=x.get("original_macro_adjusted_interval"),
                        confidence=x.get("original_macro_adjusted_confidence"),
                    ),
                    extra=dict(
                        role=x.get("original_macro_role"),
                        missing_bounds=x.get("full_roster_missing_difference_bounds"),
                        resampling=x.get("resampling_sensitivity"),
                        draw_file=x["draw_file"],
                    ),
                )


def load_supervised():
    index = c.read(p.ROOTS[3] / "accepted_analysis_index.json")
    diag = c.read(p.ROOTS[3] / "diagnostics/results.json")
    gpu_ids = {
        (x["endpoint"], x["budget"], x["left"], x["right"]): x["question_id"]
        for x in diag["primary"]
        if x["study"] == "gpu"
    }
    for study, entry in index["studies"].items():
        if study == "combined_context":
            continue
        for f in entry["outputs"]:
            path = c.ROOT / f
            r = c.read(path)
            if study in ["gpu", "controls"]:
                for j, x in enumerate(r["contrasts"]):
                    qid = (
                        gpu_ids.get((r["endpoint"], x["budget"], x["left"], x["right"]))
                        if study == "gpu"
                        else None
                    )
                    add(
                        3,
                        study,
                        r["endpoint"],
                        str(x["budget"]),
                        x["left"] + " minus " + x["right"],
                        x["metrics"],
                        path,
                        f"/contrasts/{j}",
                        r["n"],
                        r["labels"],
                        r["macro_labels"],
                        qid,
                    )
            elif study == "five":
                for j, x in enumerate(r["contrasts"]):
                    add(
                        3,
                        study,
                        r["endpoint"],
                        "updated/" + str(x["budget"]),
                        x["left"] + " minus " + x["right"],
                        x["supplementary"],
                        path,
                        f"/contrasts/{j}/supplementary",
                        r["n"],
                        r["labels"],
                        r["labels"],
                        extra=dict(affected=x["affected"], original=x["original"]),
                    )
                for group in ["affected_pool", "aggregate147"]:
                    x = r[group]
                    add(
                        3,
                        study,
                        r["endpoint"],
                        group,
                        "updated minus original",
                        x["change"],
                        path,
                        "/" + group + "/change",
                        r["n"],
                        r["labels"],
                        r["labels"],
                    )
            else:
                for j, x in enumerate(r["contrasts"]):
                    cells = [
                        cell
                        for cell in r["cells"]
                        if cell["endpoint"].replace("broad/", "") == x["endpoint"]
                    ]
                    c.require(len(cells) == 1, "Ambiguous supervised cell")
                    cell = cells[0]
                    modes = list(x["variations"])
                    main = "unstratified" if "unstratified" in modes else modes[0]
                    for mode, v in x["variations"].items():
                        qid = x.get("question_id") if mode == main else None
                        add(
                            3,
                            study,
                            x["endpoint"],
                            x.get("group", x.get("role", "supporting")) + "/" + mode,
                            x["left"] + " minus " + x["right"],
                            v["metrics"],
                            path,
                            f"/contrasts/{j}/variations/{mode}",
                            cell["n"],
                            cell["labels"],
                            cell["macro_labels"],
                            qid,
                            extra=dict(
                                inference_mode=mode,
                                main_mode=main,
                                draw_file=v.get("draws"),
                                per_pool=x.get("per_pool"),
                            ),
                        )


def load_matched():
    index = c.read(p.ROOTS[4] / "accepted_analysis_index.json")
    inner = c.read(c.ROOT / index["analysis_index"])
    for endpoint, entry in inner["endpoints"].items():
        path = c.ROOT / entry["path"]
        r = c.read(path)
        for key, x in r["contrasts"].items():
            if "metrics" not in x:
                continue
            qid = x.get("question_id")

            if x.get("family_size") and qid is None:
                qid = r["unit"]["id"] + "|" + key
            add(
                4,
                "matched",
                endpoint,
                "main",
                key,
                x["metrics"],
                path,
                "/contrasts/" + key,
                x["n_common"],
                r["labels"],
                r["macro_labels"],
                qid,
                x["common_valid_ids"],
                extra={
                    k: x.get(k)
                    for k in [
                        "full_roster_missing_difference_bounds",
                        "sensitivities",
                        "variation",
                        "resampling_sensitivity",
                        "draw_file",
                    ]
                },
            )


def load_sq1(part):
    index = c.read(p.ROOTS[part] / "accepted_analysis_index.json")
    for f in index["outputs"]:
        path = c.ROOT / f
        r = c.read(path)
        view = r.get("view", "paired")
        for key, x in r["contrasts"].items():
            add(
                part,
                r["unit"]["study"],
                r["unit"]["endpoint"],
                view,
                key,
                x["metrics"],
                path,
                "/contrasts/" + key,
                r["n_common"],
                r["labels"],
                r["macro_labels"],
                x.get("question_id"),
                r["common_ids"],
                extra={
                    k: x.get(k)
                    for k in [
                        "full_roster_missing_bounds",
                        "sensitivities",
                        "by_repeat",
                        "leave_one_repeat_out",
                        "by_pool",
                        "leave_one_pool_out",
                    ]
                },
            )


def load_sq3():
    index = c.read(p.ROOTS[7] / "accepted_analysis_index.json")
    for f in index["outputs"]:
        path = c.ROOT / f
        r = c.read(path)
        for key, x in r["columns_results"].items():
            if not x["contrast"]:
                continue
            add(
                7,
                r["study"],
                r["endpoint"],
                r["id"],
                key,
                x["metrics"],
                path,
                "/columns_results/" + key,
                r["n_common"],
                r["labels"],
                r["macro_labels"],
                x["new_micro_question"],
                r["common_ids"],
                extra=dict(
                    descriptive=r["descriptive"],
                    strict_point=x.get("strict_point"),
                    variation=x.get("variation"),
                    missing_bounds=x.get("full_roster_missing_bounds"),
                    selection_notes=r.get("selection_notes"),
                ),
            )


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("x", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    k: (
                        json.dumps(v, ensure_ascii=False, allow_nan=False)
                        if isinstance(v, (list, dict))
                        else v
                    )
                    for k, v in row.items()
                }
            )


def main():
    c.require((p.BASE / "analysis_plan.json").exists(), "Freeze first")
    for fn in [
        load_retrieval,
        load_supervised,
        load_matched,
        lambda: load_sq1(5),
        lambda: load_sq1(6),
        load_sq3,
    ]:
        fn()
    c.require(
        len({x["id"] for x in ROWS}) == len(ROWS), "Duplicate normalized contrast ID"
    )
    families = c.read(p.BASE / "analysis_plan.json")["families"]
    membership = {
        qid: (name, f) for name, f in families.items() for qid in f["members"]
    }
    questions = [r for r in ROWS if r["question_id"]]
    c.require(len(questions) == len(membership) == 77, "Question count mismatch")
    c.require(
        Counter(r["question_id"] for r in questions) == Counter(membership.keys()),
        "Question membership mismatch",
    )
    for row in ROWS:
        if row["question_id"]:
            name, family = membership[row["question_id"]]
            row["family"] = name
            row["family_size"] = family["m"]
            conf = row["metrics"]["micro"]["adjusted_confidence"]
            c.require(
                abs(conf - family["confidence"]) < 1e-12, "Frozen confidence changed"
            )
            row["bootstrap_lower_tail_expected_draws"] = 10000 * (1 - conf) / 2
            row["adjusted_boundary_distance_pp"] = 100 * min(
                abs(x) for x in row["metrics"]["micro"]["adjusted_interval"]
            )
            row["near_zero_adjusted_endpoint"] = (
                row["adjusted_boundary_distance_pp"] <= 0.25
            )
        else:
            row["family"] = None
            row["family_size"] = None

    reverse_groups = {}
    for row in ROWS:
        vals = tuple(round(abs(row["metrics"][m]["value"]), 14) for m in METRICS)
        signature = (
            row["part"],
            row["study"],
            row["endpoint"],
            row["view"],
            row["cohort_sha256"],
            vals,
        )
        reverse_groups.setdefault(signature, []).append(row["id"])
    summary = dict(
        rows=len(ROWS),
        by_part=dict(Counter(str(r["part"]) for r in ROWS)),
        questions=77,
        families=17,
        question_classes=dict(
            Counter(r["metrics"]["micro"]["adjusted_class"] for r in questions)
        ),
        question_matched95_classes=dict(
            Counter(r["matched95_interpretation"] for r in questions)
        ),
        all_classes=dict(Counter(r["matched95_interpretation"] for r in ROWS)),
        scope_changes=[
            r["id"] for r in ROWS if r["full_vs_restricted_class95_changed"]
        ],
        primary_near_boundary=[
            r["question_id"] for r in questions if r["near_zero_adjusted_endpoint"]
        ],
        reused_core_rows_not_counted=23,
        source_sha256=c.BINDINGS,
        warnings=[
            "Rows include overlapping cohorts, reverse contrasts and sensitivities; counts are coverage checks, not independent tests.",
            "The 17 adjustments remain separate; no global significance guarantee.",
            "Different interval classifications are not a test of a difference between metric effects.",
        ],
    )
    p.save("comparison_ledger.json", ROWS)
    p.save("cohorts.json", COHORTS)
    p.save("ledger_checks.json", summary)
    flat = []
    for row in ROWS:
        z = {k: v for k, v in row.items() if k not in ["metrics", "extra"]}
        for m, v in row["metrics"].items():
            for k, value in v.items():
                z[m + "_" + k] = value
        flat.append(z)
    write_csv(p.BASE / "comparison_ledger.csv", flat)
    write_csv(p.BASE / "prelisted_questions.csv", [r for r in flat if r["question_id"]])
    changes = [
        r
        for r in flat
        if r["matched95_interpretation"]
        not in [
            "same_supported_direction",
            "both_uncertain_same_direction",
            "descriptive_same_direction",
        ]
    ]
    write_csv(p.BASE / "metric_sensitive_inventory.csv", changes)
    print(
        "Combined ledger:",
        summary["rows"],
        "rows;",
        summary["by_part"],
        "; 77 questions / 17 families",
        flush=True,
    )
    print("Prelisted question statuses:", summary["question_classes"], flush=True)
    print(
        "Matched-95 question comparisons:",
        summary["question_matched95_classes"],
        flush=True,
    )


if __name__ == "__main__":
    main()
