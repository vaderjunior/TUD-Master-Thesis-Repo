import json
import socket
from unittest.mock import patch
import numpy as np
import pandas as pd
from analysis.knowledge.allocation import results as q

a = q.a
b = q.b
ROOT = q.ROOT
SUBSET = "en_dev_eval_severity"
DEST = (
    ROOT
    / "Thesis_Log/astra_result/evidence/sq1_allocation_severity_diagnostics_20260924_v1.json"
)
LABELS = ["low", "medium", "high"]
COLS = LABELS + ["none_gate_false", "none_gate_true", "invalid"]


def matrix(rows, outputs, keep):
    conf = {g: {p: 0 for p in COLS} for g in LABELS}
    for row in rows:
        rid = str(row["id"])
        if rid not in keep:
            continue
        out = outputs[rid]
        result = out["result"]
        a.require(
            (result is not None) == (out["n_valid"] > 0), "Contradictory validity"
        )
        if result is None:
            column = "invalid"
        elif result["severity"] is not None:
            a.require(
                result["hate"] and result["severity"] in LABELS,
                "Invalid gated severity",
            )
            column = result["severity"]
        else:
            column = "none_gate_true" if result["hate"] else "none_gate_false"
        conf[row["severity"]][column] += 1
    return conf


def summarize_matrix(conf):
    scores = {}
    for label in LABELS:
        tp = conf[label][label]
        fp = sum((conf[gold][label] for gold in LABELS if gold != label))
        fn = sum((conf[label][col] for col in COLS if col not in (label, "invalid")))
        scores[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "per_label_f1": scores,
        "macro_f1": sum(scores.values()) / 3,
        "correct": sum((conf[x][x] for x in LABELS)),
        "assigned_too_high": sum(
            (
                conf[g][p]
                for g in LABELS
                for p in LABELS
                if LABELS.index(p) > LABELS.index(g)
            )
        ),
        "assigned_too_low": sum(
            (
                conf[g][p]
                for g in LABELS
                for p in LABELS
                if LABELS.index(p) < LABELS.index(g)
            )
        ),
        **{col: sum((conf[g][col] for g in LABELS)) for col in COLS[3:]},
    }


def main():
    a.require(not DEST.exists(), "Preserve completed severity check")
    p = q.validate(SUBSET)
    folder = q.DEST / SUBSET
    rp = folder / "results.json"
    result = a.read(rp)
    cp = folder / "complete.json"
    complete = a.read(cp)
    ep = ROOT / complete["evidence"]
    a.require(
        a.digest(rp)
        == complete["results_sha256"]
        == complete["evidence_sha256"]
        == a.digest(ep),
        "Score receipt/evidence differs",
    )
    a.require(
        result["plan_sha256"]
        == complete["plan_sha256"]
        == a.digest(folder / "plan.json"),
        "Scoring plan differs",
    )
    a.require(
        result["gold_rosters_scored"] == [SUBSET] and result["new_API_calls"] == 0,
        "Unexpected scoring work",
    )
    a.require(complete["assembled"] == result["assembled"], "Export manifest differs")
    rows = pd.read_parquet(ROOT / f"data/processed/{SUBSET}.parquet").to_dict("records")
    a.require(
        len(rows) == 45
        and all((bool(row["gate"]) and row["severity"] in LABELS for row in rows)),
        "Gold roster differs",
    )
    ids = [str(row["id"]) for row in rows]
    settings = {}
    verified = {}
    check_count = 0
    for name, record in result["assembled"].items():
        path = ROOT / name
        a.require(a.digest(path) == record["sha256"], "Export bytes changed")
        verified[name] = record["sha256"]
        exported = b.core.rows(path)
        a.require(
            record["rows"] == len(exported) == 45
            and [row["id"] for row in exported] == ids,
            "Export order/coverage differs",
        )
        ident = (exported[0]["condition"], exported[0]["replicate"])
        a.require(ident not in settings, "Duplicate export setting")
        settings[ident] = {}
        for row in exported:
            a.require(
                (row["condition"], row["replicate"]) == ident, "Mixed export setting"
            )
            source = ROOT / row["source"]
            h = a.digest(source)
            a.require(
                h
                == row["source_sha256"]
                == result["verified_prediction_hashes"][row["source"]],
                "Export source differs",
            )
            raw = a.read(source)
            job = raw["identity"]
            a.require(
                job["key"] == row["key"] == "en/" + row["id"]
                and job["condition"] == ident[0]
                and (job["repeat"] == ident[1]),
                "Export attribution differs",
            )
            a.require(
                row["result"] == raw["result"] and row["n_valid"] == raw["n_valid"],
                "Export prediction differs",
            )
            settings[ident][row["id"]] = row
            verified[row["source"]] = h
            check_count += 1
    q.check_settings(settings, ids, a.read(q.BASE / "analysis_plan.json"))
    common = [
        rid
        for rid in ids
        if all((outputs[rid]["result"] is not None for outputs in settings.values()))
    ]
    paired = result["paired_comparison"]
    a.require(common == paired["common_valid_ids"], "Shared mask differs")
    a.require(
        paired["bootstrap_draws"] == 10000
        and paired["bootstrap_seed"] == 20260924
        and (paired["independent_bootstrap_setting_values"] == 90000),
        "Bootstrap scope differs",
    )
    diagnostics = {}
    for arm in q.CONDITIONS:
        runs = []
        for rep in range(3):
            outputs = settings[arm, rep]
            own = matrix(rows, outputs, set(ids))
            matched = matrix(rows, outputs, set(common))
            own_stats = summarize_matrix(own)
            common_stats = summarize_matrix(matched)
            score = result["conditions"][arm]["runs"][rep]["score"]
            a.require(own == score["severity_confusion"], "Export confusion differs")
            a.require(
                abs(own_stats["macro_f1"] - score["macro_f1_valid_only"]) < 1e-14,
                "Export own-valid F1 differs",
            )
            a.require(
                all(
                    (
                        abs(own_stats["per_label_f1"][l] - score["per_label"][l]["f1"])
                        < 1e-14
                        for l in LABELS
                    )
                ),
                "Export per-label F1 differs",
            )
            runs.append(
                {
                    "repeat": rep,
                    "own_confusion": own,
                    "own_summary": own_stats,
                    "common_confusion": matched,
                    "common_summary": common_stats,
                }
            )
        summary = q.summarize([run["common_summary"]["macro_f1"] for run in runs])
        saved = paired["conditions"][arm]["common_valid_summary"]
        for key in ("values", "mean", "sample_sd", "min", "max", "range"):
            a.require(
                np.allclose(summary[key], saved[key], rtol=0, atol=1e-14),
                "Shared-mask repeats/mean/SD differ: " + key,
            )
        a.require(
            summary["replicate_kind"] == saved["replicate_kind"],
            "Replicate meaning differs",
        )
        diagnostics[arm] = {
            "runs": runs,
            "common_summary": summary,
            "mean_common_errors": {
                key: float(np.mean([run["common_summary"][key] for run in runs]))
                for key in (
                    "correct",
                    "assigned_too_high",
                    "assigned_too_low",
                    "none_gate_false",
                    "none_gate_true",
                    "invalid",
                )
            },
            "mean_common_label_f1": {
                l: float(
                    np.mean([run["common_summary"]["per_label_f1"][l] for run in runs])
                )
                for l in LABELS
            },
        }
    for name, weights in q.CONTRASTS.items():
        delta = sum(
            (
                w * diagnostics[arm]["common_summary"]["mean"]
                for arm, w in weights.items()
            )
        )
        a.require(
            abs(delta - paired["contrasts"][name]["mean_difference"]) < 1e-14
            and paired["contrasts"][name]["family_ci"] is None,
            "Severity contrast differs",
        )
    for path in (
        rp,
        cp,
        ep,
        folder / "plan.json",
        ROOT / "analysis/knowledge/allocation/validate_severity.py",
    ):
        verified[b.rel(path)] = a.digest(path)
    report = {
        "created_utc": a.utc(),
        "status": "severity_exports_and_categorical_errors_independently_checked",
        "subset": SUBSET,
        "export_files": len(settings),
        "exported_rows_verified": check_count,
        "common_valid_ids": common,
        "common_gold_support": {
            label: sum(
                (str(row["id"]) in common and row["severity"] == label for row in rows)
            )
            for label in LABELS
        },
        "diagnostics": diagnostics,
        "verified_hashes": verified,
        "new_API_calls": 0,
        "gold_rosters_scored": [SUBSET],
        "limits": "Three runs of the same 45 texts, not 135 independent texts per condition. Confusions are descriptive; no new adaptive contrast, retry or model selection.",
    }
    a.save_new(DEST, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows_verified": check_count,
                "common_gold_support": report["common_gold_support"],
                "conditions": {
                    arm: {key: value for key, value in d.items() if key != "runs"}
                    for arm, d in diagnostics.items()
                },
                "evidence_sha256": a.digest(DEST),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", q.forbidden), patch.object(
        socket, "create_connection", q.forbidden
    ), patch.object(b.core, "live_client", q.forbidden):
        main()
