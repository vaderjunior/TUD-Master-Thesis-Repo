"""Banks diagnostics."""

import json
import socket
from unittest.mock import patch
import pandas as pd
import numpy as np
from analysis.knowledge.banks import primary as p

a = p.a
ROOT = p.ROOT
DEST = p.DEST
OUT = ROOT / "Thesis_Log/astra_result/evidence/core_primary_diagnostics_20260917_v1.json"


def main():
    a.require(not OUT.exists(), "Preserve diagnostics")
    receipt = a.read(DEST / "complete.json")
    path = ROOT / receipt["evidence"]
    r = a.read(path)
    a.require(a.digest(path) == receipt["evidence_sha256"], "Primary evidence changed")
    result = {}
    hashes = {p.b.rel(path): a.digest(path)}
    for subset, info in r["endpoints"].items():
        gold_path = ROOT / f"data/processed/{subset}.parquet"
        a.require(
            a.digest(gold_path) == r["verified_input_hashes"][p.b.rel(gold_path)], "Gold changed"
        )
        gold = {str(row["id"]): row for row in pd.read_parquet(gold_path).to_dict("records")}
        for name, c in info["contrasts"].items():
            if not c["primary"] and c["right"] != "encoder_full":
                continue
            ids = c["common_valid_ids"]
            sides = {}
            for side in ("left", "right"):
                system = c[side]
                names = (
                    info["fewshot_groups"][system]["conditions"]
                    if system in info["fewshot_groups"]
                    else [system]
                )
                runs = []
                for condition in names:
                    for rep in range(3):
                        path = DEST / "assembled" / subset / f"{condition}_rep{rep}.jsonl"
                        h = a.digest(path)
                        a.require(h == r["assembled"][p.b.rel(path)]["sha256"], "Assembly changed")
                        hashes[p.b.rel(path)] = h
                        rows = p.keyed(p.b.rows(path))
                        counts = {label: [0, 0, 0] for label in info["endpoint"]["all_labels"]}
                        for rid in ids:
                            pred = rows[rid]["result"]
                            a.require(pred is not None, "Common-valid mask includes null")
                            if info["endpoint"]["dimension"] == "hate":
                                y = {"hate" if bool(gold[rid]["gate"]) else "not_hate"}
                                x = {"hate" if pred["hate"] else "not_hate"}
                            else:
                                y = set(gold[rid]["legal"])
                                x = set(pred["legal"])
                            for label, arr in counts.items():
                                if label in y and label in x:
                                    arr[0] += 1
                                elif label in x:
                                    arr[1] += 1
                                elif label in y:
                                    arr[2] += 1
                        runs.append(
                            {
                                "condition": condition,
                                "replicate": rep,
                                "labels": {
                                    label: {
                                        "tp": v[0],
                                        "fp": v[1],
                                        "fn": v[2],
                                        "f1": (
                                            2 * v[0] / (2 * v[0] + v[1] + v[2])
                                            if sum(v) > 0
                                            else 0.0
                                        ),
                                    }
                                    for label, v in counts.items()
                                },
                            }
                        )
                means = {
                    label: {
                        field: float(np.mean([run["labels"][label][field] for run in runs]))
                        for field in ("tp", "fp", "fn", "f1")
                    }
                    for label in counts
                }
                macro = float(
                    np.mean([means[label]["f1"] for label in info["endpoint"]["macro_labels"]])
                )
                a.require(
                    np.isclose(macro, c[side + "_mean_on_common_valid"], atol=1e-14, rtol=0),
                    "Paired label breakdown differs from primary scorer",
                )
                sides[side] = {
                    "system": system,
                    "runs": runs,
                    "mean_per_label": means,
                    "macro": macro,
                }
            result[subset + "/" + name] = {
                "n_common_valid": len(ids),
                "common_valid_ids": ids,
                **sides,
                "left_minus_right_label_means": {
                    label: {
                        field: sides["left"]["mean_per_label"][label][field]
                        - sides["right"]["mean_per_label"][label][field]
                        for field in ("tp", "fp", "fn", "f1")
                    }
                    for label in counts
                },
            }
    payload = {
        "created_utc": a.utc(),
        "status": "paired_primary_error_breakdowns_verified",
        "source_sha256": a.digest(ROOT / "analysis/knowledge/banks/diagnostics.py"),
        "primary_results_sha256": receipt["evidence_sha256"],
        "contrasts": result,
        "verified_hashes": hashes,
        "limits": "Descriptive label/error breakdown on the already-declared common-valid masks; no new significance tests. Counts averaged across decoder repeats, example-set repeats or training seeds as appropriate. No retuning or resampling predictions.",
        "actual_API_calls": 0,
    }
    a.save_new(OUT, payload)
    print(
        json.dumps(
            {
                key: {
                    "n": x["n_common_valid"],
                    "left": x["left"]["mean_per_label"],
                    "right": x["right"]["mean_per_label"],
                }
                for key, x in result.items()
            },
            indent=2,
        )
    )
    print("Evidence SHA256: " + a.digest(OUT))


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", p.forbidden), patch.object(
        socket, "create_connection", p.forbidden
    ):
        main()
