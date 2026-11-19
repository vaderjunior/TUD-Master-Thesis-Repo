from src.hsrag.paths import PROJECT_ROOT
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.metrics import f1_score

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/pooled_fineheads_20260930_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def close(x, y):
    assert np.allclose(x, y, atol=1e-12, rtol=0), (x, y)


def run():
    root_receipt = read(BASE / "complete.json")
    attempt = ROOT / root_receipt["attempt"]
    assert sha(attempt / "complete.json") == root_receipt["receipt_sha256"]
    receipt = read(attempt / "complete.json")
    assert sha(BASE / "plan.json") == receipt["plan_sha256"]
    for name, expected in receipt["outputs"].items():
        assert sha(attempt / name) == expected
    inputs = read(attempt / "inputs.json")
    for name, expected in inputs.items():
        assert sha(ROOT / name) == expected
    plan = read(BASE / "plan.json")
    rosters = read(BASE / "rosters.json")
    original_exports = {}
    for name, path in plan["exports"].items():
        original_exports[name] = {
            row.get("id", row.get("item_id")): row
            for row in [
                json.loads(line)
                for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
                if line
            ]
        }
    grouped = defaultdict(list)
    identities = set()
    exported_rows = 0
    with (attempt / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["dimension"], row["condition"], row["repeat_or_seed"]
            assert (*key, row["id"]) not in identities
            identities.add((*key, row["id"]))
            source_name = (
                f"{row['condition']}/{row['repeat_or_seed']}"
                if row["condition"].startswith("encoder")
                else f"{row['origin']}/{row['condition']}/{row['repeat_or_seed']}"
            )
            source = original_exports[source_name][row["id"]]["result"]
            actual = row["prediction"]
            assert actual["valid"] == (source is not None)
            assert actual["gate"] == (None if source is None else source["hate"])
            value = None if source is None else source[row["dimension"]]
            labels = (
                value
                if row["dimension"] == "target_group"
                else [value] if value is not None else []
            )
            assert actual["labels"] == (labels or [])
            grouped[key].append(row)
            exported_rows += 1
    assert exported_rows == receipt["exported_rows"] == 4185
    result = read(attempt / "results.json")
    saved_draws = np.load(attempt / "bootstrap_differences.npz", allow_pickle=False)
    contrasts_checked = 0
    metric_checks = 0
    for dim, endpoint in result["endpoints"].items():
        roster = rosters[dim]
        ids = [row["id"] for row in roster]
        labels = endpoint["roster"]["labels"]
        gold = np.array(
            [[label in row["gold"] for label in labels] for row in roster], dtype=bool
        )
        arrays = {}
        for condition, details in endpoint["conditions"].items():
            fvalues = []
            for repeat, run in enumerate(details["runs"]):
                rows = grouped[dim, condition, repeat]
                assert [row["id"] for row in rows] == ids
                assert all(
                    row["gold"] == expected["gold"]
                    and row["cluster"] == expected["cluster"]
                    for row, expected in zip(rows, roster)
                )
                valid = np.array([row["prediction"]["valid"] for row in rows])
                pred = np.array(
                    [
                        [label in row["prediction"]["labels"] for label in labels]
                        for row in rows
                    ],
                    dtype=bool,
                )
                arrays[condition, repeat] = valid, pred
                computed = float(
                    f1_score(gold[valid], pred[valid], average="macro", zero_division=0)
                )
                close(computed, run["own_valid"]["macro_f1"])
                assert int(valid.sum()) == run["own_valid"]["valid"]
                assert int((~valid).sum()) == run["own_valid"]["null"]
                for li, label in enumerate(labels):
                    tp = int((gold[valid, li] & pred[valid, li]).sum())
                    fp = int((~gold[valid, li] & pred[valid, li]).sum())
                    fn = int((gold[valid, li] & ~pred[valid, li]).sum())
                    assert [tp, fp, fn] == [
                        run["own_valid"]["per_label"][label][key]
                        for key in ["tp", "fp", "fn"]
                    ]
                fvalues.append(computed)
                metric_checks += 1
            close(fvalues, details["own_valid_values"])
            close(np.mean(fvalues), details["own_valid_mean"])
        for name, contrast in endpoint["contrasts"].items():
            settings = [
                (c, r) for c in [contrast["left"], contrast["right"]] for r in range(3)
            ]
            common = np.stack([arrays[k][0] for k in settings]).all(axis=0)
            kept = np.flatnonzero(common)

            ordered = sorted(kept.tolist(), key=lambda i: roster[i]["cluster"])
            assert (
                len(ordered)
                == len({roster[i]["cluster"] for i in ordered})
                == contrast["clusters"]
            )
            y = gold[ordered]
            x = np.stack([arrays[k][1][ordered] for k in settings], axis=1)
            rng = np.random.default_rng(contrast["seed"])
            indexes = rng.integers(
                0, len(ordered), size=(contrast["draws"], len(ordered))
            )
            delta = []
            for start in range(0, contrast["draws"], 250):
                ix = indexes[start : start + 250]
                truth, pred = y[ix, None, :], x[ix]
                tp = (truth & pred).sum(axis=1)
                fp = (~truth & pred).sum(axis=1)
                fn = (truth & ~pred).sum(axis=1)
                denominator = 2 * tp + fp + fn
                score = np.divide(
                    2 * tp,
                    denominator,
                    out=np.zeros_like(tp, dtype=float),
                    where=denominator != 0,
                ).mean(axis=-1)
                delta.extend(
                    (score[:, :3].mean(axis=1) - score[:, 3:].mean(axis=1)).tolist()
                )
            close(delta, saved_draws[dim + "__" + name])
            close(np.quantile(delta, [0.025, 0.975]), contrast["ci95"])
            if contrast["family_ci975"] is not None:
                close(np.quantile(delta, [0.0125, 0.9875]), contrast["family_ci975"])
            contrasts_checked += 1
    for name, expected in inputs.items():
        assert sha(ROOT / name) == expected
    summary = {
        "status": "pooled_fineheads_independently_verified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(Path(__file__)),
        "plan_sha256": sha(BASE / "plan.json"),
        "results_receipt_sha256": sha(attempt / "complete.json"),
        "verified_inputs": len(inputs),
        "verified_exported_rows": exported_rows,
        "own_valid_metric_checks": metric_checks,
        "all_draws_verified": contrasts_checked * 10000,
        "contrast_intervals_checked": contrasts_checked,
        "new_model_calls": 0,
        "raw_response_replay": False,
    }
    with (BASE / "independent_check_v1.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(summary, flush=True)


if __name__ == "__main__":
    with patch.object(
        socket.socket, "connect", side_effect=RuntimeError("Offline verification")
    ):
        run()
