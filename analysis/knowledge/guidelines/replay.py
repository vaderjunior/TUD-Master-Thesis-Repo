"""Recalculate the saved English guideline results without a model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.hsrag.paths import project_path
from . import metrics
from .design import CONDITIONS, PRIMARY, SECONDARY


def analyze(matrix, draws=10000):
    if draws < 1:
        raise ValueError("draws must be positive")
    if (
        matrix["conditions"] != list(CONDITIONS)
        or matrix["repeats"] != [0, 1, 2]
        or matrix["missing_value"] != -1
    ):
        raise ValueError("The saved condition, repeat or missing-value definition differs")
    items = matrix["items"]
    if not items or len({item["key"] for item in items}) != len(items):
        raise ValueError("Input identities must be nonempty and unique")
    if any(item["gate"] is not False for item in items):
        raise ValueError("This study evaluates known-benign diagnostic texts")
    gates = metrics.validate_gates(matrix["values"])
    if gates.shape != (len(items), len(CONDITIONS), 3):
        raise ValueError("Prediction matrix shape differs from its roster")
    common = metrics.common_mask(gates)
    rng = np.random.default_rng(20260925)
    scopes = {}
    groups = [("all", np.arange(len(items)))]
    groups.extend(
        (name, np.array([i for i, item in enumerate(items) if item["functionality"] == name]))
        for name in sorted({item["functionality"] for item in items})
    )
    for name, indices in groups:
        summary = metrics.counts_and_rates(gates[indices], list(CONDITIONS))
        selected = indices[common[indices]]
        templates = [items[i]["templ_id"] for i in selected]
        unique, _ = metrics.cluster_structure(templates)
        summary.update(
            scope=name,
            roster_templates=len({items[i]["templ_id"] for i in indices}),
            common_templates=len(unique),
            excluded_keys=[items[i]["key"] for i in indices if not common[i]],
        )
        if len(selected):
            picks = rng.integers(0, len(unique), size=(draws, len(unique)), dtype=np.int32)
            rates = metrics.cluster_rates(gates[selected], templates, picks)
            summary["contrasts"] = metrics.contrasts(
                summary,
                rates,
                list(CONDITIONS),
                {**PRIMARY, **SECONDARY},
                PRIMARY,
                0.9875,
                name == "all",
            )
            summary["mean_rate_descriptive_95"] = {
                condition: metrics.interval(rates[:, i], 0.95)
                for i, condition in enumerate(CONDITIONS)
            }
            summary["draws_sha256"] = hashlib.sha256(picks.astype("<i4").tobytes()).hexdigest()
        else:
            summary["contrasts"] = {}
        scopes[name] = summary
    return {
        "method": {
            "seed": 20260925,
            "draws": draws,
            "unit": "template",
            "estimate": "item-weighted false-positive rate, averaged over three fixed repeats",
            "mask": "valid in every condition and repeat",
            "primary_family_size": 4,
            "primary_family_confidence": 0.9875,
            "quantiles": "linear",
            "scope_order": list(scopes),
        },
        "scopes": scopes,
        "source_input_items": len(items),
        "scorable_items": int(common.sum()),
        "missing_predictions": int((gates == -1).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay the English guideline prediction matrix.")
    parser.add_argument(
        "--matrix", type=Path, default=project_path("data/evaluations/english_guidelines.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new file")
    if not args.matrix.is_file():
        parser.error("Prediction matrix is missing: " + str(args.matrix))
    raw = args.matrix.read_bytes()
    report = analyze(json.loads(raw), args.draws)
    report["matrix_sha256"] = hashlib.sha256(raw).hexdigest()
    report["source_sha256"] = {
        "metrics": hashlib.sha256(Path(metrics.__file__).read_bytes()).hexdigest(),
        "replay": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
