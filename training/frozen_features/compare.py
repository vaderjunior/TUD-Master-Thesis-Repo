import argparse
from pathlib import Path
import numpy as np
from analysis.common import study as c
from analysis.common import result_reader as s
from training.frozen_features import broader as b
from analysis.common.api import save_new


def prepare():
    b.validate()
    paths = [
        Path(__file__),
        Path(s.__file__),
        Path(c.__file__),
        b.BASE / "plan.json",
        c.ROOT / "analysis/common/scoring.py",
        c.ROOT / "training/original/validate.py",
        c.BASE / "encoders/plan.json",
    ]
    save_new(
        b.BASE / "paired_plan.json",
        {
            "created_utc": c.utc(),
            "status": "frozen_before_aligned_neural_predictions",
            "inputs": {c.rel(p): c.sha(p) for p in paths},
            "primary_family": list(b.PRIMARY),
            "policy": "Exact wider endpoint/gold contract from compact broad plan. Four primary BGE-minus-original-compact contrasts at 98.75%; full-data RoBERTa and other endpoints descriptive. 10000 paired text/template-cluster draws plus source-stratified/item sensitivities. Require complete independently assembled aligned neural outputs before scoring.",
        },
    )
    print(
        "Paired compact comparison implementation frozen; neural predictions pending.",
        flush=True,
    )


def main():
    paired_plan = c.read(b.BASE / "paired_plan.json")
    c.verify(paired_plan["inputs"])
    done = b.BASE / "paired_results.json"
    if done.exists():
        previous = c.read(done)
        c.verify(previous["inputs"])
        print("Verified completed paired comparison; skipped.", flush=True)
        return
    p = b.validate()
    receipt = c.read(b.BASE / "predictions_complete.json")
    c.verify(receipt["outputs"])
    path = c.BASE / "encoders/assembled/complete.json"
    neural = c.read(path)
    c.require(
        neural["status"] == "all_logits_verified_and_gated_predictions_assembled",
        "Aligned neural audit missing",
    )
    for key in ("inputs", "outputs", "source_sha256"):
        c.verify(neural[key])
    bindings = {
        c.rel(path): c.sha(path),
        c.rel(b.BASE / "paired_plan.json"): c.sha(b.BASE / "paired_plan.json"),
        c.rel(b.BASE / "plan.json"): c.sha(b.BASE / "plan.json"),
        c.rel(Path(__file__)): c.sha(Path(__file__)),
        **neural["outputs"],
        **receipt["outputs"],
    }
    predictions = {
        "bge": [
            {
                r["key"]: r["result"]
                for r in c.rows(b.BASE / f"predictions_seed{seed}.jsonl")
            }
            for seed in (42, 43, 44)
        ]
    }
    for arm in ("encoder_kb", "encoder_full"):
        predictions[arm] = [
            {
                r["key"]: r["result"]
                for r in c.rows(c.BASE / f"encoders/assembled/{arm}_seed{seed}.jsonl")
            }
            for seed in range(3)
        ]
    groups, _ = s.scientific_rosters("broad")
    results = {}
    for name, ep in p["endpoints"].items():
        phase, dim = name.split("/")
        rows = groups[phase]
        keys = [r["id"] for r in rows]
        byid = {r["id"]: r for r in rows}
        values = {
            arm: [s.contributions(rows, {k: out[k] for k in keys}, ep) for out in seeds]
            for arm, seeds in predictions.items()
        }
        ids = values["bge"][0]["ids"]
        clusters = [
            (
                byid[k]["functional_cluster"]
                if phase == "en_hatecheck_eval"
                else byid[k]["text_cluster"]
            )
            for k in ids
        ]
        sources = [byid[k]["source"] for k in ids]
        comparisons = {}
        for arm in ("encoder_kb", "encoder_full"):
            family = (
                4 if arm == "encoder_kb" and name in p["primary_contrasts"] else None
            )
            comparison = s.contrast(
                values["bge"],
                values[arm],
                clusters,
                sources=sources,
                family=family,
                draws=10000,
            )
            comparison["sensitivity"] = s.contrast(
                values["bge"],
                values[arm],
                ids if phase == "en_hatecheck_eval" else clusters,
                sources=sources,
                stratify=phase != "en_hatecheck_eval",
                draws=10000,
            )
            comparisons[arm] = comparison
        summaries = {
            arm: [s.summarize(v, rows, dim) for v in vv] for arm, vv in values.items()
        }
        results[name] = {
            "endpoint": ep,
            "runs": summaries,
            "means": {
                arm: float(np.mean([v["macro_f1_valid_only"] for v in vv]))
                for arm, vv in summaries.items()
            },
            "bge_minus": comparisons,
        }
        print("Paired compact comparison checked:", name, flush=True)
    save_new(
        b.BASE / "paired_results.json",
        {
            "created_utc": c.utc(),
            "status": "all_paired_compact_comparisons_complete",
            "inputs": bindings,
            "results": results,
            "primary_family": p["primary_contrasts"],
            "primary_confidence": 0.9875,
            "limits": p["exposure"],
            "new_model_calls": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["prepare", "score"], nargs="?", default="score"
    )
    args = parser.parse_args()
    with c.offline():
        prepare() if args.mode == "prepare" else main()
