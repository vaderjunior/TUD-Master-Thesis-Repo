from collections import defaultdict
import hashlib
from pathlib import Path
import socket
from types import FunctionType
from unittest.mock import patch
import numpy as np
from analysis.feedback.insertion import archive as r
from analysis.feedback.supplements import scores as base
from analysis.feedback.balanced import scores as balanced
from analysis.feedback.insertion import scores as frozen

POLICIES = balanced.POLICIES
GROUPS = {
    "reference": ["reference"],
    **{p: [f"{p}_s{s}" for s in r.SEEDS] for p in POLICIES[1:]},
}


class Data(base.Data):
    __init__ = FunctionType(
        base.Data.__init__.__code__, vars(base) | {"STUDIES": r.STUDIES}
    )


def representatives(data, ids):
    clusters = defaultdict(list)
    for k in ids:
        clusters[data.roster[k]["cluster"]].append(k)
    selected = {
        min(
            v,
            key=lambda k: hashlib.sha256(
                ("A14-representative|" + k).encode()
            ).hexdigest(),
        )
        for v in clusters.values()
    }
    return [k for k in ids if k in selected]


def deletion_diagnostics(data, primary):
    result = {}
    for name in ["benign_false_positive_rate", "hateful_false_negative_rate"]:
        scored = primary[name + "/primary"]
        ids = scored["common_ids"]
        truth = scored["gold_hate"]
        by_pool = []
        by_batch = []

        def rates(chosen, seeds):
            pos = data.indices(chosen)
            correction = [
                data.kpos[f"error_balanced_s{s}", rep]
                for s in seeds
                for rep in range(3)
            ]
            reference = [data.kpos["reference", rep] for rep in range(3)]
            values = {
                p: float((data.gate[np.ix_(kk, pos)] != truth).mean())
                for p, kk in [("error_balanced", correction), ("reference", reference)]
            }
            return dict(
                common_n=len(chosen),
                rates=values,
                difference=values["error_balanced"] - values["reference"],
            )

        for seed in r.SEEDS:
            by_pool.append(
                dict(
                    omitted_pool_seed=seed,
                    **rates(ids, [s for s in r.SEEDS if s != seed]),
                )
            )
        for stage in sorted({data.records[k, "reference", 0]["stage"] for k in ids}):
            chosen = [
                k for k in ids if data.records[k, "reference", 0]["stage"] != stage
            ]
            by_batch.append(dict(omitted_stage=stage, **rates(chosen, r.SEEDS)))
        result[name] = dict(
            leave_one_pool_out=by_pool,
            leave_one_batch_out=by_batch,
            scope="Supporting point estimates on fixed primary common masks; pools/repeats are not independent cases.",
        )
    return result


def main():
    done = r.read(r.REVIEW / "complete.json")
    r.verify(done["inputs"])
    r.verify(done["outputs"])
    complete = r.DEST / "scoring_complete.json"
    if complete.exists():
        r.verify(r.read(complete)["inputs"])
        r.verify(r.read(complete)["outputs"])
        print("Existing A14 scores verified.")
        return
    sources = {r.rel(Path(__file__)): r.sha(Path(__file__))}
    r.save(
        r.DEST / "scoring_plan.json",
        dict(
            created_utc=r.utc(),
            analysis_plan_sha256=r.sha(r.DEST / "analysis_plan.json"),
            inputs=sources,
            scope="Implement frozen A14 primaries and the recorded supporting checks on full and common-six-excluded rosters. No changed original source or model call.",
        ),
    )
    records = list(r.rows(r.REVIEW / "predictions.jsonl"))
    data = Data("five_slot", records)
    roster = list(r.rows(r.EXEC.parent / "roster.jsonl"))
    science = r.read(r.EXEC.parent / "scientific_plan_v1.json")
    exclusion = r.read(r.EXEC.parent / "cold_replay_review_v1/sensitivity_plan.json")[
        "excluded_keys"
    ]
    outputs = {}

    def put(name, value):
        path = r.DEST / (name + ".json")
        r.save(path, value)
        outputs[r.rel(path)] = r.sha(path)
        print(
            "Scored",
            name,
            (
                f"{value['common_n']}/{value['original_n']} common"
                if "common_n" in value
                else ""
            ),
            flush=True,
        )

    for scope, excluded in [("full", []), ("exclude_replay", exclusion)]:
        scores = frozen.sensitivity_scores(
            roster, data.records, science, excluded, science["bootstrap_draws"]
        )
        put(scope + "/primary", scores)
        eligible = [k for k in data.ids if k not in excluded]
        for truth, label, seed in [
            (False, "benign", 2026101214),
            (True, "hateful", 2026101215),
        ]:
            ids = [k for k in eligible if data.roster[k]["gold"]["hate"] == truth]
            for suffix, strict in [
                ("all_policies", False),
                ("all_policies_strict", True),
            ]:
                put(
                    scope + "/" + label + "/" + suffix,
                    balanced.gate(data, ids, GROUPS, seed, strict),
                )
        benign = [k for k in eligible if not data.roster[k]["gold"]["hate"]]
        put(
            scope + "/benign/source_stratified",
            balanced.gate(data, benign, GROUPS, 2026101216, stratified=True),
        )
        for index, source in enumerate(
            sorted({data.roster[k]["source"] for k in benign})
        ):
            ids = [k for k in benign if data.roster[k]["source"] == source]
            put(
                scope + "/benign/source_" + source,
                balanced.gate(data, ids, GROUPS, 2026101220 + index),
            )
        for suffix, strict in [("all_policies", False), ("all_policies_strict", True)]:
            put(
                scope + "/binary/" + suffix,
                balanced.binary_compare(data, eligible, GROUPS, 2026101217, strict),
            )
        hateful = [k for k in eligible if data.roster[k]["gold"]["hate"]]
        for suffix, ids, strict in [
            ("all_policies", hateful, False),
            ("all_policies_strict", hateful, True),
            ("representatives", representatives(data, hateful), False),
        ]:
            put(
                scope + "/types/" + suffix,
                base.human_compare(data, ids, GROUPS, 2026101218, strict),
            )
        put(
            scope + "/supporting_deletions",
            deletion_diagnostics(data, scores["primary"]),
        )
    r.verify(sources)
    r.save(
        complete,
        dict(
            created_utc=r.utc(),
            status="scored_requires_independent_check",
            inputs={
                **sources,
                r.rel(r.REVIEW / "complete.json"): r.sha(r.REVIEW / "complete.json"),
                **done["outputs"],
                r.rel(r.DEST / "analysis_plan.json"): r.sha(
                    r.DEST / "analysis_plan.json"
                ),
                r.rel(r.DEST / "scoring_plan.json"): r.sha(
                    r.DEST / "scoring_plan.json"
                ),
            },
            outputs=outputs,
            new_model_calls=0,
        ),
    )


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with patch.object(socket.socket, "connect", r.forbid), patch.object(
        socket, "create_connection", r.forbid
    ), threadpool_limits(limits=1):
        main()
