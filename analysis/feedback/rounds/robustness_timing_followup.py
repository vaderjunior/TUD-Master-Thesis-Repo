from pathlib import Path
from datetime import datetime
import numpy as np
from analysis.feedback.rounds import robustness_inputs as a


def main():
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    matrix = list(a.rows(a.READER / "matrix.jsonl"))
    ex = dict(np.load(a.OUT / "execution_arrays.npz", allow_pickle=False))
    spans = np.ptp(ex["completed_unix"], axis=(2, 3))
    flagged = np.argwhere(spans > 300)
    keys = {meta[i]["key"] for i, r in flagged}
    clusters = {meta[i]["inference_cluster"] for i, r in flagged}
    observed = []
    for i, rr in flagged:
        observed.append(
            dict(
                key=meta[i]["key"],
                input_index=int(i),
                round=int(rr + 1),
                population=meta[i]["population"],
                gold_hate=meta[i]["gold_hate"],
                inference_cluster=meta[i]["inference_cluster"],
                span_seconds=float(spans[i, rr]),
                batch=int(ex["stage"][i, rr, 0, 0]),
                all_round_common=meta[i]["masks"]["all_round_common"],
                outputs=[
                    dict(
                        condition=condition,
                        repeat=d,
                        completed_unix=float(ex["completed_unix"][i, rr, c, d]),
                        result=matrix[i]["predictions"][rr][c][d]["result"],
                    )
                    for c, condition in enumerate(a.CONDITIONS)
                    for d in range(3)
                ],
            )
        )
    rows = []
    for pop in a.POPS:
        for end in ["FPR", "FNR"]:
            base = [
                i
                for i, m in enumerate(meta)
                if m["population"] == pop
                and m["gold_hate"] == (end == "FNR")
                and m["masks"]["all_round_common"]
            ]
            for label, ids in [
                ("fixed", base),
                ("omit_timing_key", [i for i in base if meta[i]["key"] not in keys]),
                (
                    "omit_timing_cluster",
                    [i for i in base if meta[i]["inference_cluster"] not in clusters],
                ),
            ]:
                for rr in range(4):
                    rates = {}
                    for pol, cs in {
                        "reference": [0],
                        "adaptive": [1, 4, 7],
                        "static": [2, 5, 8],
                        "random": [3, 6, 9],
                    }.items():
                        errors = sum(
                            matrix[i]["predictions"][rr][c][d]["result"]["hate"]
                            != meta[i]["gold_hate"]
                            for i in ids
                            for c in cs
                            for d in range(3)
                        )
                        rates[pol] = dict(
                            errors=errors,
                            denominator=len(ids) * len(cs) * 3,
                            rate=errors / (len(ids) * len(cs) * 3),
                        )
                    rows.append(
                        dict(
                            population=pop,
                            endpoint=end,
                            view=label,
                            round=rr + 1,
                            n=len(ids),
                            removed=len(base) - len(ids),
                            policies=rates,
                            contrasts={
                                "adaptive-"
                                + p: rates["adaptive"]["rate"]
                                - rates[p]["rate"]
                                for p in ["static", "random", "reference"]
                            },
                        )
                    )
    runtimepath = (
        a.STUDY / "execution/concurrency_20261024_v1/live_runtime_complete.json"
    )
    runtime = a.read(runtimepath)
    markers = []
    bindings = a.bindings(
        [
            a.OUT / "details.json",
            a.OUT / "details_independent_verification.json",
            a.OUT / "execution_arrays.npz",
            a.READER / "matrix.jsonl",
            runtimepath,
        ]
    )
    for path, digest in runtime["marker_hashes"].items():
        if "/r1_evaluation/" in path and "/evaluation_01/" in path:
            p = a.ROOT / path
            a.need(a.sha(p) == digest, "Provenance marker changed")
            r = a.read(p)
            start = a.ROOT / r["original_attempt"] / "start.json"
            a.need(a.sha(start) == r["start_sha256"], "Attempt start changed")
            saved = a.read(start)
            markers.append(
                dict(
                    marker=path,
                    effective_workers=r["effective_workers"],
                    attempt=r["original_attempt"],
                    start=saved,
                )
            )
            bindings.update(a.bindings([p, start]))
    a.write(
        a.OUT / "timing_followup.json",
        dict(
            status="timing_outlier_followup_ready_for_final_readback",
            created_utc=a.pure.utc(),
            source_sha256=a.sha(Path(__file__)),
            inputs=bindings,
            rule="Post-generation follow-up after inspecting the timing table: report every within-input completion span above 300 seconds, selected by timing only. Omit the affected key and its whole inference cluster on fixed all-round cohorts; descriptive points, no new hypothesis or confidence interval.",
            flagged=observed,
            rows=rows,
            first_batch_attempts=markers,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        f"Timing follow-up saved: {len(flagged)} input-round spans above 300 seconds; no saved output changed.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()
