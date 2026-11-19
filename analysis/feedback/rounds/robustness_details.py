from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from analysis.feedback.rounds import robustness_inputs as a


def main():
    proof = a.read(a.OUT / "independent_verification.json")
    a.need(
        proof["status"] == "part7_independent_verification_pass",
        "Independent robustness proof required",
    )
    analysis = a.read(a.OUT / "analysis.json")
    records = analysis["records"]
    ex = dict(np.load(a.OUT / "execution_arrays.npz", allow_pickle=False))
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    omission = defaultdict(list)
    for r in a.rows(a.OUT / "omissions.jsonl"):
        omission[(r["population"], r["endpoint"], r["view"], r["grouping"])].append(r)
    summaries = []
    for group, rs in omission.items():
        good = [r for r in rs if r["points"] is not None]
        for col in ["adaptive-static", "adaptive-random", "adaptive-reference"]:
            j = rs[0]["columns"].index(col)
            base = rs[0]["full_points"][j]
            ranked = sorted(
                good, key=lambda r: (-abs(r["points"][j] - base), r["group"])
            )
            values = [r["points"][j] for r in good]
            summaries.append(
                dict(
                    population=group[0],
                    endpoint=group[1],
                    view=group[2],
                    grouping=group[3],
                    contrast=col,
                    groups=len(rs),
                    defined=len(good),
                    full_point=base,
                    minimum=min(values) if values else None,
                    maximum=max(values) if values else None,
                    negative=sum(x < 0 for x in values),
                    zero=sum(x == 0 for x in values),
                    positive=sum(x > 0 for x in values),
                    largest_changes=[
                        dict(
                            group=r["group"],
                            removed=r["removed"],
                            point=r["points"][j],
                            shift=r["points"][j] - base,
                        )
                        for r in ranked[:5]
                    ],
                )
            )
    variation = []
    for pop in a.POPS:
        for endpoint in ["FPR", "FNR"]:
            for kind in ["stream", "repeat", "single_cell"]:
                rs = [
                    r
                    for r in analysis["stream_repeat_variation"]
                    if r["population"] == pop
                    and r["endpoint"] == endpoint
                    and r["omitted_kind"] == kind
                ]
                for col in ["adaptive-static", "adaptive-random", "adaptive-reference"]:
                    values = [r["contrasts"][col] for r in rs]
                    variation.append(
                        dict(
                            population=pop,
                            endpoint=endpoint,
                            kind=kind,
                            contrast=col,
                            n=rs[0]["n"],
                            values=values,
                            minimum=min(values),
                            maximum=max(values),
                            negative=sum(x < 0 for x in values),
                            zero=sum(x == 0 for x in values),
                            positive=sum(x > 0 for x in values),
                        )
                    )
    timing = []
    for rr in range(4):
        for pop in a.POPS:
            ids = [i for i, m in enumerate(meta) if m["population"] == pop]
            t = ex["completed_unix"][ids, rr]
            spans = t.max(axis=(1, 2)) - t.min(axis=(1, 2))
            top = sorted(
                range(len(ids)), key=lambda j: (-spans[j], meta[ids[j]]["key"])
            )[:5]
            timing.append(
                dict(
                    population=pop,
                    round=rr + 1,
                    inputs=len(ids),
                    completion_span_seconds_quantiles=np.quantile(
                        spans, [0, 0.5, 0.9, 0.99, 1]
                    ),
                    different_batch_within_input=sum(
                        len(set(ex["stage"][i, rr].ravel())) != 1 for i in ids
                    ),
                    largest_completion_spans=[
                        dict(
                            key=meta[ids[j]]["key"],
                            seconds=float(spans[j]),
                            batch=int(ex["stage"][ids[j], rr, 0, 0]),
                        )
                        for j in top
                    ],
                )
            )
    execution = a.read(a.OUT / "execution.json")
    ph = execution["groups"]
    sequence = []
    initial = ph["phase/source_initial"]
    prior_last = initial["last_unix"]
    for rr in range(1, 5):
        if rr > 1:
            for s in a.STREAMS:
                source = ph[f"phase/r{rr}_source_s{s}"]
                a.need(
                    source["first_unix"] > prior_last,
                    "Source phase precedes completed prior evaluation",
                )
            prior_last = max(
                ph[f"phase/r{rr}_source_s{s}"]["last_unix"] for s in a.STREAMS
            )
        evaluation = ph[f"phase/r{rr}_evaluation"]
        a.need(
            evaluation["first_unix"] > prior_last,
            "Evaluation precedes completed source screening",
        )
        sequence.append(
            dict(
                round=rr,
                last_source_completion_unix=prior_last,
                first_evaluation_completion_unix=evaluation["first_unix"],
                last_evaluation_completion_unix=evaluation["last_unix"],
            )
        )
        prior_last = evaluation["last_unix"]

    duplicate_records = []
    dups = a.read(a.READER / "duplicate_vote_labels.json")
    a.need(len(dups) == 4, "Duplicate-label inventory changed")
    for row in dups:
        result_path = (a.ROOT / row["item_receipt"]).with_name("result.json")
        a.need(
            a.sha(result_path) == row["result_sha256"], "Duplicate-label result changed"
        )
        saved = a.read(result_path)
        valid = [v for v in saved["run_predictions"] if v is not None]
        n = len(valid)
        a.need(n > 0, "Duplicate-label case has no vote")
        yes = sum(v["hate"] for v in valid)
        gate = yes * 2 > n
        counter = Counter(label for v in valid for label in set(v[row["field"]]))
        alt = (
            sorted(label for label, count in counter.items() if count * 2 > n)
            if gate
            else []
        )
        actual = saved["result"][row["field"]]
        a.need(
            saved["run_predictions"][row["vote"]][row["field"]] == row["labels"],
            "Duplicate labels changed",
        )
        duplicate_records.append(
            dict(
                **row,
                n_valid=n,
                gold_known=next(m for m in meta if m["key"] == row["key"])[
                    "known_fine_gold"
                ]["target_groups"],
                actual_final_gate=gate,
                actual_final_labels=actual,
                hypothetical_vote_deduplicated_labels=alt,
                changed_labels=actual != alt,
                binary_changed=False,
                note="Counterfactual per-vote deduplication only; no saved prediction, final scoring or protocol changed.",
            )
        )
    family = [
        dict(record=name, contrast=c, point=r["points"][c], interval=iv)
        for name, r in records.items()
        if r["family_level_sensitivity"] is not None
        for c, iv in r["family_level_sensitivity"].items()
    ]
    a.write(
        a.OUT / "details.json",
        dict(
            created_utc=a.pure.utc(),
            omission_summaries=summaries,
            variation_summaries=variation,
            timing=timing,
            phase_completion_order=sequence,
            duplicate_label_sensitivity=duplicate_records,
            exploratory_9875_intervals=family,
            exploratory_9875_excluding_zero=sum(
                r["interval"][0] > 0 or r["interval"][1] < 0 for r in family
            ),
            replay_keys=a.read(a.READER / "complete.json")["replay_keys"],
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    a.write(
        a.OUT / "details_complete.json",
        dict(
            status="part7_reporting_details_ready",
            created_utc=a.pure.utc(),
            source_sha256=a.sha(Path(__file__)),
            inputs=a.bindings(
                [
                    a.OUT / f
                    for f in [
                        "independent_verification.json",
                        "analysis.json",
                        "omissions.jsonl",
                        "execution_arrays.npz",
                        "execution.json",
                    ]
                ]
                + [a.READER / "duplicate_vote_labels.json"]
            ),
            artifacts=a.bindings([a.OUT / "details.json"]),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "Reporting diagnostics and four duplicate-label counterfactuals saved.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()
