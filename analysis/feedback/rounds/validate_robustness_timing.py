from fractions import Fraction
from pathlib import Path
from datetime import datetime
import numpy as np
from analysis.feedback.rounds import robustness_inputs as a


def main():
    saved = a.read(a.OUT / "timing_followup.json")
    a.check(saved["inputs"])
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    matrix = list(a.rows(a.READER / "matrix.jsonl"))
    with np.load(a.OUT / "execution_arrays.npz", allow_pickle=False) as arrays:
        times = arrays["completed_unix"]
        batches = arrays["stage"]
    expected = []
    for i, m in enumerate(meta):
        a.need(m["key"] == matrix[i]["key"], "Matrix order changed")
        for rr in range(4):
            values = [float(times[i, rr, c, d]) for c in range(10) for d in range(3)]
            span = max(values) - min(values)
            if span > 300:
                expected.append((i, rr, span))
    a.need(len(expected) == len(saved["flagged"]) == 1, "Timing outlier set changed")
    for (i, rr, span), item in zip(expected, saved["flagged"]):
        a.need(
            (item["input_index"], item["round"], item["key"], item["inference_cluster"])
            == (i, rr + 1, meta[i]["key"], meta[i]["inference_cluster"]),
            "Outlier identity differs",
        )
        a.need(
            item["span_seconds"] == span and item["batch"] == int(batches[i, rr, 0, 0]),
            "Span or batch differs",
        )
        a.need(len(item["outputs"]) == 30, "Output coverage differs")
        for j, o in enumerate(item["outputs"]):
            c, d = divmod(j, 3)
            a.need(
                o["condition"] == a.CONDITIONS[c]
                and o["repeat"] == d
                and o["completed_unix"] == float(times[i, rr, c, d]),
                "Output timing differs",
            )
            a.need(
                o["result"] == matrix[i]["predictions"][rr][c][d]["result"],
                "Saved answer differs",
            )
    keys = {meta[i]["key"] for i, rr, span in expected}
    clusters = {meta[i]["inference_cluster"] for i, rr, span in expected}
    seen = set()
    estimates = 0
    for item in saved["rows"]:
        pop, end, view, rr = (
            item[x] for x in ["population", "endpoint", "view", "round"]
        )
        signature = (pop, end, view, rr)
        a.need(signature not in seen, "Duplicate timing sensitivity")
        seen.add(signature)
        base = [
            i
            for i, m in enumerate(meta)
            if m["population"] == pop
            and m["gold_hate"] == (end == "FNR")
            and m["masks"]["all_round_common"]
        ]
        ids = [
            i
            for i in base
            if view == "fixed"
            or (view == "omit_timing_key" and meta[i]["key"] not in keys)
            or (
                view == "omit_timing_cluster"
                and meta[i]["inference_cluster"] not in clusters
            )
        ]
        a.need(
            len(ids) == item["n"] and len(base) - len(ids) == item["removed"],
            "Timing cohort differs",
        )
        rates = {}
        for policy, columns in [
            ("reference", (0,)),
            ("adaptive", (1, 4, 7)),
            ("static", (2, 5, 8)),
            ("random", (3, 6, 9)),
        ]:
            count = 0
            for i in ids:
                for c in columns:
                    for d in range(3):
                        gate = matrix[i]["predictions"][rr - 1][c][d]["result"]["hate"]
                        a.need(type(gate) is bool, "Invalid vote entered fixed cohort")
                        if gate != meta[i]["gold_hate"]:
                            count += 1
            denominator = len(ids) * len(columns) * 3
            rate = Fraction(count, denominator)
            rates[policy] = rate
            result = item["policies"][policy]
            a.need(
                result["errors"] == count
                and result["denominator"] == denominator
                and result["rate"] == float(rate),
                "Timing policy count differs",
            )
            estimates += 1
        for control in ["static", "random", "reference"]:
            a.need(
                abs(
                    item["contrasts"]["adaptive-" + control]
                    - float(rates["adaptive"] - rates[control])
                )
                < 2e-16,
                "Timing contrast differs",
            )
            estimates += 1
    a.need(len(seen) == 48 and estimates == 336, "Timing sensitivity coverage differs")
    a.need(len(saved["first_batch_attempts"]) == 2, "Attempt history differs")
    starts = []
    for item in saved["first_batch_attempts"]:
        marker = a.read(a.ROOT / item["marker"])
        path = a.ROOT / item["attempt"] / "start.json"
        start = a.read(path)
        a.need(
            a.sha(path) == marker["start_sha256"] and start == item["start"],
            "Attempt start differs",
        )
        a.need(
            marker["effective_workers"] == item["effective_workers"] == 128,
            "Worker provenance differs",
        )
        starts.append(datetime.fromisoformat(start["started_utc"]).timestamp())
    times_of_flagged = [o["completed_unix"] for o in saved["flagged"][0]["outputs"]]
    a.need(
        min(starts) < min(times_of_flagged) < max(starts) < max(times_of_flagged),
        "Outlier does not straddle the resumed attempt",
    )
    final = next(
        r
        for r in saved["rows"]
        if (r["population"], r["endpoint"], r["view"], r["round"])
        == ("en_hatecheck_eval", "FNR", "omit_timing_cluster", 4)
    )
    a.need(
        final["removed"] == 18 and final["contrasts"]["adaptive-reference"] == 0,
        "Influential-component result differs",
    )
    a.write(
        a.OUT / "timing_independent_verification.json",
        dict(
            status="part7_timing_independent_verification_pass",
            created_utc=a.pure.utc(),
            source_sha256=a.sha(Path(__file__)),
            inputs=a.bindings(
                [
                    a.OUT / "timing_followup.json",
                    a.OUT / "metadata.jsonl",
                    a.OUT / "execution_arrays.npz",
                    a.READER / "matrix.jsonl",
                ]
            ),
            input_rounds_checked=len(meta) * 4,
            outliers=1,
            outputs=30,
            sensitivity_rows=48,
            exact_estimates=estimates,
            attempt_starts=2,
            whole_cluster_removed=18,
            new_Qwen_calls=0,
            new_fits=0,
        ),
    )
    print(
        "Independent timing readback PASS: 8,308 input-rounds, one outlier, 336 estimates, both saved attempts.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()
