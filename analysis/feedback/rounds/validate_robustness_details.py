from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import ast
import hashlib
import json
import re
import numpy as np
from analysis.feedback.rounds import robustness_inputs as a


def main():
    complete = a.read(a.OUT / "failure_review_complete.json")
    details_complete = a.read(a.OUT / "details_complete.json")
    for done in [complete, details_complete]:
        a.check(done["inputs"])
        a.check(done["artifacts"])
    records = list(a.rows(a.OUT / "invalid_vote_details.jsonl"))
    a.need(len(records) == 8220, "Invalid-vote count")
    summary = a.read(a.OUT / "failure_review.json")
    groups = defaultdict(list)
    seen = set()
    previous_receipt = None
    receipt = None
    for j, r in enumerate(records):
        a.need(r["original_vote"] not in seen, "Duplicate vote detail")
        seen.add(r["original_vote"])
        path = a.ROOT / r["original_vote"]
        a.need(a.sha(path) == r["original_vote_sha256"], "Original vote changed")
        v = a.read(path)
        if r["item_receipt"] != previous_receipt:
            previous_receipt = r["item_receipt"]
            a.need(
                a.sha(a.ROOT / previous_receipt) == r["item_receipt_sha256"],
                "Original item changed",
            )
            receipt = a.read(a.ROOT / previous_receipt)
        key = f'items/{path.parent.parent.name}/votes/{r["vote"]}.json'
        a.need(
            receipt["hashes"][key] == r["original_vote_sha256"],
            "Vote not bound to item",
        )
        a.need(
            v["result"] is None
            and v["parse_failure"]
            and v["repairs"] == r["repairs"] == 2,
            "Vote no longer terminal-invalid",
        )
        error = str(v["error"])
        a.need(error == r["error"], "Final error text differs")
        matched = re.search(r" at (\(.*\))$", error)
        fieldpath = ast.literal_eval(matched[1]) if matched else None
        a.need(
            r["path"] == (list(fieldpath) if fieldpath else None)
            and r["field"] == (str(fieldpath[0]) if fieldpath else "unresolved"),
            "Error field/path differs",
        )
        a.need(
            r["category"]
            == (
                "schema_violation"
                if error.startswith("schema violation:")
                else "json_or_other_validation"
            ),
            "Error category differs",
        )
        texts = v["raw_outputs"]
        a.need(
            r["raw_turns"] == len(texts) == 3
            and r["raw_text_sha256"]
            == [hashlib.sha256(t.encode("utf8")).hexdigest() for t in texts],
            "Raw text digest differs",
        )
        a.need(
            r["distinct_texts"] == len(set(texts))
            and r["identical_repair_texts"] == all(t == texts[0] for t in texts),
            "Repair repetition differs",
        )
        ok = False
        scalar = None
        try:
            parsed = json.JSONDecoder().decode(texts[-1])
            ok = True
            if fieldpath:
                scalar = parsed
                for key in fieldpath:
                    scalar = scalar[key]
        except (ValueError, KeyError, IndexError, TypeError):
            scalar = None
        if scalar is not None and (
            not isinstance(scalar, (str, int, float, bool)) or len(str(scalar)) > 120
        ):
            scalar = "<noncompact value>"
        a.need(
            r["strict_json_last_output"] == ok and r["rejected_scalar"] == scalar,
            "Rejected scalar diagnostic differs",
        )
        for name in [
            "all",
            "role/" + r["role"],
            "phase/" + r["phase"],
            "aggregate/" + ("null" if r["null_aggregate"] else "partial"),
        ]:
            groups[name].append(r)
        if (j + 1) % 2000 == 0:
            print(f"Independent invalid-vote review: {j+1}/8220", flush=True)
    for name, rs in groups.items():
        g = summary[name]
        a.need(
            g["votes"] == len(rs) and g["inputs"] == len({r["key"] for r in rs}),
            "Invalid-vote group size differs",
        )
        for dest, src in [
            ("categories", "category"),
            ("fields", "field"),
            ("errors", "error"),
            ("repairs", "repairs"),
            ("turns", "raw_turns"),
            ("distinct_texts", "distinct_texts"),
        ]:
            a.need(
                g[dest] == dict(Counter(str(r[src]) for r in rs)),
                "Invalid-vote grouped values differ",
            )
        a.need(
            g["rejected_scalars"]
            == dict(
                Counter(
                    json.dumps(r["rejected_scalar"], sort_keys=True, ensure_ascii=False)
                    for r in rs
                    if r["rejected_scalar"] is not None
                )
            ),
            "Rejected scalar counts differ",
        )
        a.need(
            g["identical_repair_texts"] == sum(r["identical_repair_texts"] for r in rs)
            and g["strict_json"] == sum(r["strict_json_last_output"] for r in rs),
            "Repair pattern totals differ",
        )
    details = a.read(a.OUT / "details.json")
    analysis = a.read(a.OUT / "analysis.json")
    omissions = list(a.rows(a.OUT / "omissions.jsonl"))
    for s in details["omission_summaries"]:
        matches = [
            r
            for r in omissions
            if all(r[f] == s[f] for f in ["population", "endpoint", "view", "grouping"])
        ]
        good = [r for r in matches if r["points"] is not None]
        j = matches[0]["columns"].index(s["contrast"])
        values = [r["points"][j] for r in good]
        a.need(
            s["groups"] == len(matches) and s["defined"] == len(good),
            "Omission-summary support differs",
        )
        a.need(
            s["minimum"] == (min(values) if values else None)
            and s["maximum"] == (max(values) if values else None),
            "Omission extrema differ",
        )
        a.need(
            [s["negative"], s["zero"], s["positive"]]
            == [
                sum(x < 0 for x in values),
                sum(x == 0 for x in values),
                sum(x > 0 for x in values),
            ],
            "Omission signs differ",
        )
        expected = sorted(
            good, key=lambda r: (-abs(r["points"][j] - s["full_point"]), r["group"])
        )[:5]
        a.need(
            s["largest_changes"]
            == [
                dict(
                    group=r["group"],
                    removed=r["removed"],
                    point=r["points"][j],
                    shift=r["points"][j] - s["full_point"],
                )
                for r in expected
            ],
            "Omission influential groups differ",
        )
    for s in details["variation_summaries"]:
        rs = [
            r
            for r in analysis["stream_repeat_variation"]
            if r["population"] == s["population"]
            and r["endpoint"] == s["endpoint"]
            and r["omitted_kind"] == s["kind"]
        ]
        values = [r["contrasts"][s["contrast"]] for r in rs]
        a.need(
            values == s["values"]
            and min(values) == s["minimum"]
            and max(values) == s["maximum"],
            "Variation summary differs",
        )
    ex = dict(np.load(a.OUT / "execution_arrays.npz", allow_pickle=False))
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    for s in details["timing"]:
        ids = [i for i, m in enumerate(meta) if m["population"] == s["population"]]
        rr = s["round"] - 1
        spans = [
            max(ex["completed_unix"][i, rr].ravel())
            - min(ex["completed_unix"][i, rr].ravel())
            for i in ids
        ]
        a.need(
            np.array_equal(
                np.quantile(spans, [0, 0.5, 0.9, 0.99, 1]),
                s["completion_span_seconds_quantiles"],
            ),
            "Completion span differs",
        )
        a.need(
            s["different_batch_within_input"]
            == sum(len(set(ex["stage"][i, rr].ravel())) != 1 for i in ids)
            == 0,
            "Input batch pairing differs",
        )
    duplicate_changes = 0
    for r in details["duplicate_label_sensitivity"]:
        p = (a.ROOT / r["item_receipt"]).with_name("result.json")
        a.need(a.sha(p) == r["result_sha256"], "Duplicate vote result changed")
        saved = a.read(p)
        valid = [v for v in saved["run_predictions"] if v is not None]
        all_labels = set(x for v in valid for x in v[r["field"]])
        new = (
            sorted(
                x
                for x in all_labels
                if sum(x in v[r["field"]] for v in valid) > len(valid) / 2
            )
            if saved["result"]["hate"]
            else []
        )
        a.need(
            saved["result"][r["field"]] == r["actual_final_labels"]
            and new == r["hypothetical_vote_deduplicated_labels"]
            and (new != r["actual_final_labels"]) == r["changed_labels"],
            "Duplicate-vote counterfactual differs",
        )
        a.need(
            saved["result"]["hate"] == r["actual_final_gate"]
            and not r["binary_changed"],
            "Duplicate-vote binary claim differs",
        )
        duplicate_changes += r["changed_labels"]
    family = [
        (name, col, iv)
        for name, r in analysis["records"].items()
        if r["family_level_sensitivity"] is not None
        for col, iv in r["family_level_sensitivity"].items()
    ]
    a.need(
        len(family) == len(details["exploratory_9875_intervals"])
        and sum(iv[0] > 0 or iv[1] < 0 for _, _, iv in family)
        == details["exploratory_9875_excluding_zero"],
        "Exploratory family-level summary differs",
    )
    a.write(
        a.OUT / "details_independent_verification.json",
        dict(
            status="part7_details_independent_verification_pass",
            created_utc=a.pure.utc(),
            source_sha256=a.sha(Path(__file__)),
            inputs=a.bindings(
                [
                    a.OUT / "failure_review_complete.json",
                    a.OUT / "details_complete.json",
                ]
            ),
            invalid_votes=8220,
            distinct_vote_files=len(seen),
            null_aggregate_votes=726,
            omission_summaries=len(details["omission_summaries"]),
            variation_summaries=len(details["variation_summaries"]),
            timing_groups=len(details["timing"]),
            duplicate_label_cases=4,
            duplicate_label_output_changes=duplicate_changes,
            exploratory_9875_intervals=len(family),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "All invalid-vote evidence and reporting details independently PASS.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()
