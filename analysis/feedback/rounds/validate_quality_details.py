from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
from analysis.feedback.rounds import quality_inputs as a
from analysis.feedback.transfer.validate_quality import near

FIELDS = {
    "hate_type": "gold_hate_types",
    "target_group": "gold_target_groups",
    "severity": "gold_severity",
    "legal": "gold_legal",
}


def known(r, h):
    return r[FIELDS[h]] is not None and (h == "legal" or r["gold_hate"])


def conditions(p):
    return (
        [0]
        if p == "reference"
        else [1 + 3 * s + ["adaptive", "static", "random"].index(p) for s in range(3)]
    )


def mean(scores, p):
    return sum(scores[..., ci, rep, :] for ci in conditions(p) for rep in range(3)) / (
        3 * len(conditions(p))
    )


def sig(v):
    return hashlib.sha256(
        (
            json.dumps(v, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        ).encode()
    ).hexdigest()


def main():
    receipt = a.read(a.OUT / "diagnostics_complete.json")
    for path, digest in {**receipt["inputs"], **receipt["artifacts"]}.items():
        a.need(a.sha(a.ROOT / path) == digest, "Diagnostic evidence changed")
    a.need(
        a.read(a.OUT / "independent_verification.json")["status"]
        == "part6_independent_verification_pass",
        "Main numerical verification required",
    )
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    analysis = a.read(a.OUT / "analysis.json")
    records = analysis["records"]
    contract = a.read(a.OUT / "analysis_contract.json")
    labels = contract["labels"]
    diag = a.read(a.OUT / "diagnostics.json")
    paths = list(a.rows(a.OUT / "input_quality_paths.jsonl"))
    selections = a.read(a.OUT / "case_selection.json")
    with np.load(a.OUT / "prediction_codes.npz", allow_pickle=False) as f:
        codes = {h: f[h] for h in f.files}
    truth = np.asarray([m["gold_hate"] for m in meta])[:, None, None, None]
    binary = codes["gates"] == truth
    exact = {
        h: np.zeros_like(binary) for h in ["hate_type", "target_group", "severity"]
    }
    joint = binary.copy()
    for i, m in enumerate(meta):
        for h in exact:
            if not known(m, h):
                continue
            gold = (
                labels[h].index(m[FIELDS[h]])
                if h == "severity"
                else sum(1 << labels[h].index(v) for v in set(m[FIELDS[h]]))
            )
            exact[h][i] = codes[h][i] == gold
            joint[i] &= exact[h][i]
    exact["known_answer"] = joint
    for name, detail in diag["per_label"].items():
        r = records[name]
        ids = r["indices"]
        h = r["head"]
        rounds = detail["rounds"]
        a.need(
            rounds == ([1, 2, 3, 4] if r["view"] == "fixed" else [4]),
            "Label detail rounds differ",
        )
        pred = codes[h][ids][:, np.asarray(rounds) - 1]
        gold = [meta[i][FIELDS[h]] for i in ids]
        for lab, row in detail["labels"].items():
            gp = np.asarray([g == lab if h == "severity" else lab in g for g in gold])
            pp = (
                (pred == labels[h].index(lab))
                if h == "severity"
                else (pred & (1 << labels[h].index(lab))) > 0
            )
            tp = (pp & gp[:, None, None, None]).sum(axis=0)
            fp = (pp & ~gp[:, None, None, None]).sum(axis=0)
            fn = (~pp & gp[:, None, None, None]).sum(axis=0)
            expected = np.stack([tp, fp, fn], axis=-1)
            a.need(
                np.array_equal(expected, row["counts"]), "Per-label bit counts differ"
            )

            def ratio(x, y):
                return np.divide(x, y, out=np.zeros(x.shape, dtype=float), where=y != 0)

            scores = np.stack(
                [
                    ratio(tp, tp + fp),
                    ratio(tp, tp + fn),
                    ratio(2 * tp, 2 * tp + fp + fn),
                ],
                axis=-1,
            )
            near(row["run_precision_recall_f1"], scores, "Independent label P/R/F1")
            near(
                row["policy_precision_recall_f1"],
                np.stack([mean(scores, p) for p in a.POLICIES], axis=-2),
                "Independent label policy means",
            )
            a.need(
                row["common_positive_support"] == int(gp.sum())
                and row["in_macro"] == (lab in r["macro_labels"]),
                "Label support/macro differs",
            )
        shape = diag["set_and_severity"][name]
        if h == "severity":
            confusion = np.zeros((len(rounds), 10, 3, 3, 4), dtype=int)
            for ii, g in enumerate(gold):
                for ri in range(len(rounds)):
                    for ci in range(10):
                        for rep in range(3):
                            confusion[
                                ri,
                                ci,
                                rep,
                                labels[h].index(g),
                                int(pred[ii, ri, ci, rep]) + 1,
                            ] += 1
            a.need(
                np.array_equal(confusion, shape["confusion"]),
                "Severity confusion differs",
            )
            missing = (
                (pred < 0) & (codes["gates"][ids][:, np.asarray(rounds) - 1] == 1)
            ).sum(axis=0)
            a.need(
                np.array_equal(missing, shape["none_with_positive_gate"]),
                "Positive-gate missing severity differs",
            )
        else:
            goldbits = [sum(1 << labels[h].index(v) for v in set(g)) for g in gold]
            goldsize = [v.bit_count() for v in goldbits]
            a.need(
                np.array_equal(
                    np.bincount(goldsize, minlength=len(labels[h]) + 1),
                    shape["gold_cardinality"],
                ),
                "Gold cardinality differs",
            )
            a.need(
                shape["known_empty"] == sum(v == 0 for v in goldsize),
                "Known-empty count differs",
            )
            near(shape["mean_gold_labels"], sum(goldsize) / len(ids), "Gold mean size")
            hist = np.zeros((len(rounds), 10, 3, len(labels[h]) + 1), dtype=int)
            categories = {
                v: np.zeros((len(rounds), 10, 3), dtype=int)
                for v in shape["categories"]
            }
            for ii, g in enumerate(goldbits):
                for ri in range(len(rounds)):
                    for ci in range(10):
                        for rep in range(3):
                            p = int(pred[ii, ri, ci, rep])
                            e = (p & ~g).bit_count()
                            m = (g & ~p).bit_count()
                            hist[ri, ci, rep, p.bit_count()] += 1
                            for category, value in dict(
                                exact=p == g,
                                contains_gold=m == 0,
                                only_extra=e > 0 and m == 0,
                                only_missing=e == 0 and m > 0,
                                both_extra_missing=e > 0 and m > 0,
                                extra_labels=e,
                                missing_labels=m,
                            ).items():
                                categories[category][ri, ci, rep] += value
            a.need(
                np.array_equal(hist, shape["predicted_cardinality"]),
                "Predicted cardinality differs",
            )
            for category, value in categories.items():
                a.need(
                    np.array_equal(value, shape["categories"][category]),
                    "Set partition differs",
                )

    for record in diag["pair_transitions"]:
        expected = {}
        ids = record["inputs"]
        h = record["head"]
        lr = record["left_round"] - 1
        rr = record["right_round"] - 1
        for i in ids:
            for stream in range(3):
                left = (
                    0
                    if record["left"] == "reference"
                    else conditions(record["left"])[stream]
                )
                right = conditions(record["right"])[stream]
                for rep in range(3):
                    le = bool(exact[h][i, lr, left, rep])
                    re = bool(exact[h][i, rr, right, rep])
                    lg = codes["gates"][i, lr, left, rep] == 1
                    rg = codes["gates"][i, rr, right, rep] == 1
                    lb = bool(binary[i, lr, left, rep])
                    rb = bool(binary[i, rr, right, rep])
                    part = (
                        "both_positive"
                        if lg and rg
                        else (
                            "gained_positive_gate"
                            if rg
                            else "lost_positive_gate" if lg else "both_negative"
                        )
                    )
                    for cat in ["all", part]:
                        c = expected.setdefault(
                            cat,
                            dict(
                                pairs=0,
                                keys=set(),
                                exact_fix=0,
                                exact_regression=0,
                                both_exact=0,
                                neither_exact=0,
                                binary_fix=0,
                                binary_fix_with_exact=0,
                                binary_fix_without_exact=0,
                                binary_regression=0,
                            ),
                        )
                        c["pairs"] += 1
                        c["keys"].add(i)
                        c["exact_fix"] += not le and re
                        c["exact_regression"] += le and not re
                        c["both_exact"] += le and re
                        c["neither_exact"] += not le and not re
                        c["binary_fix"] += not lb and rb
                        c["binary_fix_with_exact"] += not lb and rb and re
                        c["binary_fix_without_exact"] += not lb and rb and not re
                        c["binary_regression"] += lb and not rb
        for name, actual in record["partitions"].items():
            wanted = expected.get(
                name,
                dict(
                    pairs=0,
                    keys=set(),
                    exact_fix=0,
                    exact_regression=0,
                    both_exact=0,
                    neither_exact=0,
                    binary_fix=0,
                    binary_fix_with_exact=0,
                    binary_fix_without_exact=0,
                    binary_regression=0,
                ),
            )
            wanted["unique_texts"] = len(wanted.pop("keys"))
            a.need(wanted == actual, "Independent gate/quality transition differs")
    expected_paths = {}
    for pop in a.POPS:
        for head in ["hate_type", "target_group", "severity", "known_answer"]:
            for i in records[f"population/{pop}|{head}|fixed"]["indices"]:
                expected_paths[i, head] = True
    a.need(len(paths) == len(expected_paths) == 3172, "Input/head path count differs")
    for p in paths:
        i = p["input_index"]
        h = p["head"]
        a.need((i, h) in expected_paths, "Unexpected path")
        expected_paths.pop((i, h))
        ref = exact[h][i][:, 0, :]
        rg = codes["gates"][i][:, 0, :] == 1
        a.need(
            np.array_equal(ref, p["reference_exact_by_repeat"])
            and np.array_equal(rg, p["reference_gate_by_repeat"]),
            "Reference quality path differs",
        )
        for policy, w in p["policies"].items():
            ex = exact[h][i][:, conditions(policy), :].reshape(4, 9)
            g = (codes["gates"][i][:, conditions(policy), :] == 1).reshape(4, 9)
            counts = ex.sum(axis=1)
            a.need(
                np.array_equal(ex, w["exact_by_repeat"])
                and np.array_equal(g, w["gate_by_repeat"])
                and np.array_equal(counts, w["exact_counts"]),
                "Policy quality path differs",
            )
            path = "".join(
                "B" if x > 3 * y else "W" if x < 3 * y else "T"
                for x, y in zip(counts, ref.sum(axis=1))
            )
            a.need(
                w["relative_path"] == path
                and w["always_exact"] == bool(ex.all())
                and w["never_exact"] == bool(not ex.any())
                and w["improved"] == bool(counts[3] > counts[0])
                and w["worsened"] == bool(counts[3] < counts[0]),
                "Path summary differs",
            )
    for r in diag["path_summaries"]:
        pp = [
            p["policies"][r["policy"]]
            for p in paths
            if p["population"] == r["population"] and p["head"] == r["head"]
        ]
        a.need(
            r["inputs"] == len(pp)
            and r["all36_exact"] == sum(p["always_exact"] for p in pp)
            and r["never_exact"] == sum(p["never_exact"] for p in pp)
            and r["retained_mean_benefit"]
            == sum(p["relative_path"] == "BBBB" for p in pp),
            "Grouped path counts differ",
        )
        a.need(
            r["relative_paths"] == dict(Counter(p["relative_path"] for p in pp))
            and r["improved"] == sum(p["improved"] for p in pp)
            and r["worsened"] == sum(p["worsened"] for p in pp)
            and r["equal"] == sum(not p["improved"] and not p["worsened"] for p in pp),
            "Grouped path variation differs",
        )
    for s in selections:
        pop, head, cat = s["population"], s["head"], s["category"]
        ids = records[f"population/{pop}|{head}|fixed"]["indices"]
        eligible = []
        for i in ids:
            ref = exact[head][i][:, 0, :]
            ad = exact[head][i][:, conditions("adaptive"), :].reshape(4, 9)
            rg = codes["gates"][i, 3, 0, :] == 1
            ag = (codes["gates"][i, 3][conditions("adaptive")] == 1).reshape(9)
            if cat == "retained_mean_exact_benefit":
                yes = bool(np.all(ad.sum(axis=1) > 3 * ref.sum(axis=1)))
            elif cat == "fine_regression_both_gates_positive":
                yes = bool(np.any(np.tile(ref[3], 3) & ~ad[3] & np.tile(rg, 3) & ag))
            elif cat == "gate_fix_without_fine_exact":
                yes = bool(np.any(~np.tile(rg, 3) & ag & ~ad[3]))
            elif cat == "known_empty_gold":
                yes = meta[i][FIELDS[head]] == []
            else:
                delta = int(ad[3].sum()) - 3 * int(ref[3].sum())
                yes = meta[i]["any_fine_gold"] and (
                    delta > 0 if cat == "final_known_exact_benefit" else delta < 0
                )
            if yes:
                eligible.append(i)
        rank = lambda i: hashlib.sha256(
            f'A17-part6-case-v1|{pop}/{head}|{cat}|{meta[i]["key"]}'.encode()
        ).hexdigest()
        selected = min(eligible, key=rank) if eligible else None
        a.need(
            s["eligible_candidates"] == len(eligible)
            and s["input_index"] == selected
            and s["status"] == ("selected" if selected is not None else "empty"),
            "Declared case selection differs",
        )

    cases = a.read(a.OUT / "cases_v1/complete.json")
    matrix = list(a.rows(a.READER / "matrix.jsonl"))
    original = list(a.rows(a.STUDY / "evaluation.jsonl"))
    pb = {
        (r["phase"], r["key"], r["condition"]): r
        for r in a.rows(a.PART3 / "prompt_bindings.jsonl")
        if r["phase"].endswith("_evaluation")
    }
    source_records = a.read(a.STUDY / "gate_only_records.json")
    prompts = outputs = 0
    for path, digest in cases["input_bindings"].items():
        a.need(a.sha(a.ROOT / path) == digest, "Case input changed")
    for entry in cases["catalog"]:
        a.need(a.sha(a.ROOT / entry["file"]) == entry["sha256"], "Case copy changed")
        case = a.read(a.ROOT / entry["file"])
        i = case["input_index"]
        a.need(
            case["gold"] == original[i]
            and case["metadata"] == meta[i]
            and case["quality_paths"] == [p for p in paths if p["input_index"] == i]
            and case["selection_slots"]
            == [s for s in selections if s["input_index"] == i],
            "Case metadata/paths differ",
        )
        for frame in case["frames"]:
            rnd = frame["round"]
            a.need(
                [v["condition"] for v in frame["conditions"]] == a.CONDITIONS,
                "Case conditions incomplete",
            )
            for ci, v in enumerate(frame["conditions"]):
                p = v["prompt"]
                bound = pb[f"r{rnd}_evaluation", case["key"], v["condition"]]
                a.need(
                    sig(p["messages"])
                    == bound["messages_sha256"]
                    == p["messages_sha256"]
                    and sig(p["context"])
                    == bound["context_sha256"]
                    == p["context_sha256"],
                    "Case prompt differs",
                )
                a.need(
                    v["correction_record"]
                    == (source_records[p["source_id"]] if p["source_id"] else None),
                    "Case correction differs",
                )
                prompts += 1
                for rep, out in enumerate(v["outputs"]):
                    expected = matrix[i]["predictions"][rnd - 1][ci][rep]
                    a.need(
                        out["repeat"] == rep
                        and out["condition"] == v["condition"]
                        and out["key"] == case["key"]
                        and out["phase"] == f"r{rnd}_evaluation",
                        "Case output identity differs",
                    )
                    a.need(
                        all(
                            out[f] == expected[f]
                            for f in ["result", "n_valid", "uncertain"]
                        )
                        and out["messages_sha256"] == bound["messages_sha256"],
                        "Case output differs",
                    )
                    outputs += 1
    a.need(
        prompts == 40 * cases["unique_inputs"]
        and outputs == 120 * cases["unique_inputs"],
        "Complete case coverage differs",
    )
    a.write(
        a.OUT / "details_independent_verification.json",
        dict(
            status="part6_details_independent_verification_pass",
            created_utc=a.pure.utc(),
            inputs={
                a.rel(a.OUT / f): a.sha(a.OUT / f)
                for f in [
                    "diagnostics_complete.json",
                    "cases_v1/complete.json",
                    "independent_verification.json",
                ]
            },
            source_sha256=a.sha(Path(__file__)),
            label_views=len(diag["per_label"]),
            shape_views=len(diag["set_and_severity"]),
            pair_views=len(diag["pair_transitions"]),
            quality_paths=len(paths),
            case_slots=len(selections),
            unique_cases=cases["unique_inputs"],
            empty_slots=len(cases["empty_slots"]),
            prompts=prompts,
            predictions=outputs,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        f'Independent details PASS: {len(paths)} quality paths, {len(selections)} case slots and {cases["unique_inputs"]} complete case bundles.',
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()
