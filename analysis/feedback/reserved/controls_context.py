from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from analysis.feedback.reserved import controls as a
from analysis.feedback.reserved import scoring as report
from analysis.common import prompt as renderer


def sig(value):
    return hashlib.sha256(
        (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def examples(p):
    groups = p["context"]["example_groups"]
    a.need(
        len(groups) == 1 and groups[0][0] == "Labelled examples",
        "Group heading changed",
    )
    return groups[0][1]


def check_intervention(base, prompt, mode, correction=None):

    expected = deepcopy(base["context"])
    ordinary = expected["example_groups"][0][1]
    a.need(len(ordinary) == 5, "Reference does not contain five examples")
    if mode == "reference4":
        expected["example_groups"][0][1] = ordinary[:4]
    elif mode in ("reserved5", "masked5", "extended6"):
        a.need(correction is not None, "Missing correction")
        correction = deepcopy(correction)
        a.need(
            type(correction["meta"]["gate"]) is bool, "Unknown correction source label"
        )
        a.need(
            all(
                (
                    correction["meta"].get(k) is None
                    for k in ("hate_types", "target_groups", "severity", "legal")
                )
            ),
            "Invented fine label",
        )
        if mode == "masked5":
            correction["meta"]["gate"] = None
        expected["example_groups"][0][1] = ordinary[
            : 5 if mode == "extended6" else 4
        ] + [correction]
    else:
        a.need(mode == "reference5", "Unsupported deterministic intervention")
    a.need(
        prompt["context"] == expected, "Context differs beyond declared intervention"
    )
    a.need(
        prompt["text"] == base["text"] and prompt["messages"][0] == base["messages"][0],
        "Query or system changed",
    )


def check_masked_message(reserved, masked):
    correction = examples(reserved)[-1]
    gate = correction["meta"]["gate"]
    quoted = '- "' + " ".join(correction["text"].split()) + '"\n  labels: '
    annotated = quoted + "hate=" + str(gate).lower()
    body = reserved["messages"][1]["text"]
    a.need(body.count(annotated) == 1, "Ambiguous correction block")
    a.need(
        masked["messages"]
        == [
            reserved["messages"][0],
            {"role": "user", "text": body.replace(annotated, quoted, 1)},
        ],
        "Mask changes more than binary annotation",
    )
    a.need(
        renderer.corrected_gold(examples(masked)[-1]["meta"]) == "",
        "Masked unknown rendered as a label",
    )


def prompt_blocks(conditions):
    block = {}
    last = None
    for p in a.io.rows(a.io.STUDY / "prompts.jsonl"):
        if p["key"] != last:
            if block:
                yield block
            block = {}
            last = p["key"]
        if p["condition"] in conditions:
            a.need(p["condition"] not in block, "Duplicate frozen prompt")
            block[p["condition"]] = p
    if block:
        yield block


def summary_numbers(values):
    values = list(values)
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05_p95": np.quantile(values, [0.05, 0.95]).tolist(),
    }


def main():
    done = a.io.read(a.OUT / "statistics_complete.json")
    a.io.verify(a.io.ROOT, done["artifacts"])
    a.io.verify(a.io.ROOT, done["bindings"])
    a.need(
        not (a.OUT / "context_complete.json").exists(),
        "Preserve earlier context review",
    )
    roster, science, predictions, assembly = a.load()
    conditions = a.conditions_for(science)
    bykey = {r["key"]: r for r in roster}
    selection = (
        a.io.ROOT
        / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection"
    )
    pools = [
        p
        for p in a.io.read(selection / "actual_pools.json")["pools"]
        if p["condition"].startswith("error_balanced_s")
    ]
    records = a.io.read(selection / "gate_only_records.json")
    poolnames = {p["condition"] for p in pools}
    choices = [
        x
        for x in a.io.rows(a.io.STUDY / "choices.jsonl")
        if x["condition"] in poolnames
    ]
    paths = [
        Path(__file__),
        Path(renderer.__file__),
        Path(report.__file__),
        a.io.ROOT / "src/hsrag/prompt.py",
        a.io.ROOT / "prompts/classify_v1.txt",
        a.io.ROOT / "config/taxonomy.yaml",
        selection / "actual_pools.json",
        selection / "gate_only_records.json",
        a.io.STUDY / "prompts.jsonl",
        a.io.STUDY / "choices.jsonl",
    ]
    bindings = {a.io.rel(p): a.io.sha(p) for p in paths}
    a.write(
        a.OUT / "context_started.json",
        {"created_utc": a.io.utc(), "bindings": bindings, "new_model_calls": 0},
    )
    metadata = {}
    ledgers = []
    seen = set()
    checked = Counter()
    condition_tokens = defaultdict(list)
    for block in prompt_blocks(conditions):
        a.need(set(block) == set(conditions), "Incomplete relevant prompt block")
        ref = block["reference5"]
        key = ref["key"]
        a.need(key not in seen, "Duplicate input block")
        seen.add(key)
        a.need(ref["text"] == bykey[key]["text"], "Roster query differs")
        for condition, p in block.items():
            a.need(
                sig(p["context"]) == p["context_sha256"]
                and sig(p["messages"]) == p["messages_sha256"],
                "Prompt hash differs",
            )
            a.need(
                renderer.render_messages(p["text"], "en", p["context"], "corrected")
                == p["messages"],
                "Frozen prompt no longer reproduced",
            )
            a.need(
                p["context"]["definitions"] == ref["context"]["definitions"]
                and p["context"]["guidelines"] == ref["context"]["guidelines"],
                "Abstract context changed",
            )
            a.need(
                all(
                    (
                        predictions[key, condition, r]["messages_sha256"]
                        == p["messages_sha256"]
                        for r in range(3)
                    )
                ),
                "Prediction-to-prompt binding differs",
            )
            checked["prompts_rehashed_rerendered_and_linked"] += 1
            condition_tokens[condition].append(p["reference_input_tokens"])
        check_intervention(ref, ref, "reference5")
        check_intervention(ref, block["reference4"], "reference4")
        meta, _ = report.prompt_diagnostics(block.values(), choices, records, pools)
        metadata.update(meta)
        for pool in pools:
            name = pool["condition"]
            ps = {mode: block[mode + "__" + name] for mode in a.MODES}
            reserved = ps["reserved5"]
            correction = examples(reserved)[-1]
            source = records[reserved["source_id"]]
            a.need(
                correction["id"] == source["id"]
                and correction["text"] == source["text"]
                and (correction["meta"] == source["meta"]),
                "Correction content differs from source record",
            )
            a.need(
                correction["id"] not in {e["id"] for e in examples(ref)},
                "Correction overlaps ordinary example",
            )
            a.need(
                " ".join(correction["text"].split()) != " ".join(ref["text"].split()),
                "Same-text source leakage",
            )
            for mode in ("reserved5", "masked5", "extended6"):
                check_intervention(ref, ps[mode], mode, correction)
            check_masked_message(reserved, ps["masked5"])
            checked["label_only_message_pairs"] += 1
            checked["retention_position_pairs"] += 1
            n = meta[key, "natural5__" + name]
            r = meta[key, "reserved5__" + name]
            fifth = examples(ref)[4]
            source_gate = correction["meta"]["gate"]
            natids = n["delivered_correction_ids"]
            natural_examples = examples(ps["natural5"])
            natural_ids = {e["id"] for e in natural_examples}
            row = {
                "key": key,
                "source": bykey[key]["source"],
                "gold_hate": bykey[key]["gold"]["hate"],
                "pool": name,
                "selected_source": reserved["source_id"],
                "selected_gate": source_gate,
                "selected_cosine": reserved["selected_correction_cosine"],
                "natural_delivered_ids": natids,
                "natural_delivered_gates": [
                    d["meta"]["gate"] for d in n["delivered_annotations"]
                ],
                "natural_delivers_selected": reserved["source_id"] in natids,
                "natural_identical_reference": n["identical_to_reference"],
                "natural_identical_reserved": ps["natural5"]["messages_sha256"]
                == reserved["messages_sha256"],
                "natural_first_four_kept": sum(
                    (e["id"] in natural_ids for e in examples(ref)[:4])
                ),
                "natural_removed_ordinary_count": len(n["removed_ordinary"]),
                "natural_example_count": n["example_count"],
                "removed_fifth": {"id": fifth["id"], "meta": fifth["meta"]},
                "removed_fifth_known_fine_dimensions": [
                    k
                    for k in ("hate_types", "target_groups", "severity", "legal")
                    if fifth["meta"].get(k) is not None
                ],
                "tokens": {
                    mode: ps[mode]["reference_input_tokens"] for mode in a.MODES
                },
                "reference5_tokens": ref["reference_input_tokens"],
                "reference4_tokens": block["reference4"]["reference_input_tokens"],
                "messages_sha256": {
                    mode: ps[mode]["messages_sha256"] for mode in a.MODES
                },
            }
            a.need(
                all(
                    (
                        ps[m]["source_id"] == row["selected_source"]
                        and ps[m]["selected_correction_cosine"]
                        == row["selected_cosine"]
                        for m in a.MODES
                    )
                ),
                "Modes use different selected correction",
            )
            ledgers.append(row)
        if len(seen) % 100 == 0:
            print("Frozen contexts checked: " + str(len(seen)) + "/675", flush=True)
    a.need(
        seen == set(bykey) and len(metadata) == 9450 and (len(ledgers) == 2025),
        "Context matrix incomplete",
    )
    a.jsonlines(a.OUT / "prompt_metadata.jsonl", metadata.values())
    a.jsonlines(a.OUT / "query_pool_contexts.jsonl", ledgers)
    overall = Counter()
    distribution = defaultdict(Counter)
    tokens = defaultdict(list)
    pool_info = {}
    for pool in pools:
        pool_info[pool["condition"]] = {
            "size": len(pool["ids"]),
            "source_labels": dict(
                Counter((str(records[s]["meta"]["gate"]) for s in pool["ids"]))
            ),
            "earlier_status": pool["actual_status_counts"],
        }
    for x in ledgers:
        overall["query_pool_pairs"] += 1
        overall["natural_any_delivery"] += bool(x["natural_delivered_ids"])
        overall["natural_delivered_records"] += len(x["natural_delivered_ids"])
        overall["natural_selected_delivery"] += x["natural_delivers_selected"]
        overall["natural_identical_reference"] += x["natural_identical_reference"]
        overall["natural_identical_reserved"] += x["natural_identical_reserved"]
        overall["natural_first_four_changed"] += x["natural_first_four_kept"] != 4
        overall["natural_multiple_corrections"] += len(x["natural_delivered_ids"]) > 1
        distribution["selected_gate"][str(x["selected_gate"])] += 1
        distribution["selected_gate_on_" + ("hateful" if x["gold_hate"] else "benign")][
            str(x["selected_gate"])
        ] += 1
        distribution["selected_source"][x["selected_source"]] += 1
        distribution["selected_cosine"][
            (
                "<0.4"
                if x["selected_cosine"] < 0.4
                else "0.4_to_0.6" if x["selected_cosine"] < 0.6 else ">=0.6"
            )
        ] += 1
        distribution["natural_delivered_gate"].update(
            (str(v) for v in x["natural_delivered_gates"])
        )
        distribution["natural_removed_ordinary_count"][
            str(x["natural_removed_ordinary_count"])
        ] += 1
        distribution["removed_fifth_gate"][
            str(x["removed_fifth"]["meta"].get("gate"))
        ] += 1
        distribution["removed_fifth_any_fine"][
            str(bool(x["removed_fifth_known_fine_dimensions"]))
        ] += 1
        for k in x["removed_fifth_known_fine_dimensions"]:
            distribution["removed_fifth_known_fine_dimensions"][k] += 1
        for mode, n in x["tokens"].items():
            tokens[mode + "-reference5"].append(n - x["reference5_tokens"])
        for first, second in [
            ("reserved5", "masked5"),
            ("extended6", "reserved5"),
            ("reserved5", "natural5"),
        ]:
            tokens[first + "-" + second].append(
                x["tokens"][first] - x["tokens"][second]
            )
        tokens["reference4-reference5"].append(
            x["reference4_tokens"] - x["reference5_tokens"]
        )
    overlap = []
    for i, p in enumerate(pools):
        for q in pools[i + 1 :]:
            u, v = (set(p["ids"]), set(q["ids"]))
            overlap.append(
                {
                    "a": p["condition"],
                    "b": q["condition"],
                    "intersection": len(u & v),
                    "union": len(u | v),
                }
            )
    summary = {
        "checked": dict(checked),
        "overall": dict(overall),
        "distributions": {k: dict(v) for k, v in distribution.items()},
        "pools": pool_info,
        "overlap": overlap,
        "unique_pool_sources": len({s for p in pools for s in p["ids"]}),
        "cosine": summary_numbers((x["selected_cosine"] for x in ledgers)),
        "token_differences": {k: summary_numbers(v) for k, v in tokens.items()},
        "condition_tokens": {
            k: summary_numbers(v) for k, v in condition_tokens.items()
        },
        "unit": "dependent query/pool pairs; 2025 pairs represent 675 texts and three overlapping fixed pools",
        "label_control": "All 2025 reserved/masked pairs differ only by removing the known binary annotation. Unknown is rendered blank, never hate=false.",
        "extended_control": "Retains ordinary fifth example and shifts same correction from position five to six; more tokens and changed position are bundled.",
        "token_measure": "Frozen local tokenizer estimate per prompt, not measured billed tokens; reference4 differences repeated across three pools.",
    }
    a.write(a.OUT / "context_summary.json", summary)
    transitions(roster, science, predictions, metadata, ledgers)
    a.io.verify(a.io.ROOT, bindings)
    artifact_paths = [
        a.OUT / n
        for n in (
            "context_started.json",
            "prompt_metadata.jsonl",
            "query_pool_contexts.jsonl",
            "context_summary.json",
            "mechanism_transition_pairs.jsonl",
            "mechanism_transition_summary.json",
            "natural_reference_variation.json",
            "reference_repeat_variation.json",
        )
    ]
    a.write(
        a.OUT / "context_complete.json",
        {
            "status": "part3_context_delivery_and_transition_checks_complete",
            "created_utc": a.io.utc(),
            "bindings": bindings,
            "artifacts": {a.io.rel(p): a.io.sha(p) for p in artifact_paths},
            "new_model_calls": 0,
            "context_checks": dict(checked),
        },
    )
    print(
        json.dumps(
            {
                "context_checks": dict(checked),
                "delivery": dict(overall),
                "new_model_calls": 0,
            }
        ),
        flush=True,
    )


def pair_update(group, key, treatment, control, gold):
    group["query_pool_repeat_pairs"] += 1
    group["keys"].add(key)
    te = treatment != gold
    ce = control != gold
    group["treatment_errors"] += te
    group["control_errors"] += ce
    group[
        (
            "fix"
            if ce and (not te)
            else (
                "regression"
                if te and (not ce)
                else "unchanged_error" if te else "unchanged_correct"
            )
        )
    ] += 1


def finish_groups(groups):
    result = {}
    for name, values in groups.items():
        row = dict(values)
        row["unique_texts"] = len(row.pop("keys"))
        n = row["query_pool_repeat_pairs"]
        row["treatment_rate"] = row["treatment_errors"] / n
        row["control_rate"] = row["control_errors"] / n
        row["difference"] = (row["treatment_errors"] - row["control_errors"]) / n
        result[name] = row
    return result


def transitions(roster, science, predictions, meta, ledgers):
    results = a.io.read(a.OUT / "mechanism_and_sensitivities.json")
    index = {(x["key"], x["pool"]): x for x in ledgers}
    specs = [s for s in a.rates.contrasts(science) if s["family"] == "mechanism"]
    groups = defaultdict(lambda: Counter(keys=set()))
    pairs = []
    for spec in specs:
        result = results["full/ordinary_valid/" + spec["id"]]
        mode = spec["control_conditions"][0].split("__")[0]
        for key in result["common_ids"]:
            for t, c in zip(spec["treatment_conditions"], spec["control_conditions"]):
                pool = t.split("__", 1)[1]
                x = index[key, pool]
                identical = (
                    meta[key, t]["messages_sha256"] == meta[key, c]["messages_sha256"]
                )
                cosine = x["selected_cosine"]
                dimensions = {
                    "all": "all",
                    "source": x["source"],
                    "selected_gate": str(x["selected_gate"]),
                    "selected_matches_gold": str(x["selected_gate"] == x["gold_hate"]),
                    "natural_delivery": (
                        "selected"
                        if x["natural_delivers_selected"]
                        else "other_only" if x["natural_delivered_ids"] else "none"
                    ),
                    "identical_message": str(identical),
                    "cosine": (
                        "<0.4"
                        if cosine < 0.4
                        else "0.4_to_0.6" if cosine < 0.6 else ">=0.6"
                    ),
                    "removed_fifth_gate": str(x["removed_fifth"]["meta"].get("gate")),
                    "removed_fifth_any_fine": str(
                        bool(x["removed_fifth_known_fine_dimensions"])
                    ),
                }
                for rep in range(3):
                    th = predictions[key, t, rep]["result"]["hate"]
                    ch = predictions[key, c, rep]["result"]["hate"]
                    for dimension, value in dimensions.items():
                        pair_update(
                            groups[spec["id"] + "/" + dimension + "/" + value],
                            key,
                            th,
                            ch,
                            x["gold_hate"],
                        )
                    pairs.append(
                        {
                            "contrast": spec["id"],
                            "key": key,
                            "pool": pool,
                            "repeat": rep,
                            "treatment_hate": th,
                            "control_hate": ch,
                            "gold_hate": x["gold_hate"],
                            "identical_messages": identical,
                            "selected_source": x["selected_source"],
                            "selected_gate": x["selected_gate"],
                        }
                    )
    a.jsonlines(a.OUT / "mechanism_transition_pairs.jsonl", pairs)
    final = finish_groups(groups)
    for spec in specs:
        row = final[spec["id"] + "/all/all"]
        expected = results["full/ordinary_valid/" + spec["id"]]
        a.need(
            abs(row["difference"] - expected["difference"]) < 1e-12,
            "Transition difference disagrees with frozen rate",
        )
    a.write(
        a.OUT / "mechanism_transition_summary.json",
        {
            "groups": final,
            "interpretation": "Descriptive dependent pairs on each mechanism common-valid mask. Unequal subgroup pool counts; no independent-pair confidence intervals, causal explanation, randomised relevance or label-polarity effect.",
        },
    )
    groups = defaultdict(lambda: Counter(keys=set()))
    missing = Counter()
    for x in ledgers:
        key = x["key"]
        c = "natural5__" + x["pool"]
        endpoint = "FNR" if x["gold_hate"] else "FPR"
        names = [
            endpoint + "/all",
            endpoint + "/identical:" + str(x["natural_identical_reference"]),
            endpoint + "/delivery:" + str(bool(x["natural_delivered_ids"])),
        ]
        for rep in range(3):
            t, b = (predictions[key, c, rep], predictions[key, "reference5", rep])
            if t["result"] is None or b["result"] is None:
                for name in names:
                    missing[name] += 1
                continue
            for name in names:
                pair_update(
                    groups[name],
                    key,
                    t["result"]["hate"],
                    b["result"]["hate"],
                    x["gold_hate"],
                )
    a.write(
        a.OUT / "natural_reference_variation.json",
        {
            "groups": finish_groups(groups),
            "missing_pairs": dict(missing),
            "interpretation": "Same-prompt reruns vary. Reference outputs reused across pools; pairwise-valid descriptive pairs, not new independent observations.",
        },
    )
    repeat = defaultdict(Counter)
    for row in roster:
        name = "FNR" if row["gold"]["hate"] else "FPR"
        for r1, r2 in ((0, 1), (0, 2), (1, 2)):
            first, second = (
                predictions[row["key"], "reference5", r1],
                predictions[row["key"], "reference5", r2],
            )
            repeat[name]["eligible_repeat_pairs"] += 1
            if first["result"] is None or second["result"] is None:
                repeat[name]["missing_pairs"] += 1
                continue
            repeat[name]["valid_repeat_pairs"] += 1
            repeat[name]["different_gate"] += (
                first["result"]["hate"] != second["result"]["hate"]
            )
    a.write(
        a.OUT / "reference_repeat_variation.json",
        {
            "groups": {k: dict(v) for k, v in repeat.items()},
            "interpretation": "Three overlapping repeat pairs per original text; descriptive variation only.",
        },
    )


if __name__ == "__main__":
    with a.io.offline():
        main()
