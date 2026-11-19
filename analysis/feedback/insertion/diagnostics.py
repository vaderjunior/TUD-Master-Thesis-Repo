from collections import Counter, defaultdict
from pathlib import Path
import socket
from statistics import mean
from types import FunctionType
from unittest.mock import patch
import numpy as np
from analysis.feedback.insertion import archive as r
from analysis.feedback.insertion import common as c
from analysis.feedback.balanced import diagnostics as prior
from analysis.feedback.balanced import validate_scores as independent


def prompt_metadata(roster):
    pools = r.read(c.SELECTION / "actual_pools.json")
    poolmap = {p["condition"]: p for p in pools["pools"]}
    records = r.read(c.SELECTION / "gate_only_records.json")
    recordmap = {v["id"]: (key, v) for key, v in records.items()}
    source_rows = list(r.rows(c.CANDIDATES / "roster.jsonl"))
    source_index = {v["id"]: i for i, v in enumerate(source_rows)}
    sv = np.load(c.CANDIDATES / "query_vectors.npy", allow_pickle=False, mmap_mode="r")
    query_rows = list(r.rows(c.CORE / "dev_inputs.jsonl"))
    query_index = {v["key"]: i for i, v in enumerate(query_rows)}
    qv = np.load(
        c.CORE / "retrieval_v1/query_vectors.npy", allow_pickle=False, mmap_mode="r"
    )
    out = {}
    sources = set()
    largest_error = 0.0
    count = 0
    for p in r.rows(c.BASE / "prompts.jsonl"):
        key, condition = p["key"], p["condition"]
        row = roster[key]
        r.require(
            p["text"] == row["text"] == query_rows[query_index[key]]["text"],
            "Query text differs",
        )
        r.require(
            r.sig(p["messages"]) == p["messages_sha256"]
            and r.sig(p["context"]) == p["context_sha256"],
            "Prompt hash differs",
        )
        examples = [v for label, group in p["context"]["example_groups"] for v in group]
        r.require(
            len(examples) == 5
            and all(v["meta"]["kind"] == "example" for v in examples),
            "Ordinary five-slot budget differs",
        )
        ids = [v["id"] for v in examples]
        r.require(len(set(ids)) == 5, "Repeated example")
        expected = (
            set()
            if condition == "reference"
            else {records[k]["id"] for k in poolmap[condition]["ids"]}
        )
        delivered = [v for v in examples if v["id"] in recordmap]
        r.require(
            [v["id"] for v in delivered] == p["inserted_ids_delivered"]
            and {v["id"] for v in delivered} <= expected,
            "Correction pool/delivery mismatch",
        )
        gates = []
        cosines = []
        origins = []
        positions = []
        for v in delivered:
            source_id, source = recordmap[v["id"]]
            meta = v["meta"]
            gates.append(source["meta"]["gate"])
            origins.append(source_id)
            sources.add(source_id)
            r.require(
                meta["gate"] == source["meta"]["gate"]
                and all(
                    meta.get(k) is None
                    for k in ["hate_types", "target_groups", "severity", "legal"]
                ),
                "Inserted unannotated head filled in",
            )
            r.require(
                v["text"]
                == source["text"]
                == source_rows[source_index[source_id]]["text"],
                "Inserted source text differs",
            )
            q = qv[query_index[key]]
            vec = sv[source_index[source_id]]
            cosine = float(
                np.dot(q.astype(float), vec.astype(float))
                / (np.linalg.norm(q.astype(float)) * np.linalg.norm(vec.astype(float)))
            )
            for hit in p["dense_candidates"]:
                if hit["id"] == v["id"]:
                    largest_error = max(largest_error, abs(cosine - hit["score"]))
            cosines.append(cosine)
            positions.append(ids.index(v["id"]) + 1)
        r.require(gates == p["delivered_gates"], "Delivered label polarity differs")
        out[key, condition] = dict(
            key=key,
            condition=condition,
            messages_sha256=p["messages_sha256"],
            example_ids=ids,
            abstract_sha256=r.sig(
                {k: p["context"][k] for k in ["definitions", "guidelines"]}
            ),
            inserted_ids=[v["id"] for v in delivered],
            source_ids=origins,
            gates=gates,
            cosines=cosines,
            ranks=positions,
            input_tokens=p["reference_input_tokens"],
            dense_candidate_insertions=sum(
                v["id"] in expected for v in p["dense_candidates"]
            ),
            dense_mmr_insertions=sum(
                v["id"] in expected for v in p["channels"]["dense"]
            ),
            bm25_insertions=sum(v["id"] in expected for v in p["channels"]["bm25"]),
        )
        count += 1
    r.require(
        count == 12825 and len(out) == count and largest_error < 2e-6,
        "Prompt roster/cosine reconstruction differs",
    )
    for (key, condition), v in out.items():
        ref = out[key, "reference"]
        r.require(
            v["abstract_sha256"] == ref["abstract_sha256"], "Abstract knowledge changed"
        )
        v["message_equals_reference"] = v["messages_sha256"] == ref["messages_sha256"]
        v["ordered_examples_equal_reference"] = v["example_ids"] == ref["example_ids"]
        v["old_examples_removed"] = [
            i for i in ref["example_ids"] if i not in v["example_ids"]
        ]
        v["new_existing_examples"] = [
            i
            for i in v["example_ids"]
            if i not in ref["example_ids"] and i not in v["inserted_ids"]
        ]
        if not v["gates"]:
            v["delivery_category"] = (
                "no_delivery_identical"
                if v["message_equals_reference"]
                else "no_delivery_changed"
            )
        elif all(v["gates"]):
            v["delivery_category"] = "delivered_hateful_only"
        elif not any(v["gates"]):
            v["delivery_category"] = "delivered_benign_only"
        else:
            v["delivery_category"] = "delivered_mixed"
        if v["ordered_examples_equal_reference"]:
            r.require(
                v["message_equals_reference"],
                "Same examples but different rendered content",
            )
    return out, dict(
        prompts=count,
        five_slot_checks=count,
        constant_abstract_checks=count,
        largest_delivered_cosine_error=largest_error,
        distinct_delivered_source_records=len(sources),
        source_vectors_sha256=r.sha(c.CANDIDATES / "query_vectors.npy"),
        query_vectors_sha256=r.sha(c.CORE / "retrieval_v1/query_vectors.npy"),
    )


def delivery(roster, prompts, omit):
    pools = r.read(c.SELECTION / "actual_pools.json")
    conditions = ["reference"] + [p["condition"] for p in pools["pools"]]
    mapping = {p["condition"]: p for p in pools["pools"]}
    out = {}
    for condition in conditions:
        scopes = {}
        for truth, label in [(None, "all"), (False, "benign"), (True, "hateful")]:
            values = [
                v
                for (k, cond), v in prompts.items()
                if cond == condition
                and k not in omit
                and (truth is None or roster[k]["gold"]["hate"] == truth)
            ]
            used = Counter(i for v in values for i in v["source_ids"])
            similarity = [x for v in values for x in v["cosines"]]
            scopes[label] = dict(
                inputs=len(values),
                with_correction=sum(bool(v["gates"]) for v in values),
                with_benign=sum(False in v["gates"] for v in values),
                with_hateful=sum(True in v["gates"] for v in values),
                categories=dict(Counter(v["delivery_category"] for v in values)),
                delivered_count_histogram=dict(
                    Counter(len(v["gates"]) for v in values)
                ),
                rank_counts=dict(Counter(i for v in values for i in v["ranks"])),
                unique_delivered_sources=len(used),
                source_use_counts=dict(used),
                unused_added_sources=sorted(
                    set(mapping.get(condition, {}).get("ids", [])) - set(used)
                ),
                mean_cosine=mean(similarity) if similarity else None,
                mean_input_tokens=mean(v["input_tokens"] for v in values),
                mean_reference_token_difference=mean(
                    v["input_tokens"] - prompts[v["key"], "reference"]["input_tokens"]
                    for v in values
                ),
                mean_old_examples_removed=mean(
                    len(v["old_examples_removed"]) for v in values
                ),
                with_inserted_dense_candidate=sum(
                    v["dense_candidate_insertions"] > 0 for v in values
                ),
                with_inserted_after_mmr=sum(
                    v["dense_mmr_insertions"] > 0 for v in values
                ),
                with_inserted_bm25=sum(v["bm25_insertions"] > 0 for v in values),
            )
        out[condition] = scopes
    return out


def transitions(roster, records, prompts, scope):
    output = {}
    for truth, pop, endpoint in [
        (False, "benign", "benign_false_positive_rate"),
        (True, "hateful", "hateful_false_negative_rate"),
    ]:
        common = r.read(r.DEST / scope / pop / "all_policies.json")["common_ids"]
        primary = r.read(r.DEST / scope / "primary.json")["primary"][
            endpoint + "/primary"
        ]
        for policy in [
            "error_positive",
            "error_balanced",
            "correct_positive",
            "correct_balanced",
            "random_positive",
            "random_balanced",
        ]:
            for mask, ids in [("all_policies_common", common)] + (
                [("primary_common", primary["common_ids"])]
                if policy == "error_balanced"
                else []
            ):
                counts = defaultdict(Counter)
                members = defaultdict(set)
                gates = defaultdict(Counter)
                similarities = defaultdict(Counter)
                for seed in r.SEEDS:
                    condition = f"{policy}_s{seed}"
                    for key in ids:
                        context = prompts[key, condition]
                        category = context["delivery_category"]
                        members[category].add(key)
                        sim = max(context["cosines"], default=None)
                        bin_name = (
                            "none"
                            if sim is None
                            else (
                                "<.4"
                                if sim < 0.4
                                else (
                                    ".4-.5"
                                    if sim < 0.5
                                    else ".5-.6" if sim < 0.6 else ">=.6"
                                )
                            )
                        )
                        gold = set(roster[key]["gold"]["hate_type"] or [])
                        for rep in range(3):
                            before = records[key, "reference", rep]["result"]
                            after = records[key, condition, rep]["result"]
                            r.require(
                                before is not None and after is not None,
                                "Null transition entered mask",
                            )
                            a = before["hate"] != truth
                            b = after["hate"] != truth
                            for v in [counts[category], similarities[bin_name]]:
                                v["pairs"] += 1
                                v["reference_errors"] += a
                                v["edited_errors"] += b
                                v["fixed"] += a and not b
                                v["harmed"] += not a and b
                                v["type_sets_changed"] += set(
                                    before["hate_type"]
                                ) != set(after["hate_type"])
                            if truth:
                                v = gates[
                                    f'{int(before["hate"])}->{int(after["hate"])}'
                                ]
                                v["pairs"] += 1
                                gained = set(after["hate_type"]) - set(
                                    before["hate_type"]
                                )
                                lost = set(before["hate_type"]) - set(
                                    after["hate_type"]
                                )
                                v["correct_labels_gained"] += len(gained & gold)
                                v["correct_labels_lost"] += len(lost & gold)
                                v["wrong_labels_gained"] += len(gained - gold)
                                v["wrong_labels_lost"] += len(lost - gold)
                total = 9 * len(ids)
                net = sum(
                    v["edited_errors"] - v["reference_errors"] for v in counts.values()
                )
                expected = (
                    primary["difference"]
                    if mask == "primary_common"
                    else r.read(r.DEST / scope / pop / "all_policies.json")[
                        "contrasts"
                    ][policy + "_minus_reference"]["difference"]
                )
                r.require(
                    sum(v["pairs"] for v in counts.values()) == total
                    and abs(net / total - expected) < 1e-12,
                    "Exposure decomposition differs from scored effect",
                )
                output[pop + "/" + policy + "/" + mask] = dict(
                    common_n=len(ids),
                    dependent_pool_repeat_pairs=total,
                    difference=net / total,
                    categories={
                        k: dict(v)
                        | dict(
                            unique_inputs=len(members[k]),
                            conditional_difference=(
                                v["edited_errors"] - v["reference_errors"]
                            )
                            / v["pairs"],
                            contribution=(v["edited_errors"] - v["reference_errors"])
                            / total,
                        )
                        for k, v in counts.items()
                    },
                    cosine_bins={k: dict(v) for k, v in similarities.items()},
                    fine_labels_by_gate_transition={
                        k: dict(v) for k, v in gates.items()
                    },
                    limitation="Observational exposure categories and arbitrary matched decoder indices; repeated pairs are dependent. Contributions sum to rate differences, not macro-F1. Identical messages still receive independent planned generations.",
                )
    return output


def historical(roster, records, prompts):
    oldbase = r.ROOT / "experiments/astra/sq3_balanced_results_20261010_v1/review"
    done = r.read(oldbase / "complete.json")
    p = oldbase / "predictions.jsonl"
    r.require(r.sha(p) == done["outputs"][r.rel(p)], "Prior six-slot export changed")
    old = {(v["key"], v["condition"], v["repeat"]): v for v in r.rows(p)}
    r.require(len(old) == len(records) == 38475, "History has a different matrix")
    same = []
    changed = []
    for key in roster:
        equal = True
        for rep in range(3):
            a = old[key, "reference", rep]
            b = records[key, "reference", rep]
            r.require(
                a["gold"] == b["gold"] == roster[key]["gold"], "Historical gold changed"
            )
            equal &= a["messages_sha256"] == b["messages_sha256"]
        (same if equal else changed).append(key)
    conditions = r.read(c.BASE / "scientific_plan_v1.json")["conditions"]
    common = [
        k
        for k in sorted(roster)
        if all(
            table[k, cond, rep]["result"] is not None
            for table in [old, records]
            for cond in conditions
            for rep in range(3)
        )
    ]
    output = {}
    for phase, table in [("prior_six_slot", old), ("current_five_slot", records)]:
        groups = {}
        for policy in [
            "reference",
            "error_positive",
            "error_balanced",
            "correct_positive",
            "correct_balanced",
            "random_positive",
            "random_balanced",
        ]:
            cs = (
                ["reference"]
                if policy == "reference"
                else [f"{policy}_s{s}" for s in r.SEEDS]
            )
            gate = {}
            types = []
            for truth, name in [(False, "benign"), (True, "hateful")]:
                ids = [k for k in common if roster[k]["gold"]["hate"] == truth]
                gate[name] = mean(
                    independent.primitives(ids, (cond, rep), table, roster, "gate")[
                        "error_rate"
                    ]
                    for cond in cs
                    for rep in range(3)
                )
            ids = [k for k in common if roster[k]["gold"]["hate"]]
            types = [
                independent.primitives(ids, (cond, rep), table, roster, "types")[
                    "macro_f1"
                ]
                for cond in cs
                for rep in range(3)
            ]
            groups[policy] = dict(
                false_positive_rate=gate["benign"],
                false_negative_rate=gate["hateful"],
                type_macro_f1=mean(types),
            )
        output[phase] = groups
    return dict(
        original_inputs=675,
        identical_reference_messages=len(same),
        changed_reference_keys=changed,
        common_all_114_outputs=len(common),
        common_by_polarity=dict(
            Counter(
                "hateful" if roster[k]["gold"]["hate"] else "benign" for k in common
            )
        ),
        policies=output,
        inputs={
            r.rel(p): r.sha(p),
            r.rel(oldbase / "complete.json"): r.sha(oldbase / "complete.json"),
        },
        limitation="Contextual phase comparison on joint valid support. Five versus six slots, retrieval, example replacements and generation timing differ; the reference itself may change after fresh retrieval. No causal delivery-policy effect or checkpoint-identity claim.",
    )


def main():
    checked = r.read(r.DEST / "scoring_independent_check.json")
    r.verify(checked["inputs"])
    target = r.DEST / "diagnostics_v2/complete.json"
    if target.exists():
        r.verify(r.read(target)["inputs"])
        r.verify(r.read(target)["outputs"])
        print("Existing diagnostics verified.")
        return
    inputs = {
        r.rel(Path(__file__)): r.sha(Path(__file__)),
        r.rel(r.DEST / "analysis_plan.json"): r.sha(r.DEST / "analysis_plan.json"),
        r.rel(r.DEST / "scoring_independent_check.json"): r.sha(
            r.DEST / "scoring_independent_check.json"
        ),
        r.rel(c.BASE / "prompts.jsonl"): r.sha(c.BASE / "prompts.jsonl"),
        r.rel(c.SELECTION / "gate_only_records.json"): r.sha(
            c.SELECTION / "gate_only_records.json"
        ),
    }
    r.save(
        r.DEST / "diagnostics_v2/plan.json",
        dict(
            created_utc=r.utc(),
            inputs=inputs,
            scope="Declared descriptive delivery/transition/history/cost checks; no new primary family or model calls.",
        ),
    )
    roster = {v["key"]: v for v in r.rows(c.BASE / "roster.jsonl")}
    records = {
        (v["key"], v["condition"], v["repeat"]): v
        for v in r.rows(r.REVIEW / "predictions.jsonl")
    }
    prompts, checks = prompt_metadata(roster)
    outputs = {}

    def put(name, value):
        path = r.DEST / "diagnostics_v2" / (name + ".json")
        r.save(path, value)
        outputs[r.rel(path)] = r.sha(path)
        print("Saved diagnostic", name, flush=True)

    put("prompt_checks", checks)
    p = r.DEST / "diagnostics_v2/prompt_metadata.jsonl"
    with p.open("x", encoding="utf8", newline="\n") as f:
        for key in sorted(prompts):
            f.write(
                __import__("json").dumps(prompts[key], sort_keys=True, allow_nan=False)
                + "\n"
            )
    outputs[r.rel(p)] = r.sha(p)
    excluded = r.read(c.BASE / "cold_replay_review_v1/sensitivity_plan.json")[
        "excluded_keys"
    ]
    for scope, omit in [("full", []), ("exclude_replay", excluded)]:
        put(scope + "_delivery", delivery(roster, prompts, omit))
        put(scope + "_transitions", transitions(roster, records, prompts, scope))
    put(
        "pools",
        dict(
            pools=r.read(c.SELECTION / "actual_pools.json"),
            overlaps=r.read(c.SELECTION / "summary.json")["pool_overlaps"],
        ),
    )
    put("historical", historical(roster, records, prompts))
    operational = FunctionType(prior.operational.__code__, vars(prior) | {"r": r})(
        records
    )
    operational["limitation"] = (
        "Saved request/vote time excludes restart downtime; elapsed wall time includes it. Hidden work behind transport failures is unknown. The reported model alias is not a backend checkpoint identifier. Source selection additionally inspected 1024 existing labels and ran 3072 baseline predictions. A14 used original bounded retries, without an extra-call recovery amendment."
    )
    put("operational", operational)
    r.verify(inputs)
    r.save(
        target,
        dict(
            status="supporting_diagnostics_complete",
            created_utc=r.utc(),
            inputs=inputs,
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
