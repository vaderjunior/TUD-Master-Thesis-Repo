from collections import Counter, defaultdict
import json
import socket
from unittest.mock import patch
import numpy as np
from analysis.feedback.initial.archive import ROOT
from analysis.feedback.initial.archive import DEST
from analysis.feedback.initial.archive import REVIEW
from analysis.feedback.initial.archive import STUDIES
from analysis.feedback.initial.archive import LABELS
from analysis.feedback.initial.archive import SEEDS
from analysis.feedback.initial.archive import read
from analysis.feedback.initial.archive import rows
from analysis.feedback.initial.archive import sha
from analysis.feedback.initial.archive import save
from analysis.feedback.initial.archive import rel
from analysis.feedback.initial.archive import verify
from analysis.feedback.initial.archive import require
from analysis.feedback.initial.archive import utc
from analysis.feedback.initial.archive import forbid
from analysis.feedback.initial.scores import Data
from analysis.feedback.initial.scores import POLICIES


def subset(data, ids, conditions):
    ids = sorted(ids)
    keys = [data.kpos[c, r] for c in conditions for r in range(3)]
    common = data.common(ids, keys)
    if not common:
        return {"original_n": len(ids), "common_valid_n": 0, "common_valid_ids": []}
    result = {
        "original_n": len(ids),
        "common_valid_n": len(common),
        "common_valid_ids": common,
        "excluded_ids": sorted(set(ids) - set(common)),
        "conditions": {},
    }
    for c in conditions:
        ms = [data.metric(common, data.kpos[c, r]) for r in range(3)]
        result["conditions"][c] = {
            k: np.mean([m[k] for m in ms], axis=0).tolist()
            for k in [
                "macro_f1",
                "per_label_f1",
                "contains",
                "exact",
                "predicted_labels",
                "missed_hate_rate",
            ]
        }
    return result


def changes(data, ids, left, right):
    ids = sorted(ids)
    keys = [data.kpos[c, r] for c in [left, right] for r in range(3)]
    common = data.common(ids, keys)
    if not common:
        return {"original_n": len(ids), "common_valid_n": 0}
    p = data.positions(common)
    a = np.stack([data.x[data.kpos[left, r], p] for r in range(3)])
    b = np.stack([data.x[data.kpos[right, r], p] for r in range(3)])
    ag = np.stack([data.gate[data.kpos[left, r], p] for r in range(3)])
    bg = np.stack([data.gate[data.kpos[right, r], p] for r in range(3)])
    ac = np.all(~data.y[p] | a, axis=-1)
    bc = np.all(~data.y[p] | b, axis=-1)
    identical = sum(
        (
            data.runs[left, 0][i]["messages_sha256"]
            == data.runs[right, 0][i]["messages_sha256"]
            for i in common
        )
    )
    result = {
        "original_n": len(ids),
        "common_valid_n": len(common),
        "common_valid_ids": common,
        "identical_prompt_inputs": identical,
        "type_set_change_rate": float(np.any(a != b, axis=-1).mean()),
        "gate_flip_rate": float((ag != bg).mean()),
        "became_hate_rate": float((ag & ~bg).mean()),
        "ceased_hate_rate": float((~ag & bg).mean()),
    }
    if all((data.roster[i]["gold"]["hate"] for i in common)):
        result.update(
            became_contains_rate=float((ac & ~bc).mean()),
            lost_contains_rate=float((~ac & bc).mean()),
        )
    return result


def main():
    complete = read(DEST / "scoring_complete.json")
    verify(complete["outputs"])
    require(
        read(DEST / "independent_verification.json")["status"] == "passed",
        "Independent scoring verification required",
    )
    exported = list(rows(REVIEW / "predictions.jsonl"))
    op = read(REVIEW / "summary.json")
    data = Data("explanations", exported)
    science = read(STUDIES["explanations"] / "scientific_plan_v1.json")
    choices = list(rows(STUDIES["explanations"] / "choices.jsonl"))
    lookup = {(c["pool_seed"], c["query_id"]): c for c in choices}
    source_roster = {
        r["id"]: r
        for r in rows(
            ROOT
            / "experiments/astra/sq3_full_kb_feedback_20260928_v1/feasibility_v1/roster.jsonl"
        )
    }
    coverage = {}
    slices = {}
    noise = {}
    compatibility = {}
    for seed in SEEDS:
        conditions = [
            "reference",
            *[f"{p}_s{seed}" for p in POLICIES if p != "reference"],
        ]
        for pop, ids in [
            ("hateful", science["rosters"]["transfer"]),
            ("benign", science["rosters"]["benign"]),
        ]:
            selected = [lookup[seed, i] for i in ids]
            coverage[f"{pop}_s{seed}"] = {
                "pairs": len(selected),
                "applied": sum((c["apply"] for c in selected)),
                "explanation_available": sum(
                    (c["explanation_available"] for c in selected)
                ),
                "explanation_delivered_selective": sum(
                    (c["apply"] and c["explanation_available"] for c in selected)
                ),
                "distinct_selected_sources": len({c["source_id"] for c in selected}),
                "cosine_quantiles": np.quantile(
                    [c["cosine"] for c in selected], [0, 0.25, 0.5, 0.75, 1]
                ).tolist(),
            }
            if pop == "hateful":
                masks = {
                    "all": lambda c: True,
                    "explanation_available": lambda c: c["explanation_available"],
                    "plain_fallback": lambda c: not c["explanation_available"],
                    "accepted": lambda c: c["apply"],
                    "skipped": lambda c: not c["apply"],
                    "accepted_and_explained": lambda c: c["apply"]
                    and c["explanation_available"],
                }
                for name, predicate in masks.items():
                    ii = [i for i in ids if predicate(lookup[seed, i])]
                    slices[f"{name}_s{seed}"] = subset(data, ii, conditions)
                    cov = Counter()
                    for i in ii:
                        q = set(data.roster[i]["gold"]["hate_type"])
                        c = lookup[seed, i]
                        sg = set(source_roster[c["source_id"]]["gold"]["hate_type"])
                        cov["pairs"] += 1
                        cov["covers_all_gold_types"] += q <= sg
                        cov["any_gold_type_overlap"] += bool(q & sg)
                        cov["disjoint"] += not bool(q & sg)
                    compatibility[f"{name}_s{seed}"] = dict(cov)
            fallback = [i for i in ids if not lookup[seed, i]["explanation_available"]]
            skipped = [i for i in ids if not lookup[seed, i]["apply"]]
            noise[f"{pop}_fallback_specific_plain_s{seed}"] = changes(
                data, fallback, f"specific_always_s{seed}", f"plain_always_s{seed}"
            )
            noise[f"{pop}_skipped_selective_reference_s{seed}"] = changes(
                data, skipped, f"plain_selective_s{seed}", "reference"
            )
            for key in [
                f"{pop}_fallback_specific_plain_s{seed}",
                f"{pop}_skipped_selective_reference_s{seed}",
            ]:
                require(
                    noise[key]["identical_prompt_inputs"]
                    == noise[key]["common_valid_n"],
                    "Frozen fallback/skip messages differ",
                )
    tokens = defaultdict(list)
    for p in rows(STUDIES["explanations"] / "prompt_inventory.jsonl"):
        tokens[p["policy"]].append(p["reference_input_tokens"])
    summary = {
        "coverage": coverage,
        "type_compatibility_descriptive": compatibility,
        "hateful_subsets": slices,
        "identical_prompt_noise": noise,
        "local_input_tokens_by_policy": {
            p: {
                "n_prompts": len(v),
                "min": min(v),
                "mean": float(np.mean(v)),
                "max": max(v),
            }
            for p, v in tokens.items()
        },
        "limits": [
            "Availability and acceptance subsets are descriptive; all-text primaries remain authoritative.",
            "Gold overlap is measured only after retrieval; it never chooses sources.",
            "Identical-prompt differences are fresh stochastic outputs, not evidence of an explanation/correction effect.",
            "Specific/generic explanations are active information/length changes, not a pure internal-reasoning test.",
        ],
    }
    save(DEST / "explanations/delivery_diagnostics.json", summary)
    data = Data("taxonomy", exported)
    prompts = {
        (p["item_id"], p["condition"]): p
        for p in rows(STUDIES["taxonomy"] / "prompts.jsonl")
    }
    hateful = [i for i in data.ids if data.roster[i]["population"] == "hateful_types"]
    b_ids = [i for i in data.ids if data.roster[i]["population"] == "benign"]
    manip = {}
    slice_results = {}
    unchanged = {}

    def target_examples(p, l):
        found = []
        for _, examples in p["context"].get("example_groups", []):
            for e in examples:
                types = e["meta"].get("hate_types")
                if isinstance(types, str):
                    types = json.loads(types)
                if types and l in types:
                    found.append(e["id"])
        return found

    for l in LABELS:
        counts = defaultdict(Counter)
        subsets = defaultdict(list)
        for i in data.ids:
            full = prompts[i, "rag_full"]
            schema = prompts[i, f"rag_schema_{l}"]
            removed = prompts[i, f"rag_removed_{l}"]
            require(
                schema["context"] == removed["context"]
                and schema["messages"][1] == removed["messages"][1],
                "Removed/schema context changed",
            )
            require(
                not target_examples(schema, l),
                "Withdrawn examples reached schema-only prompt",
            )
            require(
                all(
                    (
                        d["id"] != "def-hate_type-" + l
                        for d in schema["context"].get("definitions", [])
                    )
                ),
                "Withdrawn definition reached schema-only prompt",
            )
            ex = bool(target_examples(full, l))
            definition = any(
                (
                    d["id"] == "def-hate_type-" + l
                    for d in full["context"].get("definitions", [])
                )
            )
            same = full["messages_sha256"] == schema["messages_sha256"]
            pop = data.roster[i]["population"]
            c = counts[pop]
            c["inputs"] += 1
            c["full_target_example_reached"] += ex
            c["full_target_definition_reached"] += definition
            c["either_target_component_reached"] += ex or definition
            c["identical_full_schema_messages"] += same
            if pop == "hateful_types":
                subsets["identical_messages" if same else "changed_messages"].append(i)
                subsets[
                    (
                        "target_component_reached"
                        if ex or definition
                        else "no_target_component_reached"
                    )
                ].append(i)
        manip[l] = {p: dict(v) for p, v in counts.items()}
        for name, ids in subsets.items():
            slice_results[l + "/" + name] = subset(
                data, ids, [f"rag_removed_{l}", f"rag_schema_{l}", "rag_full"]
            )
        for pop, ids in [("hateful", hateful), ("benign", b_ids)]:
            same = [
                i
                for i in ids
                if prompts[i, "rag_full"]["messages_sha256"]
                == prompts[i, f"rag_schema_{l}"]["messages_sha256"]
            ]
            unchanged[l + "/" + pop] = changes(
                data, same, "rag_full", f"rag_schema_{l}"
            )
    prep = read(STUDIES["taxonomy"] / "summary.json")
    inventory = read(STUDIES["taxonomy"] / "scope_summary.json")["inventory"]
    kb_summary = {
        l: {k: v for k, v in inv.items() if k != "removed_ids"}
        for l, inv in inventory.items()
    }
    full_kb = {
        r["id"]: r
        for r in rows(
            ROOT / "experiments/astra/full_kb_core_20260915_v1/kb/full/records.jsonl"
        )
    }
    for l, inv in inventory.items():
        lost = [full_kb[i] for i in inv["removed_ids"]]
        co = Counter(
            (
                t
                for r in lost
                if r["kind"] == "example"
                for t in r["meta"].get("hate_types") or []
                if t != l
            )
        )
        kb_summary[l].update(
            removed_examples=sum((r["kind"] == "example" for r in lost)),
            removed_definitions=sum((r["kind"] == "definition" for r in lost)),
            collateral_other_type_annotations=dict(co),
        )
    for l in LABELS:
        for pop in ["hateful_types", "benign"]:
            require(
                manip[l][pop]["full_target_example_reached"]
                == prep["delivery"][l][pop + "_full_target_example_reached"],
                "Example delivery differs from preparation",
            )
            require(
                manip[l][pop]["full_target_definition_reached"]
                == prep["delivery"][l][pop + "_full_target_definition_reached"],
                "Definition delivery differs from preparation",
            )
            require(
                manip[l][pop]["identical_full_schema_messages"]
                == prep["delivery"][l][pop + "_same_user_context_as_full"],
                "Unchanged prompt count differs",
            )
    save(
        DEST / "taxonomy/delivery_diagnostics.json",
        {
            "manipulations": manip,
            "hateful_subsets": slice_results,
            "identical_prompt_noise": unchanged,
            "kb_inventory": kb_summary,
            "local_input_tokens_range": [
                prep["local_input_tokens_min"],
                prep["local_input_tokens_max"],
            ],
            "preparation_seconds": prep["seconds"],
            "scope_seconds": prep["scope"]["seconds"],
            "replay": prep["replay"],
            "limits": [
                "Knowledge restoration bundles example, definition, corpus size/composition changes.",
                "Removed/schema-only context equality is verified; full/schema-only equality subsets only diagnose unchanged exposure.",
                "Outcomes on identical messages remain stochastic; subgroup associations do not identify a pure mechanism.",
            ],
        },
    )
    nulls = Counter(((r["study"], r["id"]) for r in exported if r["result"] is None))
    errors = Counter()
    terminal = Counter()
    removed_terminal = Counter()
    for f in op["terminal_errors"]:
        for e in f["terminal_errors"]:
            error = e["error"]
            field = (
                error.split(" at (")[-1].split(",")[0].strip("() '\"")
                if " at (" in error
                else error
            )
            terminal[f["study"] + "/" + field] += 1
        raw = read(ROOT / f["source"])
        allowed = set(LABELS)
        if "_removed_" in f["condition"]:
            allowed.remove(f["condition"].split("_removed_")[1])
        from analysis.common import schema as isolated

        parser = isolated.adapter(ROOT / f["schema_path"]).bundle.parser
        for e in f["terminal_errors"]:
            text = raw["raw_runs"][e["vote"]][-1]
            try:
                obj = json.loads(parser.extract_json(text))
                bad = set(obj.get("hate_type", [])) - allowed
            except (ValueError, TypeError, AttributeError):
                bad = set()
            for label in bad:
                errors[f["study"] + "/" + str(label)] += 1
            if (
                "_removed_" in f["condition"]
                and f["condition"].split("_removed_")[1] in bad
            ):
                removed_terminal[f["condition"]] += 1
    agreement = {}
    for study in STUDIES:
        dd = Data(study, exported)
        for condition in sorted({c for c, r in dd.keys}):
            c = Counter()
            for i in dd.ids:
                rr = [dd.runs[condition, r][i]["result"] for r in range(3)]
                if any((v is None for v in rr)):
                    c["incomplete_repeat_triple"] += 1
                    continue
                c["complete_repeat_triples"] += 1
                c["all_three_gate_agree"] += len({v["hate"] for v in rr}) == 1
                c["all_three_type_sets_agree"] += (
                    len({tuple(v["hate_type"]) for v in rr}) == 1
                )
            agreement[study + "/" + condition] = dict(c)
    reliability = {
        "nulls_by_input": [
            {"study": s, "id": i, "null_predictions": n}
            for (s, i), n in sorted(nulls.items())
        ],
        "terminal_invalid_vote_fields": dict(terminal),
        "terminal_disallowed_hate_type_labels": dict(errors),
        "terminal_votes_returning_withdrawn_type": dict(removed_terminal),
        "repeat_agreement": agreement,
        "retry_events": op["retry_events"],
        "costs": op["by_policy_costs"],
        "observed_usage": op["operational"]["observed_usage"],
        "stage_wall_hours_including_suspension": {
            s: r["wall_seconds_including_restarts"] / 3600
            for s, r in op["stage_operational"].items()
        },
        "limits": [
            "All completed nulls retained. Parse/schema failures differ from failed HTTP attempts; all 12 transport/HTTP retry failures recovered.",
            "Stage wall and summed client durations include laptop suspension and must not be called pure server compute.",
            "Unknown server work for timed-out attempts is not measured; usage totals cover successful responses only.",
            "Repeated answers share inputs and conditions; agreement is descriptive, not additional independent evaluation data.",
        ],
    }
    save(DEST / "reliability_diagnostics.json", reliability)
    outputs = [
        DEST / "explanations/delivery_diagnostics.json",
        DEST / "taxonomy/delivery_diagnostics.json",
        DEST / "reliability_diagnostics.json",
    ]
    save(
        DEST / "diagnostics_complete.json",
        {
            "status": "descriptive_delivery_and_reliability_diagnostics_complete",
            "created_utc": utc(),
            "source_sha256": sha(__file__),
            "inputs": {
                rel(DEST / "scoring_complete.json"): sha(
                    DEST / "scoring_complete.json"
                ),
                rel(DEST / "independent_verification.json"): sha(
                    DEST / "independent_verification.json"
                ),
                rel(REVIEW / "complete.json"): sha(REVIEW / "complete.json"),
            },
            "outputs": {rel(p): sha(p) for p in outputs},
            "new_model_calls": 0,
        },
    )
    print(
        "Delivery, same-prompt, compatibility, validity, cost and repeat diagnostics saved.",
        flush=True,
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ):
        main()
