"""German guidelines context."""

from collections import Counter, defaultdict
from contextlib import contextmanager
from itertools import combinations, product
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import re
import time
import numpy as np
from analysis.knowledge.german_guidelines import hate as h
from analysis.knowledge.german_guidelines import legal as l
from analysis.knowledge.german_guidelines import prepare as preparation

c, reader, require = (h.c, h.reader, h.require)
BASE = c.ROOT / "experiments/astra/german_guidelines_context_analysis_20261113_v1"
CONTRACT = BASE / "analysis_contract_v2.json"
PHASES = ("de_test", "de_legal_test", "de_dev_eval", "de_legal_dev_eval")
CONDITIONS = h.CONDITIONS


@contextmanager
def offline():
    with h.offline(), patch.object(
        preparation.Guidelines, "select", c.forbidden
    ), patch.object(preparation, "load_vectors", c.forbidden):
        yield


def save(path, value):
    require(c.normal(path).is_relative_to(BASE), "Part 4 write outside new analysis")
    c.save(path, h.safe(value))


def write_rows(path, values):
    require(c.normal(path).is_relative_to(BASE), "Part 4 rows outside new analysis")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    h.safe(value), ensure_ascii=False, sort_keys=True, allow_nan=False
                )
                + "\n"
            )


def tokens(text):
    return re.findall("\\w+", text.lower())


def case_gold(row):
    return {"gold_hate": row.get("gate"), "gold_legal": row.get("legal")}


def partition_messages(messages, hits):
    """Check the actual bullet block and return everything around it."""
    require(
        len(messages) == 2
        and messages[0]["role"] == "system"
        and (messages[1]["role"] == "user"),
        "Unexpected message roles",
    )
    marker = "## Annotation guidelines\n"
    body = messages[1]["text"]
    start = body.index(marker)
    expected = marker + "\n".join(("- " + hit["text"] for hit in hits))
    require(
        body[start : start + len(expected)] == expected,
        "Delivered guideline bullets differ from context",
    )
    require(
        body[start + len(expected) :].startswith("\n\n## "),
        "Unexpected guideline block boundary",
    )
    return {
        "system": messages[0]["text"],
        "before": body[:start],
        "after": body[start + len(expected) :],
    }


def replacement(before, general, legal_id, inventory):
    old_legal = [rid for rid in before if inventory[rid]["dimension"] == "legal"]
    old_general = [rid for rid in before if inventory[rid]["dimension"] == "hate"]
    retained_general = [rid for rid in old_general if rid in general]
    removed_general = [rid for rid in old_general if rid not in general]
    introduced_general = [rid for rid in general if rid not in old_general]
    if before == general:
        change = "same_ordered_guidelines"
    elif set(before) == set(general):
        change = "same_guidelines_reordered"
    else:
        change = "guideline_membership_changed"
    status = (
        "no_original_legal"
        if not old_legal
        else (
            "retained_original_legal"
            if legal_id in old_legal
            else "new_legal_despite_original_legal"
        )
    )
    return dict(
        original_legal_count=len(old_legal),
        original_legal=old_legal,
        original_general=old_general,
        retained_general=retained_general,
        removed_general=removed_general,
        introduced_general=introduced_general,
        general_change=change,
        added_legal_id=legal_id,
        added_legal_status=status,
        added_legal_original_slot=(
            before.index(legal_id) + 1 if legal_id in before else None
        ),
        removed_original_legal_in_separate=[
            rid for rid in old_legal if rid != legal_id
        ],
    )


def lexical_observation(text, rule, trace):
    query_tokens, rule_tokens = (set(tokens(text)), set(tokens(rule["text"])))
    overlap = sorted(query_tokens & rule_tokens)
    rid = rule["id"]
    idx = trace["candidates"].index(rid)
    bm25_positive = trace["bm25_scores"][idx] > 0
    no_url_tokens = set(tokens(re.sub("https?://\\S+", "", text, flags=re.I)))
    return {
        "shared_tokens": overlap,
        "saved_bm25_score": trace["bm25_scores"][idx],
        "bm25_rank": (
            trace["bm25_order"].index(rid) + 1 if rid in trace["bm25_order"] else None
        ),
        "dense_rank": (
            trace["dense_order"].index(rid) + 1 if rid in trace["dense_order"] else None
        ),
        "only_t_overlap": overlap == ["t"],
        "positive_bm25": bm25_positive,
        "only_t_from_url": overlap == ["t"]
        and "t" not in no_url_tokens
        and bool(re.search("https?://\\S+", text, re.I)),
    }


def trace_changes(trace):
    old, new = (trace["mixed_diagnostic"], trace["general"])
    ids = set(new["candidates"])
    old_dense = [rid for rid in old["dense_order"] if rid in ids]
    old_bm = [rid for rid in old["bm25_order"] if rid in ids]
    common_bm = set(old_bm) & set(new["bm25_order"])
    dense_scores_same = all(
        (
            abs(
                old["dense_scores"][old["candidates"].index(rid)]
                - new["dense_scores"][new["candidates"].index(rid)]
            )
            <= 1e-12
            for rid in ids
        )
    )
    require(dense_scores_same, "Saved dense similarities changed")
    return {
        "dense_scores_same_to_1e12": dense_scores_same,
        "filtered_old_dense_top10_differs": old_dense != new["dense_order"],
        "filtered_old_bm25_top10_differs": old_bm != new["bm25_order"],
        "bm25_shared_hit_order_differs": [rid for rid in old_bm if rid in common_bm]
        != [rid for rid in new["bm25_order"] if rid in common_bm],
        "bm25_scores_changed": any(
            (
                abs(
                    old["bm25_scores"][old["candidates"].index(rid)]
                    - new["bm25_scores"][new["candidates"].index(rid)]
                )
                > 1e-12
                for rid in ids
            )
        ),
    }


def context_rows():
    roster = {r["key"]: r for r in c.rows(c.BASE / "roster.jsonl")}
    inventory = {r["id"]: r for r in c.rows(c.BASE / "guideline_records.jsonl")}
    traces = {r["key"]: r for r in c.rows(c.BASE / "retrieval_traces.jsonl")}
    require(
        len(inventory) == 19 and len(roster) == len(traces) == 3677,
        "Saved roster/inventory mismatch",
    )
    require(
        Counter(((v["lang"], v["dimension"]) for v in inventory.values()))
        == {("en", "hate"): 11, ("de", "legal"): 8},
        "Inventory changed",
    )
    compact, hashes, group = ([], {}, {})
    for row in c.rows(c.BASE / "prompts.jsonl"):
        key, cond = (row["key"], row["condition"])
        require(cond in CONDITIONS and (key, cond) not in hashes, "Duplicate prompt")
        require(
            c.signature(row["messages"]) == row["messages_sha256"]
            and c.signature(row["context"]) == row["context_sha256"],
            "Prompt/context hash changed",
        )
        require(row["population"] == roster[key]["phase"], "Population mismatch")
        hits = row["context"]["guidelines"]
        require(
            len(hits) == (3 if cond == "general_2_legal_1" else 2)
            and len({r["id"] for r in hits}) == len(hits),
            "Wrong or duplicate guidelines",
        )
        for hit in hits:
            rec = inventory[hit["id"]]
            require(
                hit["text"] == rec["text"]
                and all(
                    (
                        hit["meta"][k] == rec[k]
                        for k in ("lang", "source", "dimension", "kind", "label")
                    )
                ),
                "Guideline text/metadata differs",
            )
        row["skeleton"] = partition_messages(row["messages"], hits)
        hashes[key, cond] = row["messages_sha256"]
        require(
            not group or key in group, "Prepared prompt order unexpectedly interleaved"
        )
        group.setdefault(key, {})[cond] = row
        if len(group[key]) < 3:
            continue
        by = group.pop(key)
        m, g, a = (by[name] for name in CONDITIONS)
        ids = {
            name: [v["id"] for v in by[name]["context"]["guidelines"]]
            for name in CONDITIONS
        }
        require(
            all((by[name]["skeleton"] == m["skeleton"] for name in CONDITIONS)),
            "Something outside guideline bullets changed",
        )
        for name in CONDITIONS:
            require(
                {k: v for k, v in by[name]["context"].items() if k != "guidelines"}
                == {k: v for k, v in m["context"].items() if k != "guidelines"},
                "Definition/example content changed",
            )
        require(
            ids["general_2_legal_1"][:2] == ids["general_2"]
            and a["context"]["guidelines"][:2] == g["context"]["guidelines"],
            "General prefix changed in added-rule condition",
        )
        require(
            all((inventory[rid]["dimension"] == "hate" for rid in ids["general_2"]))
            and inventory[ids["general_2_legal_1"][2]]["dimension"] == "legal",
            "Wrong bucket",
        )
        require(
            m["messages_sha256"]
            == m["original_messages_sha256"]
            == roster[key]["original_messages_sha256"],
            "Original baseline changed",
        )
        trace = traces[key]
        for bucket, selected in [
            ("mixed_diagnostic", ids["mixed_2"]),
            ("general", ids["general_2"]),
            ("legal", [ids["general_2_legal_1"][2]]),
        ]:
            require(
                trace[bucket]["selected"]
                == selected
                == trace[bucket]["fused_order"][: len(selected)],
                "Saved trace selection differs",
            )
        require(
            hashlib.sha256(roster[key]["text"].encode()).hexdigest()
            == trace["query_text_sha256"],
            "Trace query differs",
        )
        change = replacement(
            ids["mixed_2"], ids["general_2"], ids["general_2_legal_1"][2], inventory
        )
        require(
            change["original_legal_count"] == roster[key]["original_legal_guidelines"],
            "Old legal count mismatch",
        )
        same = m["messages"] == g["messages"]
        require(
            same == (ids["mixed_2"] == ids["general_2"]),
            "Guideline identity and exact-message equality disagree",
        )
        ctx = m["context"]
        examples = [v for _, rows in ctx["example_groups"] for v in rows]
        require(
            len(ctx["definitions"]) == 2 and len(examples) == 5,
            "Other slot counts differ",
        )
        annotations = [
            v
            for v in examples
            if v["meta"].get("legal") is not None
            and (not v["meta"].get("illustrative_only"))
        ]
        record = dict(
            key=key,
            phase=roster[key]["phase"],
            source=roster[key]["source"],
            text_cluster=roster[key]["text_cluster"],
            guideline_ids=ids,
            **change,
            same_messages_mixed_general=same,
            non_guideline_messages_sha256=c.signature(m["skeleton"]),
            messages_sha256={name: by[name]["messages_sha256"] for name in CONDITIONS},
            context_sha256={name: by[name]["context_sha256"] for name in CONDITIONS},
            reference_input_tokens={
                name: by[name]["reference_input_tokens"] for name in CONDITIONS
            },
            trace_changes=trace_changes(trace),
            selected_lexical={
                name: {
                    rid: lexical_observation(
                        roster[key]["text"], inventory[rid], trace[bucket]
                    )
                    for rid in ids[name]
                }
                for name, bucket in [
                    ("mixed_2", "mixed_diagnostic"),
                    ("general_2", "general"),
                ]
            },
            added_legal_lexical=lexical_observation(
                roster[key]["text"], inventory[change["added_legal_id"]], trace["legal"]
            ),
            unchanged_legal_material={
                "legal_definitions": sum(
                    (v["meta"].get("dimension") == "legal" for v in ctx["definitions"])
                ),
                "legal_annotated_examples": len(annotations),
                "positive_legal_examples": sum(
                    (bool(v["meta"]["legal"]) for v in annotations)
                ),
                "empty_legal_examples": sum(
                    (v["meta"]["legal"] == [] for v in annotations)
                ),
                "legal_context_illustrations": sum(
                    (bool(v["meta"].get("illustrative_only")) for v in examples)
                ),
            },
            example_ids=[v["id"] for v in examples],
            definition_ids=[v["id"] for v in ctx["definitions"]],
        )
        compact.append(record)
    require(
        not group and len(compact) == 3677 and (len(hashes) == 11031),
        "Incomplete context matrix",
    )
    return (compact, inventory, roster, hashes)


def exposure(rows, inventory):
    n = len(rows)
    freq = {
        cond: {
            rid: {"inputs": 0, "positions": {str(i): 0 for i in range(1, 4)}}
            for rid in inventory
        }
        for cond in CONDITIONS
    }
    for row in rows:
        for cond, ids in row["guideline_ids"].items():
            for position, rid in enumerate(ids, 1):
                freq[cond][rid]["inputs"] += 1
                freq[cond][rid]["positions"][str(position)] += 1
    summaries = {}
    for name in (
        "original_legal_count",
        "general_change",
        "added_legal_status",
        "added_legal_id",
    ):
        summaries[name] = dict(Counter((str(row[name]) for row in rows)))
    token_stats = {}
    for left, right in [
        ("general_2", "mixed_2"),
        ("general_2_legal_1", "mixed_2"),
        ("general_2_legal_1", "general_2"),
    ]:
        values = np.asarray(
            [
                r["reference_input_tokens"][left] - r["reference_input_tokens"][right]
                for r in rows
            ]
        )
        token_stats[left + "_minus_" + right] = {
            "min": int(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "max": int(values.max()),
            "p95": float(np.quantile(values, 0.95)),
        }
    return dict(
        n=n,
        counts=summaries,
        rule_frequency=freq,
        tokens=token_stats,
        same_messages=sum((r["same_messages_mixed_general"] for r in rows)),
        general_retained_counts=dict(
            Counter((str(len(r["retained_general"])) for r in rows))
        ),
        general_removed_counts=dict(
            Counter((str(len(r["removed_general"])) for r in rows))
        ),
        legal_removed_in_separate_counts=dict(
            Counter((str(len(r["removed_original_legal_in_separate"])) for r in rows))
        ),
        zero_original_legal={
            name: sum(
                (
                    r["original_legal_count"] == 0 and r["general_change"] == name
                    for r in rows
                )
            )
            for name in (
                "same_ordered_guidelines",
                "same_guidelines_reordered",
                "guideline_membership_changed",
            )
        },
        trace_change_counts={
            key: sum((r["trace_changes"][key] for r in rows))
            for key in rows[0]["trace_changes"]
        },
        unchanged_legal_material={
            name: {
                "inputs_with_any": sum(
                    (r["unchanged_legal_material"][name] > 0 for r in rows)
                ),
                "slot_total": sum((r["unchanged_legal_material"][name] for r in rows)),
            }
            for name in rows[0]["unchanged_legal_material"]
        },
        lexical={
            cond: {
                "positive_selected_general_slots": sum(
                    (
                        x["positive_bm25"]
                        for r in rows
                        for rid, x in r["selected_lexical"][cond].items()
                        if inventory[rid]["dimension"] == "hate"
                    )
                ),
                "only_t_positive_slots": sum(
                    (
                        x["positive_bm25"] and x["only_t_overlap"]
                        for r in rows
                        for rid, x in r["selected_lexical"][cond].items()
                        if inventory[rid]["dimension"] == "hate"
                    )
                ),
                "only_url_t_positive_slots": sum(
                    (
                        x["positive_bm25"] and x["only_t_from_url"]
                        for r in rows
                        for rid, x in r["selected_lexical"][cond].items()
                        if inventory[rid]["dimension"] == "hate"
                    )
                ),
                "only_url_t_input_keys": [
                    r["key"]
                    for r in rows
                    if any(
                        (
                            x["positive_bm25"] and x["only_t_from_url"]
                            for rid, x in r["selected_lexical"][cond].items()
                            if inventory[rid]["dimension"] == "hate"
                        )
                    )
                ],
            }
            for cond in ("mixed_2", "general_2")
        },
    )


def score_group(arrays, selected, legal):
    y, p = (arrays["gold"][selected], arrays["predictions"][:, :, selected])
    require(len(y) > 0 and (p >= 0).all(), "Subgroup requires nonempty shared mask")
    summaries = {}
    for i, cond in enumerate(CONDITIONS):
        if legal:
            summaries[cond] = l.summarize(*l.totals(y, p[i]), [0, 1, 2])
        else:
            summaries[cond] = h.summarize(h.contributions(y, p[i]).sum(axis=-2))
    metric_names = (
        (
            "macro_f1",
            "exact_correct",
            "exact_set_accuracy",
            "extra_labels",
            "missed_labels",
        )
        if legal
        else ("macro_f1", "fp", "fn", "total_errors", "fpr", "fnr")
    )
    contrasts = []
    for left, right in h.PAIRS:
        diffs = {
            name: (
                summaries[left]["mean"][name] - summaries[right]["mean"][name]
                if summaries[left]["mean"][name] is not None
                and summaries[right]["mean"][name] is not None
                else None
            )
            for name in metric_names
        }
        contrasts.append({"left": left, "right": right, "differences": diffs})
    return {
        "n": len(y),
        "gold_support": (
            {label: int(y[:, j].sum()) for j, label in enumerate(l.LABELS)}
            if legal
            else {"not_hate": int((y == 0).sum()), "hate": int(y.sum())}
        ),
        "common": summaries,
        "contrasts": contrasts,
        "inference": "Descriptive context subgroup; no additional confidence/significance test or causal rule attribution.",
    }


def decision_variation(results, keys, phase):
    fields = ("hate", "legal", "hate_type", "target_group", "severity")

    def value(result, field):
        v = result[field]
        return tuple(sorted(set(v))) if isinstance(v, list) else v

    def mismatch(a, b, field):
        if field == "all_structured_fields":
            return any((value(a, f) != value(b, f) for f in fields))
        return value(a, field) != value(b, field)

    valid = [
        key
        for key in keys
        if all(
            (results[key, cond, r] is not None for cond in CONDITIONS for r in range(3))
        )
    ]
    rows = {}
    for field in (*fields, "all_structured_fields"):
        across = [
            [
                sum(
                    (
                        mismatch(
                            results[key, "mixed_2", a],
                            results[key, "general_2", b],
                            field,
                        )
                        for key in valid
                    )
                )
                for b in range(3)
            ]
            for a in range(3)
        ]
        within = {
            cond: [
                sum(
                    (
                        mismatch(results[key, cond, a], results[key, cond, b], field)
                        for key in valid
                    )
                )
                for a, b in combinations(range(3), 2)
            ]
            for cond in ("mixed_2", "general_2")
        }
        rows[field] = {
            "cross_all_nine_counts": across,
            "mean_cross_count": float(np.mean(across)),
            "mean_cross_rate": float(np.mean(across)) / len(valid) if valid else None,
            "within_three_pair_counts": within,
            "within_mean_count": {k: float(np.mean(v)) for k, v in within.items()},
        }
    return {
        "phase": phase,
        "full_identical_prompt_inputs": len(keys),
        "common_valid_inputs": len(valid),
        "common_keys": valid,
        "fields": rows,
        "meaning": "Same initial mixed/general messages, independent generated answers; rationale text ignored. Non-scored fields describe disagreement, not accuracy. Do not subtract this subgroup from intervention effects.",
    }


def own_sources():
    return [
        Path(__file__),
        c.ROOT / "tests/knowledge/german_guidelines/test_context.py",
        c.ROOT / "analysis/knowledge/german_guidelines/validate_context.py",
        Path(h.__file__),
        Path(l.__file__),
    ]


def prepare():
    c.validate()
    _, cache = reader.read_verified()
    _, hp = h.verified()
    _, lp = l.verified()
    tests = sorted((BASE / "tests").glob("tests_*/report.json"))
    require(
        tests
        and c.read(tests[-1])["passed"]
        and (c.read(tests[-1])["sources"] == c.bindings(own_sources())),
        "Current context tests must pass",
    )
    save(
        CONTRACT,
        {
            "status": "part4_descriptive_context_plan_with_casebook_schema_repair",
            "created_utc": c.a.utc(),
            "repair": "The first offline attempt saved descriptive group tables, then stopped while exporting cases because hate-only rows omit legal gold. Preserve the incomplete run and original sources under repair_01. Missing legal gold now remains null, never an empty set. No metric, subgroup, selection rule or scientific comparison changes. The original contract remains unchanged; this version binds the implementation repair.",
            "inputs": c.bindings(
                [
                    c.PLAN,
                    c.PROTOCOL,
                    reader.ANALYSIS_PLAN,
                    reader.BASE / "complete.json",
                    cache / "manifest.json",
                    h.BASE / "complete.json",
                    hp / "manifest.json",
                    l.BASE / "complete.json",
                    lp / "manifest.json",
                    tests[-1],
                    BASE / "analysis_contract.json",
                    *(BASE / "repair_01").glob("*.py"),
                    *own_sources(),
                ]
            ),
            "already_known": "Parts 2 and 3 aggregate outcomes are known. This step is descriptive mechanism inspection, not a newly confirmatory analysis.",
            "context": "Validate each saved prompt text, guideline block, record metadata and trace-selected IDs. Bind every cached result to its initial prompt hash. No retrieval, embedding, tokenizer or model rerun.",
            "exposure": "Count inputs, not repeat/vote copies. Full rosters and original common-valid masks separate. Summarise all rules, positions, replacements, retained legal rules, reference-token changes and unchanged legal definitions/examples by endpoint and source.",
            "subgroups": "Original 0/1/2 legal slots; added legal rule identity; retained versus new legal rule; exact ordered context replacement pattern; exact-message and zero-original-legal change subgroups. All nonempty groups saved with fixed original label universe; scores are descriptive, with no extra significance tests.",
            "pattern_reporting": "Top eight patterns per TEST endpoint ranked by full-roster frequency then lexical pattern ID; no selection by favourable scores. All patterns and small groups retained in artifacts.",
            "lexical": "Inspect literal query/rule token overlaps using the frozen regex and saved scores; record URL-only t overlap and changes to saved candidate lists, ranks and scores. Do not recalculate retrieval scores.",
            "identical_prompts": "The 194 exact mixed/general message pairs are a descriptive reference. Compare all nine cross-repeat pairs and three unordered within-condition pairs, retain common mask, ignore rationale. No noise subtraction or randomized-control claim.",
            "context_examples": "For each endpoint, select the smallest SHA256(part4-context|key) within each nonempty structural group: unchanged; zero-legal membership change; zero-legal reorder; one-legal retaining general; one-legal replacing general; two-legal retaining old legal; two-legal adding new legal; general rule with URL-only t overlap. Selection never uses outcomes.",
            "limits": "Rules and groups differ in text composition, language, prompt length and lexical pool. Associations do not show internal reasoning or individual rule causality. Gold category versus rule identity is an exposure cross-tab, not an automatic relevance judgment.",
            "deferred": "Outcome-stratified detailed cases and per-source effect analysis remain Part 5; exclusion/temporal robustness Part 6; resources/synthesis Part 7; writer handover Part 8.",
            "actual_Qwen_calls": 0,
        },
    )
    print("Part 4 descriptive context contract recorded.", flush=True)


def run():
    contract = c.read(CONTRACT)
    c.verify(contract["inputs"])
    c.validate()
    _, cache = reader.read_verified()
    _, hp = h.verified()
    _, lp = l.verified()
    rows, inventory, roster, prompt_hashes = context_rows()
    by_key = {r["key"]: r for r in rows}
    results, evidence = ({}, {})
    for row in c.rows(cache / "predictions.jsonl"):
        ident = (row["key"], row["condition"], row["repeat"])
        require(
            ident not in results and row["messages_sha256"] == prompt_hashes[ident[:2]],
            "Result/context binding differs",
        )
        results[ident], evidence[ident] = (row["result"], row["evidence"])
    require(len(results) == 33093, "Prediction matrix incomplete")
    setup = c.read(cache / "prompt_setup.json")
    require(
        sorted((r["key"] for r in rows if r["same_messages_mixed_general"]))
        == sorted(setup["same_prompt_keys"]),
        "Identical-message group differs",
    )
    out = BASE / "runs" / f"run_{time.time_ns()}"
    out.mkdir(parents=True)
    write_rows(out / "contexts.jsonl", rows)
    save(out / "inventory.json", list(inventory.values()))
    selected_cases = defaultdict(list)
    rules = {
        "unchanged": lambda r: r["same_messages_mixed_general"],
        "zero_legal_membership_change": lambda r: r["original_legal_count"] == 0
        and r["general_change"] == "guideline_membership_changed",
        "zero_legal_reorder": lambda r: r["original_legal_count"] == 0
        and r["general_change"] == "same_guidelines_reordered",
        "one_legal_retains_general": lambda r: r["original_legal_count"] == 1
        and len(r["retained_general"]) == 1,
        "one_legal_replaces_general": lambda r: r["original_legal_count"] == 1
        and len(r["retained_general"]) == 0,
        "two_legal_retains_old_legal": lambda r: r["original_legal_count"] == 2
        and r["added_legal_status"] == "retained_original_legal",
        "two_legal_adds_new_legal": lambda r: r["original_legal_count"] == 2
        and r["added_legal_status"] == "new_legal_despite_original_legal",
        "url_only_t_overlap": lambda r: any(
            (
                x["only_t_from_url"] and x["positive_bm25"]
                for x in r["selected_lexical"]["general_2"].values()
            )
        ),
    }
    for phase in PHASES:
        full = [r for r in rows if r["phase"] == phase]
        source_path = lp if "legal" in phase else hp
        ep_result = c.read(source_path / phase / "results.json")
        keys = c.read(source_path / phase / "input_roster.json")["keys"]
        with np.load(source_path / phase / "arrays.npz", allow_pickle=False) as handle:
            arr = {
                name: handle[name] for name in ("gold", "predictions", "common_mask")
            }
        common = [by_key[k] for k, on in zip(keys, arr["common_mask"]) if on]
        require(
            [r["key"] for r in common] == ep_result["common_keys"],
            "Original endpoint mask changed",
        )
        groups = {
            "original_legal_count": defaultdict(list),
            "added_legal_status": defaultdict(list),
            "added_legal_id": defaultdict(list),
            "replacement_pattern": defaultdict(list),
        }
        for i, key in enumerate(keys):
            if not arr["common_mask"][i]:
                continue
            row = by_key[key]
            for name in (
                "original_legal_count",
                "added_legal_status",
                "added_legal_id",
            ):
                groups[name][str(row[name])].append(i)
            pattern = c.signature(row["guideline_ids"])
            groups["replacement_pattern"][pattern].append(i)
        outcomes = {
            name: {
                key: score_group(arr, selected, "legal" in phase)
                for key, selected in buckets.items()
            }
            for name, buckets in groups.items()
        }
        patterns = {}
        for row in full:
            pattern = c.signature(row["guideline_ids"])
            if pattern not in patterns:
                patterns[pattern] = {
                    "guideline_ids": row["guideline_ids"],
                    "full_count": 0,
                    "common_count": 0,
                }
            patterns[pattern]["full_count"] += 1
        for pattern, value in outcomes["replacement_pattern"].items():
            patterns[pattern]["common_count"] = value["n"]
        top = sorted(patterns, key=lambda key: (-patterns[key]["full_count"], key))[:8]
        cross = {}
        if "legal" in phase:
            for rid, indices in groups["added_legal_id"].items():
                gold = arr["gold"][indices]
                cross[rid] = {
                    "n": len(indices),
                    "empty_gold": int((gold.sum(axis=1) == 0).sum()),
                    "gold_categories": dict(zip(l.LABELS, gold.sum(axis=0).tolist())),
                }
        output = {
            "phase": phase,
            "full_exposure": exposure(full, inventory),
            "common_exposure": exposure(common, inventory),
            "by_source_full": {
                source: exposure([r for r in full if r["source"] == source], inventory)
                for source in sorted({r["source"] for r in full})
            },
            "by_source_common": {
                source: exposure(
                    [r for r in common if r["source"] == source], inventory
                )
                for source in sorted({r["source"] for r in common})
            },
            "descriptive_outcomes": outcomes,
            "replacement_patterns": patterns,
            "top_eight_patterns": top,
            "legal_gold_by_added_rule": cross,
            "fixed_main_results": ep_result["common"],
            "identical_prompt_variation": decision_variation(
                results,
                [r["key"] for r in full if r["same_messages_mixed_general"]],
                phase,
            ),
        }
        identical_indices = [
            i
            for i, key in enumerate(keys)
            if arr["common_mask"][i] and by_key[key]["same_messages_mixed_general"]
        ]
        output["identical_prompt_outcomes"] = (
            score_group(arr, identical_indices, "legal" in phase)
            if identical_indices
            else None
        )
        output["zero_legal_change_outcomes"] = {}
        for change in (
            "same_ordered_guidelines",
            "same_guidelines_reordered",
            "guideline_membership_changed",
        ):
            selected = [
                i
                for i, key in enumerate(keys)
                if arr["common_mask"][i]
                and by_key[key]["original_legal_count"] == 0
                and (by_key[key]["general_change"] == change)
            ]
            if selected:
                output["zero_legal_change_outcomes"][change] = score_group(
                    arr, selected, "legal" in phase
                )
        count_metrics = (
            ("extra_labels", "missed_labels", "exact_correct")
            if "legal" in phase
            else ("fp", "fn", "total_errors")
        )
        for ci, (left, right) in enumerate(h.PAIRS):
            for name in count_metrics:
                total = sum(
                    (
                        v["contrasts"][ci]["differences"][name]
                        for v in outcomes["original_legal_count"].values()
                    )
                )
                require(
                    abs(
                        total
                        - (
                            ep_result["common"][left]["mean"][name]
                            - ep_result["common"][right]["mean"][name]
                        )
                    )
                    < 1e-09,
                    "Subgroup error totals do not recover main results",
                )
        save(out / phase / "results.json", output)
        for reason, predicate in rules.items():
            pool = [r for r in full if predicate(r)]
            if pool:
                chosen = min(
                    pool,
                    key=lambda r: hashlib.sha256(
                        ("part4-context|" + r["key"]).encode()
                    ).hexdigest(),
                )
                selected_cases[chosen["key"]].append(reason)
        print(
            phase
            + ": saved delivery, replacements, identical prompts and descriptive context groups.",
            flush=True,
        )
    examples = {}
    for row in c.rows(c.BASE / "prompts.jsonl"):
        if row["key"] in selected_cases:
            examples.setdefault(row["key"], {})[row["condition"]] = row
    case_rows = []
    for key in sorted(selected_cases):
        case_rows.append(
            {
                "key": key,
                "phase": roster[key]["phase"],
                "source": roster[key]["source"],
                "selection_reasons": selected_cases[key],
                "text": roster[key]["text"],
                **case_gold(roster[key]),
                "context_change": by_key[key],
                "saved_prompts": examples[key],
                "predictions": {
                    cond: [
                        {
                            "result": results[key, cond, r],
                            "evidence": evidence[key, cond, r],
                        }
                        for r in range(3)
                    ]
                    for cond in CONDITIONS
                },
            }
        )
    write_rows(out / "context_examples.jsonl", case_rows)
    save(
        out / "summary.json",
        {
            "status": "saved_context_delivery_inspected",
            "contexts_checked": 11031,
            "result_prompt_bindings_checked": 33093,
            "identical_mixed_general_messages": sum(
                (r["same_messages_mixed_general"] for r in rows)
            ),
            "context_examples": len(case_rows),
            "full": exposure(rows, inventory),
            "new_retrieval_calls": 0,
            "actual_Qwen_calls": 0,
            "D_writes": 0,
        },
    )
    c.verify(contract["inputs"])
    c.validate()
    save(
        out / "source_and_preservation_check.json",
        {
            "status": "frozen_sources_and_earlier_parts_unchanged",
            "hashes": contract["inputs"],
            "actual_Qwen_calls": 0,
            "retrieval_reruns": 0,
            "D_writes": 0,
            "latex_edits": 0,
        },
    )
    save(
        out / "manifest.json",
        {
            "status": "part4_contexts_pending_independent_check",
            "run": c.ref(out),
            "created_utc": c.a.utc(),
            "outputs": c.bindings((p for p in out.rglob("*") if p.is_file())),
            "sources": c.bindings(own_sources()),
            "actual_Qwen_calls": 0,
        },
    )
    print(json.dumps({"manifest": c.ref(out / "manifest.json")}), flush=True)


def verified():
    done = c.read(BASE / "complete.json")
    require(
        done["status"] == "part4_numerically_verified",
        "Part 4 independent check missing",
    )
    c.verify(done["bindings"])
    manifest = c.read(c.ROOT / done["manifest"])
    c.verify(manifest["outputs"])
    return (manifest, c.ROOT / manifest["run"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    with offline():
        prepare() if args.mode == "prepare" else run()
