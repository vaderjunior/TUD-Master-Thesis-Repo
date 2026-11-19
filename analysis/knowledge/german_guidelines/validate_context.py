from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
from analysis.knowledge.german_guidelines import context as x
from analysis.knowledge.german_guidelines.verify_legal import (
    reference as legal_reference,
    same,
)
from analysis.knowledge.german_guidelines.verify_hate import rates as hate_reference


def check_summary(saved, y, p, legal):
    rows = [
        legal_reference(y, row, [0, 1, 2])[0] if legal else hate_reference(y, row)
        for row in p
    ]
    for i, row in enumerate(rows):
        same(saved["repeats"][i], row)
    for name in rows[0]:
        values = np.asarray(
            [np.nan if row[name] is None else row[name] for row in rows], dtype=float
        )
        same(saved["mean"][name], values.mean())
        same(saved["sample_sd"][name], values.std(ddof=1))


def verify(manifest_path):
    with x.offline():
        c = x.c
        contract = c.read(x.CONTRACT)
        c.verify(contract["inputs"])
        c.validate()
        manifest = c.read(manifest_path)
        x.require(
            manifest["status"] == "part4_contexts_pending_independent_check",
            "Wrong context checkpoint",
        )
        c.verify(manifest["outputs"])
        c.verify(manifest["sources"])
        out = c.ROOT / manifest["run"]
        rows = {r["key"]: r for r in c.rows(out / "contexts.jsonl")}
        inventory = {r["id"]: r for r in c.rows(c.BASE / "guideline_records.jsonl")}
        roster = {r["key"]: r for r in c.rows(c.BASE / "roster.jsonl")}
        traces = {r["key"]: r for r in c.rows(c.BASE / "retrieval_traces.jsonl")}
        prompts = defaultdict(dict)
        checks = Counter()
        for row in c.rows(c.BASE / "prompts.jsonl"):
            key, cond = (row["key"], row["condition"])
            ids = [r["id"] for r in row["context"]["guidelines"]]
            saved = rows[key]
            x.require(saved["guideline_ids"][cond] == ids, "Saved rule IDs differ")
            body = row["messages"][1]["text"]
            expected = "## Annotation guidelines\n" + "\n".join(
                ("- " + inventory[rid]["text"] for rid in ids)
            )
            x.require(expected in body, "Actual prompt lacks expected rule wording")
            before, after = body.split(expected, 1)
            x.require(after.startswith("\n\n## "), "Guideline block boundary differs")
            masked = [
                row["messages"][0],
                {"role": "user", "text": before + "<GUIDELINES>" + after},
            ]
            prompts[key][cond] = {
                "ids": ids,
                "hash": row["messages_sha256"],
                "masked": masked,
                "ref_tokens": row["reference_input_tokens"],
                "context": row["context"],
                "messages": row["messages"],
            }
            x.require(
                c.signature(row["messages"]) == saved["messages_sha256"][cond],
                "Prompt hash differs",
            )
            checks["prompt_text_checks"] += 1
        for key, p in prompts.items():
            r, original = (rows[key], roster[key])
            m, g, a = [p[k] for k in x.CONDITIONS]
            x.require(
                m["masked"] == g["masked"] == a["masked"], "Non-guideline text differs"
            )
            for cond in x.CONDITIONS:
                x.require(
                    {k: v for k, v in p[cond]["context"].items() if k != "guidelines"}
                    == {k: v for k, v in m["context"].items() if k != "guidelines"},
                    "Other context differs",
                )
            old_general = [
                rid for rid in m["ids"] if inventory[rid]["dimension"] == "hate"
            ]
            old_legal = [
                rid for rid in m["ids"] if inventory[rid]["dimension"] == "legal"
            ]
            x.require(
                a["ids"][:2] == g["ids"] and len(a["ids"]) == 3,
                "Added rule prefix differs",
            )
            same(r["original_legal_count"], len(old_legal))
            x.require(
                r["original_legal"] == old_legal
                and r["original_general"] == old_general,
                "Old rule type mismatch",
            )
            x.require(
                r["retained_general"] == [v for v in old_general if v in g["ids"]],
                "Retained general mismatch",
            )
            x.require(
                r["removed_general"] == [v for v in old_general if v not in g["ids"]],
                "Removed general mismatch",
            )
            x.require(
                r["introduced_general"]
                == [v for v in g["ids"] if v not in old_general],
                "Introduced general mismatch",
            )
            status = (
                "no_original_legal"
                if not old_legal
                else (
                    "retained_original_legal"
                    if a["ids"][2] in old_legal
                    else "new_legal_despite_original_legal"
                )
            )
            x.require(r["added_legal_status"] == status, "Legal retention mismatch")
            change = (
                "same_ordered_guidelines"
                if m["ids"] == g["ids"]
                else (
                    "same_guidelines_reordered"
                    if set(m["ids"]) == set(g["ids"])
                    else "guideline_membership_changed"
                )
            )
            x.require(r["general_change"] == change, "General change category differs")
            x.require(
                r["removed_original_legal_in_separate"]
                == [rid for rid in old_legal if rid != a["ids"][2]],
                "Removed legal IDs differ",
            )
            same(r["same_messages_mixed_general"], m["messages"] == g["messages"])
            examples = [
                hit for _, hits in m["context"]["example_groups"] for hit in hits
            ]
            annotations = [
                hit
                for hit in examples
                if hit["meta"].get("legal") is not None
                and (not hit["meta"].get("illustrative_only"))
            ]
            same(
                r["unchanged_legal_material"],
                {
                    "legal_definitions": sum(
                        (
                            hit["meta"].get("dimension") == "legal"
                            for hit in m["context"]["definitions"]
                        )
                    ),
                    "legal_annotated_examples": len(annotations),
                    "positive_legal_examples": sum(
                        (bool(hit["meta"]["legal"]) for hit in annotations)
                    ),
                    "empty_legal_examples": sum(
                        (hit["meta"]["legal"] == [] for hit in annotations)
                    ),
                    "legal_context_illustrations": sum(
                        (bool(hit["meta"].get("illustrative_only")) for hit in examples)
                    ),
                },
            )
            for cond in ("mixed_2", "general_2"):
                for rid, observation in r["selected_lexical"][cond].items():
                    overlap = sorted(
                        set(re.findall("\\w+", original["text"].lower()))
                        & set(re.findall("\\w+", inventory[rid]["text"].lower()))
                    )
                    x.require(
                        observation["shared_tokens"] == overlap, "Token overlap differs"
                    )
                    without = re.sub("https?://\\S+", "", original["text"], flags=re.I)
                    url_only = (
                        overlap == ["t"]
                        and "t" not in re.findall("\\w+", without.lower())
                        and (
                            re.search("https?://\\S+", original["text"], re.I)
                            is not None
                        )
                    )
                    same(observation["only_t_from_url"], url_only)
                    trace = traces[key][
                        "mixed_diagnostic" if cond == "mixed_2" else "general"
                    ]
                    idx = trace["candidates"].index(rid)
                    same(observation["saved_bm25_score"], trace["bm25_scores"][idx])
                    same(observation["positive_bm25"], trace["bm25_scores"][idx] > 0)
                    same(observation["only_t_overlap"], overlap == ["t"])
                    for name in ("dense", "bm25"):
                        expected_rank = (
                            trace[name + "_order"].index(rid) + 1
                            if rid in trace[name + "_order"]
                            else None
                        )
                        same(observation[name + "_rank"], expected_rank)
            old_trace, new_trace = (
                traces[key]["mixed_diagnostic"],
                traces[key]["general"],
            )
            ids = set(new_trace["candidates"])
            old_dense = [rid for rid in old_trace["dense_order"] if rid in ids]
            old_bm = [rid for rid in old_trace["bm25_order"] if rid in ids]
            shared_bm = set(old_bm) & set(new_trace["bm25_order"])
            same(
                r["trace_changes"],
                {
                    "dense_scores_same_to_1e12": all(
                        (
                            abs(
                                old_trace["dense_scores"][
                                    old_trace["candidates"].index(rid)
                                ]
                                - new_trace["dense_scores"][
                                    new_trace["candidates"].index(rid)
                                ]
                            )
                            <= 1e-12
                            for rid in ids
                        )
                    ),
                    "filtered_old_dense_top10_differs": old_dense
                    != new_trace["dense_order"],
                    "filtered_old_bm25_top10_differs": old_bm
                    != new_trace["bm25_order"],
                    "bm25_shared_hit_order_differs": [
                        rid for rid in old_bm if rid in shared_bm
                    ]
                    != [rid for rid in new_trace["bm25_order"] if rid in shared_bm],
                    "bm25_scores_changed": any(
                        (
                            abs(
                                old_trace["bm25_scores"][
                                    old_trace["candidates"].index(rid)
                                ]
                                - new_trace["bm25_scores"][
                                    new_trace["candidates"].index(rid)
                                ]
                            )
                            > 1e-12
                            for rid in ids
                        )
                    ),
                },
            )
            checks["replacement_and_literal_overlap_checks"] += 1
        _, cache = x.reader.read_verified()
        outputs = {}
        for row in c.rows(cache / "predictions.jsonl"):
            key, cond, rep = (row["key"], row["condition"], row["repeat"])
            x.require(
                row["messages_sha256"] == prompts[key][cond]["hash"],
                "Generated result bound to wrong prompt",
            )
            outputs[key, cond, rep] = row["result"]
            checks["result_prompt_bindings"] += 1
        _, hp = x.h.verified()
        _, lp = x.l.verified()
        for phase in x.PHASES:
            result = c.read(out / phase / "results.json")
            full = [r for r in rows.values() if r["phase"] == phase]
            numeric = lp if "legal" in phase else hp
            keys = c.read(numeric / phase / "input_roster.json")["keys"]
            with np.load(numeric / phase / "arrays.npz", allow_pickle=False) as f:
                y, p, mask = (f["gold"], f["predictions"], f["common_mask"])
            common = [rows[key] for key, on in zip(keys, mask) if on]
            checks_to_run = [
                (result["full_exposure"], full),
                (result["common_exposure"], common),
            ]
            for suffix, selected in [("full", full), ("common", common)]:
                for source in sorted({r["source"] for r in selected}):
                    checks_to_run.append(
                        (
                            result["by_source_" + suffix][source],
                            [r for r in selected if r["source"] == source],
                        )
                    )
            for saved, selected in checks_to_run:
                same(saved["n"], len(selected))
                for field in (
                    "original_legal_count",
                    "general_change",
                    "added_legal_status",
                    "added_legal_id",
                ):
                    x.require(
                        saved["counts"][field]
                        == dict(Counter((str(r[field]) for r in selected))),
                        "Exposure histogram differs",
                    )
                same(
                    saved["same_messages"],
                    sum((r["same_messages_mixed_general"] for r in selected)),
                )
                for name, field in [
                    ("general_retained_counts", "retained_general"),
                    ("general_removed_counts", "removed_general"),
                    (
                        "legal_removed_in_separate_counts",
                        "removed_original_legal_in_separate",
                    ),
                ]:
                    same(
                        saved[name],
                        dict(Counter((str(len(r[field])) for r in selected))),
                    )
                same(
                    saved["zero_original_legal"],
                    {
                        name: sum(
                            (
                                r["original_legal_count"] == 0
                                and r["general_change"] == name
                                for r in selected
                            )
                        )
                        for name in (
                            "same_ordered_guidelines",
                            "same_guidelines_reordered",
                            "guideline_membership_changed",
                        )
                    },
                )
                same(
                    saved["trace_change_counts"],
                    {
                        name: sum((r["trace_changes"][name] for r in selected))
                        for name in selected[0]["trace_changes"]
                    },
                )
                for name, counts in saved["unchanged_legal_material"].items():
                    same(
                        counts,
                        {
                            "inputs_with_any": sum(
                                (
                                    r["unchanged_legal_material"][name] > 0
                                    for r in selected
                                )
                            ),
                            "slot_total": sum(
                                (r["unchanged_legal_material"][name] for r in selected)
                            ),
                        },
                    )
                for cond, counts in saved["lexical"].items():
                    observations = [
                        o
                        for r in selected
                        for rid, o in r["selected_lexical"][cond].items()
                        if inventory[rid]["dimension"] == "hate"
                    ]
                    same(
                        counts["positive_selected_general_slots"],
                        sum((o["positive_bm25"] for o in observations)),
                    )
                    same(
                        counts["only_t_positive_slots"],
                        sum(
                            (
                                o["positive_bm25"] and o["only_t_overlap"]
                                for o in observations
                            )
                        ),
                    )
                    same(
                        counts["only_url_t_positive_slots"],
                        sum(
                            (
                                o["positive_bm25"] and o["only_t_from_url"]
                                for o in observations
                            )
                        ),
                    )
                    same(
                        counts["only_url_t_input_keys"],
                        [
                            r["key"]
                            for r in selected
                            if any(
                                (
                                    o["positive_bm25"] and o["only_t_from_url"]
                                    for rid, o in r["selected_lexical"][cond].items()
                                    if inventory[rid]["dimension"] == "hate"
                                )
                            )
                        ],
                    )
                for cond in x.CONDITIONS:
                    freq = Counter(
                        (v for r in selected for v in prompts[r["key"]][cond]["ids"])
                    )
                    for rid, counts in saved["rule_frequency"][cond].items():
                        same(counts["inputs"], freq[rid])
                        for pos in range(1, 4):
                            same(
                                counts["positions"][str(pos)],
                                sum(
                                    (
                                        len(prompts[r["key"]][cond]["ids"]) >= pos
                                        and prompts[r["key"]][cond]["ids"][pos - 1]
                                        == rid
                                        for r in selected
                                    )
                                ),
                            )
                for left, right in [
                    ("general_2", "mixed_2"),
                    ("general_2_legal_1", "mixed_2"),
                    ("general_2_legal_1", "general_2"),
                ]:
                    v = np.array(
                        [
                            prompts[r["key"]][left]["ref_tokens"]
                            - prompts[r["key"]][right]["ref_tokens"]
                            for r in selected
                        ]
                    )
                    same(
                        saved["tokens"][left + "_minus_" + right],
                        dict(
                            min=int(v.min()),
                            max=int(v.max()),
                            mean=float(v.mean()),
                            median=float(np.median(v)),
                            p95=float(np.quantile(v, 0.95)),
                        ),
                    )
                checks["exposure_and_frequency_tables"] += 1
            for family, values in result["descriptive_outcomes"].items():
                for group, saved in values.items():
                    selected = [
                        i
                        for i, key in enumerate(keys)
                        if mask[i]
                        and (
                            x.c.signature(rows[key]["guideline_ids"]) == group
                            if family == "replacement_pattern"
                            else str(rows[key][family]) == group
                        )
                    ]
                    same(saved["n"], len(selected))
                    for ci, cond in enumerate(x.CONDITIONS):
                        check_summary(
                            saved["common"][cond],
                            y[selected],
                            p[ci, :, selected].swapaxes(0, 1),
                            "legal" in phase,
                        )
                    checks["descriptive_subgroups"] += 1
            for saved, selected in [
                (
                    result["identical_prompt_outcomes"],
                    [
                        i
                        for i, key in enumerate(keys)
                        if mask[i] and rows[key]["same_messages_mixed_general"]
                    ],
                ),
                *[
                    (
                        v,
                        [
                            i
                            for i, key in enumerate(keys)
                            if mask[i]
                            and rows[key]["original_legal_count"] == 0
                            and (rows[key]["general_change"] == name)
                        ],
                    )
                    for name, v in result["zero_legal_change_outcomes"].items()
                ],
            ]:
                same(saved["n"], len(selected))
                for ci, cond in enumerate(x.CONDITIONS):
                    check_summary(
                        saved["common"][cond],
                        y[selected],
                        p[ci, :, selected].swapaxes(0, 1),
                        "legal" in phase,
                    )
                checks["identical_and_zero_legal_subgroups"] += 1
            variation = result["identical_prompt_variation"]
            same_keys = [
                key
                for key in keys
                if prompts[key]["mixed_2"]["messages"]
                == prompts[key]["general_2"]["messages"]
            ]
            valid = [
                key
                for key in same_keys
                if all(
                    (
                        outputs[key, cond, r] is not None
                        for cond in x.CONDITIONS
                        for r in range(3)
                    )
                )
            ]
            x.require(
                variation["common_keys"] == valid, "Identical prompt mask differs"
            )

            def equality(a, b, field):
                if field == "all_structured_fields":
                    return all(
                        (
                            equality(a, b, f)
                            for f in (
                                "hate",
                                "legal",
                                "hate_type",
                                "target_group",
                                "severity",
                            )
                        )
                    )
                return (
                    set(a[field]) == set(b[field])
                    if isinstance(a[field], list)
                    else a[field] == b[field]
                )

            for field, v in variation["fields"].items():
                counts = [
                    [
                        sum(
                            (
                                not equality(
                                    outputs[key, "mixed_2", a],
                                    outputs[key, "general_2", b],
                                    field,
                                )
                                for key in valid
                            )
                        )
                        for b in range(3)
                    ]
                    for a in range(3)
                ]
                same(v["cross_all_nine_counts"], counts)
                same(v["mean_cross_count"], np.mean(counts))
                for cond in ("mixed_2", "general_2"):
                    expected = [
                        sum(
                            (
                                not equality(
                                    outputs[key, cond, a], outputs[key, cond, b], field
                                )
                                for key in valid
                            )
                        )
                        for a, b in combinations(range(3), 2)
                    ]
                    same(v["within_three_pair_counts"][cond], expected)
                checks["identical_prompt_field_checks"] += 1
            patterns = result["replacement_patterns"]
            tally = Counter((c.signature(r["guideline_ids"]) for r in full))
            x.require(set(patterns) == set(tally), "Pattern roster differs")
            for k, n in tally.items():
                same(patterns[k]["full_count"], n)
            x.require(
                result["top_eight_patterns"]
                == sorted(tally, key=lambda k: (-tally[k], k))[:8],
                "Pattern report selection differs",
            )
            if "legal" in phase:
                for rid, v in result["legal_gold_by_added_rule"].items():
                    selected = [
                        i
                        for i, key in enumerate(keys)
                        if mask[i] and rows[key]["added_legal_id"] == rid
                    ]
                    same(v["n"], len(selected))
                    same(v["empty_gold"], int((y[selected].sum(axis=1) == 0).sum()))
                    same(
                        v["gold_categories"],
                        dict(zip(x.l.LABELS, y[selected].sum(axis=0))),
                    )
        examples = list(c.rows(out / "context_examples.jsonl"))
        expected_examples = defaultdict(list)
        for phase in x.PHASES:
            candidates = [r for r in rows.values() if r["phase"] == phase]
            pools = {
                "unchanged": [
                    r for r in candidates if r["same_messages_mixed_general"]
                ],
                "zero_legal_membership_change": [
                    r
                    for r in candidates
                    if r["original_legal_count"] == 0
                    and r["general_change"] == "guideline_membership_changed"
                ],
                "zero_legal_reorder": [
                    r
                    for r in candidates
                    if r["original_legal_count"] == 0
                    and r["general_change"] == "same_guidelines_reordered"
                ],
                "one_legal_retains_general": [
                    r
                    for r in candidates
                    if r["original_legal_count"] == 1
                    and len(r["retained_general"]) == 1
                ],
                "one_legal_replaces_general": [
                    r
                    for r in candidates
                    if r["original_legal_count"] == 1 and (not r["retained_general"])
                ],
                "two_legal_retains_old_legal": [
                    r
                    for r in candidates
                    if r["original_legal_count"] == 2
                    and r["added_legal_status"] == "retained_original_legal"
                ],
                "two_legal_adds_new_legal": [
                    r
                    for r in candidates
                    if r["original_legal_count"] == 2
                    and r["added_legal_status"] == "new_legal_despite_original_legal"
                ],
                "url_only_t_overlap": [
                    r
                    for r in candidates
                    if any(
                        (
                            o["positive_bm25"] and o["only_t_from_url"]
                            for o in r["selected_lexical"]["general_2"].values()
                        )
                    )
                ],
            }
            for reason, pool in pools.items():
                if pool:
                    key = sorted(
                        pool,
                        key=lambda r: hashlib.sha256(
                            ("part4-context|" + r["key"]).encode()
                        ).hexdigest(),
                    )[0]["key"]
                    expected_examples[key].append(reason)
        x.require(
            {r["key"]: r["selection_reasons"] for r in examples}
            == dict(expected_examples),
            "Deterministic context case selection differs",
        )
        for row in examples:
            key = row["key"]
            same(row["gold_legal"], roster[key].get("legal"))
            same(row["gold_hate"], roster[key].get("gate"))
            for cond in x.CONDITIONS:
                x.require(
                    row["saved_prompts"][cond]["messages"]
                    == prompts[key][cond]["messages"],
                    "Case context differs",
                )
                for rep in range(3):
                    x.require(
                        row["predictions"][cond][rep]["result"]
                        == outputs[key, cond, rep],
                        "Case answer differs",
                    )
            checks["context_examples_bound"] += 1
        c.verify(contract["inputs"])
        c.verify(manifest["outputs"])
        save = {
            "status": "independent_saved_context_analysis_pass",
            "checks": dict(checks),
            "actual_Qwen_calls": 0,
            "retrieval_reruns": 0,
            "D_writes": 0,
        }
        x.save(out / "independent_check.json", save)
        x.save(
            x.BASE / "complete.json",
            {
                "status": "part4_numerically_verified",
                "created_utc": c.a.utc(),
                "manifest": c.ref(manifest_path),
                "bindings": c.bindings(
                    [
                        manifest_path,
                        out / "independent_check.json",
                        x.CONTRACT,
                        *x.own_sources(),
                    ]
                ),
                "actual_Qwen_calls": 0,
                "D_writes": 0,
            },
        )
        print(json.dumps(save), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    verify(Path(parser.parse_args().manifest))
