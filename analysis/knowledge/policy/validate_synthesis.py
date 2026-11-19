import argparse
import ast
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
from unittest.mock import patch
import numpy as np
from analysis.knowledge.policy import synthesis as p
from analysis.knowledge.policy import report_tables as report

c = p.c


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def at(root, pointer):
    result = root
    if pointer:
        assert pointer[0] == "/"
        for part in pointer[1:].split("/"):
            name = part.replace("~1", "/").replace("~0", "~")
            result = result[int(name)] if isinstance(result, list) else result[name]
    return result


def verify():
    cache = {}
    field_count = 0
    bindings = {}
    data = load(p.BASE / "comparison_catalogue.json")
    rows = data["records"]
    byid = {r["id"]: r for r in rows}
    assert len(rows) == len(byid) == 421
    assert dict(Counter((r["study"] for r in rows))) == data["by_study"]
    assert dict(Counter((r["metric"] for r in rows))) == data["by_metric"]
    for row in rows:
        path = c.ROOT / row["source"]
        if row["source"] not in cache:
            cache[row["source"]] = load(path)
            assert c.sha(path) == row["source_sha256"]
            bindings[row["source"]] = row["source_sha256"]
        for field, ptr in row["source_fields"].items():
            assert row[field] == at(cache[row["source"]], ptr), (row["id"], field)
            field_count += 1
        interval = (
            row["family_interval"]
            if row["family_interval"] is not None
            else row["ci95"]
        )
        confidence = (
            row["family_confidence"] if row["family_confidence"] is not None else 0.95
        )
        assert (
            interval == row["preferred_interval"]
            and confidence == row["preferred_confidence"]
        )
        assert row["better_direction"] == (
            "lower" if row["metric"] == "false_positive_rate" else "higher"
        )
    expected_macro = 7 + sum(
        (
            sum(
                (
                    len(v["contrasts"])
                    for v in load(p.RESULT / "evidence" / name)["endpoints"].values()
                )
            )
            for name in (
                "sq1_strategy_completion_20260921_v1.json",
                "sq1_components_completion_20260924_v1.json",
                "sq1_allocation_completion_20260925_v1.json",
                "sq1_pool_completion_20260928_v1.json",
            )
        )
    )
    expected_micro = sum(
        (
            sum(
                (
                    len(load(c.ROOT / path)["contrasts"])
                    for path in load(folder / "accepted_analysis_index.json")["outputs"]
                )
            )
            for folder in (p.BANKS, p.COMPONENTS)
        )
    )
    assert expected_macro == 224 and expected_micro == 130
    assert data["by_metric"] == dict(
        false_positive_rate=10,
        macro_f1=expected_macro + 12 + 18,
        micro_f1=expected_micro + 9,
        mean_pairwise_repeat_agreement=18,
    )
    cohorts = load(p.BASE / "cohort_relationships.json")
    sets = {}
    for group in cohorts["groups"]:
        if group["study"] == "complete_policy":
            source = load(c.ROOT / group["source"])[group["endpoint"]]["keys"]
            mask = np.load(c.ROOT / group["common_mask_source"], allow_pickle=False)[
                "common"
            ].tolist()
            original_common = [k for k, on in zip(source, mask, strict=True) if on]
        else:
            v = load(c.ROOT / group["source"])
            lang = group["endpoint"][:2] + "/"
            source = [
                key if key[:3] in ("en/", "de/") else lang + key for key in v["ids"]
            ]
            original_common = [
                key if key[:3] in ("en/", "de/") else lang + key
                for key in v["common_ids"]
            ]
        assert (
            sorted(source) == group["roster_ids"]
            and sorted(original_common) == group["common_ids"]
        )
        sets[group["id"]] = (set(source), set(original_common))
    for pair in cohorts["pairs"]:
        left, right = (sets[pair["left"]], sets[pair["right"]])
        assert pair["shared_roster"] == len(left[0].intersection(right[0]))
        assert pair["shared_common"] == len(left[1].intersection(right[1]))
        assert pair["roster_n"] == [len(left[0]), len(right[0])]
        assert pair["common_n"] == [len(left[1]), len(right[1])]
    assert len(sets) == 28 and len(cohorts["pairs"]) == 98

    def overlap(a, b, which):
        return len(sets[a][which] & sets[b][which])

    assert (
        overlap(
            "pools/en_dev_eval_sq3_types/paired",
            "components/en_dev_eval_sq3_types/factorial",
            1,
        )
        == 465
    )
    assert (
        overlap(
            "strategy/en_dev_eval_sq3_types/paired", "complete_policy/en_hate_type", 0
        )
        == 0
    )
    assert (
        overlap(
            "german_guidelines/de_legal_test/legal/paired",
            "complete_policy/de_legal",
            0,
        )
        == 177
    )
    assert (
        overlap(
            "german_guidelines/de_legal_test/legal/paired",
            "complete_policy/de_legal",
            1,
        )
        == 174
    )
    assert (
        overlap(
            "strategy/en_dev_eval_sq3_types/paired",
            "strategy/en_dev_eval_types/paired",
            0,
        )
        == 0
    )
    appendix = (p.BASE / "numerical_appendix.md").read_text(encoding="utf-8")
    tables = load(p.BASE / "report_tables.json")
    published = 0
    for table in tables:
        rendered = table["markdown"].splitlines()[2:]
        assert len(rendered) == len(table["choices"])
        for line, choice in zip(rendered, table["choices"], strict=True):
            fields = [x.strip() for x in line.split("|")[1:-1]]
            row = byid[choice["id"]]
            assert (
                fields[0] == choice["label"]
                and int(fields[1].replace(",", "")) == row["n"]
            )
            assert fields[2] == format(row["difference"] * 100, "+.2f")
            assert (
                fields[3]
                == "["
                + ", ".join(
                    (format(x * 100, "+.2f") for x in row["preferred_interval"])
                )
                + "]"
            )
            assert fields[4].startswith(
                f"{row['preferred_confidence'] * 100:.4f}".rstrip("0").rstrip(".") + "%"
            )
            assert ("family" in fields[4]) == (row["family_interval"] is not None)
            published += 1
    appendix_rows = [line for line in appendix.splitlines() if line.startswith("| `")]
    assert len(appendix_rows) == 421
    for line in appendix_rows:
        fields = [x.strip() for x in line.split("|")[1:-1]]
        row = byid[fields[0].strip("`")]
        assert fields[1] == str(row["n"]) and fields[2] == format(
            row["difference"] * 100, "+.4f"
        )
        assert (
            fields[3]
            == "["
            + ", ".join((format(x * 100, "+.4f") for x in row["preferred_interval"]))
            + "]"
        )
        assert ("family" in fields[4]) == (row["family_interval"] is not None)
    claims = load(p.BASE / "scientific_claims.json")
    assert len(claims["claims"]) == 15
    for claim in claims["claims"]:
        for identity in claim["comparison_ids"]:
            assert identity in byid
        c.verify(claim["context_reports"])
    assert claims["comparison_catalogue_sha256"] == c.sha(
        p.BASE / "comparison_catalogue.json"
    )
    assert claims["cohort_relationships_sha256"] == c.sha(
        p.BASE / "cohort_relationships.json"
    )
    legal = byid[
        "complete_policy:de_legal:complete_policy_minus_retrieved_policy:macro_f1"
    ]
    extra = byid[
        "complete_policy:de_legal:complete_policy_minus_complete_definitions:macro_f1"
    ]
    sev = byid[
        "complete_policy:en_severity:complete_policy_minus_retrieved_policy:macro_f1"
    ]
    agreement = byid[
        "complete_policy:en_hate_type:complete_policy_minus_retrieved_policy:mean_pairwise_repeat_agreement"
    ]
    alloc = byid["allocation:en_dev_eval_main:proportional_minus_flat2:macro_f1"]
    assert legal["preferred_interval"][0] > 0 and legal["source_role"] == "primary"
    assert (
        extra["preferred_interval"][0] < 0 < extra["preferred_interval"][1]
        and extra["source_role"] == "supporting"
    )
    assert sev["family_interval"] is None and sev["source_role"] == "supporting"
    assert (
        agreement["preferred_interval"][1] < 0
        and agreement["metric"] == "mean_pairwise_repeat_agreement"
    )
    assert alloc["ci95"][0] > 0 and alloc["preferred_interval"][0] < 0
    oracle = byid["guideline_oracle:en_benign:matched_minus_none:false_positive_rate"]
    assert oracle["better_direction"] == "lower" and oracle["preferred_interval"][1] < 0
    macro = byid["components:en_dev_eval_targets:examples_alone:macro_f1"]
    micro = byid["components:en_dev_eval_targets:examples_alone:micro_f1"]
    assert (
        macro["preferred_interval"][1] < 0
        and micro["preferred_interval"][0] < 0 < micro["preferred_interval"][1]
    )
    facts = []

    def fact(path, ptr, format_spec, scale, expected):
        value = at(load(path), ptr)
        assert format(value * scale, format_spec) == expected, (ptr, value, expected)
        facts.append(
            dict(
                source=c.ref(path),
                source_sha256=c.sha(path),
                pointer=ptr,
                value=value,
                scale=scale,
                formatted=expected,
            )
        )

    path = p.run_root() / "de_legal/results.json"
    for ptr, fmt, scale, expected in [
        ("/point/exact/0", ".2f", 100, "62.90"),
        ("/point/exact/2", ".2f", 100, "73.82"),
        ("/point/empty_gold_n", "d", 1, "96"),
        ("/point/nonempty_gold_n", "d", 1, "81"),
        ("/point/empty_gold_exact/0", ".2f", 100, "57.64"),
        ("/point/empty_gold_exact/2", ".2f", 100, "76.74"),
        ("/point/nonempty_gold_jaccard/0", ".3f", 1, "0.796"),
        ("/point/nonempty_gold_jaccard/2", ".3f", 1, "0.779"),
        ("/point/recall/0/2", ".3f", 1, "0.900"),
        ("/point/recall/2/2", ".3f", 1, "0.711"),
        ("/point/repeatability/stable_correct/0", ".2f", 100, "55.37"),
        ("/point/repeatability/stable_correct/2", ".2f", 100, "63.84"),
        ("/point/repeatability/stable_wrong/0", ".2f", 100, "22.60"),
        ("/point/repeatability/stable_wrong/2", ".2f", 100, "9.04"),
    ]:
        fact(path, ptr, fmt, scale, expected)
    assert load(path)["meta"]["all_labels"][2] == "insult_defamation"
    for ptr, expected in [
        ("/point/repeatability/pairwise_agreement/0", "67.61"),
        ("/point/repeatability/pairwise_agreement/2", "63.91"),
    ]:
        fact(p.run_root() / "en_hate_type/results.json", ptr, ".2f", 100, expected)
    for cond, expected in zip(c.CONDITIONS, ("66.32", "106.81", "168.64"), strict=True):
        ptr = "/groups/condition~1" + cond + "/usage/all_success/total_tokens/sum"
        fact(p.P6 / "resources.json", ptr, ".2f", 1e-06, expected)
    fact(p.P6 / "timing.json", "/wall/invocation_s", ".3f", 1 / 3600, "10.501")
    fact(
        p.P4 / "context_summary.json",
        "/populations/de_hate/any_dimension_definition_A/hate",
        "d",
        1,
        "38",
    )
    context = load(p.P4 / "context_summary.json")["counts"]
    assert context["examples_replaced"] == 0 and context["all_example_groups_unchanged"]
    assert (
        context["definitions_added_per_B_or_C"] == 19
        and context["guidelines_added_per_C"] == 17
    )
    res = load(p.P6 / "resources.json")["groups"]
    assert [res["condition/" + cond]["null_final"] for cond in c.CONDITIONS] == [
        31,
        0,
        0,
    ]
    links = 0
    captured = {}

    def capture(path, text):
        captured[c.ref(path)] = text.rstrip() + "\n"

    captured_json = {}
    with patch.object(report, "write_new", capture), patch.object(
        c, "save", lambda path, value: captured_json.update({c.ref(path): value})
    ):
        report.build()
    assert captured[c.ref(p.BASE / "numerical_appendix.md")] == appendix
    assert captured_json[c.ref(p.BASE / "report_tables.json")] == tables
    for source_path in [
        c.ROOT / "analysis/knowledge/policy/synthesis.py",
        c.ROOT / "analysis/knowledge/policy/report_tables.py",
        c.ROOT / "analysis/knowledge/policy/validate_synthesis.py",
    ]:
        ast.parse(source_path.read_text(encoding="utf-8"))
    c.verify(load(p.BASE / "analysis_contract.json")["inputs"])
    c.save(
        p.BASE / "final_verification.json",
        dict(
            status="PASS",
            created_utc=c.a.utc(),
            comparison_rows=len(rows),
            source_fields=field_count,
            source_documents=len(cache),
            source_bindings=bindings,
            cohort_rosters=len(sets),
            same_dimension_overlap_rows=len(cohorts["pairs"]),
            publication_tables=len(tables),
            main_numeric_table_rows=published,
            appendix_numeric_rows=len(appendix_rows),
            scientific_claims=len(claims["claims"]),
            narrative_machine_fact_checks=facts,
            numerical_tables_reproduced_exactly=True,
            publication_bindings=c.bindings(
                [
                    p.BASE / "scientific_claims.json",
                    p.BASE / "comparison_catalogue.json",
                    p.BASE / "cohort_relationships.json",
                    p.BASE / "numerical_appendix.md",
                    p.BASE / "report_tables.json",
                ]
            ),
            all_frozen_sources_unchanged=True,
            new_F1_calculations=0,
            new_bootstraps=0,
            new_calls=0,
            semantic_checks="Original families retained; oracle FPR direction; legal primary versus conditional extra guidelines; severity supporting; agreement separate from correctness; positive ordinary versus unresolved adjusted allocation; macro versus micro target qualification; identical sample counts with different retained IDs.",
            manual_review="Report read chapter by chapter against original reports. Legal trade-offs, actual delivery, missing outputs, output-selected sensitivities, late exposure, subgroup limitations, repeated-text reuse, no example displacement, no full interaction and token/runtime boundaries retained.",
        ),
    )
    print("Source fields, cohorts and numerical tables verified.", flush=True)


if __name__ == "__main__":
    with c.offline():
        verify()
