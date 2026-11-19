from collections import Counter
from analysis.knowledge.policy import synthesis as p

c = p.c


def write_new(path, text):
    c.require(c.normal(path).is_relative_to(c.ROOT), "Output outside project")
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text.rstrip() + "\n")


def build():
    values = c.read(p.BASE / "comparison_catalogue.json")
    rows = {x["id"]: x for x in values["records"]}
    tables = []

    def table(title, choices):
        lines = [
            "| Within-study comparison | Shared texts | Change, pp | Interval, pp | Interval level |",
            "| --- | ---: | ---: | --- | --- |",
        ]
        for name, key in choices:
            x = rows[key]
            lo, hi = x["preferred_interval"]
            level = f"{100 * x['preferred_confidence']:.4f}".rstrip("0").rstrip(".")
            lines.append(
                f"| {name} | {x['n']:,} | {100 * x['difference']:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] | {level}%"
                + (" family" if x["family_interval"] is not None else " descriptive")
                + " |"
            )
        rendered = "\n".join(lines)
        tables.append(
            dict(
                title=title,
                choices=[dict(label=name, id=key) for name, key in choices],
                markdown=rendered,
            )
        )
        return rendered

    bank_table = table(
        "Example banks and retrieval",
        [
            (
                "Core: full − curated compact, English hate macro",
                "core:en_dev_eval_main:full_minus_small:macro_f1",
            ),
            (
                "Core: full − curated compact, German legal macro",
                "core:de_legal_dev_eval:full_minus_small:macro_f1",
            ),
            (
                "Core: full − curated compact, main type macro",
                "core:en_dev_eval_sq3_types:full_minus_small:macro_f1",
            ),
            (
                "New pool study: full − random compact, main type macro",
                "pools:en_dev_eval_sq3_types:full_reference_minus_uniform_compact:macro_f1",
            ),
            (
                "New pool study: curated − random compact, main type macro",
                "pools:en_dev_eval_sq3_types:curated_small_minus_uniform_compact:macro_f1",
            ),
            (
                "New pool study: full − random compact, main type micro",
                "pools:en_dev_eval_sq3_types:full_reference_minus_uniform_compact:micro_f1",
            ),
            (
                "New pool study: curated − random compact, main type micro",
                "pools:en_dev_eval_sq3_types:curated_small_minus_uniform_compact:micro_f1",
            ),
            (
                "Strategy: hybrid − BM25, main type macro",
                "strategy:en_dev_eval_sq3_types:hybrid_minus_bm25:macro_f1",
            ),
            (
                "Strategy: dense − hybrid, main type macro",
                "strategy:en_dev_eval_sq3_types:dense_minus_hybrid:macro_f1",
            ),
        ],
    )
    component_table = table(
        "Component and definition-allocation comparisons",
        [
            (
                "Add five examples to definitions + guidelines, type macro",
                "components:en_dev_eval_sq3_types:examples_given_definitions_guidelines:macro_f1",
            ),
            (
                "Same example contrast, type micro",
                "components:en_dev_eval_sq3_types:examples_given_definitions_guidelines:micro_f1",
            ),
            (
                "Add two definitions to examples + guidelines, type macro",
                "components:en_dev_eval_sq3_types:definitions_given_examples_guidelines:macro_f1",
            ),
            (
                "Add two guidelines to examples + definitions, type macro",
                "components:en_dev_eval_sq3_types:guidelines_given_examples_definitions:macro_f1",
            ),
            (
                "Examples alone − no retrieved context, target macro",
                "components:en_dev_eval_targets:examples_alone:macro_f1",
            ),
            (
                "Same target contrast, micro",
                "components:en_dev_eval_targets:examples_alone:micro_f1",
            ),
            (
                "Seven pooled definitions − two, type macro",
                "allocation:en_dev_eval_sq3_types:flat7_minus_flat2:macro_f1",
            ),
            (
                "Seven allocated definitions − two, type macro",
                "allocation:en_dev_eval_sq3_types:proportional_minus_flat2:macro_f1",
            ),
            (
                "Seven allocated definitions − two, English hate macro",
                "allocation:en_dev_eval_main:proportional_minus_flat2:macro_f1",
            ),
        ],
    )
    guideline_table = table(
        "Benign guideline delivery",
        [
            (
                "Matched rule − no guidelines, false-positive rate",
                "guideline_oracle:en_benign:matched_minus_none:false_positive_rate",
            ),
            (
                "Matched rule − alternative rule, false-positive rate",
                "guideline_oracle:en_benign:matched_minus_alternative:false_positive_rate",
            ),
            (
                "Matched rule − filler only, false-positive rate",
                "guideline_oracle:en_benign:matched_minus_filler:false_positive_rate",
            ),
            (
                "Natural retrieval − no guidelines, false-positive rate",
                "guideline_oracle:en_benign:natural_minus_none:false_positive_rate",
            ),
        ],
    )
    german_table = table(
        "German guideline separation",
        [
            (
                "General-only − mixed guidelines, German TEST hate macro",
                "german_guidelines:de_test:general_2_minus_mixed_2:macro_f1",
            ),
            (
                "General + legal slot − mixed guidelines, German TEST hate macro",
                "german_guidelines:de_test:general_2_legal_1_minus_mixed_2:macro_f1",
            ),
            (
                "General-only − mixed guidelines, TEST legal macro",
                "german_guidelines:de_legal_test:general_2_minus_mixed_2:macro_f1",
            ),
            (
                "General + legal slot − mixed guidelines, TEST legal macro",
                "german_guidelines:de_legal_test:general_2_legal_1_minus_mixed_2:macro_f1",
            ),
        ],
    )
    policy_table = table(
        "Complete policy versus original retrieved policy",
        [
            (
                "English hate, macro",
                "complete_policy:en_hate:complete_policy_minus_retrieved_policy:macro_f1",
            ),
            (
                "English types, macro",
                "complete_policy:en_hate_type:complete_policy_minus_retrieved_policy:macro_f1",
            ),
            (
                "German hate, macro",
                "complete_policy:de_hate:complete_policy_minus_retrieved_policy:macro_f1",
            ),
            (
                "German legal, macro",
                "complete_policy:de_legal:complete_policy_minus_retrieved_policy:macro_f1",
            ),
            (
                "English targets, macro",
                "complete_policy:en_target_group:complete_policy_minus_retrieved_policy:macro_f1",
            ),
            (
                "English severity, macro",
                "complete_policy:en_severity:complete_policy_minus_retrieved_policy:macro_f1",
            ),
            (
                "German legal, complementary micro",
                "complete_policy:de_legal:complete_policy_minus_retrieved_policy:micro_f1",
            ),
            (
                "English types, repeat agreement",
                "complete_policy:en_hate_type:complete_policy_minus_retrieved_policy:mean_pairwise_repeat_agreement",
            ),
        ],
    )
    stages_table = table(
        "Legal expansion stages",
        [
            (
                "All definitions − retrieved policy, legal macro",
                "complete_policy:de_legal:complete_definitions_minus_retrieved_policy:macro_f1",
            ),
            (
                "Complete policy − all definitions, legal macro",
                "complete_policy:de_legal:complete_policy_minus_complete_definitions:macro_f1",
            ),
            (
                "All definitions − retrieved policy, legal micro",
                "complete_policy:de_legal:complete_definitions_minus_retrieved_policy:micro_f1",
            ),
            (
                "Complete policy − all definitions, legal micro",
                "complete_policy:de_legal:complete_policy_minus_complete_definitions:micro_f1",
            ),
        ],
    )
    c.save(p.BASE / "report_tables.json", tables)
    lines = [
        "# Part 7 numerical appendix",
        "",
        "All values below are copied from accepted within-study results. Nothing is rescored. Differences and intervals are in percentage points; sample sizes count shared texts, not calls or repeat pairs. A lower false-positive rate is better; the other displayed metrics use higher values as better.",
        "",
        "Family levels are local to their original questions. The full catalogue is not a new multiple-testing family. Core complementary micro and SQ2 bridging results remain in their existing reports and are not recopied into this catalogue.",
        "",
    ]
    for study in values["by_study"]:
        lines += [
            "## " + study,
            "",
            "| Comparison ID | N | Difference, pp | Interval, pp | Level |",
            "| --- | ---: | ---: | --- | --- |",
        ]
        for x in values["records"]:
            if x["study"] != study:
                continue
            lo, hi = x["preferred_interval"]
            level = f"{x['preferred_confidence'] * 100:.4f}".rstrip("0").rstrip(".")
            lines.append(
                f"| `{x['id']}` | {x['n']} | {100 * x['difference']:+.4f} | [{100 * lo:+.4f}, {100 * hi:+.4f}] | {level}% "
                + ("family" if x["family_interval"] is not None else "descriptive")
                + " |"
            )
        lines.append("")
    write_new(p.BASE / "numerical_appendix.md", "\n".join(lines))
    print("Numerical tables and appendix saved.", flush=True)


if __name__ == "__main__":
    with c.offline():
        build()
