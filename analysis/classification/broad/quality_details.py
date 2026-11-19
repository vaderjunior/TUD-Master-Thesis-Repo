from collections import Counter
from pathlib import Path
import numpy as np
from analysis.classification.broad import quality as q

NOTES = {
    "Q01": (
        "Full retains the annotated propaganda label without the additional incitement label from small. Its legal set stays correct while the hate decision changes across repeats.",
        "Legal gold is the only observed field here. Neither the changing hate decisions nor the explanations about German law are validated by this annotation. No legal advice is inferred.",
        "legal set precision and independent branch",
    ),
    "Q02": (
        "Full consistently substitutes propaganda for the annotated incitement/public-peace label, whereas small matches both annotated labels. Both identify insult/defamation.",
        "The first full rationale discusses racist abuse but does not explain its added propaganda label. A later saved rationale speculates about the ambiguous word Kitchen; neither that speculation nor any legal interpretation is adjudicated here.",
        "legal substitution and explanation coverage",
    ),
    "Q03": (
        "The legal gold is an observed empty set. Both retrieval systems nevertheless assign legal labels, with full adding incitement/public-peace to insult/defamation.",
        "The text is hostile, but this does not independently establish a legal violation. These are false assignments against the frozen legal annotations; the annotation/model boundary is not resolved by accepting the model rationale.",
        "known-empty legal false assignments",
    ),
    "Q04": (
        "Full matches grievance alone. Small adds stereotyping, and no retrieval adds still more types. The scored improvement is fewer additional type labels.",
        "The first full rationale frames the alleged in-group victims as the target, while small describes the out-group as the target. Only type and hate are annotated; the correct type set does not validate the target explanation or the extra legal/severity claims.",
        "type specificity without explanation validation",
    ),
    "Q05": (
        "Small retains the annotated hate/grievance interpretation in every repeat. Full and no retrieval take a questioning or distancing reading and suppress all fine fields.",
        "The fragment permits a disagreement about stance. Preserve its original labels and lack of surrounding context; this example does not establish that quotation marks always mean endorsement or always excuse a statement.",
        "gate-mediated type loss and stance ambiguity",
    ),
    "Q06": (
        "Both full and small retain the annotated irony label but also assign stereotyping in every repeat, so neither exact type set matches.",
        "The joke visibly invokes a stereotype, making this a useful annotation-granularity caution: an additional label is an error against the supplied set, not automatically a nonsensical description. Other predicted fields have no gold here.",
        "overlapping types and annotation granularity",
    ),
    "Q07": (
        "Full restores the hateful decision and high severity, while small classifies the insults as non-hate and withholds severity. Full still misses the annotated sexual-orientation target in two repeats.",
        "A severity fix alone is not a complete annotated-answer fix. The model explanations illustrate the protected-group versus personal-abuse boundary, without proving the original context or the target identity.",
        "gate restoration with incomplete targets",
    ),
    "Q08": (
        "Both retrieval systems identify the sexual-orientation target. Full chooses medium severity in every repeat; small and no retrieval choose the annotated high value.",
        "The full rationale explicitly uses absence of direct physical violence to justify medium. The frozen gold is mapped perceived intensity, so a violence-based rationale is not a substitute for the recorded severity construct.",
        "severity construct mismatch",
    ),
    "Q09": (
        "All systems give a negative hate decision, preventing the annotated high severity and both target labels from being delivered.",
        "The rationale distinguishes sexual harassment of an individual from protected-group hate. Retain this definition/context boundary rather than silently relabeling the row or calling the gate-blocked severity low.",
        "shared gate loss and annotation boundary",
    ),
    "Q10": (
        "Full supplies the annotated gender-only target. Small consistently adds disability, apparently reading a derogatory intelligence term as a target attribute.",
        "The gold does not annotate disability. This illustrates a possible lexical overextension, not evidence about the actual person. Types and legal labels are unannotated, even when their explanations sound specific.",
        "extra target from insult wording",
    ),
    "Q11": (
        "Small supplies the annotated hate decision and gender target. Full suppresses both in every repeat despite recognizing gendered mockery in its explanation.",
        "The first full rationale says gender is insufficient and treats humour as a reason against hate; this conflicts with the available protected-gender taxonomy more directly than a simple missed word. No retrieval context has yet been inspected to explain why.",
        "gate rationale conflicts with schema",
    ),
    "Q12": (
        "Full consistently returns the sexual-orientation target but omits the annotated race target. Small introduces national origin and sometimes also loses sexual orientation.",
        "The visible text mentions nationality and points to someone outside the text; the race gold is not directly explained by the available sentence. Keep the original annotation and missing contextual limits. Full being more restrained is not exact annotated correctness.",
        "target omissions and incomplete context",
    ),
    "Q13": (
        "Full matches race plus low severity with a positive hate decision. Small raises severity to medium and often adds gender; no retrieval raises severity to high.",
        "This is a joint fix only for the three observed fields. Author identity, reclaimed use and wider context are unknown, and the unannotated type/legal outputs remain unvalidated.",
        "joint specificity and scope of correctness",
    ),
    "Q14": (
        "Small matches the annotated gender/low answer; full reads the text as harassment rather than group hate and suppresses the fields.",
        "The small rationale adds a non-consensual or exploitative interpretation that is not established by the sentence alone. Its scored joint success does not validate that explanation.",
        "correct labels with unsupported rationale detail",
    ),
    "Q15": (
        "All systems retain a positive gate but miss at least one annotated target and rate severity above the annotated low value. Full keeps only national origin; small also supplies religion.",
        "The religious collective-blame wording is visible, yet target mapping and the intensity tercile still require the frozen annotation definitions. Do not infer that a high-sounding rationale overrides low gold.",
        "combined target omission and severity disagreement",
    ),
    "Q16": (
        "All three systems return non-hate and an empty target set. The target-only score counts that empty set as correct, but the annotated hate decision is positive, so the gate-aware answer is wrong.",
        "The criticism targets a corporation rather than an evident protected group. This illustrates an annotation/taxonomy boundary and the danger of calling an empty-set score complete answer quality; the gold is preserved.",
        "empty-set inflation and definition mismatch",
    ),
}


def run():
    receipt = q.c.read(q.BASE / "analysis_complete.json")
    q.r.verify(receipt["outputs"])
    q.r.verify(receipt["sources"])
    review = q.c.read(q.BASE / "independent_review.json")
    q.r.verify(review["sources"])
    cases = q.c.read(q.BASE / "cases.json")
    q.require(set(NOTES) == {x["case_id"] for x in cases}, "Manual notes incomplete")
    notes = []
    for case in cases:
        reading, qualification, theme = NOTES[case["case_id"]]
        notes.append(
            dict(
                case_id=case["case_id"],
                key=case["key"],
                reading=reading,
                qualification=qualification,
                theme=theme,
                gold_changed=False,
                all_nine_decisions_read=True,
                displayed_rationales="First outer repeat of full and small; no best-rationale selection.",
                source_case_sha256=q.c.signature(case),
            )
        )
    q.save(
        q.BASE / "manual_case_review.json",
        dict(
            status="all_16_selected_cases_and_144_delivered_decisions_reviewed",
            sources=q.bindings(
                [Path(__file__), q.BASE / "cases.json", q.BASE / "case_selection.json"]
            ),
            cases=notes,
            not_prevalence_sample=True,
            no_retrieval_context_or_internal_reasoning_claim=True,
        ),
    )
    summaries = {}
    n_checked = 0
    for path in sorted((q.BASE / "endpoints").glob("*/results.json")):
        out = q.c.read(path)
        dim = out["endpoint"].split("/")[1]
        labels = out["labels"]
        settings = [tuple(v) for v in out["settings"]]
        with np.load(q.ROOT / out["encoded_path"], allow_pickle=False) as z:
            y = z["gold"]
            p = z["predicted"]
            g = z["gate"]
            mask = z["common_nine"]
            card = y[mask].sum(1)
        bycond = {}
        for cond in q.CONDITIONS:
            ix = [settings.index((cond, j)) for j in range(3)]
            pp = p[ix][:, mask]
            gg = g[ix][:, mask]
            yy = y[mask]
            exact = (pp == yy).all(2)
            gate_exact = exact if dim == "legal" else exact & gg
            strata = []
            for k in range(len(labels) + 1):
                m = card == k
                n = int(m.sum())
                strata.append(
                    dict(
                        gold_cardinality=k,
                        n=n,
                        exact_set_by_repeat=(
                            exact[:, m].mean(1).tolist() if n else [None] * 3
                        ),
                        gate_exact_by_repeat=(
                            gate_exact[:, m].mean(1).tolist() if n else [None] * 3
                        ),
                        gate_exact_mean=float(gate_exact[:, m].mean()) if n else None,
                    )
                )
            d = dict(
                cardinality_strata=strata,
                conditional_gate_exact_mean=(
                    float(
                        np.mean(
                            [
                                v["exact_given_gate"]
                                for v in out["common_nine"][cond]["repeats"]
                            ]
                        )
                    )
                    if dim != "legal"
                    else None
                ),
            )
            details = out["common_nine"][cond]["detail"]
            if dim == "severity":
                d.update(
                    severity_confusion_mean=np.mean(
                        [v["confusion"] for v in details], axis=0
                    ).tolist(),
                    severity_counts_mean={
                        k: float(np.mean([v["counts"][k] for v in details]))
                        for k in details[0]["counts"]
                    },
                    conditional_ordinal_mae_mean=float(
                        np.mean([v["conditional_ordinal_mae"] for v in details])
                    ),
                )
            if dim == "legal":
                legal_rows = []
                for j, v in enumerate(details):
                    n0 = v["known_empty"]
                    n1 = v["annotated_positive"]
                    tn = v["any_legal_tn"]
                    exact1 = (
                        sum(
                            v["by_predicted_hate"][b]["exact"]
                            for b in ("False", "True")
                        )
                        - tn
                    )
                    legal_rows.append(
                        dict(
                            known_empty=n0,
                            known_positive=n1,
                            empty_gold_flagged=v["empty_gold_flagged"],
                            empty_gold_flagged_rate=v["empty_gold_flagged_rate"],
                            positive_gold_unflagged=v["positive_gold_unflagged"],
                            positive_gold_unflagged_rate=v[
                                "positive_gold_unflagged_rate"
                            ],
                            exact_on_empty=tn / n0,
                            exact_on_positive=exact1 / n1,
                            exact_positive_count=exact1,
                        )
                    )
                    m = yy.any(1)
                    q.require(
                        exact1 == int(exact[j, m].sum())
                        and tn == int(exact[j, ~m].sum()),
                        "Legal exact subgroups differ",
                    )
                    n_checked += 2
                d["legal_repeats"] = legal_rows
                d["legal_mean"] = {
                    k: float(np.mean([v[k] for v in legal_rows])) for k in legal_rows[0]
                }
            bycond[cond] = d
        contributions = []
        for right in ("rag_small", "zero_shot"):
            for j, label in enumerate(labels):
                delta = (
                    out["common_nine"]["rag_full"]["per_label_mean"][j]["f1"]
                    - out["common_nine"][right]["per_label_mean"][j]["f1"]
                )
                inc = label in out["macro_labels"]
                contributions.append(
                    dict(
                        reference=right,
                        label=label,
                        label_f1_difference=delta,
                        in_original_macro=inc,
                        contribution_to_fixed_macro=(
                            delta / len(out["macro_labels"]) if inc else None
                        ),
                    )
                )
            if out["macro_labels"]:
                total = sum(
                    v["contribution_to_fixed_macro"]
                    for v in contributions
                    if v["reference"] == right and v["in_original_macro"]
                )
                a = np.mean(
                    [v["macro_f1"] for v in out["common_nine"]["rag_full"]["repeats"]]
                )
                b = np.mean(
                    [v["macro_f1"] for v in out["common_nine"][right]["repeats"]]
                )
                q.require(
                    abs(total - (a - b)) < 1e-14,
                    "Label decomposition does not sum to fixed macro change",
                )
                n_checked += 1
        summaries[out["endpoint"]] = dict(
            common_nine_n=out["common_nine_n"],
            conditions=bycond,
            label_f1_decomposition=contributions,
        )
    normal = q.c.read(q.BASE / "normalization_diagnostic.json")
    counters = {}
    seen = 0
    for row in q.reader.rows(q.reader.BASE / "predictions.jsonl"):
        key = (row["phase"], row["condition"], row["repeat"])
        ct = counters.setdefault(key, Counter())
        seen += 1
        ct["predictions"] += 1
        ct["parser_normalisations"] += row["normalisations"]
        ct["parser_gate_normalised"] += row["gate_normalised"]
        valid = row["result"] is not None
        ct["valid" if valid else "invalid"] += 1
        if not valid:
            continue
        votes = [v for v in row["votes"] if v is not None]
        result = row["result"]
        majority = {
            field: [
                lab
                for lab in sorted({lab for vote in votes for lab in vote[field]})
                if sum(vote[field].count(lab) for vote in votes) * 2 > len(votes)
            ]
            for field in ("hate_type", "target_group", "legal")
        }
        severity = sorted(
            (v["severity"] for v in votes if v["severity"] is not None),
            key=["low", "medium", "high"].index,
        )
        majority["severity"] = severity[(len(severity) - 1) // 2] if severity else None
        q.require(
            result["hate"] == (sum(v["hate"] for v in votes) * 2 > len(votes)),
            "Independent final gate differs",
        )
        if result["hate"]:
            for field in ("hate_type", "target_group", "severity"):
                ct["positive_empty_" + field] += int(not majority[field])
                q.require(majority[field] == result[field], "Open gate field differs")
        else:
            ct["final_negative"] += 1
            ct["negative_with_positive_inner_vote"] += int(
                any(v["hate"] for v in votes)
            )
            for field in ("hate_type", "target_group", "severity"):
                ct["final_gate_clears_" + field] += int(bool(majority[field]))
                q.require(not result[field], "Closed gate field not empty")
        q.require(
            majority["legal"] == result["legal"], "Independent legal vote differs"
        )
    for row in normal["rows"]:
        expected = counters[row["population"], row["condition"], row["repeat"]]
        actual = {
            k: v
            for k, v in row.items()
            if k not in ("population", "condition", "repeat")
        }
        q.require(
            all(expected[k] == v for k, v in actual.items())
            and all(actual.get(k, 0) == v for k, v in expected.items()),
            "Normalization inventory differs",
        )
        n_checked += len(actual)
    normtot = {}
    for cond in q.CONDITIONS:
        ct = Counter()
        for (phase, condition, rep), v in counters.items():
            if condition == cond:
                ct.update(v)
        normtot[cond] = dict(ct)
    q.require(
        seen == 194643
        and sum(v.get("parser_gate_normalised", 0) for v in counters.values()) == 303,
        "Normalization total mismatch",
    )
    q.save(
        q.BASE / "additional_summaries.json",
        dict(
            status="descriptive_decompositions_checked",
            sources=q.bindings(
                [
                    Path(__file__),
                    q.BASE / "analysis_complete.json",
                    q.BASE / "independent_review.json",
                ]
            ),
            endpoints=summaries,
            normalization_by_condition=normtot,
            normalization_rows_checked=seen,
            additional_numeric_checks=n_checked,
            caveat="Cardinality strata and label contributions are descriptive on the all-nine mask; no new hypothesis, tuning, legal advice, causal reasoning or source-weighted estimate.",
        ),
    )
    print(
        "All selected cases reviewed; all 194,643 vote aggregates and descriptive decompositions checked.",
        flush=True,
    )


if __name__ == "__main__":
    with q.r.original.offline():
        run()
