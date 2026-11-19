from collections import defaultdict
import numpy as np
from analysis.classification.combined import quality as q

a = q.a
p = q.p


def close(x, y, msg):
    a.require(abs(x - y) < 1e-10, msg)


def components(x00, x10, x01, x11):
    return dict(
        gate_first=x10 - x00,
        gate_second=x11 - x01,
        head_first=x01 - x00,
        head_second=x11 - x10,
        gate=((x10 - x00) + (x11 - x01)) / 2,
        head=((x01 - x00) + (x11 - x10)) / 2,
        interaction=x11 - x10 - x01 + x00,
        total=x11 - x00,
    )


def run():
    a.g.offline()
    a.verify(a.read(q.BASE / "analysis_contract.json")["sources"])
    out = {}
    verification = 0
    cutoff = []
    cross = []
    for endpoint in (
        "en_test/hate_type",
        "larger_dev/target_group",
        "larger_dev/severity",
    ):
        slug = endpoint.replace("/", "__")
        old = a.read(q.GC / (slug + "_results.json"))
        for v in old["cutoff_components"]:
            stem = f"{v['budget']}/{v['family']}/"
            sc = {
                k: old["absolute"][stem + k]["value"]
                for k in ("default", "gate_only", "head_only", "selected")
            }
            calc = components(
                sc["default"], sc["gate_only"], sc["head_only"], sc["selected"]
            )
            for k, x in calc.items():
                close(x, v[k]["value"], "Original cutoff factorization differs")
                verification += 1
            cutoff.append(
                dict(
                    endpoint=endpoint,
                    **v,
                    scores=sc,
                    source=a.rel(q.GC / (slug + "_results.json")),
                    interpretation="Same fitted weights, fixed saved probabilities, changed TRAIN-selected gate/head cutoffs; average both change orders. Supporting original intervals only.",
                )
            )
        old = a.read(q.GE / (slug + "_cross.json"))
        for v in old["comparisons"]:
            sc = v["scores"]
            calc = components(
                *[sc[k]["value"] for k in ("before", "gate_only", "head_only", "after")]
            )
            for k, x in calc.items():
                close(
                    x,
                    v["contributions"][k]["value"],
                    "Original cross-model factorization differs",
                )
                verification += 1
            cross.append(
                dict(
                    endpoint=endpoint,
                    **v,
                    source=a.rel(q.GE / (slug + "_cross.json")),
                    interpretation="Arithmetic hybrids of saved supervised gates/heads; not measured internal causal mediation or a tested deployable ensemble.",
                )
            )
    q.save(
        "controlled_factorizations.json",
        dict(cutoff=cutoff, cross_model=cross, reconciled_components=verification),
    )
    transitions = []
    for e in q.ENDPOINTS:
        slug = e.replace("/", "__")
        quality = a.read(q.BASE / "endpoints" / (slug + ".json"))
        score = a.read(p.BASE / "endpoints" / slug / "scores.json")
        for mode in ("own_valid", "common_nine"):
            for name, v in quality[mode]["groups"].items():
                before = (
                    name[: -len("selected")] + "default"
                    if name.endswith("selected")
                    else None
                )
                if before not in quality[mode]["groups"]:
                    continue
                prev = quality[mode]["groups"][before]
                mi = quality["models"]
                a.require(
                    [(mi[i]["pool"], mi[i]["seed"]) for i in prev["model_indices"]]
                    == [(mi[i]["pool"], mi[i]["seed"]) for i in v["model_indices"]],
                    "Cutoff pair fit identities differ",
                )
                b = dict(
                    macro_f1=score[mode]["groups"][before]["mean"]["macro_f1"],
                    **{
                        k: prev["mean"][k]
                        for k in (
                            "gate_aware_exact",
                            "exact_set",
                            "gate_pass",
                            "predicted_labels",
                            "extra_labels",
                            "missing_labels",
                        )
                    },
                )
                s = dict(
                    macro_f1=score[mode]["groups"][name]["mean"]["macro_f1"],
                    **{k: v["mean"][k] for k in b if k != "macro_f1"},
                )
                transitions.append(
                    dict(
                        endpoint=e,
                        mask=mode,
                        before=before,
                        after=name,
                        default=b,
                        selected=s,
                        delta={k: s[k] - b[k] if s[k] is not None else None for k in b},
                        qualification="Same retained fit family and seed/pool; cutoffs fixed using TRAIN/validation, never selected on this evaluation.",
                    )
                )
    q.save("cutoff_quality_connections.json", transitions)

    mm = a.read(a.MATCHED / "metrics.json")["results"]
    dd = a.read(a.MATCHED / "diagnostics.json")
    qq = a.read(a.MATCHED / "quality_intervals.json")["results"]
    matched = []
    for endpoint in (
        "en_dev_eval_main/hate",
        "en_dev_eval_sq3_types/hate_type",
        "de_legal_dev_eval/legal",
    ):
        for control, c in mm[endpoint]["contrasts"].items():
            phase = endpoint.split("/")[0]
            d = dd[phase]["paired"][control]
            v = dict(
                endpoint=endpoint,
                control=control,
                confidence=c["family_confidence"],
                family_size=c["family_size"],
                interval=c["family_interval"],
                difference=c["mean_difference"],
                n=c["n_common_valid"],
                reference_F1=c["left_common_mean"],
                control_F1=c["right_common_mean"],
                reference_exact=d["left_mean"]["exact_known_valid"],
                control_exact=d["right_mean"]["exact_known_valid"],
            )
            v["exact_difference"] = v["reference_exact"] - v["control_exact"]
            qualkey = f"{phase}/{control}/known_exact"
            if qualkey in qq:
                v["exact_interval"] = qq[qualkey]["ci95"]
                close(
                    v["exact_difference"],
                    qq[qualkey]["difference"],
                    "Matched exact difference drift",
                )
            else:
                v["exact_interval"] = None
            v["quality_role"] = (
                "Exploratory nominal 95%, known-field agreement on original masks; heterogeneous known fields on main hate cohort, not all-field accuracy."
            )
            matched.append(v)
    q.save("matched_mechanism_connections.json", matched)
    normalization = defaultdict(lambda: defaultdict(int))
    for r in a.read(q.BQ / "normalization_diagnostic.json")["rows"]:
        for k, v in r.items():
            if k not in ("condition", "population", "repeat"):
                normalization[r["condition"]][k] += v
    rationales = a.read(q.ROB / "derived_findings.json")["rationales"]
    for co, v in rationales.items():
        close(
            v["any_field_differs"] / v["valid_predictions"],
            v["fraction_any_differs"],
            "Rationale fraction drift",
        )
        a.require(
            v["valid_predictions"] == normalization[co]["valid"],
            "Rationale validity population drift",
        )
    q.save(
        "aggregation_and_rationales.json",
        dict(
            normalization=dict(normalization),
            rationales=rationales,
            interpretation="Final parsed/voted fields only. Copied rationale comes from one gate-agreeing inner response. A differing final field is not an adjudicated explanation error; absence of mismatch is not verified faithfulness.",
            raw_limit="Parser normalization can remove fine assertions. No latent Qwen ungated head, pre-parser accuracy or oracle gate rescue is inferred.",
        ),
    )
    cases = a.read(q.CTX / "case_contexts.json")
    prior = (
        a.read(q.CTX / "manual_review.json")["notes_by_case"]
        if "notes_by_case" in a.read(q.CTX / "manual_review.json")
        else None
    )
    notes = {
        "C01": (
            "Helpful boundary calibration",
            "Five non-hate sexual-content examples accompany the full-system correct benign decision, whereas small supplies five hate-labelled examples. This is a contextual association, not an isolated negative-example intervention.",
        ),
        "C11": (
            "Adverse boundary transfer",
            "Four similar sexual examples labelled non-hate accompany a missed gold-hate input. The mechanism suggested by C01 therefore has an observable cost at a nearby boundary; source annotations are retained.",
        ),
        "C08": (
            "Target boundary confusion",
            "The full context includes kill-ISIS examples, including explicit anti-Islam extensions. Full treats the query as a religious attack, while small stays benign. Topical proximity can transfer an inappropriate target interpretation.",
        ),
        "Q01": (
            "Useful annotation specificity",
            "Full shows five propaganda-only examples and returns the exact observed legal set. Small examples often combine propaganda/incitement and its answers add incitement. Hate gold is unknown; this is corpus legal-set agreement, not a legal determination.",
        ),
        "Q04": (
            "Useful close example with conflicting neighbours",
            "Full repeatedly returns grievance alone for a close template; small adds stereotyping. Other near-template full examples carry incitement or a negative gate, so close wording is not uniformly reliable and is not exact-duplicate leakage.",
        ),
        "S19": (
            "Correct gate with unsupported background",
            "The query assigns blame to a daughter-in-law rather than a son without mentioning infertility. Full context mentions infertility and its rationale imports it. Gold gate agreement does not validate this detail, fine fields, or internal faithfulness.",
        ),
        "S20": (
            "Stance-changing continuation",
            "A close opening in a non-hate retrieved example continues by defending girls worth; the query instead criticizes their moral/behavioural upbringing. Full misses the gold-hate input despite four other hate-labelled examples. Neither majority label counts nor topical similarity fully explains the result.",
        ),
        "S22": (
            "Conflicting bank label boundary",
            "Five examples are labelled non-hate, including Women should never be allowed to rule. This hateful wording has a negative dataset label; full misses the hateful query. Preserve the contradiction without relabelling gold or describing the example as semantically positive.",
        ),
    }
    selected = []
    for cid, (theme, note) in notes.items():
        x = next(c for c in cases if cid in c["case_ids"])
        selected.append(
            dict(
                case_id=cid,
                key=x["key"],
                theme=theme,
                interpretation=note,
                gold=x["gold"],
                context_sha256={
                    co: x["prompts"][co]["context_sha256"]
                    for co in ("rag_full", "rag_small")
                },
                example_ids={
                    co: [
                        h["id"]
                        for _, hh in x["prompts"][co]["context"]["example_groups"]
                        for h in hh
                    ]
                    for co in ("rag_full", "rag_small")
                },
                saved_answers=[
                    dict(
                        condition=r["condition"],
                        repeat=r["repeat"],
                        result=r["result"],
                        result_path=r["result_path"],
                    )
                    for r in x["predictions"]
                ],
            )
        )
    q.save(
        "case_integration.json",
        dict(
            selection="Illustrative integration of eight already selected/reviewed prior cases, retaining favourable and adverse mechanisms. Not a random sample or prevalence estimate. The complete prior 96-input casebook remains authoritative.",
            source=a.rel(q.CTX / "case_contexts.json"),
            cases=selected,
        ),
    )

    ladder = [
        (
            "retrieval_package",
            "Controlled system contrast",
            "Full versus no retrieval changes definitions, guidelines and examples together; fixed generator endpoint/decoding.",
            "Useful package effect on measured cohorts; no isolated example-only, retrieval-algorithm or model-internal mechanism.",
        ),
        (
            "bank_composition",
            "Controlled bank/system contrast",
            "Full/small deliver identical rule channels and five examples; examples, order, source and annotation composition change.",
            "Not pure bank size, token count or semantic relevance. Query-conditioned context associations are not mediation.",
        ),
        (
            "matched_coverage",
            "Controlled prompt intervention",
            "Replaces example identities while matching source/language/gate and annotation availability/cardinality.",
            "Exact label distribution, wording and length still differ. Main type benefit is specific to exposed DEV.",
        ),
        (
            "annotation_permutation",
            "Controlled prompt intervention",
            "Permutes complete annotation blocks among the same example texts.",
            "Supports contribution of correct association for type F1; includes unchanged rendered permutations; exact type gain not established.",
        ),
        (
            "cutoffs",
            "Deterministic saved-output intervention",
            "Same weights/probabilities; gate and fine cutoffs switched separately, in both orders.",
            "Identifies decision-policy effects on saved outputs, not learning improvements or future calibrated reliability.",
        ),
        (
            "cross_heads",
            "Deterministic saved-output intervention",
            "Swap existing supervised gates/heads and average both change orders.",
            "Arithmetic attribution only; conditional on source models/pools, not a newly trained hybrid or internal causal mediation.",
        ),
        (
            "training_coverage",
            "Verified design plus association",
            "Total-text pools give unequal fine annotations; selected settings fixed on TRAIN.",
            "Few-label degeneracy is observed; no isolated coverage or LR mechanism, no claim that three learning rates caused all gains.",
        ),
        (
            "context_cases",
            "Illustrative observation",
            "Actual saved text, annotations, output fields and rationale.",
            "Helpful and distracting context coexist; examples cannot establish prevalence, causality or faithful internal reasoning.",
        ),
        (
            "rationale_assembly",
            "Observed implementation and trace",
            "Final voted fields may differ from fields in the one response supplying the rationale.",
            "A consistency risk, not a measured explanation-falsehood rate; correct labels/repeat agreement do not validate explanations.",
        ),
    ]
    q.save(
        "mechanism_evidence_map.json",
        [
            dict(id=i, evidence_level=l, what_is_observed=w, claim_limit=c)
            for i, l, w, c in ladder
        ],
    )
    q.save(
        "mechanisms_complete.json",
        dict(
            created_utc=a.utc(),
            cutoff_components_verified=verification,
            cutoff_connections=len(transitions),
            matched_contrasts=len(matched),
            illustrative_cases=len(selected),
            sources=a.bind([__file__, q.BASE / "analysis_contract.json"]),
            new_experiments=0,
        ),
    )
    print(
        "Mechanism integration complete:",
        verification,
        "factorization components,",
        len(transitions),
        "cutoff views,",
        len(matched),
        "matched contrasts.",
        flush=True,
    )


if __name__ == "__main__":
    run()
