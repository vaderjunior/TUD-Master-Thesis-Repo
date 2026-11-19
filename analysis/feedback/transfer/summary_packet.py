from analysis.feedback.transfer import summary as p
from pathlib import Path
import os

a = p.a
OUT = p.OUT
E = "../../experiments/astra/sq3_transfer_part10_20261023_v1"
NAMES = {
    "en_test_FPR": "English TEST false alarms",
    "en_test_FNR": "English TEST missed hate",
    "en_hatecheck_eval_FPR": "HateCheck false alarms",
    "en_hatecheck_eval_FNR": "HateCheck missed hate",
}


def table(head, rows):
    cell = lambda v: str(v).replace("|", "/").replace("\n", " ")
    return "\n".join(
        [
            "| " + " | ".join(head) + " |",
            "|" + "|".join(["---"] * len(head)) + "|",
            *["| " + " | ".join(cell(v) for v in row) + " |" for row in rows],
        ]
    )


def pct(x):
    return f"{100*x:.2f}%"


def pp(x):
    return f"{100*x:+.3f}"


def ci(x):
    return "[" + ", ".join(pp(v) for v in x) + "]"


def make_tables():
    d = a.read(OUT / "display_values.json")
    tabs = {}
    tabs["primary"] = table(
        [
            "Endpoint",
            "Common texts / clusters",
            "Reference",
            "Feedback",
            "Change pp",
            "Primary 98.75% interval pp",
        ],
        [
            [
                NAMES[q["endpoint"]],
                f"{q['common_n']:,} / {q['clusters']:,}",
                pct(q["reference"]),
                pct(q["feedback"]),
                pp(q["difference"]),
                ci(q["interval98_75"]),
            ]
            for q in d["primary"]
        ],
    )
    tabs["classifier"] = table(
        [
            "Endpoint",
            "Classifier before → after",
            "Qwen before → after",
            "Classifier change pp",
            "Qwen change pp",
            "Difference in change pp",
            "Separate primary 98.75% interval",
        ],
        [
            [
                NAMES[q["endpoint"]],
                pct(q["estimates"]["linear_baseline"])
                + " → "
                + pct(q["estimates"]["linear_updated"]),
                pct(q["estimates"]["qwen_reference"])
                + " → "
                + pct(q["estimates"]["qwen_updated"]),
                pp(q["estimates"]["linear_change"]),
                pp(q["estimates"]["qwen_change"]),
                pp(q["estimates"]["difference_in_change"]),
                ci(q["primary_interval98_75"]),
            ]
            for q in d["classifier"]
        ],
    )
    names = {
        ("binary", "macro_f1"): "Binary macro F1",
        ("binary", "balanced_accuracy"): "Balanced accuracy",
        ("known_answer", "known_answer_exact"): "Known-field exact agreement",
        ("hate_type", "macro_f1"): "Type macro F1",
        ("hate_type", "exact_set"): "Exact type set",
        ("target_group", "macro_f1"): "Target macro F1",
        ("target_group", "exact_set"): "Exact target set",
        ("severity", "macro_f1"): "Severity macro F1",
        ("severity", "exact_set"): "Exact severity",
        ("severity", "severity_provided"): "Severity category provided",
    }
    tabs["quality"] = table(
        [
            "Population",
            "Metric",
            "Common texts",
            "Reference",
            "Feedback",
            "Change pp",
            "Supporting 95% interval pp",
        ],
        [
            [
                q["population"],
                names[(q["head"], q["metric"])],
                f"{q['common_n']:,}",
                pct(q["reference"]),
                pct(q["feedback"]),
                pp(q["difference"]),
                ci(q["interval95"]),
            ]
            for q in d["quality"]
        ],
    )
    tabs["study_map"] = table(
        ["Study", "Question / scope", "What it establishes", "Boundary"],
        [
            [
                "Ordinary insertion",
                "40 corrections compete within the full KB; exposed DEV",
                "Rare delivery: 13/2,025 main contexts",
                "Availability of corrections is not their delivery",
            ],
            [
                "Earlier reserved-correction matrix",
                "675 exposed DEV texts, 74 conditions, delivery/label/context/policy controls",
                "Guaranteed delivery exposes an error trade-off; controls bound mechanism claims",
                "Its 97.5% primary and 99.1667% control families are separate",
            ],
            [
                "Human-derived supplement",
                "Three source corrections and 21 other selected texts",
                "Source uptake; no main complete-answer transfer gain",
                "Two novices, disagreement and identical-message variation remain",
            ],
            [
                "This broader transfer",
                "12,030 eligible English TEST + 3,728 HateCheck; four fixed conditions",
                "English trade-off; no demonstrated overall complete-quality gain",
                "Previously exposed datasets; no new label/context control arms",
            ],
            [
                "Principal same-label classifier comparison",
                "Same three forty-label pools and transfer masks; eight saved fits",
                "Small refit changes, opposing absolute errors, four planned change contrasts",
                "Binary head; cumulative updates still pending",
            ],
            [
                "Successive simulated rounds",
                "Four rounds, three streams and selection policies",
                "Prepared; no reviewed outcome yet",
                "Separate accepted experiment, not completed by this synthesis",
            ],
        ],
    )
    return tabs


def make_claims():
    specs = [
        (
            "T01",
            "The prospectively fixed policy compares five ordinary examples with four ordinary plus one retrieved binary correction; Qwen weights stay fixed.",
            "Same example count does not equalize tokens, fine information or update mechanism.",
            "experiments/astra/sq3_transfer_20261012_v1/STUDY_PROTOCOL.md",
            "Methodology",
        ),
        (
            "T02",
            "English false alarms rise +3.182 pp and missed hate falls −1.358 pp in the original four-endpoint 98.75% family.",
            "Use different benign/hateful denominators; do not subtract the two rates as overall gain.",
            "experiments/astra/sq3_transfer_part2_20261023_v1/rates.json",
            "Results_SQ3",
        ),
        (
            "T03",
            "HateCheck false alarms rise +1.394 pp in the primary analysis; missed-hate benefit is unresolved.",
            "Representative-cluster FPR intervals include zero; representative FNR upper endpoint equals zero.",
            "experiments/astra/sq3_transfer_part2_20261023_v1/rates.json",
            "Results_SQ3 / limitations",
        ),
        (
            "T04",
            "All three natural datasets retain the English trade-off; referring counter-speech is a conspicuous HateCheck problem.",
            "Source/function intervals are supporting 95%, not new adjusted discoveries or source interactions.",
            "experiments/astra/sq3_transfer_part3_20261023_v1/rates.json",
            "Supporting results",
        ),
        (
            "T05",
            "English binary macro F1 and joint known-field exact agreement decline; exact type sets also decline.",
            "Known-field exactness checks only annotated eligible fields; no text supplies every fine field and legal has no gold.",
            "experiments/astra/sq3_transfer_part4_20261023_v1/quality_results.json",
            "Results_SQ3",
        ),
        (
            "T06",
            "Target and severity macro-F1 changes are uncertain; more severity coverage coexists with stronger high-category bias.",
            "Do not manufacture fine labels, replace unknown with empty, or call an uncertain result equivalent.",
            "experiments/astra/sq3_transfer_part4_20261023_v1/set_and_severity_diagnostics.json",
            "Supporting results / discussion",
        ),
        (
            "T07",
            "Hateful corrections are delivered in 63.33% of English and 65.93% of HateCheck assignments despite balanced pools.",
            "Label swaps and fine-annotation loss describe associations; this run does not isolate their causal effects.",
            "experiments/astra/sq3_transfer_part5_20261023_v1/exposure.json",
            "Discussion",
        ),
        (
            "T08",
            "The fifth ordinary example is displaced and sometimes carries fine annotations.",
            "Fifth means prompt position, not necessarily the least similar or least useful example.",
            "experiments/astra/sq3_transfer_part5_20261023_v1/displacement_details.json",
            "Methodology / discussion",
        ),
        (
            "T09",
            "English mean correctness improves on 929 texts and worsens on 1,470; 845/1,156 matched missed-hate fixes retain a fine error.",
            "Pairs reuse reference outputs; cases are mechanically selected descriptions, not independent trials or causal explanations.",
            "experiments/astra/sq3_transfer_part6_20261023_v1/transition_summary.json",
            "Results_SQ3 / appendix",
        ),
        (
            "T10",
            "English common-case directions survive declared sensitivities, but adverse invalid-vote completion can erase the missed-hate benefit.",
            "Identification bounds are not confidence intervals; observed strict filtering changes the cohort.",
            "experiments/astra/sq3_transfer_part7_20261023_v1/vote_completion_bounds.json",
            "Limitations",
        ),
        (
            "T11",
            "All 276 null aggregates result from terminal schema-invalid hate-type votes; outputs were preserved.",
            "Nulls are not 276 unanswered network requests; no invalid booleans were salvaged.",
            "experiments/astra/sq3_transfer_part7_20261023_v1/failure_detail_summary.json",
            "Reliability / appendix",
        ),
        (
            "T12",
            "The English trade-off appears in both observed worker eras.",
            "Different inputs and service periods do not prove concurrency equivalence or an unchanged backend.",
            "experiments/astra/sq3_transfer_part7_20261023_v1/era_rates.json",
            "Limitations",
        ),
        (
            "T13",
            "The principal classifier comparison distinguishes relative adaptation from absolute error rates.",
            "Difference in change is classifier delta minus Qwen delta; a favourable contrast is not proof of useful classifier learning.",
            "experiments/astra/sq3_transfer_part8_20261023_v1/rates.json",
            "Results_SQ3",
        ),
        (
            "T14",
            "Classifier refits change little and are cheap on cached vectors, but retain substantial missed hate.",
            "Binary head only, 55,753 prior TRAIN rows; prior embeddings and model selection are inherited, not free.",
            "experiments/astra/sq3_transfer_part9_20261023_v1/classifier_resources.json",
            "Discussion / resources",
        ),
        (
            "T15",
            "Scientific recorded usage is 552,493,301 tokens; mean English serving tokens are about 3,009 versus 3,012.",
            "Observed tokens exclude unknown remote work; cost similarity is not equivalence or a deployment winner.",
            "experiments/astra/sq3_transfer_part9_20261023_v1/resource_summary.json",
            "Resources",
        ),
        (
            "T16",
            "The broader study and principal static comparison are complete; simulated rounds and cumulative refits remain open.",
            "Do not mark all SQ3 complete, launch a queue, or retrospectively change the future policy.",
            "Thesis_Log/astra_result/astra_remaining_work_before_writing_20261023_v85.md",
            "Conclusion / pending status",
        ),
    ]
    return [
        dict(
            id=i,
            claim=c,
            qualification=q,
            source=s,
            source_sha256=a.sha(p.ROOT / s) if (p.ROOT / s).exists() else None,
            writer_destination=w,
        )
        for i, c, q, s, w in specs
    ]


def report(t):
    return f"""# Broader correction transfer — final scientific synthesis

23 October 2026. **All ten review parts of this study are complete.** The fixed correction policy changes the error balance on the broader English evaluation: fewer hateful texts are missed, but more benign texts are flagged. It does not demonstrate an overall improvement in complete-answer quality. A small classifier refitted with the same retained labels changes little and makes a different absolute error trade-off. Similar observed Qwen token cost and inexpensive classifier updates do not establish an overall practical winner.

This closes the broader-transfer study and its principal static same-feedback comparison. It does **not close all SQ3**: successive simulated correction rounds, their matching cumulative classifier updates and final combined SQ3 synthesis remain. No new Qwen calls, model checks, training, embeddings, retrieval, annotations or queues were used for this synthesis. Writer-owned LaTeX/PPT and acknowledgements remain untouched.

## The question and the fixed comparison

The question is whether a correction to an earlier mistake helps classify other texts. For each eligible input, the reference supplies five ordinary examples from the full knowledge base. Feedback retains the same first four and replaces the fifth with one retrieved correction from a fixed forty-record pool. Each pool contains twenty benign and twenty hateful earlier errors; all three pools were fixed before the previous reserved-correction study's outcomes. Query text and cached representations select the correction; query gold does not route it. The correction provides only its binary label. Qwen's parameters, task, taxonomy, definitions and guidelines remain fixed.

The study contains 12,030 English TEST texts from HateXplain, Implicit Hate and Measuring Hate Speech, and 3,728 HateCheck diagnostic texts, analysed separately. Four conditions × three decoder repeats produce 189,096 voted outputs; each requests three ordinary votes. The reference has three real outputs per text, while feedback averages the three repeats within each of three overlapping pools equally. Neither reused reference comparisons nor repeated generations create independent texts. The pools have 120 memberships but only 98 distinct source records.

The shared source/overlap audit excludes specified source, DEV and TRAIN-exposed components. The conservative cosine boundary is 0.9495, with text/prefix and transitive connections retained. This is a bounded separation audit, not proof of no conceivable semantic overlap. The original first-pass retrieval contexts stay authoritative; the inherited common twenty-input replay exclusion is a sensitivity, not a claim that exact cold replay succeeded. These TEST datasets were previously used elsewhere in the thesis and are **not a new untouched holdout**. HateCheck measures functional coverage rather than deployment prevalence.

All conditions were generated fresh in this campaign. The experiment extends evaluation breadth; it is not a repeat of the previous 74-condition control matrix or the two-person feedback experiment:

{t['study_map']}

The studies are connected conceptually, without pooling their populations, uncertainty families or execution periods into a formal cross-study effect estimate. Familiar-category restoration remains positive evidence about a different form of external adaptation; a mixed correction result does not erase it.

## Primary error effects

The main mask requires every condition and repeat to have a usable aggregate. It retains 11,884 English and 3,724 HateCheck texts. False alarms use benign texts; missed hate uses hateful texts. Effects are feedback minus reference in percentage points. The frozen four-endpoint family uses 10,000 paired whole-inference-cluster resamples, PCG64 seeds 2026101216–2026101219 and **98.75% intervals**. All dependent outputs remain together. Inference conditions on these fixed pools and saved repeats.

{t['primary']}

The English trade-off is established under that primary analysis. Every natural source and each fixed pool has the same directions. Source-specific 95% intervals are supporting evidence, not new family-adjusted discoveries or tests that one source benefits more. The higher HateXplain false-alarm rate is partly contextualized by its original offensive-versus-hateful distinction, but the increase also occurs among unanimously normal-labelled texts. The fixed label mapping remains unchanged.

HateCheck false alarms rise in the primary analysis, with a representative-cluster qualification below. Missed hate starts low and its tiny mean change is unresolved. Counter-speech referring to hate is the clearest supporting location of the false-alarm increase: 18.10% → 23.25%, +5.16 pp [nominal 95%: +1.22, +9.06], across 140 common texts in seventeen connected groups. It contributes 44.5% of the net HateCheck FPR increase arithmetically. That is not a causal percentage or a new adjusted subgroup discovery. Every hateful function's change interval includes or touches zero; small improvements and new errors offset.

The full [primary analysis](astra_sq3_transfer_part2_primary_20261023_v1.md) and [all sources/functions](astra_sq3_transfer_part3_sources_20261023_v1.md) preserve every pool, source, function and template. The [current primary figure](../../experiments/astra/sq3_transfer_part2_20261023_v1/figures_v2/primary_error_rates.png) is suitable for the main thesis; all twenty-five current scientific figures are indexed in the [figure handover]({E}/figure_inventory.md).

## The complete recorded answer does not improve overall

Each real run is scored before averaging, so F1 is not computed by pooling dependent repeated predictions. Fine heads retain predeclared population support and are evaluated only where gold is known. Unknown is not an empty set. Nine hateful natural texts have known-empty target sets and remain scored. Type gold is confined to eligible Implicit Hate cases, natural targets to HateXplain/MHS, severity to MHS, and HateCheck fine gold to targets. Neither population supplies legal gold.

**Known-field exact agreement** requires the correct gate and all eligible fields actually annotated for that text. No text has every fine field annotated, so this is not complete-schema accuracy. The differing annotation profiles must accompany any quoted combined number.

{t['quality']}

The main English binary macro F1, balanced accuracy, known-field agreement and exact type-set changes have supporting intervals below zero. Exact type sets fall from 11.34% to 10.02%, even though containment of all recorded type labels is much higher. More extra labels accompany the decline: returning a recorded label plus additional labels is not an exact answer. Extra labels indicate disagreement with recorded gold, not automatically semantic invalidity.

Target and severity macro-F1 changes remain uncertain. Severity categories are returned more often, while high-category bias and conditional rank error increase. Coverage alone cannot establish improved severity judgment. HateCheck binary and known-field scores decline in the main analysis, but their representative-cluster intervals include zero; target findings remain uncertain and can change point direction under representative weighting. The [complete quality report](astra_sq3_transfer_part4_quality_20261023_v1.md) retains all label supports, cardinalities, gate partitions, omission checks and missing-output bounds.

## Delivery and individual changes explain the limitations without proving a mechanism

Every planned correction was delivered and the first four ordinary examples were preserved. Balanced pools did not produce balanced delivered labels: hateful corrections appear in 63.33% of English and 65.93% of HateCheck assignments, whereas the removed fifth examples are hateful in 32.16% and 37.61%. Opposite-label swaps accompany opposite error movements. This helps locate the observed trade-off, but retrieved text, label, similarity and annotation content change together; it does not isolate a label-only causal effect.

The fifth example is a prompt position, not necessarily the least similar example. Replacing it removes at least one known fine annotation in 30.85% of English and 36.67% of HateCheck prompts. The new correction supplies no replacement fine label. The associations between removed annotations and exact-answer changes are descriptive: this broader study has no masked-label or retained-fifth arms. Earlier controls cannot manufacture those missing arms on this population. All 47,274 correction assignments and exact messages remain in [Part 5](astra_sq3_transfer_part5_context_20261023_v1.md).

On the common English cohort, mean binary correctness improves for 929 texts, worsens for 1,470 and is equal for 9,485. There are useful local fixes, persistent failures and new errors. Of 1,156 matched missed-hate-fix comparisons, 845 still have at least one known fine-field error. These are dependent paired occurrences, not 1,156 independent texts. Only 27 texts satisfy the stringent observed pattern of reference wrong in all three outputs and all nine feedback outputs correct; 35 satisfy the reverse. Less uniform benefits are not denied by that stringent definition.

Identical prompts can yield different saved answers. Binary outputs are more stable than five-field answers, and stable can still mean wrong. The mechanically selected [43-case book](../../experiments/astra/sq3_transfer_part6_20261023_v1/case_index.md) and [eleven classifier-comparison cases](../../experiments/astra/sq3_transfer_part8_20261023_v1/case_index.md) retain actual contexts, annotations and repeats, including empty selection categories. Cases illustrate observable transitions; they do not reveal an internal reasoning mechanism or justify post-hoc label changes.

## Robustness and failure boundaries

The English error directions survive all six declared common-case mask views, source-stratified resampling, fixed-pool and batch omissions. Removing the common twenty replay-related inputs changes little. However, selecting only usable outputs is distinct from making guarantees about missing votes. The 276 null output slots affect 150 texts; 5,188 outputs have fewer than three valid votes. Strict means all votes are valid, not unanimous.

Null-only full-roster bounds retain the English trade-off. If invalid vote slots receive adverse binary completions while surviving valid votes stay fixed, English FNR change can lie between −2.955 and +0.445 pp. Broader strict-unknown bounds are wider. These are identification bounds, not confidence intervals. Thus the common-case missed-hate benefit is not a worst-case full-roster guarantee. The FPR increase remains under all three missingness assumptions.

All 276 null aggregates result from three terminal-invalid votes with hate-type schema failures after both allowed repairs. In 818 of their 828 votes, `insult_defamation`, a legal category, was emitted as a hate type. These are responses that arrived and failed the declared schema, not 276 unanswered network requests. Every invalid vote remains invalid; no raw boolean is retrospectively salvaged.

HateCheck representative-cluster FPR intervals include zero, and representative FNR intervals end at exactly zero. Sparse and connected templates limit stronger claims. The classifier's main HateCheck FNR difference-in-change is also exactly zero; these zero boundaries must not be rounded into established effects.

Immutable resume inventories assign 31,565 outputs to eight workers and 157,531 to 128, with one input spanning the change. Both homogeneous periods retain the English within-period trade-off. They contain different texts and service conditions, so they do not establish causal concurrency effects, equivalence or an unchanged backend. All reported responses use the same model alias, which is not a backend-version certificate. [Part 7](astra_sq3_transfer_part7_robustness_20261023_v1.md) retains these boundaries and the original failed reader attempts and amendments.

## The principal same-feedback classifier comparison

The comparator is a binary logistic head on frozen BGE-M3 representations, with the original C=10, balanced weights and strict probability >0.6 threshold. It begins with 55,753 permitted TRAIN texts and refits after appending forty source records. Qwen sees one retrieved correction per query from the same pool. Equal retained labels do not mean equal per-query exposure, prior pretraining, fine information or compute. The matched-gradient TRAIN variant and >0.5 threshold remain supporting sensitivities, never retrospectively chosen winners.

Eight of the 38 already verified static fits supply this comparison. No fit is repeated. The four original planned contrasts are **(classifier after − before) − (Qwen after − before)**, with a separate 98.75% family and seeds 2026101224–2026101227. Negative means a more favourable classifier change, not necessarily fewer final errors or useful learning.

{t['classifier']}

English FPR change favours the classifier mainly because Qwen adds false alarms. English FNR change favours Qwen. The classifier's own principal change intervals all include zero. HateCheck FPR relative change favours the classifier even though Qwen has the lower final FPR point; the difference in FNR change is exactly zero while their final FNRs are very different, 60.70% for the classifier and 1.18% for Qwen.

Absolute English errors also oppose each other: the updated classifier has 18.09% FPR and 37.36% FNR, versus Qwen's 38.93% and 14.53%. No overall winner follows. The principal broad comparison provides evidence beyond the earlier supporting 675-text DEV analysis, but different populations and uncertainty roles prevent replacing its historical values or treating both as untouched replications. [Part 8](astra_sq3_transfer_part8_classifier_20261023_v1.md) keeps all settings, sources/functions, bounds, probability movements and cases. Cumulative-round fitting and review remain pending.

## Resources and practical meaning

The scientific run used 567,288 ordinary calls, 36,784 repairs and 4,195 extra transport attempts: 608,267 POST attempts, plus 24 separate availability GETs. Recorded input/output usage is **552,493,301 tokens**. Repairs consume 8.03%; all null-output costs remain included. English full-roster tokens per voted prediction are 3,008.71 for reference and 3,011.86 for feedback, difference +3.15 [supporting 95%: −5.48, +11.81]. Lower ordinary usage is offset by repairs. Similar observed totals do not establish equivalence or better value.

Generation ends after 53.85 calendar hours; the historically captured final completion-file timestamp gives 60.93 hours, including pauses/checks and an uncertain post-audit window. Summed POST duration is 2,236.13 overlapping request-hours, not GPU-hours. Uninterrupted 128-worker batches achieve about 9,747 voted outputs/hour during generation and 7,635 through audits, versus 1,162 and 1,130 in the two uninterrupted eight-worker batches. This observed throughput is useful operational evidence, not a guaranteed future multiplier.

The 38 inherited classifier fits total 34.15 solver minutes; the three principal updates take about 55–60 seconds each on cached vectors. The eight consumed fits and three updates are nested subsets, not additional fits. Shared embeddings, retrieval preparation and earlier selection are inherited. Source screening used 1,024 existing labels and 8,264,261 tokens already counted in the additions programme. The separate capacity probes' 12,073,826 observed tokens are excluded from this scientific row. Unknown remote work, monetary cost, energy and unmeasured effort remain unknown.

The new [partial resource ledger v7](astra_resource_evidence_20261023_v7.md) adds this study exactly once to the unchanged eleven prior programmes: twelve remote programmes, 2,028,441 successful text replies and 1,860,042,318 reported tokens. It is not a whole-thesis cost or a bill. The complete [resource review](astra_sq3_transfer_part9_resources_20261023_v1.md) includes failed and repaired work, inherited classifier costs and hypothetical prevalence/error-cost scenarios. Those scenarios illustrate assumptions; they neither estimate real harm values nor choose a deployment policy.

## What the final SQ3 argument can say

External knowledge can change behaviour without retraining the generator. Source uptake, actual delivery, transfer and complete-answer quality are different steps. Earlier ordinary insertion often failed at delivery. Guaranteed delivery then exposed an error trade-off. This prospectively fixed broader evaluation shows the same qualitative English trade-off across sources and no overall complete-quality gain. The small classifier provides a concrete alternative use of the same labels with limited adaptation and a different error profile. The evidence supports a conditional adaptation claim, not a generally self-improving or deployed expert-feedback system.

The result does not imply that all feedback fails, that every model-added label is wrong, or that all classifier updates are ineffective. It describes this fixed retrieval/update policy, these overlapping single-source binary pools, previously exposed evaluation data and the saved generations. No pool, threshold, correction weight, source blacklist or future policy is selected from the results. Positive category-restoration evidence and the qualified two-person findings remain part of SQ3.

## Verification and handover

The final check revalidates 1,380 immutable source/artifact bindings and the current nine-part proof chain. Fresh direct arithmetic from the original 15,758-row assembled decisions reproduces all four primary points, four principal classifier contrasts and fifteen main quality quantities, including run-wise F1 and known-field exactness. Sixteen headline interval endpoints are separately recomputed from sorted saved draws. No unresolved numerical discrepancy was found.

The prior independent proofs already regenerated **335 distinct bootstrap sets / 77,220,000 saved scalar values** across Parts 2–9. This synthesis reuses those bound proofs rather than claiming a new full bootstrap or raw-response replay. The twenty-five current figures, original tables, complete case ledgers and reader/layout amendments remain preserved and indexed. Part 8's external figure-receipt correction remains authoritative for its two self-hash defects; no historical receipt is rewritten. The [verification report](astra_sq3_transfer_part10_verification_20261023_v1.md) states the fresh and reused scope explicitly.

The [writing handover](astra_sq3_transfer_writing_handoff_20261023_v1.md) supplies exact tables, current figure paths, sixteen claim-to-evidence entries, replacement locations and the word-budget boundary. Journal v16 and roadmap v85 record the completed study. Actual thesis/PPT integration and acknowledgement belong to the writing session. Its current draft still describes this study as running; publication of the handover is not evidence that the draft has changed.

**Four of the nine experimental items are now complete; five remain open, and all sixteen inventory entries are retained.** The next SQ3 work is readiness for the separate successive simulated-round study and matching cumulative classifier updates, followed by final combined SQ3 synthesis. No next queue starts automatically. Saved equal-budget SQ2 work and university setup remain preserved; the 9 November experiment-stop target is unchanged. [Current roadmap](astra_remaining_work_before_writing_20261023_v85.md).
"""


def handoff():
    return f"""# Writing handover — completed broader correction transfer and principal classifier comparison

23 October 2026. **Ready for staged integration.** All ten review parts of this broader-transfer study are complete, including final verification. This handover adds reviewed outcomes; it does not close all SQ3. The separate writer owns LaTeX, presentation edits, builds, visual checks and `writing_evidence/astra_sync_state.json`. No acknowledgement was created here.

## Read these first

1. [Final synthesis](astra_sq3_transfer_final_results_20261023_v1.md), then the [verification scope](astra_sq3_transfer_part10_verification_20261023_v1.md).
2. [Exact display tables]({E}/display_tables.md) and [machine-readable values]({E}/display_values.json). Read the [claim map]({E}/claim_map.md) alongside prose, not just the rounded figures.
3. [Figure inventory]({E}/figure_inventory.md): all twenty-five current PNG/SVG pairs, their hashes and intended roles. Keep old layouts and earlier studies intact.
4. [Resource ledger v7](astra_resource_evidence_20261023_v7.md) and [roadmap v85](astra_remaining_work_before_writing_20261023_v85.md). Earlier reports' pending paragraphs are dated history, not current status.

New register ID: **`{p.UID}`**. The experiment-side [update register](../astra_writing_updates.json) lists all authoritative source hashes. Read every new/changed register item before writing: the shorter-SQ2 result and later operational readiness updates also arrived since the inspected draft. Acknowledge only actual integration, using the existing [sync protocol](../astra_writing_sync.md).

## What the inspected writing checkpoint already contains

The read-only snapshot of `Thesis_Final/writing_status.md` is dated 20 October. It reports integration of the reserved-correction study, human-derived supplement and supporting same-label DEV comparison. `Results_SQ3.tex` still says broader transfer is running, and its same-label subsection centres on the 675-text DEV comparison. Those descriptions now need updating, while their historical findings remain valid in their own scope. The snapshot hashes are recorded in the synthesis contract; do not assume an independently changing writer tree stays at this version.

The writer reports a conservative planning count of 14,317 words against a 14,500 editorial ceiling and the supplied 15,000 maximum: only 183 planning words remained at that checkpoint. Therefore **replace and consolidate existing passages rather than append the entire research report**. Recount the actual current draft after editing. Detailed subgroup, case, resource and robustness tables belong in the existing supporting PDF or appendix. Do not reduce type size or alter the template to accommodate them.

## Concrete integration locations

| Existing writer file/location | Intended change |
|---|---|
| `sections/Thesis_Chapters/Methodology.tex` and `Experiments.tex` | Add the fixed broader design, source separation, all-arm common mask, four-endpoint 98.75% family and separate classifier contrast family. Keep previously exposed TEST and HateCheck dependence explicit. |
| `Results_SQ3.tex`, after `subsec:reserved-results` | Summarise broader transfer with the four primary rates; distinguish it from the earlier 675-text/74-condition controls. State the complete-quality result and annotation coverage limit. |
| `Results_SQ3.tex`, `subsec:same-label-results` | Make the planned broad comparison the main evidence. Retain the existing DEV analysis as supporting history in the supplement/appendix, with its original numbers and supporting 95% role. |
| `Results_SQ3.tex`, `subsec:sq3-diagnostics` | Replace the running/pending broad-study statement. Retain the separate pending successive-round and cumulative-classifier work. |
| `Discussion.tex` and `Conclusion.tex` | Explain delivery versus useful transfer, opposing false-alarm/miss costs, known-field quality, cheap but weak fixed refits, missing-vote limits and no overall policy winner. |
| `Thesis_Appendices.tex` and `supporting_results.tex` | Carry source/function tables, full quality/support definitions, cases, all declared masks and failure/era qualifications. Link exhaustive artifacts rather than copy every diagnostic into main text. |
| Existing resource table | Extend the eleven-programme ledger with the single reviewed transfer row. Use v7's twelve-programme totals; keep local fits, inherited selection and capacity diagnostics separate. |
| `claim_sources.md`, writer source archives and binding manifests | Extend the existing map with the new source hashes and qualifications. Do not discard old claim IDs or silently relabel historical evidence. |
| Presentation and speaker notes | Consolidate the feedback story around delivery, broader trade-off and complete quality. Prefer replacing an existing feedback/classifier display; keep detailed diagnostics in backup slides. |
| `writing_status.md` and writer acknowledgement | Record what was actually integrated, still deferred, built and visually checked; do not infer completion from this handover. |

The inspected presentation status describes DRAFT_v5 with twelve main slides / fourteen minutes and sixteen hidden backups. Those are the writer's recorded constraints, not a fresh presentation render by this session. Native PowerPoint opening and rehearsal remain the writer/user's separate tasks.

## Compact main-text narrative to adapt

> The prospectively fixed broader evaluation compared five ordinary retrieved examples with four ordinary examples and one binary correction from each of three balanced forty-record pools. On 12,030 eligible English TEST texts and 3,728 HateCheck cases, all conditions were generated afresh with three decoder repeats and three votes per output. The common-valid English analysis showed more false alarms (+3.182 percentage points; 98.75% interval +2.634 to +3.728) and fewer missed hateful texts (−1.358; −2.000 to −0.718). HateCheck false alarms increased in the primary analysis, while its missed-hate change remained unresolved. These are separate benign and hateful denominators, not an overall utility score.
>
> The broader result did not establish an overall complete-answer improvement. English binary macro F1 fell from 69.09% to 67.53%, known-field exact agreement from 52.48% to 50.09%, and exact type sets from 11.34% to 10.02%. Known-field agreement checks only annotated eligible fields; no text has every fine field annotated. Target and severity macro-F1 changes remained uncertain. Corrections were delivered, but their label balance and displaced fine information varied with the selected context, so these associations do not isolate a mechanism.
>
> Refitting the frozen-feature classifier with the same forty labels produced small changes and opposing absolute errors: fewer English false alarms but substantially more missed hate than Qwen. The separately planned difference-in-change family distinguished this adaptation comparison from final accuracy. Similar Qwen token use and inexpensive cached-feature refits did not identify an overall practical winner. Previously exposed evaluation data, overlapping simulated feedback pools, missing-vote worst cases and nonrandom execution periods limit stronger generalization. Successive simulated rounds and cumulative refits remain separate pending work.

This is replacement material, not a requested word-count increase. Use the exact tables if rounding or phrasing changes. The narrative's source is this study; do not attach the previous 97.5% family or the DEV comparator's supporting 95% intervals to these new primary results.

## Display selection and non-negotiable qualifications

For the main thesis, use the four-row primary table or Part 2's current primary figure. For the principal classifier comparison, use Part 8's current absolute-error figure together with the separate change-contrast table or a concise description. Absolute and relative adaptation evidence answer different questions. Keep full quality and robustness displays in the supplement if the word budget requires it. The primary and classifier tables each use their own four-contrast **98.75% family**; source, quality, context and resource intervals remain supporting 95%. Their uncertainty is conditional on saved pools and repeats.

Do not describe this as new expert feedback, a newly untouched holdout, German transfer, a complete-schema/legal accuracy evaluation, a causal concurrency experiment, or proof that all corrections fail. Never drop nulls asymmetrically, convert unknown labels into negatives, turn a zero endpoint into significance, or treat a favourable relative classifier change as a final-performance win. The known-field metric is not an all-field gold standard. The fifth displaced example is not necessarily the least relevant. All detailed qualifications are attached to the sixteen entries in the claim map.

The headline classifier HateCheck FNR change contrast is **exactly zero**, despite very different final FNRs. The representative HateCheck Qwen FNR interval ends at **exactly zero**. The earlier reserved study's strict/excluded FNR zero and tiny-positive type-F1 boundary remain separate historical facts; preserve them rather than replacing them with the broader values.

## Scope after integration

This study's whole-experiment box closes; the classifier's principal static comparison is complete. The larger classifier-update box stays open for cumulative-round fits/review. **Four of nine experimental items are complete and five remain open.** SQ3 still needs the accepted successive simulated-round experiment, its matching cumulative classifier updates and final combined synthesis. The two other unstarted Qwen studies are SQ2. The human review and its feedback run are already complete; no new forms are needed.

Use the current roadmap for SQ2 checkpoint/university setup and deadlines. The experiment stop remains 9 November, with generation preferably finished by 6 November. This packet neither adopts a new scope reduction nor starts any queue. Actual writing integration, source-bound acknowledgement, build/visual verification and word-budget checks remain to be performed by the writer.
"""


def verify_report():
    return f"""# Broader-transfer Part 10 — final verification scope

23 October 2026. **PASS: no unresolved numerical discrepancy in the current final synthesis.** This is an offline final reconciliation, not a new model run, reannotation, complete raw-response replay or new hypothesis family.

The [contract]({E}/analysis_contract.json) binds 1,380 immutable sources and artifacts from Parts 1–9, their current completion/proof chain, the old resource ledger and frozen protocols. Current hashes reconcile. Historical files that say a later part was pending remain unchanged; their later completion receipts establish the current state. In particular, Part 4's diagnostic completion, Part 6's case completion, Part 7's failure verification and Part 8's explicit figure-receipt correction must accompany their earlier-stage numerical receipts.

Fresh direct arithmetic reads all 15,758 original matrix rows and reconstructs all 189,096 Qwen decisions/validity slots, the common masks, binary error counts and declared gold eligibility. It checks strict classifier thresholding from saved probabilities and identical Qwen arrays, then reconstructs four primary rates/effects and four principal classifier contrasts with Fraction arithmetic. It separately reconstructs fifteen main quality quantities from per-run confusion/label counts: binary macro F1 and balanced accuracy, known-field exactness, available type/target/severity macro F1 and exact sets, plus severity coverage. Full-roster fixed label support and absent gold are checked.

All sixteen interval endpoints for the eight headline contrasts are recomputed by sorting the saved draws and manually interpolating the declared 98.75% percentiles. Exact-zero representative HateCheck FNR endpoints and the exact-zero classifier FNR change contrast remain zero. This pass does not generate new bootstrap draws. The prior independent proofs already regenerated **335 distinct sets, 77,220,000 scalar values**, excluding reused source sets from the sum. Their data/source bindings are retained; this figure describes reused proof coverage, not 77 million new computations in Part 10.

Every current scientific figure is indexed with PNG/SVG hashes, including corrected layout versions and preserved earlier files. Prior table/case publication checks remain bound. Final packet checks verify the displayed numbers, claim/source links, current figure selection, twelve-programme ledger sums, journal embedded-image bytes, writer-source register and preserved navigation. Reader/layout corrections remain reporting amendments, not scientific changes. Existing full automatic raw audits are reused.

The resource extension preserves every one of the previous eleven programme rows and adds the reviewed transfer row once. Screening, nested classifier fits, human effort and capacity probes retain their stated boundaries. The new total is partial remote-programme accounting, not total thesis compute or cost.

The [machine-readable verification]({E}/verification.json), [exact values]({E}/display_values.json), [final completion receipt]({E}/complete.json) and [publication check]({E}/publication/final_check.json) give the precise binding scope. Final status: broader transfer and principal static comparison complete; successive rounds, cumulative classifier work and whole-SQ3 closure remain pending. No writer-owned file or acknowledgement is changed by this verification.
"""


def resource_report():
    old = p.OLDRESOURCE.read_text(encoding="utf8")
    ledger = a.read(OUT / "partial_programme_ledger.json")
    fields = (
        "ordinary_slots",
        "repair_slots",
        "retry_or_recovered_failure_attempts",
        "attempt_slots",
        "successful_responses",
        "reported_tokens",
    )
    lines = []
    for q in ledger["programmes"]:
        src = q["source"]
        target = os.path.relpath(p.ROOT / src, p.RESULT).replace("\\", "/")
        lines.append([f"[{q['programme']}]({target})", *[f"{q[k]:,}" for k in fields]])
    lines.append(
        ["**Twelve-programme total**", *[f"{ledger['totals'][k]:,}" for k in fields]]
    )
    retained = old.split("## New human-feedback row", 1)[1].split(
        "The accepted future queue remains separate from incurred costs.", 1
    )[0]
    return f"""# Resource evidence and practical costs — partial ledger v7

23 October 2026. Adds the scientifically reviewed [broader-transfer study](astra_sq3_transfer_final_results_20261023_v1.md) exactly once to the unchanged eleven rows in [v6](astra_resource_evidence_20261020_v6.md). The old ledger/report remain preserved. No new model call, fit or embedding is incurred by this update.

**Scope: twelve nonoverlapping remote programmes, not the entire thesis cost.** Ordinary logical calls, repair calls, extra transport attempts and successful text replies are distinct. Unknown work behind failures remains unknown. The earlier nine programmes total 880,940,304 tokens; reserved delivery, human feedback and this transfer are separate later rows.

{table(['Programme','Ordinary slots','Repair calls','Extra attempts','Accounted POST slots','Successful replies','Reported tokens'],lines)}

## New broader-transfer row

15,758 inputs × four conditions × three repeats = 189,096 voted outputs. The row contains 567,288 ordinary calls, 36,784 repair calls and 4,195 extra attempts, totalling 608,267 POST attempts and 604,072 successful text replies. Twenty-four availability GETs are separate. Recorded 473,767,201 prompt + 78,726,100 completion tokens = **552,493,301 tokens**, including 44,353,529 repair tokens (8.03%). Every null and under-three-vote output retains its expense.

There are 4,174 ReadTimeouts, seven ConnectTimeouts and fourteen HTTP 502 responses; possible remote work behind them is unmeasured. The 276 null aggregates instead arise from terminal schema-invalid hate-type votes after responses arrived. Repairs recover schema validity for 15,046 of 21,849 initially invalid votes; correctness is a separate question. Terminal-invalid vote paths use 16,652,313 repair tokens already inside the total.

Elapsed time is 53.85 hours through the last generation summary, 54.01 through the last batch audit, and 60.93 through the historically captured final-file timestamp. The last is a filesystem proxy, not a measured GPU clock. Two explicit gaps total 5.11 hours. Summed POST durations are 2,236.13 overlapping request-hours and recorded retry waiting another 2.33 overlapping hours. No bill, energy or GPU-occupancy estimate is manufactured.

On English TEST, full-roster tokens per voted prediction are 3,008.71 for reference and 3,011.86 for feedback. Difference +3.15 has supporting 95% interval [−5.48, +11.81]. Similar observed cost does not establish equivalence or a useful overall upgrade. Read [Part 9](astra_sq3_transfer_part9_resources_20261023_v1.md) for all 136 groups, input costs, repair depth, failed attempts and chronology.

## Classifier, screening and capacity boundaries

All 38 existing static classifier fits total 2,048.98 solver seconds (34.15 minutes), with 2,292.83 seconds of monitored worker time. The eight fits consumed in the principal/supporting broad comparison and three principal updates are nested subsets, not new fits to add. Three shared cached vector files total 373,707,136 bytes; this excludes the BGE model and total infrastructure storage. Original encoding, indexing, curation and prior selection effort are not comprehensively timed. Local fitting is not added to a remote-call ledger.

The original source screening's 8,264,261 tokens already belong to SQ3 additions, and its 1,024 labels are existing dataset labels. The 120 current correction memberships contain 98 distinct sources; overlap does not create new annotation effort. The separate capacity-test checkpoint records 13,495 POST attempts, 48 GETs and 12,073,826 observed tokens, with 25 timeouts. These are excluded diagnostics, not an extra scientific programme row. The friends' 96 judgments / 100 participant-minutes remain inherited once.

## Preserved human-feedback row and earlier qualifications

The following material preserves the earlier incurred-cost boundaries from v6. Its references to a current run describe the corresponding historical row, not a second charge.

{retained}

## Current completion and accounting boundary

Four experimental items are now scientifically complete and five remain open. Broader transfer and its principal static classifier comparison are reviewed; successive simulated rounds and cumulative classifier updates remain. Shorter SQ2 training is reviewed, equal-budget SQ2 is paused with university setup pending, and the two separate SQ2 Qwen studies remain. Future planned calls are not incurred cost, and none starts automatically.

The [machine-readable twelve-programme ledger]({E}/partial_programme_ledger.json) preserves the previous eleven rows byte-for-value and the [Part 9 staged addition](../../experiments/astra/sq3_transfer_part9_20261023_v1/resource_addition.json). The [final verification]({E}/verification.json) checks sums and exact-once attribution. Local training/embedding studies still need whole-project consolidation; this remains a partial ledger. [Current roadmap](astra_remaining_work_before_writing_20261023_v85.md).
"""


def main():
    a.need(
        (OUT / "verification.json").exists() and not p.REPORT.exists(),
        "Verified new packet only",
    )
    tables = make_tables()
    p.save(
        OUT / "display_tables.md",
        "# Final synthesis — exact display tables\n\nUnits and uncertainty roles stay attached to each table. All values come from the fresh direct checks and bound prior estimates.\n\n"
        + "\n\n".join("## " + name + "\n\n" + value for name, value in tables.items())
        + "\n",
    )
    p.save(OUT / "display_tables.json", tables)
    claims = make_claims()
    p.save(OUT / "claim_map.json", claims)
    p.save(
        OUT / "claim_map.md",
        "# Final synthesis — claim map\n\n"
        + table(
            [
                "ID",
                "Permitted claim",
                "Required qualification",
                "Source",
                "Writer destination",
            ],
            [
                [
                    q["id"],
                    q["claim"],
                    q["qualification"],
                    "[evidence](../../../../" + q["source"] + ")",
                    q["writer_destination"],
                ]
                for q in claims
            ],
        )
        + "\n",
    )
    figures = a.read(OUT / "analysis_contract.json")["current_figures"]
    p.save(OUT / "figure_inventory.json", figures)
    p.save(
        OUT / "figure_inventory.md",
        "# Current scientific figures\n\nAll twenty-five were previously visually reviewed and remain unchanged. Use current layouts below. The complete part reports define masks, units, intervals and qualifications; do not detach them when reproducing a plot. Main-text candidates: Part 2 primary error rates and Part 8 absolute error rates. Quality/context/resource and all other diagnostics can go in the supplement.\n\n"
        + table(
            ["Part", "Figure", "PNG", "Editable SVG"],
            [
                [
                    q["part"],
                    q["name"],
                    "[PNG](../../../../" + q["png"] + ")",
                    "[SVG](../../../../" + q["svg"] + ")",
                ]
                for q in figures
            ],
        )
        + "\n",
    )
    p.save(p.REPORT, report(tables))
    p.save(p.HANDOFF, handoff())
    p.save(p.VERIFY_REPORT, verify_report())
    p.save(p.RESOURCE, resource_report())
    p.save(
        OUT / "packet_created.json",
        dict(
            status="synthesis_packet_created_final_publication_checks_pending",
            created_utc=a.utc(),
            script_sha256=a.sha(Path(__file__)),
            artifacts={
                a.rel(path): a.sha(path)
                for path in (
                    p.REPORT,
                    p.HANDOFF,
                    p.VERIFY_REPORT,
                    p.RESOURCE,
                    OUT / "display_tables.md",
                    OUT / "display_tables.json",
                    OUT / "claim_map.json",
                    OUT / "claim_map.md",
                    OUT / "figure_inventory.json",
                    OUT / "figure_inventory.md",
                )
            },
            new_Qwen_calls=0,
            new_fits=0,
        ),
    )
    print(
        "Scientific synthesis, exact displays, claim/figure maps, writer handover and resource v7 prepared. Publication checks still pending."
    )


if __name__ == "__main__":
    with a.offline():
        main()
