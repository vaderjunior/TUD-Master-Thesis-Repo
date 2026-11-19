import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
import socket
from unittest.mock import patch
from analysis.common import feedback_inputs as c
from analysis.common import prompt as renderer
from analysis.feedback.summary.validate_closing_feasibility import blocks

COMMON_ANALYSIS = {
    "decoder_repeats": 3,
    "votes_per_prediction": 3,
    "bootstrap_draws": 10000,
    "bootstrap_unit": "original text overlap cluster; keep all repeats, pools and category pairs together",
    "zero_denominator_f1": 0,
    "failed_outputs": "retain every null; no replacement generation",
    "sensitivities": [
        "all-three-valid-votes mask",
        "one representative per overlap cluster",
        "null-as-error lower bound and opposite upper bound",
    ],
    "scope": "Previously exposed DEV. New prospective supplement after original results; not untouched confirmation.",
    "stopping": "complete frozen roster; infrastructure/integrity stops only; never inspect scores to choose continuation",
    "old_primary_family": "closed and unchanged; no new outcome is retroactively added to it",
    "cross_study_interpretation": "Study-specific intervals; no single confirmatory omnibus claim across these additions.",
}


def prep_start(root, paths, protocol, analysis, conditions):
    c.require(not (root / "prepared.json").exists(), "Already prepared")
    frozen = c.preserve_originals()
    c.write(root / "STUDY_PROTOCOL.md", protocol)
    inputs = c.inputs_for_study(
        [
            *paths,
            __file__,
            root / "STUDY_PROTOCOL.md",
            c.ROOT / "analysis/feedback/summary/validate_closing_feasibility.py",
        ]
    )
    c.write(
        root / "preparation_plan.json",
        {
            "created_utc": c.utc(),
            "inputs": inputs,
            "original_bindings": frozen,
            "model_calls": 0,
        },
    )
    science = {
        **COMMON_ANALYSIS,
        **analysis,
        "conditions": conditions,
        "preparation_plan_sha256": c.sha(root / "preparation_plan.json"),
    }
    c.write(root / "scientific_plan_v1.json", science)
    return science


def finish(root, prompts, roster, summary):
    c.jsonlines(root / "roster.jsonl", roster)
    c.jsonlines(root / "prompts.jsonl", prompts)
    c.require(
        len({(p["key"], p["condition"]) for p in prompts}) == len(prompts),
        "Duplicate prompt identity",
    )
    counts = [p["reference_input_tokens"] for p in prompts]
    summary.update(
        prompt_identities=len(prompts),
        predictions=len(prompts) * 3,
        ordinary_calls=len(prompts) * 9,
        input_tokens_min=min(counts),
        input_tokens_max=max(counts),
        new_provider_calls=0,
        cuda_jobs=0,
    )
    c.write(root / "summary.json", summary)
    c.verify(c.read(root / "preparation_plan.json")["inputs"])
    c.preserve_originals()
    c.write(
        root / "prepared.json",
        {
            "status": "prepared_requires_independent_check",
            "outputs": c.bindings((p for p in root.iterdir() if p.is_file())),
            "created_utc": c.utc(),
        },
    )
    print(json.dumps({"study": root.name, **summary}, ensure_ascii=False), flush=True)


def factorial():
    root = c.BASE / "factorial"
    protocol = "# Focused examples × definitions supplement\n\nThe roster is every input/category pair in the completed visible-block audit for which both example and definition blocks differ: 227 pairs, 183 texts, all seven labels. No outcome filters. All four cells receive fresh outputs; historical responses are never spliced into the primary. Examples (E) and definitions (D) independently take their saved withdrawn (0) or full-KB (1) block. Guidelines, input, full seven-label schema and slot budgets stay fixed. E0D0 and E1D1 must reproduce the original schema-only and full messages byte for byte. Crossed cells use the same renderer.\n\nThere are 193 hateful pairs with 70 target positives, and 34 benign pairs. Two co-primary effects are the marginal example-block effect ((F10-F00)+(F11-F01))/2 and marginal definition-block effect ((F01-F00)+(F11-F10))/2. Compute each category's target F1 on its hateful subset, average repeats within category and then equally average seven categories. Report 97.5% paired cluster-bootstrap intervals (Bonferroni two-contrast family, 10,000 draws). These are effects within the both-changed subset, not general population or single-token effects. Co-labels, other examples and length may change with the example block.\n\nUse a common per-pair mask requiring all four cells and three repeats to be non-null. Cluster sampling retains every eligible category pair from the same original overlap cluster; pairs and decoder repeats are not independent sample units. Fix seven categories and zero-denominator F1=0. If a category has no retained positive support, do not quietly remove it; mark the planned primary unestimable and show descriptive results. Bootstrap replicates without positive support use the declared zero-denominator convention. Report per-category supports, precision/recall, TP/FP/FN, joint restoration, interaction, other-label regressions and benign hate false positives as supporting diagnostics, with unadjusted 95% intervals clearly identified. No category selection by effect. Explicitly report all nulls and shared full-prompt identities across category pairs; fresh responses still carry clustered dependence.\n"
    conditions = ["E0D0", "E1D0", "E0D1", "E1D1"]
    prep_start(
        root,
        [
            c.REVIEW / "rendered_block_audit.json",
            c.OLD_TAX / "prompts.jsonl",
            c.OLD_TAX / "roster.jsonl",
            c.OLD_TAX / "complete.json",
        ],
        protocol,
        {
            "primary": [
                "marginal_example_block_effect",
                "marginal_definition_block_effect",
            ],
            "primary_interval": 0.975,
            "bootstrap_seed": 202610041,
            "mask": "per pair, all 12 condition/repeat outputs valid",
        },
        conditions,
    )
    inventory = c.read(c.REVIEW / "rendered_block_audit.json")[
        "both_changed_pair_inventory"
    ]
    old = {
        (p["item_id"], p["condition"]): p for p in c.rows(c.OLD_TAX / "prompts.jsonl")
    }
    old_roster = {r["id"]: r for r in c.rows(c.OLD_TAX / "roster.jsonl")}
    c.require(
        len(inventory) == 227 and len({r["id"] for r in inventory}) == 183,
        "Frozen pair inventory differs",
    )
    tok = c.tokenizer()
    prompts = []
    roster = []
    endpoint_matches = 0
    for pair in inventory:
        iid, label = (pair["id"], pair["label"])
        r = old_roster[iid]
        low, full = (old[iid, "rag_schema_" + label], old[iid, "rag_full"])
        lo, hi = (blocks(low), blocks(full))
        c.require(
            lo["examples"] != hi["examples"]
            and lo["definitions"] != hi["definitions"]
            and (lo["guidelines"] == hi["guidelines"]),
            "Not a both-changed pair",
        )
        key = r["key"] + "::" + label
        roster.append(
            {
                **r,
                "key": key,
                "original_key": r["key"],
                "target_label": label,
                "target_positive": label in (r["gold"]["hate_type"] or []),
            }
        )
        for condition in conditions:
            e, d = (int(condition[1]), int(condition[3]))
            context = deepcopy(low["context"])
            context["example_groups"] = deepcopy(
                (full if e else low)["context"]["example_groups"]
            )
            context["definitions"] = deepcopy(
                (full if d else low)["context"]["definitions"]
            )
            messages = renderer.render_messages(r["text"], "en", context, "corrected")
            c.require(
                messages[0] == full["messages"][0] == low["messages"][0],
                "System prompt drift",
            )
            if condition in ("E0D0", "E1D1"):
                c.require(
                    messages == (full if e else low)["messages"],
                    "Endpoint is not original payload",
                )
                endpoint_matches += 1
            prompts.append(
                {
                    "key": key,
                    "item_id": iid,
                    "condition": condition,
                    "target_label": label,
                    "population": r["population"],
                    "lang": "en",
                    "text": r["text"],
                    "context": context,
                    "messages": messages,
                    "context_sha256": c.sig(context),
                    "messages_sha256": c.sig(messages),
                    "reference_input_tokens": c.token_count(tok, messages),
                    "scientific_plan_sha256": c.sha(root / "scientific_plan_v1.json"),
                }
            )
    c.require(
        len(prompts) == 908 and endpoint_matches == 454, "Factorial count differs"
    )
    finish(
        root,
        prompts,
        roster,
        {
            "original_texts": 183,
            "pairs": 227,
            "endpoint_message_matches": endpoint_matches,
            "by_label": {
                l: dict(
                    Counter(
                        (
                            "positive" if r["target_positive"] else "negative"
                            for r in roster
                            if r["target_label"] == l
                        )
                    )
                )
                for l in c.LABELS
            },
        },
    )


SOURCE_NOTES = {
    "11518": "Interpretation describes aggression rather than the annotated threatening subtype; meaning is not a label rationale.",
    "11642": "Meaning captures an invader frame but not the full inferiority rationale.",
    "13011": "Target is inferred through racial coding; meaning does not explain the incitement label.",
    "13860": "Interpretation reframes praise of a supremacist as a group belief; not a direct label rationale.",
    "13920": "World-domination inference extends beyond the quoted report; keep as attributed interpretation.",
    "15757": "Target field names Russians, while the hostile action is directed at Muslims. Role mismatch; plain fallback.",
    "16604": "Second-class-citizens interpretation needs context beyond the visible border/law statement.",
    "1769": "Generalizing a report about three people to the whole group is an annotator inference.",
    "18828": "Minorities-taking-over interpretation resolves pronouns/context beyond explicit wording.",
    "19612": "Target field includes a spelling error and multiple groups; preserve raw annotation, no correction.",
    "19737": "Journalist occupation and named references leave protected-target/hate interpretation context dependent.",
    "19746": "One-word source does not identify immigrants; target is additional annotation information.",
    "19819": "Denial-of-rights implication goes beyond the explicit pun; not both label rationales.",
    "19841": "Annotation replaces the addressed journalist/liberal-parrot target with immigrants without a visible bridge; plain fallback.",
    "19848": "Sarcastic border/tunnel statement; polarity of endorsement remains context dependent.",
    "19916": "Target supplied by annotation; implication concerns trivializing racism.",
    "19922": "Political preference paraphrase does not establish hate or a protected-group target.",
    "19968": "Annotation target is immigrants while source uses religious coding; broad target interpretation.",
    "20317": "Taxpayer-money meaning omits the coercive deportation/marking aspect of the threatening label.",
    "20485": "Muslim target inferred from named people and context not explicit group wording.",
    "20720": "Target explicitly unspecified by annotation; retain that uncertainty, never invent a group.",
    "20792": "Meaning omits the source call for violence; not a complete rationale.",
    "20869": "Source threat is clear but immigrants are identified only by the extra annotation.",
    "21533": "Meaning summarizes demographic threat framing, not an unambiguous stereotype rationale.",
    "5942": "Fragmentary meaning but understandable grievance interpretation.",
    "6455": "Meaning supplies demographic-threat context absent from the short slogan.",
    "846": "Inferiority interpretation differs from the source grievance label; no label rewriting.",
    "8601": "Liar implication omits the genocide/grievance framing.",
    "8945": "Minorities-inferior interpretation abstracts a superiority claim; not incitement-specific.",
    "9287": "Meaning supplies racial-policy context and does not by itself explain incitement.",
}
UNSUITABLE = {"implicit_hate-15757", "implicit_hate-19841"}
NOTE_PREFIX = "  Source interpretation note (an annotation, not a fact about a group or the new input): "
GENERIC_SENTENCES = [
    "This note concerns the labelled source.",
    "Its labels describe that example.",
    "Consider the wording and available context.",
    "Do not assume that another text has the same labels.",
    "Similarity alone does not determine the answer.",
    "Use the classification instructions for the new input.",
    "The additional example is one piece of evidence.",
    "Keep the source and the new input distinct.",
]


def interpretation_note(annotation):
    return (
        "Annotator-supplied target: "
        + json.dumps(annotation["target"], ensure_ascii=False)
        + ". Annotator interpretation of the source: "
        + json.dumps(annotation["implied_statement"], ensure_ascii=False)
        + "."
    )


def generic_note(tok, target):
    candidates = [
        " ".join(GENERIC_SENTENCES[:i]) for i in range(1, len(GENERIC_SENTENCES) + 1)
    ]
    return min(
        candidates,
        key=lambda s: (
            abs(len(tok.encode(s, add_special_tokens=False).ids) - target),
            len(s),
        ),
    )


def add_note(plain, note):
    messages = deepcopy(plain["messages"])
    suffix = "\n\n## Text to classify (language: en)\n" + plain["text"]
    c.require(messages[1]["text"].endswith(suffix), "Query boundary differs")
    messages[1]["text"] = (
        messages[1]["text"][: -len(suffix)] + "\n" + NOTE_PREFIX + note + suffix
    )
    return messages


def human():
    root = c.BASE / "human_meaning_v2"
    protocol = "# Existing human implied meanings as correction information\n\nKeep the original 558-input roster (350 hateful, 208 benign), three 40-example correction pools, natural selected source, five ordinary examples plus one correction, and full schema. Generate fresh reference/plain/human/generic outputs: ten settings × 558 inputs × three repeats × three votes = 50,220 ordinary calls. Reference and plain messages must exactly match the saved original study. No query's own stage-3 annotation is accessed when constructing its prediction. Only the already-selected correction source's exact-text annotation is eligible.\n\nSource-only assistant review found 41 of 58 sources have exact human annotations. Two are unsuitable because the supplied target is visibly assigned to the wrong role/entity (15757, 19841); both human and generic arms fall back to plain for these and for missing annotations. Preserve all raw strings and review notes. Do not rewrite target or meaning, manufacture explanations, or select by query outcomes. Remaining interpretations may need context, omit aspects of the source, or not explain the specific gold subtype. Retain and flag these: the experiment tests additional interpretation information, not verified label rationales. Missing annotation is not an empty-label claim. Assistant review is not the planned independent two-human annotation audit.\n\nHuman notes explicitly attribute target and implied meaning to an annotator. The generic arm adds whole-sentence source-use reminders of approximately comparable token length. Record every length difference. This is an active generic reminder control, not a semantically neutral or exactly length-matched placebo. Both arms use the same outer attribution prefix and exact same fallback mask. Extra target/meaning information has annotation cost, and no causal human-versus-assistant authorship claim is permitted.\n\nOne primary contrast: human minus plain seven-type macro F1 on the original 350 hateful inputs, equally averaging three pools and three repeats, with a paired overlap-cluster bootstrap and 95% interval, seed 202610042, 10,000 draws. Common primary mask requires all 18 human/plain outputs for an input. Keep all seven labels; zero denominator=0. This is a new one-contrast supplement, separate from the closed original family. No claim of general uplift from a supporting subgroup alone.\n\nSupporting results: human versus generic, plain/human/generic versus fresh reference, individual labels, hate recall, benign false-positive rate on the 208 mixed-source texts and source strata, all-repeat/pool run values, null counts, tokens and calls. Also report (a) a common input/pool availability stratum fixed from source cards before generation, (b) the exact set where both old assistant and new human information were available, and (c) flagged source-context cases. These are conditional coverage diagnostics, not randomized authorship comparisons. Pool-specific masks retain all repeats and cluster shared texts. No new source selection, retrieval tuning, gold edits, extra head labels, selective cutoff changes or favorable resampling.\n"
    conditions = ["reference"] + [
        f"{policy}_s{seed}"
        for seed in c.SEEDS
        for policy in ("plain", "human", "generic")
    ]
    paths = [
        c.OLD_EX / "prompts.jsonl",
        c.OLD_EX / "roster.jsonl",
        c.OLD_EX / "choices.jsonl",
        c.OLD_EX / "complete.json",
        c.REVIEW / "extension_feasibility.json",
        c.OLD_FEEDBACK / "update_payloads_v1/added_records.jsonl",
        c.ROOT / "data/raw/implicit_hate/implicit_hate_v1_stg3_posts.tsv",
    ]
    prep_start(
        root,
        paths,
        protocol,
        {
            "primary": ["human_minus_plain_seven_type_macro_f1"],
            "primary_interval": 0.95,
            "bootstrap_seed": 202610042,
            "mask": "per input, all 18 human/plain pool/repeat outputs valid",
            "source_quality_exclusions": sorted(UNSUITABLE),
            "no_evaluation_outcomes_used_for_source_review": True,
        },
        conditions,
    )
    sources = {
        r["provenance"]["source_id"]: r
        for r in c.rows(c.OLD_FEEDBACK / "update_payloads_v1/added_records.jsonl")
    }
    detail = c.read(c.REVIEW / "extension_feasibility.json")["human_annotations"][
        "sources_detail"
    ]
    with (c.ROOT / "data/raw/implicit_hate/implicit_hate_v1_stg3_posts.tsv").open(
        encoding="utf8", newline=""
    ) as stream:
        raw = list(csv.DictReader(stream, delimiter="\t"))
    tok = c.tokenizer()
    cards = {}
    for sid in sorted(detail):
        source = sources[sid]
        matched = [
            {"target": r["target"], "implied_statement": r["implied_statement"]}
            for r in raw
            if r["post"] == source["text"] and r["implied_statement"].strip()
        ]
        c.require(matched == detail[sid]["exact_annotations"], "Raw exact join differs")
        unique = {c.sig(r): r for r in matched}
        c.require(len(unique) <= 1, "Multiple meanings require a new frozen rule")
        available = bool(unique) and sid not in UNSUITABLE
        human = interpretation_note(next(iter(unique.values()))) if available else ""
        hn = len(tok.encode(human, add_special_tokens=False).ids) if human else 0
        generic = generic_note(tok, hn) if available else ""
        gn = len(tok.encode(generic, add_special_tokens=False).ids) if generic else 0
        number = sid.removeprefix("implicit_hate-")
        note = (
            SOURCE_NOTES.get(
                number,
                "Plausible source interpretation; not independently verified and not guaranteed to justify every type label.",
            )
            if matched
            else "No exact human meaning annotation; plain fallback."
        )
        cards[sid] = {
            "source_id": sid,
            "source_text": source["text"],
            "source_text_sha256": (
                c.sha_text(source["text"])
                if hasattr(c, "sha_text")
                else __import__("hashlib").sha256(source["text"].encode()).hexdigest()
            ),
            "gold_types": source["meta"]["hate_types"],
            "raw_annotations": matched,
            "eligible": available,
            "assistant_review_note": note,
            "reviewer": "assistant, source-only; not a human judgment",
            "review_flagged": number in SOURCE_NOTES,
            "human_note": human,
            "generic_note": generic,
            "human_note_tokens": hn,
            "generic_note_tokens": gn,
            "generic_minus_human_tokens": gn - hn,
            "old_assistant_available": detail[sid]["assistant_status"] == "available",
        }
    c.require(
        len(cards) == 58
        and sum((bool(r["raw_annotations"]) for r in cards.values())) == 41
        and (sum((r["eligible"] for r in cards.values())) == 39),
        "Coverage changed",
    )
    c.jsonlines(root / "source_review_cards.jsonl", list(cards.values()))
    old = {
        (p["item_id"], p["condition"]): p for p in c.rows(c.OLD_EX / "prompts.jsonl")
    }
    choices = {
        (r["query_id"], r["pool_seed"]): r for r in c.rows(c.OLD_EX / "choices.jsonl")
    }
    roster = [
        {**r, "population": r["cohort"]} for r in c.rows(c.OLD_EX / "roster.jsonl")
    ]
    prompts = []
    availability = []
    unchanged = 0
    for r in roster:
        ref = old[r["id"], "reference"]
        for cond in conditions:
            if cond == "reference":
                original = ref
                context = deepcopy(ref["context"])
                messages = deepcopy(ref["messages"])
                sid = None
                card = None
                policy = "reference"
            else:
                policy, seed_string = cond.split("_s")
                seed = int(seed_string)
                original = old[r["id"], f"plain_always_s{seed}"]
                context = deepcopy(original["context"])
                feedback_id = choices[r["id"], seed]["source_id"]
                sid = feedback_id.removeprefix("sq3fb-")
                card = cards[sid]
                c.require(
                    original["source_id"] == feedback_id,
                    "Saved source identity differs",
                )
                messages = (
                    add_note(original, card[policy + "_note"])
                    if policy in ("human", "generic") and card["eligible"]
                    else deepcopy(original["messages"])
                )
                if policy == "plain":
                    availability.append(
                        {
                            "key": r["key"],
                            "pool_seed": seed,
                            "source_id": sid,
                            "human_eligible": card["eligible"],
                            "old_assistant_available": card["old_assistant_available"],
                            "source_flagged": card["review_flagged"],
                            "population": r["population"],
                        }
                    )
            if policy in ("reference", "plain") or not card["eligible"]:
                c.require(
                    messages == original["messages"], "Reference/plain/fallback drift"
                )
                unchanged += 1
            prompts.append(
                {
                    "key": r["key"],
                    "item_id": r["id"],
                    "condition": cond,
                    "population": r["population"],
                    "lang": "en",
                    "text": r["text"],
                    "source_id": sid,
                    "note_eligible": bool(card and card["eligible"]),
                    "context": context,
                    "messages": messages,
                    "context_sha256": c.sig(context),
                    "messages_sha256": c.sig(messages),
                    "reference_input_tokens": c.token_count(tok, messages),
                    "scientific_plan_sha256": c.sha(root / "scientific_plan_v1.json"),
                }
            )
    c.jsonlines(root / "availability.jsonl", availability)
    c.write(
        root / "SOURCE_REVIEW.md",
        "# Source-only quality review\n\nRaw human annotations are preserved, not endorsed as facts. The assistant reviewed all 41 available entries; two receive plain fallback for apparent target-role mismatches. Seventeen lack exact annotations. Thirty-nine are retained with limitations. This is separate from independent human review of evaluation texts.\n\n"
        + "\n".join(
            (
                f"- `{sid}`: {('retained' if v['eligible'] else 'plain fallback')}. {v['assistant_review_note']}"
                for sid, v in cards.items()
            )
        )
        + "\n",
    )
    finish(
        root,
        prompts,
        roster,
        {
            "source_cards": 58,
            "exact_human_available": 41,
            "human_retained": 39,
            "unavailable_or_unsuitable": 19,
            "exact_plain_reference_fallback_matches": unchanged,
            "availability": {
                pop: {
                    "pairs": sum((r["population"] == pop for r in availability)),
                    "human_eligible": sum(
                        (
                            r["population"] == pop and r["human_eligible"]
                            for r in availability
                        )
                    ),
                }
                for pop in sorted({r["population"] for r in availability})
            },
            "generic_minus_human_tokens_min": min(
                (r["generic_minus_human_tokens"] for r in cards.values())
            ),
            "generic_minus_human_tokens_max": max(
                (r["generic_minus_human_tokens"] for r in cards.values())
            ),
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", choices=["factorial", "human"])
    args = p.parse_args()
    with patch.object(
        socket.socket, "connect", side_effect=AssertionError("Offline preparation")
    ):
        (factorial if args.study == "factorial" else human)()


if __name__ == "__main__":
    main()
