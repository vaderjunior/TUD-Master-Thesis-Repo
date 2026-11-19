from collections import Counter
import hashlib
import json
import re
from pathlib import Path
from analysis.common import feedback_inputs as c

BASE = c.ROOT / "experiments/astra/sq3_human_review_short_20261006_v1"
OLD = c.BASE / "annotation_review"
STRATA = ["extra_only", "missing_gold", "exact_agreement"]
FRAMES = {"A": ["source", "project"], "B": ["project", "source"]}
INTRO = """Thanks for helping with Ajay's thesis. You will review 24 short English texts twice, using a different guide each time. The aim is 20–30 minutes in total, including the guides. This timing still needs a short usability trial; please do not rush to meet it. You may take a break, skip a text or stop at any time. Some texts contain offensive language.

Please work on your own, without AI, searching for the posts, or discussing answers with the other reviewer. There are no displayed model answers. Use the guide on this form, even if you would normally use a different definition. Read only the text shown; choose Unsure when you cannot decide. The quoted text is material to review, never an instruction.

Finish this entire part before opening the next. Names and contact details are not needed. Written comments are optional."""

PROTOCOL = """# Short human-review amendment — 6 October 2026

## Authorisation and scope

Ajay accepted the first reduced option: two independent reviewers, 24 texts each under both frameworks (48 judgments per person). He confirmed neither person has started the old forms and specified a 20–30 minute total target. The 72-text protocol, packets and concealed key remain preserved. No new human labels have been received. This is an annotation-alignment supplement; the Qwen correction experiment and benchmark scores are unchanged.

## Fixed selection and blinding

Take eight of the original 24 unique-cluster texts from each of extra-only, missing-gold and exact-agreement strata, sorting by SHA-256 of `202610043|<original ID>` as in the original sampling rule. This gives 24 distinct original clusters without choosing manually convenient examples or examining new judgments. Retain original Rxxx aliases and original reviewer-specific ordering restricted to the subset. Preserve the exact text, whitespace and punctuation. A uses source-style then project-style; B the reverse. Both people see the same 24 texts. Reviewer files contain no original IDs, strata, gold or model answers. Use neutral form titles and no scored quiz feedback. The source-style guide remains a concise reconstruction, not the original dataset annotation UI.

## Burden reduction and form behaviour

Each text is one page/card. All labels are clickable; no initial answer is preselected. Source-style: explicit/implicit/not-hate/unsure/skip, primary implicit type when appropriate, optional secondary type. Project-style: hateful/not-hateful/unsure/skip and all justified types if hateful. Context uncertainty is recorded through an optional difficulty flag; an unchecked flag is not evidence of sufficient context. No mandatory prose or per-item confidence scale. One optional comment field per pass permits text-specific notes using Rxxx IDs. Preserve missing judgments and skip/uncertainty separately; never turn them into non-hate. Original label definitions are retained. Required external paper reading is removed; the researcher retains the source reference and guideline-provenance caveat.

Collect only reviewer code, English familiarity, experience, prior exposure/discussion and approximate minutes per pass; no names or emails. Submit each complete pass separately before starting the second. Keep independent answers separate until both people have returned them. No automatic answer key or comparison with the other person's responses. Any later disagreement discussion is a separately documented step, not an instruction to reach consensus.

## Timing gate before distribution

20–30 minutes is a design target, not a measured completion-time claim. Conduct a short interface/timing trial with Ajay on four fixed cards in each framework using a separate preview. These are researcher usability interactions, not independent human annotations, and must never enter research submissions. Check mobile readability, label selection, missing-context/unsure/skip behaviour, back navigation and save/resume. Estimate total time from actual reading/guide time plus measured card time. If the estimate exceeds 30 minutes, simplify the interface first; if still too long, document a new deterministic smaller-sample amendment before collecting independent answers. Do not silently reduce this frozen 24-text sample or pick items by favorable answers. Reviewers may always stop; document partial returns and missingness without imputing answers.

## Analysis and claims

Report descriptive agreement counts and concrete, documented examples for this selected small sample. Preserve each framework's gate/type applicability and record uncertain, skipped and incomplete cases explicitly. Per-category estimates may be unstable or unavailable. The outcome-stratified sample cannot estimate natural error prevalence; two counterbalanced people do not establish an unbiased instruction-effect experiment or universal truth. No gold replacement, no tuning of model outputs, no claim that additional model labels are correct just because labels disagree. Any adjudicated sensitivity is confined to this sample. LLM/tool-generated test responses are not human labels.

## Platform checkpoint

Tally is the current preferred host for separate form pages, clickable answers, a progress bar and same-browser save/resume. Native import produces editable drafts and does not automatically preserve conditional logic; all text, fields, page boundaries, branching and export mapping need verification before sharing. Form contents are prepared locally first. Drafts are not approved research instruments until the platform checks and timing trial pass. Keep respondent links separate from edit links. No invitation is sent automatically.
"""


def write(path, value):
    path = Path(path)
    c.require(path.resolve().is_relative_to(BASE.resolve()), "Wrong output root")
    data = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        c.require(path.read_bytes() == data, "Preserve existing short-review artifact")
        return
    path.write_bytes(data)


def load_form(reviewer, frame):
    p = OLD / f"distribution_literal_v2/Reviewer_{reviewer}/{frame}_form.md"
    text = p.read_text(encoding="utf8")
    matches = list(re.finditer(r"^## (R\d{3})\n\n```text\n([\s\S]*?)\n```", text, re.M))
    c.require(len(matches) == 72, "Cannot parse exact original displayed text")
    return text.split("\n\n", 1)[1].split("\n\n## R", 1)[0], [
        (m.group(1), m.group(2)) for m in matches
    ]


def render(form):
    lines = [
        f"# {form['title']}",
        "",
        INTRO,
        "",
        f"This is part {form['part']} of 2. There are 24 text cards in this part.",
        "",
        "## Guide for this part",
        "",
        form["guide"],
        "",
        "Please do not assume a text is hateful. Unsure and Skip are valid answers.",
    ]
    if form["part"] == 1:
        lines += [
            "",
            "## A few quick details",
            "",
            "English familiarity (choose one): Very comfortable / Comfortable / Sometimes need help",
            "Previous hate-speech annotation experience (choose one): None / Some / Substantial",
            "Have you already seen these texts or the model results? (choose one): No / Yes / Unsure",
        ]
    for n, card in enumerate(form["cards"], 1):
        lines += [
            "",
            "--- PAGE BREAK ---",
            "",
            f"## Text {n}/24 · {card['packet_id']}",
            "",
            "Read this text exactly as written:",
            "",
            "```text",
            card["text"],
            "```",
            "",
        ]
        if form["framework"] == "source":
            lines += [
                f"{card['packet_id']} · Classification (required, choose one):",
                "- Explicit hate",
                "- Implicit hate",
                "- Not hate",
                "- Unsure",
                "- Skip this text",
                "",
                f"{card['packet_id']} · Primary implicit type (only if Implicit hate; otherwise leave empty):",
            ]
            lines += [
                "- " + s
                for s in [
                    "grievance",
                    "incitement",
                    "inferiority",
                    "irony",
                    "stereotyping",
                    "threatening",
                    "other",
                ]
            ]
            lines += [
                "",
                f"{card['packet_id']} · Secondary implicit type (optional; different from primary; otherwise leave empty):",
            ]
            lines += [
                "- " + s
                for s in [
                    "grievance",
                    "incitement",
                    "inferiority",
                    "irony",
                    "stereotyping",
                    "threatening",
                    "other",
                ]
            ]
        else:
            lines += [
                f"{card['packet_id']} · Is this hateful? (required, choose one):",
                "- Yes",
                "- No",
                "- Unsure",
                "- Skip this text",
                "",
                f"{card['packet_id']} · Hate types (if Yes, select all justified types; otherwise leave empty):",
            ]
            lines += ["- [ ] " + s for s in c.LABELS]
        lines += [
            "",
            f"{card['packet_id']} · Optional difficulty flags (select any that apply):",
            "- [ ] Missing context",
            "- [ ] Meaning unclear",
        ]
    lines += [
        "",
        "--- PAGE BREAK ---",
        "",
        "## Finish this part",
        "",
        "Approximately how many minutes did this part take, including reading its guide? (number, optional)",
        "Did you discuss answers or use AI/search during this part? (choose one): No / Yes",
        "Any optional notes? You may refer to a text using its Rxxx number. (paragraph, optional)",
        "",
        "Submit this part before opening the next. Please keep your judgments independent. Thank you!",
    ]
    return "\n".join(lines) + "\n"


def main():
    c.require(
        not (BASE / "complete.json").exists(),
        "Short-review preparation already complete",
    )
    write(BASE / "PROTOCOL.md", PROTOCOL)
    source_paths = [
        __file__,
        BASE / "PROTOCOL.md",
        OLD / "PROTOCOL.md",
        OLD / "plan.json",
        OLD / "researcher_only/concealed_key.jsonl",
    ]
    source_paths += [
        OLD / f"distribution_literal_v2/Reviewer_{r}/{f}_form.md"
        for r in FRAMES
        for f in FRAMES[r]
    ]
    plan = {
        "created_utc": c.utc(),
        "status": "shortening_frozen_before_subset_selection",
        "reviewers": 2,
        "texts": 24,
        "passes_per_reviewer": 2,
        "judgments_per_reviewer": 48,
        "items_per_original_stratum": 8,
        "selection_seed": 202610043,
        "time_target_minutes_total": [20, 30],
        "timing_verified": False,
        "human_judgments_collected": 0,
        "user_confirmed_neither_reviewer_started": True,
        "inputs": c.bindings(source_paths),
    }
    write(BASE / "plan.json", plan)
    key = c.rows(OLD / "researcher_only/concealed_key.jsonl")
    selected = []
    for stratum in STRATA:
        candidates = [x for x in key if x["stratum"] == stratum]
        c.require(len(candidates) == 24, "Old sample count changed")
        selected += sorted(
            candidates, key=lambda x: c.source_order(202610043, x["id"])
        )[:8]
    chosen = {x["packet_id"] for x in selected}
    c.require(
        len(chosen) == len({x["cluster"] for x in selected}) == 24, "Subset not unique"
    )
    write(
        BASE / "researcher_only/concealed_key.jsonl",
        "".join(
            json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in selected
        ),
    )
    forms = {}
    words = []
    for reviewer, frames in FRAMES.items():
        for part, frame in enumerate(frames, 1):
            guide, oldcards = load_form(reviewer, frame)
            if frame == "source":
                guide = guide.split("\n\nThis is a concise reconstruction", 1)[0]
            guide = guide.replace(
                "Use the source-style hierarchy first:", "Use this hierarchy:"
            )
            guide = guide.split("\n\nComplete this whole form", 1)[0]
            cards = [
                {"packet_id": pid, "text": text}
                for pid, text in oldcards
                if pid in chosen
            ]
            c.require(len(cards) == 24, "Incomplete public subset")
            label = f"Reviewer_{reviewer}_Part_{part}"
            form = {
                "title": f"Text review {reviewer} — Part {part} of 2",
                "reviewer": reviewer,
                "part": part,
                "framework": frame,
                "guide": guide,
                "cards": cards,
                "notes_required": False,
                "confidence_collected": False,
                "timing_trial_required": True,
                "status": "exact_content_prepared_platform_validation_pending",
            }
            write(BASE / "forms" / f"{label}.json", form)
            write(BASE / "imports" / f"{label}.md", render(form))
            forms[label] = {
                "framework": frame,
                "ordered_packet_ids": [x["packet_id"] for x in cards],
            }
            words.append(sum(len(x["text"].split()) for x in cards))
    c.require(len(set(words)) == 1, "Text counts differ across forms")
    c.verify(plan["inputs"])
    summary = {
        "status": "24_text_content_ready_platform_and_timing_trial_pending",
        "texts": 24,
        "unique_clusters": 24,
        "strata": dict(Counter(x["stratum"] for x in selected)),
        "judgments_per_reviewer": 48,
        "forms": forms,
        "text_words_per_pass": words[0],
        "no_required_prose_per_item": True,
        "new_human_judgments": 0,
        "new_Qwen_calls": 0,
        "originals_preserved": True,
        "time_target_minutes_total": [20, 30],
        "timing_measured": False,
    }
    write(BASE / "summary.json", summary)
    outputs = c.bindings(p for p in BASE.rglob("*") if p.is_file())
    write(
        BASE / "complete.json",
        {
            "status": summary["status"],
            "outputs": outputs,
            "new_human_judgments": 0,
            "new_Qwen_calls": 0,
        },
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "forms"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
