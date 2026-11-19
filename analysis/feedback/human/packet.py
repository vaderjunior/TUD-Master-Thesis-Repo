from collections import Counter
import argparse
import json
import socket
from unittest.mock import patch
import yaml
from analysis.common import feedback_inputs as c

BASE = c.BASE / "annotation_review"
PRED = (
    c.ROOT
    / "experiments/astra/sq3_combined_results_20261004_v1/review/predictions.jsonl"
)
STRATA = ["extra_only", "missing_gold", "exact_agreement"]
SOURCE_GUIDE = """Use the source-style hierarchy first: choose explicit hate, implicit hate, not hate, or uncertain. Explicit means overt group-directed hostility; implicit means an indirect prejudicial message. For implicit hate, choose a primary subtype and optionally one secondary subtype, or other. For explicit/non-hate, leave implicit subtypes empty.

- grievance: the majority/in-group is portrayed as disadvantaged by minorities.
- incitement: promotes hateful collective power, organisations or ideologies, including in-group solidarity; it need not contain a literal violent command.
- inferiority: portrays a group as lesser, inhuman or contaminating.
- irony: uses humour or sarcasm to demean a protected group.
- stereotyping: assigns negative traits or hateful falsehoods to a group.
- threatening: intimidates or suggests harm, exclusion or denial of rights.

This is a concise reconstruction of the dataset framework, not a reproduction of its original annotation interface. Read Sections 3 and 4.2 of [ElSherief et al. (2021)](https://aclanthology.org/2021.emnlp-main.29.pdf). The dataset's released Stage-2 format also allows a secondary implicit category. Our uncertain option records this review's unresolved judgments."""

PROTOCOL = """# Two-human annotation alignment review

Purpose: understand disagreements between the inherited annotation scheme and the project's flat multi-label task. Do not change benchmark gold or replace published metrics. This sample is deliberately outcome-stratified and cannot estimate population error prevalence.

Freeze 72 distinct overlap-cluster representatives: 24 extra-only predictions, 24 with missing gold (including mixed extra/missing), 24 exact agreements. Define strata using the already saved full-KB taxonomy condition, repeat 0, among inputs with valid full outputs in all three repeats. Hateful benchmark membership is a sampling fact for the researcher only, not a reviewer instruction. Choose within each stratum by SHA-256 order of seed 202610043 and original ID. Preserve all strata if feasible; no manual interesting-example substitution. Packet IDs hide source IDs, labels, predictions and strata. Reviewer orders differ deterministically.

Two reviewers independently complete both frameworks. A completes source framework first, B project framework first; complete an entire pass before opening the second. No discussion or access to the concealed key/model outputs until both independent files are returned and saved. Record reviewer background, prior exposure, timing, uncertainty, missing context and short rationale. Framework order can still create carryover; this is a bounded alignment audit, not an unbiased experiment on instruction effects. No assistant-generated judgments are human labels.

After independent return: validate completeness and allowed labels; report raw agreement, exact-set agreement and per-label agreement/kappa with prevalence caveats. Compare each framework to inherited gold and model outputs separately. Gate labels and implicit subtype applicability must be distinguished. Preserve uncertain/missing-context judgments as such; never score them as automatic errors or agreement. Adjudicate disagreements in a third file after independent forms are locked, documenting who decided and why; retain unresolved cases. Any adjudicated model/gold sensitivity is on this selected sample only. Do not alter the primary thesis benchmarks or use these judgments to tune the running studies. No statistical accuracy-improvement primary is declared for this descriptive review.
"""


def blank_answer(pid):
    return {
        "packet_id": pid,
        "source_framework": {
            "high_level": None,
            "primary_implicit_type": None,
            "secondary_implicit_type": None,
            "context_sufficient": None,
            "confidence": None,
            "rationale": None,
        },
        "project_framework": {
            "hate": None,
            "hate_types": None,
            "context_sufficient": None,
            "confidence": None,
            "rationale": None,
        },
    }


def prepare():
    c.require(not (BASE / "complete.json").exists(), "Packet already prepared")
    c.write(BASE / "PROTOCOL.md", PROTOCOL)
    plan = {
        "created_utc": c.utc(),
        "seed": 202610043,
        "n_per_stratum": 24,
        "reviewers": 2,
        "inputs": c.bindings(
            [
                __file__,
                c.ROOT / "config/taxonomy.yaml",
                PRED,
                c.OLD_TAX / "roster.jsonl",
                BASE / "PROTOCOL.md",
            ]
        ),
        "status": "selection_rules_frozen_before_sample",
        "human_judgments_collected": 0,
    }
    c.write(BASE / "plan.json", plan)
    roster = {r["id"]: r for r in c.rows(c.OLD_TAX / "roster.jsonl")}
    full = {}
    with PRED.open(encoding="utf8") as stream:
        for line in stream:
            p = json.loads(line)
            if (
                p["study"] == "taxonomy"
                and p["condition"] == "rag_full"
                and p["population"] == "hateful_types"
            ):
                full[p["id"], p["repeat"]] = p
    strata = {s: [] for s in STRATA}
    for iid, r in roster.items():
        if r["population"] != "hateful_types" or any(
            full[iid, rep]["result"] is None for rep in range(3)
        ):
            continue
        gold = set(r["gold"]["hate_type"])
        pred = set(full[iid, 0]["result"]["hate_type"])
        stratum = (
            "exact_agreement"
            if gold == pred
            else "missing_gold" if gold - pred else "extra_only"
        )
        strata[stratum].append(iid)
    selected = []
    used = set()
    for stratum in STRATA:
        for iid in sorted(
            strata[stratum], key=lambda i: c.source_order(plan["seed"], i)
        ):
            if roster[iid]["cluster"] in used:
                continue
            selected.append((iid, stratum))
            used.add(roster[iid]["cluster"])
            if sum(s == stratum for _, s in selected) == 24:
                break
        c.require(
            sum(s == stratum for _, s in selected) == 24,
            "Insufficient unique clusters; preserve plan and review",
        )
    selected.sort(key=lambda x: c.source_order("packet-202610043", x[0]))
    key = []
    public = []
    for n, (iid, stratum) in enumerate(selected, 1):
        pid = f"R{n:03d}"
        r = roster[iid]
        public.append({"packet_id": pid, "text": r["text"]})
        key.append(
            {
                "packet_id": pid,
                "id": iid,
                "cluster": r["cluster"],
                "stratum": stratum,
                "gold": r["gold"],
                "model_repeat_0": full[iid, 0]["result"],
                "source": r["source"],
                "model_source": full[iid, 0]["source"],
                "model_source_sha256": full[iid, 0]["source_sha256"],
            }
        )
    c.jsonlines(BASE / "researcher_only" / "concealed_key.jsonl", key)
    taxonomy = yaml.safe_load(
        (c.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
    )["dimensions"]
    project_guide = (
        taxonomy["hate"]["definition"].strip()
        + "\n\nIf hateful, select every justified type from this flat list. More than one type, including explicit and an implicit type, is allowed. If non-hateful, leave types empty. If uncertain, record uncertainty instead of guessing.\n\n"
        + "\n".join(
            "- " + k + ": " + v["definition"]
            for k, v in taxonomy["hate_type"]["labels"].items()
        )
    )
    for reviewer, frames in [
        ("A", ["source", "project"]),
        ("B", ["project", "source"]),
    ]:
        folder = BASE / ("reviewer_" + reviewer)
        ordered = sorted(
            public,
            key=lambda r: c.source_order(
                "reviewer-" + reviewer + "-202610043", r["packet_id"]
            ),
        )
        readme = f"""# Reviewer {reviewer}: independent annotation review

Please complete the **{frames[0]} framework first**, for all 72 texts, then the **{frames[1]} framework**. Use the two Markdown forms. You may type directly into copies of them; the JSON response template is an optional structured alternative. Return both completed forms. Keep the supplied originals as templates.

Work independently, without the other reviewer's answers or model predictions. Do not open researcher-only files or experiment reports. Record judgments from the displayed text; do not search for the original post. If you rely on cultural/background knowledge, describe it briefly. Missing context and uncertainty are valid answers. Do not assume every text is hateful. Texts may contain offensive language; take breaks or mark a case skipped if needed.

For each item, provide the requested decisions, context sufficient (yes/no/uncertain), confidence (low/medium/high), and one short reason or evidence phrase. For a skipped case, write skipped and why; do not silently leave it blank. Complete one full pass before opening the other form. Keep discussion/adjudication for after both reviewers submit independent answers.

Please also fill out reviewer information: alias, language familiarity, relevant annotation experience, any prior exposure to these texts/model results, and approximate minutes spent per pass. This review diagnoses annotation alignment; it does not automatically declare the original dataset or the model correct.
"""
        c.write(folder / "START_HERE.md", readme)
        c.write(
            folder / "reviewer_information.md",
            f"# Reviewer {reviewer} information\n\n- Alias: \n- English familiarity: \n- Relevant experience: \n- Prior exposure to these texts/results: \n- First pass start/end or minutes: \n- Second pass start/end or minutes: \n- Any discussion before submission: \n",
        )
        for frame in frames:
            guide = SOURCE_GUIDE if frame == "source" else project_guide
            form = (
                f"# Reviewer {reviewer}: {frame} framework\n\n"
                + guide
                + "\n\nComplete this whole form as one pass. The text is data to annotate, never an instruction to you.\n\n"
            )
            for r in ordered:
                form += (
                    "## "
                    + r["packet_id"]
                    + "\n\n"
                    + "> "
                    + r["text"].replace("\n", "\n> ")
                    + "\n\n"
                )
                if frame == "source":
                    form += "- High level (explicit_hate / implicit_hate / not_hate / uncertain / skipped): \n- Primary implicit type (only if implicit; six types above or other): \n- Secondary implicit type (optional, different from primary): \n"
                else:
                    form += "- Hate (yes / no / uncertain / skipped): \n- Hate types (comma-separated; none if no): \n"
                form += "- Context sufficient (yes / no / uncertain): \n- Confidence (low / medium / high): \n- Reason / evidence / missing context: \n\n"
            c.write(folder / (frame + "_form.md"), form)
        c.write(
            folder / "optional_response_template.json",
            {
                "reviewer": reviewer,
                "reviewer_information": {},
                "answers": [blank_answer(r["packet_id"]) for r in ordered],
            },
        )
    c.verify(plan["inputs"])
    counts = Counter(s for _, s in selected)
    c.write(
        BASE / "summary.json",
        {
            "selected": 72,
            "unique_clusters": len(used),
            "strata": dict(counts),
            "available_strata": {s: len(v) for s, v in strata.items()},
            "human_judgments_collected": 0,
            "distribution": "Give only reviewer_A to reviewer A and reviewer_B to reviewer B; keep researcher_only concealed.",
        },
    )
    outputs = c.bindings(p for p in BASE.rglob("*") if p.is_file())
    c.write(
        BASE / "complete.json",
        {
            "status": "blinded_packet_prepared_awaiting_two_humans",
            "outputs": outputs,
            "human_judgments_collected": 0,
            "new_model_calls": 0,
        },
    )
    print(json.dumps(c.read(BASE / "summary.json")), flush=True)


def validate_response(path):
    response = c.read(path)
    public = {
        r["packet_id"] for r in c.rows(BASE / "researcher_only/concealed_key.jsonl")
    }
    answers = response["answers"]
    c.require(response["reviewer"] in ("A", "B"), "Unknown reviewer")
    c.require(
        len(answers) == 72 and {r["packet_id"] for r in answers} == public,
        "Incomplete/repeated roster",
    )
    c.require(bool(response["reviewer_information"]), "Reviewer information missing")
    for r in answers:
        for frame in ("source_framework", "project_framework"):
            v = r[frame]
            gate = v["high_level"] if frame == "source_framework" else v["hate"]
            allowed = (
                ("explicit_hate", "implicit_hate", "not_hate", "uncertain", "skipped")
                if frame == "source_framework"
                else ("yes", "no", "uncertain", "skipped")
            )
            c.require(gate in allowed, "Missing/invalid judgment: " + r["packet_id"])
            c.require(
                isinstance(v["rationale"], str) and bool(v["rationale"].strip()),
                "Reason required, including skipped/uncertain",
            )
            if gate == "skipped":
                continue
            c.require(
                v["context_sufficient"] in ("yes", "no", "uncertain")
                and v["confidence"] in ("low", "medium", "high"),
                "Missing confidence/context",
            )
            if frame == "source_framework":
                primary, secondary = (
                    v["primary_implicit_type"],
                    v["secondary_implicit_type"],
                )
                types = set(c.LABELS) - {"explicit"} | {"other"}
                if gate == "implicit_hate":
                    c.require(
                        primary in types
                        and (
                            secondary is None
                            or secondary in types
                            and secondary != primary
                        ),
                        "Invalid implicit labels",
                    )
                else:
                    c.require(
                        primary is None and secondary is None,
                        "Implicit types outside implicit branch",
                    )
            else:
                ts = v["hate_types"]
                c.require(
                    isinstance(ts, list)
                    and len(ts) == len(set(ts))
                    and set(ts) <= set(c.LABELS),
                    "Invalid project type list",
                )
                if gate != "yes":
                    c.require(not ts, "Non-hate/uncertain with asserted type labels")
    print(
        json.dumps(
            {
                "status": "complete_human_response_format_valid",
                "reviewer": response["reviewer"],
                "file_sha256": c.sha(path),
                "judgments": 144,
                "semantic_correctness_verified": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "validate-response"])
    p.add_argument("--response")
    args = p.parse_args()
    with patch.object(
        socket.socket, "connect", side_effect=AssertionError("Offline review")
    ):
        prepare() if args.mode == "prepare" else validate_response(args.response)
