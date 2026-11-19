from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

from analysis.feedback.human import response_audit as decoder

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_human_review_results_20261010_v1"
SHORT = ROOT / "experiments/astra/sq3_human_review_short_20261006_v1"
OLD = ROOT / "experiments/astra/sq3_additions_20261004_v1/annotation_review"
LABELS = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
STRATA = ["extra_only", "missing_gold", "exact_agreement"]
FORMS = {
    "Reviewer_A_Part_1": ("A", "source", "jaOkaR"),
    "Reviewer_A_Part_2": ("A", "project", "q42kG8"),
    "Reviewer_B_Part_1": ("B", "project", "810RZk"),
    "Reviewer_B_Part_2": ("B", "source", "ODWo7k"),
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [
        json.loads(s) for s in Path(path).read_text(encoding="utf8").splitlines() if s
    ]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(path, value):
    data = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.read_bytes() == data,
            "Preserve existing analysis artifact: " + str(path),
        )
    else:
        path.write_bytes(data)


def agreement(pairs):

    n = len(pairs)
    if not n:
        return {
            "n": 0,
            "agree": 0,
            "rate": None,
            "kappa": None,
            "expected_agreement": None,
            "cells": [],
        }
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    same = sum(a == b for a, b in pairs)
    pe = sum(left[k] * right[k] for k in set(left) | set(right)) / (n * n)
    cells = [
        {"left": a, "right": b, "n": count}
        for (a, b), count in sorted(Counter(pairs).items(), key=lambda z: str(z[0]))
    ]
    return {
        "n": n,
        "agree": same,
        "rate": same / n,
        "kappa": (same / n - pe) / (1 - pe) if pe < 1 else None,
        "expected_agreement": pe,
        "cells": cells,
    }


def binary_name(card):
    if card["binary_hate"] is not None:
        return "Hate" if card["binary_hate"] else "Not hate"
    return card["gate"]


def annotation_set(card):

    if card["binary_hate"] is None:
        return None
    if not card["binary_hate"]:
        return set()
    if card["gate"] == "Explicit hate":
        return {"explicit"}
    return set(card["types"]) if card["type_state"] == "assigned" else None


def set_agreement(pairs, labels):
    n = len(pairs)
    exact = sum(a == b for a, b in pairs)
    per_label = {}
    for label in labels:
        per_label[label] = agreement([(label in a, label in b) for a, b in pairs])
        per_label[label]["left_positive"] = sum(label in a for a, _ in pairs)
        per_label[label]["right_positive"] = sum(label in b for _, b in pairs)
    return {
        "n": n,
        "exact": exact,
        "exact_rate": exact / n if n else None,
        "mean_jaccard": (
            sum(len(a & b) / len(a | b) if a | b else 1 for a, b in pairs) / n
            if n
            else None
        ),
        "per_label": per_label,
    }


def form_summary(form, decoded, export):
    meta = {k: v for k, v in decoded["raw_fields"].items() if "/" not in k}
    cards = decoded["cards"]
    subs = export["submissions"][0]
    fine = [c for c in cards if c["type_state"] == "assigned"]
    return {
        "form_key": form,
        "framework": decoded["framework"],
        "submission_id": decoded["submission_id"],
        "respondent_id": subs.get("respondentId"),
        "submitted_at": subs.get("submittedAt"),
        "metadata": meta,
        "n": len(cards),
        "gates": dict(Counter(c["gate"] for c in cards)),
        "binary_with_unknown": dict(Counter(binary_name(c) for c in cards)),
        "type_states": dict(Counter(c["type_state"] for c in cards)),
        "multiple_types": sum(len(c["types"]) > 1 for c in fine),
        "explicit_plus_other": sum(
            "explicit" in c["types"] and len(c["types"]) > 1 for c in fine
        ),
        "assigned_type_count": sum(len(c["types"]) for c in fine),
        "type_counts": dict(Counter(t for c in fine for t in c["types"])),
        "difficulty": dict(Counter(f for c in cards for f in c["difficulty"])),
        "any_difficulty_cards": [c["id"] for c in cards if c["difficulty"]],
        "guide_toggle_checked_on_final_saved_response": sum(
            c["guide_open"] for c in cards
        ),
        "review_flags": [
            {"id": c["id"], "flags": c["review_flags"]}
            for c in cards
            if c["review_flags"]
        ],
    }


def compare_reviewers(left, right, frame):
    ids = sorted(left)
    resolved = [
        i
        for i in ids
        if left[i]["binary_hate"] is not None and right[i]["binary_hate"] is not None
    ]
    fine = [
        i for i in ids if left[i]["type_state"] == right[i]["type_state"] == "assigned"
    ]
    primary = (
        [i for i in fine if left[i]["gate"] == right[i]["gate"] == "Implicit hate"]
        if frame == "source"
        else []
    )
    same_codes = [
        i
        for i in ids
        if annotation_set(left[i]) is not None and annotation_set(right[i]) is not None
    ]
    return {
        "coarse_including_uncertain_skip": agreement(
            [(left[i]["gate"], right[i]["gate"]) for i in ids]
        ),
        "binary_including_uncertain_skip": agreement(
            [(binary_name(left[i]), binary_name(right[i])) for i in ids]
        ),
        "binary_both_resolved": {
            **agreement(
                [(left[i]["binary_hate"], right[i]["binary_hate"]) for i in resolved]
            ),
            "ids": resolved,
        },
        "type_sets_both_applicable_assigned": {
            **set_agreement(
                [(set(left[i]["types"]), set(right[i]["types"])) for i in fine],
                LABELS if frame == "project" else LABELS[1:] + ["other"],
            ),
            "ids": fine,
        },
        "source_primary_both_implicit": {
            **agreement(
                [(left[i]["primary"][0], right[i]["primary"][0]) for i in primary]
            ),
            "ids": primary,
        },
        "complete_response_code_match": {
            "n": len(same_codes),
            "exact": sum(
                annotation_set(left[i]) == annotation_set(right[i]) for i in same_codes
            ),
            "ids": same_codes,
        },
        "unresolved_either": [i for i in ids if i not in resolved],
    }


def compare_guides(source, project):
    ids = sorted(source)
    resolved = [
        i
        for i in ids
        if source[i]["binary_hate"] is not None
        and project[i]["binary_hate"] is not None
    ]
    applicable = [
        i
        for i in ids
        if source[i]["type_state"] == project[i]["type_state"] == "assigned"
    ]
    return {
        "binary_transitions_all": agreement(
            [(binary_name(source[i]), binary_name(project[i])) for i in ids]
        ),
        "binary_both_resolved": {
            **agreement(
                [
                    (source[i]["binary_hate"], project[i]["binary_hate"])
                    for i in resolved
                ]
            ),
            "ids": resolved,
        },
        "hateful_in_both_assigned_implicit_source": {
            **set_agreement(
                [
                    (set(source[i]["types"]), set(project[i]["types"]))
                    for i in applicable
                ],
                LABELS + ["other"],
            ),
            "ids": applicable,
        },
        "source_explicit_to_project": [
            {
                "id": i,
                "project_gate": project[i]["gate"],
                "project_types": project[i]["types"],
            }
            for i in ids
            if source[i]["gate"] == "Explicit hate"
        ],
        "source_implicit_to_project_explicit": [
            i
            for i in ids
            if source[i]["gate"] == "Implicit hate"
            and "explicit" in project[i]["types"]
        ],
    }


def alignment(cards, key, reference):
    ids = sorted(cards)
    get_ref = lambda i: key[i]["gold" if reference == "dataset" else "model_repeat_0"]
    resolved = [i for i in ids if cards[i]["binary_hate"] is not None]
    available = [i for i in ids if annotation_set(cards[i]) is not None]
    positive = [i for i in available if cards[i]["binary_hate"]]
    rows_out = []
    for i in available:
        hs, rs = annotation_set(cards[i]), set(get_ref(i)["hate_type"])
        rows_out.append(
            {
                "id": i,
                "human_code": sorted(hs),
                "reference_types": sorted(rs),
                "extra_to_reference": sorted(hs - rs),
                "missing_from_reference": sorted(rs - hs),
                "exact_code": hs == rs,
                "reference_all_included": rs <= hs,
            }
        )
    return {
        "reference": reference,
        "binary_resolved": {
            **agreement(
                [(cards[i]["binary_hate"], get_ref(i)["hate"]) for i in resolved]
            ),
            "ids": resolved,
        },
        "complete_response_codes": {
            "n": len(available),
            "exact": sum(x["exact_code"] for x in rows_out),
            "all_reference_types_included": sum(
                x["reference_all_included"] for x in rows_out
            ),
            "human_extra_occurrences": sum(
                len(x["extra_to_reference"]) for x in rows_out
            ),
            "human_omitted_occurrences": sum(
                len(x["missing_from_reference"]) for x in rows_out
            ),
        },
        "positive_human_complete_codes": {
            "n": len(positive),
            "exact": sum(
                annotation_set(cards[i]) == set(get_ref(i)["hate_type"])
                for i in positive
            ),
            "ids": positive,
        },
        "per_item_codes": rows_out,
    }


def occurrence_review(key, project_a, project_b, which):
    items = []
    for pid in sorted(key):
        gold, model = set(key[pid]["gold"]["hate_type"]), set(
            key[pid]["model_repeat_0"]["hate_type"]
        )
        labels = (
            model - gold
            if which == "model_extra"
            else gold - model if which == "model_omitted" else gold & model
        )
        a, b = annotation_set(project_a[pid]), annotation_set(project_b[pid])
        for label in sorted(labels):
            av, bv = (None if a is None else label in a), (
                None if b is None else label in b
            )
            state = (
                "unresolved_either"
                if av is None or bv is None
                else (
                    "both_selected"
                    if av and bv
                    else "one_selected" if av or bv else "neither_selected"
                )
            )
            items.append(
                {
                    "id": pid,
                    "label": label,
                    "stratum": key[pid]["stratum"],
                    "A_selected": av,
                    "B_selected": bv,
                    "state": state,
                }
            )
    counts = dict(Counter(x["state"] for x in items))
    return {
        "occurrences": len(items),
        "texts": len({x["id"] for x in items}),
        "counts": counts,
        "both_selected_texts": sorted(
            {x["id"] for x in items if x["state"] == "both_selected"}
        ),
        "per_reviewer": {
            r: {
                "known": sum(x[r + "_selected"] is not None for x in items),
                "selected": sum(x[r + "_selected"] is True for x in items),
                "not_selected": sum(x[r + "_selected"] is False for x in items),
                "unknown": sum(x[r + "_selected"] is None for x in items),
            }
            for r in ["A", "B"]
        },
        "by_stratum": {
            s: dict(Counter(x["state"] for x in items if x["stratum"] == s))
            for s in STRATA
        },
        "items": items,
    }


def main():
    inputs = {}
    verified = []

    def bind(path):
        path = Path(path)
        inputs[rel(path)] = sha(path)
        return path

    bind(BASE / "ANALYSIS_PLAN.md")
    for plan_path in [SHORT / "plan.json", OLD / "plan.json"]:
        bind(plan_path)
        for path, expected in read(plan_path)["inputs"].items():
            require(sha(ROOT / path) == expected, "Frozen input changed: " + path)
            inputs[path] = expected
    key_rows = rows(bind(SHORT / "researcher_only/concealed_key.jsonl"))
    old_key = rows(bind(OLD / "researcher_only/concealed_key.jsonl"))
    expected_rows = []
    for s in STRATA:
        candidate = [x for x in old_key if x["stratum"] == s]
        expected_rows += sorted(
            candidate,
            key=lambda x: hashlib.sha256(("202610043|" + x["id"]).encode()).hexdigest(),
        )[:8]
    require(key_rows == expected_rows, "Selected sample changed")
    key = {x["packet_id"]: x for x in key_rows}
    require(
        len(key)
        == len({x["id"] for x in key_rows})
        == len({x["cluster"] for x in key_rows})
        == 24,
        "Repeated ID/cluster",
    )
    require(
        Counter(x["stratum"] for x in key_rows) == Counter({s: 8 for s in STRATA}),
        "Wrong strata",
    )
    roster_path = (
        ROOT / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1/roster.jsonl"
    )
    roster = {x["id"]: x for x in rows(bind(roster_path))}
    for k in key_rows:
        record = roster[k["id"]]
        require(
            record["gold"] == k["gold"]
            and record["cluster"] == k["cluster"]
            and record["population"] == "hateful_types",
            "Wrong roster join",
        )
        model_path = bind(ROOT / k["model_source"])
        require(
            sha(model_path) == k["model_source_sha256"], "Saved model result changed"
        )
        model = read(model_path)
        require(
            model["identity"]["repeat"] == 0
            and model["identity"]["condition"] == "taxonomy::rag_full",
            "Wrong model condition",
        )
        require(model["identity"]["key"] == "en/" + k["id"], "Wrong model input")
        require(
            {n: model["result"][n] for n in k["model_repeat_0"]} == k["model_repeat_0"],
            "Concealed model labels differ",
        )
        g, p = set(k["gold"]["hate_type"]), set(k["model_repeat_0"]["hate_type"])
        require(
            k["stratum"]
            == (
                "exact_agreement"
                if g == p
                else "missing_gold" if g - p else "extra_only"
            ),
            "Wrong sampling stratum",
        )
    verified.append(
        "Frozen sample, unique clusters, original gold and raw repeat-0 full-KB outputs match"
    )
    data = {}
    summaries = {}
    texts = {}
    exports = {}
    full_decoded = {}
    for form, (reviewer, frame, fid) in FORMS.items():
        inspect = read(bind(BASE / (form + "_inspection.json")))
        require(
            inspect["form_id"] == fid
            and inspect["ui_version"] == 5
            and inspect["submissions"] == 1,
            "Missing/duplicate/changed form",
        )
        export_path = bind(Path(inspect["export_path"]))
        actual = read(bind(Path(inspect["form_path"])))
        require(
            actual["id"] == fid
            and actual["status"] == "PUBLISHED"
            and actual["numberOfSubmissions"] == 1,
            "Wrong form state",
        )
        mapping = read(
            bind(SHORT / "platform/blueprints_v5" / (form + "_mapping.json"))
        )
        bind(SHORT / "platform/blueprints_v5" / (form + ".json"))
        export = read(export_path)
        require(
            export["hasMore"] is False and len(export["submissions"]) == 1,
            "Incomplete/duplicate export",
        )
        decoded = decoder.decode(export, mapping, fid)
        require(
            len(decoded) == 1 and not decoded[0]["synthetic_only"],
            "Expected one human return",
        )
        result = decoded[0]
        require(result["framework"] == frame, "Wrong framework")
        form_def = read(bind(SHORT / "forms" / (form + ".json")))
        require(
            form_def["framework"] == frame and form_def["reviewer"] == reviewer,
            "Wrong assignment",
        )
        positions = {block["uuid"]: n for n, block in enumerate(actual["blocks"])}
        live_order = sorted(
            mapping["texts"], key=lambda pid: positions[mapping["texts"][pid]]
        )
        require(
            live_order == [x["packet_id"] for x in form_def["cards"]],
            "Live display order mismatch",
        )

        decoded_by_id = {card["id"]: card for card in result["cards"]}
        result["cards"] = [decoded_by_id[pid] for pid in live_order]
        require({x["id"] for x in result["cards"]} == set(key), "Wrong aliases")
        for card in form_def["cards"]:
            pid = card["packet_id"]
            require(
                card["text"] == roster[key[pid]["id"]]["text"],
                "Displayed text differs from original input",
            )
            require(
                pid not in texts or texts[pid] == card["text"],
                "Different text across forms",
            )
            texts[pid] = card["text"]
        data[reviewer, frame] = {c["id"]: c for c in result["cards"]}
        summaries[form] = form_summary(form, result, export)
        full_decoded[form] = result
        exports[form] = {
            "original_export": rel(export_path),
            "sha256": sha(export_path),
        }
    require(
        len({s["submission_id"] for s in summaries.values()}) == 4,
        "Duplicate submission identity",
    )
    verified.append(
        "Exactly one complete real response per form, matching all 96 cards and option mappings"
    )
    profile = {}
    for reviewer in ["A", "B"]:
        first, second = (
            summaries["Reviewer_" + reviewer + "_Part_1"],
            summaries["Reviewer_" + reviewer + "_Part_2"],
        )
        require(
            first["respondent_id"] == second["respondent_id"],
            "Reviewer parts have different respondent IDs; inspect",
        )
        require(
            datetime.fromisoformat(first["submitted_at"].replace("Z", "+00:00"))
            < datetime.fromisoformat(second["submitted_at"].replace("Z", "+00:00")),
            "Submission order differs",
        )
        profile[reviewer] = {
            "reported_minutes": [
                first["metadata"].get("minutes"),
                second["metadata"].get("minutes"),
            ],
            "reported_total_minutes": first["metadata"]["minutes"]
            + second["metadata"]["minutes"],
            "familiarity": first["metadata"].get("english"),
            "experience": first["metadata"].get("experience"),
            "prior_exposure": first["metadata"].get("exposure"),
            "discussion_or_AI_search": [
                first["metadata"].get("independence"),
                second["metadata"].get("independence"),
            ],
            "submission_order_matches_assigned_parts": True,
        }
    require(
        summaries["Reviewer_A_Part_1"]["respondent_id"]
        != summaries["Reviewer_B_Part_1"]["respondent_id"],
        "A/B respondent IDs identical; inspect",
    )
    combined = []
    for pid in sorted(key):
        combined.append(
            {
                **key[pid],
                "text": texts[pid],
                "judgments": {
                    reviewer + "_" + frame: data[reviewer, frame][pid]
                    for reviewer in ["A", "B"]
                    for frame in ["source", "project"]
                },
            }
        )
    comparison = {
        frame: compare_reviewers(data["A", frame], data["B", frame], frame)
        for frame in ["source", "project"]
    }
    guide_comparison = {
        reviewer: compare_guides(data[reviewer, "source"], data[reviewer, "project"])
        for reviewer in ["A", "B"]
    }
    aligns = {
        r
        + "_"
        + f: {ref: alignment(data[r, f], key, ref) for ref in ["dataset", "model"]}
        for r in ["A", "B"]
        for f in ["source", "project"]
    }
    extra = {
        which: occurrence_review(key, data["A", "project"], data["B", "project"], which)
        for which in ["model_extra", "model_omitted", "model_and_gold"]
    }
    strata = {}
    for s in STRATA:
        ids = [pid for pid, k in key.items() if k["stratum"] == s]
        strata[s] = {
            "n": len(ids),
            "ids": ids,
            "forms": {
                r + "_" + f: dict(Counter(binary_name(data[r, f][i]) for i in ids))
                for r in ["A", "B"]
                for f in ["source", "project"]
            },
        }
    common = [
        i
        for i in key
        if all(
            data[r, f][i]["binary_hate"] is not None
            for r in ["A", "B"]
            for f in ["source", "project"]
        )
    ]
    common_positive = [
        i
        for i in key
        if all(
            data[r, f][i]["binary_hate"] is True
            for r in ["A", "B"]
            for f in ["source", "project"]
        )
    ]
    result = {
        "date": "2026-10-10",
        "status": "returns_audited_descriptive_analysis_independence_clarification_pending",
        "sample": {
            "texts": 24,
            "unique_clusters": 24,
            "judgments": 96,
            "reviewers": 2,
            "strata": dict(Counter(x["stratum"] for x in key_rows)),
            "dataset_hateful": sum(x["gold"]["hate"] for x in key_rows),
            "model_hateful": sum(x["model_repeat_0"]["hate"] for x in key_rows),
            "gold_type_counts": dict(
                Counter(t for x in key_rows for t in x["gold"]["hate_type"])
            ),
            "model_type_counts": dict(
                Counter(t for x in key_rows for t in x["model_repeat_0"]["hate_type"])
            ),
        },
        "forms": summaries,
        "reviewer_profiles": profile,
        "between_reviewers": comparison,
        "within_reviewers": guide_comparison,
        "reference_alignment": aligns,
        "model_occurrence_review_project_guide": extra,
        "strata": strata,
        "common_resolved_all_four": sorted(common),
        "common_hateful_all_four": sorted(common_positive),
        "items": combined,
        "new_model_calls": 0,
        "original_scores_changed": False,
        "qualifications": [
            "Outcome-stratified selected sample, not population accuracy.",
            "Source explicit subtype fields are inapplicable, not negative subtype labels.",
            "Two ordered passes can have carryover; no causal guide claim.",
            "Reviewer B Part 1 reports discussion or AI/search; clarification pending.",
            "All timing/background/conduct fields are self-reports.",
            "Guide toggles are final saved states, not interaction history.",
        ],
        "verified": verified,
    }
    bind(ROOT / "analysis/feedback/human/agreements.py")
    bind(ROOT / "analysis/feedback/human/response_audit.py")
    save(BASE / "decoded_returns.json", full_decoded)
    save(BASE / "result_summary.json", result)
    save(BASE / "input_manifest.json", {"inputs": inputs, "exports": exports})
    print(
        json.dumps(
            {
                "forms": {
                    k: {
                        "gates": v["gates"],
                        "multi": v["multiple_types"],
                        "explicit_plus_other": v["explicit_plus_other"],
                    }
                    for k, v in summaries.items()
                },
                "between_reviewers": comparison,
                "within_reviewers": guide_comparison,
                "extra_counts": {
                    k: {
                        z: v[z]
                        for z in ["occurrences", "texts", "counts", "per_reviewer"]
                    }
                    for k, v in extra.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
