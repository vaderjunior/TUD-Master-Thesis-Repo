from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_human_review_results_20261010_v1"
FORMS = {
    "Reviewer_A_Part_1": ("A", "source"),
    "Reviewer_A_Part_2": ("A", "project"),
    "Reviewer_B_Part_1": ("B", "project"),
    "Reviewer_B_Part_2": ("B", "source"),
}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


CHECKS = []


def check(ok, description):
    if not ok:
        raise AssertionError(description)
    CHECKS.append(description)


def cat(pairs, published, name):
    count = len(pairs)
    equal = len([p for p in pairs if p[0] == p[1]])
    check(published["n"] == count and published["agree"] == equal, name + "/n/equal")
    check(
        Counter(
            (x["left"], x["right"]) for x in published["cells"] for _ in range(x["n"])
        )
        == Counter(pairs),
        name + "/cells",
    )
    if count == 0:
        check(published["kappa"] is None and published["rate"] is None, name + "/empty")
        return
    lm, rm = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    products = sum(lm[k] * rm[k] for k in set(lm) | set(rm))
    check(
        math.isclose(published["rate"], float(Fraction(equal, count)), abs_tol=1e-12),
        name + "/rate",
    )
    check(
        math.isclose(
            published["expected_agreement"],
            float(Fraction(products, count * count)),
            abs_tol=1e-12,
        ),
        name + "/pe",
    )
    kappa = (
        Fraction(count * equal - products, count * count - products)
        if products != count * count
        else None
    )
    check(
        (
            published["kappa"] is None
            if kappa is None
            else math.isclose(published["kappa"], float(kappa), abs_tol=1e-12)
        ),
        name + "/kappa",
    )


def type_sets(pairs, published, name):
    check(published["n"] == len(pairs), name + "/n")
    check(published["exact"] == sum(a == b for a, b in pairs), name + "/exact")
    if pairs:
        j = sum(
            (Fraction(len(a & b), len(a | b)) if a | b else Fraction(1))
            for a, b in pairs
        ) / len(pairs)
        check(
            math.isclose(float(j), published["mean_jaccard"], abs_tol=1e-12),
            name + "/jaccard",
        )
    for label, value in published["per_label"].items():
        cat([(label in a, label in b) for a, b in pairs], value, name + "/" + label)
        check(
            value["left_positive"] == sum(label in a for a, _ in pairs),
            name + "/" + label + "/left",
        )
        check(
            value["right_positive"] == sum(label in b for _, b in pairs),
            name + "/" + label + "/right",
        )


def main():
    summary = read(BASE / "result_summary.json")
    primary = read(BASE / "decoded_returns.json")
    manifest = read(BASE / "input_manifest.json")
    for path, wanted in manifest["inputs"].items():
        check(sha(ROOT / path) == wanted, "hash/" + path)
    data = {}
    code = {}
    expanded = {}
    for form, (reviewer, frame) in FORMS.items():
        inspect = read(BASE / (form + "_inspection.json"))
        live, export = read(inspect["form_path"]), read(inspect["export_path"])
        fields = {x["uuid"]: x for x in live["blocks"]}
        questions = {x["id"]: x for x in export["questions"]}
        check(len(questions) == len(export["questions"]), form + "/unique questions")
        sub = export["submissions"][0]
        values = {}
        for response in sub["responses"]:
            q = questions[response["questionId"]]
            check(q["title"] not in values, form + "/unique answer/" + q["title"])
            answer = response["answer"]
            if isinstance(answer, list):
                answer = [fields[uid]["payload"]["text"] for uid in answer]
            values[q["title"]] = answer
        wanted = {x["id"]: x for x in primary[form]["cards"]}
        check(len(wanted) == 24, form + "/24")
        data[reviewer, frame] = {}
        code[reviewer, frame] = {}
        for pid, expected in wanted.items():
            get = lambda suffix: values.get(pid + " · " + suffix, []) or []
            gate_list = get(
                "Classification" if frame == "source" else "Is this hateful?"
            )
            check(len(gate_list) == 1, form + "/" + pid + "/single gate")
            gate = gate_list[0]
            hate = (
                True
                if gate in ["Explicit hate", "Implicit hate", "Yes"]
                else False if gate in ["Not hate", "No"] else None
            )
            applicable = gate == ("Implicit hate" if frame == "source" else "Yes")
            if frame == "source":
                first = get("Primary implicit type (if applicable)")
                second = get(
                    "Secondary implicit type (optional; different from primary)"
                )
                types = set(first + [x for x in second if x != "No secondary type"])
                missing = not first
            else:
                first, second = [], []
                types = set(get("Hate types (select all justified types)"))
                missing = not types
            flags = []
            if first and second == first:
                flags.append("duplicate_primary_secondary")
            if not applicable and types:
                flags.append("ignored_nonapplicable_types")
            if applicable and missing:
                flags.append("unassigned_applicable_type")
            rebuilt = {
                "id": pid,
                "gate": gate,
                "binary_hate": hate,
                "types": sorted(types) if applicable else [],
                "type_state": (
                    ("unassigned" if missing else "assigned")
                    if applicable
                    else "not_applicable"
                ),
                "primary": first if applicable else [],
                "secondary": second if applicable else [],
                "difficulty": get("Optional difficulty flags"),
                "guide_open": bool(get("Guide (optional)")),
                "review_flags": flags,
            }
            check(
                rebuilt == expected,
                form + "/" + pid + "/all independently decoded fields",
            )
            data[reviewer, frame][pid] = rebuilt
            code[reviewer, frame][pid] = (
                None
                if hate is None
                else (
                    set()
                    if hate is False
                    else (
                        {"explicit"}
                        if gate == "Explicit hate"
                        else types if not missing else None
                    )
                )
            )
        expanded[form] = {
            "values_by_question_title": values,
            "cards": data[reviewer, frame],
        }
        check(
            dict(Counter(c["gate"] for c in data[reviewer, frame].values()))
            == summary["forms"][form]["gates"],
            form + "/coarse counts",
        )
        check(
            sum(len(c["types"]) > 1 for c in data[reviewer, frame].values())
            == summary["forms"][form]["multiple_types"],
            form + "/multiple",
        )
    bn = lambda c: (
        ("Hate" if c["binary_hate"] else "Not hate")
        if c["binary_hate"] is not None
        else c["gate"]
    )
    ids = sorted(data["A", "source"])
    for frame in ["source", "project"]:
        aa, bb, pub = (
            data["A", frame],
            data["B", frame],
            summary["between_reviewers"][frame],
        )
        cat(
            [(aa[i]["gate"], bb[i]["gate"]) for i in ids],
            pub["coarse_including_uncertain_skip"],
            frame + "/coarse",
        )
        cat(
            [(bn(aa[i]), bn(bb[i])) for i in ids],
            pub["binary_including_uncertain_skip"],
            frame + "/binary all",
        )
        valid = [
            i
            for i in ids
            if aa[i]["binary_hate"] is not None and bb[i]["binary_hate"] is not None
        ]
        cat(
            [(aa[i]["binary_hate"], bb[i]["binary_hate"]) for i in valid],
            pub["binary_both_resolved"],
            frame + "/binary resolved",
        )
        check(valid == pub["binary_both_resolved"]["ids"], frame + "/resolved cohort")
        fine = [
            i for i in ids if aa[i]["type_state"] == bb[i]["type_state"] == "assigned"
        ]
        check(
            fine == pub["type_sets_both_applicable_assigned"]["ids"],
            frame + "/type cohort",
        )
        type_sets(
            [(set(aa[i]["types"]), set(bb[i]["types"])) for i in fine],
            pub["type_sets_both_applicable_assigned"],
            frame + "/types",
        )
        if frame == "source":
            cat(
                [(aa[i]["primary"][0], bb[i]["primary"][0]) for i in fine],
                pub["source_primary_both_implicit"],
                "source primary",
            )
    for reviewer in ["A", "B"]:
        ss, pp, pub = (
            data[reviewer, "source"],
            data[reviewer, "project"],
            summary["within_reviewers"][reviewer],
        )
        cat(
            [(bn(ss[i]), bn(pp[i])) for i in ids],
            pub["binary_transitions_all"],
            reviewer + "/guide transitions",
        )
        valid = [
            i
            for i in ids
            if ss[i]["binary_hate"] is not None and pp[i]["binary_hate"] is not None
        ]
        cat(
            [(ss[i]["binary_hate"], pp[i]["binary_hate"]) for i in valid],
            pub["binary_both_resolved"],
            reviewer + "/guide resolved",
        )
        fine = [
            i for i in ids if ss[i]["type_state"] == pp[i]["type_state"] == "assigned"
        ]
        type_sets(
            [(set(ss[i]["types"]), set(pp[i]["types"])) for i in fine],
            pub["hateful_in_both_assigned_implicit_source"],
            reviewer + "/guide types",
        )
    key = {x["packet_id"]: x for x in summary["items"]}
    for reviewer in ["A", "B"]:
        for frame in ["source", "project"]:
            cards = data[reviewer, frame]
            labelsets = code[reviewer, frame]
            pub = summary["reference_alignment"][reviewer + "_" + frame]
            for reference, ref_name in [
                ("dataset", "gold"),
                ("model", "model_repeat_0"),
            ]:
                resolved = [i for i in ids if cards[i]["binary_hate"] is not None]
                cat(
                    [
                        (cards[i]["binary_hate"], key[i][ref_name]["hate"])
                        for i in resolved
                    ],
                    pub[reference]["binary_resolved"],
                    reviewer + frame + reference + "/gate",
                )
                annotated = [i for i in ids if labelsets[i] is not None]
                exact = sum(
                    labelsets[i] == set(key[i][ref_name]["hate_type"])
                    for i in annotated
                )
                check(
                    pub[reference]["complete_response_codes"]["n"] == len(annotated)
                    and pub[reference]["complete_response_codes"]["exact"] == exact,
                    reviewer + frame + reference + "/complete code",
                )
    for name, pub in summary["model_occurrence_review_project_guide"].items():
        independent = []
        for i in ids:
            g, m = set(key[i]["gold"]["hate_type"]), set(
                key[i]["model_repeat_0"]["hate_type"]
            )
            selected = (
                m - g
                if name == "model_extra"
                else g - m if name == "model_omitted" else g & m
            )
            for label in sorted(selected):
                left, right = code["A", "project"][i], code["B", "project"][i]
                av, bv = None if left is None else label in left, (
                    None if right is None else label in right
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
                independent.append(
                    {
                        "id": i,
                        "label": label,
                        "stratum": key[i]["stratum"],
                        "A_selected": av,
                        "B_selected": bv,
                        "state": state,
                    }
                )
        check(
            independent == pub["items"],
            name + "/every independently reconstructed occurrence",
        )
        check(
            Counter(x["state"] for x in independent) == pub["counts"], name + "/counts"
        )
    result = {
        "status": "passed",
        "checks": len(CHECKS),
        "checks_passed": CHECKS,
        "raw_cards_independently_reconstructed": 96,
        "different_decode_basis": "Live option block texts and question titles; no production mapping or decoder imports",
        "arithmetic_basis": "Exact Fraction kappa arithmetic and separately constructed contingencies",
        "source_sha256": sha(__file__),
        "summary_sha256": sha(BASE / "result_summary.json"),
        "human_judgments_added": 0,
        "new_model_calls": 0,
    }
    for path, data_out in [
        (BASE / "independent_decoded_returns.json", expanded),
        (BASE / "independent_check.json", result),
    ]:
        encoded = (
            json.dumps(data_out, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf8")
        if path.exists():
            check(path.read_bytes() == encoded, "Preserve existing output")
        else:
            path.write_bytes(encoded)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "checks",
                    "raw_cards_independently_reconstructed",
                    "human_judgments_added",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
