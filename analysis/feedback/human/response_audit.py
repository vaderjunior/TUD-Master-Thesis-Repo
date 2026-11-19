import json
from pathlib import Path
from analysis.feedback.human.forms import guide as b

QA = b.BASE / "platform/deep_qa_v1"


def decode(data, mapping, expected_form_id, allow_synthetic=False):
    assert data["hasMore"] is False, "Incomplete export: fetch every page first"
    bygroup = {f["group_uuid"]: key for key, f in mapping["fields"].items()}
    assert len(bygroup) == len(mapping["fields"])
    qmap = {}
    for q in data["questions"]:
        assert q["formId"] == expected_form_id and not q["isDeleted"]
        refs = {f.get("blockGroupUuid", f["uuid"]) for f in q["fields"]}
        assert (
            len(refs) == 1 and next(iter(refs)) in bygroup
        ), "Unexpected question identity"
        key = bygroup[next(iter(refs))]
        assert q["title"] == mapping["fields"][key]["title"] and q["id"] not in qmap
        qmap[q["id"]] = key
    assert len(qmap) == len(mapping["fields"]) and len(set(qmap.values())) == len(qmap)
    output = []
    seen = set()
    for sub in data["submissions"]:
        assert sub["formId"] == expected_form_id and sub["isCompleted"] is True
        assert sub["id"] not in seen
        seen.add(sub["id"])
        raw = {}
        present = set()
        for response in sub["responses"]:
            assert (
                response["formId"] == expected_form_id
                and response["submissionId"] == sub["id"]
            )
            assert response["questionId"] in qmap
            key = qmap[response["questionId"]]
            assert key not in present
            present.add(key)
            field = mapping["fields"][key]
            answer = response["answer"]
            if "option_uuids" in field:
                assert answer is None or isinstance(
                    answer, list
                ), "Unexpected choice encoding"
                answer = answer or []
                options = {uid: label for label, uid in field["option_uuids"].items()}
                assert len(set(answer)) == len(answer) and all(
                    uid in options for uid in answer
                )
                if field["type"] == "MULTIPLE_CHOICE_OPTION":
                    assert len(answer) <= 1
                raw[key] = [options[uid] for uid in answer]
            else:
                raw[key] = answer
        synthetic = "ASTRA_QA_ONLY_" in str(raw.get("notes", ""))
        assert (
            allow_synthetic or not synthetic
        ), "Synthetic QA response cannot enter human analysis"
        cards = []
        for pid in mapping["texts"]:
            value = raw.get(pid + "/gate", [])
            assert len(value) == 1, "Missing required classification"
            gate = value[0]
            flags = []
            framework = mapping["framework"]
            applicable = (
                (gate == "Implicit hate") if framework == "source" else (gate == "Yes")
            )
            if framework == "source":
                primary = raw.get(pid + "/primary", [])
                secondary = raw.get(pid + "/secondary", [])
                types = primary + [x for x in secondary if x != "No secondary type"]
                if primary and secondary == primary:
                    flags.append("duplicate_primary_secondary")
                missing_type = not primary
            else:
                types = raw.get(pid + "/types", [])
                primary = []
                secondary = []
                missing_type = not types
            if not applicable and types:
                flags.append("ignored_nonapplicable_types")
            if applicable and missing_type:
                flags.append("unassigned_applicable_type")
            cards.append(
                {
                    "id": pid,
                    "gate": gate,
                    "binary_hate": (
                        True
                        if gate in {"Yes", "Explicit hate", "Implicit hate"}
                        else False if gate in {"No", "Not hate"} else None
                    ),
                    "types": sorted(set(types)) if applicable else [],
                    "type_state": (
                        ("unassigned" if missing_type else "assigned")
                        if applicable
                        else "not_applicable"
                    ),
                    "primary": primary if applicable else [],
                    "secondary": secondary if applicable else [],
                    "difficulty": raw.get(pid + "/flags", []),
                    "guide_open": bool(raw.get(pid + "/guide_help", [])),
                    "review_flags": flags,
                }
            )
        assert len(cards) == 24
        output.append(
            {
                "submission_id": sub["id"],
                "form_id": expected_form_id,
                "synthetic_only": synthetic,
                "form_key": mapping["form_key"],
                "framework": mapping["framework"],
                "raw_fields": raw,
                "answered_field_keys": sorted(present),
                "cards": cards,
            }
        )
    return output


def fixture_export(role):
    matches = []
    for file in QA.glob("*_" + role + "_get/response.json"):
        value = b.read(file)
        if "submissions" in value:
            matches.append((file, value))
    assert matches
    return max(matches, key=lambda x: x[0].parent.name)


def audit_fixture(role):
    file, data = fixture_export(role)
    mapping = b.read(QA / (role + "_mapping.json"))
    fixture = b.read(QA / (role + "_fixture.json"))
    expected = b.read(QA / (role + "_ui_expectations.json"))
    decoded = decode(data, mapping, fixture["form_id"], allow_synthetic=True)
    assert len(decoded) == 1 and decoded[0]["synthetic_only"]
    item = decoded[0]
    cards = {x["id"]: x for x in item["cards"]}
    comparisons = []
    for exp in expected["checks"]:
        got = cards[exp["id"]]
        assert got["gate"] == exp["gate"]
        assert (
            sorted(got["difficulty"]) == sorted(exp.get("flags", []))
            and got["guide_open"] == exp["guide"]
        )
        if role == "source":
            assert got["primary"] == ([exp["primary"]] if exp.get("primary") else [])
            assert got["secondary"] == (
                [exp["secondary"]] if exp.get("secondary") else []
            )
            types = [
                exp[x]
                for x in ["primary", "secondary"]
                if exp.get(x) and exp[x] != "No secondary type"
            ]
        else:
            types = exp.get("types", [])
        assert got["types"] == sorted(set(types))
        comparisons.append(
            {"id": exp["id"], "matched": True, "review_flags": got["review_flags"]}
        )
    assert len(comparisons) == 24 and expected["marker"] in item["raw_fields"]["notes"]
    assert str(item["raw_fields"]["minutes"]) in {"0", "0.0"} and item["raw_fields"][
        "independence"
    ] == ["Yes"]
    assert len(expected["recovery"]) == 8 and all(
        x["saved"] for x in expected["recovery"]
    )
    b.save(QA / (role + "_decoded_qa.json"), item)
    result = {
        "role": role,
        "form_id": fixture["form_id"],
        "synthetic_only": True,
        "cards_matched": 24,
        "reopened_saved_cards_verified": 8,
        "export": str(file),
        "export_sha256": b.sha(file),
        "ui_expectations_sha256": b.sha(QA / (role + "_ui_expectations.json")),
        "checks": comparisons,
    }
    b.save(QA / (role + "_roundtrip_verified.json"), result)
    return result


if __name__ == "__main__":
    results = [audit_fixture(role) for role in ["source", "project"]]
    print(
        json.dumps(
            {
                "scripted_cards_exported_and_matched": sum(
                    r["cards_matched"] for r in results
                ),
                "saved_cards_recovered": sum(
                    r["reopened_saved_cards_verified"] for r in results
                ),
                "independent_human_judgments": 0,
            }
        )
    )
