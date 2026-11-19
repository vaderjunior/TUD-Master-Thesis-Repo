import json
from analysis.feedback.human.forms import base as prior

BASE = prior.BASE
OUT = BASE / "platform/blueprints_v2"
read, save, sha, plain = prior.read, prior.save, prior.sha, prior.plain


def build(key):
    form, payload, mapping = prior.build(key)
    prior.check(form, payload, mapping)
    fields = mapping["fields"]
    for block in payload["blocks"]:
        if block["type"] == "FORM_TITLE":
            block["groupType"] = "FORM_TITLE"
        if block["type"] != "CONDITIONAL_LOGIC":
            continue
        rule = block["payload"]
        condition = rule["conditionals"][0]["payload"]

        condition["field"]["questionType"] = "MULTIPLE_CHOICE_OPTION"
        gate = next(
            f
            for f in fields.values()
            if f["group_uuid"] == condition["field"]["blockGroupUuid"]
        )
        prefix = gate["key"].split("/")[0] + "/"
        targets = [
            f["group_uuid"]
            for f in fields.values()
            if f["key"].startswith(prefix) and f["hidden_initially"]
        ]
        rule["actions"][0]["payload"]["showBlocks"] = targets
    mapping["native_reference_version"] = 2
    return form, payload, mapping


def check(form, payload, mapping):
    from jsonschema import Draft7Validator, FormatChecker

    spec = read(prior.SCHEMA)
    schema = spec["paths"]["/forms/{formId}"]["patch"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    errors = list(
        Draft7Validator(
            {**schema, "components": spec["components"]}, format_checker=FormatChecker()
        ).iter_errors(payload)
    )
    assert not errors, [(list(e.path), e.message[:300]) for e in errors[:3]]
    source, p0, m0 = prior.build(mapping["form_key"])
    result = prior.check(source, p0, m0)
    expected_form, expected_payload, expected_mapping = build(mapping["form_key"])
    assert (
        form == expected_form
        and payload == expected_payload
        and mapping == expected_mapping
    )
    groups = {x["groupUuid"] for x in payload["blocks"]}
    for block in payload["blocks"]:
        if block["type"] != "CONDITIONAL_LOGIC":
            continue
        rule = block["payload"]
        condition = rule["conditionals"][0]["payload"]
        assert condition["field"]["questionType"] == "MULTIPLE_CHOICE_OPTION"
        assert condition["field"]["uuid"] == condition["field"]["blockGroupUuid"]
        assert condition["field"]["uuid"] in groups
        assert all(x in groups for x in rule["actions"][0]["payload"]["showBlocks"])
    return {**result, "native_question_group_references": True, "version": 2}


def main():
    results = {}
    for source in sorted((BASE / "forms").glob("Reviewer_*.json")):
        form, payload, mapping = build(source.stem)
        result = check(form, payload, mapping)
        save(OUT / (source.stem + ".json"), payload)
        save(OUT / (source.stem + "_mapping.json"), mapping)
        results[source.stem] = {
            **result,
            "payload_sha256": sha(OUT / (source.stem + ".json")),
            "mapping_sha256": sha(OUT / (source.stem + "_mapping.json")),
        }
    save(
        OUT / "verification.json",
        {
            "forms": results,
            "source_sha256": sha(
                prior.ROOT / "analysis/feedback/human/forms/groups.py"
            ),
            "prior_source_sha256": sha(
                prior.ROOT / "analysis/feedback/human/forms/base.py"
            ),
            "network_calls": 0,
            "human_judgments": 0,
            "live_UI_validation": "pending",
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
