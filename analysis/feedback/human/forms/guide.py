import copy
import html
import json
import uuid
from analysis.feedback.human.forms import base as original
from analysis.feedback.human.forms import submit as prior

BASE = prior.BASE
OUT = BASE / "platform/blueprints_v4"
read, save, sha, plain = prior.read, prior.save, prior.sha, prior.plain


def old_uid(key, name):
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "astra/short-human-review/20261006/tally-v1/" + key + "/" + name,
        )
    )


def build(key):
    form = read(BASE / "forms" / (key + ".json"))
    payload = copy.deepcopy(read(prior.OUT / (key + ".json")))
    mapping = copy.deepcopy(read(prior.OUT / (key + "_mapping.json")))
    helpers = {}
    out = []
    for card in form["cards"]:
        pid = card["packet_id"]
        helper = original.Builder(key)
        field = helper.choice(
            pid + "/guide_help",
            pid + " · Guide (optional)",
            ["Show this part's guide"],
            multiple=True,
        )
        body = helper.text(pid + "/guide_body", form["guide"], hidden=True)
        helper.block(
            pid + "/guide_logic",
            "CONDITIONAL_LOGIC",
            {
                "logicalOperator": "AND",
                "conditionals": [
                    {
                        "uuid": helper.uid(pid + "/guide_condition"),
                        "type": "SINGLE",
                        "payload": {
                            "field": {
                                "uuid": field["group_uuid"],
                                "blockGroupUuid": field["group_uuid"],
                                "type": "INPUT_FIELD",
                                "questionType": "CHECKBOX",
                                "title": field["title"],
                            },
                            "comparison": "IS_NOT_EMPTY",
                            "value": "",
                        },
                    }
                ],
                "actions": [
                    {
                        "uuid": helper.uid(pid + "/guide_action"),
                        "type": "SHOW_BLOCKS",
                        "payload": {"showBlocks": [body["groupUuid"]]},
                    }
                ],
            },
        )
        helpers[mapping["texts"][pid]] = helper.blocks
        mapping["fields"][pid + "/guide_help"] = field
        mapping.setdefault("help_texts", {})[pid] = body["uuid"]
    for block in payload["blocks"]:
        if block["uuid"] == old_uid(key, "introduction"):
            text = plain(block["payload"]["html"])
            if form["part"] == 2:
                text = text.replace(
                    "Finish this entire part before opening the next. ", ""
                )
            text += "\n\nIf you take a break, return using the same browser and device. Answers reach Ajay only after you press Submit and see the confirmation. The optional guide on each card repeats this part’s instructions."
            block["payload"]["html"] = html.escape(text).replace("\n", "<br>")
        if block["uuid"] == old_uid(key, "guide-reminder"):
            text = (
                plain(block["payload"]["html"])
                + " Click a selected optional answer again to clear it."
            )
            block["payload"]["html"] = html.escape(text)
        if block["uuid"] == old_uid(key, "finish-note"):
            text = (
                "Press Submit and wait for the confirmation. Then open your Part 2 link."
                if form["part"] == 1
                else "Press Submit and wait for the confirmation. This completes your review; there is no third part."
            )
            block["payload"]["html"] = html.escape(
                text + " Thank you. No names or contact details are requested."
            )
        out.append(block)
        out.extend(helpers.get(block["uuid"], []))
    payload["blocks"] = out
    mapping["ui_version"] = 4
    mapping["interpretation"][
        "guide_help"
    ] = "usability-only field; never a hate/type label"
    return form, payload, mapping


def check(form, payload, mapping):
    from jsonschema import Draft7Validator, FormatChecker

    spec = read(original.SCHEMA)
    schema = spec["paths"]["/forms/{formId}"]["patch"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    errs = list(
        Draft7Validator(
            {**schema, "components": spec["components"]}, format_checker=FormatChecker()
        ).iter_errors(payload)
    )
    assert not errs, [(list(e.path), e.message[:200]) for e in errs[:3]]
    key = mapping["form_key"]
    before = read(prior.OUT / (key + ".json"))
    old_mapping = read(prior.OUT / (key + "_mapping.json"))
    byid = {z["uuid"]: z for z in payload["blocks"]}
    assert len(byid) == len(payload["blocks"])
    allowed_text_changes = {
        old_uid(key, x) for x in ["introduction", "guide-reminder", "finish-note"]
    }
    for z in before["blocks"]:
        if z["uuid"] not in allowed_text_changes:
            assert byid[z["uuid"]] == z
    assert [
        z["uuid"]
        for z in payload["blocks"]
        if z["uuid"] in {x["uuid"] for x in before["blocks"]}
    ] == [z["uuid"] for z in before["blocks"]]
    for k, v in old_mapping["fields"].items():
        assert mapping["fields"][k] == v
    pages = [[]]
    for z in payload["blocks"]:
        if z["type"] == "PAGE_BREAK":
            pages.append([])
        else:
            pages[-1].append(z)
    assert len(pages) == 26
    for n, card in enumerate(form["cards"], 1):
        pid = card["packet_id"]
        post = byid[mapping["texts"][pid]]
        guide = byid[mapping["help_texts"][pid]]
        assert post in pages[n] and guide in pages[n]
        assert plain(post["payload"]["html"]) == card["text"]
        assert (
            plain(guide["payload"]["html"]) == form["guide"]
            and guide["payload"]["isHidden"] is True
        )
        h = mapping["fields"][pid + "/guide_help"]
        assert not h["required"] and not h["hidden_initially"]
        rules = [r for r in pages[n] if r["type"] == "CONDITIONAL_LOGIC"]
        assert len(rules) == 2
        help_rule = next(
            r
            for r in rules
            if r["payload"]["conditionals"][0]["payload"]["field"]["uuid"]
            == h["group_uuid"]
        )
        assert help_rule["payload"]["actions"][0]["payload"]["showBlocks"] == [
            guide["groupUuid"]
        ]
    assert all(
        f["required"] == name.endswith("/gate") for name, f in mapping["fields"].items()
    )
    return {
        "cards": 24,
        "pages": 26,
        "exact_posts": 24,
        "exact_on_card_guides": 24,
        "unchanged_scoring_fields": len(old_mapping["fields"]),
        "no_defaults": all(
            not z["payload"].get("hasDefaultAnswer", False) for z in payload["blocks"]
        ),
        "ui_version": 4,
        "independent_human_judgments": 0,
    }


def main():
    result = {}
    for p in sorted((BASE / "forms").glob("Reviewer_*.json")):
        form, payload, mapping = build(p.stem)
        checks = check(form, payload, mapping)
        save(OUT / (p.stem + ".json"), payload)
        save(OUT / (p.stem + "_mapping.json"), mapping)
        result[p.stem] = {
            **checks,
            "payload_sha256": sha(OUT / (p.stem + ".json")),
            "mapping_sha256": sha(OUT / (p.stem + "_mapping.json")),
        }
    save(
        OUT / "verification.json",
        {
            "forms": result,
            "source_sha256": sha(
                original.ROOT / "analysis/feedback/human/forms/guide.py"
            ),
            "timing": "pending",
            "live_guide_toggle": "pending",
            "collection_authorized": False,
        },
    )
    print(
        json.dumps(
            {"forms": 4, "checked": True, "ui_version": 4, "unchanged_posts": 96}
        )
    )


if __name__ == "__main__":
    main()
