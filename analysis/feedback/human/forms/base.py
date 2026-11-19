from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import uuid

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_human_review_short_20261006_v1"
OUT = BASE / "platform/blueprints_v1"
SCHEMA = BASE / "platform/api_docs/openapi.json"
TYPES = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
IMPLICIT = [
    "grievance",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "other",
]


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    data = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf8")
    if p.exists():
        assert p.read_bytes() == data, f"Preserve existing artifact: {p}"
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(data)


class Plain(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def plain(value):
    p = Plain()
    p.feed(value)
    return "".join(p.parts)


class Builder:
    def __init__(self, key):
        self.key = key
        self.blocks = []
        self.fields = {}
        self.texts = {}

    def uid(self, key):
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "astra/short-human-review/20261006/tally-v1/" + self.key + "/" + key,
            )
        )

    def block(self, key, kind, payload, group=None, group_type=None):
        b = {
            "uuid": self.uid(key),
            "type": kind,
            "groupUuid": group or self.uid(key),
            "groupType": group_type or kind,
            "payload": payload,
        }
        self.blocks.append(b)
        return b

    def text(self, key, value, kind="TEXT", hidden=False):
        return self.block(
            key,
            kind,
            {"html": html.escape(value).replace("\n", "<br>"), "isHidden": hidden},
        )

    def choice(
        self, key, title, labels, *, multiple=False, required=False, hidden=False
    ):
        title_block = self.block(
            key + "/title",
            "TITLE",
            {"html": html.escape(title), "isHidden": hidden},
            group_type="QUESTION",
        )
        group = self.uid(key + "/group")
        options = []
        for n, label in enumerate(labels):
            p = {
                "text": label,
                "index": n,
                "isFirst": n == 0,
                "isLast": n == len(labels) - 1,
                "isRequired": required,
                "isHidden": hidden,
                "hasDefaultAnswer": False,
                "randomize": False,
                "hasOtherOption": False,
                "isOtherOption": False,
                "hasMinChoices": False,
                "hasMaxChoices": False,
            }
            if not multiple:
                p["allowMultiple"] = False
            options.append(
                self.block(
                    key + "/option/" + str(n),
                    "CHECKBOX" if multiple else "MULTIPLE_CHOICE_OPTION",
                    p,
                    group=group,
                    group_type="CHECKBOXES" if multiple else "MULTIPLE_CHOICE",
                )
            )
        field = {
            "key": key,
            "title": title,
            "title_uuid": title_block["uuid"],
            "group_uuid": group,
            "type": options[0]["type"],
            "required": required,
            "hidden_initially": hidden,
            "option_uuids": {str(o["payload"]["text"]): o["uuid"] for o in options},
            "block_uuids": [title_block["uuid"], *[o["uuid"] for o in options]],
        }
        self.fields[key] = field
        return field

    def input(self, key, title, kind):
        t = self.block(
            key + "/title", "TITLE", {"html": html.escape(title)}, group_type="QUESTION"
        )
        i = self.block(
            key + "/input",
            kind,
            {"isRequired": False, "hasDefaultAnswer": False, "placeholder": ""},
        )
        self.fields[key] = {
            "key": key,
            "title": title,
            "title_uuid": t["uuid"],
            "group_uuid": i["groupUuid"],
            "type": kind,
            "required": False,
            "hidden_initially": False,
            "block_uuids": [t["uuid"], i["uuid"]],
        }

    def page(self, index, name):
        return self.block(
            "page/" + str(index),
            "PAGE_BREAK",
            {
                "index": index,
                "isFirst": index == 0,
                "isLast": index == 24,
                "name": name,
                "isThankYouPage": False,
                "button": {"label": "Next"},
            },
        )

    def show_if(self, key, field, answer, targets):
        self.block(
            key,
            "CONDITIONAL_LOGIC",
            {
                "logicalOperator": "AND",
                "conditionals": [
                    {
                        "uuid": self.uid(key + "/condition"),
                        "type": "SINGLE",
                        "payload": {
                            "field": {
                                "uuid": field["group_uuid"],
                                "blockGroupUuid": field["group_uuid"],
                                "type": "INPUT_FIELD",
                                "questionType": field["type"],
                                "title": field["title"],
                            },
                            "comparison": "IS",
                            "value": field["option_uuids"][answer],
                        },
                    }
                ],
                "actions": [
                    {
                        "uuid": self.uid(key + "/action"),
                        "type": "SHOW_BLOCKS",
                        "payload": {
                            "showBlocks": [
                                uid for f in targets for uid in f["block_uuids"]
                            ]
                        },
                    }
                ],
            },
        )


def build(key):
    form = read(BASE / "forms" / f"{key}.json")
    b = Builder(key)
    b.block(
        "form-title",
        "FORM_TITLE",
        {"title": form["title"], "html": html.escape(form["title"])},
        group_type="TEXT",
    )
    md = (BASE / "imports" / f"{key}.md").read_text(encoding="utf8")
    introduction = md.split("\n", 1)[1].split("## Guide for this part", 1)[0].strip()
    b.text("introduction", introduction)
    b.text("guide-title", "Guide for this part", "HEADING_2")
    b.text("guide", form["guide"])
    b.text(
        "guide-reminder",
        "Please do not assume a text is hateful. Unsure and Skip are valid answers. Select only justified types; an empty type answer will remain unassigned. You can use the Back button to revise an answer.",
    )
    if form["part"] == 1:
        b.text("details-heading", "A few quick details", "HEADING_2")
        b.choice(
            "english",
            "English familiarity",
            ["Very comfortable", "Comfortable", "Sometimes need help"],
        )
        b.choice(
            "experience",
            "Previous hate-speech annotation experience",
            ["None", "Some", "Substantial"],
        )
        b.choice(
            "exposure",
            "Have you already seen these texts or the model results?",
            ["No", "Yes", "Unsure"],
        )
    for n, card in enumerate(form["cards"], 1):
        pid = card["packet_id"]
        b.page(n - 1, f"Text {n}/24")
        b.text(pid + "/heading", f"Text {n}/24 · {pid}", "HEADING_2")
        b.text(pid + "/read", "Read this text exactly as written:")
        t = b.block(
            pid + "/text",
            "TEXT",
            {
                "html": '<span style="white-space: pre-wrap;">'
                + html.escape(card["text"])
                + "</span>"
            },
        )
        b.texts[pid] = t["uuid"]
        if form["framework"] == "source":
            gate = b.choice(
                pid + "/gate",
                pid + " · Classification",
                [
                    "Explicit hate",
                    "Implicit hate",
                    "Not hate",
                    "Unsure",
                    "Skip this text",
                ],
                required=True,
            )
            primary = b.choice(
                pid + "/primary",
                pid + " · Primary implicit type (if applicable)",
                IMPLICIT,
                hidden=True,
            )
            secondary = b.choice(
                pid + "/secondary",
                pid + " · Secondary implicit type (optional; different from primary)",
                [*IMPLICIT, "No secondary type"],
                hidden=True,
            )
            b.show_if(pid + "/logic", gate, "Implicit hate", [primary, secondary])
        else:
            gate = b.choice(
                pid + "/gate",
                pid + " · Is this hateful?",
                ["Yes", "No", "Unsure", "Skip this text"],
                required=True,
            )
            types = b.choice(
                pid + "/types",
                pid + " · Hate types (select all justified types)",
                TYPES,
                multiple=True,
                hidden=True,
            )
            b.show_if(pid + "/logic", gate, "Yes", [types])
        b.choice(
            pid + "/flags",
            pid + " · Optional difficulty flags",
            ["Missing context", "Meaning unclear"],
            multiple=True,
        )
    b.page(24, "Finish this part")
    b.text("finish", "Finish this part", "HEADING_2")
    b.input(
        "minutes",
        "Approximately how many minutes did this part take, including reading its guide?",
        "INPUT_NUMBER",
    )
    b.choice(
        "independence",
        "Did you discuss answers or use AI/search during this part?",
        ["No", "Yes"],
    )
    b.input(
        "notes",
        "Any optional notes? You may refer to a text using its Rxxx number.",
        "TEXTAREA",
    )
    b.text(
        "finish-note",
        "Thank you. Submit this part before opening the next link. Your answers are saved to Ajay’s Tally account when you submit. No names or contact details are requested.",
    )
    result = {
        "name": form["title"],
        "status": "DRAFT",
        "blocks": b.blocks,
        "settings": {
            "language": "en",
            "hasProgressBar": True,
            "saveForLater": True,
            "pageAutoJump": False,
            "hasPartialSubmissions": False,
            "hasSelfEmailNotifications": False,
            "hasRespondentEmailNotifications": False,
        },
    }
    mapping = {
        "form_key": key,
        "source_form_sha256": sha(BASE / "forms" / f"{key}.json"),
        "framework": form["framework"],
        "reviewer": form["reviewer"],
        "part": form["part"],
        "fields": b.fields,
        "texts": b.texts,
        "interpretation": {
            "No secondary type": "no secondary label; not a category",
            "unanswered_type": "unassigned/missing; not inferred from gate",
            "nonapplicable_type": "retain raw answer, but flag/ignore for framework scoring when gate makes type inapplicable",
            "duplicate_primary_secondary": "flag for review; do not silently count twice",
            "unsure_skip_unanswered": "distinct from non-hate",
            "test_interactions": "not human judgments",
        },
    }
    return form, result, mapping


def check(form, payload, mapping):
    from jsonschema import Draft7Validator, FormatChecker

    spec = read(SCHEMA)
    schema = spec["paths"]["/forms/{formId}"]["patch"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    validator = Draft7Validator(
        {**schema, "components": spec["components"]}, format_checker=FormatChecker()
    )
    errors = list(validator.iter_errors(payload))
    assert not errors, [(list(e.path), e.message[:500]) for e in errors[:3]]
    blocks = payload["blocks"]
    by_id = {x["uuid"]: x for x in blocks}
    assert len(by_id) == len(blocks)
    assert payload["status"] == "DRAFT" and len(form["cards"]) == 24
    assert (
        plain(
            by_id[
                str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        "astra/short-human-review/20261006/tally-v1/"
                        + mapping["form_key"]
                        + "/guide",
                    )
                )
            ]["payload"]["html"]
        )
        == form["guide"]
    )
    assert [x["payload"]["index"] for x in blocks if x["type"] == "PAGE_BREAK"] == list(
        range(25)
    )
    pages = [[]]
    for x in blocks:
        if x["type"] == "PAGE_BREAK":
            pages.append([])
        else:
            pages[-1].append(x)
    assert len(pages) == 26
    for n, card in enumerate(form["cards"], 1):
        pid = card["packet_id"]
        t = by_id[mapping["texts"][pid]]
        assert t in pages[n] and plain(t["payload"]["html"]) == card["text"]
        group = mapping["fields"][pid + "/gate"]
        assert group["required"]
        assert all(
            by_id[x] in pages[n]
            for f in mapping["fields"].values()
            if f["key"].startswith(pid + "/")
            for x in f["block_uuids"]
        )
        rules = [x for x in pages[n] if x["type"] == "CONDITIONAL_LOGIC"]
        assert len(rules) == 1
        rule = rules[0]["payload"]
        condition = rule["conditionals"][0]["payload"]
        action = rule["actions"][0]
        assert condition["field"]["blockGroupUuid"] == group["group_uuid"]
        assert (
            condition["value"]
            == group["option_uuids"][
                "Implicit hate" if form["framework"] == "source" else "Yes"
            ]
        )
        expected = [
            f
            for f in mapping["fields"].values()
            if f["key"].startswith(pid + "/") and f["hidden_initially"]
        ]
        assert action["type"] == "SHOW_BLOCKS" and action["payload"]["showBlocks"] == [
            i for f in expected for i in f["block_uuids"]
        ]
        for f in expected:
            for uid in f["block_uuids"]:
                assert by_id[uid]["payload"].get("isHidden") is True
            assert not f["required"]
    for f in mapping["fields"].values():
        assert f["required"] == f["key"].endswith("/gate")
        for uid in f.get("option_uuids", {}).values():
            p = by_id[uid]["payload"]
            assert p["hasDefaultAnswer"] is False and not p["randomize"]
            assert p["isRequired"] == f["required"]
    return {
        "cards": 24,
        "pages": 26,
        "exact_texts_checked": 24,
        "guide_exact": True,
        "conditional_rules": 24,
        "required_gates": 24,
        "other_required_fields": 0,
        "question_fields": len(mapping["fields"]),
        "blocks": len(blocks),
        "schema_valid": True,
        "status": "offline_blueprint_verified_not_uploaded",
    }


def main():
    results = {}
    for path in sorted((BASE / "forms").glob("Reviewer_*.json")):
        form, payload, mapping = build(path.stem)
        result = check(form, payload, mapping)
        save(OUT / (path.stem + ".json"), payload)
        save(OUT / (path.stem + "_mapping.json"), mapping)
        results[path.stem] = {
            **result,
            "payload_sha256": sha(OUT / (path.stem + ".json")),
            "mapping_sha256": sha(OUT / (path.stem + "_mapping.json")),
        }
    assert len(results) == 4
    save(
        OUT / "verification.json",
        {
            "forms": results,
            "source_sha256": sha(Path(__file__)),
            "openapi_sha256": sha(SCHEMA),
            "new_human_judgments": 0,
            "network_calls": 0,
            "platform_status": "pending_user_key_choice_and_live_validation",
            "timing_measured": False,
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
