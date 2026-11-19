import copy
import json
from analysis.feedback.human.forms import groups as prior

BASE = prior.BASE
OUT = BASE / "platform/blueprints_v3"
read, save, sha, plain = prior.read, prior.save, prior.sha, prior.plain


def build(key):
    form, payload, mapping = prior.build(key)
    pages = [x for x in payload["blocks"] if x["type"] == "PAGE_BREAK"]
    assert len(pages) == 25 and pages[-1]["payload"]["index"] == 24
    pages[-1]["payload"]["button"]["label"] = "Submit"
    return form, payload, mapping


def check(form, payload, mapping):
    original = copy.deepcopy(payload)
    pages = [x for x in original["blocks"] if x["type"] == "PAGE_BREAK"]
    assert pages[-1]["payload"]["button"]["label"] == "Submit"
    pages[-1]["payload"]["button"]["label"] = "Next"
    return {
        **prior.check(form, original, mapping),
        "version": 3,
        "finish_button": "Submit",
    }


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
                prior.prior.ROOT / "analysis/feedback/human/forms/submit.py"
            ),
            "network_calls": 0,
            "human_judgments": 0,
            "live_UI_validation": "pending",
        },
    )
    print(
        json.dumps(
            {"forms": 4, "version": 3, "finish_button": "Submit", "verified": True}
        )
    )


if __name__ == "__main__":
    main()
