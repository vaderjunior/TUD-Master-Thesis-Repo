from collections import Counter, defaultdict
from pathlib import Path
from analysis.common import replay as c
from analysis.classification.validate_preparation import label

BASE = c.BASE / "evaluation_preparation"


def main():
    amendment = c.read(BASE / "coverage_amendment.json")
    c.verify(amendment["inputs"])
    receipt = c.read(BASE / "coverage_preparation.json")
    c.verify(receipt["outputs"])
    kb = {
        r["id"]: r
        for r in c.rows(c.CORE / "kb/full/records.jsonl")
        if r["kind"] == "example"
    }
    old = {
        r["key"]: r
        for r in c.rows(BASE / "matched_prompts.jsonl")
        if r["condition"] == "retrieved_complete"
    }
    rows = c.rows(BASE / "coverage_matched_prompts.jsonl")
    seen = set()
    slots = 0
    coverage = defaultdict(Counter)
    for r in rows:
        assert (r["key"], r["condition"]) not in seen
        seen.add((r["key"], r["condition"]))
        reference = old[r["key"]]
        ids = r["example_text_ids"]
        original = reference["example_text_ids"]
        assert r["annotation_source_ids"] == ids and len(ids) == len(set(ids)) == 5
        assert (
            r["messages"][0] == reference["messages"][0]
            and r["scopes"] == reference["scopes"]
        )
        assert r["messages_sha256"] == c.signature(r["messages"])
        expect = []
        for k, (newid, oldid) in enumerate(zip(ids, original)):
            a, b = (kb[newid], kb[oldid])
            am, bm = (a["meta"], b["meta"])
            assert (
                a["lang"] == b["lang"]
                and a["source"] == b["source"]
                and (am.get("gate") == bm.get("gate"))
            )
            for f in ["target_groups", "hate_types", "legal"]:
                assert (am.get(f) is None) == (bm.get(f) is None)
                if am.get(f) is not None:
                    assert len(am[f]) == len(bm[f])
            assert (am.get("severity") is None) == (bm.get("severity") is None)
            assert bool(am.get("astra_aux_stgb")) == bool(bm.get("astra_aux_stgb"))
            assert newid not in original and c.normalized(a["text"]) != c.normalized(
                b["text"]
            )
            expect.append(
                str(k + 1)
                + '. "'
                + " ".join(a["text"].split())
                + '"\n   labels: '
                + label(am)
            )
            slots += 1
            for scope in r["scopes"]:
                dim = (
                    "hate_type"
                    if scope["dimension"].startswith("hate_type")
                    else scope["dimension"]
                )
                field = c.original.GOLD_FIELDS[dim]
                coverage[scope["subset"] + "/" + r["condition"]]["slots"] += 1
                coverage[scope["subset"] + "/" + r["condition"]][
                    "annotated_slots"
                ] += int(am.get(field) is not None)
        text = r["messages"][1]["text"]
        originaltext = reference["messages"][1]["text"]
        left, middle = text.split("## Example context\n", 1)
        body, right = middle.split("\n\n## Text to classify", 1)
        assert left == originaltext.split("## Example context\n", 1)[0]
        assert right == originaltext.split("\n\n## Text to classify", 1)[1]
        assert body == "\n".join(expect)
    assert len(rows) == 2823 and receipt["fallback_count"] == 0
    c.save(
        BASE / "coverage_verification.json",
        {
            "status": "independent_coverage_controls_pass",
            "created_utc": c.utc(),
            "source_sha256": c.sha(Path(__file__)),
            "amendment_sha256": c.sha(BASE / "coverage_amendment.json"),
            "prompts": len(rows),
            "slots": slots,
            "unchanged_slots": 0,
            "fallbacks": 0,
            "annotation_delivery": dict(coverage),
            "new_model_calls": 0,
        },
    )
    print(
        "Independent coverage checks passed:",
        len(rows),
        "prompts;",
        slots,
        "slots; all source/gate/availability/cardinality matches; no unchanged fallback.",
    )


if __name__ == "__main__":
    with c.network_guard():
        main()
