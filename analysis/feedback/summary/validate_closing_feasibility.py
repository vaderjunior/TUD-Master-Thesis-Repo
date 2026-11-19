from collections import Counter
import csv
from pathlib import Path
import re
from analysis.feedback.summary.closing_review import ROOT
from analysis.feedback.summary.closing_review import DEST
from analysis.feedback.summary.closing_review import STUDIES
from analysis.feedback.summary.closing_review import LABELS
from analysis.feedback.summary.closing_review import read
from analysis.feedback.summary.closing_review import rows
from analysis.feedback.summary.closing_review import sha
from analysis.feedback.summary.closing_review import rel
from analysis.feedback.summary.closing_review import save
from analysis.feedback.summary.closing_review import verify
from analysis.feedback.summary.closing_review import require
from analysis.common.prompt import corrected_gold


def blocks(p):
    context = p["context"]
    flatten = lambda s: " ".join(str(s).split())
    ds = "## Label definitions\n" + "\n".join(
        (
            "- "
            + str(d["meta"].get("dimension"))
            + (" / " + d["meta"]["label"] if d["meta"].get("label") else "")
            + ": "
            + flatten(d["text"])
            for d in context["definitions"]
        )
    )
    gs = "## Annotation guidelines\n" + "\n".join(
        ("- " + d["text"] for d in context["guidelines"])
    )
    template = (ROOT / "prompts/classify_v1.txt").read_text(encoding="utf8")
    sections = re.split("^=== (\\w+) ===$", template, flags=re.MULTILINE)
    caution = {
        sections[i]: sections[i + 1].strip() for i in range(1, len(sections), 2)
    }["EXAMPLES_CAUTION"]
    es = []
    for name, examples in context["example_groups"]:
        require(
            not any(
                (
                    e["meta"].get("illustrative_only") or e["via"] == "static"
                    for e in examples
                )
            ),
            "Unexpected group rendering",
        )
        es.append(
            "## "
            + name
            + "\n"
            + caution
            + "\n\n"
            + "\n".join(
                (
                    '- "'
                    + flatten(e["text"])
                    + '"\n  labels: '
                    + corrected_gold(e["meta"])
                    for e in examples
                )
            )
        )
    result = {"definitions": ds, "guidelines": gs, "examples": "\n\n".join(es)}
    user = p["messages"][1]["text"]
    prefix = "\n\n".join([ds, gs, *es]) + "\n\n## Text to classify (language: en)\n"
    require(user.startswith(prefix), "Independent visible block renderer differs")
    return result


def main():
    prior = read(DEST / "extension_feasibility.json")
    verify(prior["inputs"])
    inputs = {
        rel(p): sha(p)
        for p in [
            Path(__file__),
            ROOT / "analysis/common/prompt.py",
            ROOT / "prompts/classify_v1.txt",
            ROOT / "src/hsrag/prompt.py",
            DEST / "extension_feasibility.json",
        ]
    }
    prompts = {
        (r["item_id"], r["condition"]): r
        for r in rows(STUDIES["taxonomy"] / "prompts.jsonl")
    }
    ids = sorted({i for i, c in prompts})
    rendered = {
        k: blocks(p)
        for k, p in prompts.items()
        if k[1] == "rag_full" or k[1].startswith("rag_schema_")
    }
    inventory = {}
    checks = 0
    for label in LABELS:
        c = Counter()
        for i in ids:
            f = rendered[i, "rag_full"]
            s = rendered[i, "rag_schema_" + label]
            e = f["examples"] != s["examples"]
            d = f["definitions"] != s["definitions"]
            g = f["guidelines"] != s["guidelines"]
            c["inputs"] += 1
            c["example_block_changed"] += e
            c["definition_block_changed"] += d
            c["guideline_block_changed"] += g
            c[f"E{int(e)}_D{int(d)}"] += 1
            require(
                (not (e or d or g))
                == (
                    prompts[i, "rag_full"]["messages_sha256"]
                    == prompts[i, "rag_schema_" + label]["messages_sha256"]
                ),
                "Visible block/hash disagreement",
            )
            checks += 1
        inventory[label] = dict(c)
    with (ROOT / "data/raw/implicit_hate/implicit_hate_v1_stg3_posts.tsv").open(
        encoding="utf8", newline=""
    ) as f:
        raw = list(csv.DictReader(f, delimiter="\t"))
    wanted = prior["human_annotations"]["sources_detail"]
    selected = {
        r["id"]: r
        for r in rows(
            ROOT
            / "experiments/astra/sq3_full_kb_feedback_20260928_v1/feasibility_v1/roster.jsonl"
        )
        if r["id"] in wanted
    }
    for i, v in wanted.items():
        matching = [
            r
            for r in raw
            if r["post"] == selected[i]["text"] and r["implied_statement"].strip()
        ]
        require(
            len(matching) == v["usable_exact_rows"], "Independent source match differs"
        )
        checks += 1
        require(
            [
                {"target": r["target"], "implied_statement": r["implied_statement"]}
                for r in matching
            ]
            == v["exact_annotations"],
            "Source annotation bytes differ",
        )
        checks += 1
    outputs = {
        "status": "passed",
        "checks": checks,
        "inputs": inputs,
        "rendered_block_inventory": inventory,
        "preliminary_inventory_note": "extension_feasibility.json compares raw context objects including metadata. This file is authoritative for model-visible example/definition/guideline differences. No live payloads created.",
        "human_exact_join_sources": sum(
            (v["usable_exact_rows"] > 0 for v in wanted.values())
        ),
        "new_model_calls": 0,
    }
    verify(inputs)
    save("feasibility_verification.json", outputs)
    print(
        {"status": "passed", "checks": checks, "rendered_block_inventory": inventory},
        flush=True,
    )


if __name__ == "__main__":
    main()
