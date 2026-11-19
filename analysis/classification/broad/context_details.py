from collections import Counter, defaultdict
from analysis.classification.broad import contexts as x


def run():
    x.r.verify(x.c.read(x.BASE / "analysis_complete.json")["outputs"])
    catalog = x.c.read(x.BASE / "hit_catalog.json")
    group = x.c.read(x.BASE / "group_membership.json")
    ds = defaultdict(dict)
    for d in x.reader.rows(x.BASE / "prompt_descriptors.jsonl"):
        ds[d["condition"]][d["key"]] = d
    extra = {}
    for name, keys in group.items():
        extra[name] = {}
        for co in x.CONDITIONS[:2]:
            dd = [ds[co][k] for k in keys]
            value = {}
            for ch in ("definitions", "guidelines", "examples"):
                ids = [ident for d in dd for ident in d["ids"][ch]]
                counts = Counter(ids)
                value[ch] = dict(
                    hit_counts=dict(counts),
                    languages=dict(Counter(catalog[i]["lang"] for i in ids)),
                    sources=dict(Counter(catalog[i]["source"] for i in ids)),
                    normalized_unique_texts=len(
                        {x.normalized(catalog[i]["text"]) for i in ids}
                    ),
                    within_prompt_normalized_duplicate_inputs=sum(
                        len({x.normalized(catalog[i]["text"]) for i in d["ids"][ch]})
                        < len(d["ids"][ch])
                        for d in dd
                    ),
                )
            value["full_examples_from_small_candidates"] = sum(
                d["examples_in_small"] for d in dd
            )
            extra[name][co] = value
    x.save(x.BASE / "channel_inventory.json", extra)
    cases = x.c.read(x.BASE / "case_contexts.json")
    guide = []
    for ident, record in sorted(catalog.items()):
        if record["kind"] != "example":
            guide.append(ident + " | " + record["text"])
    directory = x.BASE / "inspection"
    directory.mkdir(exist_ok=True)
    with (directory / "definitions_and_guidelines.txt").open("x", encoding="utf8") as f:
        f.write("\n\n".join(guide) + "\n")

    def order(case):
        return min(case["case_ids"])

    seen = set()
    batches = []
    for start in range(0, len(cases), 8):
        rows = []
        block = sorted(cases, key=order)[start : start + 8]
        for case in block:
            rows.append(
                "\nCASE "
                + ",".join(case["case_ids"])
                + " "
                + case["key"]
                + "\nQUERY "
                + case["gold"]["text"]
            )
            rows.append(
                "GOLD "
                + str(
                    {
                        k: case["gold"].get(k)
                        for k in (
                            "gate",
                            "hate_types",
                            "target_groups",
                            "severity",
                            "legal",
                        )
                    }
                )
            )
            for co in x.CONDITIONS[:2]:
                results = sorted(
                    [p for p in case["predictions"] if p["condition"] == co],
                    key=lambda p: p["repeat"],
                )
                rows.append(
                    co
                    + " ANSWERS "
                    + str(
                        [
                            (
                                {
                                    k: p["result"].get(k)
                                    for k in (
                                        "hate",
                                        "hate_type",
                                        "target_group",
                                        "severity",
                                        "legal",
                                    )
                                }
                                if p["result"]
                                else None
                            )
                            for p in results
                        ]
                    )
                )
                ctx = case["prompts"][co]["context"]
                rows.append(
                    "DEFS "
                    + str([h["id"] for h in ctx["definitions"]])
                    + " GUIDES "
                    + str([h["id"] for h in ctx["guidelines"]])
                )
                for h in x.examples(ctx):
                    rows.append(
                        h["id"]
                        + " | "
                        + x.reader.prompt.corrected_gold(h["meta"])
                        + " | "
                        + (
                            "[already printed in earlier batch]"
                            if h["id"] in seen
                            else h["text"]
                        )
                    )
                    seen.add(h["id"])
            rows.append(
                "FULL RATIONALE r0 "
                + next(
                    p["result"]["reasoning"]
                    for p in case["predictions"]
                    if p["condition"] == "rag_full" and p["repeat"] == 0
                )
            )
        path = directory / f"cases_{start+1:03d}_{min(start+8,len(cases)):03d}.txt"
        with path.open("x", encoding="utf8") as f:
            f.write("\n".join(rows) + "\n")
        batches.append(
            dict(path=x.c.rel(path), case_ids=[case["case_ids"] for case in block])
        )
    x.save(
        x.BASE / "context_reading_inventory.json",
        dict(
            batches=batches,
            case_inputs=len(cases),
            unique_examples=len(seen),
            rule="Every fixed prior selected input; repeated example text is printed at its first occurrence, all labels and ranks stay in case_contexts.json. No text truncation.",
            sources=x.bindings(
                [
                    x.BASE / "case_contexts.json",
                    x.BASE / "channel_inventory.json",
                    x.ROOT / "analysis/classification/broad/context_details.py",
                ]
            ),
            files=x.bindings(list(directory.iterdir())),
        ),
    )
    print(
        "Context reading batches prepared:",
        len(cases),
        "inputs,",
        len(seen),
        "distinct example texts.",
    )


if __name__ == "__main__":
    with x.r.original.offline():
        run()
