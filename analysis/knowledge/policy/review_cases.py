import argparse
from collections import Counter
import json
import numpy as np
from analysis.knowledge.policy import labels as p

c = p.c
B = p.BASE


def text_file(name, text):
    with (B / name).open("x", encoding="utf8", newline="\n") as f:
        f.write(text)


def cases():
    contract, root, roster, endpoints, arrays = p.load_inputs()
    selected = c.read(B / "case_selection.json")
    keys = {v["key"] for v in selected if v["key"]}
    features = {
        v["key"]: v for v in c.rows(B / "context_features.jsonl") if v["key"] in keys
    }
    contexts = {
        v["key"]: v["context"]
        for v in c.rows(B / "selected_original_contexts.jsonl")
        if v["key"] in keys
    }
    reader_done = c.read(p.PART3 / "reader_complete.json")
    index = {}
    bindings = {}
    for name, expected in reader_done["reader_files"].items():
        path = c.ROOT / name
        c.require(c.sha(path) == expected, "Reader changed")
        for entry in c.read(path):
            if entry["key"] not in keys:
                continue
            rawpath = p.reader.LIVE / "items" / entry["stamp"] / "result.json"
            c.require(c.sha(rawpath) == entry["result_sha256"], "Case result changed")
            raw = c.read(rawpath)
            bindings[c.ref(rawpath)] = entry["result_sha256"]
            result = raw["result"]
            c.require(
                {k: v for k, v in result.items() if k != "reasoning"}
                == entry["result"],
                "Case reader differs",
            )
            index[entry["key"], entry["condition"], entry["repeat"]] = dict(
                result=result,
                stage=entry["stage"],
                result_path=c.ref(rawpath),
                result_sha256=entry["result_sha256"],
                n_valid=entry["n_valid"],
                repairs=entry["repairs"],
            )
    c.require(len(index) == len(keys) * 9, "Incomplete case results")
    cards = []
    seen = {}
    compact = []
    full = [
        "# Part 4 selected case evidence",
        "",
        "Illustrative, outcome-selected examples. Selection did not read text or explanations. All nine final answers and original five examples are preserved in case_cards.json. Offensive language is original research material. Dataset labels are the scoring reference, not a fresh adjudication. Model explanations are not evidence of internal reasoning.",
        "",
    ]
    for slot in selected:
        key = slot["key"]
        if key is None:
            continue
        if key in seen:
            cards[seen[key]]["slots"].append(slot)
            continue
        seen[key] = len(cards)
        card = dict(
            case_id=f"CP4-{len(cards) + 1:02d}",
            key=key,
            slots=[slot],
            input=roster[key],
            context=features[key],
            original_context=contexts[key],
            answers={
                cond: [index[key, cond, r] for r in range(3)] for cond in p.CONDITIONS
            },
        )
        cards.append(card)
    for card in cards:
        r = card["input"]
        f = card["context"]
        header = f"{card['case_id']} — {card['key']}"
        slots = "; ".join(
            (
                v["endpoint"]
                + " "
                + v["direction"]
                + " "
                + v["left"]
                + " vs "
                + v["right"]
                for v in card["slots"]
            )
        )
        lines = [
            "## " + header,
            "",
            slots,
            "",
            "Query: " + r["text"],
            "",
            "Gold (null = unavailable): "
            + json.dumps(
                {
                    k: r.get(k)
                    for k in (
                        "gate",
                        "hate_types",
                        "target_groups",
                        "severity",
                        "legal",
                    )
                },
                ensure_ascii=False,
            ),
            "",
            "Original definitions: " + ", ".join(f["definitions_A"]),
            "Original guidelines: " + ", ".join(f["guidelines_A"]),
            "Input tokens A/B/C: "
            + str([f["settings"][cond]["tokens"] for cond in p.CONDITIONS]),
            "",
        ]
        for short, cond in zip("ABC", p.CONDITIONS):
            results = [v["result"] for v in card["answers"][cond]]
            for endpoint in dict.fromkeys((v["endpoint"] for v in card["slots"])):
                field = endpoints[endpoint]["dimension"]
                state = [
                    (
                        {"hate": q["hate"], field: q[field]}
                        if field not in ("hate", "legal")
                        else q[field]
                    )
                    for q in results
                ]
                lines.append(
                    short
                    + " "
                    + endpoint
                    + ": "
                    + json.dumps(state, ensure_ascii=False)
                )
            lines.append(
                short + " repeat 1 explanation: " + str(results[0].get("reasoning", ""))
            )
        lines.append("")
        compact.extend(lines)
        full.extend(lines)
        full.append("Original five examples (identical across conditions):")
        for _, group in card["original_context"]["example_groups"]:
            for example in group:
                full.extend(
                    [
                        "",
                        example["id"] + ": " + example["text"],
                        json.dumps(example["meta"], ensure_ascii=False, sort_keys=True),
                    ]
                )
        full.append("")
    p.save(B / "case_cards.json", cards)
    text_file("case_cards_for_review.md", "\n".join(compact) + "\n")
    text_file("case_evidence_appendix.md", "\n".join(full) + "\n")
    p.save(
        B / "case_bindings.json",
        dict(
            raw_result_files=bindings,
            case_texts=len(cards),
            slots=len(selected),
            result_files=len(index),
        ),
    )
    print(
        "Case cards:",
        len(cards),
        "unique texts;",
        len(index),
        "hashed raw final answers.",
        flush=True,
    )


def f3(value):
    return f"{value:.3f}"


def pc(value):
    return f"{100 * value:.2f}"


def numbers(seq, style=f3):
    return " / ".join((style(v) for v in seq))


def table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(("---" for _ in headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join((str(v).replace("|", "/") for v in row)) + " |")
    lines.append("")


def tables():
    data = c.read(B / "groups_and_sources.json")
    summary = c.read(B / "context_summary.json")
    exposure = c.read(B / "label_definition_exposure.json")
    out = [
        "# Part 4 descriptive numerical appendix",
        "",
        "A = retrieved policy; B = all definitions with original guidelines; C = all definitions and guidelines. Values are means of three separately scored repeats. All performance tables use the unchanged common-nine cohorts. Exposure inventory uses the full eligible population unless marked common. No subgroup confidence intervals or causal effects are claimed. F1 retains every frozen label, including any with no gold positives in a subgroup. Counts averaged over repeats can be fractional.",
        "",
        "## Policy delivery across the eligible populations",
        "",
    ]
    table(
        out,
        [
            "Endpoint",
            "N",
            "Any task definition in A",
            "0 / 1 / 2 legal guides in A",
            "Mean input tokens A/B/C",
        ],
        [
            [
                e,
                v["n"],
                v["any_dimension_definition_A"][data[e]["meta"]["dimension"]],
                " / ".join(
                    (str(v["legal_guidelines_A"].get(str(i), 0)) for i in range(3))
                ),
                numbers(
                    [v["tokens"][cond]["mean_tokens"] for cond in p.CONDITIONS],
                    lambda v: f"{v:.1f}",
                ),
            ]
            for e, v in summary["populations"].items()
            if e in data
        ],
    )
    for e, value in data.items():
        out.extend(["## " + p.NAMES[e], ""])
        sources = {k: v for k, v in value["groups"].items() if k.startswith("source/")}
        table(
            out,
            ["Source", "N", "Macro F1 A/B/C", "Micro F1 A/B/C", "Exact % A/B/C"],
            [
                [
                    k[7:],
                    v["n"],
                    numbers(v["macro_f1"]),
                    numbers(v["micro_f1"]),
                    numbers(v["exact"], pc),
                ]
                for k, v in sources.items()
            ],
        )
        if value["meta"]["dimension"] == "hate":
            table(
                out,
                [
                    "Source",
                    "Gold negative / positive",
                    "FP A/B/C",
                    "FN A/B/C",
                    "FPR % A/B/C",
                    "FNR % A/B/C",
                ],
                [
                    [
                        k[7:],
                        f"{v['negative_n']} / {v['positive_n']}",
                        numbers(
                            np.mean(v["confusion_repeats"]["fp"], axis=1),
                            lambda x: f"{x:.2f}",
                        ),
                        numbers(
                            np.mean(v["confusion_repeats"]["fn"], axis=1),
                            lambda x: f"{x:.2f}",
                        ),
                        numbers(v["false_positive_rate"], pc),
                        numbers(v["false_negative_rate"], pc),
                    ]
                    for k, v in sources.items()
                ],
            )
        table(
            out,
            [
                "Source",
                "C-A fixes / regressions per repeat",
                "C-A stable fixes / regressions",
                "Texts with more / fewer correct repeats",
            ],
            [
                [
                    k[7:],
                    f"{np.mean(v['transitions'][0]['fixes_repeats']):.2f} / {np.mean(v['transitions'][0]['regressions_repeats']):.2f}",
                    f"{v['transitions'][0]['stable_fixes']} / {v['transitions'][0]['stable_regressions']}",
                    f"{v['transitions'][0]['more_correct_repeat_texts']} / {v['transitions'][0]['fewer_correct_repeat_texts']}",
                ]
                for k, v in sources.items()
            ],
        )
        counts = np.array(value["whole"]["counts"])
        tp, fp, fn = (counts[..., i] for i in range(3))
        precision = p.metric.divide(tp, tp + fp).mean(axis=1)
        recall = p.metric.divide(tp, tp + fn).mean(axis=1)
        table(
            out,
            [
                "Label",
                "Gold N",
                "Precision A/B/C",
                "Recall A/B/C",
                "F1 A/B/C",
                "Mean TP A/B/C",
                "Mean FP A/B/C",
                "Mean FN A/B/C",
            ],
            [
                [
                    label,
                    value["whole"]["gold_support"][j],
                    numbers(precision[:, j]),
                    numbers(recall[:, j]),
                    numbers(np.array(value["whole"]["per_label_f1"])[:, j]),
                    numbers(tp.mean(axis=1)[:, j], lambda x: f"{x:.2f}"),
                    numbers(fp.mean(axis=1)[:, j], lambda x: f"{x:.2f}"),
                    numbers(fn.mean(axis=1)[:, j], lambda x: f"{x:.2f}"),
                ]
                for j, label in enumerate(value["meta"]["all_labels"])
            ],
        )
        for prefix in (
            "gold_definition_coverage/",
            "dimension_definition_A/",
            "legal_guidelines_A/",
            "frozen_A_token_quartile/",
            "definition_relative_order_reversed/",
            "guideline_relative_order_reversed/",
        ):
            out.extend(["### " + prefix.rstrip("/"), ""])
            table(
                out,
                ["Group", "N", "Macro F1 A/B/C", "Exact % A/B/C", "Gold support"],
                [
                    [
                        k[len(prefix) :],
                        v["n"],
                        numbers(v["macro_f1"]),
                        numbers(v["exact"], pc),
                        str(v["gold_support"]),
                    ]
                    for k, v in value["groups"].items()
                    if k.startswith(prefix) and v["n"]
                ],
            )
        if value["meta"]["dimension"] != "hate":
            out.extend(
                [
                    "### Gold-positive matching-definition exposure",
                    "",
                    "This is recall within retrieved-definition strata, not the causal value of a definition. Small cells need particular care.",
                    "",
                ]
            )
            table(
                out,
                [
                    "Label",
                    "Definition already in A",
                    "Gold-positive N",
                    "Recall % A/B/C",
                ],
                [
                    [
                        v["label"],
                        v["definition_A_present"],
                        v["n"],
                        numbers(v["predicted_positive_rate"], pc),
                    ]
                    for v in exposure
                    if v["endpoint"] == e and v["gold_positive"]
                ],
            )
        if "severity_gold_by_prediction" in value:
            matrix = np.array(
                value["severity_gold_by_prediction"]["counts_repeats"]
            ).mean(axis=1)
            out.extend(["### Severity confusion: mean counts per repeat", ""])
            table(
                out,
                ["Condition", "Gold band", "Predicted low", "medium", "high", "absent"],
                [
                    [cond, band, *[f"{x:.2f}" for x in matrix[i, j]]]
                    for i, cond in enumerate("ABC")
                    for j, band in enumerate(("low", "medium", "high"))
                ],
            )
        out.append(
            "Identical wrong state in all six A/C final answers: "
            + str(value["persistent_identical_wrong_count"])
            + " texts.\n"
        )
    text_file("numerical_appendix.md", "\n".join(out) + "\n")
    print("Descriptive numerical appendix saved.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cases", "tables"])
    args = parser.parse_args()
    with c.offline():
        globals()[args.mode]()
