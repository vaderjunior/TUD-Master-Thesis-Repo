from pathlib import Path
import json
import numpy as np
from analysis.feedback.transfer import transitions as p
from analysis.feedback.transfer import transitions_metrics as k

a = p.a
OUT = p.OUT
FINE = [
    ("en_test", "hate_type"),
    ("en_test", "target_group"),
    ("en_test", "severity"),
    ("en_hatecheck_eval", "target_group"),
]


def slots(meta, matrix, x):
    strict = [i for i, r in enumerate(matrix) if r["masks"]["strict_all_votes"]]
    score = x["scores"]
    result = {}
    states = {i: k.profile(score[i, ..., 0].tolist()) for i in strict}
    for pop in ("en_test", "en_hatecheck_eval"):
        for gold in (0, 1):
            for state in (
                "consistent_fix",
                "consistent_regression",
                "persistently_wrong",
            ):
                name = f"binary/{pop}/gold_{gold}/{state}"
                result[name] = [
                    i
                    for i in strict
                    if meta[i]["population"] == pop
                    and int(meta[i]["gold_hate"]) == gold
                    and states[i]["stability"] == state
                ]
    for source in ("hatexplain", "implicit_hate", "mhs"):
        for state in ("consistent_fix", "consistent_regression"):
            result[f"source/{source}/{state}"] = [
                i
                for i in strict
                if meta[i]["source"] == source and states[i]["stability"] == state
            ]
    for pop in ("en_test", "en_hatecheck_eval"):
        ids = [i for i in strict if meta[i]["population"] == pop]
        result[f"variation/{pop}/opposing_pool_directions"] = [
            i for i in ids if states[i]["opposing_pool_directions"]
        ]
        result[f"variation/{pop}/equal_average_with_changes"] = [
            i
            for i in ids
            if states[i]["direction"] == "equal_average" and states[i]["offsetting"]
        ]
        result[f"variation/{pop}/gate_repeat_variation"] = [
            i
            for i in ids
            if any(len(set(x["codes"][i, c, :, 0])) > 1 for c in range(4))
        ]
        result[f"variation/{pop}/identical_feedback_prompt_variation"] = [
            i
            for i in ids
            if any(
                x["messages"][i, c] == x["messages"][i, d]
                and np.any(x["codes"][i, c] != x["codes"][i, d])
                for c in range(1, 4)
                for d in range(c + 1, 4)
            )
        ]
    for pop, head in FINE:
        h = k.HEADS.index(head)
        ids = [
            i
            for i in strict
            if meta[i]["population"] == pop and k.eligible(meta[i], head)
        ]
        result[f"fine/{pop}/{head}/one_pool_consistent_fix"] = [
            i
            for i in ids
            if np.all(score[i, 0, :, h] == 0)
            and any(np.all(score[i, c, :, h] == 1) for c in range(1, 4))
        ]
        result[f"fine/{pop}/{head}/one_pool_consistent_regression"] = [
            i
            for i in ids
            if np.all(score[i, 0, :, h] == 1)
            and any(np.all(score[i, c, :, h] == 0) for c in range(1, 4))
        ]
        result[f"fine/{pop}/{head}/binary_fix_fine_still_wrong"] = [
            i
            for i in ids
            if np.all(score[i, 0, :, 0] == 0)
            and any(
                np.all(score[i, c, :, 0] == 1)
                and np.all(score[i, c, :, h] == 0)
                and np.all(score[i, c, :, 1] == 0)
                for c in range(1, 4)
            )
        ]
    return result


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
            *[
                "| "
                + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row)
                + " |"
                for row in rows
            ],
        ]
    )


def main():
    done = a.read(OUT / "calculated.json")
    assembled = a.read(OUT / "assembled.json")
    for name, digest in {**done["artifacts"], **assembled["artifacts"]}.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed case input")
    a.need(
        not (OUT / "case_selection.json").exists(), "Preserve previous case selection"
    )
    p.save(
        "case_selection_contract.json",
        dict(
            created_utc=a.utc(),
            script_sha256=a.sha(Path(__file__)),
            rule="Minimum fixed SHA256 slot/key among all strict-36-vote inputs meeting the saved slot criteria. No replacement for duplicate selected keys. Empty slots remain explicit. Do not inspect text or raw reasoning to choose cases.",
            binary_consistency="All three reference aggregates wrong/correct and all nine feedback aggregates correct/wrong. Persistent error means all twelve binary decisions disagree with recorded gold.",
            fine_consistency="Within one correction pool all three fine exact answers change wrong-to-right or right-to-wrong relative to all three reference repeats. All other pools remain visible.",
            binary_fix_fine_still_wrong="In at least one pool all three binary answers fix the three reference mistakes, but the specified fine head and joint known-field exact remain wrong in all three.",
            identical_message_case="At least one feedback-pool pair has identical message hash but a different canonical five-field output in an aligned repeat. No label-only causal attribution.",
            prior_contexts="All eight Part 5 cases retained regardless of outcomes or strict validity.",
            slots=38,
            prior_context_slots=8,
            new_Qwen_calls=0,
        ),
    )
    meta = list(a.rows(p.P4 / "metadata.jsonl"))
    matrix = list(a.rows(a.OUT / "matrix.jsonl"))
    roster = list(a.rows(a.OUT / "roster.jsonl"))
    bykey = {m["key"]: i for i, m in enumerate(meta)}
    with np.load(OUT / "arrays.npz", allow_pickle=False) as f:
        x = {n: f[n] for n in ("scores", "codes", "messages", "n_valid")}
    candidates = slots(meta, matrix, x)
    a.need(len(candidates) == 38, "Case slot coverage")
    chosen = {}
    for name, ids in sorted(candidates.items()):
        idx = min(ids, key=lambda i: k.rank(name, meta[i]["key"])) if ids else None
        chosen[name] = dict(
            candidate_count=len(ids),
            input_index=idx,
            key=meta[idx]["key"] if idx is not None else None,
            selection_hash=k.rank(name, meta[idx]["key"]) if idx is not None else None,
            eligibility="strict_all_votes",
            candidate_indices=ids,
            status="selected" if idx is not None else "no_eligible_case",
        )
    prior = a.read(p.P5 / "context_examples.json")
    for name, c in prior.items():
        key = c["query_context"]["key"]
        idx = bykey[key]
        chosen["part5_context/" + name] = dict(
            candidate_count=1,
            input_index=idx,
            key=key,
            selection_hash=c["selection_hash"],
            eligibility="original_outcome_independent_part5_choice",
            candidate_indices=[idx],
            status="selected",
        )
    p.save("case_selection.json", chosen)
    selected = sorted(
        {q["input_index"] for q in chosen.values() if q["input_index"] is not None}
    )
    keys = {meta[i]["key"] for i in selected}
    contexts = {key: {} for key in keys}
    actual = {key: {} for key in keys}
    for r in a.rows(a.STUDY / "prompts.jsonl"):
        if r["key"] in keys:
            contexts[r["key"]][r["condition"]] = r
    for path in sorted((a.OUT / "assembly").glob("transfer_*.jsonl")):
        for r in a.rows(path):
            if r["key"] in keys:
                actual[r["key"]][r["condition"] + "/" + str(r["repeat"])] = {
                    name: r[name]
                    for name in (
                        "result",
                        "votes",
                        "n_valid",
                        "uncertain",
                        "messages_sha256",
                        "item_receipt",
                        "item_receipt_sha256",
                        "result_sha256",
                        "stage",
                        "execution_period",
                        "repeat",
                        "condition",
                    )
                }
    qctx = {
        r["key"]: r for r in a.rows(p.P5 / "query_contexts.jsonl") if r["key"] in keys
    }
    delivery = {key: [] for key in keys}
    for r in a.rows(p.P5 / "delivery.jsonl"):
        if r["key"] in keys:
            delivery[r["key"]].append(r)
    catalog = a.read(p.P5 / "source_catalog.json")
    index = []
    for number, i in enumerate(selected, 1):
        m = meta[i]
        key = m["key"]
        caseid = f"case_{number:02d}"
        ss = [name for name, q in chosen.items() if q["input_index"] == i]
        a.need(
            len(contexts[key]) == 4 and len(actual[key]) == 12,
            "Incomplete selected case",
        )
        for c, condition in enumerate(p.CONDITIONS):
            prompt = contexts[key][condition]
            a.need(
                prompt["messages_sha256"] == x["messages"][i, c], "Selected case prompt"
            )
            for repeat in range(3):
                a.need(
                    actual[key][condition + "/" + str(repeat)]["messages_sha256"]
                    == prompt["messages_sha256"],
                    "Selected case output prompt",
                )
        profiles = {
            head: k.profile(x["scores"][i, ..., h].tolist())
            for h, head in enumerate(k.HEADS)
        }
        value = dict(
            case_id=caseid,
            input_index=i,
            key=key,
            selected_slots=ss,
            metadata=m,
            roster=roster[i],
            masks=matrix[i]["masks"],
            profiles=profiles,
            predictions=matrix[i]["predictions"],
            actual_records=actual[key],
            prompts=contexts[key],
            query_context=qctx[key],
            delivered_corrections=delivery[key],
            source_records={
                r["source_id"]: catalog[r["source_id"]] for r in delivery[key]
            },
            interpretation_boundary="Recorded outputs and dataset gold only; no claim about internal reasoning or expert relabelling. Cases are deliberately selected categories, not representative prevalence.",
        )
        p.save("cases/" + caseid + ".json", value)
        rows = []
        for c, condition in enumerate(p.CONDITIONS):
            for repeat in range(3):
                record = actual[key][condition + "/" + str(repeat)]
                v = record["result"]
                tag = "Reference" if c == 0 else "Pool " + str(c + 440)
                rows.append(
                    [
                        tag,
                        repeat + 1,
                        record["n_valid"],
                        "missing" if v is None else int(v["hate"]),
                        *[
                            (
                                "missing"
                                if v is None
                                else json.dumps(v[f], ensure_ascii=False)
                            )
                            for f in ("hate_type", "target_group", "severity", "legal")
                        ],
                        "missing" if v is None else x["scores"][i, c, repeat, 1],
                        ",".join(
                            "null" if q is None else str(int(q["hate"]))
                            for q in record["votes"]
                        ),
                    ]
                )
        corrections = []
        for d in delivery[key]:
            source = catalog[d["source_id"]]["record"]
            corrections.append(
                f"### Pool {d['pool_index']+440}: {d['source_id']}\n\n```text\n{source['text']}\n```\n\nShown binary label: `{source['meta']['gate']}`; all fine annotations unknown. Cosine {d['cosine']:.6f}; removed-example cosine {d['removed_cosine']:.6f}. Complete original prompt and prior source screening are in the case JSON.\n"
            )
        gold = {
            name: m[name]
            for name in (
                "gold_hate",
                "gold_hate_types",
                "gold_target_groups",
                "gold_severity",
                "gold_legal",
                "eligible_fine_gold",
            )
        }
        text = f"""# {caseid}: {key}

Selected slots: {', '.join(ss)}. These slots deliberately illustrate outcomes; their counts are not population prevalence. This case uses recorded dataset gold, without fresh human adjudication. Strict all-36-vote input: {matrix[i]['masks']['strict_all_votes']}.

```text
{roster[i]['text']}
```

## Recorded gold and scoring eligibility

```json
{json.dumps(gold,ensure_ascii=False,indent=2)}
```

## All twelve saved aggregate outputs

Known exact means gate plus available eligible fine gold, not validation of every schema field. Legal outputs have no gold in this evaluation. Votes show the three raw saved binary decisions; `null` is a failed vote. Full five-field raw votes and receipt bindings are in [{caseid}.json]({caseid}.json).

{table(['Condition','Repeat','Valid votes','Hate','Types','Targets','Severity','Legal','Known exact','Raw hate votes'],rows)}

## Four retained ordinary examples

```json
{json.dumps(qctx[key]['ordinary_hits'][:4],ensure_ascii=False,indent=2)}
```

## Removed fifth example

```json
{json.dumps(qctx[key]['ordinary_hits'][4],ensure_ascii=False,indent=2)}
```

## Corrections inserted in its place

{''.join(corrections)}

The complete original four system/user message pairs, exact query-specific metadata, all thirty-six saved votes (including any missing votes), source labels, twelve result bindings, masks and per-head transition profiles are preserved in [{caseid}.json]({caseid}.json). No output was regenerated. The [source roster](../../sq3_transfer_part1_20261023_v1/roster.jsonl) retains the source-specific original annotations; the case JSON includes this text's original roster row.
"""
        p.save("cases/" + caseid + ".md", text.encode("utf8"))
        index.append(
            dict(
                case_id=caseid,
                key=key,
                input_index=i,
                population=m["population"],
                source=m["source"],
                gold_hate=m["gold_hate"],
                slots=ss,
                binary_profile=profiles["binary"],
                known_answer_profile=profiles["known_answer"],
                json="cases/" + caseid + ".json",
                markdown="cases/" + caseid + ".md",
            )
        )
    p.save("case_index.json", index)
    lines = []
    for row in index:
        profile = row["binary_profile"]
        lines.append(
            [
                f"[{row['case_id']}]({row['markdown']})",
                row["key"],
                int(row["gold_hate"]),
                profile.get("reference_correct"),
                profile.get("pool_correct"),
                profile.get("stability", "unavailable"),
                "; ".join(row["slots"]),
            ]
        )
    empty = [name for name, q in chosen.items() if q["input_index"] is None]
    md = (
        "# Mechanically selected output cases\n\n38 outcome-defined strict-vote slots plus all eight earlier context examples. Each slot uses its smallest fixed identity hash, not appealing text. Cases may serve multiple slots; no substitute is chosen to force unique anecdotes. Empty slots remain below. All four conditions, three aggregate repeats and original three votes per output are available for each case.\n\n"
        + table(
            [
                "Case",
                "Text key",
                "Recorded hate",
                "Reference correct / 3",
                "Pool correct / 3 each",
                "Binary consistency",
                "Selected slots",
            ],
            lines,
        )
        + "\n\n## Slots with no eligible strict case\n\n"
        + ("\n".join("- " + name for name in empty) if empty else "None.")
        + "\n"
    )
    p.save("case_index.md", md.encode("utf8"))
    files = [
        OUT / "case_selection_contract.json",
        OUT / "case_selection.json",
        OUT / "case_index.json",
        OUT / "case_index.md",
    ] + sorted((OUT / "cases").glob("*"))
    p.save(
        "cases_complete.json",
        dict(
            status="part6_cases_assembled_independent_verification_pending",
            created_utc=a.utc(),
            source_sha256=a.sha(Path(__file__)),
            calculated_sha256=a.sha(OUT / "calculated.json"),
            artifacts={a.rel(f): a.sha(f) for f in files},
            slots=len(chosen),
            empty_slots=empty,
            unique_cases=len(index),
            raw_vote_slots=len(index) * 36,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        f"Saved {len(index)} complete cases for {len(chosen)} slots; {len(empty)} empty slots retained."
    )


if __name__ == "__main__":
    with a.offline():
        main()
