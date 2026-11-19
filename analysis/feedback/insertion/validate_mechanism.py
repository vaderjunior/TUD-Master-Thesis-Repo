from collections import Counter, defaultdict
import json
from pathlib import Path
import socket
from statistics import mean
from unittest.mock import patch
from analysis.feedback.insertion import archive as r
from analysis.feedback.insertion import common as c

OUT = r.DEST / "mechanism_check_v1"


def main():
    target = OUT / "complete.json"
    if target.exists():
        done = r.read(target)
        r.verify(done["inputs"])
        r.verify(done["outputs"])
        print("Existing mechanism review verified.")
        return
    done = r.read(r.DEST / "diagnostics_v2/complete.json")
    r.verify(done["inputs"])
    r.verify(done["outputs"])
    paths = [
        Path(__file__),
        r.DEST / "diagnostics_v2/complete.json",
        r.REVIEW / "summary.json",
        r.REVIEW / "predictions.jsonl",
        c.BASE / "prompts.jsonl",
        c.SELECTION / "actual_pools.json",
        c.SELECTION / "gate_only_records.json",
        c.CORE / "kb/full/records.jsonl",
    ]
    inputs = {r.rel(p): r.sha(p) for p in paths}
    r.save(
        OUT / "plan.json",
        dict(
            created_utc=r.utc(),
            inputs=inputs,
            scope="Post-score descriptive check of all saved RRF rankings, delivery metadata, identical-message contrasts, all exposed cases and terminal schema failures. No new hypothesis family, fitting, model calls or score replacement.",
        ),
    )
    rows = {
        (v["key"], v["condition"], v["repeat"]): v
        for v in r.rows(r.REVIEW / "predictions.jsonl")
    }
    meta = {
        (v["key"], v["condition"]): v
        for v in r.rows(r.DEST / "diagnostics_v2/prompt_metadata.jsonl")
    }
    roster = {v["key"]: v for v in r.rows(c.BASE / "roster.jsonl")}
    pools = {
        v["condition"]: v for v in r.read(c.SELECTION / "actual_pools.json")["pools"]
    }
    source = r.read(c.SELECTION / "gate_only_records.json")
    original = {v["id"]: v for v in r.rows(c.CORE / "kb/full/records.jsonl")}
    refs = {v["key"]: v for v in r.rows(c.BASE / "retrieval/reference/prompts.jsonl")}
    counts = defaultdict(Counter)
    details = []
    replacements = []
    checks = Counter()
    for p in r.rows(c.BASE / "prompts.jsonl"):
        key, cond = p["key"], p["condition"]
        m = meta[key, cond]
        expected = (
            set()
            if cond == "reference"
            else {source[i]["id"] for i in pools[cond]["ids"]}
        )

        scores = defaultdict(float)
        for ch in ["dense", "bm25"]:
            hits = p["channels"][ch]
            r.require(len({v["id"] for v in hits}) == len(hits), "Repeated channel ID")
            for i, hit in enumerate(hits):
                scores[hit["id"]] += 1 / (61 + i)
        order = sorted(scores, key=lambda i: (-scores[i], i))[:5]
        actual = [
            v["id"] for label, group in p["context"]["example_groups"] for v in group
        ]
        r.require(order == actual == m["example_ids"], "Independent RRF order differs")
        found = [v for v in order if v in expected]
        r.require(
            found == m["inserted_ids"] == p["inserted_ids_delivered"],
            "Independent delivered IDs differ",
        )
        r.require(
            all(v in original or v in expected for v in order),
            "Unknown example delivered",
        )
        flags = {
            name: [v["id"] for v in hits if v["id"] in expected]
            for name, hits in [
                ("dense30", p["dense_candidates"]),
                ("mmr10", p["channels"]["dense"]),
                ("bm2510", p["channels"]["bm25"]),
            ]
        }
        r.require(
            len(flags["dense30"]) == m["dense_candidate_insertions"]
            and len(flags["mmr10"]) == m["dense_mmr_insertions"]
            and len(flags["bm2510"]) == m["bm25_insertions"],
            "Independent stage counts differ",
        )
        ref = refs[key]
        equal = p["messages"] == ref["messages"]
        r.require(
            equal == m["message_equals_reference"], "Exact message equality differs"
        )
        checks["rrf_reconstructions"] += 1
        checks["direct_message_comparisons"] += 1
        if cond == "reference":
            continue
        policy = cond.split("_s202610")[0]
        v = counts[policy]
        v["pool_input_combinations"] += 1
        for name, values in flags.items():
            v["with_" + name] += bool(values)
        v["with_either_top10"] += bool(flags["mmr10"] or flags["bm2510"])
        v["with_final5"] += bool(found)
        v["identical_messages"] += equal
        v["no_delivery_changed"] += not found and not equal
        v["delivered_records"] += len(found)
        v["benign_inputs"] += not roster[key]["gold"]["hate"]
        v["benign_inputs_with_delivery"] += not roster[key]["gold"]["hate"] and bool(
            found
        )
        if not equal and not found:
            channel_flags = {
                name: [x["id"] for x in values] != [x["id"] for x in other]
                for name, values, other in [
                    ("dense30", p["dense_candidates"], ref["dense_candidates"]),
                    ("mmr10", p["channels"]["dense"], ref["channels"]["dense"]),
                    ("bm2510", p["channels"]["bm25"], ref["channels"]["bm25"]),
                ]
            }
            replacements.append(
                dict(
                    key=key,
                    condition=cond,
                    channel_order_changed=channel_flags,
                    removed=m["old_examples_removed"],
                    new_existing=m["new_existing_examples"],
                    order_only=set(order) == set(meta[key, "reference"]["example_ids"]),
                )
            )
        if found:
            channel_ranks = {
                i: {
                    ch: next(
                        (
                            j + 1
                            for j, x in enumerate(p["channels"][ch])
                            if x["id"] == i
                        ),
                        None,
                    )
                    for ch in ["dense", "bm25"]
                }
                for i in found
            }
            rr = {
                phase: [rows[key, cc, rep]["result"] for rep in range(3)]
                for phase, cc in [("reference", "reference"), ("edited", cond)]
            }
            details.append(
                dict(
                    key=key,
                    condition=cond,
                    policy=policy,
                    gold=roster[key]["gold"],
                    source_ids=m["source_ids"],
                    gates=m["gates"],
                    final_ranks=m["ranks"],
                    channel_ranks=channel_ranks,
                    cosines=m["cosines"],
                    reference_gates=[
                        None if q is None else q["hate"] for q in rr["reference"]
                    ],
                    edited_gates=[
                        None if q is None else q["hate"] for q in rr["edited"]
                    ],
                    reference_types=[
                        None if q is None else q["hate_type"] for q in rr["reference"]
                    ],
                    edited_types=[
                        None if q is None else q["hate_type"] for q in rr["edited"]
                    ],
                    messages_sha256=p["messages_sha256"],
                    tokens=m["input_tokens"],
                    removed=m["old_examples_removed"],
                )
            )

    inventories = {}
    for cond, pool in pools.items():
        path = c.BASE / "retrieval" / cond / "records.jsonl"
        n = 0
        added = {}
        full_ids = set()
        for rec in r.rows(path):
            n += 1
            if rec["id"] in original:
                full_ids.add(rec["id"])
                r.require(rec == original[rec["id"]], "Original corpus entry changed")
            else:
                added[rec["id"]] = rec
        expected = {source[i]["id"]: c.ordinary(source[i]) for i in pool["ids"]}
        r.require(
            full_ids == set(original) and added == expected and n == len(original) + 40,
            "Actual inserted corpus differs",
        )
        inventories[cond] = dict(
            total=n,
            original_records=len(full_ids),
            inserted=len(added),
            sha256=r.sha(path),
        )
        print(cond + ": all original entries and 40 insertions verified.", flush=True)
    r.require(
        checks["rrf_reconstructions"] == 12825
        and len(details) == sum(v["with_final5"] for v in counts.values()),
        "Coverage differs",
    )
    summary = r.read(r.REVIEW / "summary.json")
    terminal = Counter()
    invalid_values = Counter()
    nulls = defaultdict(Counter)
    null_details = []
    exact_repeats = Counter()
    allowed = {
        "hate_type": set(r.LABELS),
        "target_group": {
            "race",
            "religion",
            "gender",
            "sexual_orientation",
            "disability",
            "national_origin",
            "other",
        },
        "legal": {"propaganda", "incitement_public_peace", "insult_defamation"},
        "severity": {"low", "medium", "high"},
    }

    for item in summary["terminal_errors"]:
        raw = r.read(r.ROOT / item["source"])
        record = rows[item["key"], item["condition"], item["repeat"]]
        r.require(
            r.sha(r.ROOT / item["source"]) == record["source_sha256"],
            "Terminal result changed",
        )
        for failure in item["terminal_errors"]:
            terminal[failure["error"]] += 1
            turns = raw["raw_runs"][failure["vote"]]
            exact_repeats["terminal_votes"] += 1
            exact_repeats["last_repair_byte_identical_to_previous"] += (
                len(turns) > 1 and turns[-1] == turns[-2]
            )
            try:
                value = json.loads(turns[-1])
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(value, dict):
                continue
            for field, labels in allowed.items():
                vals = value.get(field)
                vals = vals if isinstance(vals, list) else [vals]
                for token in vals:
                    if (
                        token is not None
                        and isinstance(token, str)
                        and token not in labels
                    ):
                        invalid_values[field + "::" + token] += 1
        if item["null"]:
            k = item["key"]
            nulls[k]["whole_nulls"] += 1
            for failure in item["terminal_errors"]:
                err = failure["error"]
                nulls[k][err] += 1
            null_details.append(
                dict(
                    key=k,
                    condition=item["condition"],
                    repeat=item["repeat"],
                    gold=record["gold"],
                    terminal_errors=item["terminal_errors"],
                    source=item["source"],
                )
            )
    r.require(
        sum(terminal.values()) == 566 and len(null_details) == 36,
        "Terminal failure count differs",
    )

    flags = []
    versus_ref = Counter()
    comparisons = Counter()
    for scope in ["full", "exclude_replay"]:
        for pop in ["benign", "hateful", "types", "binary"]:
            for path in sorted((r.DEST / scope / pop).glob("*.json")):
                value = r.read(path)
                for endpoint, obj in (value.get("endpoints") or {pop: value}).items():
                    for name, q in obj.get("contrasts", {}).items():
                        lo, hi = q["ci95"]
                        comparisons["directed_contrasts"] += 1
                        if name.endswith("_minus_reference"):
                            versus_ref["total"] += 1
                        if lo > 0 or hi < 0:
                            flags.append(
                                dict(
                                    path=r.rel(path),
                                    endpoint=endpoint,
                                    contrast=name,
                                    difference=q["difference"],
                                    ci95=[lo, hi],
                                )
                            )
                            if name.endswith("_minus_reference"):
                                versus_ref["excluding_zero"] += 1
    output = dict(
        status="independent_prompt_mechanism_and_failure_checks_passed",
        checks=dict(checks),
        original_inventory=dict(
            records=len(original),
            english_examples=sum(
                x["meta"].get("kind") == "example" and x["meta"].get("lang") == "en"
                for x in original.values()
            ),
        ),
        pool_inventories=inventories,
        policies={k: dict(v) for k, v in counts.items()},
        totals=dict(sum(counts.values(), Counter())),
        exposure_cases=details,
        no_delivery_changed_cases=replacements,
        failed_outputs=dict(
            terminal_failure_reasons=dict(terminal),
            invalid_last_reply_tokens=dict(invalid_values),
            repeated_last_replies=dict(exact_repeats),
            null_inputs={k: dict(v) for k, v in nulls.items()},
            null_jobs=null_details,
        ),
        supporting_interval_scan=dict(
            counts=dict(comparisons),
            policy_versus_reference=dict(versus_ref),
            flags=flags,
            caveat="Supporting 95% intervals, many dependent contrasts, scopes and masks. These flags are exploratory and not multiplicity-controlled; they do not replace the co-primary null findings.",
        ),
        scope="Ordinary dense/BM25 compete in parallel; do not describe BM25 as a sequential stage after MMR. Condition/pool/input counts are dependent. RRF checked from saved traces without regenerating approximate retrieval. Terminal JSON scanning never repairs scientific predictions.",
        new_model_calls=0,
    )
    r.save(OUT / "results.json", output)
    r.verify(inputs)
    outputs = {r.rel(OUT / "results.json"): r.sha(OUT / "results.json")}
    r.save(
        target,
        dict(
            status="passed",
            created_utc=r.utc(),
            inputs=inputs,
            outputs=outputs,
            new_model_calls=0,
        ),
    )
    print(
        json.dumps(
            dict(
                checks=dict(checks),
                totals=output["totals"],
                versus_reference=dict(versus_ref),
                exploratory_flags=len(flags),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", r.forbid), patch.object(
        socket, "create_connection", r.forbid
    ):
        main()
