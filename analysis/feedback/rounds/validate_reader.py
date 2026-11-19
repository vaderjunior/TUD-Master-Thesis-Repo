from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from analysis.feedback.rounds import reader as io


def final_from_votes(votes):
    valid = [v for v in votes if v is not None]
    if not valid:
        return None, False
    n = len(valid)
    yes = len([v for v in valid if v["hate"] is True])
    result = {"hate": yes > n / 2}
    for field in ["hate_type", "target_group", "legal"]:
        union = sorted(set(x for v in valid for x in v[field]))
        result[field] = [
            x for x in union if sum(v[field].count(x) for v in valid) > n / 2
        ]
    severity = [v["severity"] for v in valid if v["severity"] is not None]
    severity.sort(key=["low", "medium", "high"].index)
    result["severity"] = severity[(len(severity) - 1) // 2] if severity else None
    if result["hate"] is False:
        result["hate_type"], result["target_group"], result["severity"] = [], [], None
    return result, yes == n / 2


def verify():
    started = time.monotonic()
    receipt = io.read(io.READER / "complete.json")
    for name, digest in {**receipt["input_bindings"], **receipt["artifacts"]}.items():
        io.need(
            io.sha(io.ROOT / name) == digest, "Reader input/artifact changed: " + name
        )
    matrix = list(io.rows(io.READER / "matrix.jsonl"))
    roster = list(io.rows(io.READER / "roster.jsonl"))
    io.need(len(matrix) == len(roster) == 2077, "Wrong matrix size")
    by_key = {v["key"]: i for i, v in enumerate(matrix)}
    conditions = receipt["conditions"]
    seen = set()
    counts, accounting, usage = Counter(), Counter(), Counter()
    references = Counter()
    phase_ids = {}
    for stage in receipt["stage_receipts"]:
        path = io.ROOT / stage["receipt"]
        io.need(io.sha(path) == stage["receipt_sha256"], "Stage receipt changed")
        done = io.read(path)
        io.need(
            io.sha(io.ROOT / stage["predictions"]) == done["predictions_sha256"],
            "Assembly changed",
        )
        for name, digest in done["inputs"].items():
            io.need(io.sha(io.ROOT / name) == digest, "Audit binding changed")
        local = Counter()
        for v in io.rows(io.ROOT / stage["predictions"]):
            key = (v["phase"], v["key"], v["condition"], v["repeat"])
            io.need(key not in seen, "Duplicate assembled identity")
            seen.add(key)
            ident = {
                "experiment_sha256": v["phase_plan_sha256"],
                "condition": v["role"] + "::" + v["condition"],
                "key": v["key"],
                "repeat": v["repeat"],
            }
            payload = (
                json.dumps(
                    ident, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2
                )
                + "\n"
            ).encode("utf8")
            item_id = hashlib.sha256(payload).hexdigest()
            io.need(
                v["item_receipt"].endswith("/items/" + item_id + "/complete.json"),
                "Wrong independent identity",
            )
            io.need(
                "/phases/" + v["phase"] + "/" in v["item_receipt"], "Wrong phase path"
            )
            phase_ids[v["phase"]] = v["phase_plan_sha256"]
            final, tie = final_from_votes(v["votes"])
            io.need(
                final == v["result"] and tie == v["uncertain"],
                "Independent vote aggregation differs",
            )
            io.need(
                v["n_valid"] == len([x for x in v["votes"] if x is not None]),
                "Independent vote count differs",
            )
            io.need(
                v["effective_worker_limit"] == 128 and "execution_period" not in v,
                "Wrong execution metadata",
            )
            if v["role"] == "evaluation":
                expected_policy = v["condition"].split("_s")[0]
                expected_stream = (
                    None
                    if v["condition"] == "reference"
                    else int(v["condition"].split("_s")[1])
                )
                io.need(
                    v["policy"] == expected_policy
                    and v["stream"] == expected_stream
                    and v["source_phase_stream"] is None,
                    "Explicit evaluation axes differ",
                )
            else:
                io.need(
                    v["policy"] == v["condition"]
                    and v["stream"] == v["source_phase_stream"],
                    "Source axes differ",
                )
            local["predictions"] += 1
            local["nulls"] += final is None
            local["valid_votes_" + str(v["n_valid"])] += 1
            local["gate_ties"] += tie
            for field in [
                "gate_normalised",
                "normalisations",
                "repairs",
                "parse_failures",
            ]:
                local[field] += v[field]
            accounting.update(v["accounting"])
            usage.update(v["observed_usage"])
            if v["role"] == "evaluation":
                i = by_key[v["key"]]
                m = matrix[i]
                actual = m["predictions"][v["round"] - 1][
                    conditions.index(v["condition"])
                ][v["repeat"]]
                io.need(
                    actual == {k: v[k] for k in ["result", "n_valid", "uncertain"]},
                    "Matrix output differs from assembly",
                )
                io.need(m["population"] == v["population"], "Wrong population join")
                if v["condition"] == "reference":
                    references[v["round"]] += 1
        io.need(
            local == Counter(stage["counts"]) and local["predictions"] == done["rows"],
            "Stage totals differ",
        )
        counts.update(local)
        print(
            "Independent reader check: " + stage["phase"] + "/" + stage["stage"],
            flush=True,
        )
    io.need(
        len(phase_ids) == len(set(phase_ids.values())) == 14,
        "Phase namespaces not distinct",
    )
    io.need(
        counts == Counter(receipt["counts"])
        and accounting == Counter(receipt["accounting"])
        and usage == Counter(receipt["observed_usage"]),
        "Global independent totals differ",
    )
    io.need(
        references == Counter({r: 2077 * 3 for r in range(1, 5)}),
        "Reference was duplicated or lost",
    )
    masks = {
        p: {
            str(g): {
                "total": 0,
                "clusters": set(),
                "round_common": [0] * 4,
                "round_strict": [0] * 4,
                "all_round_common": 0,
                "all_round_strict": 0,
            }
            for g in [False, True]
        }
        for p in ["en_test", "en_hatecheck_eval"]
    }
    replay = set(receipt["replay_keys"])
    for i, (m, r) in enumerate(zip(matrix, roster)):
        io.need(
            m["input_index"] == i
            and m["key"] == r["key"]
            and m["gold_hate"] == r["gate"],
            "Gold/order differs",
        )
        for field in ["hate_types", "target_groups", "severity", "legal"]:
            io.need(
                m["gold_" + field] == r.get(field)
                and m["known_fine_gold"][field] == (r.get(field) is not None),
                "Missing gold semantics differ",
            )
        for field in [
            "inference_cluster",
            "functional_cluster",
            "overlap_cluster",
            "text_cluster",
            "source",
            "population",
        ]:
            io.need(m[field] == r[field], "Metadata differs")
        validity = [
            not any(v["result"] is None for c in rr for v in c)
            for rr in m["predictions"]
        ]
        strict = [
            not any(v["n_valid"] != 3 for c in rr for v in c) for rr in m["predictions"]
        ]
        expected = {
            "round_common": validity,
            "round_strict": strict,
            "all_round_common": all(validity),
            "all_round_strict": all(strict),
            "replay_exclusion_member": m["key"] in replay,
        }
        io.need(m["masks"] == expected, "Independent mask differs")
        mm = masks[m["population"]][str(m["gold_hate"])]
        mm["total"] += 1
        mm["clusters"].add(m["inference_cluster"])
        for rnd in range(4):
            mm["round_common"][rnd] += validity[rnd]
            mm["round_strict"][rnd] += strict[rnd]
        mm["all_round_common"] += all(validity)
        mm["all_round_strict"] += all(strict)
    for pop in masks.values():
        for mm in pop.values():
            mm["clusters"] = len(mm["clusters"])
    io.need(masks == receipt["mask_counts"], "Independent mask totals differ")
    result = {
        "status": "independent_reader_verification_pass",
        "created_utc": io.pure.utc(),
        "reader_complete_sha256": io.sha(io.READER / "complete.json"),
        "counts": dict(counts),
        "independent_vote_reaggregations": len(seen),
        "distinct_phase_namespaces": len(phase_ids),
        "evaluation_slots_compared_to_assembly": 249240,
        "reference_slots_per_round": dict(references),
        "mask_counts": masks,
        "elapsed_seconds": time.monotonic() - started,
        "verifier_sha256": io.sha(Path(__file__)),
        "new_Qwen_calls": 0,
        "new_fits": 0,
        "scientific_effects_scored": False,
    }
    io.write(io.READER / "independent_verification.json", result)
    print(
        json.dumps(
            {k: result[k] for k in ["status", "counts", "elapsed_seconds"]}, indent=2
        )
    )


if __name__ == "__main__":
    with io.pure.offline():
        verify()
