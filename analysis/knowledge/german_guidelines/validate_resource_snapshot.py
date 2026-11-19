from collections import Counter
from pathlib import Path
import argparse
import json
import math
import numpy as np
from analysis.knowledge.german_guidelines import resources as x

c = x.c


def same(a, b, label):
    if isinstance(b, dict):
        x.require(isinstance(a, dict) and set(a) == set(b), label + " keys")
        for k in b:
            same(a[k], b[k], label + "/" + k)
    elif isinstance(b, (list, tuple)):
        x.require(len(a) == len(b), label + " length")
        for i, (aa, bb) in enumerate(zip(a, b)):
            same(aa, bb, label + "/" + str(i))
    elif b is None or isinstance(b, (str, bool)):
        x.require(a == b, label + " value")
    else:
        x.require(
            math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-07),
            label + " numeric",
        )


def ref_stats(values):
    z = sorted(map(float, values))
    n = len(z)

    def quant(q):
        i = (n - 1) * q
        a = int(i)
        return z[a] + (i - a) * (z[min(a + 1, n - 1)] - z[a])

    return {
        "n": n,
        "sum": math.fsum(z),
        "mean": math.fsum(z) / n if n else None,
        "min": z[0] if n else None,
        "median": quant(0.5) if n else None,
        "p90": quant(0.9) if n else None,
        "p95": quant(0.95) if n else None,
        "p99": quant(0.99) if n else None,
        "max": z[-1] if n else None,
    }


def verify(out):
    manifest = c.read(out / "manifest.json")
    c.verify(manifest["outputs"])
    c.verify(c.read(x.BASE / "analysis_contract.json")["bindings"])
    _, rp = x.reader.read_verified()
    meta = list(c.rows(out / "prediction_resources.jsonl"))
    with np.load(out / "event_resources.npz", allow_pickle=False) as f:
        events = f["events"]
    at = 0
    allcounts = Counter()
    for i, row in enumerate(c.rows(rp / "predictions.jsonl")):
        m = meta[i]
        x.require(
            all(
                (
                    m[k] == row[k]
                    for k in (
                        "phase",
                        "condition",
                        "key",
                        "repeat",
                        "stage",
                        "source",
                        "n_valid",
                        "repairs",
                    )
                )
            ),
            "Prediction metadata mismatch",
        )
        vote_seconds = [0.0, 0.0, 0.0]
        for event in row["events"]:
            ok = event["action"] == "ok"
            u = event["model_response_metadata"].get("usage") if ok else None
            tokens = [
                u.get(k, -1) if u and u.get(k) is not None else -1
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            ]
            expected = [
                i,
                int(event["kind"] == "repair"),
                int(ok),
                event["attempt"],
                event["elapsed_s"],
                event["backoff_elapsed_s"],
                *tokens,
            ]
            same(events[at].tolist(), expected, "saved event")
            at += 1
            vote_seconds[event["vote"]] += (
                event["elapsed_s"] + event["backoff_elapsed_s"]
            )
        same(m["vote_chain_s"], vote_seconds, "vote chain")
        same(m["request_chain_s"], sum(vote_seconds), "prediction chain")
        allcounts.update(row["accounting"])
    x.require(
        at == len(events) and i + 1 == len(meta) == 33093, "Incomplete resource matrix"
    )
    resources = c.read(out / "resources.json")
    for name, saved in resources["groups"].items():
        if name == "all":
            indices = list(range(len(meta)))
        elif name.startswith("condition/"):
            indices = [
                i for i, v in enumerate(meta) if v["condition"] == name.split("/")[1]
            ]
        else:
            ph, co = name.split("/")
            indices = [
                i
                for i, v in enumerate(meta)
                if v["phase"] == ph and v["condition"] == co
            ]
        chosen = events[np.isin(events[:, 0].astype(int), indices)]
        success = chosen[:, 2] == 1
        expected = {
            "predictions": len(indices),
            "nulls": sum((meta[i]["n_valid"] == 0 for i in indices)),
            "repaired_predictions": sum((meta[i]["repairs"] > 0 for i in indices)),
            "ordinary_logical_calls": 3 * len(indices),
            "repair_logical_calls": sum((meta[i]["repairs"] for i in indices)),
            "physical_POST_attempts": len(chosen),
            "successful_responses": int(success.sum()),
            "failed_attempts": int((~success).sum()),
            "retry_attempts": int((chosen[:, 3] > 0).sum()),
            "backoff_seconds": math.fsum(chosen[:, 5]),
            "all_transport_seconds": math.fsum(chosen[:, 4]),
            "successful_latency_s": ref_stats(chosen[success, 4]),
            "ordinary_latency_s": ref_stats(chosen[success & (chosen[:, 1] == 0), 4]),
            "repair_latency_s": ref_stats(chosen[success & (chosen[:, 1] == 1), 4]),
            "failed_attempt_latency_s": ref_stats(chosen[~success, 4]),
            "prediction_request_chain_s": ref_stats(
                [meta[i]["request_chain_s"] for i in indices]
            ),
            "vote_request_chain_s": ref_stats(
                [t for i in indices for t in meta[i]["vote_chain_s"]]
            ),
            "usage": {},
        }
        for kind, mask in [
            ("all_success", success),
            ("ordinary", success & (chosen[:, 1] == 0)),
            ("repair", success & (chosen[:, 1] == 1)),
        ]:
            ev = chosen[mask]
            expected["usage"][kind] = {"responses": len(ev)}
            for field, col in [
                ("prompt_tokens", 6),
                ("completion_tokens", 7),
                ("total_tokens", 8),
            ]:
                vals = [v for v in ev[:, col] if v >= 0]
                expected["usage"][kind][field] = {
                    "known_responses": len(vals),
                    "unknown_responses": len(ev) - len(vals),
                    **ref_stats(vals),
                }
        same(saved, expected, name)
    live = c.read(c.OUTPUT / "live/complete.json")
    same(
        dict(allcounts),
        {
            k: v
            for k, v in live["counts"].items()
            if k not in ("predictions", "null_results")
        },
        "live ledger",
    )
    pt = list(c.rows(out / "prompt_resources.jsonl"))
    saved = c.read(out / "prompt_tokens.json")
    for phase, conditions in saved["conditions"].items():
        for co, z in conditions.items():
            rr = [
                v
                for v in pt
                if v["condition"] == co and (phase == "all" or v["phase"] == phase)
            ]
            known = [v for v in rr if v["initial_server_prompt_tokens"] is not None]
            same(
                z,
                {
                    "prompts": len(rr),
                    "stable_complete_server_usage": len(known),
                    "reference_tokens": ref_stats([v["reference_tokens"] for v in rr]),
                    "server_prompt_tokens": ref_stats(
                        [v["initial_server_prompt_tokens"] for v in known]
                    ),
                    "server_minus_reference": ref_stats(
                        [
                            v["initial_server_prompt_tokens"] - v["reference_tokens"]
                            for v in known
                        ]
                    ),
                },
                "prompt statistics",
            )
    ptmap = {(v["key"], v["condition"]): v for v in pt}
    for phase, pairs in saved["paired"].items():
        for pair, z in pairs.items():
            left, right = pair.split("_minus_")
            rr = [
                v
                for v in pt
                if v["condition"] == left and (phase == "all" or v["phase"] == phase)
            ]
            pairs_known = [
                (v, ptmap[v["key"], right])
                for v in rr
                if v["initial_server_prompt_tokens"] is not None
                and ptmap[v["key"], right]["initial_server_prompt_tokens"] is not None
            ]
            ds = [
                a["initial_server_prompt_tokens"] - b["initial_server_prompt_tokens"]
                for a, b in pairs_known
            ]
            same(
                z,
                {
                    "matched_inputs": len(ds),
                    "token_difference": ref_stats(ds),
                    "relative_mean_difference": (
                        sum(ds)
                        / sum(
                            (b["initial_server_prompt_tokens"] for a, b in pairs_known)
                        )
                        if pairs_known
                        else None
                    ),
                    "unchanged_inputs": sum((d == 0 for d in ds)),
                    "shorter_inputs": sum((d < 0 for d in ds)),
                    "longer_inputs": sum((d > 0 for d in ds)),
                },
                "paired prompt tokens",
            )
    timeline = c.read(out / "timing.json")
    c.verify(timeline["receipt_bindings"])
    stages = timeline["stages"]
    wall = timeline["wall"]
    starts = [x.reader.timestamp(v["start_utc"]) for v in stages]
    ends = [x.reader.timestamp(v["finish_utc"]) for v in stages]
    active = sum(((b - a).total_seconds() for a, b in zip(starts, ends)))
    gaps = sum(
        ((starts[j + 1] - ends[j]).total_seconds() for j in range(len(stages) - 1))
    )
    same(wall["attempt_seconds"], active, "active wall")
    same(wall["between_attempt_seconds"], gaps, "between attempts")
    same(wall["first_to_last_attempt_seconds"], active + gaps, "span")
    for st in stages:
        indices = [i for i, v in enumerate(meta) if v["stage"] == st["stage"]]
        ev = events[np.isin(events[:, 0].astype(int), indices)]
        same(st["predictions"], len(indices), "stage predictions")
        same(st["POST_attempts"], len(ev), "stage POST")
        same(st["request_seconds"], math.fsum(ev[:, 4]), "stage request time")
        x.require(
            st["workers"] == st["peak_submitted"] == 128
            and (not st["stopped"])
            and (not st["errors"]),
            "Unexpected stage outcome",
        )
    storage = c.read(out / "storage.json")
    for side in ("D_study", "C_preparation"):
        z = storage[side]
        same(
            sum((v["files"] for v in z["groups"].values())),
            z["files"],
            "group file count",
        )
        same(
            sum((v["bytes"] for v in z["groups"].values())),
            z["logical_bytes"],
            "group file bytes",
        )
        same(
            sum((v["bytes"] for v in z["by_basename_or_suffix"].values())),
            z["logical_bytes"],
            "filetype bytes",
        )
    types = storage["D_study"]["by_basename_or_suffix"]
    x.require(
        types["intent.json"]["files"] == types["outcome.json"]["files"] == len(events),
        "Storage POST record count differs",
    )
    x.require(
        types["result.json"]["files"] == 33093
        and types["job.json"]["files"] == 33093
        and (types["complete.json"]["files"] == 33094),
        "Storage prediction record count differs",
    )
    scores = c.read(out / "synthesis_scores.json")
    _, rob = x.r.verified()
    old = c.read(rob / "original_coverage_bounds_repeats.json")
    for phase, v in scores.items():
        same(v, {k: old[phase][k] for k in v}, "Unchanged synthesis scores")
    for pair, v in c.read(out / "practical_tradeoffs.json").items():
        left, right = pair.split("_minus_")
        a = scores["de_test"]["common"][left]["mean"]
        b = scores["de_test"]["common"][right]["mean"]
        same(
            v["equal_cost_error_change"],
            a["fp"] + a["fn"] - b["fp"] - b["fn"],
            "Error cost illustration",
        )
        if v["illustrative_missed_hate_to_false_alarm_break_even"] is not None:
            same(
                v["mean_extra_FP"]
                + v["mean_extra_FN"]
                * v["illustrative_missed_hate_to_false_alarm_break_even"],
                0,
                "Break-even cost",
            )
    c.verify(manifest["outputs"])
    c.verify(c.read(x.BASE / "analysis_contract.json")["bindings"])
    x.save(
        out / "independent_check.json",
        {
            "status": "part7_resource_checks_pass",
            "checked_predictions": len(meta),
            "checked_POST_events": len(events),
            "resource_groups": len(resources["groups"]),
            "prompt_records": len(pt),
            "stages": len(stages),
            "storage_method": "Arithmetic and expected durable-file counts independently reconciled; metadata walk is not a content audit.",
            "actual_Qwen_calls": 0,
            "D_writes": 0,
        },
    )
    x.save(
        x.BASE / "complete.json",
        {
            "status": "part7_resources_verified",
            "created_utc": c.a.utc(),
            "manifest": c.ref(out / "manifest.json"),
            "bindings": c.bindings(
                [
                    out / "manifest.json",
                    out / "independent_check.json",
                    x.BASE / "analysis_contract.json",
                    *x.sources(),
                ]
            ),
            "actual_Qwen_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "Resource checks passed",
                "predictions": len(meta),
                "events": len(events),
                "groups": len(resources["groups"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    with x.offline():
        verify(c.ROOT / args.run)
