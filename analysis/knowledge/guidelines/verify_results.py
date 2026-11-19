"""Check saved guideline results against their original records."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
import time
from unittest.mock import patch
import numpy as np
from analysis.knowledge.guidelines import run as b

a = b.a
ROOT = b.ROOT
DEST = b.STUDY / "analysis_results_v1"
OUT = ROOT / "Thesis_Log/astra_result/evidence/sq1_guideline_final_verification_20260926_v1.json"


def forbidden(*args, **kwargs):
    raise AssertionError("Offline independent verification only")


def equal(actual, expected, path="root"):
    if isinstance(expected, dict):
        a.require(set(actual) == set(expected), "Key mismatch: " + path)
        for key in expected:
            equal(actual[key], expected[key], path + "/" + str(key))
    elif isinstance(expected, (list, tuple)):
        a.require(len(actual) == len(expected), "Length mismatch: " + path)
        for i, (left, right) in enumerate(zip(actual, expected)):
            equal(left, right, path + "/" + str(i))
    elif isinstance(expected, float):
        a.require(
            isinstance(actual, (float, int))
            and math.isclose(actual, expected, rel_tol=0, abs_tol=2e-12),
            "Numeric mismatch: " + path,
        )
    else:
        a.require(actual == expected, "Value mismatch: " + path)


def verify(hashes):
    for name, digest in hashes.items():
        a.require(a.digest(ROOT / name) == digest, "Changed evidence/source: " + name)


def distribution(values):
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def independent_interval(samples, confidence):
    ordered = sorted((float(x) for x in samples))
    out = []
    for probability in ((1 - confidence) / 2, 1 - (1 - confidence) / 2):
        position = (len(ordered) - 1) * probability
        low = math.floor(position)
        high = math.ceil(position)
        out.append(ordered[low] + (position - low) * (ordered[high] - ordered[low]))
    return out


def main():
    tick = time.perf_counter()
    a.require(
        not OUT.exists() and (not (DEST / "independent_check.json").exists()),
        "Preserve completed verification",
    )
    b.validate()
    receipt_path = DEST / "complete.json"
    receipt = a.read(receipt_path)
    a.require(
        receipt["status"] == "complete_roster_scored_independent_verification_pending",
        "Wrong scoring stage",
    )
    verify(receipt["published_hashes"])
    verify(receipt["verified_bindings"])
    evidence_path = ROOT / receipt["evidence"]
    evidence = a.read(evidence_path)
    a.require(a.digest(evidence_path) == receipt["evidence_sha256"], "Evidence binding differs")
    analysis = a.read(b.BASE / "analysis_plan.json")
    schedule = a.read(b.BASE / "schedule.json")
    inputs = b.core.rows(b.preparation.INVENTORY / "inputs.jsonl")
    bykey = {x["key"]: x for x in inputs}
    conditions = analysis["conditions"]
    key_to_index = {x["key"]: i for i, x in enumerate(inputs)}
    matrix = a.read(DEST / "gate_matrix.json")
    expected_meta = [
        {k: x[k] for k in ("key", "item_id", "templ_id", "functionality", "text_sha256", "gate")}
        for x in inputs
    ]
    equal(matrix["items"], expected_meta)
    equal(matrix["conditions"], conditions)
    equal(matrix["repeats"], [0, 1, 2])
    a.require(
        matrix["missing_value"] == -1 and all((x["gate"] is False for x in inputs)),
        "Gold/missing convention differs",
    )
    exported = b.core.rows(DEST / "predictions.jsonl")
    jobs = {
        (j["key"], j["condition"], j["repeat"]): (s["id"], j)
        for s in schedule["stages"]
        for j in s["jobs"]
    }
    a.require(len(exported) == len(jobs) == 10395, "Exported roster differs")
    output_hashes = a.read(DEST / "source_output_hashes.json")
    values = np.full((693, 5, 3), -2, dtype=np.int8)
    seen = set()
    operations = {c: Counter() for c in conditions}
    stage_audits = {}
    for i in range(1, 5):
        review = a.read(b.BASE / "stage_reviews" / f"stage_{i:02d}.json")
        stage_audits[f"stage_{i:02d}"] = a.read(ROOT / review["evidence"])
    inspected_hashes = {}
    recomputed_nulls = []
    independent_gates = 0
    for number, row in enumerate(exported, 1):
        identity = (row["key"], row["condition"], row["repeat"])
        a.require(identity in jobs and identity not in seen, "Wrong/duplicate exported identity")
        seen.add(identity)
        stage, job = jobs[identity]
        meta = bykey[row["key"]]
        folder = b.item_dir(b.BASE / "live", job, receipt["plan_sha256"])
        path = folder / "result.json"
        completion = folder / "complete.json"
        audit = stage_audits[stage]
        for p in (path, completion):
            data = a.storage_path(p).read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            rel = b.rel(p)
            a.require(
                sha == output_hashes[rel] == audit["verified_output_hashes"][rel],
                "Source output binding differs",
            )
            inspected_hashes[rel] = sha
            if p == path:
                raw = json.loads(data)
        valid = [vote for vote in raw["run_predictions"] if vote is not None]
        expected_gate = None if not valid else sum((x["hate"] for x in valid)) > len(valid) / 2
        expected_tie = bool(valid) and sum((x["hate"] for x in valid)) == len(valid) / 2
        a.require(
            raw["result"] is None
            and expected_gate is None
            or (raw["result"] is not None and raw["result"]["hate"] == expected_gate),
            "Independent gate vote differs",
        )
        a.require(
            raw["n_valid"] == len(valid) and raw["uncertain"] == expected_tie,
            "Independent validity/tie differs",
        )
        expected = {
            "key": job["key"],
            "item_id": meta["item_id"],
            "condition": job["condition"],
            "repeat": job["repeat"],
            "functionality": meta["functionality"],
            "templ_id": meta["templ_id"],
            "hate": expected_gate,
            "n_valid": raw["n_valid"],
            "gate_tie": raw["uncertain"],
            "repairs": raw["repairs"],
            "parse_failures": raw["parse_failures"],
            "gate_normalised": raw["gate_normalised"],
            "stage": stage,
            "messages_sha256": job["messages_sha256"],
            "source_result": b.rel(path),
            "source_sha256": inspected_hashes[b.rel(path)],
        }
        equal(row, expected)
        ii = key_to_index[row["key"]]
        cc = conditions.index(row["condition"])
        rr = row["repeat"]
        values[ii, cc, rr] = -1 if expected_gate is None else int(expected_gate)
        if expected_gate is None:
            recomputed_nulls.append(expected)
        op = operations[row["condition"]]
        for field in ("repairs", "parse_failures", "gate_normalised", "normalisations"):
            op[field] += raw[field]
        op["items"] += 1
        op["nulls"] += expected_gate is None
        op["gate_ties"] += expected_tie
        op["valid_votes_" + str(len(valid))] += 1
        op["items_with_repairs"] += raw["repairs"] > 0
        for field, value in raw["accounting"].items():
            op["accounting_" + field] += value
        independent_gates += 1
        if number % 2500 == 0:
            print(f"Independently source-checked {number}/10395 predictions", flush=True)
    equal(values.tolist(), matrix["values"])
    equal(inspected_hashes, output_hashes)
    equal({c: dict(x) for c, x in operations.items()}, evidence["operations_by_condition"])
    equal(recomputed_nulls, evidence["null_predictions"])
    common_indices = [
        i
        for i in range(len(inputs))
        if all((values[i, c, r] != -1 for c in range(5) for r in range(3)))
    ]
    excluded = [
        {k: inputs[i][k] for k in ("key", "item_id", "templ_id", "functionality")}
        for i in range(len(inputs))
        if i not in common_indices
    ]
    equal(excluded, evidence["excluded_from_common"])
    scope_names = ["all", *sorted({x["functionality"] for x in inputs})]
    a.require(set(evidence["scopes"]) == set(scope_names), "Scope set differs")
    contrasts = {**analysis["primary_contrasts"], **analysis["secondary_contrasts"]}
    rng = np.random.default_rng(analysis["bootstrap_seed"])
    arrays = np.load(DEST / "bootstrap_arrays.npz", allow_pickle=False)
    verified_draws = 0
    max_error = 0.0
    setting_checks = 0
    contrast_checks = 0
    for name in scope_names:
        indices = [i for i, x in enumerate(inputs) if name == "all" or x["functionality"] == name]
        chosen = [i for i in indices if i in common_indices]
        common_templates = sorted({inputs[i]["templ_id"] for i in chosen})
        roster_templates = sorted({inputs[i]["templ_id"] for i in indices})
        scope = evidence["scopes"][name]
        for key, expected in {
            "scope": name,
            "roster_items": len(indices),
            "common_items": len(chosen),
            "roster_keys": [inputs[i]["key"] for i in indices],
            "common_keys": [inputs[i]["key"] for i in chosen],
            "roster_templates": roster_templates,
            "common_templates": common_templates,
            "excluded_keys": [inputs[i]["key"] for i in indices if i not in chosen],
            "templates_lost": [x for x in roster_templates if x not in common_templates],
        }.items():
            equal(scope[key], expected)
        summaries = {}
        expected_settings = []
        for cc, condition in enumerate(conditions):
            shared = []
            own = []
            lows = []
            highs = []
            for rr in range(3):
                known = [int(values[i, cc, rr]) for i in indices if values[i, cc, rr] != -1]
                fp = sum(known)
                null = len(indices) - len(known)
                common_fp = sum((int(values[i, cc, rr]) for i in chosen))
                rate = fp / len(known) if known else None
                cr = common_fp / len(chosen) if chosen else None
                low = fp / len(indices)
                high = (fp + null) / len(indices)
                expected_settings.append(
                    {
                        "condition": condition,
                        "repeat": rr,
                        "roster_items": len(indices),
                        "valid": len(known),
                        "null": null,
                        "false_positives": fp,
                        "own_valid_rate": rate,
                        "common_items": len(chosen),
                        "common_false_positives": common_fp,
                        "common_rate": cr,
                        "full_roster_bounds": [low, high],
                    }
                )
                shared.append(cr)
                own.append(rate)
                lows.append(low)
                highs.append(high)
                setting_checks += 1
            summaries[condition] = {
                "common": distribution(shared) if chosen else None,
                "own_valid": distribution(own) if all((x is not None for x in own)) else None,
                "full_roster_mean_bounds": [statistics.mean(lows), statistics.mean(highs)],
            }
        equal(scope["settings"], expected_settings)
        equal(scope["conditions"], summaries)
        if not chosen:
            a.require(
                scope["status"] == "unscorable_no_common_items" and (not scope["contrasts"]),
                "Empty scope scored",
            )
            continue
        draws = arrays[name + "__draws"]
        saved = arrays[name + "__mean_rates"]
        expected_draws = rng.integers(
            0, len(common_templates), size=(10000, len(common_templates)), dtype=np.int32
        )
        a.require(np.array_equal(draws, expected_draws), "Saved random draws differ")
        members = [
            np.array([i for i in chosen if inputs[i]["templ_id"] == tid])
            for tid in common_templates
        ]
        expanded_rates = np.empty((len(draws), len(conditions)))
        for j, draw in enumerate(draws):
            selected = np.concatenate([members[int(cluster)] for cluster in draw])
            expanded_rates[j] = values[selected].mean(axis=(0, 2))
        error = float(np.max(np.abs(expanded_rates - saved)))
        max_error = max(max_error, error)
        a.require(error < 2e-15, "Expanded bootstrap differs from grouped calculation")
        verified_draws += len(draws)
        equal(
            scope["bootstrap_mean_rate_descriptive_95"],
            {c: independent_interval(expanded_rates[:, i], 0.95) for i, c in enumerate(conditions)},
        )
        a.require(set(scope["contrasts"]) == set(contrasts), "Contrast set differs")
        for contrast, weights in contrasts.items():
            left = next((c for c, w in weights.items() if w == 1))
            right = next((c for c, w in weights.items() if w == -1))
            ls = summaries[left]
            rs = summaries[right]
            samples = (
                expanded_rates[:, conditions.index(left)]
                - expanded_rates[:, conditions.index(right)]
            )
            family = (
                independent_interval(samples, 0.9875)
                if name == "all" and contrast in analysis["primary_contrasts"]
                else None
            )
            expected = {
                "left": left,
                "right": right,
                "mean_difference": ls["common"]["mean"] - rs["common"]["mean"],
                "per_repeat_difference": [
                    x - y for x, y in zip(ls["common"]["values"], rs["common"]["values"])
                ],
                "descriptive_95_interval": independent_interval(samples, 0.95),
                "primary_family_interval": family,
                "primary_family_confidence": 0.9875 if family is not None else None,
                "family_interval_excludes_zero": (
                    family[1] < 0 or family[0] > 0 if family is not None else None
                ),
                "full_roster_missing_output_difference_bounds": [
                    ls["full_roster_mean_bounds"][0] - rs["full_roster_mean_bounds"][1],
                    ls["full_roster_mean_bounds"][1] - rs["full_roster_mean_bounds"][0],
                ],
            }
            equal(scope["contrasts"][contrast], expected)
            contrast_checks += 1
        print("Expanded-template verification passed: " + name, flush=True)
    a.require(len(arrays.files) == 2 * len(scope_names), "Unexpected bootstrap arrays")
    totals = Counter()
    account = Counter()
    usage = Counter()
    for audit in stage_audits.values():
        totals.update(audit["counts"])
        account.update(audit["accounting"])
        usage.update(audit["observed_usage_sums"])
    equal(evidence["execution_counts"], dict(totals))
    equal(evidence["accounting"], dict(account))
    equal(evidence["observed_usage"], dict(usage))
    delivery = a.read(DEST / "prompt_delivery.json")
    prompt_rows = b.core.rows(ROOT / a.read(b.PLAN)["prompts"])
    cat = {(row["key"], row["condition"]): row for row in prompt_rows}
    a.require(
        len(cat) == len(prompt_rows) == 3465 and all(("context" in row for row in prompt_rows)),
        "Full frozen contexts required",
    )
    languages = Counter()
    lead_count = 0
    filler_count = 0
    lengths = {c: [] for c in conditions}
    for row, meta in zip(delivery["items"], inputs):
        key = meta["key"]
        natural = cat[key, "natural"]["context"]["guidelines"]
        matched = cat[key, "matched_rule"]["context"]["guidelines"]
        alternative = cat[key, "alternative_rule"]["context"]["guidelines"]
        nids = [x["id"] for x in natural]
        langs = [x["meta"].get("lang") for x in natural]
        expected = {
            "key": key,
            "functionality": meta["functionality"],
            "matched_lead": matched[0]["id"],
            "filler": matched[1]["id"],
            "alternative_lead": alternative[0]["id"],
            "natural_ids": nids,
            "natural_languages": langs,
            "matched_lead_retrieved": matched[0]["id"] in nids,
            "filler_retrieved": matched[1]["id"] in nids,
        }
        equal(row, expected)
        languages.update((str(x) for x in langs))
        lead_count += expected["matched_lead_retrieved"]
        filler_count += expected["filler_retrieved"]
        for c in conditions:
            lengths[c].append(sum((len(x["text"]) for x in cat[key, c]["messages"])))
    a.require(len(delivery["items"]) == 693, "Delivery roster length differs")
    equal(delivery["natural_language_slots"], dict(languages))
    equal(delivery["matched_lead_delivered"], lead_count)
    equal(delivery["filler_delivered"], filler_count)
    for function, summary in delivery["by_functionality"].items():
        subset = [row for row in delivery["items"] if row["functionality"] == function]
        equal(
            summary,
            {
                "inputs": len(subset),
                "matched_lead_delivered": sum((x["matched_lead_retrieved"] for x in subset)),
                "filler_delivered": sum((x["filler_retrieved"] for x in subset)),
            },
        )
    equal(
        delivery["message_character_lengths"],
        {
            c: {
                "minimum": min(v),
                "maximum": max(v),
                "mean": statistics.mean(v),
                "median": statistics.median(v),
            }
            for c, v in lengths.items()
        },
    )
    equal(evidence["prompt_delivery_summary"], {k: v for k, v in delivery.items() if k != "items"})
    verify(receipt["published_hashes"])
    verify(receipt["verified_bindings"])
    record = {
        "created_utc": a.utc(),
        "status": "guideline_all_results_independently_verified",
        "plan_sha256": receipt["plan_sha256"],
        "main_evidence": b.rel(evidence_path),
        "main_evidence_sha256": a.digest(evidence_path),
        "source_sha256": a.digest(Path(__file__)),
        "verified_source_output_hashes": len(inspected_hashes),
        "independent_gate_votes": independent_gates,
        "independent_setting_count_rate_bound_checks": setting_checks,
        "independent_contrast_checks": contrast_checks,
        "independent_expanded_bootstrap_draws": verified_draws,
        "maximum_bootstrap_absolute_difference": max_error,
        "common_items": len(common_indices),
        "null_outputs": len(recomputed_nulls),
        "all_exported_rows_source_verified": True,
        "verified_bindings": receipt["verified_bindings"],
        "verified_published_hashes": receipt["published_hashes"],
        "elapsed_seconds": time.perf_counter() - tick,
        "actual_API_calls": 0,
        "new_CUDA_training_retrieval": False,
    }
    a.save_new(OUT, record)
    a.save_new(
        DEST / "independent_check.json",
        {
            "created_utc": a.utc(),
            "status": record["status"],
            "evidence": b.rel(OUT),
            "evidence_sha256": a.digest(OUT),
            "main_evidence_sha256": a.digest(evidence_path),
            "source_sha256": record["source_sha256"],
            "actual_API_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k not in ("verified_bindings", "verified_published_hashes")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        main()
