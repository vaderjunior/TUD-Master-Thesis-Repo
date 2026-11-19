"""Guidelines results."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch
import numpy as np
from analysis.knowledge.guidelines import run as b
from analysis.knowledge.guidelines import metrics as m

a = b.a
ROOT = b.ROOT
DEST = b.STUDY / "analysis_results_v1"
EVIDENCE = ROOT / "Thesis_Log/astra_result/evidence/sq1_guideline_final_results_20260926_v1.json"
SOURCE_FILES = [
    x
    for x in (
        "analysis/knowledge/guidelines/metrics.py",
        "tests/knowledge/guidelines/test_metrics.py",
        "analysis/knowledge/guidelines/results.py",
        "analysis/knowledge/guidelines/verify_results.py",
    )
]


def forbidden(*args, **kwargs):
    raise AssertionError("Offline guideline scoring only")


def verify(hashes):
    for name, digest in hashes.items():
        a.require(a.digest(ROOT / name) == digest, "Changed scoring input: " + name)


def checked_read(path, expected):
    data = a.storage_path(path).read_bytes()
    a.require(
        hashlib.sha256(data).hexdigest() == expected, "Changed audited output: " + b.rel(path)
    )
    return json.loads(data)


def run():
    tick = time.perf_counter()
    a.require(
        not (DEST / "complete.json").exists() and (not EVIDENCE.exists()),
        "Preserve completed results",
    )
    a.require(not (DEST / "analysis_contract.json").exists(), "Preserve partial analysis")
    plan = b.validate()
    plan_sha = a.digest(b.PLAN)
    analysis = a.read(b.BASE / "analysis_plan.json")
    a.require(
        analysis["no_partial_scoring"]
        and analysis["bootstrap_draws"] == 10000
        and (analysis["bootstrap_seed"] == 20260925)
        and (analysis["bootstrap_unit"] == "templ_id")
        and (analysis["primary_family_confidence"] == 0.9875)
        and (analysis["repeats"] == [0, 1, 2])
        and (tuple(analysis["conditions"]) == b.CONDITIONS),
        "Analysis contract differs",
    )
    tests_path = DEST / "development/metrics_tests_passed.json"
    tests = a.read(tests_path)
    a.require(
        tests["passed"] is True and tests["tests_run"] == 10, "Passing metrics tests required"
    )
    verify(tests["source_hashes"])
    schedule = a.read(b.BASE / "schedule.json")
    inputs = b.core.rows(b.preparation.INVENTORY / "inputs.jsonl")
    a.require(
        len(inputs) == 693
        and len({x["key"] for x in inputs}) == 693
        and all((x["gate"] is False and x["lang"] == "en" for x in inputs)),
        "Input scope differs",
    )
    a.require(
        [x["item_id"] for x in inputs] == analysis["ordered_item_ids"]
        and len({x["templ_id"] for x in inputs}) == 302,
        "Frozen ordered roster differs",
    )
    bindings = {
        b.rel(b.PLAN): plan_sha,
        b.rel(b.BASE / "analysis_plan.json"): a.digest(b.BASE / "analysis_plan.json"),
        b.rel(b.preparation.INVENTORY / "inputs.jsonl"): a.digest(
            b.preparation.INVENTORY / "inputs.jsonl"
        ),
        b.rel(tests_path): a.digest(tests_path),
    }
    bindings.update({name: a.digest(ROOT / name) for name in SOURCE_FILES})
    audits = []
    for i, stage in enumerate(schedule["stages"], 1):
        b.previous_review(i, schedule, plan_sha)
        review_path = b.BASE / "stage_reviews" / f"stage_{i:02d}.json"
        review = a.read(review_path)
        audit_path = ROOT / review["evidence"]
        audit = checked_read(audit_path, review["evidence_sha256"])
        acceptance_path = b.STUDY / f"stage{i}_review_v1/audit_acceptance.json"
        acceptance = a.read(acceptance_path)
        diag_path = ROOT / acceptance["diagnostics"]
        diag = checked_read(diag_path, acceptance["diagnostics_sha256"])
        expected_status = (
            f"guideline_stage_{i:02d}_reviewed_no_rerun_needed"
            if i < 4
            else "guideline_stage_04_reviewed_with_one_retained_null"
        )
        a.require(
            acceptance["status"] == diag["status"] == expected_status
            and acceptance["plan_sha256"] == plan_sha
            and (acceptance["main_audit_sha256"] == a.digest(audit_path)),
            "Additional review mismatch",
        )
        verify(acceptance["verified_bindings"])
        a.require(
            audit["counts"]["items"] == stage["items"]
            and diag["checked_predictions"] == stage["items"]
            and (audit["counts"] == diag["counts"]),
            "Stage audit scope differs",
        )
        for path in (review_path, audit_path, acceptance_path, diag_path):
            bindings[b.rel(path)] = a.digest(path)
        audits.append(audit)
    live = b.BASE / "live"
    a.require(
        {p.name for p in (live / "stages").iterdir()} == {s["id"] for s in schedule["stages"]},
        "Unexpected stage history",
    )
    index = {row["key"]: i for i, row in enumerate(inputs)}
    ci = {condition: i for i, condition in enumerate(b.CONDITIONS)}
    gates = np.full((693, 5, 3), -2, dtype=np.int8)
    rows = []
    output_hashes = {}
    operations = {c: Counter() for c in b.CONDITIONS}
    for stage, audit in zip(schedule["stages"], audits):
        for job in stage["jobs"]:
            folder = b.item_dir(live, job, plan_sha)
            path = folder / "result.json"
            receipt_path = folder / "complete.json"
            rel = b.rel(path)
            receipt_rel = b.rel(receipt_path)
            a.require(
                rel in audit["verified_output_hashes"]
                and receipt_rel in audit["verified_output_hashes"],
                "Output not in reviewed audit",
            )
            result = checked_read(path, audit["verified_output_hashes"][rel])
            checked_read(receipt_path, audit["verified_output_hashes"][receipt_rel])
            output_hashes[rel] = audit["verified_output_hashes"][rel]
            output_hashes[receipt_rel] = audit["verified_output_hashes"][receipt_rel]
            ii = index[job["key"]]
            cc = ci[job["condition"]]
            rr = job["repeat"]
            a.require(gates[ii, cc, rr] == -2, "Duplicate prediction identity")
            answer = result["result"]
            a.require(answer is None or type(answer["hate"]) is bool, "Invalid hate result")
            a.require((answer is None) == (result["n_valid"] == 0), "Validity mismatch")
            gate = -1 if answer is None else int(answer["hate"])
            gates[ii, cc, rr] = gate
            op = operations[job["condition"]]
            for field in ("repairs", "parse_failures", "gate_normalised", "normalisations"):
                op[field] += result[field]
            op["items"] += 1
            op["nulls"] += answer is None
            op["gate_ties"] += result["uncertain"]
            op["valid_votes_" + str(result["n_valid"])] += 1
            op["items_with_repairs"] += result["repairs"] > 0
            for field, value in result["accounting"].items():
                op["accounting_" + field] += value
            row = {
                "key": job["key"],
                "item_id": inputs[ii]["item_id"],
                "condition": job["condition"],
                "repeat": rr,
                "functionality": inputs[ii]["functionality"],
                "templ_id": inputs[ii]["templ_id"],
                "hate": None if gate == -1 else bool(gate),
                "n_valid": result["n_valid"],
                "gate_tie": result["uncertain"],
                "repairs": result["repairs"],
                "parse_failures": result["parse_failures"],
                "gate_normalised": result["gate_normalised"],
                "stage": stage["id"],
                "messages_sha256": job["messages_sha256"],
                "source_result": rel,
                "source_sha256": output_hashes[rel],
            }
            rows.append(row)
        print("Loaded reviewed predictions: " + stage["id"], flush=True)
    a.require(
        len(rows) == 10395 and np.isin(gates, [-1, 0, 1]).all(), "Missing or extra predictions"
    )
    common = m.common_mask(gates)
    a.require(common.any(), "No common-valid items")
    excluded = [
        {k: inputs[i][k] for k in ("key", "item_id", "templ_id", "functionality")}
        for i in np.flatnonzero(~common)
    ]
    nulls = [row for row in rows if row["hate"] is None]
    a.require(
        len(nulls) == 1
        and nulls[0]["key"] == "en/hatecheck-2892"
        and (nulls[0]["condition"] == "natural")
        and (nulls[0]["repeat"] == 2),
        "Unexpected failure roster",
    )
    contrasts = {**analysis["primary_contrasts"], **analysis["secondary_contrasts"]}
    a.require(len(contrasts) == 10, "Pairwise contrast scope differs")
    rng = np.random.default_rng(analysis["bootstrap_seed"])
    scopes = {}
    arrays = {}
    scope_specs = [
        ("all", np.arange(len(inputs))),
        *[
            (
                function,
                np.array([i for i, x in enumerate(inputs) if x["functionality"] == function]),
            )
            for function in sorted({x["functionality"] for x in inputs})
        ],
    ]
    for scope, indices in scope_specs:
        own_gates = gates[indices]
        own_meta = [inputs[i] for i in indices]
        stats = m.counts_and_rates(own_gates, list(b.CONDITIONS))
        selected = indices[common[indices]]
        metas = [inputs[i] for i in selected]
        templates, groups = m.cluster_structure([x["templ_id"] for x in metas])
        all_templates = sorted({x["templ_id"] for x in own_meta})
        stats.update(
            {
                "scope": scope,
                "roster_keys": [x["key"] for x in own_meta],
                "common_keys": [x["key"] for x in metas],
                "roster_templates": all_templates,
                "common_templates": templates,
                "excluded_keys": [x["key"] for i, x in zip(indices, own_meta) if not common[i]],
                "templates_lost": [x for x in all_templates if x not in templates],
            }
        )
        if not len(selected):
            stats.update({"status": "unscorable_no_common_items", "contrasts": {}})
        else:
            draws = rng.integers(
                0,
                len(templates),
                size=(analysis["bootstrap_draws"], len(templates)),
                dtype=np.int32,
            )
            boot = m.cluster_rates(gates[selected], [x["templ_id"] for x in metas], draws)
            stats["contrasts"] = m.contrasts(
                stats,
                boot,
                list(b.CONDITIONS),
                contrasts,
                analysis["primary_contrasts"],
                analysis["primary_family_confidence"],
                scope == "all",
            )
            stats["status"] = "scored"
            stats["bootstrap_mean_rate_descriptive_95"] = {
                c: m.interval(boot[:, j], 0.95) for j, c in enumerate(b.CONDITIONS)
            }
            arrays[scope + "__draws"] = draws
            arrays[scope + "__mean_rates"] = boot
        scopes[scope] = stats
    prompt_rows = b.core.rows(ROOT / plan["prompts"])
    cat = {(row["key"], row["condition"]): row for row in prompt_rows}
    a.require(
        len(cat) == len(prompt_rows) == 3465 and all(("context" in row for row in prompt_rows)),
        "Full frozen contexts required",
    )
    delivery = []
    lengths = {c: [] for c in b.CONDITIONS}
    natural_languages = Counter()
    for row in inputs:
        key = row["key"]
        natural = cat[key, "natural"]
        matched_ids = [x["id"] for x in cat[key, "matched_rule"]["context"]["guidelines"]]
        alt_ids = [x["id"] for x in cat[key, "alternative_rule"]["context"]["guidelines"]]
        natural_ids = [x["id"] for x in natural["context"]["guidelines"]]
        a.require(
            len(matched_ids) == len(alt_ids) == len(natural_ids) == 2, "Guideline slot mismatch"
        )
        languages = [x["meta"].get("lang") for x in natural["context"]["guidelines"]]
        natural_languages.update((str(lang) for lang in languages))
        delivery.append(
            {
                "key": key,
                "functionality": row["functionality"],
                "matched_lead": matched_ids[0],
                "filler": matched_ids[1],
                "alternative_lead": alt_ids[0],
                "natural_ids": natural_ids,
                "natural_languages": languages,
                "matched_lead_retrieved": matched_ids[0] in natural_ids,
                "filler_retrieved": matched_ids[1] in natural_ids,
            }
        )
        for c in b.CONDITIONS:
            lengths[c].append(sum((len(message["text"]) for message in cat[key, c]["messages"])))
    prompt_delivery = {
        "items": delivery,
        "natural_language_slots": dict(natural_languages),
        "matched_lead_delivered": sum((x["matched_lead_retrieved"] for x in delivery)),
        "filler_delivered": sum((x["filler_retrieved"] for x in delivery)),
        "by_functionality": {
            function: {
                "inputs": sum((x["functionality"] == function for x in delivery)),
                "matched_lead_delivered": sum(
                    (
                        x["functionality"] == function and x["matched_lead_retrieved"]
                        for x in delivery
                    )
                ),
                "filler_delivered": sum(
                    (x["functionality"] == function and x["filler_retrieved"] for x in delivery)
                ),
            }
            for function in sorted({x["functionality"] for x in inputs})
        },
        "message_character_lengths": {
            c: {
                "minimum": min(values),
                "maximum": max(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
            }
            for c, values in lengths.items()
        },
        "length_unit": "Unicode code points across internal role/text messages; not token counts.",
    }
    totals = Counter()
    account = Counter()
    usage = Counter()
    for audit in audits:
        totals.update(audit["counts"])
        account.update(audit["accounting"])
        usage.update(audit["observed_usage_sums"])
    verify(bindings)
    contract = {
        "created_utc": a.utc(),
        "plan_sha256": plan_sha,
        "analysis_sha256": a.digest(b.BASE / "analysis_plan.json"),
        "source_hashes": {name: bindings[name] for name in SOURCE_FILES},
        "numpy_version": np.__version__,
        "tests": b.rel(tests_path),
        "tests_sha256": bindings[b.rel(tests_path)],
        "sampling": "NumPy default_rng, seed 20260925; sorted template IDs. Generate global draws then six groups in sorted functionality order from one generator. Every draw is shared by all conditions and repeats in its scope. Save every draw.",
        "quantiles": "NumPy linear percentile interpolation; paired item-weighted rate differences; three decoder repeats fixed.",
        "nulls": "-1 internal missing value; never a correct non-hate. One shared mask across 15 settings; own-valid and conservative full-roster bounds also retained.",
        "scope": "All completed 693 inputs; no added evaluation examples, new model calls or changed prompts.",
    }
    a.save_new(DEST / "analysis_contract.json", contract)
    with (DEST / "bootstrap_arrays.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with (DEST / "predictions.jsonl").open("x", encoding="utf8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    compact = [
        {k: x[k] for k in ("key", "item_id", "templ_id", "functionality", "text_sha256", "gate")}
        for x in inputs
    ]
    a.save_new(
        DEST / "gate_matrix.json",
        {
            "items": compact,
            "conditions": list(b.CONDITIONS),
            "repeats": [0, 1, 2],
            "values": gates.tolist(),
            "missing_value": -1,
        },
    )
    a.save_new(DEST / "source_output_hashes.json", output_hashes)
    a.save_new(DEST / "prompt_delivery.json", prompt_delivery)
    evidence = {
        "created_utc": a.utc(),
        "status": "complete_roster_scored_independent_verification_pending",
        "plan_sha256": plan_sha,
        "analysis_sha256": a.digest(b.BASE / "analysis_plan.json"),
        "source_hashes": contract["source_hashes"],
        "verified_bindings": bindings,
        "verified_result_and_completion_hashes": len(output_hashes),
        "scopes": scopes,
        "null_predictions": nulls,
        "excluded_from_common": excluded,
        "execution_counts": dict(totals),
        "accounting": dict(account),
        "observed_usage": dict(usage),
        "operations_by_condition": {c: dict(v) for c, v in operations.items()},
        "prompt_delivery_summary": {k: v for k, v in prompt_delivery.items() if k != "items"},
        "bootstrap_draws_per_scope": analysis["bootstrap_draws"],
        "bootstrap_seed": analysis["bootstrap_seed"],
        "actual_API_calls": 0,
        "actual_CUDA_training_retrieval": False,
        "elapsed_seconds": time.perf_counter() - tick,
        "qualifications": analysis["limits"]
        + [
            "Historical plan limits about preparation readiness describe their earlier checkpoint; all preparation and all four live stages are now accepted.",
            "Intervals condition on three fixed decoder repeats and frozen first-pass retrieval, not all decoder/provider or retrieval uncertainty.",
            "The sole null is retained and one input leaves the shared mask for all settings. Complete-roster bounds remain visible.",
        ],
    }
    a.save_new(EVIDENCE, evidence)
    outputs = [
        DEST / name
        for name in (
            "analysis_contract.json",
            "bootstrap_arrays.npz",
            "predictions.jsonl",
            "gate_matrix.json",
            "source_output_hashes.json",
            "prompt_delivery.json",
        )
    ]
    outputs.append(EVIDENCE)
    a.save_new(
        DEST / "complete.json",
        {
            "created_utc": a.utc(),
            "status": evidence["status"],
            "plan_sha256": plan_sha,
            "evidence": b.rel(EVIDENCE),
            "evidence_sha256": a.digest(EVIDENCE),
            "published_hashes": {b.rel(p): a.digest(p) for p in outputs},
            "verified_bindings": bindings,
            "actual_API_calls": 0,
            "next": "Independent source/metric/expanded-cluster verification before publication.",
        },
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "scored_predictions": len(rows),
                "common_items": int(common.sum()),
                "nulls": len(nulls),
                "common_templates": len(scopes["all"]["common_templates"]),
                "conditions": {c: scopes["all"]["conditions"][c]["common"] for c in b.CONDITIONS},
                "primary_contrasts": {
                    name: scopes["all"]["contrasts"][name] for name in analysis["primary_contrasts"]
                },
                "evidence_sha256": a.digest(EVIDENCE),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        run()
