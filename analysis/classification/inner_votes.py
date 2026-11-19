import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import unittest
import numpy as np
from analysis.common import replay as c

ROOT = c.ROOT
CORE = c.CORE
BASE = c.BASE / "votes"
API = CORE / "api_execution_v1"
SOURCE = Path(__file__).resolve()
COMMON = Path(c.__file__).resolve()
SUMMARY = (
    ROOT / "Thesis_Log/astra_result/evidence/core_completion_summary_20260920_v1.json"
)
FOCAL = ("en_dev_eval_main", "de_legal_dev_eval")


def accounting(events, slots=None):
    out = Counter()
    for e in events:
        if slots is not None and e["vote"] not in slots:
            continue
        out["post_attempts"] += 1
        out["elapsed_plus_backoff_seconds"] += e["elapsed_s"] + e["backoff_elapsed_s"]
        out["retry_attempts"] += e["attempt"] > 0
        out["ordinary_attempts" if e["kind"] == "ordinary" else "repair_attempts"] += 1
        if e["action"] == "ok":
            out["successful_responses"] += 1
            metadata = e.get("model_response_metadata") or {}
            usage = metadata.get("usage")
            if usage is None:
                out["successful_responses_without_usage"] += 1
            else:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if usage.get(k) is not None:
                        out[k] += usage[k]
        else:
            out["non_success_attempts"] += 1
    return dict(out)


def prepare():
    assert not (BASE / "plan.json").exists()
    ep = c.read(API / "scoring_plan.json")["endpoints"]
    summary = c.read(SUMMARY)
    paths = [
        SOURCE,
        COMMON,
        ROOT / "tests/classification/test_offline_closure.py",
        SUMMARY,
        API / "plan.json",
        API / "schedule.json",
        API / "scoring_plan.json",
        CORE / "dev_inputs.jsonl",
        ROOT / "config/taxonomy.yaml",
        ROOT / "src/hsrag/vote.py",
        ROOT / "src/hsrag/schema.py",
        ROOT / "analysis/common/scoring.py",
        ROOT / "experiments/astra/qwen_downtime_inventory_20261006_v1/inventory.json",
    ]
    paths += [ROOT / f"data/processed/{k}.parquet" for k in ep]
    paths += [ROOT / v["source_results"] for v in summary["endpoints"].values()]
    hashes = {c.rel(p): c.sha(p) for p in paths}
    for v in summary["endpoints"].values():
        assert hashes[v["source_results"]] == v["source_sha256"]
    plan = {
        "created_utc": c.utc(),
        "status": "policy_frozen_before_new_vote_scores",
        "scope": "Existing full/small core RAG; seven original DEV rosters; no new inference.",
        "endpoints": ep,
        "conditions": ["rag_full", "rag_small"],
        "repeats": [0, 1, 2],
        "single": "Always inner slot 0, after its original bounded repair process. Null remains null; no first-valid substitution.",
        "voted": "Original stored three-slot aggregate; independently reconstruct all decision fields.",
        "focal": {
            "conditions": ["rag_full"],
            "endpoints": list(FOCAL),
            "family_size": 2,
            "confidence": 0.975,
        },
        "supporting": "Other 12 endpoint/KB differences use descriptive 95% intervals; all 14 shown, no winner selection.",
        "contrast": "Mean of three outer-repeat macro-F1 values: voted minus slot0. Common validity across both modes and all three repeats.",
        "uncertainty": {
            "draws": 10000,
            "seed": 20261006,
            "primary_unit": "NFKC/casefold/whitespace-normalized exact text cluster",
            "item_sensitivity": True,
            "fixed_fitted_repeats": True,
        },
        "missing": "Preserve frozen gold/scorable/fixed-label masks, own-valid F1, common-valid contrasts and conservative full-roster failure bounds. Never impute null as negative.",
        "reliability": "Within-prediction slot pairs and between-outer-repeat pairs computed from decisions. Report unavailable pairs, endpoint equality, gate equality, label Jaccard and per-label positive/disagreement counts.",
        "cost": "Observed all-slot work versus saved slot-0 work with its repairs/retries. Counterfactual deployment estimate, not observed new deployment, exact wall speedup, GPU work or billing.",
        "scope_limits": "Retrospective exposed-DEV analysis; no change to original primary family or selected main system. No human-review or unfinished balanced-study outputs.",
        "input_hashes": hashes,
    }
    c.save(BASE / "plan.json", plan)
    print("Frozen vote-analysis plan " + c.sha(BASE / "plan.json"), flush=True)


def load_plan():
    p = c.read(BASE / "plan.json")
    c.verify(p["input_hashes"])
    return p


def extract():
    plan = load_plan()
    assert not (BASE / "records.jsonl").exists()
    jobs = [
        j
        for st in c.read(API / "schedule.json")["stages"]
        for j in st["jobs"]
        if j["condition"] in plan["conditions"]
    ]
    dev = {x["key"]: x for x in c.rows(CORE / "dev_inputs.jsonl")}
    plan_sha = c.sha(API / "plan.json")
    assert len(jobs) == 7770

    def one(job):
        identity = {
            "experiment_sha256": plan_sha,
            **{k: job[k] for k in ("condition", "repeat", "key")},
        }
        directory = API / "live/items" / c.signature(identity)
        cp = directory / "complete.json"
        rp = directory / "result.json"
        receipt = c.read(cp)
        source_sha = c.sha(rp)
        assert receipt["status"] == "complete"
        assert receipt["hashes"][rp.relative_to(API / "live").as_posix()] == source_sha
        r = c.read(rp)
        assert r["identity"] == identity and r["execution_mode"] == "live"
        assert r["n_runs"] == len(r["run_predictions"]) == 3
        slots = [c.decision(x) for x in r["run_predictions"]]
        original = c.decision(r["result"])
        assert c.independent_aggregate(slots) == original
        assert sum((x is not None for x in slots)) == r["n_valid"]
        for vote in range(3):
            path = directory / "votes" / f"{vote}.json"
            assert (
                c.sha(path)
                == receipt["hashes"][path.relative_to(API / "live").as_posix()]
            )
            assert c.decision(c.read(path)["result"]) == slots[vote]
        allcost = accounting(r["events"])
        single = accounting(r["events"], {0})
        assert allcost["post_attempts"] == r["accounting"]["post_attempts"]
        assert (
            abs(sum(r["latencies"]) - allcost["elapsed_plus_backoff_seconds"]) < 1e-05
        )
        item = dev[job["key"]]
        return {
            "key": job["key"],
            "item_id": item["item_id"],
            "subset": job["subset"],
            "condition": job["condition"],
            "repeat": job["repeat"],
            "text_sha256": __import__("hashlib")
            .sha256(item["text"].encode())
            .hexdigest(),
            "normalized_text_cluster": __import__("hashlib")
            .sha256(c.normalized(item["text"]).encode())
            .hexdigest(),
            "source_result": c.rel(rp),
            "source_result_sha256": source_sha,
            "source_receipt_sha256": c.sha(cp),
            "slots": slots,
            "voted": original,
            "n_valid": r["n_valid"],
            "cost_all": allcost,
            "cost_slot0": single,
            "latency_slot0": r["latencies"][0],
            "latency_all": sum(r["latencies"]),
        }

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(one, jobs), 1):
            records.append(row)
            if i % 1000 == 0:
                print(f"Copied and verified saved vote records: {i}/7770", flush=True)
    assert len({(x["key"], x["condition"], x["repeat"]) for x in records}) == 7770
    c.save_rows(BASE / "records.jsonl", records)
    c.save(
        BASE / "extraction.json",
        {
            "created_utc": c.utc(),
            "plan_sha256": c.sha(BASE / "plan.json"),
            "records_sha256": c.sha(BASE / "records.jsonl"),
            "records": len(records),
            "votes_verified": 3 * len(records),
            "independent_decision_reconstructions": len(records),
            "actual_API_calls": 0,
        },
    )
    print("Extraction complete; original records preserved.", flush=True)


def analyse():
    plan = load_plan()
    extraction = c.read(BASE / "extraction.json")
    assert extraction["records_sha256"] == c.sha(BASE / "records.jsonl")
    assert not (BASE / "results.json").exists()
    rr = c.rows(BASE / "records.jsonl")
    by = defaultdict(dict)
    for x in rr:
        by[x["subset"], x["condition"], x["repeat"]][x["item_id"]] = x
    summary = c.read(SUMMARY)
    oldchecks = 0
    results = {}
    contrast_count = 0
    for subset, endpoint in plan["endpoints"].items():
        gold = c.gold_rows(subset)
        ids = [str(x["id"]) for x in gold]
        assert ids == endpoint["scorable_ids"]
        source_report = c.read(ROOT / summary["endpoints"][subset]["source_results"])
        old = (
            source_report["endpoints"][subset]["conditions"]
            if "endpoints" in source_report
            else source_report["conditions"]
        )
        result = {"endpoint": endpoint, "conditions": {}}
        for condition in plan["conditions"]:
            samples = [by[subset, condition, rep] for rep in range(3)]
            assert all((set(x) == set(ids) for x in samples))
            allc = {}
            runs = {}
            agreements = {}
            costs = {}
            clusters = [samples[0][rid]["normalized_text_cluster"] for rid in ids]
            for mode in ("slot0", "voted"):
                curves = []
                table = []
                for rep in range(3):
                    pred = {
                        rid: (
                            samples[rep][rid]["slots"][0]
                            if mode == "slot0"
                            else samples[rep][rid]["voted"]
                        )
                        for rid in ids
                    }
                    values = c.contributions(gold, pred, endpoint)
                    score = c.summarize(values)
                    curves.append(values)
                    score["replicate"] = rep
                    table.append(score)
                    if mode == "voted":
                        prior = next(
                            (x for x in old[condition]["runs"] if x["replicate"] == rep)
                        )["score"]
                        assert (
                            abs(
                                prior["macro_f1_valid_only"]
                                - score["macro_f1_valid_only"]
                            )
                            < 1e-12
                        )
                        assert prior["per_label_counts"] == score["per_label_counts"]
                        oldchecks += 1
                allc[mode] = curves
                runs[mode] = {"runs": table, "summary": c.means(table)}
                pairs = [
                    (
                        (
                            samples[a][rid]["slots"][0]
                            if mode == "slot0"
                            else samples[a][rid]["voted"]
                        ),
                        (
                            samples[b][rid]["slots"][0]
                            if mode == "slot0"
                            else samples[b][rid]["voted"]
                        ),
                    )
                    for rid in ids
                    for a, b in combinations(range(3), 2)
                ]
                agreements["outer_" + mode] = c.pair_agreement(
                    pairs, endpoint["dimension"], endpoint["all_labels"]
                )
                total = Counter()
                for sample in samples:
                    for rid in ids:
                        total.update(
                            sample[rid]["cost_slot0" if mode == "slot0" else "cost_all"]
                        )
                costs[mode] = dict(total)
            inner = [
                (row["slots"][a], row["slots"][b])
                for sample in samples
                for row in sample.values()
                for a, b in combinations(range(3), 2)
            ]
            agreements["inner_slots"] = c.pair_agreement(
                inner, endpoint["dimension"], endpoint["all_labels"]
            )
            agreements["slot0_vs_voted"] = c.pair_agreement(
                [
                    (row["slots"][0], row["voted"])
                    for sample in samples
                    for row in sample.values()
                ],
                endpoint["dimension"],
                endpoint["all_labels"],
            )
            family = 2 if condition == "rag_full" and subset in FOCAL else None
            contrast = c.bootstrap(
                allc["voted"], allc["slot0"], clusters, family_size=family
            )
            item = c.bootstrap(
                allc["voted"], allc["slot0"], list(range(len(ids))), family_size=family
            )
            contrast_count += 1
            failure = Counter()
            for sample in samples:
                for row in sample.values():
                    failure["predictions"] += 1
                    failure["slot0_null"] += row["slots"][0] is None
                    failure["voted_null"] += row["voted"] is None
                    failure["slot0_failure_rescued_by_later_votes"] += (
                        row["slots"][0] is None and row["voted"] is not None
                    )
                    failure[f"valid_slots_{row['n_valid']}"] += 1
            result["conditions"][condition] = {
                "modes": runs,
                "voted_minus_slot0": contrast,
                "item_bootstrap_sensitivity": item,
                "agreement": agreements,
                "costs": costs,
                "validity": dict(failure),
                "reported_token_ratio_voted_over_slot0": costs["voted"]["total_tokens"]
                / costs["slot0"]["total_tokens"],
                "attempt_ratio_voted_over_slot0": costs["voted"]["post_attempts"]
                / costs["slot0"]["post_attempts"],
            }
            print(
                f"Scored {subset}/{condition}: {contrast['mean_difference']:+.6f}; original scores reproduce.",
                flush=True,
            )
        results[subset] = result
    assert oldchecks == 42 and contrast_count == 14
    c.save(
        BASE / "results.json",
        {
            "created_utc": c.utc(),
            "status": "complete",
            "plan_sha256": c.sha(BASE / "plan.json"),
            "records_sha256": c.sha(BASE / "records.jsonl"),
            "endpoints": results,
            "original_repeat_scores_verified": oldchecks,
            "contrasts": contrast_count,
            "bootstrap_draws": contrast_count * 20000,
            "actual_API_calls": 0,
        },
    )
    c.save(
        BASE / "complete.json",
        {
            "status": "complete",
            "created_utc": c.utc(),
            "results_sha256": c.sha(BASE / "results.json"),
            "plan_sha256": c.sha(BASE / "plan.json"),
        },
    )


def report():
    q = c.read(BASE / "results.json")
    assert c.read(BASE / "complete.json")["results_sha256"] == c.sha(
        BASE / "results.json"
    )
    text = [
        "# One saved answer versus three-answer voting",
        "",
        "6 October 2026. Completed retrospective offline analysis; no new Qwen calls, training or CUDA. The original core results and the selected main system remain unchanged.",
        "",
        "The single-answer condition is always response slot 0 after its original bounded repairs. The voted condition is the original three-slot result. Three outer decoder repeats remain separate. All 42 old RAG repeat scores and their confusion counts reproduce.",
        "",
        "## Accuracy and uncertainty",
        "",
        "Means use each mode's own valid outputs. Differences/intervals use the common-valid texts across both modes and all three repeats. Intervals resample normalized exact-text clusters; fitted decoder repeats are fixed. Two focal full-KB contrasts use a 97.5% family interval; the other contrasts are descriptive 95%.",
        "",
        "| Endpoint | KB | Single mean F1 | Voted mean F1 | Paired voted−single | Interval | Common texts |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for ep, x in q["endpoints"].items():
        for kb, z in x["conditions"].items():
            d = z["voted_minus_slot0"]
            ci = d["family_interval"] or d["ci95"]
            level = "97.5%" if d["family_interval"] else "95%"
            text.append(
                f"| {ep} | {kb} | {z['modes']['slot0']['summary']['mean']:.6f} | {z['modes']['voted']['summary']['mean']:.6f} | {d['mean_difference']:+.6f} | {level} [{ci[0]:+.6f}, {ci[1]:+.6f}] | {d['n_common_valid']} |"
            )
    text += [
        "",
        "## Reliability, validity and cost",
        "",
        "Agreement is recomputed from actual decisions. It does not use the misleading multilabel/severity `agreement` metadata. Pairwise rates condition on both outputs being valid; all missing-pair counts remain in the JSON.",
        "",
        "| Endpoint | KB | Single / voted nulls | Single / voted outer agreement | Slot-0 versus voted agreement | Voted / single reported-token ratio |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for ep, x in q["endpoints"].items():
        for kb, z in x["conditions"].items():
            a = z["agreement"]
            v = z["validity"]
            text.append(
                f"| {ep} | {kb} | {v.get('slot0_null', 0)} / {v.get('voted_null', 0)} | {a['outer_slot0']['endpoint_exact_agreement']:.3%} / {a['outer_voted']['endpoint_exact_agreement']:.3%} | {a['slot0_vs_voted']['endpoint_exact_agreement']:.3%} | {z['reported_token_ratio_voted_over_slot0']:.3f} |"
            )
    text += [
        "",
        "These ratios compare the successful-response token usage actually recorded for all slots with the recorded slot-0 work. Keeping only slot 0 is a counterfactual cost estimate: the experiment actually made all requests. It is not an exact wall-time speedup, complete remote compute measurement or bill. Failed/administratively interrupted remote work can be unknown.",
        "",
        "## Every run and supporting diagnostics",
        "",
        "The machine report contains all single/voted run scores, per-label confusion counts, exact-set accuracy, label cardinality, extra/missing labels, full-roster failure bounds, slot/outer-repeat disagreements, positive-label disagreement denominators, request/repair/retry/token counts and paired item-bootstrap sensitivity. Nulls are never converted to negative labels.",
        "",
        "This is exposed-DEV evidence. No result selects a new vote count for the already frozen ongoing studies, tunes prompts, changes a threshold or changes their original uncertainty families.",
        "",
        "Evidence: [plan](../../experiments/astra/offline_closure_20261006_v1/votes/plan.json), [all runs/results](../../experiments/astra/offline_closure_20261006_v1/votes/results.json), [completion](../../experiments/astra/offline_closure_20261006_v1/votes/complete.json).",
    ]
    path = (
        ROOT / "Thesis_Log/astra_result/astra_single_three_vote_results_20261006_v1.md"
    )
    with path.open("x", encoding="utf8", newline="\n") as f:
        f.write("\n".join(text) + "\n")
    print("Wrote " + c.rel(path), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "extract", "analyse", "report"])
    mode = p.parse_args().mode
    with c.network_guard():
        globals()[mode]()


if __name__ == "__main__":
    main()
