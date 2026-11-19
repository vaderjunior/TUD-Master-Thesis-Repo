import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import socket
from unittest.mock import patch
from analysis.feedback.summary import p3_audit as u
from analysis.feedback.source import eligibility as f
from analysis.common import prompt as renderer

BASE = f.BASE
PLAN = BASE / "scientific_plan_v1.json"
SEEDS = [202609281, 202609282, 202609283]
MAX_K = 40
MIN_K = 20
GROUP = "Additional labelled examples"
LABELS = list(u.LABELS)


def choose_updates(eligible, gold, runs):
    u.require(
        len(runs) == 3 and len(set(eligible)) == len(eligible),
        "Three selection repeats and unique IDs required",
    )
    groups = {"error": [], "correct": [], "invalid": []}
    for i in sorted(eligible):
        p = [r[i]["result"] for r in runs]
        status = (
            "invalid"
            if any((v is None for v in p))
            else (
                "correct"
                if sum((gold[i] <= set(v["hate_type"]) for v in p)) >= 2
                else "error"
            )
        )
        groups[status].append(i)
    k = min(MAX_K, len(groups["error"]), len(groups["correct"]))
    u.require(
        k >= MIN_K,
        "Insufficient matched candidates: pause before updates; no top-up or retry to change classification",
    )
    levels = [(k * j + 3) // 4 for j in range(1, 5)]
    sets = []
    for seed in SEEDS:
        for arm in ("error", "correct"):
            ranked = sorted(
                groups[arm],
                key=lambda i: (
                    hashlib.sha256(f"{seed}/{arm}/{i}".encode()).hexdigest(),
                    i,
                ),
            )
            for level, n in enumerate(levels, 1):
                sets.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "level": level,
                        "size": n,
                        "ids": ranked[:n],
                    }
                )
    return {"groups": groups, "k": k, "levels": levels, "sets": sets}


def append_feedback(context, record):
    out = copy.deepcopy(context)
    u.require(
        GROUP not in {name for name, _ in out["example_groups"]},
        "Do not append a second feedback slot",
    )
    regular = [h for _, rr in out["example_groups"] for h in rr]
    u.require(
        len(regular) == 5 and len(out["definitions"]) == len(out["guidelines"]) == 2,
        "Frozen baseline context required",
    )
    u.require(
        record["id"] not in {h["id"] for h in regular},
        "Record is already an ordinary example",
    )
    u.require(
        record["kind"] == "feedback"
        and record["lang"] == "en"
        and (record["meta"]["gate"] is True),
        "Wrong correction type/language/gate",
    )
    u.require(
        record["meta"]["hate_types"]
        and set(record["meta"]["hate_types"]) <= set(u.LABELS),
        "Invalid correction type labels",
    )
    u.require(
        all(
            (
                record["meta"].get(k) is None
                for k in ("target_groups", "severity", "legal")
            )
        ),
        "Unannotated dimensions must stay missing",
    )
    hit = {
        "id": record["id"],
        "text": record["text"],
        "meta": copy.deepcopy(record["meta"]),
        "via": "rrf",
    }
    out["example_groups"].append([GROUP, [hit]])
    u.require(
        out["definitions"] == context["definitions"]
        and out["guidelines"] == context["guidelines"]
        and (out["example_groups"][:-1] == context["example_groups"]),
        "Base context changed",
    )
    return out


def tests():
    ids = [f"e{i}" for i in range(45)] + [f"c{i}" for i in range(45)] + ["bad"]
    gold = {i: {"explicit"} for i in ids}
    runs = [
        {
            i: {
                "result": (
                    None
                    if i == "bad"
                    else {
                        "hate_type": [] if i.startswith("e") else ["explicit", "irony"]
                    }
                )
            }
            for i in ids
        }
        for _ in range(3)
    ]
    result = choose_updates(ids, gold, runs)
    u.require(
        result["k"] == 40
        and result["levels"] == [10, 20, 30, 40]
        and (len(result["sets"]) == 24),
        "Budget/cap fixture",
    )
    u.require(
        result == choose_updates(list(reversed(ids)), gold, list(reversed(runs))),
        "Order-independent selection",
    )
    u.require(
        result["groups"]["invalid"] == ["bad"]
        and "bad" not in sum([r["ids"] for r in result["sets"]], []),
        "Invalid selection fixture",
    )
    for seed in SEEDS:
        e = [r for r in result["sets"] if r["seed"] == seed and r["arm"] == "error"]
        c = [r for r in result["sets"] if r["seed"] == seed and r["arm"] == "correct"]
        for x, y in zip(e, c):
            u.require(
                x["size"] == y["size"] and (not set(x["ids"]) & set(y["ids"])),
                "Arm matching fixture",
            )
        for arm in (e, c):
            for low, high in zip(arm, arm[1:]):
                u.require(
                    high["ids"][: len(low["ids"])] == low["ids"],
                    "Nested budget fixture",
                )
    little = [f"e{i}" for i in range(21)] + [f"c{i}" for i in range(25)]
    u.require(
        choose_updates(little, gold, runs)["levels"] == [6, 11, 16, 21],
        "Predeclared smaller budget fixture",
    )
    try:
        choose_updates(["e0", "c0"], gold, runs)
    except AssertionError:
        pass
    else:
        raise AssertionError("Too-small budget accepted")

    def hit(i):
        return {"id": i, "text": "Fixture " + i, "meta": {}, "via": "fixture"}

    ctx = {
        "definitions": [hit("d1"), hit("d2")],
        "guidelines": [hit("g1"), hit("g2")],
        "example_groups": [["Labelled examples", [hit(f"x{i}") for i in range(5)]]],
    }
    old = copy.deepcopy(ctx)
    rec = {
        "id": "test-feedback",
        "kind": "feedback",
        "lang": "en",
        "text": "Distinct correction fixture text.",
        "meta": {
            "gate": True,
            "hate_types": ["explicit"],
            "target_groups": None,
            "severity": None,
            "legal": None,
        },
    }
    updated = append_feedback(ctx, rec)
    u.require(
        ctx == old and len(updated["example_groups"]) == 2,
        "Context immutability fixture",
    )
    before = renderer.render_messages("Fixture input", "en", ctx, "corrected")
    after = renderer.render_messages("Fixture input", "en", updated, "corrected")
    u.require(
        before[0] == after[0]
        and GROUP in after[-1]["text"]
        and (rec["text"] in after[-1]["text"])
        and ("hate_type=[explicit]" in after[-1]["text"]),
        "Visible correction fixture",
    )
    u.require(
        "target_group=[]" not in after[-1]["text"]
        and "legal=[]" not in after[-1]["text"],
        "Missing annotation rendering fixture",
    )
    try:
        append_feedback(updated, rec)
    except AssertionError:
        pass
    else:
        raise AssertionError("Second feedback slot accepted")
    bad = copy.deepcopy(rec)
    bad["meta"]["target_groups"] = []
    try:
        append_feedback(ctx, bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing-as-empty correction accepted")
    paths = [
        Path(__file__),
        Path(f.__file__),
        Path(u.__file__),
        ROOT_PATH / "analysis/common/prompt.py",
        ROOT_PATH / "src/hsrag/prompt.py",
        ROOT_PATH / "prompts/classify_v1.txt",
        ROOT_PATH / "config/taxonomy.yaml",
    ]
    evidence = {
        "passed": True,
        "fixtures": 12,
        "created_utc": u.utc(),
        "source_hashes": {u.rel(p): u.digest(p) for p in paths},
    }
    path = BASE / "development" / ("design_tests_" + u.digest(__file__)[:16] + ".json")
    u.save(path, evidence)
    print(json.dumps(evidence, indent=2), flush=True)


ROOT_PATH = u.ROOT


def freeze():
    u.require(not PLAN.exists(), "Scientific plan is immutable")
    evidence = u.read(f.EVIDENCE)
    u.verify(evidence["input_hashes"])
    u.verify(evidence["outputs"])
    test_path = (
        BASE / "development" / ("design_tests_" + u.digest(__file__)[:16] + ".json")
    )
    test = u.read(test_path)
    u.require(test["passed"], "Design fixtures not passed")
    u.verify(test["source_hashes"])
    historic = u.read(u.EVIDENCE)
    core = u.read(f.CORE / "plan.json")
    goldpath = "data/processed/en_dev_eval_sq3_types.parquet"
    u.require(
        u.digest(ROOT_PATH / goldpath)
        == historic["input_hashes"][goldpath]
        == core["input_hashes"][goldpath],
        "Gold source differs from existing audits",
    )
    roster = f.jl(f.OUT / "roster.jsonl")
    byid = {r["id"]: r for r in roster}
    eligible = evidence["eligible_ids"]
    transfer = sorted((r["id"] for r in roster if r["role"] != "batches"))
    pool = sorted((r["id"] for r in roster if r["role"] == "pool"))
    held = sorted((r["id"] for r in roster if r["role"] == "held_out"))
    execution = sorted(set(eligible) | set(transfer))
    selection = sorted(set(eligible) | set(pool))
    u.require(
        (len(transfer), len(held), len(pool), len(execution), len(selection))
        == (350, 121, 229, 466, 345),
        "Unexpected execution roster",
    )
    base_records = {r["id"]: r for r in f.jl(f.CORE / "kb/full/records.jsonl")}
    prompts = f.jl(f.OUT / "baseline_prompts.jsonl")
    for item in prompts:
        ctx = item["context"]
        for h in (
            ctx["definitions"]
            + ctx["guidelines"]
            + [h for _, rr in ctx["example_groups"] for h in rr]
        ):
            rec = base_records[h["id"]]
            meta = {
                **{k: rec[k] for k in ("kind", "lang", "source", "dimension", "label")},
                **rec["meta"],
            }
            u.require(
                all((h["meta"].get(k) == v for k, v in meta.items()))
                and set(h["meta"]) <= set(meta) | {"_ranks"},
                "Frozen annotation mismatch",
            )
    adjacency = {r["id"]: set() for r in roster}
    for pair in evidence["near_pairs"]:
        if pair["edge"]:
            adjacency[pair["left"]].add(pair["right"])
            adjacency[pair["right"]].add(pair["left"])
    seen = set()
    components = []
    for iid in sorted(adjacency):
        if iid in seen:
            continue
        todo = [iid]
        component = set()
        while todo:
            nxt = todo.pop()
            if nxt in component:
                continue
            component.add(nxt)
            todo.extend(adjacency[nxt] - component)
        seen |= component
        components.append(sorted(component))
    u.require(
        sorted(components) == evidence["components"],
        "Independent graph reconstruction differs",
    )
    for group in components:
        if set(group) & set(transfer):
            u.require(
                not set(group) & set(eligible),
                "Eligible correction touches transfer cluster",
            )
    hist = [
        {
            r["id"]: r["result"]
            for r in f.jl(f.SCORES / f"assembled/rag_full_rep{k}.jsonl")
        }
        for k in range(3)
    ]
    codes = []
    for iid in eligible:
        predictions = [r[iid] for r in hist]
        codes.append(
            "invalid"
            if any((p is None for p in predictions))
            else (
                "correct"
                if sum(
                    (
                        set(byid[iid]["gold"]["hate_type"]).issubset(p["hate_type"])
                        for p in predictions
                    )
                )
                >= 2
                else "error"
            )
        )
    for status in ("error", "correct", "invalid"):
        u.require(
            codes.count(status)
            == evidence["historical_capacity_screened_candidates"]["counts"].get(
                status, 0
            ),
            "Independent historical supply differs",
        )
    sources = {
        **evidence["outputs"],
        **test["source_hashes"],
        u.rel(f.EVIDENCE): u.digest(f.EVIDENCE),
        u.rel(test_path): u.digest(test_path),
        u.rel(u.EVIDENCE): u.digest(u.EVIDENCE),
    }
    selection_predictions = len(selection) * 3
    reference_predictions = len(execution) * 3
    level_predictions = len(execution) * 3 * 2 * 3
    stages = [
        {
            "stage": "baseline_selection_and_reference",
            "predictions": selection_predictions + reference_predictions,
        },
        {"stage": "budgets_01_02", "predictions": 2 * level_predictions},
        {"stage": "budget_03", "predictions": level_predictions},
        {"stage": "budget_04", "predictions": level_predictions},
    ]
    plan = {
        "status": "scientific_design_frozen_execution_not_ready",
        "created_utc": u.utc(),
        "study": "sq3_full_kb_feedback_20260928_v1",
        "question": "Does selecting correctly labelled examples from the full-KB baseline errors transfer better than adding the same number of already-correct examples from the same allowed pool?",
        "selection_history": "Post-selection DEV design, informed by earlier small/full results and a separately recorded historical feasibility count; all new responses remain ungenerated.",
        "inputs": sources,
        "core_plan_sha256": u.digest(f.CORE / "plan.json"),
        "base_records_sha256": u.digest(f.CORE / "kb/full/records.jsonl"),
        "rosters": {
            "eligible_candidates": eligible,
            "transfer": transfer,
            "holdout": held,
            "sentinel_selection_pool": pool,
            "selection_phase": selection,
            "evaluation_phase": execution,
            "excluded_candidate_ids": sorted(evidence["excluded_candidates"]),
        },
        "model": {
            "name": "Qwen/Qwen3-VL-32B-Instruct",
            "temperature": 1.0,
            "votes_per_prediction": 3,
            "decoder_repeats_per_setting": 3,
            "historical_predictions_reused_as_new_results": False,
            "generation_seed": "Do not claim deterministic independent random seeds; record actual API request fields and backend metadata.",
        },
        "selection": {
            "criterion": "Gold type set contained in final predicted type set in at least 2 of 3 fresh selection-phase repeats; errors have at most 1 success. Any whole-output null excludes the candidate from both arms.",
            "phase": "Fresh full-KB no-feedback outputs on 345 texts. The 121 held-out texts have no selection-phase call. Selection-phase outputs do not enter the headline reference average.",
            "max_k": MAX_K,
            "minimum_k": MIN_K,
            "fallback": "K=min(40,n_errors,n_correct) after selection. If K<20, stop before any update; freeze a separately justified amendment without inspecting update outcomes. Never retry completed invalid outputs or top up from TRAIN.",
            "budgets": "ceil(K/4), ceil(K/2), ceil(3K/4), K; normally 10,20,30,40 per arm.",
            "seeds": SEEDS,
            "ranking": "SHA256 UTF8 of decimal seed/arm/item_id, ascending; separate error and correct lists, then nested prefixes. No replacement within a list. Draws may overlap and remain recorded.",
            "type_balance": "Match number, allowed source pool, gold availability and delivered-slot count. Do not force equal label marginals: their differences are part of the selection policy, and must be reported.",
            "fixed_selection": "Select all candidate pools once from the common fresh baseline; do not select new errors from intervening update outcomes. These are nested update budgets, not online error-correction rounds.",
        },
        "retrieval": {
            "base": "Reuse the 467 verified saved full-KB contexts; execute only the 466 retained texts. Five ordinary examples, two definitions and two guidelines remain identical per text across arms and budgets.",
            "additional": "One separate neutral Additional labelled examples slot from the selected sidecar pool. Both update arms have six labelled examples; no-update reference has five.",
            "feedback_dense": "Exact float64 cosine over the saved BGE-M3 text vectors, top min(10,K), descending score then ascending record ID. This replaces approximate indexing only for the <=40-record added bucket, not the full-base retriever.",
            "feedback_bm25": 'BM25Okapi defaults k1=1.5,b=0.75,epsilon=0.25; tokens re.findall(r"\\w+",text.lower()); sort descending score then ascending record ID; take top10 and discard nonpositive scores.',
            "fusion": "Sum 1/(60+one-based rank) across available channels; descending fused score then ascending record ID; keep one. No MMR on the feedback bucket.",
            "source_labels": "Only source-annotated hate=True and hate_type labels; target_group, severity and legal remain unannotated, not empty negatives. Record origin, selection phase, pool draw, budget and source hashes.",
            "holdout_gold": "Never used to choose delivered examples. Label relevance is calculated later for diagnosis only.",
            "case": "Keep each added source text verbatim; new IDs and KB records isolated from original active KB/global taxonomy. Added-vector source is the existing same-text query cache, with documented provenance.",
        },
        "reference_and_sentinel": {
            "reference": "Three fresh no-update evaluation-phase repeats on all 466 retained texts, distinct from selection-phase calls, even where prompts are identical. Do not reuse the three selected baseline outcomes as a no-regression reference.",
            "sentinel": "Within the original 229 pool texts, all gold labels contained in all 3 valid selection-phase repeats. Freeze IDs before reference/update scoring. Compare fresh no-update and update flip rates on exactly those IDs; report size/support, without forcing an 80-text target.",
            "selected_text_diagnostic": "Evaluate the 116 candidate texts too, separating selected correction/control members and not-selected candidates. Gains on selected texts may be retrieval of their own answer and are not transfer.",
        },
        "scoring": {
            "primary": "One primary contrast: final-K error-selected minus correct-selected mean seven-type macro-F1 on the fixed 350 transfer texts, average three pool draws and three decoder repeats within each draw.",
            "labels": LABELS,
            "ineligible": "Prediction gate false yields empty type predictions; zero denominator label F1=0; gold label set is fixed across conditions and every resample.",
            "mask": "Use one common-valid transfer mask across all final-K error/control settings (18), publish all failures and full-roster wrong/right bounds; never turn invalid output into non-hate. Other contrasts use explicitly named matched masks.",
            "uncertainty": "10000 paired cluster bootstrap draws, seed20260928; freeze overlap components now (350 transfer clusters currently). Resample same text clusters across all settings and recompute each setting before averaging. Conditional on fixed pools/repeats; no pooling votes or treating repeats as new independent texts.",
            "intervals": "Primary 95% percentile interval for the single declared primary contrast; other contrasts descriptive pointwise95. Old DEV exposure prevents claiming untouched confirmation.",
            "secondary": "Earlier budgets; each update arm versus fresh no-update; separate121 holdout; mean contains, exact-set match, per-label precision/recall/F1, label count, missed hate among gold-hateful texts, sentinel flips, selected-text repair and retrieval relevance.",
            "replication": "Report all decoder values, within-pool mean/sampleSD, between-pool mean spread separately. Shared baseline reference is not triplicated as independent runs. Earlier budgets are dependent nested interventions.",
            "interpretation": "No significant improvement is not equivalence or proof all feedback fails. Contains-based error selection permits extra predicted labels. No overall binary hate discrimination conclusion from this all-hateful subset.",
            "stopping": "Run the fixed budget matrix once; no accuracy-based extension, repeat selection, pool replacement or backend retry to improve scores. Retain adverse/null results.",
        },
        "budget": {
            "selection_predictions": selection_predictions,
            "reference_predictions": reference_predictions,
            "predictions_per_update_budget": level_predictions,
            "update_settings": 24,
            "total_predictions": sum((s["predictions"] for s in stages)),
            "ordinary_responses": sum((s["predictions"] for s in stages)) * 3,
            "ordinary_plus_two_repairs_per_vote_ceiling": sum(
                (s["predictions"] for s in stages)
            )
            * 9,
            "execution_batches": stages,
            "transport": "Exact retry/time/token/concurrency caps and phase-aware identities require a separate frozen execution plan before API use. This ceiling is arithmetic, not authorization for unlimited retries.",
        },
        "readiness": {
            "ready_to_run": False,
            "remaining": [
                "Implement and test exact feedback selection/ranking and explicit context assembly, including the omitted-feedback helper regression guard.",
                "Prepare and independently verify the baseline payload catalog and phase-distinct jobs; freeze transport, failure, accounting and resumption contracts.",
                "Mock full stage and zero-call restart; provide one PowerShell user launcher with a stage parameter. Review baseline before selecting real update pools.",
                "After baseline, freeze actual selected pools/K/sentinel IDs and all update payloads; audit before subsequent user batches.",
            ],
            "long_run_owner": "Ajay",
            "cuda_required_for_this_feedback_study": False,
            "next": "Offline execution preparation, not new model results.",
        },
        "limits": [
            "Historical small-KB experiment used adaptive per-round selection and TRAIN top-ups; this matched fixed-baseline budget study is a scientific counterpart, not an exact replication or causal interaction test of KB size.",
            "Added examples have a guaranteed dedicated slot; findings do not establish that ordinary insertion among71091 training examples is equally effective.",
            "Original full-base first-pass contexts and their accepted retrieval limitations are inherited; no new base-index retrieval or silent rerun.",
            "No German/taxonomy/encoder adaptation run is part of this feedback matrix. Those remain separately scoped later work.",
        ],
    }
    u.require(
        plan["budget"]["total_predictions"] == 35985
        and plan["budget"]["ordinary_responses"] == 107955,
        "Budget arithmetic differs",
    )
    u.verify(sources)
    u.save(PLAN, plan)
    u.save(
        BASE / "design_verification_v1.json",
        {
            "passed": True,
            "created_utc": u.utc(),
            "plan_sha256": u.digest(PLAN),
            "feasibility_sha256": u.digest(f.EVIDENCE),
            "tests": test,
            "verified_contexts": len(prompts),
            "graph_components": len(components),
            "transfer_clusters": evidence["evaluation_clusters"],
            "historical_capacity_independent_counts": {
                "error": codes.count("error"),
                "correct": codes.count("correct"),
                "invalid": codes.count("invalid"),
            },
            "source_hashes": sources,
            "new_model_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan_sha256": u.digest(PLAN),
                "budget": plan["budget"],
                "ready_to_run": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["tests", "freeze"])
    args = p.parse_args()
    with patch.object(socket.socket, "connect", u.forbid), patch.object(
        socket, "create_connection", u.forbid
    ):
        tests() if args.mode == "tests" else freeze()
