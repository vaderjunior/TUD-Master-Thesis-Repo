import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import socket
import time
import unicodedata
from unittest.mock import patch
import numpy as np
import pandas as pd
from analysis.feedback.summary import p3_audit as u

ROOT = u.ROOT
BASE = ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1"
OUT = BASE / "feasibility_v1"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
RET = CORE / "retrieval_v1"
SCORES = CORE / "secondary_results_v4/en_dev_eval_sq3_types"
EVIDENCE = (
    ROOT / "Thesis_Log/astra_result/evidence/sq3_feedback_feasibility_20260928_v1.json"
)
THRESHOLD = 0.95
MARGIN = 0.0005
LABELS = list(u.LABELS)


def key(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def prefixkey(text):
    return key(text).lstrip(": ")


def sha_text(text):
    return hashlib.sha256(str(text).encode("utf8")).hexdigest()


def order_id(i, seed=20260928):
    return (hashlib.sha256(f"{seed}/{i}".encode()).hexdigest(), i)


def jl(path):
    return list(u.jsonl(path))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def signature(obj):
    return hashlib.sha256(
        (
            json.dumps(
                obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        ).encode()
    ).hexdigest()


def components(ids, edges):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in edges:
        a, b = (find(x), find(y))
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return sorted([sorted(v) for v in groups.values()])


def eligibility(ids, roles, edges, base_exposed):
    groups = components(ids, edges)
    chosen = []
    excluded = {}
    cluster_map = {}
    for members in groups:
        label = "sq3c-" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        for i in members:
            cluster_map[i] = label
        candidates = [i for i in members if roles[i] == "batches"]
        if not candidates:
            continue
        if any((roles[i] != "batches" for i in members)):
            for i in candidates:
                excluded[i] = "component_touches_evaluation"
        elif any((i in base_exposed for i in members)):
            for i in candidates:
                excluded[i] = "component_touches_base_example"
        else:
            representative = min(candidates, key=order_id)
            chosen.append(representative)
            for i in candidates:
                if i != representative:
                    excluded[i] = "candidate_component_representative_kept"
    return (sorted(chosen), excluded, groups, cluster_map)


def capacity(ids, gold, runs):
    counts = Counter()
    statuses = {}
    per_label = defaultdict(Counter)
    for iid in ids:
        values = [
            (
                None
                if r[iid]["result"] is None
                else gold[iid] <= set(r[iid]["result"]["hate_type"])
            )
            for r in runs
        ]
        if any((v is None for v in values)):
            status = "invalid"
        else:
            status = "correct" if sum(values) >= 2 else "error"
        statuses[iid] = {"majority_status": status, "contains": values}
        counts[status] += 1
        if all((v is True for v in values)):
            counts["unanimous_correct"] += 1
        if all((v is False for v in values)):
            counts["unanimous_error"] += 1
        for label in gold[iid]:
            per_label[label][status] += 1
    return {
        "n": len(ids),
        "counts": dict(counts),
        "matched_capacity": min(counts["correct"], counts["error"]),
        "label_counts": {k: dict(v) for k, v in per_label.items()},
        "items": statuses,
    }


def batches(ids, gold, n=4):

    bylabel = defaultdict(list)
    for i in ids:
        bylabel[tuple(sorted(gold[i]))].append(i)
    result = [[] for _ in range(n)]
    for labels, items in sorted(bylabel.items(), key=lambda p: (-len(p[1]), p[0])):
        local = [0] * n
        for i in sorted(items, key=order_id):
            j = min(range(n), key=lambda j: (local[j], len(result[j]), j))
            result[j].append(i)
            local[j] += 1
    return [sorted(group) for group in result]


def tests():
    u.require(key(" A\xa0B ") == key("a b"), "Unicode canonical fixture")
    u.require(prefixkey(": a  b") == prefixkey("A B"), "Known prefix fixture")
    ids = ["a", "b", "c", "d", "e", "f"]
    roles = {i: "held_out" if i == "a" else "batches" for i in ids}
    good, ex, groups, _ = eligibility(
        ids, roles, [("a", "b"), ("b", "c"), ("d", "e")], {"d"}
    )
    u.require(
        good == ["f"]
        and ex["c"] == "component_touches_evaluation"
        and (ex["e"] == "component_touches_base_example"),
        "Transitive quarantine fixture",
    )
    good, ex, _, _ = eligibility(
        ["b", "c"], {"b": "batches", "c": "batches"}, [("b", "c")], set()
    )
    u.require(len(good) == 1 and len(ex) == 1, "Candidate-only component dedup fixture")
    g = {"a": {"explicit"}, "b": {"grievance"}, "c": {"explicit", "grievance"}}

    def result(labels):
        return {"result": None if labels is None else {"hate_type": labels}}

    runs = [
        {"a": result(["explicit"]), "b": result([]), "c": result(None)},
        {"a": result([]), "b": result(["grievance"]), "c": result(["explicit"])},
        {
            "a": result(["explicit", "irony"]),
            "b": result([]),
            "c": result(["explicit", "grievance"]),
        },
    ]
    cap = capacity(list(g), g, runs)
    u.require(
        cap["counts"] == {"correct": 1, "error": 1, "invalid": 1}
        and cap["matched_capacity"] == 1,
        "Contains/majority/null fixture",
    )
    split = batches(list(g), g)
    u.require(
        sorted(sum(split, [])) == sorted(g) and split == batches(list(reversed(g)), g),
        "Deterministic batching fixture",
    )
    test = {
        "passed": True,
        "fixtures": 6,
        "created_utc": u.utc(),
        "script_sha256": u.digest(__file__),
    }
    u.save(
        BASE
        / "development"
        / ("feasibility_tests_" + test["script_sha256"][:16] + ".json"),
        test,
    )
    print(json.dumps(test, indent=2))


def run():
    u.require(
        not EVIDENCE.exists() and (not (OUT / "plan.json").exists()),
        "Preserve previous feasibility execution",
    )
    tested = (
        BASE
        / "development"
        / ("feasibility_tests_" + u.digest(__file__)[:16] + ".json")
    )
    test = u.read(tested)
    u.require(
        test["passed"] and test["script_sha256"] == u.digest(__file__),
        "Current passing tests needed",
    )
    paths = [
        Path(__file__),
        tested,
        CORE / "plan.json",
        CORE / "kb/full/records.jsonl",
        CORE / "dev_inputs.jsonl",
        RET / "plan.json",
        RET / "query_vectors.npy",
        RET / "full_vectors.npy",
        RET / "query_vector_sources.jsonl",
        RET / "record_vector_sources.jsonl",
        RET / "full/prompts.jsonl",
        RET / "full/complete.json",
        SCORES / "complete.json",
        ROOT / "data/processed/en_dev_eval_sq3_types.parquet",
        ROOT / "data/processed/en_train.parquet",
        ROOT / "kb/records_sq3_ctl_r4.jsonl",
        ROOT / "scripts/apply_feedback.py",
        ROOT / "scripts/make_sq3_subset.py",
        ROOT / "analysis/common/prompt.py",
        ROOT / "src/hsrag/prompt.py",
        ROOT / "src/hsrag/retrieve.py",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "config/taxonomy.yaml",
    ]
    paths.extend((SCORES / f"assembled/rag_full_rep{i}.jsonl" for i in range(3)))
    hashes = {u.rel(p): u.digest(p) for p in paths}
    core = u.read(CORE / "plan.json")
    ret = u.read(RET / "plan.json")
    done = u.read(SCORES / "complete.json")
    u.require(
        u.digest(CORE / "plan.json")
        == "5c610f4fe320b5e0f185315933d47b388ab645b20b52eed738541a1579184e4e",
        "Core plan changed",
    )
    for p in (CORE / "kb/full/records.jsonl", CORE / "dev_inputs.jsonl"):
        u.require(
            hashes[u.rel(p)] == core["generated_hashes"][u.rel(p)],
            "Core generated file differs",
        )
    for name, value in ret["generated_hashes"].items():
        u.require(hashes[name] == value, "Cached-vector artifact differs")
    u.require(
        hashes[u.rel(RET / "full/prompts.jsonl")]
        == u.read(RET / "full/complete.json")["prompt_sha256"],
        "Base prompts differ",
    )
    for i in range(3):
        p = SCORES / f"assembled/rag_full_rep{i}.jsonl"
        u.require(
            hashes[u.rel(p)] == done["assembled"][u.rel(p)]["sha256"],
            "Core prediction export differs",
        )
    u.save(
        OUT / "plan.json",
        {
            "created_utc": u.utc(),
            "input_hashes": hashes,
            "threshold": THRESHOLD,
            "borderline_margin": MARGIN,
            "screen": "NFKC/casefold/whitespace and leading-colon normalization plus float64 cosine on the existing normalized BGE-M3 vectors; all within-roster pairs and all candidate-to-base example pairs.",
            "quarantine": "Any candidate component touching a protected evaluation role or a base-example match is ineligible; otherwise keep one stable hashed-ID representative per candidate-only component. No gold/prediction-based overlap decision.",
            "evaluation_policy": "Keep original 350 transfer texts and their 229/121 roles; preserve graph clusters for later uncertainty. Do not revise old experiments.",
            "capacity": "Historical full core predictions used only for planning supply, not future correction picks. Majority contains across three valid saved repeats; any whole-output null makes selection ineligible.",
            "scope": "Offline feasibility, no training, new embeddings, live retrieval or classifier calls; no TEST data scoring.",
        },
    )
    df = pd.read_parquet(ROOT / "data/processed/en_dev_eval_sq3_types.parquet")
    rows = {str(r.id): r for r in df.itertuples(index=False)}
    ids = sorted(rows)
    roles = {i: rows[i].sq3_role for i in ids}
    gold = {i: set(rows[i].hate_types) for i in ids}
    candidates = [i for i in ids if roles[i] == "batches"]
    evaluation = [i for i in ids if roles[i] != "batches"]
    u.require(
        len(ids) == 467 and len(candidates) == 117 and (len(evaluation) == 350),
        "Role counts changed",
    )
    records = jl(CORE / "kb/full/records.jsonl")
    dev = jl(CORE / "dev_inputs.jsonl")
    qsrc = jl(RET / "query_vector_sources.jsonl")
    rsrc = jl(RET / "record_vector_sources.jsonl")
    u.require(
        len(records) == 71131
        and len(dev) == len(qsrc) == 1295
        and (len(records) == len(rsrc)),
        "Cached roster differs",
    )
    q = np.load(RET / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    v = np.load(RET / "full_vectors.npy", mmap_mode="r", allow_pickle=False)
    for vectors, n in [(q, len(dev)), (v, len(records))]:
        u.require(
            vectors.dtype == np.float32 and vectors.shape == (n, 1024),
            "Wrong vector shape/type",
        )
        for start in range(0, n, 512):
            arr = np.asarray(vectors[start : start + 512], dtype=np.float64)
            u.require(
                bool(np.isfinite(arr).all())
                and bool(
                    np.allclose(np.linalg.norm(arr, axis=1), 1, atol=0.0002, rtol=0)
                ),
                "Invalid vector",
            )
    dev_ix = {r["key"]: i for i, r in enumerate(dev)}
    for i, (r, s) in enumerate(zip(dev, qsrc)):
        u.require(
            r["key"] == s["key"] and sha_text(r["text"]) == s["text_sha256"],
            "Query vector source identity differs",
        )
    for r, s in zip(records, rsrc):
        u.require(
            r["id"] == s["id"] and sha_text(r["text"]) == s["text_sha256"],
            "Record vector source identity differs",
        )
    for iid in ids:
        u.require(
            dev[dev_ix["en/" + iid]]["text"] == str(rows[iid].text),
            "Input text differs",
        )
    x = np.asarray([q[dev_ix["en/" + i]] for i in ids], dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ix = {i: j for j, i in enumerate(ids)}
    sim = x @ x.T
    edges = []
    near_pairs = []
    for j, iid in enumerate(ids):
        for k in range(j + 1, len(ids)):
            other = ids[k]
            normalized = key(rows[iid].text) == key(rows[other].text)
            prefix = prefixkey(rows[iid].text) == prefixkey(rows[other].text)
            score = float(sim[j, k])
            trigger = score >= THRESHOLD or prefix
            if trigger:
                edges.append((iid, other))
            if trigger or abs(score - THRESHOLD) <= MARGIN:
                independent = float(
                    sum((float(t) * float(s) for t, s in zip(x[j], x[k])))
                )
                u.require(
                    abs(independent - score) < 1e-12, "Independent cosine mismatch"
                )
                near_pairs.append(
                    {
                        "left": iid,
                        "right": other,
                        "left_role": roles[iid],
                        "right_role": roles[other],
                        "cosine": score,
                        "normalized_equal": normalized,
                        "prefix_equal": prefix,
                        "edge": trigger,
                    }
                )
    print(
        f"Checked all {len(ids) * (len(ids) - 1) // 2:,} within-roster pairs; {len(edges)} overlap-screen edges.",
        flush=True,
    )
    cx = x[[ix[i] for i in candidates]]
    best = np.full(len(candidates), -np.inf)
    bestids = [None] * len(candidates)
    base_hits = []
    base_exposed = set()
    basekeys = defaultdict(list)
    for r in records:
        if r["kind"] == "example":
            basekeys[prefixkey(r["text"])].append(r["id"])
    for j, i in enumerate(candidates):
        for rid in basekeys[prefixkey(rows[i].text)]:
            base_hits.append(
                {"candidate": i, "record": rid, "method": "normalized_or_prefix"}
            )
            base_exposed.add(i)
    base_pairs = 0
    for start in range(0, len(records), 512):
        selected = [
            k
            for k in range(start, min(start + 512, len(records)))
            if records[k]["kind"] == "example"
        ]
        if not selected:
            continue
        arr = np.asarray(v[selected], dtype=np.float64)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        sims = cx @ arr.T
        base_pairs += sims.size
        for j, i in enumerate(candidates):
            k = int(sims[j].argmax())
            s = float(sims[j, k])
            if s > best[j]:
                best[j] = s
                bestids[j] = records[selected[k]]["id"]
            for k in np.flatnonzero(sims[j] >= THRESHOLD - MARGIN):
                score = float(sims[j, k])
                independent = float(
                    sum((float(t) * float(s) for t, s in zip(cx[j], arr[k])))
                )
                u.require(abs(independent - score) < 1e-12, "Base cosine mismatch")
                base_hits.append(
                    {
                        "candidate": i,
                        "record": records[selected[k]]["id"],
                        "method": "cosine",
                        "cosine": score,
                        "threshold_match": score >= THRESHOLD,
                    }
                )
                if score >= THRESHOLD:
                    base_exposed.add(i)
    eligible, excluded, groups, cluster = eligibility(ids, roles, edges, base_exposed)
    u.require(
        not set(eligible) & set(evaluation) and (not set(eligible) & base_exposed),
        "Unsafe candidate",
    )
    for group in groups:
        if set(group) & set(evaluation):
            u.require(not set(group) & set(eligible), "Transitive evaluation leak")
        u.require(
            len(set(group) & set(eligible)) <= 1, "Duplicate candidate representative"
        )
    print(
        f"Checked {base_pairs:,} candidate/base pairs. {len(eligible)}/{len(candidates)} candidates survive the frozen screen.",
        flush=True,
    )
    roster = [
        {
            "id": i,
            "key": "en/" + i,
            "text": str(rows[i].text),
            "lang": "en",
            "source": str(rows[i].source),
            "role": roles[i],
            "gold": {"hate": bool(rows[i].gate), "hate_type": sorted(gold[i])},
            "cluster": cluster[i],
            "correction_eligible": i in eligible,
            "exclusion_reason": excluded.get(i),
        }
        for i in ids
    ]
    write_rows(OUT / "roster.jsonl", roster)
    payloads = {r["key"]: r for r in jl(RET / "full/prompts.jsonl")}
    kept = []
    from analysis.common import prompt as prompt

    full_lookup = {r["id"]: r for r in records}
    for i in ids:
        item = payloads["en/" + i]
        context = item["context"]
        examples = [h for _, rr in context["example_groups"] for h in rr]
        u.require(
            len(examples) == 5
            and len(context["definitions"]) == len(context["guidelines"]) == 2,
            "Base context budget differs",
        )
        u.require(
            all(
                (
                    full_lookup[h["id"]]["text"] == h["text"]
                    for h in context["definitions"] + context["guidelines"] + examples
                )
            ),
            "Context source text differs",
        )
        u.require(
            prompt.render_messages(str(rows[i].text), "en", context, "corrected")
            == item["messages"],
            "Saved baseline render differs",
        )
        u.require(
            signature(item["messages"]) == item["messages_sha256"]
            and signature(context) == item["context_sha256"],
            "Prompt digest differs",
        )
        kept.append(item)
    write_rows(OUT / "baseline_prompts.jsonl", kept)
    runs = []
    for j in range(3):
        output = jl(SCORES / f"assembled/rag_full_rep{j}.jsonl")
        run = {r["id"]: r for r in output}
        u.require(
            len(run) == len(output) == 467 and set(run) == set(ids),
            "Historical prediction identities differ",
        )
        u.require(
            all((r["condition"] == "rag_full" and r["replicate"] == j for r in output)),
            "Historical condition differs",
        )
        runs.append(run)
    eligible_capacity = capacity(eligible, gold, runs)
    all_capacity = capacity(candidates, gold, runs)
    batch_ids = batches(eligible, gold)
    batch_capacity = [capacity(group, gold, runs) for group in batch_ids]
    old_controls = [
        r
        for r in jl(ROOT / "kb/records_sq3_ctl_r4.jsonl")
        if r["kind"] == "feedback" and r["meta"]["origin_item_id"] not in rows
    ]
    topups = [
        {
            "origin": r["meta"]["origin_item_id"],
            "old_feedback_id": r["id"],
            "full_text_matches": basekeys[prefixkey(r["text"])],
        }
        for r in old_controls
    ]
    data_train = pd.read_parquet(ROOT / "data/processed/en_train.parquet")
    train_byid = {str(r.id): r for r in data_train.itertuples(index=False)}
    for r in old_controls:
        u.require(
            r["text"] == str(train_byid[r["meta"]["origin_item_id"]].text),
            "Old control source differs",
        )
    supports = lambda selected: dict(Counter((l for i in selected for l in gold[i])))
    known = [
        p
        for p in near_pairs
        if {p["left"], p["right"]} == {"implicit_hate-21540", "implicit_hate-7570"}
    ]
    u.require(
        len(known) == 1 and known[0]["edge"] and ("implicit_hate-21540" in excluded),
        "Known prefix duplicate not quarantined",
    )
    result = {
        "status": "feasibility_complete",
        "created_utc": u.utc(),
        "input_hashes": hashes,
        "plan": u.rel(OUT / "plan.json"),
        "plan_sha256": u.digest(OUT / "plan.json"),
        "tests": test,
        "counts": {
            "roster": len(ids),
            "original_candidates": len(candidates),
            "eligible_candidates": len(eligible),
            "evaluation": len(evaluation),
            "full_records": len(records),
            "base_example_pairs": base_pairs,
            "within_roster_pairs": len(ids) * (len(ids) - 1) // 2,
            "overlap_edges": len(edges),
            "baseline_payloads_verified": len(kept),
        },
        "role_counts": dict(Counter(roles.values())),
        "role_support": {
            role: supports([i for i in ids if roles[i] == role])
            for role in sorted(set(roles.values()))
        },
        "eligible_ids": eligible,
        "eligible_support": supports(eligible),
        "excluded_candidates": excluded,
        "near_pairs": near_pairs,
        "base_near_pairs": base_hits,
        "candidate_nearest_base": [
            {"id": i, "record": bestids[j], "cosine": float(best[j])}
            for j, i in enumerate(candidates)
        ],
        "components": groups,
        "evaluation_clusters": len({cluster[i] for i in evaluation}),
        "historical_topups": topups,
        "historical_capacity_all_candidates": all_capacity,
        "historical_capacity_screened_candidates": eligible_capacity,
        "historical_four_batch_diagnostic": {
            "batch_ids": batch_ids,
            "capacities": batch_capacity,
            "total_minimum_per_batch_capacity": sum(
                (r["matched_capacity"] for r in batch_capacity)
            ),
        },
        "outputs": {
            u.rel(p): u.digest(p)
            for p in (OUT / "roster.jsonl", OUT / "baseline_prompts.jsonl")
        },
        "limits": [
            "Heuristic overlap screen, not a proof of semantic independence. Query/base embeddings are reused from the prior validated core caches.",
            "Historical capacity counts inform planning only; future selected corrections require the new specified baseline. No present full-KB adaptation performance result.",
            "DEV texts and results have been inspected before; holdout means not eligible for correction selection, not researcher-unseen TEST.",
            "The existing corrected from_retrieval helper omits feedback; the new prompt assembly must explicitly add the neutral feedback group and test its presence.",
        ],
        "new_model_calls": 0,
        "new_embeddings": False,
        "new_retrieval": False,
        "new_training": False,
        "new_TEST_scoring": False,
    }
    u.verify(hashes)
    u.save(EVIDENCE, result)
    u.save(
        OUT / "complete.json",
        {
            "status": result["status"],
            "evidence": u.rel(EVIDENCE),
            "evidence_sha256": u.digest(EVIDENCE),
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "exclusions": excluded,
                "historical_capacity": {
                    k: v for k, v in eligible_capacity.items() if k != "items"
                },
                "historical_batch_capacities": [
                    {
                        "n": r["n"],
                        "counts": r["counts"],
                        "matched_capacity": r["matched_capacity"],
                    }
                    for r in batch_capacity
                ],
                "topups_existing_in_full": sum(
                    (bool(r["full_text_matches"]) for r in topups)
                ),
                "topups_total": len(topups),
                "near_pairs": near_pairs,
                "base_near_pairs": base_hits,
                "evidence_sha256": u.digest(EVIDENCE),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["tests", "run"])
    args = p.parse_args()
    with patch.object(socket.socket, "connect", u.forbid), patch.object(
        socket, "create_connection", u.forbid
    ):
        tests() if args.mode == "tests" else run()
