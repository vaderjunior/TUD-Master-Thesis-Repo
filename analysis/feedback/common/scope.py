import os

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"
import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.common import runtime as c

BASE = c.SHARED / "scope"
PLAN = BASE / "plan.json"


def consumed_inputs():

    bound = {}

    def consume(path, mapping):
        name = c.rel(path)
        c.require(name in mapping, "Missing historical input binding: " + name)
        c.require(
            c.sha(path) == mapping[name], "Historical consumed input changed: " + name
        )
        bound[name] = mapping[name]

    parents = [
        (c.CANDIDATES / "prepared.json", "outputs"),
        (c.SELECTION / "complete.json", "outputs"),
        (c.PREP / "complete.json", "outputs"),
        (c.LATER / "retrieval_v3/full/freeze/complete.json", "outputs"),
        (c.LATER / "retrieval_v3/vectors/complete.json", "outputs"),
        (c.CORE / "retrieval_v1/plan.json", "generated_hashes"),
        (c.CORE / "plan.json", "generated_hashes"),
        (c.A15 / "execution/plan.json", "input_hashes"),
    ]
    wanted = [
        c.CANDIDATES / n
        for n in (
            "roster.jsonl",
            "query_vectors.npy",
            "query_vector_sources.jsonl",
            "prompts.jsonl",
            "scope_details.json",
            "scope_complete.json",
        )
    ]
    wanted += [
        c.SELECTION / "actual_pools.json",
        c.SELECTION / "gate_only_records.json",
        c.PREP / "broad_roster.jsonl",
        c.LATER / "retrieval_v3/full/prompts.jsonl",
        c.LATER / "retrieval_v3/vectors/query_vectors.npy",
        c.LATER / "retrieval_v3/vectors/query_origins.jsonl",
        c.CORE / "retrieval_v1/full_vectors.npy",
        c.CORE / "retrieval_v1/query_vectors.npy",
        c.CORE / "retrieval_v1/query_vector_sources.jsonl",
        c.CORE / "dev_inputs.jsonl",
        c.CORE / "kb/full/records.jsonl",
        c.A15 / "roster.jsonl",
    ]
    mapping = {}
    for path, field in parents:
        d = c.read(path)
        for name, h in d[field].items():
            c.require(
                name not in mapping or mapping[name] == h, "Conflicting parent binding"
            )
            mapping[name] = h
        bound[c.rel(path)] = c.sha(path)
    for path in wanted:
        consume(path, mapping)
    broad_done = c.read(c.LATER / "retrieval_v3/complete.json")
    for path in [
        c.LATER / "replay_review_20261009_v1/sensitivity_plan.json",
        c.LATER / "replay_review_20261009_v1/review.json",
    ]:
        (
            consume(path, broad_done["outputs"])
            if c.rel(path) in broad_done["outputs"]
            else consume(path, {broad_done["review"]: broad_done["review_sha256"]})
        )
    for path in [
        c.LATER / "retrieval_v3/complete.json",
        c.SHARED / "SCOPE_PROTOCOL.md",
        Path(__file__),
        Path(c.__file__),
    ]:
        bound[c.rel(path)] = c.sha(path)
    return bound


def prepare():
    if PLAN.exists():
        c.verify(c.read(PLAN)["inputs"])
        c.verify(c.read(PLAN)["prepared"])
        print("Shared scope inputs already frozen.")
        return
    start = time.perf_counter()
    print(c.resources(2.0, 3.0), flush=True)
    guard = c.active_guard()
    inputs = consumed_inputs()
    broad = list(c.rows(c.PREP / "broad_roster.jsonl"))
    origins = list(c.rows(c.LATER / "retrieval_v3/vectors/query_origins.jsonl"))
    bvec = np.load(
        c.LATER / "retrieval_v3/vectors/query_vectors.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    c.require(
        len(broad) == len(origins) == 21627 and bvec.shape == (21627, 1024),
        "Broad vector roster differs",
    )
    selected = []
    vectors = []
    for i, r in enumerate(broad):
        if r["phase"] not in ("en_test", "en_hatecheck_eval"):
            continue
        c.require(
            origins[i]["key"] == r["key"]
            and origins[i]["exact_text_sha256"] == c.text_sha(r["text"]),
            "Broad exact vector identity differs",
        )
        c.require(
            type(r["gate"]) is bool and "hate" in r["endpoints"],
            "Missing evaluation gate",
        )
        selected.append(
            dict(
                r,
                role="evaluation",
                node_id="evaluation|" + r["key"],
                vector_origin=origins[i],
            )
        )
        vectors.append(bvec[i])
    c.require(len(selected) == 15769, "Transfer universe differs")
    sources = list(c.rows(c.CANDIDATES / "roster.jsonl"))
    sorigins = list(c.rows(c.CANDIDATES / "query_vector_sources.jsonl"))
    sv = np.load(c.CANDIDATES / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    c.require(
        len(sources) == len(sorigins) == 1024 and sv.shape == (1024, 1024),
        "Source vectors differ",
    )
    for i, r in enumerate(sources):
        c.require(
            sorigins[i]["id"] == r["id"]
            and sorigins[i]["text_sha256"] == c.text_sha(r["text"]),
            "Candidate exact vector identity differs",
        )
        selected.append(
            dict(
                r,
                role="source",
                node_id="source|" + r["key"],
                gate=r["gold"]["hate"],
                vector_origin=sorigins[i],
            )
        )
        vectors.append(sv[i])
    dev = list(c.rows(c.A15 / "roster.jsonl"))
    core = list(c.rows(c.CORE / "dev_inputs.jsonl"))
    ci = {r["key"]: i for i, r in enumerate(core)}
    co = list(c.rows(c.CORE / "retrieval_v1/query_vector_sources.jsonl"))
    cv = np.load(
        c.CORE / "retrieval_v1/query_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    c.require(len(dev) == 675 and cv.shape == (1295, 1024), "Development scope differs")
    for r in dev:
        i = ci[r["key"]]
        c.require(
            core[i]["text"] == r["text"]
            and co[i]["key"] == r["key"]
            and (co[i]["text_sha256"] == c.text_sha(r["text"])),
            "DEV exact vector identity differs",
        )
        selected.append(
            dict(
                r,
                role="development",
                node_id="development|" + r["key"],
                gate=r["gold"]["hate"],
                vector_origin=co[i],
            )
        )
        vectors.append(cv[i])
    x = np.stack(vectors)
    c.require(
        x.shape == (17468, 1024) and x.dtype == np.float32 and np.isfinite(x).all(),
        "Shared vectors malformed",
    )
    c.require(
        np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=0.0002),
        "Shared vector norms malformed",
    )
    train = []
    for i, r in enumerate(c.rows(c.CORE / "kb/full/records.jsonl")):
        if r["lang"] == "en" and r["kind"] == "example":
            train.append(
                dict(
                    id=r["id"],
                    text=r["text"],
                    vector_index=i,
                    aliases=list(
                        r["meta"].get("astra_train_ids", [r["id"].removeprefix("ex-")])
                    ),
                )
            )
    c.require(len(train) == 55753, "TRAIN scope differs")
    c.jsonlines(BASE / "records.jsonl", selected)
    c.array(BASE / "vectors.npy", x)
    c.jsonlines(BASE / "train_texts.jsonl", train)
    c.save(
        PLAN,
        {
            "created_utc": c.utc(),
            "inputs": inputs,
            "prepared": c.bindings(
                [
                    BASE / "records.jsonl",
                    BASE / "vectors.npy",
                    BASE / "train_texts.jsonl",
                ]
            ),
            "active_A15": guard,
            "scope": {
                "evaluation": 15769,
                "candidates": 1024,
                "development": 675,
                "train": 55753,
            },
            "threshold": c.NEAR,
            "precision": "float64 normalized cached FP32 vectors; exact dense block products, no ANN",
            "chunk_rows": 256,
            "new_model_calls": 0,
            "assembly_seconds": time.perf_counter() - start,
        },
    )
    print(
        "Shared scope frozen: 17,468 nodes; 55,753 TRAIN references; no outcomes or models.",
        flush=True,
    )


def matrices():
    x = np.asarray(np.load(BASE / "vectors.npy", allow_pickle=False), dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    train = list(c.rows(BASE / "train_texts.jsonl"))
    allv = np.load(
        c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    t = np.asarray(allv[[r["vector_index"] for r in train]], dtype=np.float64)
    c.require(
        np.isfinite(t).all()
        and np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=0.0002),
        "TRAIN vectors malformed",
    )
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return (x, t)


def calculate(limit=0):
    p = c.read(PLAN)
    c.verify(p["inputs"])
    c.verify(p["prepared"])
    print(c.resources(2.0, 2.0), flush=True)
    x, t = matrices()
    start = time.perf_counter()
    chunks = []
    for offset in range(0, len(x), 256):
        path = BASE / "chunks" / f"block_{offset:06d}.json"
        end = min(offset + 256, len(x))
        if path.exists():
            d = c.read(path)
            c.require(
                d["plan_sha256"] == c.sha(PLAN)
                and d["start"] == offset
                and (d["end"] == end),
                "Chunk identity differs",
            )
        else:
            began = time.perf_counter()
            scores = x[offset:end] @ t.T
            nearest = np.argmax(scores, axis=1)
            maximum = scores[np.arange(end - offset), nearest].tolist()
            del scores
            scores = x[offset:end] @ x.T
            edges = []
            for local, j in zip(*np.nonzero(scores >= c.NEAR)):
                i = offset + int(local)
                if i < int(j):
                    edges.append([i, int(j), float(scores[local, j])])
            del scores
            d = {
                "plan_sha256": c.sha(PLAN),
                "start": offset,
                "end": end,
                "nearest_train_indices": nearest.tolist(),
                "nearest_train_cosines": maximum,
                "edges": edges,
                "seconds": time.perf_counter() - began,
                "new_model_calls": 0,
            }
            c.save(path, d)
        chunks.append(path)
        if offset == 0 or end == len(x) or end % 2048 == 0:
            print(
                f"Scope exact overlaps {end}/{len(x)}; elapsed {time.perf_counter() - start:.1f}s; no Qwen.",
                flush=True,
            )
        if limit and len(chunks) >= limit:
            print(
                {
                    "pilot_blocks": len(chunks),
                    "seconds": time.perf_counter() - start,
                    "approx_remaining_seconds": d["seconds"] * (len(x) - end) / 256,
                },
                flush=True,
            )
            return
    c.save(
        BASE / "calculated.json",
        {
            "status": "all_exact_scope_blocks_saved",
            "plan_sha256": c.sha(PLAN),
            "outputs": c.bindings(chunks),
            "nodes": len(x),
            "new_model_calls": 0,
        },
    )


def exact_edges(nodes):
    groups = defaultdict(list)
    for i, r in enumerate(nodes):
        groups["normal|" + c.norm(r["text"])].append(i)
        groups["prefix|" + c.prefix(r["text"])].append(i)
        groups["id|" + str(r["id"])].append(i)
    return sorted({(g[0], i) for g in groups.values() for i in g[1:]})


def finalize():
    if (BASE / "complete.json").exists():
        c.verify(c.read(BASE / "complete.json")["outputs"])
        print("Shared scope already verified.")
        return
    p = c.read(PLAN)
    c.verify(p["inputs"])
    c.verify(p["prepared"])
    calc = c.read(BASE / "calculated.json")
    c.verify(calc["outputs"])
    nodes = list(c.rows(BASE / "records.jsonl"))
    train = list(c.rows(BASE / "train_texts.jsonl"))
    n = len(nodes)
    parent = c.Union(n)
    edges = []
    train_values = {}
    for path in calc["outputs"]:
        d = c.read(c.ROOT / path)
        for i, j, value in d["edges"]:
            parent.join(i, j)
            edges.append((i, j))
        for k, (j, value) in enumerate(
            zip(d["nearest_train_indices"], d["nearest_train_cosines"])
        ):
            train_values[d["start"] + k] = (j, value)
    c.require(set(train_values) == set(range(n)), "Incomplete nearest-TRAIN matrix")
    ex = exact_edges(nodes)
    for i, j in ex:
        parent.join(i, j)
    aliases = {s for r in train for s in [r["id"], *r["aliases"]]}
    norms = {c.norm(r["text"]) for r in train}
    prefixes = {c.prefix(r["text"]) for r in train}
    train_flags = {
        i: dict(
            identity=str(r["id"]) in aliases,
            normalized=c.norm(r["text"]) in norms,
            prefix=c.prefix(r["text"]) in prefixes,
            near=train_values[i][1] >= c.NEAR,
        )
        for i, r in enumerate(nodes)
    }
    c.require(
        not any(
            (
                any(train_flags[i].values())
                for i, r in enumerate(nodes)
                if r["role"] == "source"
            )
        ),
        "Previously separated feedback source touches TRAIN",
    )
    adjacency = [set() for _ in range(n)]
    for i, j in [*edges, *ex]:
        adjacency[i].add(j)
        adjacency[j].add(i)
    unseen = set(range(n))
    components = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        group = []
        todo = deque([first])
        while todo:
            i = todo.popleft()
            group.append(i)
            for j in adjacency[i] & unseen:
                unseen.remove(j)
                todo.append(j)
        components.append(sorted(group))
    c.require(
        sorted(map(sorted, parent.groups())) == sorted(components),
        "Independent connected components differ",
    )
    data = []
    eligible = []
    reasons = Counter()
    component_records = []
    for group in components:
        roles = {nodes[i]["role"] for i in group}
        why = []
        if "source" in roles:
            why.append("component_touches_feedback_candidates")
        if "development" in roles:
            why.append("component_touches_A15_development")
        if any((any(train_flags[i].values()) for i in group)):
            why.append("component_touches_TRAIN")
        cid = "sq3follow-" + c.sig(sorted((nodes[i]["node_id"] for i in group)))[:20]
        component_records.append(
            {
                "cluster": cid,
                "nodes": [nodes[i]["node_id"] for i in group],
                "exclusion_reasons": why,
            }
        )
        for i in group:
            r = nodes[i]
            j, similarity = train_values[i]
            row = {
                "node_id": r["node_id"],
                "key": r["key"],
                "role": r["role"],
                "cluster": cid,
                "nearest_train_id": train[j]["id"],
                "train_cosine": similarity,
                "direct_train_flags": train_flags[i],
                "exclusion_reasons": why,
                "retained": r["role"] == "evaluation" and (not why),
            }
            data.append(row)
            if r["role"] == "evaluation":
                if why:
                    reasons.update(why)
                else:
                    eligible.append(dict(r, overlap_cluster=cid, scope_node_index=i))
    eligible.sort(key=lambda r: r["key"])
    c.require(
        eligible and {r["phase"] for r in eligible} == {"en_test", "en_hatecheck_eval"},
        "Empty transfer phase",
    )
    ix = {r["key"]: i for i, r in enumerate(eligible)}
    inf = c.Union(len(eligible))
    families = defaultdict(list)
    for i, r in enumerate(eligible):
        families[r["phase"] + "|overlap|" + r["overlap_cluster"]].append(i)
        if r["phase"] == "en_hatecheck_eval":
            c.require(
                r.get("functional_cluster") is not None,
                "HateCheck functional cluster missing",
            )
            families["functional|" + str(r["functional_cluster"])].append(i)
    for group in families.values():
        for j in group[1:]:
            inf.join(group[0], j)
    for group in inf.groups():
        cid = "sq3infer-" + c.sig(sorted((eligible[i]["key"] for i in group)))[:20]
        for i in group:
            eligible[i]["inference_cluster"] = cid
    sample = []
    for phase in ("en_test", "en_hatecheck_eval"):
        selected = c.hash_sample(
            [r for r in eligible if r["phase"] == phase],
            1024,
            "A17-evaluation-20261012",
            "inference_cluster",
        )
        sample.extend(selected)
    replay = c.read(c.LATER / "replay_review_20261009_v1/sensitivity_plan.json")[
        "exclusion_roster"
    ]
    c.require(len(replay) == 27, "Historical broader replay sensitivity changed")
    replay_keys = sorted({r["key"] for r in replay} & set(ix))
    c.jsonlines(BASE / "eligibility.jsonl", sorted(data, key=lambda r: r["node_id"]))
    c.save(BASE / "components.json", component_records)
    c.jsonlines(BASE / "evaluation.jsonl", eligible)
    c.save(
        BASE / "iteration_evaluation.json",
        {
            "keys": sorted(sample),
            "rule": "Whole inference clusters, identity hash, target >=1024 per phase",
            "outcome_selected": False,
        },
    )
    c.save(
        BASE / "sensitivity.json",
        {
            "replay_keys": replay_keys,
            "parent_replay_keys": [r["key"] for r in replay],
            "policy": "Common exclusions across all conditions/methods; primary keeps qualified first-pass prompts",
        },
    )
    summary = {
        "status": "scope_ready_for_independent_numerical_check",
        "universe": dict(Counter((r["role"] for r in nodes))),
        "eligible_by_phase": dict(Counter((r["phase"] for r in eligible))),
        "eligible_gate_counts": dict(
            Counter((r["phase"] + "/" + str(r["gate"]) for r in eligible))
        ),
        "eligible_sources": dict(
            Counter((r["phase"] + "/" + r["source"] for r in eligible))
        ),
        "excluded_evaluation": 15769 - len(eligible),
        "exclusion_reason_counts_nonexclusive": dict(reasons),
        "cosine_edges": len(edges),
        "exact_identity_edges": len(ex),
        "components": len(components),
        "inference_clusters": dict(
            Counter((eligible[g[0]]["phase"] for g in inf.groups()))
        ),
        "iteration_sample_counts": dict(
            Counter((eligible[ix[k]]["phase"] for k in sample))
        ),
        "replay_sensitivity_inputs": len(replay_keys),
        "A16_prompts": len(eligible) * 4,
        "A16_predictions": len(eligible) * 12,
        "A16_ordinary_calls": len(eligible) * 36,
        "new_model_calls": 0,
        "CUDA_used": False,
        "active_sources_unchanged": c.active_guard()["plan_sha256"]
        == p["active_A15"]["plan_sha256"],
    }
    c.save(BASE / "summary.json", summary)
    paths = [
        BASE / n
        for n in (
            "eligibility.jsonl",
            "components.json",
            "evaluation.jsonl",
            "iteration_evaluation.json",
            "sensitivity.json",
            "summary.json",
        )
    ]
    c.save(
        BASE / "prepared.json",
        {"plan_sha256": c.sha(PLAN), "outputs": c.bindings(paths)},
    )
    print(summary, flush=True)


def check():
    if (BASE / "complete.json").exists():
        c.verify(c.read(BASE / "complete.json")["outputs"])
        print("Independent scope receipt verified.")
        return
    p = c.read(PLAN)
    c.verify(p["inputs"])
    c.verify(p["prepared"])
    c.verify(c.read(BASE / "prepared.json")["outputs"])
    calc = c.read(BASE / "calculated.json")
    c.verify(calc["outputs"])
    x, t = matrices()
    expected_edges = set()
    original = {}
    for path in calc["outputs"]:
        d = c.read(c.ROOT / path)
        expected_edges.update(((i, j) for i, j, _ in d["edges"]))
        for k, (j, v) in enumerate(
            zip(d["nearest_train_indices"], d["nearest_train_cosines"])
        ):
            original[d["start"] + k] = (j, v)
    observed = set()
    max_difference = 0.0
    start = time.perf_counter()
    for startrow in range(0, len(x), 127):
        end = min(startrow + 127, len(x))
        batch = x[startrow:end]
        score = np.einsum("ik,jk->ij", batch, t, optimize=True)
        mx = score.max(axis=1)
        for k, val in enumerate(mx):
            i = startrow + k
            j, old = original[i]
            difference = abs(float(val) - old)
            max_difference = max(max_difference, difference)
            c.require(
                difference < 1e-12 and (val >= c.NEAR) == (old >= c.NEAR),
                "Independent nearest-TRAIN threshold differs",
            )
            c.require(
                abs(float(score[k, j]) - float(val)) < 1e-12,
                "Saved nearest TRAIN is not a maximizer",
            )
        del score
        score = np.einsum("ik,jk->ij", batch, x, optimize=True)
        for k, j in zip(*np.nonzero(score >= c.NEAR)):
            i = startrow + int(k)
            if i < int(j):
                observed.add((i, int(j)))
        del score
        if startrow == 0 or end == len(x) or startrow % (127 * 16) == 0:
            print(
                f"Independent overlap check {end}/{len(x)} ({time.perf_counter() - start:.1f}s)",
                flush=True,
            )
    c.require(observed == expected_edges, "Independent full cosine edge set differs")
    c.active_guard()
    c.verify(p["inputs"])
    c.verify(p["prepared"])
    outputs = c.read(BASE / "prepared.json")["outputs"] | {
        c.rel(PLAN): c.sha(PLAN),
        c.rel(BASE / "prepared.json"): c.sha(BASE / "prepared.json"),
        c.rel(BASE / "calculated.json"): c.sha(BASE / "calculated.json"),
    }
    c.save(
        BASE / "complete.json",
        {
            "status": "independent_exact_scope_verified",
            "outputs": outputs,
            "numerical_checked_nodes": len(x),
            "max_nearest_cosine_difference": max_difference,
            "independent_edges": len(observed),
            "independent_connectivity": "adjacency BFS agrees with union-find",
            "seconds": time.perf_counter() - start,
            "Qwen_calls": 0,
            "CUDA_used": False,
        },
    )
    print("Shared transfer scope verified; no model calls.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "mode", choices=["prepare", "pilot", "calculate", "finalize", "check", "all"]
    )
    args = ap.parse_args()
    with c.offline(), threadpool_limits(limits=1):
        if args.mode == "prepare":
            prepare()
        elif args.mode == "pilot":
            calculate(limit=1)
        elif args.mode == "calculate":
            calculate()
        elif args.mode == "finalize":
            finalize()
        elif args.mode == "check":
            check()
        else:
            prepare()
            calculate()
            finalize()
            check()


if __name__ == "__main__":
    main()
