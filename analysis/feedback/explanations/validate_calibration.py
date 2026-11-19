from src.hsrag.paths import PROJECT_ROOT
import os

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "2"
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import socket
import time
from types import SimpleNamespace
import unicodedata
from unittest.mock import patch
import numpy as np
from rank_bm25 import BM25Okapi

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_selective_calibration_20261001_v1"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
FB = ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1/update_payloads_v1"
ACTIVE = (
    ROOT
    / "experiments/astra/sq3_full_kb_relevance_20260930_v1/api_execution_v1/plan.json"
)
SEED = "20261001-applicability-v1"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def jl(path):
    with path.open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def h(purpose, value):
    return digest(SEED + "|" + purpose + "|" + value)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def original_bucket_class():

    filename = ROOT / "analysis/feedback/source/retrieval.py"
    tree = ast.parse(filename.read_text(encoding="utf8"))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FeedbackBucket"
    ]
    assert len(classes) == 1
    module = ast.Module(body=classes, type_ignores=[])
    env = {
        "np": np,
        "re": re,
        "BM25Okapi": BM25Okapi,
        "design": SimpleNamespace(u=SimpleNamespace(require=require)),
    }
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), env)
    return env["FeedbackBucket"]


def stats(rows, t):
    accept = np.array([r["cosine"] >= t / 100 for r in rows], dtype=bool)
    y = np.array([r["compatible"] for r in rows], dtype=bool)
    hate = np.array([r["train_gate"] for r in rows], dtype=bool)
    tp = int(np.sum(accept & y))
    fp = int(np.sum(accept & ~y))
    tn = int(np.sum(~accept & ~y))
    fn = int(np.sum(~accept & y))
    b = int(np.sum(~hate))
    ht = int(np.sum(hate))
    ba = int(np.sum(accept & ~hate))
    ha = int(np.sum(accept & hate))
    d = 5 * tp + fn + 4 * fp
    return {
        "pairs": len(rows),
        "unique_inputs": len({r["query_id"] for r in rows}),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accepted": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f0_5": 5 * tp / d if d else 0.0,
        "coverage": (tp + fp) / len(rows) if rows else None,
        "proxy_positive_pairs": tp + fn,
        "benign_pairs": b,
        "benign_accepted": ba,
        "benign_acceptance": ba / b if b else None,
        "typed_pairs": ht,
        "typed_accepted": ha,
        "typed_acceptance": ha / ht if ht else None,
    }


def check():
    start = time.perf_counter()
    assert not (BASE / "complete.json").exists(), "Preserve completed verification"
    receipt = read(BASE / "prepared.json")
    for name, expected in receipt["outputs"].items():
        assert sha(BASE / name) == expected, name
    plan = read(BASE / "plan.json")
    for name, expected in plan["inputs"].items():
        assert sha(ROOT / name) == expected, name
    records = jl(CORE / "kb/full/records.jsonl")
    dev = jl(CORE / "dev_inputs.jsonl")
    fv = np.load(
        CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    qv = np.load(
        CORE / "retrieval_v1/query_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    assert fv.shape == (71131, 1024) and qv.shape == (1295, 1024)
    for array in (fv, qv):
        assert array.dtype == np.dtype("float32")
        for i in range(0, len(array), 4096):
            a = np.array(array[i : i + 4096], dtype=np.float64)
            assert np.isfinite(a).all() and np.all(
                np.abs(np.linalg.norm(a, axis=1) - 1) < 0.0002
            )
    for rows, src, key in (
        (records, jl(CORE / "retrieval_v1/record_vector_sources.jsonl"), "id"),
        (dev, jl(CORE / "retrieval_v1/query_vector_sources.jsonl"), "key"),
    ):
        assert len(rows) == len(src)
        assert all(
            r[key] == s[key] and digest(r["text"]) == s["text_sha256"]
            for r, s in zip(rows, src)
        )
    eligible = defaultdict(list)
    sigs = {}
    unknown = 0
    for i, r in enumerate(records):
        if (r["kind"], r["lang"], r["source"]) != (
            "example",
            "en",
            "implicit_hate-train",
        ):
            continue
        gate = r["meta"].get("gate")
        ts = r["meta"].get("hate_types")
        if gate is False:
            assert ts in (None, [])
            s = "benign"
        elif gate is True and ts:
            assert len(ts) == len(set(ts)) and set(ts) <= {
                "explicit",
                "incitement",
                "inferiority",
                "irony",
                "stereotyping",
                "threatening",
                "grievance",
            }
            s = "hate:" + ",".join(sorted(ts))
        else:
            unknown += 1
            continue
        sigs[r["id"]] = s
        eligible["benign" if not gate else "hate"].append((r["id"], i))
    expected = []
    for group in ("benign", "hate"):
        assert len(eligible[group]) >= 1024
        expected += sorted(eligible[group], key=lambda v: (h("sample", v[0]), v[0]))[
            :1024
        ]
    expected.sort()
    sampled = jl(BASE / "sampled_train.jsonl")
    assert [(r["id"], r["record_row"]) for r in sampled] == expected and len(
        sampled
    ) == 2048
    for r in sampled:
        raw = records[r["record_row"]]
        assert r["text"] == raw["text"] and r["text_sha256"] == digest(raw["text"])
        assert r["train_gate"] == raw["meta"]["gate"] and r["train_types"] == raw[
            "meta"
        ].get("hate_types")
        assert r["signature"] == sigs[r["id"]] and r["sample_hash"] == h(
            "sample", r["id"]
        )
    x = np.array(fv[[r["record_row"] for r in sampled]], dtype=np.float64)
    x /= np.linalg.norm(x, axis=1)[:, None]
    v = np.array(qv, dtype=np.float64)
    v /= np.linalg.norm(v, axis=1)[:, None]
    tt = np.clip(x @ x.T, -1, 1)
    td = np.clip(x @ v.T, -1, 1)

    def nk(s):
        return (
            re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).casefold())
            .strip()
            .lstrip(": ")
        )

    tn = [nk(r["text"]) for r in sampled]
    dn = [nk(r["text"]) for r in dev]
    lookup = defaultdict(list)
    for j, n in enumerate(dn):
        lookup[n].append(j)
    expected_evidence = {}
    graph = defaultdict(set)
    touched = defaultdict(set)

    def evidence(kind, i, j, cos, exact):
        left = sampled[i]["id"]
        right = sampled[j]["id"] if kind == "train_train" else dev[j]["key"]
        edge = bool(exact or cos >= 0.95)
        expected_evidence[(kind, left, right)] = {
            "cosine": float(cos),
            "exact": exact,
            "edge": edge,
            "borderline": abs(cos - 0.95) <= 0.0005,
        }
        return edge

    for i in range(len(sampled)):
        candidates = set(np.flatnonzero(td[i] >= 0.9495).tolist()) | set(
            lookup.get(tn[i], [])
        )
        for j in candidates:
            if evidence("train_dev", i, j, td[i, j], tn[i] == dn[j]):
                touched[i].add(dev[j]["key"])
        candidates = set(np.flatnonzero(tt[i] >= 0.9495).tolist()) | {
            j for j, n in enumerate(tn) if tn[i] == n
        }
        for j in candidates:
            if j <= i:
                continue
            if evidence("train_train", i, j, tt[i, j], tn[i] == tn[j]):
                graph[i].add(j)
                graph[j].add(i)
    observed = jl(BASE / "overlap_evidence.jsonl")
    assert len(observed) == len(expected_evidence)
    for r in observed:
        e = expected_evidence.pop((r["kind"], r["left"], r["right"]))
        assert abs(r["cosine"] - e["cosine"]) < 1e-12
        assert all(r[k] == e[k] for k in ("exact", "edge", "borderline"))
    assert not expected_evidence
    visited = set()
    groups = []
    for i in range(len(sampled)):
        if i in visited:
            continue
        todo = [i]
        group = set()
        while todo:
            current = todo.pop()
            if current in group:
                continue
            group.add(current)
            visited.add(current)
            todo.extend(graph[current] - group)
        groups.append(sorted(group))
    key = lambda i: (h("sample", sampled[i]["id"]), sampled[i]["id"])
    groups.sort(key=lambda g: min(key(i) for i in g))
    expected_groups = []
    reps = []
    for gi, g in enumerate(groups):
        p = sorted({k for i in g for k in touched[i]})
        conflict = len({sampled[i]["signature"] for i in g}) > 1
        why = (
            "protected_dev_overlap"
            if p
            else "conflicting_train_annotations" if conflict else None
        )
        representative = None if why else min(g, key=key)
        expected_groups.append(
            {
                "members": g,
                "protected_keys": p,
                "annotation_conflict": conflict,
                "excluded_reason": why,
                "representative": representative,
            }
        )
        if representative is not None:
            reps.append((gi, representative))
    assert read(BASE / "components.json") == expected_groups
    roster = jl(BASE / "retained_train.jsonl")
    rmap = {r["id"]: r for r in roster}
    assert len(roster) == len(rmap) == len(reps)
    assert set(rmap) == {sampled[i]["id"] for _, i in reps}
    for gi, i in reps:
        r = rmap[sampled[i]["id"]]
        assert r["component"] == gi and r["sample_index"] == i
        assert all(r[k] == v for k, v in sampled[i].items())
    strata = defaultdict(list)
    for r in roster:
        strata[r["signature"]].append(r)
    for signature, rows in strata.items():
        first = int(h("split-start", signature), 16) % 2
        for index, r in enumerate(
            sorted(rows, key=lambda r: (h("split", r["id"]), r["id"]))
        ):
            assert r["partition"] == ("calibration", "review")[(index + first) % 2]
            assert r["split_hash"] == h("split", r["id"])
    pools = sorted(
        [
            p
            for p in read(FB / "pools.json")["pools"]
            if p["arm"] == "error" and p["size"] == 40
        ],
        key=lambda p: p["seed"],
    )
    assert [p["seed"] for p in pools] == [202609281, 202609282, 202609283]
    added = {r["id"]: r for r in jl(FB / "added_records.jsonl")}
    packet = {
        r["source_id"]: r
        for r in jl(
            ROOT
            / "experiments/astra/sq3_weekend_preparation_20261001_v2/explanation_source_packet.jsonl"
        )
    }
    assert {iid for p in pools for iid in p["ids"]} == set(packet) and len(packet) == 58
    devmap = {r["key"]: i for i, r in enumerate(dev)}
    buckets = {}
    cls = original_bucket_class()
    for p in pools:
        rs = [added[rid] for rid in p["record_ids"]]
        vs = []
        for iid, r in zip(p["ids"], rs):
            j = devmap["en/" + iid]
            src = packet[iid]
            assert (
                r["id"] == "sq3fb-" + iid and r["text"] == src["text"] == dev[j]["text"]
            )
            assert r["meta"]["gate"] is src["hate"] is True
            assert r["meta"]["hate_types"] == src["hate_types"]
            assert r["provenance"]["vector_row"] == j
            assert (
                r["provenance"]["vector_sha256"]
                == hashlib.sha256(qv[j].tobytes()).hexdigest()
            )
            vs.append(qv[j])
        buckets[p["seed"]] = cls(rs, np.array(vs))
    pairs = jl(BASE / "train_pairs.jsonl")
    seen = set()
    max_score_error = 0.0
    assert len(pairs) == 3 * len(roster)
    for n, p in enumerate(pairs, 1):
        identity = p["query_id"], p["pool_seed"]
        assert identity not in seen
        seen.add(identity)
        r = rmap[p["query_id"]]
        bucket = buckets[p["pool_seed"]]
        selected, trace = bucket.retrieve(r["text"], fv[r["record_row"]])
        index = bucket.ids.index(selected["id"])
        q = np.array(fv[r["record_row"]], dtype=np.float64)
        cos = float(bucket.vectors[index] @ (q / np.linalg.norm(q)))
        assert abs(cos - p["cosine"]) < 1e-12
        assert (
            p["selected_id"] == selected["id"]
            and p["selected_types"] == selected["meta"]["hate_types"]
        )
        assert (
            p["partition"] == r["partition"]
            and p["train_gate"] == r["train_gate"]
            and p["train_types"] == r["train_types"]
        )
        y = (
            r["train_gate"] is True
            and bool(r["train_types"])
            and set(r["train_types"]) <= set(selected["meta"]["hate_types"])
        )
        assert (
            p["compatible"] == y
            and p["retrieval"]["selected_id"] == trace["selected_id"]
        )
        for name in ("dense", "bm25", "rrf"):
            assert [t["id"] for t in trace[name]] == [
                t["id"] for t in p["retrieval"][name]
            ]
            for a, b in zip(trace[name], p["retrieval"][name]):
                error = abs(a["score"] - b["score"])
                max_score_error = max(max_score_error, error)
                assert error < 1e-10
        if n % 1500 == 0:
            print(
                "Verified natural TRAIN retrieval pairs", n, "/", len(pairs), flush=True
            )
    assert seen == {(r["id"], p["seed"]) for r in roster for p in pools}
    calibration = [r for r in pairs if r["partition"] == "calibration"]
    table = []
    objectives = []
    for t in list(range(-100, 101)) + [200]:
        m = stats(calibration, t)
        table.append({"cutoff_hundredths": t, **m})
        denominator = 5 * m["tp"] + m["fn"] + 4 * m["fp"]
        objectives.append(
            (Fraction(5 * m["tp"], denominator) if denominator else Fraction(0), t)
        )
    assert table == read(BASE / "calibration_threshold_table.json")
    cutoff = max(objectives)[1]
    policy = read(BASE / "policy.json")
    summary = read(BASE / "summary.json")
    assert (
        policy["cutoff_hundredths"] == cutoff
        and policy["cutoff"] == summary["cutoff"] == cutoff / 100
    )
    assert policy["plan_sha256"] == sha(BASE / "plan.json") and policy[
        "threshold_table_sha256"
    ] == sha(BASE / "calibration_threshold_table.json")
    assert policy["all_zero_objective"] == all(x[0] == 0 for x in objectives)
    assert (
        summary["eligible_counts"] == {k: len(v) for k, v in eligible.items()}
        and summary["excluded_unknown_annotations"] == unknown
    )
    assert (
        summary["sampled_inputs"] == 2048
        and summary["retained_inputs"] == len(roster)
        and summary["train_pairs"] == len(pairs)
    )
    assert summary["partition_gate_counts"] == dict(
        Counter(
            r["partition"] + ("/hate" if r["train_gate"] else "/benign") for r in roster
        )
    )
    assert summary["excluded_input_counts"] == dict(
        Counter(
            g["excluded_reason"]
            for g in expected_groups
            for _ in g["members"]
            if g["excluded_reason"]
        )
    )
    assert summary["collapsed_extra_inputs"] == sum(
        len(g["members"]) - 1
        for g in expected_groups
        if g["representative"] is not None
    )
    assert summary["overlap_evidence_rows"] == len(observed) and summary[
        "borderline_rows"
    ] == sum(r["borderline"] for r in observed)
    for part in ("calibration", "review"):
        selected = [p for p in pairs if p["partition"] == part]
        for name, t in (("selected", cutoff), ("always", -100), ("never", 200)):
            expected = {
                "pooled": stats(selected, t),
                "by_pool": {
                    str(p["seed"]): stats(
                        [r for r in selected if r["pool_seed"] == p["seed"]], t
                    )
                    for p in pools
                },
            }
            assert summary["summaries"][part][name] == expected
    active = read(ACTIVE)
    count = 0
    assert (
        sha(ACTIVE)
        == "d4d34f90adf144131e3e59a13cb9c6cd351b7eefc97a093ee71613ee75a0babb"
    )
    for field in ("input_hashes", "source_hashes", "generated_hashes"):
        for name, expected in active[field].items():
            assert sha(ROOT / name) == expected
            count += 1
    assert count == 79
    result = {
        "status": "training_only_calibration_independently_verified",
        "prepared_sha256": sha(BASE / "prepared.json"),
        "checker_sha256": sha(Path(__file__)),
        "policy_sha256": sha(BASE / "policy.json"),
        "verified_input_bindings": len(plan["inputs"]),
        "active_bindings_verified": count,
        "sampled_inputs": 2048,
        "retained_inputs": len(roster),
        "reconstructed_retrieval_pairs": len(pairs),
        "max_retrieval_score_error": max_score_error,
        "cutoff": cutoff / 100,
        "classifier_calls": 0,
        "seconds": time.perf_counter() - start,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (BASE / "complete.json").open("x", encoding="utf8", newline="\n") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    with patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("Offline check: network forbidden"),
    ):
        check()
