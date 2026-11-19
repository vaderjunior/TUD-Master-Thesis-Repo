from src.hsrag.paths import PROJECT_ROOT
import os

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "2"
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import socket
import time
import unicodedata
from unittest.mock import patch
import numpy as np

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_selective_calibration_20261001_v1"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
FEEDBACK = (
    ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1/update_payloads_v1"
)
PACKET = ROOT / "experiments/astra/sq3_weekend_preparation_20261001_v2"
ACTIVE = (
    ROOT
    / "experiments/astra/sq3_full_kb_relevance_20260930_v1/api_execution_v1/plan.json"
)
SEED = "20261001-applicability-v1"
LABELS = {
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
}
GRID = list(range(-100, 101)) + [200]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def rank_hash(purpose, value):
    return digest(SEED + "|" + purpose + "|" + value)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path):
    with path.open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, value):
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        f.write("\n")


def jsonlines(path, rows):
    with path.open("x", encoding="utf8", newline="\n") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def rel(path):
    return path.relative_to(ROOT).as_posix()


def verify_active():
    p = read(ACTIVE)
    require(
        sha(ACTIVE)
        == "d4d34f90adf144131e3e59a13cb9c6cd351b7eefc97a093ee71613ee75a0babb",
        "Active plan changed",
    )
    n = 0
    for key in ("input_hashes", "source_hashes", "generated_hashes"):
        for name, expected in p[key].items():
            require(sha(ROOT / name) == expected, "Active dependency changed: " + name)
            n += 1
    require(n == 79, "Unexpected active binding count")
    return n


def signature(record):
    m = record["meta"]
    gate, types = m.get("gate"), m.get("hate_types")
    if gate is False:
        require(types is None or types == [], "Benign TRAIN record with positive types")
        return "benign"
    if gate is True and isinstance(types, list) and types:
        require(
            set(types) <= LABELS and len(types) == len(set(types)),
            "Invalid observed type set",
        )
        return "hate:" + ",".join(sorted(types))
    return None


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).lstrip(": ")


def unit(vectors):
    a = np.asarray(vectors, dtype=np.float64)
    require(a.ndim == 2 and np.isfinite(a).all(), "Nonfinite or non-matrix vectors")
    norms = np.linalg.norm(a, axis=1)
    require(
        np.all(np.abs(norms - 1) < 0.0002), "Cached vectors not normalized as expected"
    )
    return a / norms[:, None]


def components(n, edges, touched, signatures, order_keys):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        x, y = find(a), find(b)
        parent[max(x, y)] = min(x, y)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = []
    for members in sorted(groups.values(), key=lambda g: min(order_keys[i] for i in g)):
        protected = sorted({k for i in members for k in touched.get(i, [])})
        conflict = len({signatures[i] for i in members}) > 1
        why = (
            "protected_dev_overlap"
            if protected
            else "conflicting_train_annotations" if conflict else None
        )
        representative = None if why else min(members, key=lambda i: order_keys[i])
        out.append(
            {
                "members": members,
                "protected_keys": protected,
                "annotation_conflict": conflict,
                "excluded_reason": why,
                "representative": representative,
            }
        )
    return out


class LocalBucket:
    """Explicit Okapi/RRF; checker uses the original rank_bm25-based class."""

    def __init__(self, records, vectors):
        order = sorted(range(len(records)), key=lambda i: records[i]["id"])
        self.records = [records[i] for i in order]
        self.ids = [r["id"] for r in self.records]
        require(
            len(self.ids) == len(set(self.ids)) and 1 <= len(self.ids) <= 40, "Bad pool"
        )
        x = np.asarray(vectors, dtype=np.float64)[order]
        self.vectors = x / np.linalg.norm(x, axis=1)[:, None]
        self.counts = [
            Counter(re.findall(r"\w+", r["text"].lower())) for r in self.records
        ]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg = sum(self.lengths) / len(self.ids)
        require(self.avg > 0, "Empty corpus")
        df = Counter(token for c in self.counts for token in c)
        n = len(self.ids)
        self.idf = {w: math.log(n - f + 0.5) - math.log(f + 0.5) for w, f in df.items()}
        floor = 0.25 * sum(self.idf.values()) / len(self.idf)
        self.idf = {w: (v if v >= 0 else floor) for w, v in self.idf.items()}

    def retrieve(self, text, vector):
        q = np.asarray(vector, dtype=np.float64)
        dense = self.vectors @ (q / np.linalg.norm(q))
        lexical = [0.0] * len(self.ids)
        for token in re.findall(r"\w+", text.lower()):
            for i, c in enumerate(self.counts):
                f = c.get(token, 0)
                lexical[i] += self.idf.get(token, 0.0) * (
                    f * 2.5 / (f + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.avg))
                )
        d = sorted(range(len(self.ids)), key=lambda i: (-float(dense[i]), self.ids[i]))[
            :10
        ]
        b = sorted(range(len(self.ids)), key=lambda i: (-lexical[i], self.ids[i]))[:10]
        b = [i for i in b if lexical[i] > 0]
        fused = {}
        for ranking in (d, b):
            for rank, i in enumerate(ranking, 1):
                fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
        final = sorted(fused, key=lambda i: (-fused[i], self.ids[i]))
        chosen = final[0]
        return (
            self.records[chosen],
            float(dense[chosen]),
            {
                "dense": [{"id": self.ids[i], "score": float(dense[i])} for i in d],
                "bm25": [{"id": self.ids[i], "score": lexical[i]} for i in b],
                "rrf": [{"id": self.ids[i], "score": fused[i]} for i in final],
                "selected_id": self.ids[chosen],
            },
        )


def measure(rows, cutoff):
    tp = fp = tn = fn = benign_accept = typed_accept = 0
    for r in rows:
        accept = r["cosine"] >= cutoff / 100
        if accept and r["compatible"]:
            tp += 1
        elif accept:
            fp += 1
        elif r["compatible"]:
            fn += 1
        else:
            tn += 1
        benign_accept += int(accept and not r["train_gate"])
        typed_accept += int(accept and r["train_gate"])
    n = len(rows)
    benign = sum(not r["train_gate"] for r in rows)
    typed = n - benign
    numerator, denominator = 5 * tp, 5 * tp + fn + 4 * fp
    return {
        "pairs": n,
        "unique_inputs": len({r["query_id"] for r in rows}),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accepted": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f0_5": numerator / denominator if denominator else 0.0,
        "coverage": (tp + fp) / n if n else None,
        "proxy_positive_pairs": tp + fn,
        "benign_pairs": benign,
        "benign_accepted": benign_accept,
        "benign_acceptance": benign_accept / benign if benign else None,
        "typed_pairs": typed,
        "typed_accepted": typed_accept,
        "typed_acceptance": typed_accept / typed if typed else None,
    }


def choose(rows):
    require(
        rows and all(r["partition"] == "calibration" for r in rows),
        "Only calibration pairs may fit cutoff",
    )
    table = []
    best = None
    for t in GRID:
        m = measure(rows, t)
        den = 5 * m["tp"] + m["fn"] + 4 * m["fp"]
        ratio = Fraction(5 * m["tp"], den) if den else Fraction(0)
        table.append({"cutoff_hundredths": t, **m})
        if best is None or (ratio, t) > best:
            best = ratio, t
    return best[1], table


def freeze():
    require(not (BASE / "plan.json").exists(), "Do not replace frozen plan")
    require(read(BASE / "tests_passed.json")["successful"], "Synthetic tests required")
    n = verify_active()
    paths = [
        CORE / "plan.json",
        CORE / "kb/full/records.jsonl",
        CORE / "dev_inputs.jsonl",
        CORE / "retrieval_v1/plan.json",
        CORE / "retrieval_v1/full_vectors.npy",
        CORE / "retrieval_v1/query_vectors.npy",
        CORE / "retrieval_v1/record_vector_sources.jsonl",
        CORE / "retrieval_v1/query_vector_sources.jsonl",
        FEEDBACK / "pools.json",
        FEEDBACK / "added_records.jsonl",
        PACKET / "explanation_source_packet.jsonl",
        PACKET / "prepared.json",
        ACTIVE,
        BASE / "CALIBRATION_PROTOCOL.md",
        BASE / "tests_passed.json",
    ]
    paths += [
        ROOT / "scratch/analysis" / name
        for name in (
            "astra_sq3_selective_calibration_v1.py",
            "astra_sq3_selective_calibration_check_v1.py",
            "test_astra_sq3_selective_calibration_v1.py",
            "astra_sq3_feedback_retrieval_v1.py",
        )
    ]
    bindings = {rel(p): sha(p) for p in paths}
    rp = read(CORE / "retrieval_v1/plan.json")
    for path in paths[4:8]:
        require(
            bindings[rel(path)] == rp["generated_hashes"][rel(path)],
            "Cache provenance mismatch",
        )
    cp = read(CORE / "plan.json")
    for path in paths[1:3]:
        require(
            bindings[rel(path)] == cp["generated_hashes"][rel(path)],
            "Core data provenance mismatch",
        )
    write(
        BASE / "plan.json",
        {
            "status": "frozen_before_calibration_scores",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": bindings,
            "seed": SEED,
            "per_gate_quota": 1024,
            "protected_core_dev_count": 1295,
            "expected_full_records": 71131,
            "near_cosine": 0.95,
            "borderline_margin": 0.0005,
            "grid_hundredths": GRID,
            "active_bindings_verified": n,
            "objective": "pooled_pair_F0.5_then_higher_cutoff",
            "proxy": "hateful_and_train_types_subset_of_selected_types",
            "environment": {
                "numpy": np.__version__,
                "rank_bm25": importlib.metadata.version("rank-bm25"),
                "python": os.sys.version,
                "blas_threads": 2,
            },
            "classifier_calls": 0,
            "evaluation_results_read": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "calibration_plan_frozen",
                "inputs": len(bindings),
                "active_bindings": n,
            }
        )
    )


def build():
    start = time.perf_counter()
    plan = read(BASE / "plan.json")
    require(
        not (BASE / "sampled_train.jsonl").exists(), "Preserve earlier build attempt"
    )
    for name, expected in plan["inputs"].items():
        require(sha(ROOT / name) == expected, "Frozen input changed: " + name)
    verify_active()
    records = lines(CORE / "kb/full/records.jsonl")
    dev = lines(CORE / "dev_inputs.jsonl")
    rsrc = lines(CORE / "retrieval_v1/record_vector_sources.jsonl")
    qsrc = lines(CORE / "retrieval_v1/query_vector_sources.jsonl")
    fv = np.load(
        CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    qv = np.load(
        CORE / "retrieval_v1/query_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    require(
        fv.shape == (71131, 1024) and qv.shape == (1295, 1024), "Cache shapes changed"
    )
    require(fv.dtype == qv.dtype == np.dtype("float32"), "Cache dtype changed")
    require(
        len(records) == len(rsrc) == len(fv) and len(dev) == len(qsrc) == len(qv),
        "Cache rows changed",
    )
    require(len({r["id"] for r in records}) == len(records), "Repeated KB IDs")
    require(len({r["key"] for r in dev}) == len(dev), "Repeated protected IDs")
    for r, s in zip(records, rsrc):
        require(
            r["id"] == s["id"] and digest(r["text"]) == s["text_sha256"],
            "KB vector alignment",
        )
    for r, s in zip(dev, qsrc):
        require(
            r["key"] == s["key"] and digest(r["text"]) == s["text_sha256"],
            "DEV vector alignment",
        )
    eligible = {"benign": [], "hate": []}
    excluded_untyped = 0
    for row, r in enumerate(records):
        if (
            r["kind"] != "example"
            or r["lang"] != "en"
            or r["source"] != "implicit_hate-train"
        ):
            continue
        s = signature(r)
        if s is None:
            excluded_untyped += 1
            continue
        eligible["benign" if s == "benign" else "hate"].append((row, r, s))
    sampled = []
    for stratum, items in eligible.items():
        require(len(items) >= 1024, "Insufficient fixed TRAIN quota: " + stratum)
        selected = sorted(
            items, key=lambda x: (rank_hash("sample", x[1]["id"]), x[1]["id"])
        )[:1024]
        for row, r, s in selected:
            sampled.append(
                {
                    "id": r["id"],
                    "record_row": row,
                    "text": r["text"],
                    "text_sha256": digest(r["text"]),
                    "train_gate": r["meta"]["gate"],
                    "train_types": r["meta"].get("hate_types"),
                    "signature": s,
                    "stratum": stratum,
                    "sample_hash": rank_hash("sample", r["id"]),
                }
            )
    sampled.sort(key=lambda r: r["id"])
    jsonlines(BASE / "sampled_train.jsonl", sampled)
    x = unit(fv[[r["record_row"] for r in sampled]])
    v = unit(qv)
    c_dev = np.clip(x @ v.T, -1, 1)
    c_train = np.clip(x @ x.T, -1, 1)
    nk = [normalize(r["text"]) for r in sampled]
    dk = [normalize(r["text"]) for r in dev]
    require(all(nk) and all(dk), "Empty normalized input")
    evidence, edges, touched = [], [], defaultdict(list)
    for i in range(len(sampled)):
        for j in range(len(dev)):
            sim = float(c_dev[i, j])
            exact = nk[i] == dk[j]
            edge = exact or sim >= 0.95
            if edge or sim >= 0.9495:
                evidence.append(
                    {
                        "kind": "train_dev",
                        "left": sampled[i]["id"],
                        "right": dev[j]["key"],
                        "cosine": sim,
                        "exact": exact,
                        "edge": edge,
                        "borderline": abs(sim - 0.95) <= 0.0005,
                    }
                )
            if edge:
                touched[i].append(dev[j]["key"])
        for j in range(i + 1, len(sampled)):
            sim = float(c_train[i, j])
            exact = nk[i] == nk[j]
            edge = exact or sim >= 0.95
            if edge or sim >= 0.9495:
                evidence.append(
                    {
                        "kind": "train_train",
                        "left": sampled[i]["id"],
                        "right": sampled[j]["id"],
                        "cosine": sim,
                        "exact": exact,
                        "edge": edge,
                        "borderline": abs(sim - 0.95) <= 0.0005,
                    }
                )
            if edge:
                edges.append((i, j))
    groups = components(
        len(sampled),
        edges,
        touched,
        [r["signature"] for r in sampled],
        [(r["sample_hash"], r["id"]) for r in sampled],
    )
    roster = []
    for gi, g in enumerate(groups):
        for i in g["members"]:
            sampled[i]["component"] = gi
        if g["representative"] is not None:
            i = g["representative"]
            roster.append({**sampled[i], "sample_index": i})
    strata = defaultdict(list)
    for r in roster:
        strata[r["signature"]].append(r)
    for s, rows in strata.items():
        start_partition = int(rank_hash("split-start", s), 16) % 2
        for i, r in enumerate(
            sorted(rows, key=lambda r: (rank_hash("split", r["id"]), r["id"]))
        ):
            r["partition"] = ("calibration", "review")[(i + start_partition) % 2]
            r["split_hash"] = rank_hash("split", r["id"])
    roster.sort(key=lambda r: r["id"])
    require(
        {(r["partition"], r["train_gate"]) for r in roster}
        == {
            ("calibration", True),
            ("calibration", False),
            ("review", True),
            ("review", False),
        },
        "Empty split stratum",
    )
    jsonlines(BASE / "overlap_evidence.jsonl", evidence)
    write(BASE / "components.json", groups)
    jsonlines(BASE / "retained_train.jsonl", roster)
    pools = [
        p
        for p in read(FEEDBACK / "pools.json")["pools"]
        if p["arm"] == "error" and p["size"] == 40
    ]
    pools.sort(key=lambda p: p["seed"])
    require(
        [p["seed"] for p in pools] == [202609281, 202609282, 202609283], "Pool changes"
    )
    added = {r["id"]: r for r in lines(FEEDBACK / "added_records.jsonl")}
    packet = {
        r["source_id"]: r for r in lines(PACKET / "explanation_source_packet.jsonl")
    }
    di = {r["key"]: i for i, r in enumerate(dev)}
    require(
        {iid for p in pools for iid in p["ids"]} == set(packet) and len(packet) == 58,
        "Correction source changes",
    )
    buckets = []
    for p in pools:
        rs = [added[rid] for rid in p["record_ids"]]
        rows = []
        for iid, r in zip(p["ids"], rs):
            pk = packet[iid]
            j = di["en/" + iid]
            require(
                r["id"] == "sq3fb-" + iid and r["meta"]["gate"] is True,
                "Correction ID/gate mismatch",
            )
            require(
                r["text"] == pk["text"] == dev[j]["text"], "Correction text mismatch"
            )
            require(
                r["meta"]["hate_types"] == pk["hate_types"] and pk["hate"] is True,
                "Correction labels mismatch",
            )
            require(
                r["provenance"]["vector_row"] == j, "Correction vector row mismatch"
            )
            require(
                r["provenance"]["vector_sha256"]
                == hashlib.sha256(qv[j].tobytes()).hexdigest(),
                "Correction vector mismatch",
            )
            rows.append(j)
        buckets.append((p, LocalBucket(rs, qv[rows])))
    pairs = []
    for n, r in enumerate(roster, 1):
        for p, bucket in buckets:
            selected, cosine, trace = bucket.retrieve(r["text"], fv[r["record_row"]])
            types = set(r["train_types"] or [])
            compatible = (
                r["train_gate"]
                and bool(types)
                and types <= set(selected["meta"]["hate_types"])
            )
            pairs.append(
                {
                    "query_id": r["id"],
                    "partition": r["partition"],
                    "pool_seed": p["seed"],
                    "train_gate": r["train_gate"],
                    "train_types": r["train_types"],
                    "selected_id": selected["id"],
                    "selected_types": selected["meta"]["hate_types"],
                    "cosine": cosine,
                    "compatible": bool(compatible),
                    "retrieval": trace,
                }
            )
        if n % 500 == 0:
            print("Ranked TRAIN inputs", n, "/", len(roster), flush=True)
    jsonlines(BASE / "train_pairs.jsonl", pairs)
    calibration = [p for p in pairs if p["partition"] == "calibration"]
    cutoff, table = choose(calibration)
    write(BASE / "calibration_threshold_table.json", table)
    policy = {
        "status": "cutoff_frozen_before_internal_review_summary",
        "cutoff_hundredths": cutoff,
        "cutoff": cutoff / 100,
        "feature": "cosine_to_unchanged_natural_hybrid_correction",
        "accept_operator": ">=",
        "skip_behavior": "exact_base_prompt_no_reselection",
        "pool_seeds": [p["seed"] for p in pools],
        "calibration_inputs": len(calibration) // 3,
        "plan_sha256": sha(BASE / "plan.json"),
        "threshold_table_sha256": sha(BASE / "calibration_threshold_table.json"),
        "fit_objective": "F0.5_compatibility_proxy",
        "evaluation_outcomes_used": False,
        "all_zero_objective": all(t["f0_5"] == 0 for t in table),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write(BASE / "policy.json", policy)
    summaries = {}
    for part in ("calibration", "review"):
        summaries[part] = {}
        selected = [p for p in pairs if p["partition"] == part]
        for name, t in (("selected", cutoff), ("always", -100), ("never", 200)):
            summaries[part][name] = {
                "pooled": measure(selected, t),
                "by_pool": {
                    str(p["seed"]): measure(
                        [r for r in selected if r["pool_seed"] == p["seed"]], t
                    )
                    for p in pools
                },
            }
    summary = {
        "status": "calibration_built_independent_check_pending",
        "cutoff": cutoff / 100,
        "eligible_counts": {k: len(v) for k, v in eligible.items()},
        "excluded_unknown_annotations": excluded_untyped,
        "sampled_inputs": len(sampled),
        "retained_inputs": len(roster),
        "train_pairs": len(pairs),
        "partition_gate_counts": dict(
            Counter(
                r["partition"] + ("/hate" if r["train_gate"] else "/benign")
                for r in roster
            )
        ),
        "excluded_input_counts": dict(
            Counter(
                g["excluded_reason"]
                for g in groups
                for _ in g["members"]
                if g["excluded_reason"]
            )
        ),
        "collapsed_extra_inputs": sum(
            len(g["members"]) - 1 for g in groups if g["representative"] is not None
        ),
        "overlap_evidence_rows": len(evidence),
        "borderline_rows": sum(r["borderline"] for r in evidence),
        "summaries": summaries,
        "classifier_calls": 0,
        "cuda_jobs": 0,
        "test_scores": 0,
        "active_bindings_verified": verify_active(),
        "seconds": time.perf_counter() - start,
    }
    write(BASE / "summary.json", summary)
    outputs = {
        p.name: sha(p)
        for p in BASE.iterdir()
        if p.is_file() and p.name not in ("prepared.json", "failure.json")
    }
    write(
        BASE / "prepared.json",
        {
            "status": "independent_check_pending",
            "outputs": outputs,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "status",
                    "cutoff",
                    "sampled_inputs",
                    "retained_inputs",
                    "train_pairs",
                    "seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["--freeze", "--build", "freeze", "build"])

    args = parser.parse_args([a.lstrip("-") for a in os.sys.argv[1:]])
    with patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("Offline calibration: network forbidden"),
    ):
        if args.action == "freeze":
            freeze()
        else:
            build()
