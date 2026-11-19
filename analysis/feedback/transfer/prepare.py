import os

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.common import runtime as c
from analysis.common import prompt as renderer
from analysis.common import feedback_inputs as tokens
from analysis.feedback.reserved.common import intervene
from analysis.feedback.source.retrieval import FeedbackBucket
from analysis.feedback.explanations.calibration import LocalBucket

BASE = c.TRANSFER
SCOPE = c.SHARED / "scope"


def load():
    scope = c.read(SCOPE / "complete.json")
    c.require(
        scope["status"] == "independent_exact_scope_verified",
        "Shared scope not verified",
    )
    c.verify(scope["outputs"])
    rows = list(c.rows(SCOPE / "evaluation.jsonl"))
    wanted = {r["key"] for r in rows}
    bases = {
        r["key"]: r
        for r in c.rows(c.LATER / "retrieval_v3/full/prompts.jsonl")
        if r["key"] in wanted
    }
    pools = [
        p
        for p in c.read(c.SELECTION / "actual_pools.json")["pools"]
        if p["selection"] == "error" and p["polarity"] == "balanced"
    ]
    records = c.read(c.SELECTION / "gate_only_records.json")
    sources = list(c.rows(c.CANDIDATES / "roster.jsonl"))
    si = {r["id"]: i for i, r in enumerate(sources)}
    sv = np.load(c.CANDIDATES / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    qv = np.load(SCOPE / "vectors.npy", mmap_mode="r", allow_pickle=False)
    c.require(
        len(bases) == len(rows)
        and len(pools) == 3
        and all((len(p["ids"]) == 40 for p in pools)),
        "A16 sources/matrix differ",
    )
    return (rows, bases, pools, records, si, sv, qv)


def sources():
    paths = [
        SCOPE / "complete.json",
        SCOPE / "plan.json",
        SCOPE / "evaluation.jsonl",
        SCOPE / "vectors.npy",
        SCOPE / "sensitivity.json",
        c.SELECTION / "actual_pools.json",
        c.SELECTION / "gate_only_records.json",
        c.CANDIDATES / "roster.jsonl",
        c.CANDIDATES / "query_vectors.npy",
        c.LATER / "retrieval_v3/full/prompts.jsonl",
        BASE / "STUDY_PROTOCOL.md",
        Path(__file__),
        Path(c.__file__),
        c.ROOT / "analysis/feedback/reserved/common.py",
        c.ROOT / "analysis/feedback/source/retrieval.py",
        c.ROOT / "analysis/feedback/explanations/calibration.py",
        c.ROOT / "analysis/common/prompt.py",
        c.ROOT / "src/hsrag/prompt.py",
        c.ROOT / "prompts/classify_v1.txt",
        c.ROOT / "config/taxonomy.yaml",
        tokens.TOK / "complete.json",
        tokens.TOK / "tokenizer.json",
    ]
    return c.bindings(paths)


def prepare():
    if (BASE / "prepared.json").exists():
        c.verify(c.read(BASE / "preparation_plan.json")["inputs"])
        c.verify(c.read(BASE / "prepared.json")["outputs"])
        print("A16 frozen prompts verified.")
        return
    c.require(not (BASE / "execution/live").exists(), "Preserve A16 live work")
    c.resources(2.0, 3.0)
    c.active_guard()
    rows, bases, pools, records, si, sv, qv = load()
    inputs = sources()
    for name, h in c.read(SCOPE / "plan.json")["inputs"].items():
        if name in inputs:
            c.require(inputs[name] == h, "A16 parent binding differs")
    if (BASE / "preparation_plan.json").exists():
        c.require(
            c.read(BASE / "preparation_plan.json")["inputs"] == inputs,
            "Interrupted A16 preparation inputs differ",
        )
    else:
        c.save(
            BASE / "preparation_plan.json",
            {
                "inputs": inputs,
                "created_utc": c.utc(),
                "new_model_calls": 0,
                "A15_outcomes_read": False,
            },
        )
    conditions = ["reference"] + [p["condition"] for p in pools]
    n = len(rows)
    labels = {}
    for phase in ("en_test", "en_hatecheck_eval"):
        rr = [r for r in rows if r["phase"] == phase]
        labels[phase] = {}
        for field in ("hate_types", "target_groups", "severity", "legal"):
            eligible = [
                r
                for r in rr
                if r.get(field) is not None and (field == "legal" or r["gate"] is True)
            ]
            support = Counter()
            for r in eligible:
                value = r[field]
                support.update(value if isinstance(value, list) else [value])
            labels[phase][field] = {
                "rows": len(eligible),
                "positive_support": dict(support),
                "support_ge_10": sorted((k for k, v in support.items() if v >= 10)),
            }
    science = {
        "conditions": conditions,
        "inputs": n,
        "predictions": n * 12,
        "ordinary_calls": n * 36,
        "pool_seeds": [p["seed"] for p in pools],
        "decoder_repeats": 3,
        "votes_per_prediction": 3,
        "primary": {
            "endpoints": [
                "en_test_FPR",
                "en_test_FNR",
                "en_hatecheck_eval_FPR",
                "en_hatecheck_eval_FNR",
            ],
            "confidence": 0.9875,
            "bootstrap_draws": 10000,
            "bootstrap_seeds": list(range(2026101216, 2026101220)),
        },
        "fixed_support": labels,
        "mandatory_replay_keys": c.read(SCOPE / "sensitivity.json")["replay_keys"],
        "protocol_sha256": c.sha(BASE / "STUDY_PROTOCOL.md"),
        "preparation_plan_sha256": c.sha(BASE / "preparation_plan.json"),
        "exposure": "Previously exposed English TEST; HateCheck functional diagnostics; no A15-dependent selection",
    }
    c.save(BASE / "scientific_plan_v1.json", science)
    science_sha = c.sha(BASE / "scientific_plan_v1.json")
    c.jsonlines(BASE / "roster.jsonl", (dict(r, population=r["phase"]) for r in rows))
    buckets = {
        p["condition"]: FeedbackBucket(
            [records[i] for i in p["ids"]], sv[[si[i] for i in p["ids"]]]
        )
        for p in pools
    }
    tok = tokens.tokenizer()
    counts = Counter()
    ranges = []
    choices = []

    def generate():
        for number, r in enumerate(rows, 1):
            old = bases[r["key"]]
            q = qv[r["scope_node_index"]]
            c.require(
                old["item_id"] == r["id"]
                and old["messages"]
                == renderer.render_messages(
                    r["text"], "en", old["context"], "corrected"
                ),
                "Base text/identity/render differs",
            )
            for cond in conditions:
                selected = None
                context = old["context"]
                messages = old["messages"]
                if cond != "reference":
                    record, trace = buckets[cond].retrieve(r["text"], q)
                    selected = record["provenance"]["source_id"]
                    hit = {
                        "id": record["id"],
                        "text": record["text"],
                        "meta": record["meta"],
                        "via": "rrf",
                    }
                    context = intervene(old["context"], "reserved5", hit)
                    messages = renderer.render_messages(
                        r["text"], "en", context, "corrected"
                    )
                    source = np.asarray(sv[si[selected]], dtype=np.float64)
                    query = np.asarray(q, dtype=np.float64)
                    similarity = float(
                        np.dot(source, query)
                        / (np.linalg.norm(source) * np.linalg.norm(query))
                    )
                    c.require(similarity < c.NEAR, "Correction overlaps transfer input")
                    choices.append(
                        {
                            "key": r["key"],
                            "condition": cond,
                            "source_id": selected,
                            "source_gate": record["meta"]["gate"],
                            "cosine": similarity,
                            "retrieval": trace,
                        }
                    )
                count = tokens.token_count(tok, messages)
                ranges.append(count)
                counts[cond] += 1
                yield {
                    "key": r["key"],
                    "item_id": r["id"],
                    "condition": cond,
                    "population": r["phase"],
                    "lang": "en",
                    "text": r["text"],
                    "context": context,
                    "context_sha256": c.sig(context),
                    "messages": messages,
                    "messages_sha256": c.sig(messages),
                    "source_id": selected,
                    "reference_input_tokens": count,
                    "scientific_plan_sha256": science_sha,
                }
            if number % 1000 == 0 or number == n:
                print(
                    f"A16 frozen prompts: {number}/{n} inputs; zero model calls.",
                    flush=True,
                )

    c.jsonlines(BASE / "prompts.jsonl", generate())
    c.jsonlines(BASE / "choices.jsonl", choices)
    c.save(
        BASE / "summary.json",
        {
            "status": "prepared_requires_independent_verification",
            "inputs": n,
            "conditions": 4,
            "prompts": n * 4,
            "predictions": n * 12,
            "ordinary_calls": n * 36,
            "conditions_counts": dict(counts),
            "token_min": min(ranges),
            "token_max": max(ranges),
            "new_model_calls": 0,
        },
    )
    c.verify(inputs)
    c.active_guard()
    c.save(
        BASE / "prepared.json",
        {
            "status": "prepared_requires_independent_check",
            "outputs": c.bindings(
                (
                    BASE / p
                    for p in [
                        "roster.jsonl",
                        "scientific_plan_v1.json",
                        "prompts.jsonl",
                        "choices.jsonl",
                        "summary.json",
                    ]
                )
            ),
            "created_utc": c.utc(),
        },
    )


def check():
    plan = c.read(BASE / "preparation_plan.json")
    c.verify(plan["inputs"])
    c.verify(c.read(BASE / "prepared.json")["outputs"])
    if (BASE / "complete.json").exists():
        done = c.read(BASE / "complete.json")
        c.require(
            done["prepared_sha256"] == c.sha(BASE / "prepared.json")
            and done["verification_sha256"] == c.sha(c.ROOT / done["verification"]),
            "A16 receipt changed",
        )
        print("A16 independent prompt receipt verified.")
        return
    rows, bases, pools, records, si, sv, qv = load()
    bykey = {r["key"]: r for r in rows}
    c.require(
        list(c.rows(BASE / "roster.jsonl"))
        == [dict(r, population=r["phase"]) for r in rows],
        "A16 gold/roster changed",
    )
    science = c.read(BASE / "scientific_plan_v1.json")
    conditions = science["conditions"]
    pools_by = {p["condition"]: p for p in pools}
    independent = {
        p["condition"]: LocalBucket(
            [records[i] for i in p["ids"]], sv[[si[i] for i in p["ids"]]]
        )
        for p in pools
    }
    choices = {(r["key"], r["condition"]): r for r in c.rows(BASE / "choices.jsonl")}
    seen = set()
    tok = tokens.tokenizer()
    delivery = Counter()
    for p in c.rows(BASE / "prompts.jsonl"):
        pair = (p["key"], p["condition"])
        c.require(pair not in seen, "Duplicate A16 prompt")
        seen.add(pair)
        r = bykey[p["key"]]
        base = bases[p["key"]]
        context = deepcopy(base["context"])
        c.require(
            p["text"] == r["text"]
            and p["population"] == r["phase"]
            and (p["item_id"] == r["id"])
            and (
                p["scientific_plan_sha256"] == c.sha(BASE / "scientific_plan_v1.json")
            ),
            "A16 attribution changed",
        )
        if p["condition"] == "reference":
            c.require(
                p["source_id"] is None and p["messages"] == base["messages"],
                "A16 reference changed",
            )
        else:
            chosen, similarity, trace = independent[p["condition"]].retrieve(
                r["text"], qv[r["scope_node_index"]]
            )
            sid = chosen["provenance"]["source_id"]
            stored = choices[pair]
            c.require(
                sid == stored["source_id"] == p["source_id"]
                and sid in pools_by[p["condition"]]["ids"],
                "Independent correction selection differs",
            )
            c.require(
                abs(similarity - stored["cosine"]) < 1e-12
                and stored["source_gate"] == chosen["meta"]["gate"],
                "Correction label/cosine differs",
            )
            for channel in ("dense", "bm25", "rrf"):
                c.require(
                    [z["id"] for z in trace[channel]]
                    == [z["id"] for z in stored["retrieval"][channel]],
                    "Independent rank order differs",
                )
                c.require(
                    all(
                        (
                            abs(a["score"] - b["score"]) < 1e-10
                            for a, b in zip(
                                trace[channel], stored["retrieval"][channel]
                            )
                        )
                    ),
                    "Independent ranking arithmetic differs",
                )
            c.require(
                all((v is None for k, v in chosen["meta"].items() if k != "gate")),
                "Unprovided fine annotation",
            )
            hits = context["example_groups"][0][1]
            c.require(
                len(hits) == 5
                and context["example_groups"][0][0] == "Labelled examples",
                "Wrong base example group",
            )
            hits.pop()
            hits.append(
                {
                    "id": chosen["id"],
                    "text": chosen["text"],
                    "meta": deepcopy(chosen["meta"]),
                    "via": "rrf",
                }
            )
            delivery[p["condition"] + "/" + str(chosen["meta"]["gate"])] += 1
        messages = renderer.render_messages(r["text"], "en", context, "corrected")
        c.require(
            p["context"] == context
            and p["messages"] == messages
            and (messages[0] == base["messages"][0]),
            "Independent context/message differs",
        )
        c.require(
            p["context_sha256"] == c.sig(context)
            and p["messages_sha256"] == c.sig(messages),
            "Prompt checksum differs",
        )
        c.require(
            p["reference_input_tokens"] == tokens.token_count(tok, messages),
            "Token accounting differs",
        )
        if len(seen) % 5000 == 0:
            print(f"A16 independent check {len(seen)}/{len(rows) * 4}", flush=True)
    c.require(
        seen == {(r["key"], cond) for r in rows for cond in conditions}
        and len(choices) == len(rows) * 3,
        "A16 comparison matrix incomplete",
    )
    c.verify(plan["inputs"])
    c.verify(c.read(BASE / "prepared.json")["outputs"])
    c.active_guard()
    checkpath = BASE / "independent_verification.json"
    c.save(
        checkpath,
        {
            "status": "every_A16_prompt_and_correction_independently_verified",
            "prompts": len(seen),
            "retrievals": len(choices),
            "delivery": dict(delivery),
            "prepared_sha256": c.sha(BASE / "prepared.json"),
            "source_sha256": c.sha(__file__),
            "new_model_calls": 0,
            "CUDA_used": False,
        },
    )
    c.save(
        BASE / "complete.json",
        {
            "status": "offline_preparation_independently_verified",
            "prepared_sha256": c.sha(BASE / "prepared.json"),
            "verification": c.rel(checkpath),
            "verification_sha256": c.sha(checkpath),
            "new_model_calls": 0,
        },
    )
    print("A16 complete independent prompt/retrieval check passed.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "check", "all"])
    args = ap.parse_args()
    with c.offline(), threadpool_limits(limits=1):
        if args.mode in ("prepare", "all"):
            prepare()
        if args.mode in ("check", "all"):
            check()
