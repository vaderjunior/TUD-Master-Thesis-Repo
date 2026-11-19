from src.hsrag.paths import PROJECT_ROOT
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os

os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ANONYMIZED_TELEMETRY="False"
)
ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = CORE / "retrieval_v1"
OUT = ROOT / "Thesis_Log/astra_result/evidence/core_retrieval_check_20260915_v1.json"
TABLE = ROOT / "Thesis_Log/astra_result/astra_core_retrieval_20260915_v1.md"
HASHES = {}


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    HASHES[rel(p)] = h.hexdigest()
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def rows(p):
    return [
        json.loads(x)
        for x in Path(p).read_text(encoding="utf8").splitlines()
        if x.strip()
    ]


def signature(x):
    return hashlib.sha256(
        (
            json.dumps(x, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        ).encode()
    ).hexdigest()


def check_vectors(filename, sourcefile, identities, idkey):
    import numpy as np

    matrix = np.load(BASE / filename, allow_pickle=False, mmap_mode="r")
    source = rows(BASE / sourcefile)
    require(
        matrix.dtype == np.float32 and matrix.shape == (len(identities), 1024),
        "Wrong assembled matrix",
    )
    require(
        [x[idkey] for x in source] == [x[idkey] for x in identities],
        "Vector mapping order differs",
    )
    byfile = defaultdict(list)
    for i, (origin, item) in enumerate(zip(source, identities)):
        require(
            origin["text_sha256"] == hashlib.sha256(item["text"].encode()).hexdigest(),
            "Vector text identity differs",
        )
        byfile[origin["file"]].append((i, origin["row"]))
    for name, pairs in byfile.items():
        if name.endswith(".npz"):
            with np.load(ROOT / name, allow_pickle=False) as z:
                vv = z["vectors"]
                meta = json.loads(str(z["metadata"].item()))
            require(
                hashlib.sha256(vv.tobytes()).hexdigest() == meta["vectors_sha256"],
                "Raw source vector checksum differs",
            )
        else:
            vv = np.asarray(read(ROOT / name)["vectors"], dtype=np.float32)
        a, b = zip(*pairs)
        require(
            np.array_equal(matrix[list(a)], vv[list(b)]),
            "Copied vectors differ from original cache",
        )
    return {"rows": len(identities), "source_files": len(byfile), "bitwise_equal": True}


def main():
    require(not OUT.exists() and (not TABLE.exists()), "Preserve completed review")
    os.chdir(ROOT)
    from analysis.common import prompt as prompt

    plan = read(BASE / "plan.json")
    done = read(BASE / "complete.json")
    require(
        done["status"] == "prepared_no_classifier_calls"
        and done["plan_sha256"] == sha(BASE / "plan.json"),
        "Setup incomplete/mismatched",
    )
    for mapping in (plan["input_hashes"], plan["generated_hashes"], done["hashes"]):
        for name, digest in mapping.items():
            require(sha(ROOT / name) == digest, f"Changed source/output: {name}")
    for f in [Path(__file__), BASE / "complete.json"]:
        sha(f)
    full = rows(CORE / "kb/full/records.jsonl")
    dev = rows(CORE / "dev_inputs.jsonl")
    require(
        len({r["text"] for r in dev}) == len(dev),
        "Duplicate query text would alias cached-provider entries",
    )
    vector_checks = {
        "records": check_vectors(
            "full_vectors.npy", "record_vector_sources.jsonl", full, "id"
        ),
        "queries": check_vectors(
            "query_vectors.npy", "query_vector_sources.jsonl", dev, "key"
        ),
    }
    summaries = {}
    saved = {}
    for pool in ("small", "full"):
        known = {r["id"]: r for r in rows(CORE / f"kb/{pool}/records.jsonl")}
        items = rows(BASE / pool / "prompts.jsonl")
        saved[pool] = {x["key"]: x for x in items}
        require(len(saved[pool]) == len(items) == 1295, "Wrong prompt counts")
        require(
            [x["key"] for x in items] == [x["key"] for x in dev], "Wrong prompt order"
        )
        cold = read(BASE / pool / "cold_check.json")
        require(
            cold["all_prompts_retrieved_again"]
            and cold["prompt_sha256"] == sha(BASE / pool / "prompts.jsonl"),
            "Cold replay incomplete",
        )
        stats = {}
        used = defaultdict(set)
        lengths = defaultdict(list)
        for x, r in zip(items, dev):
            require(all((x[k] == r[k] for k in r)), "Input identity/text/scope differs")
            ctx = x["context"]
            examples = [h for _, rr in ctx["example_groups"] for h in rr]
            require(
                len(examples) == len({h["id"] for h in examples}) == 5,
                "Wrong examples count",
            )
            require(
                len(ctx["definitions"]) == len(ctx["guidelines"]) == 2,
                "Wrong abstract budget",
            )
            for bucket, rr in [
                ("definition", ctx["definitions"]),
                ("guideline", ctx["guidelines"]),
                ("example", examples),
            ]:
                for h in rr:
                    k = known[h["id"]]
                    expected = {
                        **{
                            name: k[name]
                            for name in ("kind", "lang", "source", "dimension", "label")
                        },
                        **k["meta"],
                    }
                    require(
                        k["text"] == h["text"] and k["kind"] == bucket,
                        "Delivered content differs",
                    )
                    require(
                        {
                            name: value
                            for name, value in h["meta"].items()
                            if name != "_ranks"
                        }
                        == expected,
                        "Delivered annotation differs",
                    )
                    if bucket == "example":
                        require(k["lang"] == r["lang"], "Wrong example language")
            rendered = prompt.render_messages(r["text"], r["lang"], ctx, "corrected")
            require(
                rendered == x["messages"]
                and signature(rendered) == x["messages_sha256"],
                "Messages do not reproduce",
            )
            require(
                signature(ctx) == x["context_sha256"]
                and prompt.hit_ids(ctx) == x["retrieved"],
                "Context stamps differ",
            )
            for scope in r["scopes"]:
                name = scope["subset"]
                c = stats.setdefault(name, Counter())
                c["inputs"] += 1
                lengths[name].append(sum((len(m["text"]) for m in rendered)))
                for h in examples:
                    m = h["meta"]
                    used[name].add(h["id"])
                    c["example_slots"] += 1
                    c[f"gate_{m.get('gate')}"] += 1
                    for f in ("target_groups", "hate_types", "severity", "legal"):
                        c[f + "_annotated"] += m.get(f) is not None
        summaries[pool] = {
            k: {
                **dict(v),
                "unique_examples": len(used[k]),
                "prompt_chars_min": min(lengths[k]),
                "prompt_chars_max": max(lengths[k]),
                "prompt_chars_mean": sum(lengths[k]) / len(lengths[k]),
            }
            for k, v in stats.items()
        }
    comparisons = Counter()
    changed_abstract = []
    for r in dev:
        a, b = (saved[p][r["key"]] for p in ("small", "full"))
        comparisons["inputs"] += 1
        for part in ("definitions", "guidelines"):
            same = a["context"][part] == b["context"][part]
            comparisons[part + "_same_exact_context"] += same
            if not same:
                changed_abstract.append({"key": r["key"], "kind": part})
        ids = lambda x: [
            h["id"] for _, rr in x["context"]["example_groups"] for h in rr
        ]
        comparisons["examples_same_ordered_ids"] += ids(a) == ids(b)
        comparisons["examples_same_id_set"] += set(ids(a)) == set(ids(b))
        comparisons["system_message_equal"] += a["messages"][0] == b["messages"][0]
    history = {}
    historical_changed = []
    prior = ROOT / "experiments/astra/german_kb_comparison_20260913_v1"
    for pool, oldname in [("small", "c_canonical"), ("full", "d_full")]:
        p = prior / f"prompts_{oldname}.jsonl"
        require(
            sha(p) == read(prior / "setup_complete.json")["hashes"][rel(p)],
            "Historical German prompt hash differs",
        )
        old = {x["item_id"]: x for x in rows(p)}
        c = Counter()
        for r in dev:
            if r["lang"] != "de":
                continue
            a = saved[pool][r["key"]]
            b = old[r["item_id"]]
            require(a["text"] == b["text"], "Historical German input text differs")
            c["inputs"] += 1
            c["same_messages"] += a["messages"] == b["messages"]
            c["same_context"] += a["context"] == b["context"]
            ids = lambda x: [
                h["id"] for _, rr in x["context"]["example_groups"] for h in rr
            ]
            c["same_example_set"] += set(ids(a)) == set(ids(b))
            c["same_example_order"] += ids(a) == ids(b)
            if a["messages"] != b["messages"]:
                historical_changed.append(
                    {
                        "pool": pool,
                        "key": r["key"],
                        "same_example_set": set(ids(a)) == set(ids(b)),
                        "same_example_order": ids(a) == ids(b),
                    }
                )
        oldpool = "canonical_small" if pool == "small" else "canonical_full"
        oldrecords = [
            x
            for x in rows(prior / f"kb/{oldpool}/records.jsonl")
            if x["kind"] == "example" and x["lang"] == "de"
        ]
        newrecords = [
            x
            for x in rows(CORE / f"kb/{pool}/records.jsonl")
            if x["kind"] == "example" and x["lang"] == "de"
        ]
        require(
            oldrecords == newrecords, "Historical German pool content/order changed"
        )
        history[pool] = {**dict(c), "same_german_record_order_and_content": True}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_retrieval_preparation_audit",
        "plan_sha256": sha(BASE / "plan.json"),
        "setup_sha256": sha(BASE / "complete.json"),
        "vector_checks": vector_checks,
        "index_storage_vector_max_abs_difference": {
            pool: read(BASE / pool / "complete.json")["index_vector_max_abs_difference"]
            for pool in ("small", "full")
        },
        "prompt_summaries": summaries,
        "pool_comparisons": dict(comparisons),
        "changed_abstract_contexts": changed_abstract,
        "historical_german_prompt_comparison": history,
        "historical_german_changed_prompts": historical_changed,
        "scope": "Reconstructed every vector mapping and saved RAG prompt; both indexes replayed every query after a fresh process. No API/model forward pass, classifier predictions, outcome scoring, Priority 3 or LaTeX writing.",
        "caveats": "Cached query vectors use FP32 batch-4 outputs, not new individual forwards. Approximate Chroma indexes and combined pool can change rankings; preserve actual prompts. Identical historical German prompts alone do not authorize silent prediction reuse or establish contemporaneous service behaviour. API logger/execution protocol remain separate.",
        "verified_hashes": dict(sorted(HASHES.items())),
    }
    with OUT.open("x", encoding="utf8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    doc = [
        "# Bilingual core retrieval preparation — 2026-09-15",
        "",
        "**Preparation and audit complete; no new classifier results.**",
        "",
        "RAG needs a searchable KB and a saved prompt for each input. This stage prepares those inputs for the next SQ2/baseline comparison, with relevance to the SQ1 KB-size design. It does not fine-tune an encoder or ask Qwen to classify.",
        "",
        "Two isolated indexes contain 71,131 full records and 340 small records, including the same 40 definitions/guidelines. The assembled files contain 71,131 record vectors and 1,295 query vectors copied from checked BGE-M3 caches bit for bit. Chroma storage allows only the separately checked tiny cosine-normalization differences (recorded in the evidence). Seven focused tests passed. All 2,590 RAG prompts use five language-matched examples, two definitions and two guidelines; all delivered texts/annotations/messages reproduce. Every query was repeated after closing and reopening its index in a fresh process.",
        "",
        "No fresh embedding run, GPU training, classifier/API call or outcome scoring was needed. Original indexes and results remain intact.",
        "",
        "## Retrieved annotation coverage",
        "",
        "Counts are retrieved slots with observed labels, not unique training examples or classifier accuracy. Missing annotation is not the same as an annotated empty list.",
        "",
        "| Pool | DEV roster | Inputs | Unique examples | Gate true / false / missing slots | Target / type / severity / legal annotated slots |",
        "|---|---|---:|---:|---|---|",
    ]
    for pool, stats in summaries.items():
        for name, c in stats.items():
            doc.append(
                f"| {pool} | {name} | {c['inputs']} | {c['unique_examples']} | {c.get('gate_True', 0)} / {c.get('gate_False', 0)} / {c.get('gate_None', 0)} | {c.get('target_groups_annotated', 0)} / {c.get('hate_types_annotated', 0)} / {c.get('severity_annotated', 0)} / {c.get('legal_annotated', 0)} |"
            )
    doc += [
        "",
        "## Comparability checks",
        "",
        f"Across 1,295 paired inputs: `{dict(comparisons)}`. Changed abstract contexts: {len(changed_abstract)}. The evidence lists any changed keys.",
        "",
        f"Historical German message equivalence: `{history}`. These counts are a reuse diagnostic only. The frozen core plan still calls for new contemporaneous comparisons unless a later explicit, documented reuse amendment is made.",
        "",
        "The original hybrid/MMR ranking code and corrected renderer were preserved. A cached vector provider replaces neural query encoding; exact text matching is required and unknown inputs fail rather than invoking a model. Both pools use the same query vectors. Cached batched query arithmetic can differ slightly from old individual inference; this is recorded as an implementation/provenance qualification.",
        "",
        "## Next bounded stage",
        "",
        "Prepare isolated concurrency-safe API logging, explicit retry/repair caps, exact failure/scoring rules and staged resumable classifier execution. The 96,822-response programme remains a programme budget, not a ready batch. No user terminal action is required for this completed retrieval stage. Priority 3, adaptation and LaTeX stay parked.",
        "",
        f"Evidence: [review JSON](evidence/{OUT.name}); plan SHA-256 `{report['plan_sha256']}`; evidence SHA-256 `{sha(OUT)}`.",
    ]
    with TABLE.open("x", encoding="utf8", newline="\n") as f:
        f.write("\n".join(doc) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "evidence_sha256": sha(OUT),
                "vectors": vector_checks,
                "comparisons": dict(comparisons),
                "historical_german": history,
                "verified_files": len(report["verified_hashes"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
