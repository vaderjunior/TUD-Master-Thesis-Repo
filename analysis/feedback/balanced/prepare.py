import os

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "2"
os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
)
from collections import Counter, defaultdict
from copy import deepcopy
import socket
from unittest.mock import patch
import numpy as np
from analysis.feedback.balanced import common as c
from analysis.feedback.source.retrieval import FeedbackBucket
from analysis.common import prompt as renderer


def append_binary(context, record):

    c.require(
        len(context["definitions"]) == len(context["guidelines"]) == 2,
        "Base knowledge budget differs",
    )
    c.require(
        sum((len(v) for _, v in context["example_groups"])) == 5,
        "Base example budget differs",
    )
    c.require(
        c.GROUP not in {name for name, _ in context["example_groups"]},
        "Cannot add another correction slot",
    )
    meta = record["meta"]
    c.require(
        record["kind"] == "feedback"
        and record["lang"] == "en"
        and (type(meta["gate"]) is bool),
        "Invalid correction kind/language/gate",
    )
    c.require(
        set(meta) == {"gate", "hate_types", "target_groups", "severity", "legal"},
        "Unexpected source annotation",
    )
    c.require(
        all((meta[k] is None for k in meta if k != "gate")),
        "Only binary feedback is allowed",
    )
    c.require(
        record["id"]
        not in {h["id"] for _, rs in context["example_groups"] for h in rs},
        "Correction already in base context",
    )
    out = deepcopy(context)
    out["example_groups"].append(
        [
            c.GROUP,
            [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "meta": deepcopy(meta),
                    "via": "rrf",
                }
            ],
        ]
    )
    return out


def original_inputs():
    for base in [c.SELECTION, c.CANDIDATES]:
        done = c.read(base / "complete.json")
        if "inputs" in done:
            c.verify(done["inputs"])
        if "outputs" in done:
            c.verify(done["outputs"])
    prepared = c.read(c.CANDIDATES / "prepared.json")
    c.verify(prepared["outputs"])
    paths = [
        c.SELECTION / name
        for name in [
            "complete.json",
            "summary.json",
            "actual_pools.json",
            "gate_only_records.json",
            "candidate_statuses.json",
        ]
    ]
    paths += [
        c.CANDIDATES / name
        for name in [
            "complete.json",
            "prepared.json",
            "scientific_plan_v1.json",
            "STUDY_PROTOCOL.md",
            "roster.jsonl",
            "query_vectors.npy",
            "query_vector_sources.jsonl",
            "scope_complete.json",
            "scope_details.json",
            "cache_bindings.json",
        ]
    ]
    paths += [
        c.TAXONOMY / name for name in ["roster.jsonl", "prompts.jsonl", "complete.json"]
    ]
    paths += [
        c.CORE / "dev_inputs.jsonl",
        c.CORE / "retrieval_v1/query_vectors.npy",
        c.CORE / "retrieval_v1/query_vector_sources.jsonl",
    ]
    paths += [
        c.ROOT / name
        for name in [
            "config/taxonomy.yaml",
            "prompts/classify_v1.txt",
            "src/hsrag/prompt.py",
            "analysis/common/prompt.py",
            "analysis/feedback/balanced/common.py",
            "analysis/feedback/balanced/prepare.py",
            "analysis/feedback/source/retrieval.py",
            "analysis/feedback/source/design.py",
            "analysis/feedback/explanations/calibration.py",
        ]
    ]
    return c.bindings(paths)


def main():
    c.require(not (c.BASE / "prepared.json").exists(), "Preserve completed preparation")
    bindings = original_inputs()
    pools = c.read(c.SELECTION / "actual_pools.json")
    records = c.read(c.SELECTION / "gate_only_records.json")
    c.require(
        pools["budget"] == 40
        and len(pools["pools"]) == 18
        and (pools["seeds"] == c.SEEDS),
        "Approved pool matrix changed",
    )
    conditions = ["reference"] + [p["condition"] for p in pools["pools"]]
    c.require(
        len(set(conditions)) == 19 and len(records) == 413,
        "Condition/source matrix differs",
    )
    protocol = (c.CANDIDATES / "STUDY_PROTOCOL.md").read_text(encoding="utf-8")
    protocol += "\n## 6 October update-preparation checkpoint\n\nThe audited selection passed the original 40-example rule. All 18 actual pools are frozen in the 6 October selection results. This stage uses those exact pools and the existing 675-text evaluation roster; it neither resamples sources nor reuses old prediction responses. Four automatic-audited execution batches keep every comparison for an input together. All model responses remain pending.\n"
    c.write(c.BASE / "STUDY_PROTOCOL.md", protocol)
    bindings[c.rel(c.BASE / "STUDY_PROTOCOL.md")] = c.sha(c.BASE / "STUDY_PROTOCOL.md")
    c.write(
        c.BASE / "preparation_plan.json",
        {
            "created_utc": c.utc(),
            "inputs": bindings,
            "original_bindings": c.old.preserve_originals(),
            "new_model_calls": 0,
            "retrieval": "Original FeedbackBucket; exact float64 cosine top10 and positive BM25 top10; RRF60; ascending ID ties; no query labels",
            "base_context": "Existing taxonomy rag_full messages; no repeated ANN retrieval or changed ordinary knowledge",
        },
    )
    science = {
        "created_utc": c.utc(),
        "conditions": conditions,
        "pool_seeds": c.SEEDS,
        "pool_budget": 40,
        "actual_pools_sha256": c.sha(c.SELECTION / "actual_pools.json"),
        "gate_only_records_sha256": c.sha(c.SELECTION / "gate_only_records.json"),
        "preparation_plan_sha256": c.sha(c.BASE / "preparation_plan.json"),
        "protocol_sha256": c.sha(c.BASE / "STUDY_PROTOCOL.md"),
        "original_source_protocol_sha256": c.sha(c.CANDIDATES / "STUDY_PROTOCOL.md"),
        "decoder_repeats": 3,
        "votes_per_prediction": 3,
        "population_counts": {"hateful_types": 467, "benign": 208},
        "primary": [
            {
                "name": "benign_false_positive_rate",
                "gold_hate": False,
                "left": "error_balanced",
                "right": "error_positive",
                "bootstrap_seed": 202610044,
            },
            {
                "name": "hateful_false_negative_rate",
                "gold_hate": True,
                "left": "error_balanced",
                "right": "error_positive",
                "bootstrap_seed": 202610045,
            },
        ],
        "primary_interval": 0.975,
        "bootstrap_draws": 10000,
        "bootstrap_generator": "PCG64",
        "bootstrap_unit": "Original overlap text cluster; all pools and decoder repeats retained together",
        "primary_mask": "Within each population, all two primary arms x three pools x three repeats nonnull (18 outputs)",
        "averaging": "Rate per decoder repeat; equal average of three repeats within each pool, then three pools. Shared reference has three real repeats only.",
        "supporting": [
            "each arm versus reference",
            "matched already-correct and random controls",
            "balanced accuracy",
            "binary macro F1 at artificial prevalence",
            "annotated fine-head regressions",
            "benign source strata",
            "delivered polarity and source/query similarity",
            "actual pool overlap",
            "annotation and inference cost",
            "every run",
        ],
        "supporting_interval": 0.95,
        "sensitivities": [
            "all-three-valid-votes mask",
            "one stable representative per original overlap cluster",
            "null-as-error and opposite bound",
        ],
        "scope": "Previously exposed DEV; new matched gate-only information; no untouched confirmation or comparison isolating polarity against historical type-feedback",
        "failure_policy": "Retain nulls and failed attempts; no replacement generation or outcome-based stopping",
        "decision_rule": "A false-alarm decrease with more missed hate is a trade-off. Nonsignificant recall loss is not non-inferiority.",
        "original_primary_family": "Closed and unchanged; this is a separate prospective two-outcome family",
    }
    c.write(c.BASE / "scientific_plan_v1.json", science)
    roster = c.rows(c.TAXONOMY / "roster.jsonl")
    c.require(
        len(roster) == 675
        and Counter((r["population"] for r in roster))
        == {"hateful_types": 467, "benign": 208},
        "Original evaluation roster changed",
    )
    c.jsonlines(c.BASE / "roster.jsonl", roster)
    bases = {
        p["key"]: p
        for p in c.rows(c.TAXONOMY / "prompts.jsonl")
        if p["condition"] == "rag_full"
    }
    core = c.rows(c.CORE / "dev_inputs.jsonl")
    qmap = {r["key"]: i for i, r in enumerate(core)}
    qv = np.load(
        c.CORE / "retrieval_v1/query_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    sources = c.rows(c.CANDIDATES / "roster.jsonl")
    smap = {r["id"]: i for i, r in enumerate(sources)}
    sv = np.load(c.CANDIDATES / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    c.require(
        qv.shape == (1295, 1024) and sv.shape == (1024, 1024),
        "Cached vector matrix changed",
    )
    c.require(
        not {r["id"] for r in roster}.intersection(records),
        "Source/evaluation IDs overlap",
    )
    buckets = {
        p["condition"]: FeedbackBucket(
            [records[i] for i in p["ids"]], sv[[smap[i] for i in p["ids"]]]
        )
        for p in pools["pools"]
    }
    tok = c.old.tokenizer()
    prompts = []
    choices = []
    provenance = []
    token_counts = []
    delivery = defaultdict(Counter)
    science_sha = c.sha(c.BASE / "scientific_plan_v1.json")
    for number, r in enumerate(roster, 1):
        base = bases[r["key"]]
        core_row = core[qmap[r["key"]]]
        c.require(
            base["text"] == r["text"] == core_row["text"]
            and base["item_id"] == r["id"] == core_row["item_id"],
            "Query/vector/base identity drift",
        )
        c.require(
            renderer.render_messages(r["text"], "en", base["context"], "corrected")
            == base["messages"],
            "Original base rendering drift",
        )
        provenance.append(
            {
                "key": r["key"],
                "query_vector_file": c.rel(c.CORE / "retrieval_v1/query_vectors.npy"),
                "query_vector_row": qmap[r["key"]],
                "text_sha256": __import__("hashlib")
                .sha256(r["text"].encode())
                .hexdigest(),
            }
        )
        for condition in conditions:
            if condition == "reference":
                context = deepcopy(base["context"])
                messages = deepcopy(base["messages"])
                selected = None
            else:
                chosen, trace = buckets[condition].retrieve(
                    r["text"], qv[qmap[r["key"]]]
                )
                sid = chosen["provenance"]["source_id"]
                c.require(
                    sid != r["id"] and chosen["text"] != r["text"],
                    "Query leaked into correction",
                )
                context = append_binary(base["context"], chosen)
                messages = renderer.render_messages(
                    r["text"], "en", context, "corrected"
                )
                selected = sid
                vec = np.asarray(sv[smap[sid]], dtype=np.float64)
                query = np.asarray(qv[qmap[r["key"]]], dtype=np.float64)
                similarity = float(
                    np.dot(vec, query) / (np.linalg.norm(vec) * np.linalg.norm(query))
                )
                choices.append(
                    {
                        "key": r["key"],
                        "condition": condition,
                        "population": r["population"],
                        "source_id": sid,
                        "source_gate": chosen["meta"]["gate"],
                        "source_vector_row": smap[sid],
                        "cosine": similarity,
                        "retrieval": trace,
                    }
                )
                delivery[condition][
                    r["population"] + "/" + str(chosen["meta"]["gate"])
                ] += 1
            c.require(
                messages[0] == base["messages"][0], "System/schema changed across arms"
            )
            tokens = c.old.token_count(tok, messages)
            token_counts.append(tokens)
            prompts.append(
                {
                    "key": r["key"],
                    "item_id": r["id"],
                    "condition": condition,
                    "population": r["population"],
                    "lang": "en",
                    "text": r["text"],
                    "context": context,
                    "messages": messages,
                    "context_sha256": c.sig(context),
                    "messages_sha256": c.sig(messages),
                    "reference_input_tokens": tokens,
                    "source_id": selected,
                    "scientific_plan_sha256": science_sha,
                }
            )
        if number % 100 == 0:
            print(
                f"Prepared {number}/675 evaluation inputs, cached CPU retrieval only.",
                flush=True,
            )
    c.require(
        len(prompts) == 12825 and len(choices) == 12150, "Missing prepared comparisons"
    )
    c.jsonlines(c.BASE / "prompts.jsonl", prompts)
    c.jsonlines(c.BASE / "choices.jsonl", choices)
    c.jsonlines(c.BASE / "query_vector_provenance.jsonl", provenance)
    summary = {
        "status": "prepared_requires_independent_verification",
        "texts": 675,
        "conditions": 19,
        "prompts": 12825,
        "predictions": 38475,
        "ordinary_calls": 115425,
        "unchanged_reference_messages": 675,
        "additional_binary_slots": 12150,
        "source_count": len(records),
        "pool_count": 18,
        "pool_budget": 40,
        "input_tokens_min": min(token_counts),
        "input_tokens_max": max(token_counts),
        "delivery_by_condition": {k: dict(v) for k, v in delivery.items()},
        "new_model_calls": 0,
        "new_embeddings": 0,
        "cuda_jobs": 0,
    }
    c.write(c.BASE / "summary.json", summary)
    c.verify(bindings)
    c.old.preserve_originals()
    outputs = c.bindings((p for p in c.BASE.iterdir() if p.is_file()))
    c.write(
        c.BASE / "prepared.json",
        {
            "status": "prepared_requires_independent_check",
            "outputs": outputs,
            "created_utc": c.utc(),
        },
    )
    print(
        {
            k: summary[k]
            for k in [
                "prompts",
                "predictions",
                "ordinary_calls",
                "input_tokens_max",
                "new_model_calls",
            ]
        },
        flush=True,
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", c.forbid), patch.object(
        socket, "create_connection", c.forbid
    ):
        main()
