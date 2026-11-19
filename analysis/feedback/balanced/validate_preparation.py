import os

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "2"
from collections import Counter
import hashlib
import re
import socket
import unicodedata
from unittest.mock import patch
import numpy as np
from analysis.feedback.balanced import common as c
from analysis.feedback.explanations.calibration import LocalBucket


def main():
    c.require(
        not (c.BASE / "complete.json").exists(), "Preserve completed verification"
    )
    prepared = c.read(c.BASE / "prepared.json")
    c.verify(prepared["outputs"])
    plan = c.read(c.BASE / "preparation_plan.json")
    c.verify(plan["inputs"])
    c.verify(plan["original_bindings"])
    pools = c.read(c.SELECTION / "actual_pools.json")
    records = c.read(c.SELECTION / "gate_only_records.json")
    statuses = c.read(c.SELECTION / "candidate_statuses.json")
    source_rows = c.rows(c.CANDIDATES / "roster.jsonl")
    source_index = {r["id"]: n for n, r in enumerate(source_rows)}
    source_vec = np.load(c.CANDIDATES / "query_vectors.npy", allow_pickle=False)
    core_rows = c.rows(c.CORE / "dev_inputs.jsonl")
    core_index = {r["key"]: n for n, r in enumerate(core_rows)}
    query_vec = np.load(c.CORE / "retrieval_v1/query_vectors.npy", allow_pickle=False)
    roster = {r["key"]: r for r in c.rows(c.BASE / "roster.jsonl")}
    original_roster = {r["key"]: r for r in c.rows(c.TAXONOMY / "roster.jsonl")}
    c.require(
        roster == original_roster and len(roster) == 675,
        "Evaluation roster/gold/cluster differs",
    )
    base = {
        p["key"]: p
        for p in c.rows(c.TAXONOMY / "prompts.jsonl")
        if p["condition"] == "rag_full"
    }
    prompts = c.rows(c.BASE / "prompts.jsonl")
    choices = {(p["key"], p["condition"]): p for p in c.rows(c.BASE / "choices.jsonl")}
    science = c.read(c.BASE / "scientific_plan_v1.json")
    expected = ["reference"] + [p["condition"] for p in pools["pools"]]
    c.require(
        science["conditions"] == expected
        and science["primary_interval"] == 0.975
        and science["bootstrap_draws"] == 10000,
        "Analysis family changed",
    )
    c.require(
        [
            (p["gold_hate"], p["left"], p["right"], p["bootstrap_seed"])
            for p in science["primary"]
        ]
        == [
            (False, "error_balanced", "error_positive", 202610044),
            (True, "error_balanced", "error_positive", 202610045),
        ],
        "Frozen primary changed",
    )
    by_condition = {p["condition"]: p for p in pools["pools"]}
    reconstructed = {}
    for p in pools["pools"]:
        seed = p["seed"]
        selection = p["selection"]
        order = {}
        for gate in [True, False]:
            eligible = [
                i
                for i, v in statuses.items()
                if v["hate"] == gate
                and v["status"] != "invalid"
                and (selection == "random" or v["status"] == selection)
            ]
            order[gate] = sorted(
                eligible,
                key=lambda i: hashlib.sha256(
                    (f"{seed}/{selection}/{gate}|" + i).encode()
                ).hexdigest(),
            )
        want = (
            order[True][:40]
            if p["polarity"] == "positive"
            else order[True][:20] + order[False][:20]
        )
        c.require(
            p["ids"] == want and len(set(want)) == 40, "Exact frozen pool mismatch"
        )
        reconstructed[p["condition"]] = LocalBucket(
            [records[i] for i in want], source_vec[[source_index[i] for i in want]]
        )
    for sid, r in records.items():
        original = source_rows[source_index[sid]]
        c.require(
            r["id"] == "sq3benign-" + sid and r["text"] == original["text"],
            "Added record identity changed",
        )
        c.require(
            r["meta"]
            == {
                "gate": original["gold"]["hate"],
                "hate_types": None,
                "target_groups": None,
                "severity": None,
                "legal": None,
            },
            "Undeclared annotation in correction",
        )
    norm = lambda t: " ".join(
        unicodedata.normalize("NFKC", t).casefold().split()
    ).lstrip(": ")
    protected = {norm(r["text"]) for r in core_rows if r["lang"] == "en"}
    c.require(
        all(norm(r["text"]) not in protected for r in records.values()),
        "Normalized source/core overlap",
    )
    chosen_vectors = np.array(
        source_vec[[source_index[i] for i in sorted(records)]], dtype=np.float64
    )
    evaluated_vectors = np.array(
        query_vec[[core_index[i] for i in sorted(roster)]], dtype=np.float64
    )
    chosen_vectors /= np.linalg.norm(chosen_vectors, axis=1)[:, None]
    evaluated_vectors /= np.linalg.norm(evaluated_vectors, axis=1)[:, None]
    max_similarity = float(
        np.max(np.einsum("ik,jk->ij", chosen_vectors, evaluated_vectors, optimize=True))
    )
    c.require(max_similarity < 0.9495, "Selected source too close to evaluation input")

    verified_sources = c.read(c.CANDIDATES / "complete.json")
    c.require(
        verified_sources["verification_sha256"]
        == c.sha(c.ROOT / verified_sources["verification"]),
        "Source exclusion verification changed",
    )
    tpl = (c.ROOT / "prompts/classify_v1.txt").read_text(encoding="utf-8")
    sections = re.split(r"^=== (\w+) ===$", tpl, flags=re.M)
    cautions = {
        sections[n]: sections[n + 1].strip() for n in range(1, len(sections), 2)
    }
    tok = c.old.tokenizer()
    seen = set()
    all_reference = 0
    delivery = Counter()
    trace_checks = 0
    for number, prompt in enumerate(prompts, 1):
        key, condition = prompt["key"], prompt["condition"]
        pair = key, condition
        c.require(
            pair not in seen and key in roster and condition in expected,
            "Duplicate/unexpected prompt",
        )
        seen.add(pair)
        r = roster[key]
        old = base[key]
        c.require(
            core_rows[core_index[key]]["text"]
            == r["text"]
            == prompt["text"]
            == old["text"],
            "Query cache/text mismatch",
        )
        c.require(
            prompt["item_id"] == r["id"] and prompt["population"] == r["population"],
            "Payload roster identity changed",
        )
        c.require(
            prompt["context_sha256"] == c.sig(prompt["context"])
            and prompt["messages_sha256"] == c.sig(prompt["messages"]),
            "Payload hash mismatch",
        )
        c.require(
            prompt["scientific_plan_sha256"]
            == c.sha(c.BASE / "scientific_plan_v1.json"),
            "Scientific plan attribution differs",
        )
        c.require(
            prompt["reference_input_tokens"]
            == c.old.token_count(tok, prompt["messages"]),
            "Token count differs",
        )
        if condition == "reference":
            c.require(
                prompt["context"] == old["context"]
                and prompt["messages"] == old["messages"]
                and prompt["source_id"] is None,
                "Fresh reference is not original prompt",
            )
            all_reference += 1
            continue
        selected, similarity, trace = reconstructed[condition].retrieve(
            r["text"], query_vec[core_index[key]]
        )
        stored = choices[pair]
        sid = selected["provenance"]["source_id"]
        c.require(
            prompt["source_id"] == stored["source_id"] == sid
            and sid in by_condition[condition]["ids"],
            "Independent retrieval chose another source",
        )
        c.require(
            abs(similarity - stored["cosine"]) < 1e-12
            and stored["source_gate"] == selected["meta"]["gate"],
            "Delivered label/similarity changed",
        )
        c.require(
            stored["source_vector_row"] == source_index[sid],
            "Source/vector row differs",
        )
        for channel in ["dense", "bm25", "rrf"]:
            observed = stored["retrieval"][channel]
            c.require(
                [v["id"] for v in trace[channel]] == [v["id"] for v in observed],
                "Channel order differs",
            )
            c.require(
                np.allclose(
                    [v["score"] for v in trace[channel]],
                    [v["score"] for v in observed],
                    rtol=0,
                    atol=1e-12,
                ),
                "Retrieval score differs",
            )
            trace_checks += 1
        context = prompt["context"]
        c.require(
            context["definitions"] == old["context"]["definitions"]
            and context["guidelines"] == old["context"]["guidelines"]
            and context["example_groups"][:-1] == old["context"]["example_groups"],
            "Base blocks changed",
        )
        hit = {
            "id": selected["id"],
            "text": selected["text"],
            "meta": selected["meta"],
            "via": "rrf",
        }
        c.require(context["example_groups"][-1] == [c.GROUP, [hit]], "Wrong extra slot")
        suffix = "\n\n## Text to classify (language: en)\n" + r["text"]
        c.require(
            old["messages"][1]["text"].endswith(suffix),
            "Original query boundary differs",
        )
        block = (
            "## Additional labelled examples\n"
            + cautions["EXAMPLES_CAUTION"]
            + '\n\n- "'
            + " ".join(selected["text"].split())
            + '"\n  labels: hate='
            + str(selected["meta"]["gate"]).lower()
        )
        user = old["messages"][1]["text"][: -len(suffix)] + "\n\n" + block + suffix
        c.require(
            prompt["messages"] == [old["messages"][0], {"role": "user", "text": user}],
            "Independent literal binary prompt reconstruction differs",
        )
        delivery[str(selected["meta"]["gate"])] += 1
        if number % 1500 == 0:
            print(
                f"Independently checked {number}/12825 prompts and retrieved sources.",
                flush=True,
            )
    c.require(
        seen == {(i, p) for i in roster for p in expected}
        and len(choices) == 12150
        and all_reference == 675,
        "Incomplete payload matrix",
    )
    c.verify(prepared["outputs"])
    c.verify(plan["inputs"])
    c.verify(plan["original_bindings"])
    evidence = {
        "status": "all_sources_retrieval_and_payloads_verified",
        "created_utc": c.utc(),
        "prepared_sha256": c.sha(c.BASE / "prepared.json"),
        "checker_sha256": c.sha(__file__),
        "independent_retrieval_source_sha256": c.sha(
            c.ROOT / "analysis/feedback/explanations/calibration.py"
        ),
        "prompts": len(seen),
        "unchanged_references": all_reference,
        "retrieval_comparisons": len(choices),
        "channel_order_and_score_checks": trace_checks,
        "exact_pools_reconstructed": 18,
        "binary_records_checked": len(records),
        "delivered_binary_labels": dict(delivery),
        "max_selected_source_evaluation_cosine": max_similarity,
        "original_source_screen_receipt_sha256": c.sha(c.CANDIDATES / "complete.json"),
        "new_classifier_calls": 0,
        "new_embedding_calls": 0,
        "cuda_jobs": 0,
    }
    c.write(c.BASE / "independent_verification.json", evidence)
    c.write(
        c.BASE / "complete.json",
        {
            "status": "offline_preparation_independently_verified",
            "created_utc": c.utc(),
            "prepared_sha256": c.sha(c.BASE / "prepared.json"),
            "verification": c.rel(c.BASE / "independent_verification.json"),
            "verification_sha256": c.sha(c.BASE / "independent_verification.json"),
            "new_classifier_calls": 0,
        },
    )
    print(
        {
            k: evidence[k]
            for k in [
                "status",
                "prompts",
                "retrieval_comparisons",
                "exact_pools_reconstructed",
                "max_selected_source_evaluation_cosine",
            ]
        },
        flush=True,
    )


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with patch.object(socket.socket, "connect", c.forbid), patch.object(
        socket, "create_connection", c.forbid
    ), threadpool_limits(limits=2):
        main()
