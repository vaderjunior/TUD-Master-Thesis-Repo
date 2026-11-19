import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from analysis.feedback.human import feedback_common as c
from analysis.common import prompt as renderer
from analysis.common import feedback_inputs as tokens


def original_inputs():
    done = c.read(c.HUMAN / "complete.json")
    require = c.require
    require(
        done["status"] == "completed_descriptive_human_review",
        "Human review incomplete",
    )
    paths = [c.HUMAN / "complete.json"]
    for e in done["evidence"]:
        require(
            c.sha(c.ROOT / e["path"]) == e["sha256"],
            "Human evidence changed: " + e["path"],
        )
        paths.append(c.ROOT / e["path"])
    manifest = c.read(c.HUMAN / "input_manifest.json")
    c.verify(manifest["inputs"])
    paths += [c.ROOT / p for p in manifest["inputs"]]
    for name in (
        "Reviewer_A_Part_2_inspection.json",
        "Reviewer_B_Part_1_inspection.json",
    ):
        paths.append(c.HUMAN / name)
    active = c.read(c.A15 / "execution/plan.json")
    for name in ("roster.jsonl", "prompts.jsonl"):
        p = c.A15 / name
        require(
            active["input_hashes"][c.rel(p)] == c.sha(p), "Parent prompt input changed"
        )
        paths.append(p)
    scope_done = c.read(c.SCOPE / "complete.json")
    for name in ("components.json", "eligibility.jsonl", "plan.json"):
        p = c.SCOPE / name
        require(
            scope_done["outputs"][c.rel(p)] == c.sha(p),
            "Historical overlap result changed",
        )
        paths.append(p)
    scope = c.read(c.SCOPE / "plan.json")
    for p in (
        c.CORE / "dev_inputs.jsonl",
        c.CORE / "retrieval_v1/query_vectors.npy",
        c.CORE / "retrieval_v1/query_vector_sources.jsonl",
    ):
        require(scope["inputs"][c.rel(p)] == c.sha(p), "Cached vector evidence changed")
        paths.append(p)
    paths += [c.SCOPE / "complete.json", c.A15 / "execution/plan.json"]
    replay = [
        p
        for p in active["input_hashes"]
        if p.endswith("cold_replay_review_v1/sensitivity_plan.json")
    ]
    require(len(replay) == 1, "Expected one inherited retrieval sensitivity")
    paths.append(c.ROOT / replay[0])
    for p in (
        c.BASE / "STUDY_PROTOCOL.md",
        Path(__file__),
        Path(c.__file__),
        Path(renderer.__file__),
        Path(tokens.__file__),
        c.ROOT / "src/hsrag/prompt.py",
        c.ROOT / "config/taxonomy.yaml",
        c.ROOT / "prompts/classify_v1.txt",
        tokens.TOK / "complete.json",
        tokens.TOK / "tokenizer.json",
    ):
        paths.append(p)
    return (c.bindings(list(dict.fromkeys(paths))), replay[0])


def assemble():
    items = sorted(
        c.read(c.HUMAN / "final_result_summary.json")["items"],
        key=lambda x: x["packet_id"],
    )
    c.require(
        len(items) == 24 and len({r["id"] for r in items}) == 24,
        "Review sample differs",
    )
    corrections = [r for r in items if c.agreed_correction(r) is not None]
    c.require(
        tuple((r["packet_id"] for r in corrections)) == c.SOURCE_ALIASES,
        "Agreed correction census differs; review, do not silently replace",
    )
    refs = {}
    ids = {"en/" + r["id"] for r in items}
    for p in c.rows(c.A15 / "prompts.jsonl"):
        if p["condition"] == "reference5" and p["key"] in ids:
            refs[p["key"]] = p
    c.require(set(refs) == ids, "Parent reference missing")
    dev = list(c.rows(c.CORE / "dev_inputs.jsonl"))
    origins = list(c.rows(c.CORE / "retrieval_v1/query_vector_sources.jsonl"))
    vectors = np.load(
        c.CORE / "retrieval_v1/query_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    ix = {r["key"]: i for i, r in enumerate(dev)}
    x = []
    for r in items:
        k = "en/" + r["id"]
        i = ix[k]
        c.require(
            dev[i]["text"] == refs[k]["text"] == r["text"]
            and origins[i]["key"] == k
            and (origins[i]["text_sha256"] == c.shared.text_sha(r["text"])),
            "Exact vector/text identity mismatch",
        )
        x.append(np.asarray(vectors[i], dtype=np.float64))
    x = np.stack(x)
    c.require(
        np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1, atol=0.0002),
        "Vector norms differ",
    )
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    source_indices = [
        next((i for i, r in enumerate(items) if r["packet_id"] == s))
        for s in c.SOURCE_ALIASES
    ]
    similarities = x @ x[source_indices].T
    components = c.read(c.SCOPE / "components.json")
    component = {n: row for row in components for n in row["nodes"]}
    eligibility = {
        r["node_id"]: r
        for r in c.rows(c.SCOPE / "eligibility.jsonl")
        if r["role"] == "development"
    }
    source_clusters = {
        component["development|en/" + r["id"]]["cluster"] for r in corrections
    }
    c.require(len(source_clusters) == 3, "Correction sources overlap")
    roster = []
    choices = []
    for i, r in enumerate(items):
        k = "en/" + r["id"]
        node = "development|" + k
        cluster = component[node]["cluster"]
        e = eligibility[node]
        ranking = sorted(
            range(3), key=lambda j: (-float(similarities[i, j]), c.SOURCE_ALIASES[j])
        )
        chosen = corrections[ranking[0]]
        source = r["packet_id"] in c.SOURCE_ALIASES
        if source:
            c.require(chosen["id"] == r["id"], "Source self-retrieval failed")
        eligible = not source and cluster not in source_clusters
        row = dict(
            r,
            key=k,
            lang="en",
            population="source_uptake" if source else "other_reviewed",
            original_cluster=r["cluster"],
            cluster=cluster,
            is_correction_source=source,
            transfer_eligible=eligible,
            touches_training="component_touches_TRAIN"
            in component[node]["exclusion_reasons"],
            overlap_evidence=e,
            selected_correction=chosen["packet_id"],
        )
        roster.append(row)
        choices.append(
            {
                "key": k,
                "selected_alias": chosen["packet_id"],
                "selected_id": chosen["id"],
                "selection": "highest normalized cached-vector cosine; alias tie-break",
                "ranking": [
                    {"alias": c.SOURCE_ALIASES[j], "cosine": float(similarities[i, j])}
                    for j in ranking
                ],
                "self_delivery": source,
                "same_source_in_both_feedback_arms": True,
            }
        )
    return (roster, corrections, refs, choices)


def prepare():
    if (c.BASE / "prepared.json").exists():
        c.verify(c.read(c.BASE / "preparation_plan.json")["inputs"])
        c.verify(c.read(c.BASE / "prepared.json")["outputs"])
        print("Existing human-feedback preparation verified; no rewrite.")
        return
    c.require(
        not (c.BASE / "execution/live").exists(), "Preserve live human feedback outputs"
    )
    guard = c.active_preservation()
    inputs, replay_path = original_inputs()
    planpath = c.BASE / "preparation_plan.json"
    if planpath.exists():
        c.require(
            c.read(planpath)["inputs"] == inputs,
            "Interrupted preparation inputs changed",
        )
    else:
        c.save(
            planpath,
            {
                "created_utc": c.utc(),
                "inputs": inputs,
                "active_queue_preservation": guard,
                "new_model_calls": 0,
                "new_embeddings": 0,
            },
        )
    roster, corrections, refs, choices = assemble()
    replay_keys = c.read(c.ROOT / replay_path)["excluded_keys"]
    for r in roster:
        r["retrieval_replay_flag"] = r["key"] in replay_keys
    sources = {r["packet_id"]: r for r in corrections}
    c.jsonlines(c.BASE / "roster.jsonl", roster)
    c.jsonlines(c.BASE / "choices.jsonl", choices)
    c.save(
        c.BASE / "corrections.json",
        [
            {
                "packet_id": r["packet_id"],
                "id": r["id"],
                "text": r["text"],
                "human": c.correction_record(r, "human_feedback5"),
                "dataset": c.correction_record(r, "dataset_feedback5"),
                "original_model": r["model_repeat_0"],
                "guide": "project",
                "model_source": r["model_source"],
                "human_judgments": {
                    v: r["judgments"][v + "_project"] for v in ("A", "B")
                },
            }
            for r in corrections
        ],
    )
    science = {
        "inputs": 24,
        "conditions": list(c.CONDITIONS),
        "predictions": 216,
        "ordinary_calls": 648,
        "decoder_repeats": 3,
        "votes_per_prediction": 3,
        "source_aliases": list(c.SOURCE_ALIASES),
        "scope": "exploratory, previously selected/exposed human-review sample; one fixed feedback update",
        "primary_descriptive_comparison": ["human_feedback5", "dataset_feedback5"],
        "secondary_descriptive_comparison": ["human_feedback5", "reference5"],
        "references": ["A", "B", "dataset"],
        "primary_endpoint": "gate-and-complete-type-set agreement with each reviewer separately",
        "supporting_endpoints": [
            "gate agreement",
            "mean type-set Jaccard",
            "per-label confusion",
            "fix/regression transitions",
        ],
        "analysis_unit": "text; repeats averaged within text; reviewers reported separately",
        "inference": "no population inference, p-values, confirmatory confidence intervals, expert or real-multiround claim",
        "missing": "per-reference common-valid all-condition/all-repeat text mask; retain separate eligibility and null counts; opposing bounds",
        "sensitivities": [
            "all three votes valid",
            "both reviewers resolved",
            "exclude inherited replay keys",
            "exclude TRAIN-touching components",
            "one per overlap cluster",
        ],
        "excluded_transfer_keys": [
            r["key"] for r in roster if not r["transfer_eligible"]
        ],
        "protocol_sha256": c.sha(c.BASE / "STUDY_PROTOCOL.md"),
    }
    c.save(c.BASE / "scientific_plan_v1.json", science)
    scientific_sha = c.sha(c.BASE / "scientific_plan_v1.json")
    tok = tokens.tokenizer()
    prompts = []
    for r in roster:
        base = refs[r["key"]]
        chosen = sources[r["selected_correction"]]
        for condition in c.CONDITIONS:
            correction = (
                None
                if condition == "reference5"
                else c.correction_record(chosen, condition)
            )
            ctx = c.context(base["context"], correction)
            messages = renderer.render_messages(r["text"], "en", ctx, "corrected")
            c.require(messages[0] == base["messages"][0], "System prompt changed")
            if condition == "reference5":
                c.require(messages == base["messages"], "Reference messages changed")
            prompts.append(
                {
                    "key": r["key"],
                    "item_id": r["id"],
                    "condition": condition,
                    "population": r["population"],
                    "lang": "en",
                    "text": r["text"],
                    "context": ctx,
                    "messages": messages,
                    "context_sha256": c.sig(ctx),
                    "messages_sha256": c.sig(messages),
                    "reference_input_tokens": tokens.token_count(tok, messages),
                    "source_id": chosen["id"] if correction else None,
                    "scientific_plan_sha256": scientific_sha,
                }
            )
    c.jsonlines(c.BASE / "prompts.jsonl", prompts)
    other = [r for r in roster if r["transfer_eligible"]]
    identical = [
        r["packet_id"]
        for r in roster
        if sources[r["selected_correction"]]["packet_id"] == "R059"
    ]
    summary = {
        "status": "prepared_requires_independent_check_and_mock",
        "created_utc": c.utc(),
        "texts": 24,
        "correction_sources": 3,
        "other_texts": 21,
        "source_disjoint_transfer_texts": len(other),
        "conditions": 3,
        "prompts": len(prompts),
        "predictions": 216,
        "ordinary_calls": 648,
        "delivery_counts_all": dict(
            Counter((r["selected_correction"] for r in roster))
        ),
        "delivery_counts_other": dict(
            Counter((r["selected_correction"] for r in other))
        ),
        "identical_dataset_human_prompt_aliases": identical,
        "reviewer_coverage_other": {
            v: {
                "gate": sum((c.reference_target(r, v)[0] is not None for r in other)),
                "complete": sum(
                    (c.reference_target(r, v)[1] is not None for r in other)
                ),
            }
            for v in ("A", "B")
        },
        "both_resolved_other": sum(
            (
                all((c.reference_target(r, v)[0] is not None for v in ("A", "B")))
                for r in other
            )
        ),
        "train_touching_aliases": [
            r["packet_id"] for r in roster if r["touches_training"]
        ],
        "replay_flag_aliases": [
            r["packet_id"] for r in roster if r["retrieval_replay_flag"]
        ],
        "token_min": min((p["reference_input_tokens"] for p in prompts)),
        "token_max": max((p["reference_input_tokens"] for p in prompts)),
        "new_model_calls": 0,
        "new_embeddings": 0,
        "original_labels_changed": False,
        "new_human_judgments": 0,
    }
    c.save(c.BASE / "summary.json", summary)
    c.verify(inputs)
    c.require(
        c.active_preservation() == guard,
        "Active queue bindings changed during preparation",
    )
    outputs = [
        c.BASE / p
        for p in (
            "roster.jsonl",
            "choices.jsonl",
            "corrections.json",
            "scientific_plan_v1.json",
            "prompts.jsonl",
            "summary.json",
        )
    ]
    c.save(
        c.BASE / "prepared.json",
        {
            "status": "prepared_requires_independent_check",
            "outputs": c.bindings(outputs),
            "created_utc": c.utc(),
        },
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "inspect"])
    args = parser.parse_args()
    with c.offline():
        if args.mode == "prepare":
            prepare()
        else:
            roster, corrections, _, choices = assemble()
            print(
                {
                    "sources": [r["packet_id"] for r in corrections],
                    "transfer_eligible": sum((r["transfer_eligible"] for r in roster)),
                    "train_touching": [
                        r["packet_id"] for r in roster if r["touches_training"]
                    ],
                    "delivery": dict(Counter((x["selected_alias"] for x in choices))),
                }
            )
