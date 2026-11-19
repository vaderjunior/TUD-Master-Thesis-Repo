from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
DEST = CORE / "secondary_results_v2/en_dev_eval_targets"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p):
    return p.relative_to(ROOT).as_posix()


def records(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    result = read(DEST / "results.json")
    receipt = read(DEST / "complete.json")
    assert (
        digest(DEST / "results.json")
        == receipt["results_sha256"]
        == digest(ROOT / receipt["evidence"])
    )
    assert digest(DEST / "plan.json") == receipt["plan_sha256"] == result["plan_sha256"]
    ids = result["endpoint"]["scorable_ids"]
    labels = result["endpoint"]["all_labels"]
    assert len(ids) == len(set(ids)) == 56
    hashes = {
        rel(p): digest(p)
        for p in (
            Path(__file__).resolve(),
            DEST / "results.json",
            DEST / "complete.json",
            DEST / "plan.json",
        )
    }
    core_plan = read(CORE / "plan.json")
    api_plan = read(CORE / "api_execution_v1/plan.json")
    pools = {}
    for pool in ("small", "full"):
        kb_path = CORE / f"kb/{pool}/records.jsonl"
        prompt_path = CORE / f"retrieval_v1/{pool}/prompts.jsonl"
        assert digest(kb_path) == core_plan["generated_hashes"][rel(kb_path)]
        assert digest(prompt_path) == api_plan["input_hashes"][rel(prompt_path)]
        hashes.update(
            {rel(kb_path): digest(kb_path), rel(prompt_path): digest(prompt_path)}
        )
        kb = {r["id"]: r for r in records(kb_path)}
        selected_prompts = [
            r for r in records(prompt_path) if r["key"] in {"en/" + rid for rid in ids}
        ]
        prompts = {r["key"]: r for r in selected_prompts}
        assert len(prompts) == len(selected_prompts) == 56
        annotation = Counter()
        occurrences = Counter()
        sources = Counter()
        per_input = Counter()
        details = {}
        for key, p in prompts.items():
            example_ids = p["retrieved"]["Labelled examples"]
            assert len(example_ids) == 5
            selected = [kb[e] for e in example_ids]
            assert all(r["kind"] == "example" and r["lang"] == "en" for r in selected)
            targets = [r.get("meta", {}).get("target_groups") for r in selected]
            for value in targets:
                assert (
                    value is None
                    or isinstance(value, list)
                    and set(value) <= set(labels)
                )
                annotation[
                    (
                        "unannotated"
                        if value is None
                        else "annotated_empty" if not value else "annotated_nonempty"
                    )
                ] += 1
                if value is not None:
                    occurrences.update(set(value))
            per_input[str(sum(value is not None for value in targets))] += 1
            sources.update(r["source"] for r in selected)
            details[key] = {
                "example_ids": example_ids,
                "target_labels": targets,
                "definitions": p["retrieved"]["definitions"],
                "guidelines": p["retrieved"]["guidelines"],
            }
        assert sum(annotation.values()) == 280 and sum(per_input.values()) == 56
        pools[pool] = {
            "items": 56,
            "example_slots": 280,
            "annotation_slots": dict(annotation),
            "label_occurrences": {label: occurrences[label] for label in labels},
            "annotated_examples_per_input": dict(sorted(per_input.items())),
            "source_slots": dict(sources),
            "details": details,
        }
    equal = sum(
        pools["small"]["details"][key][kind] == pools["full"]["details"][key][kind]
        for key in pools["small"]["details"]
        for kind in ("definitions", "guidelines")
    )
    schedule = read(CORE / "api_execution_v1/schedule.json")
    stage = next(s for s in schedule["stages"] if s["id"] == "stage_08_secondary")
    inputs = {(j["key"], j["subset"]) for j in stage["jobs"]}
    assert len(inputs) == 100 and len(stage["jobs"]) == 2700
    assert not (CORE / "api_execution_v1/stages/stage_08_secondary").exists()
    out = (
        ROOT / "Thesis_Log/astra_result/evidence/core_targets_followup_20260918_v1.json"
    )
    evidence = {
        "status": "targets_saved_contexts_verified",
        "results_sha256": digest(DEST / "results.json"),
        "retrieval_contexts": pools,
        "identical_ordered_abstract_lists_out_of_112": equal,
        "next_stage_inputs": dict(Counter(subset for _, subset in inputs)),
        "next_stage_started": False,
        "verified_hashes": hashes,
        "actual_API_calls": 0,
        "new_retrieval": False,
        "new_accuracy_scoring": False,
        "interpretation": "Post-hoc descriptive context counts, including repeated slots. Missing annotations differ from annotated empty lists. Multilabel occurrence totals may exceed annotated slots. Counts do not establish a causal explanation or justify changing the frozen programme.",
    }
    with out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(evidence, f, indent=2)
        f.write("\n")
    for pool, info in pools.items():
        print(pool, json.dumps({k: v for k, v in info.items() if k != "details"}))
    print("Equal abstracts:", equal, "Next stage:", evidence["next_stage_inputs"])
    print("Follow-up SHA256:", digest(out))


if __name__ == "__main__":
    main()
