from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
DEST = CORE / "severity_results_v1"
SOURCE = Path(__file__).resolve()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rel(path):
    return path.relative_to(ROOT).as_posix()


def main():
    result = read(DEST / "results.json")
    complete = read(DEST / "complete.json")
    plan = read(DEST / "plan.json")
    assert (
        digest(DEST / "results.json")
        == complete["results_sha256"]
        == digest(ROOT / complete["evidence"])
    )
    assert (
        digest(DEST / "plan.json") == complete["plan_sha256"] == result["plan_sha256"]
    )
    hashes = {
        rel(p): digest(p)
        for p in (
            SOURCE,
            DEST / "plan.json",
            DEST / "results.json",
            DEST / "complete.json",
        )
    }
    for name, h in result["verified_input_hashes"].items():
        assert digest(ROOT / name) == h
    ids = result["endpoint"]["scorable_ids"]
    rows = 0
    for name, meta in complete["assembled"].items():
        path = ROOT / name
        assert digest(path) == meta["sha256"]
        rr = records(path)
        assert [r["id"] for r in rr] == ids and len(rr) == meta["rows"] == 45
        for row in rr:
            src = Path("\\\\?\\" + str(ROOT / row["source"]))
            assert digest(src) == row["source_sha256"]
            if row["kind"] == "qwen_decoder_repeat":
                original = read(src)
                assert (
                    original["result"] == row["result"]
                    and original["n_valid"] == row["n_valid"]
                )
            else:
                matches = [r for r in records(src) if r["key"] == row["key"]]
                assert (
                    len(matches) == 1
                    and matches[0]["result"] == row["result"]
                    and row["n_valid"] == 1
                )
        rows += len(rr)
        hashes[name] = meta["sha256"]
    assert rows == 1485 and len(complete["assembled"]) == 33
    ep = ROOT / "Thesis_Log/astra_result/evidence/core_encoder_results_20260915_v5.json"
    old = read(ep)
    hashes[rel(ep)] = digest(ep)
    for arm in ("full", "kb"):
        current = result["conditions"]["encoder_" + arm]["summary"]
        previous = old["english_dev"][arm + "/en_dev_eval_severity"]
        assert math.isclose(
            current["mean"], previous["new_mean"], abs_tol=1e-14, rel_tol=0
        )
        assert math.isclose(
            current["sample_sd"], previous["new_sample_sd"], abs_tol=1e-14, rel_tol=0
        )
        assert all(
            math.isclose(value, p["new"]["macro_f1"], abs_tol=1e-14, rel_tol=0)
            for value, p in zip(current["values"], previous["per_seed"])
        )
    core_plan = read(CORE / "plan.json")
    api_plan = read(CORE / "api_execution_v1/plan.json")
    pools = {}
    by_key = {}
    for pool in ("small", "full"):
        kb_path = CORE / f"kb/{pool}/records.jsonl"
        prompt_path = CORE / f"retrieval_v1/{pool}/prompts.jsonl"
        assert digest(kb_path) == core_plan["generated_hashes"][rel(kb_path)]
        assert digest(prompt_path) == api_plan["input_hashes"][rel(prompt_path)]
        hashes[rel(kb_path)] = digest(kb_path)
        hashes[rel(prompt_path)] = digest(prompt_path)
        kb = {r["id"]: r for r in records(kb_path)}
        prompts = {r["key"]: r for r in records(prompt_path) if r["item_id"] in ids}
        assert set(prompts) == {"en/" + rid for rid in ids}
        label_counts = Counter()
        sources = Counter()
        coverage = Counter()
        details = {}
        for key, p in prompts.items():
            examples = p["retrieved"]["Labelled examples"]
            assert len(examples) == 5
            selected = [kb[e] for e in examples]
            assert all(r["kind"] == "example" and r["lang"] == "en" for r in selected)
            labels = [r.get("meta", {}).get("severity") for r in selected]
            assert all(x is None or x in ("low", "medium", "high") for x in labels)
            label_counts.update(x if x is not None else "unannotated" for x in labels)
            sources.update(r["source"] for r in selected)
            coverage[str(sum(x is not None for x in labels))] += 1
            details[key] = {
                "example_ids": examples,
                "severity_labels": labels,
                "definitions": p["retrieved"]["definitions"],
                "guidelines": p["retrieved"]["guidelines"],
            }
        pools[pool] = {
            "items": 45,
            "example_slots": 225,
            "severity_labels_by_slot": dict(label_counts),
            "examples_with_severity_per_input_distribution": dict(
                sorted(coverage.items())
            ),
            "source_slots": dict(sources),
            "details": details,
        }
        by_key[pool] = prompts
    abstract_equal = sum(
        pools["small"]["details"][key][kind] == pools["full"]["details"][key][kind]
        for key in by_key["small"]
        for kind in ("definitions", "guidelines")
    )
    out = (
        ROOT
        / "Thesis_Log/astra_result/evidence/core_severity_followup_20260918_v1.json"
    )
    assert not out.exists()
    evidence = {
        "status": "severity_publication_and_saved_contexts_verified",
        "severity_results_sha256": digest(DEST / "results.json"),
        "verified_assembled_files": 33,
        "verified_assembled_rows": rows,
        "encoder_means_sds_and_seeds_match_prior_audit": True,
        "retrieval_contexts": pools,
        "identical_ordered_abstract_lists_out_of_90": abstract_equal,
        "verified_hashes": hashes,
        "actual_API_calls": 0,
        "new_retrieval": False,
        "new_accuracy_scoring": False,
        "interpretation": "Post-hoc descriptive saved-context counts only, including repeated slots. This does not establish the cause of the severity performance difference. Full/small comparisons change the candidate pool and composition; no new condition is selected here.",
    }
    with out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(evidence, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in evidence.items()
                if k not in ("verified_hashes", "retrieval_contexts")
            },
            indent=2,
        )
    )
    for pool, info in pools.items():
        print(pool, json.dumps({k: v for k, v in info.items() if k != "details"}))
    print("Follow-up SHA256: " + digest(out))


if __name__ == "__main__":
    main()
