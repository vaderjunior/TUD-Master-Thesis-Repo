"""Sampled banks verify design."""

import hashlib
import heapq
import json
from pathlib import Path
import socket
from analysis.knowledge.sampled_banks import verify_annotations as independent

ROOT = independent.ROOT
BASE = ROOT / "experiments/astra/sq1_pool_size_composition_20260926_v1"
CORE = independent.CORE


def main():
    socket.socket.connect = independent.no_network
    socket.create_connection = independent.no_network
    assert not (BASE / "independent_check.json").exists(), "Preserve prior verification"
    plan = independent.read(BASE / "scientific_plan.json")
    manifests = independent.read(BASE / "pool_manifests.json")
    prepared = independent.read(BASE / "preparation.json")
    assert independent.digest(BASE / "scientific_plan.json") == prepared["scientific_plan_sha256"]
    for name, expected in plan["input_hashes"].items():
        assert independent.digest(ROOT / name) == expected, name
    assert len(manifests) == 9 and len(plan["conditions"]) == len(set(plan["conditions"])) == 11
    full = independent.rows(CORE / "kb/full/records.jsonl")
    by_id = {r["id"]: r for r in full}
    coverage_plan = independent.read(
        ROOT / "experiments/astra/sq1_annotation_coverage_20260926_v1/analysis_plan.json"
    )
    reconstructed = {}
    checked = 0
    prefixes = {}
    compositions = []
    for lang in ("en", "de"):
        examples = [r for r in full if r["kind"] == "example" and r["lang"] == lang]
        for seed in (42, 43, 44):

            def key(rid):
                payload = "|".join(("astra-sq1-pools-v1", lang, str(seed), rid)).encode("utf8")
                return (hashlib.sha256(payload).hexdigest(), rid)

            reconstructed[lang, seed] = heapq.nsmallest(5000, [r["id"] for r in examples], key=key)
    for m in manifests:
        assert m["abstract_ids"] == [r["id"] for r in full if r["kind"] != "example"]
        for lang in ("en", "de"):
            sub = m["languages"][lang]
            n = sub["size"]
            ids = sub["example_ids"]
            assert n == (
                {"en": 147, "de": 153}[lang] if m["size_name"] == "compact" else int(m["size_name"])
            )
            assert sub["selection_prefix_ids"] == reconstructed[lang, m["seed"]][:n]
            chosen = set(sub["selection_prefix_ids"])
            assert len(ids) == len(set(ids)) == n and set(ids) == chosen
            assert ids == [r["id"] for r in full if r["id"] in chosen]
            selected = [by_id[rid] for rid in ids]
            assert all(
                (
                    r["kind"] == "example"
                    and r["lang"] == lang
                    and (not r["meta"].get("illustrative_only"))
                    for r in selected
                )
            )
            assert (
                independent.describe_independently(selected, coverage_plan["labels"])
                == sub["inventory"]
            )
            prefixes[lang, m["seed"], n] = set(ids)
            checked += 1
            compositions.append(
                {
                    "condition": m["condition"],
                    "lang": lang,
                    "size": n,
                    "gate": sub["inventory"]["gate"],
                    "annotated": {f: s["annotated"] for f, s in sub["inventory"]["fields"].items()},
                }
            )
    for lang in ("en", "de"):
        for seed in (42, 43, 44):
            n = 147 if lang == "en" else 153
            assert (
                prefixes[lang, seed, n]
                < prefixes[lang, seed, 1000]
                < prefixes[lang, seed, 5000]
                < set(by_id)
            )
    assert sum((r["items"] for r in plan["evaluation"]["all_rosters"].values())) == 1295
    assert 1295 * len(plan["conditions"]) * 3 == plan["budget"]["predictions"] == 42735
    assert plan["budget"]["ordinary_responses"] == 128205
    assert (
        plan["evaluation"]["primary_family"]["contrasts"]
        == 2 * len(plan["evaluation"]["focal_rosters"])
        == 6
    )
    verification = {
        "status": "scientific_design_and_18_TRAIN_subsets_verified_retrieval_not_prepared",
        "scientific_plan_sha256": independent.digest(BASE / "scientific_plan.json"),
        "pool_manifests_sha256": independent.digest(BASE / "pool_manifests.json"),
        "checker_sha256": independent.digest(Path(__file__)),
        "input_hashes_verified": len(plan["input_hashes"]),
        "language_subsets": checked,
        "nested_chains_verified": 6,
        "selection_uses_TRAIN_identity_only": True,
        "pool_composition": compositions,
        "no_API": True,
        "no_retrieval": True,
        "no_CUDA": True,
    }
    data = (
        json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf8")
    for path in [
        BASE / "independent_check.json",
        ROOT / "Thesis_Log/astra_result/evidence/sq1_pool_design_check_20260926_v1.json",
    ]:
        with path.open("xb") as f:
            f.write(data)
    print(json.dumps({k: v for k, v in verification.items() if k != "pool_composition"}, indent=2))


if __name__ == "__main__":
    main()
