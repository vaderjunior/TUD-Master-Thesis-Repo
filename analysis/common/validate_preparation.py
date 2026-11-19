from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import pandas as pd
from analysis.knowledge.banks import design as p

ROOT, BASE = (p.ROOT, p.BASE)
OUT = ROOT / "Thesis_Log/astra_result/evidence/core_preparation_check_20260915_v1.json"


def main():
    assert not OUT.exists(), "Preserve existing verification receipt"
    plan = p.u.read_json(BASE / "plan.json")
    ep = p.u.read_json(BASE / "encoder_plan.json")
    hashes = {
        **plan["input_hashes"],
        **plan["generated_hashes"],
        p.u.rel(BASE / "plan.json"): p.u.digest(BASE / "plan.json"),
        p.u.rel(Path(__file__)): p.u.digest(Path(__file__)),
    }
    assert all((p.u.digest(ROOT / n) == h for n, h in hashes.items()))
    assert p.u.environment() == plan["environment"]
    for cache in plan["base_caches"].values():
        assert all(
            (
                p.u.digest(Path(cache["snapshot"]) / n) == h
                for n, h in cache["files"].items()
            )
        )
    full = p.u.read_lines(BASE / "kb/full/records.jsonl")
    small = p.u.read_lines(BASE / "kb/small/records.jsonl")
    assert len(full) == 71131 and len(small) == 340
    byid = {r["id"]: r for r in full}
    assert len(byid) == len(full)
    assert len({p.u.canonical_text(r["text"]) for r in full}) == len(full)
    assert all((byid[r["id"]] == r for r in small))
    enex = set(p.u.read_json(p.EN / "policy.json")["final_excluded_ids"])
    deex = set(p.u.read_json(p.DE / "policy.json")["final_excluded_ids"])
    for r in full:
        if r["kind"] == "example":
            ids = r["meta"].get("astra_train_ids", [p.cid(r["id"])])
            assert not set(ids) & (enex if r["lang"] == "en" else deex)
    split_checks = {}
    text_variants = []
    for key, meta in ep["split_summaries"].items():
        arm, head = key.split("/")
        ex = deex if arm == "de" else enex
        sets = {}
        for split in ("train", "val"):
            source = pd.read_parquet(ROOT / meta["source_files"][split])
            new = pd.read_parquet(BASE / f"encoder_data/{arm}/{head}_{split}.parquet")
            keep = source.id.astype(str).map(p.cid).map(lambda rid: rid not in ex)
            assert new.reset_index(drop=True).equals(
                source.loc[keep].reset_index(drop=True)
            )
            sets[split] = {p.cid(x) for x in new.id}
            assert not sets[split] & ex
        assert not sets["train"] & sets["val"]
        rr = full if arm != "kb" else small
        lang = "de" if arm == "de" else "en"
        col = p.GOLD[head]
        expected = [
            r
            for r in rr
            if r["kind"] == "example"
            and r["lang"] == lang
            and (r["meta"].get(col) is not None)
            and (head in ("hate", "legal") or r["meta"].get("gate") is True)
        ]
        union = sets["train"] | sets["val"]
        assert len(expected) == len(union)
        frames = [
            pd.read_parquet(BASE / f"encoder_data/{arm}/{head}_{s}.parquet")
            for s in ("train", "val")
        ]
        original = {p.cid(x["id"]): x for df in frames for x in df.to_dict("records")}
        observed = set()
        for r in expected:
            ids = set(r["meta"].get("astra_train_ids", [p.cid(r["id"])])) & union
            assert len(ids) == 1
            rid = next(iter(ids))
            observed.add(rid)
            raw = original[rid]
            actual = p.u.clean(raw[col])
            wanted = r["meta"][col]
            assert (sorted(actual) if isinstance(actual, list) else actual) == (
                sorted(wanted) if isinstance(wanted, list) else wanted
            )
            assert p.u.canonical_text(raw["text"]) == p.u.canonical_text(r["text"])
            if raw["text"] != r["text"]:
                text_variants.append(
                    {
                        "arm": arm,
                        "head": head,
                        "train_id": rid,
                        "record_id": r["id"],
                        "note": "Canonical text equal but raw case/whitespace differs; encoder preserves its original source text.",
                    }
                )
        assert observed == union
        split_checks[key] = {s: len(v) for s, v in sets.items()}
    assert len(ep["train_jobs"]) == 21 and len(ep["weight_reuse"]) == 9
    assert (
        len(
            {
                (x["arm"], x["head"], x["seed"])
                for x in ep["train_jobs"] + ep["weight_reuse"]
            }
        )
        == 30
    )
    for x in ep["weight_reuse"]:
        old = p.u.read_json(ROOT / x["path"] / "meta.json")
        assert old["epochs"] == 20 and old["early_stopping"] is False
        assert not ep["split_summaries"][f"{x['arm']}/{x['head']}"]["removed_ids"][
            "train"
        ]
    inputs = p.u.read_lines(BASE / "dev_inputs.jsonl")
    bykey = {x["key"]: x for x in inputs}
    assert len(bykey) == len(inputs) == 1295
    sets = p.u.read_json(BASE / "fewshot_sets.json")
    eligible = p.u.read_json(BASE / "fewshot_eligible_sets.json")
    for x in sets + eligible:
        pool = [
            r
            for r in full
            if r["kind"] == "example"
            and r["lang"] == x["lang"]
            and (not r["meta"].get("illustrative_only"))
        ]
        if x in eligible:
            col = p.GOLD[x["head"]]
            pool = [
                r
                for r in pool
                if r["meta"].get(col) is not None
                and (x["head"] in ("hate", "legal") or r["meta"].get("gate") is True)
            ]
        assert len(pool) == x["pool_size"]
        assert random.Random(x["set_seed"]).sample(pool, 5) == x["records"]
        assert len({r["id"] for r in x["records"]}) == 5
    prompts = p.u.read_lines(BASE / "static_prompts.jsonl")
    seen = set()
    prompt_counts = Counter()
    for r in prompts:
        pair = (r["key"], r["condition"])
        assert pair not in seen
        seen.add(pair)
        row = bykey[r["key"]]
        p.check_messages(r["messages"], row["text"])
        assert (
            hashlib.sha256(p.u.json_bytes(r["messages"])).hexdigest()
            == r["messages_sha256"]
        )
        assert r["context"]["definitions"] == [] and r["context"]["guidelines"] == []
        prompt_counts[r["condition"]] += 1
        if r["condition"] == "zero_shot":
            assert r["context"]["example_groups"] == []
        else:
            groups = r["context"]["example_groups"]
            assert len(groups) == 1 and len(groups[0][1]) == 5
            assert all(
                (
                    x["via"] == "static" and x["meta"]["lang"] == row["lang"]
                    for x in groups[0][1]
                )
            )
            assert all((byid[x["id"]]["text"] == x["text"] for x in groups[0][1]))
        assert (
            p.prompt.render_messages(
                row["text"], row["lang"], r["context"], "corrected"
            )
            == r["messages"]
        )
    expected_prompts = (
        len(inputs) * 4
        + sum(map(len, plan["eligible_fewshot_conditions"]["input_keys"].values())) * 3
    )
    assert len(prompts) == plan["static_prompt_count"] == expected_prompts
    ordinary = (
        len(inputs) * 18 * 3
        + sum(map(len, plan["eligible_fewshot_conditions"]["input_keys"].values()))
        * 9
        * 3
    )
    assert ordinary == plan["budget"]["ordinary_model_votes"] == 96822
    blocks = []
    for subset, n in plan["scope_counts"].items():
        for cond in plan["conditions"]:
            blocks.append(
                {
                    "subset": subset,
                    "condition": cond["name"],
                    "repeats": 3,
                    "inputs": n,
                    "ordinary_votes": n * 9,
                }
            )
        if subset != "en_dev_eval_main":
            head = p.SCOPES[subset].removesuffix("_primary").removesuffix("_secondary")
            for seed in (42, 43, 44):
                blocks.append(
                    {
                        "subset": subset,
                        "condition": f"fewshot_eligible_{head}_s{seed}",
                        "repeats": 3,
                        "inputs": n,
                        "ordinary_votes": n * 9,
                    }
                )
    assert sum((b["ordinary_votes"] for b in blocks)) == ordinary
    assert all((p.u.digest(ROOT / n) == h for n, h in hashes.items()))
    result = {
        "id": "ASTRA-CORE-PREP-CHECK-20260915-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": p.u.digest(BASE / "plan.json"),
        "verified_hash_count": len(hashes),
        "input_hashes": hashes,
        "verified_pool_counts": plan["pool_counts"],
        "verified_encoder_splits": split_checks,
        "encoder_canonical_text_variants": text_variants,
        "verified_static_prompts": len(prompts),
        "prompt_counts": dict(prompt_counts),
        "future_api_blocks": blocks,
        "budget": plan["budget"],
        "status": "CPU preparation and independent verification passed. No trained models, new index or model predictions created.",
        "limitations": "Historical compact checkpoint reuse is supported by fixed-epoch metadata, not a recovered exact old trainer. Task-aware FS selection is a declared separate intervention; uniform sparse-label controls remain included. No new outcome has been measured, and post-selection development/test history remains applicable.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                "path": p.u.rel(OUT),
                "sha256": p.u.digest(OUT),
                "plan_sha256": result["plan_sha256"],
                "hashes": len(hashes),
                "encoder_splits": split_checks,
                "text_variants": len(text_variants),
                "static_prompts": len(prompts),
                "api_blocks": len(blocks),
                "ordinary_votes": ordinary,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
