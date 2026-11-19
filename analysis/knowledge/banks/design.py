import os

"""Banks design."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import pandas as pd
from analysis.knowledge.banks import build as u
from analysis.common import prompt as prompt

ROOT = u.ROOT
BASE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
EN = ROOT / "experiments/astra/english_kb_cleanup_20260914_v1"
DE = ROOT / "experiments/astra/german_kb_comparison_20260913_v1"
OUT = BASE / "plan.json"
SCOPES = {
    "en_dev_eval_main": "hate",
    "en_dev_eval_targets": "target_group",
    "en_dev_eval_types": "hate_type_secondary",
    "en_dev_eval_sq3_types": "hate_type_primary",
    "en_dev_eval_severity": "severity",
    "de_dev_eval": "hate",
    "de_legal_dev_eval": "legal",
}
HEADS = {
    "full": ["hate", "target_group", "hate_type", "severity"],
    "kb": ["hate", "target_group", "hate_type", "severity"],
    "de": ["hate", "legal"],
}
GOLD = {
    "hate": "gate",
    "target_group": "target_groups",
    "hate_type": "hate_types",
    "severity": "severity",
    "legal": "legal",
}


def own(path):
    path = Path(path).resolve()
    assert (
        path.is_relative_to(BASE.resolve()) and path != BASE.resolve()
    ), "Write outside isolated preparation"
    return path


def put(path, obj):
    path = own(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf8")
    if path.exists():
        assert path.read_bytes() == data, "Never overwrite differing prepared evidence"
    else:
        with path.open("xb") as f:
            f.write(data)


def jsonl(path, rows):
    path = own(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists()
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(
                json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def cid(rid):
    return str(rid).replace("ex-detox-legal-", "detox-").removeprefix("ex-")


def check_messages(messages, text):
    assert [m["role"] for m in messages] == ["system", "user"]
    assert all(
        (set(m) == {"role", "text"} and isinstance(m["text"], str) for m in messages)
    )
    assert messages[-1]["text"].endswith(text)


def combine(en, de, abstracts):
    assert all((r["kind"] == "example" and r["lang"] == "en" for r in en))
    assert all((r["kind"] == "example" and r["lang"] == "de" for r in de))
    assert all((r["kind"] in ("definition", "guideline") for r in abstracts))
    rows = abstracts + en + de
    assert len({r["id"] for r in rows}) == len(rows), "Duplicate record IDs"
    assert len({u.canonical_text(r["text"]) for r in rows}) == len(
        rows
    ), "Cross-pool text collision needs review"
    return rows


def filtered(frame, excluded):
    ids = [cid(x) for x in frame.id]
    assert len(set(ids)) == len(ids), "Non-unique canonical IDs"
    mask = [x not in excluded for x in ids]
    return (frame.loc[mask].copy(), [x for x in ids if x in excluded])


def hit(record, via="static"):
    return {
        "id": record["id"],
        "text": record["text"],
        "via": via,
        "meta": {
            **record["meta"],
            **{k: record[k] for k in ("lang", "source", "dimension", "label", "kind")},
        },
    }


def fixed_sets(records, seeds=(42, 43, 44)):
    selected = []
    for lang in ("en", "de"):
        pool = [
            r
            for r in records
            if r["kind"] == "example"
            and r["lang"] == lang
            and (not r["meta"].get("illustrative_only"))
        ]
        assert len(pool) >= 5
        for seed in seeds:
            picked = random.Random(seed).sample(pool, 5)
            selected.append(
                {
                    "lang": lang,
                    "set_seed": seed,
                    "pool_size": len(pool),
                    "records": picked,
                    "coverage": {
                        "hate_positive": sum(
                            (r["meta"].get("gate") is True for r in picked)
                        ),
                        "annotated": {
                            g: sum((r["meta"].get(g) is not None for r in picked))
                            for g in u.GOLD
                        },
                    },
                }
            )
    return selected


def record_sources(records, head):
    col = GOLD[head]
    result = {}
    for r in records:
        m = r["meta"]
        if m.get(col) is None or (
            head in ("target_group", "hate_type", "severity")
            and m.get("gate") is not True
        ):
            continue
        source_ids = m.get("astra_train_ids", [cid(r["id"])])
        result[r["id"]] = {"train_ids": source_ids, "record": r}
    return result


def eligible_sets(records, lang, head, seeds=(42, 43, 44)):
    chosen = record_sources(
        [r for r in records if r["kind"] == "example" and r["lang"] == lang], head
    )
    pool = [x["record"] for x in chosen.values()]
    assert len(pool) >= 5
    return [
        {
            "lang": lang,
            "head": head,
            "set_seed": s,
            "pool_size": len(pool),
            "records": random.Random(s).sample(pool, 5),
        }
        for s in seeds
    ]


def prepare():
    assert not OUT.exists() and (
        not (BASE / "kb").exists()
    ), "Preserve existing preparation; inspect instead of overwriting"
    encheck = (
        ROOT
        / "Thesis_Log/astra_result/evidence/english_kb_cleanup_check_20260914_v1.json"
    )
    ev = u.read_json(encheck)
    assert all((u.digest(ROOT / n) == h for n, h in ev["input_hashes"].items()))
    enreceipt = u.read_json(EN / "complete.json")
    deplan = u.read_json(DE / "plan.json")
    assert all((u.digest(ROOT / n) == h for n, h in enreceipt["outputs"].items()))
    depaths = [
        DE / f"kb/{pool}/records.jsonl"
        for pool in ("canonical_small", "canonical_full")
    ]
    for path in depaths:
        assert u.digest(path) == deplan["generated_hashes"][u.rel(path)]
    en = {
        size: u.read_lines(EN / f"{size}_english_records.jsonl")
        for size in ("small", "full")
    }
    derows = {
        size: u.read_lines(DE / f"kb/canonical_{size}/records.jsonl")
        for size in ("small", "full")
    }
    de = {
        size: [r for r in rr if r["kind"] == "example" and r["lang"] == "de"]
        for size, rr in derows.items()
    }
    abstracts = [r for r in derows["small"] if r["kind"] != "example"]
    assert (
        abstracts == [r for r in derows["full"] if r["kind"] != "example"]
        and len(abstracts) == 40
    )
    original = u.read_lines(ROOT / "kb/records.jsonl")
    assert abstracts == [r for r in original if r["kind"] != "example"]
    combined = {
        size: combine(en[size], de[size], abstracts) for size in ("small", "full")
    }
    fby = {r["id"]: r for r in combined["full"]}
    assert all((fby[r["id"]] == r for r in combined["small"]))
    assert len(combined["full"]) == 71131 and len(combined["small"]) == 340
    enex = set(u.read_json(EN / "policy.json")["final_excluded_ids"])
    deex = set(u.read_json(DE / "policy.json")["final_excluded_ids"])
    paths = [
        Path(__file__),
        ROOT / "tests/knowledge/banks/test_design.py",
        ROOT / "analysis/knowledge/banks/build.py",
        ROOT / "analysis/common/prompt.py",
        encheck,
        EN / "complete.json",
        EN / "policy.json",
        EN / "full_english_records.jsonl",
        EN / "small_english_records.jsonl",
        DE / "plan.json",
        DE / "policy.json",
        DE / "record_provenance.json",
        *depaths,
        ROOT / "kb/records.jsonl",
        ROOT / "config/config.yaml",
        ROOT / "config/taxonomy.yaml",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "training/encoder.py",
        ROOT / "scripts/make_encoder_data.py",
        ROOT / "training/predict.py",
        ROOT / "src/hsrag/prompt.py",
        ROOT / "src/hsrag/classify.py",
        ROOT / "src/hsrag/metrics.py",
        ROOT / "src/hsrag/schema.py",
    ]
    data = {}
    encoder_jobs = []
    weight_reuse = []
    meta_evidence = {}
    for arm, heads in HEADS.items():
        for head in heads:
            olddir = ROOT / f"data/encoder/{arm}"
            meta_path = olddir / f"{head}_meta.json"
            paths.append(meta_path)
            meta = u.read_json(meta_path)
            ex = deex if arm == "de" else enex
            col = GOLD[head]
            frames = {}
            removed = {}
            source_keys = {}
            for split in ("train", "val"):
                path = olddir / f"{head}_{split}.parquet"
                paths.append(path)
                old = pd.read_parquet(path)
                frames[split], removed[split] = filtered(old, ex)
                source_keys[split] = [cid(x) for x in frames[split].id]
            assert not set(source_keys["train"]) & set(source_keys["val"])
            pool = de["full"] if arm == "de" else en["small" if arm == "kb" else "full"]
            expected = record_sources(pool, head)
            union = set(source_keys["train"] + source_keys["val"])
            raw_by_id = {
                cid(row["id"]): row
                for split in frames.values()
                for row in split.to_dict("records")
            }
            mapped = set()
            for record_id, item in expected.items():
                eligible = set(item["train_ids"]) & union
                assert (
                    len(eligible) == 1
                ), f"Expected one original head annotation source: {arm}/{head}/{record_id}"
                rid = next(iter(eligible))
                mapped.add(rid)
                oldrow = raw_by_id[rid]
                rec = item["record"]
                assert u.canonical_text(oldrow["text"]) == u.canonical_text(rec["text"])
                gold = u.clean(oldrow[col])
                want = rec["meta"][col]
                assert (sorted(gold) if isinstance(gold, list) else gold) == (
                    sorted(want) if isinstance(want, list) else want
                )
            assert (
                mapped == union
            ), f"Encoder pool does not match KB annotations: {arm}/{head}"
            assert all(
                (
                    u.clean(x) is not None
                    for frame in frames.values()
                    for x in frame[col]
                )
            )
            newdir = BASE / f"encoder_data/{arm}"
            newdir.mkdir(parents=True, exist_ok=True)
            for split, df in frames.items():
                target = own(newdir / f"{head}_{split}.parquet")
                if target.exists():
                    assert (
                        pd.read_parquet(target)
                        .reset_index(drop=True)
                        .equals(df.reset_index(drop=True))
                    ), "Partial preparation data differs"
                    if not removed[split]:
                        assert u.digest(target) == u.digest(
                            olddir / f"{head}_{split}.parquet"
                        )
                elif not removed[split]:
                    shutil.copyfile(olddir / f"{head}_{split}.parquet", target)
                else:
                    df.to_parquet(target, index=False)
            newmeta = {
                "original_meta": meta,
                "arm": arm,
                "head": head,
                "kind": meta["kind"],
                "lang": meta["lang"],
                "labels": meta["labels"],
                "counts": {k: len(v) for k, v in frames.items()},
                "removed_ids": removed,
                "source_files": {
                    k: u.rel(olddir / f"{head}_{k}.parquet") for k in ("train", "val")
                },
                "policy_path": u.rel((DE if arm == "de" else EN) / "policy.json"),
                "split_policy": "Filter original TRAIN-internal train/val partitions, preserve surviving row order and partition; do not resplit or use DEV for selection.",
                "annotation_alignment": "One eligible source annotation per canonical head text; both partitions combined exactly cover the corresponding annotated KB examples.",
            }
            put(newdir / f"{head}_meta.json", newmeta)
            data[f"{arm}/{head}"] = newmeta
            mlist = []
            for seed in (0, 1, 2):
                mp = ROOT / f"models/encoder/{arm}/{head}/seed{seed}/meta.json"
                paths.append(mp)
                m = u.read_json(mp)
                mlist.append(m)
                assert (
                    m["labels"] == meta["labels"]
                    and m["seed"] == seed
                    and (m["head"] == head)
                )
            meta_evidence[f"{arm}/{head}"] = [
                {
                    k: m.get(k)
                    for k in (
                        "seed",
                        "epochs",
                        "early_stopping",
                        "precision",
                        "n_train",
                        "n_val",
                        "train_script_sha",
                        "base_model",
                        "train_seconds",
                    )
                }
                for m in mlist
            ]
            reuse = arm == "kb" and (not removed["train"])
            if reuse:
                assert all(
                    (m["early_stopping"] is False and m["epochs"] == 20 for m in mlist)
                )
                for seed in (0, 1, 2):
                    checkpoint = ROOT / f"models/encoder/{arm}/{head}/seed{seed}"
                    files = sorted((x for x in checkpoint.iterdir() if x.is_file()))
                    assert any(
                        (
                            x.name in ("model.safetensors", "pytorch_model.bin")
                            for x in files
                        )
                    )
                    paths.extend(files)
                    weight_reuse.append(
                        {
                            "arm": arm,
                            "head": head,
                            "seed": seed,
                            "path": u.rel(checkpoint),
                            "reason": "Gradient TRAIN unchanged; fixed 20 epochs, no early stopping/best-validation selection. For compact types, one reporting-only validation row removed. Preserve original training metadata.",
                            "limits": "Historical metadata supports fixed-epoch recipe; current trainer SHA differs from recorded old SHA. No claim of byte-identical historical script reconstruction. Regenerate complete predictions with new gate.",
                        }
                    )
            else:
                for seed in (0, 1, 2):
                    encoder_jobs.append(
                        {
                            "arm": arm,
                            "head": head,
                            "seed": seed,
                            "dataset": u.rel(newdir),
                            "base_model": (
                                "deepset/gbert-base" if arm == "de" else "roberta-base"
                            ),
                            "epochs": 20 if arm == "kb" else 5,
                            "early_stopping_patience": None if arm == "kb" else 2,
                            "max_seq_length": 128,
                            "batch_size": 16,
                            "eval_batch_size": 32,
                            "learning_rate": 2e-05,
                            "warmup_ratio": 0.1,
                            "weight_decay": 0.01,
                            "threshold": 0.5,
                            "precision": "fp16",
                            "class_weighting": False,
                            "resampling": False,
                            "selection": (
                                "last epoch"
                                if arm == "kb"
                                else "best internal-val macro-F1"
                            ),
                        }
                    )
    assert len(encoder_jobs) == 21 and len(weight_reuse) == 9
    inputs = {}
    scope_counts = {}
    for name, dimension in SCOPES.items():
        path = ROOT / f"data/processed/{name}.parquet"
        paths.append(path)
        df = pd.read_parquet(path, columns=["id", "text", "lang"])
        scope_counts[name] = len(df)
        assert df.id.nunique() == len(df)
        for row in df.itertuples(index=False):
            key = f"{row.lang}/{row.id}"
            if key not in inputs:
                inputs[key] = {
                    "key": key,
                    "item_id": str(row.id),
                    "lang": row.lang,
                    "text": row.text,
                    "scopes": [],
                }
            assert inputs[key]["text"] == row.text
            inputs[key]["scopes"].append({"subset": name, "dimension": dimension})
    input_rows = list(inputs.values())
    fs = fixed_sets(combined["full"])
    eligible = []
    eligible_input_keys = {}
    for lang, head in [
        ("en", "target_group"),
        ("en", "hate_type"),
        ("en", "severity"),
        ("de", "hate"),
        ("de", "legal"),
    ]:
        entries = eligible_sets(combined["full"], lang, head)
        eligible += entries
        selected = [
            r["key"]
            for r in input_rows
            if r["lang"] == lang
            and any(
                (
                    x["dimension"].removesuffix("_primary").removesuffix("_secondary")
                    == head
                    for x in r["scopes"]
                )
            )
        ]
        assert selected
        eligible_input_keys[f"{lang}/{head}"] = selected
    rows_by_lang = Counter((r["lang"] for r in input_rows))
    conditions = [
        {"name": "rag_full", "arm": "rag", "pool": "full", "repeats": 3},
        {"name": "rag_small", "arm": "rag", "pool": "small", "repeats": 3},
        {"name": "zero_shot", "arm": "zero_shot", "pool": None, "repeats": 3},
        *[
            {
                "name": f"fewshot_full_s{s}",
                "arm": "few_shot",
                "pool": "full",
                "set_seed": s,
                "repeats": 3,
            }
            for s in (42, 43, 44)
        ],
    ]
    assert sum((x["repeats"] for x in conditions)) == 18
    static = []
    for row in input_rows:
        contexts = [
            ("zero_shot", {"definitions": [], "guidelines": [], "example_groups": []})
        ]
        for x in fs:
            if x["lang"] == row["lang"]:
                contexts.append(
                    (
                        f"fewshot_full_s{x['set_seed']}",
                        {
                            "definitions": [],
                            "guidelines": [],
                            "example_groups": [
                                ["Labelled examples", [hit(r) for r in x["records"]]]
                            ],
                        },
                    )
                )
        for x in eligible:
            if row["key"] in eligible_input_keys[f"{x['lang']}/{x['head']}"]:
                contexts.append(
                    (
                        f"fewshot_eligible_{x['head']}_s{x['set_seed']}",
                        {
                            "definitions": [],
                            "guidelines": [],
                            "example_groups": [
                                ["Labelled examples", [hit(r) for r in x["records"]]]
                            ],
                        },
                    )
                )
        for name, ctx in contexts:
            messages = prompt.render_messages(
                row["text"], row["lang"], ctx, "corrected"
            )
            check_messages(messages, row["text"])
            static.append(
                {
                    "key": row["key"],
                    "condition": name,
                    "messages": messages,
                    "messages_sha256": hashlib.sha256(
                        u.json_bytes(messages)
                    ).hexdigest(),
                    "context": ctx,
                }
            )
    caches = {}
    for model in ("roberta-base", "deepset/gbert-base"):
        repo = Path(
            os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or (
                Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface")
                / "hub"
            )
        ) / ("models--" + model.replace("/", "--"))
        rev = (repo / "refs/main").read_text().strip()
        snap = repo / "snapshots" / rev
        files = sorted((x for x in snap.iterdir() if x.is_file()))
        assert any((x.suffix in (".safetensors", ".bin") for x in files))
        caches[model] = {
            "revision": rev,
            "snapshot": str(snap),
            "files": {x.name: u.digest(x) for x in files},
        }
    paths = sorted(set(paths))
    hashes = {u.rel(x): u.digest(x) for x in paths}
    for size, records in combined.items():
        jsonl(BASE / f"kb/{size}/records.jsonl", records)
    jsonl(BASE / "dev_inputs.jsonl", input_rows)
    jsonl(BASE / "static_prompts.jsonl", static)
    put(BASE / "fewshot_sets.json", fs)
    put(BASE / "fewshot_eligible_sets.json", eligible)
    put(
        BASE / "encoder_plan.json",
        {
            "train_jobs": encoder_jobs,
            "weight_reuse": weight_reuse,
            "historical_meta": meta_evidence,
            "split_summaries": data,
            "mmbert": "Completed historical backbone sensitivity retained; no new mmBERT run. Clean GBERT is the selected German baseline.",
            "parked": "No-grievance taxonomy adaptation is not included; Priority 3 remains parked.",
            "runner_status": "No new training runner or CUDA validation yet. Do not execute original trainer against these paths.",
        },
    )
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": BASE.name,
        "status": "cpu_protocol_preparation_complete_execution_not_ready",
        "pool_counts": {
            size: dict(
                total=len(rr),
                english=sum((r["kind"] == "example" and r["lang"] == "en" for r in rr)),
                german=sum((r["kind"] == "example" and r["lang"] == "de" for r in rr)),
                abstracts=sum((r["kind"] != "example" for r in rr)),
            )
            for size, rr in combined.items()
        },
        "scope_counts": scope_counts,
        "distinct_dev_inputs": len(input_rows),
        "distinct_by_language": dict(rows_by_lang),
        "conditions": conditions,
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "provider": "peasec",
        "temperature": 1.0,
        "votes_per_item": 3,
        "eligible_fewshot_conditions": {
            "sets": u.rel(BASE / "fewshot_eligible_sets.json"),
            "input_keys": eligible_input_keys,
            "set_seeds": [42, 43, 44],
            "repeats_per_set": 3,
            "english_gate_alias": "identical to uniform full pool; do not duplicate requests",
        },
        "retrieval": (
            deplan["manifests"][next(iter(deplan["manifests"]))]["retrieval"]
            if "manifests" in deplan
            else u.read_json(DE / "plan.json").get("retrieval")
        ),
        "core_design": {
            "primary_endpoints": [
                "English strict hate gate",
                "German legal three-label macro-F1",
            ],
            "secondary_endpoints": [
                "English primary types (467)",
                "English targets",
                "English severity",
                "German gate",
            ],
            "sensitivity_endpoints": ["English secondary types (103)"],
            "pairing": "Same saved prediction per language/item/condition/repeat reused across overlapping endpoint rosters, never new independent observations.",
            "fewshot": "Five uniform examples from full language pool, set seeds 42/43/44, each repeated three times. No conditioning on gold or post-draw rebalancing. Report annotation coverage.",
            "strong_fewshot_control": "Also sample five uniformly from the full pool annotated for the requested head (fine English heads restricted to TRAIN gate=True, as encoder masks). Use the same three predetermined set seeds and three decoder repeats. Retain annotated empty legal lists. No current input gold/text similarity used in selection. This is a separately named task-aware control; uniform draws remain reported.",
            "planned_contrasts": "Full-minus-small RAG, and full RAG versus annotation-eligible few-shot on each endpoint. Uniform few-shot and zero-shot remain context/control comparisons. Report FS between-set and within-set variation separately. Core RAG/FS is a package contrast, not isolated example selection.",
            "interpretation": "RAG versus FS compares the RAG package: selected examples, definitions/guidelines and different source caution. It is not a pure causal retrieval-only effect.",
            "missing_gold": "Per-dimension masks preserve None versus []; legal independent of hate gate. Freeze scoring/failure rules in execution preparation.",
            "reuse": "New RAG/ZS/FS execution planned for contemporaneous core comparisons. Historical results retained separately; German reuse needs saved-message/index equivalence before any later amendment.",
            "staging": "Budget is for the whole matrix, not one uninterrupted batch. Freeze resumable language/condition blocks and stop for review after each major batch.",
            "later": "HateCheck/TEST and claim-bearing SQ1 counterparts separately frozen later; Priority 3 and LaTeX excluded.",
        },
        "budget": {
            "base_condition_repeats": 18,
            "base_item_predictions": len(input_rows) * 18,
            "eligible_additional_item_predictions": sum(
                map(len, eligible_input_keys.values())
            )
            * 9,
            "item_predictions": len(input_rows) * 18
            + sum(map(len, eligible_input_keys.values())) * 9,
            "ordinary_model_votes": (
                len(input_rows) * 18 + sum(map(len, eligible_input_keys.values())) * 9
            )
            * 3,
            "maximum_logical_votes_with_two_repairs_each": (
                len(input_rows) * 18 + sum(map(len, eligible_input_keys.values())) * 9
            )
            * 9,
            "transport_retries": "Additional; separate accounting/caps to be frozen in runner.",
            "encoder_train_jobs": 21,
            "encoder_weight_reuse": 9,
        },
        "static_prompt_count": len(static),
        "fewshot_coverage": [
            {"lang": x["lang"], "seed": x["set_seed"], "coverage": x["coverage"]}
            for x in fs
        ],
        "base_caches": caches,
        "environment": u.environment(),
        "input_hashes": hashes,
        "preservation": "Only this new directory and new helpers/docs changed. No active KB/index, source trainer/client, checkpoint or original data/score changes.",
        "pending": [
            "Separate robust training runner, pinned model loading, CUDA smoke/reload checks and one-command user launcher",
            "Separate combined-index setup, RAG prompt checks and partial-label control coverage review",
            "Concurrent-safe API logging, response/failure/scoring contracts and bounded execution launcher",
        ],
    }
    if protocol["retrieval"] is None:
        import yaml

        protocol["retrieval"] = yaml.safe_load(
            (ROOT / "config/config.yaml").read_text(encoding="utf8")
        )["retrieval"]
    generated = {u.rel(x): u.digest(x) for x in sorted(BASE.rglob("*")) if x.is_file()}
    assert all(
        (u.digest(ROOT / n) == h for n, h in hashes.items())
    ), "Source changed during preparation"
    protocol["generated_hashes"] = generated
    put(OUT, protocol)
    print(
        json.dumps(
            {
                k: protocol[k]
                for k in (
                    "status",
                    "pool_counts",
                    "scope_counts",
                    "distinct_dev_inputs",
                    "distinct_by_language",
                    "budget",
                    "fewshot_coverage",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
