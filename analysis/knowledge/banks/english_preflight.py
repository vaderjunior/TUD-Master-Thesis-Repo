from collections import Counter, defaultdict
import json
from pathlib import Path
import pandas as pd
from analysis.knowledge.banks import build as u

ROOT = u.ROOT
BASE = ROOT / "experiments/astra/full_kb_english_preflight_20260914_v1"


def main():
    out = BASE / "preflight.json"
    u.require(not out.exists(), "Preflight already exists; preserve evidence")
    train_path = ROOT / "data/processed/en_train.parquet"
    held_paths = sorted(
        (
            f
            for f in (ROOT / "data/processed").glob("en_*.parquet")
            if "train" not in f.stem
        )
    )
    previous = ROOT / "experiments/astra/sq1_kb_size_20260911_v1"
    files = [
        Path(__file__),
        ROOT / "analysis/knowledge/banks/build.py",
        train_path,
        *held_paths,
        ROOT / "experiments/leakage_report_en_train.json",
        ROOT / "experiments/leakage_train_exclusions.json",
        previous / "plan.json",
        previous / "preparation.json",
        previous / "kb/all/records.jsonl",
        previous / "kb/small/records.jsonl",
    ]
    hashes = {u.rel(f): u.digest(f) for f in files}
    oldplan = u.read_json(previous / "plan.json")
    for pool in ("all", "small"):
        key = u.rel(previous / f"kb/{pool}/records.jsonl")
        u.require(
            hashes[key] == oldplan["generated_hashes"][key],
            "Previous frozen KB changed",
        )
    train = pd.read_parquet(
        train_path, columns=["id", "text", "lang", "source"]
    ).to_dict("records")
    u.require(len({str(r["id"]) for r in train}) == len(train), "Duplicate TRAIN IDs")
    u.require(
        all(
            (
                r["lang"] == "en" and isinstance(r["text"], str) and u.norm(r["text"])
                for r in train
            )
        ),
        "Invalid TRAIN text/language",
    )
    byid = {str(r["id"]): r for r in train}
    held, texts, id_text, sizes, empty = ([], {}, {}, {}, [])
    for path in held_paths:
        df = pd.read_parquet(path, columns=["id", "text", "lang"])
        sizes[u.rel(path)] = len(df)
        u.require(df.id.nunique() == len(df), "Duplicate held-out IDs")
        for r in df.itertuples(index=False):
            rid = str(r.id)
            u.require(
                r.lang == "en" and isinstance(r.text, str),
                "Invalid held-out text/language",
            )
            u.require(
                rid not in id_text or id_text[rid] == r.text,
                "Held-out ID/text inconsistency",
            )
            id_text[rid] = r.text
            ref = {"id": rid, "file": u.rel(path)}
            if not r.text.strip():
                empty.append(ref)
                continue
            if r.text not in texts:
                texts[r.text] = len(held)
                held.append({"text": r.text, "references": []})
            held[texts[r.text]]["references"].append(ref)
    pool_ids = {}
    for pool in ("small", "all"):
        pool_ids[pool] = {
            r["id"].removeprefix("ex-")
            for r in u.read_lines(previous / f"kb/{pool}/records.jsonl")
            if r["kind"] == "example" and r["lang"] == "en"
        }
    exact = set(texts)
    norm = {u.norm(t) for t in exact}
    exact_ids = {r["id"] for r in train if r["text"] in exact}
    norm_ids = {r["id"] for r in train if u.norm(r["text"]) in norm}
    shared = set(byid) & set(id_text)
    groups = defaultdict(list)
    for r in train:
        groups[u.canonical_text(r["text"])].append(r["id"])
    duplicates = [ids for ids in groups.values() if len(ids) > 1]
    old_report = u.read_json(ROOT / "experiments/leakage_report_en_train.json")
    matches = []
    for subset, report in old_report["subsets"].items():
        for pair in report.get("top", []):
            if pair["cosine"] >= 0.95 and pair.get("train_id") in byid:
                rid = pair["train_id"]
                matches.append(
                    {
                        **pair,
                        "subset": subset,
                        "in_previous_full": rid in pool_ids["all"],
                        "in_previous_small": rid in pool_ids["small"],
                        "train_text": byid[rid]["text"],
                        "heldout_text": id_text.get(pair["eval_id"]),
                    }
                )
    retained = [r for r in matches if r["in_previous_full"]]
    known = []
    for rid in ("implicit_hate-6023", "implicit_hate-8907"):
        known.append(
            {
                "id": rid,
                "in_train": rid in byid,
                "in_previous_full": rid in pool_ids["all"],
                "in_previous_small": rid in pool_ids["small"],
                "normalised_overlap": rid in norm_ids,
                "recorded_matches": [r for r in matches if r["train_id"] == rid],
            }
        )
    summary = {
        "raw_english_train_rows": len(train),
        "train_sources": dict(Counter((r["source"] for r in train))),
        "heldout_file_counts": sizes,
        "heldout_unique_ids": len(id_text),
        "heldout_distinct_nonempty_texts": len(held),
        "heldout_empty_references": len(empty),
        "same_train_heldout_ids": len(shared),
        "exact_overlap_train_rows": len(exact_ids),
        "normalized_overlap_train_rows": len(norm_ids),
        "canonical_duplicate_train_groups": len(duplicates),
        "previous_small_examples": len(pool_ids["small"]),
        "previous_full_examples": len(pool_ids["all"]),
        "normalized_matches_still_in_previous_full": len(norm_ids & pool_ids["all"]),
        "retained_ids_with_recorded_cosine_at_least_095": len(
            {r["train_id"] for r in retained}
        ),
        "fresh_embedding_text_count": len(train) + len(held),
        "full_train_by_heldout_similarity_pairs": len(train) * len(held),
    }
    report = {
        "id": "ASTRA-FULL-KB-ENGLISH-PREFLIGHT-20260914-v1",
        "created_utc": u.now(),
        "summary": summary,
        "exact_overlap_ids": sorted(exact_ids),
        "normalized_overlap_ids": sorted(norm_ids),
        "shared_ids": sorted(shared),
        "canonical_duplicate_train_groups": duplicates,
        "empty_heldout_references": empty,
        "recorded_near_matches_retained_in_previous_full": retained,
        "known_hatecheck_cases": known,
        "previous_frozen_plan_sha256": u.digest(previous / "plan.json"),
        "input_hashes": hashes,
        "decision": "Proceed toward a freshly audited full-English TRAIN pool and combine with the audited German policy for a coherent primary system. Keep previous pools/results unchanged.",
        "limits": "No fresh semantic similarities computed. Old report has rounded scores and no input hashes, so these are recorded warnings, not revalidated cosine values. Exact-text checks do not establish semantic clearance. Final English exclusion set and size remain pending. Read held-out identity/text/language only; no gold scoring, GPU/model calls or active KB changes.",
    }
    u.require(
        all((u.digest(ROOT / k) == v for k, v in hashes.items())),
        "Input changed during preflight",
    )
    BASE.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train_texts.jsonl", train), ("heldout_texts.jsonl", held)):
        with (BASE / name).open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["generated_hashes"] = {
        u.rel(BASE / name): u.digest(BASE / name)
        for name in ("train_texts.jsonl", "heldout_texts.jsonl")
    }
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "path": u.rel(out),
                "sha256": u.digest(out),
                "summary": summary,
                "retained_recorded_matches": [
                    {k: r[k] for k in ("subset", "train_id", "eval_id", "cosine")}
                    for r in retained
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
