"""Banks german results."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from analysis.knowledge.banks import german_audit as audit

ROOT = audit.ROOT
BASE = audit.BASE
OUT = ROOT / "Thesis_Log/astra_result/evidence/german_kb_results_20260913_v1.json"


def username_norm(text):
    return audit.util.norm(re.sub("(?<![\\w@])@[A-Za-z0-9_]+", "@user", text))


def count_annotations(rows):
    return {
        "rows": len(rows),
        "sources": dict(Counter((r["source"] for r in rows))),
        "gate": dict(Counter((str(r["meta"].get("gate")) for r in rows))),
        "annotated": {
            field: sum((r["meta"].get(field) is not None for r in rows))
            for field in ("gate", "target_groups", "hate_types", "severity", "legal")
        },
        "positive_legal": sum((bool(r["meta"].get("legal")) for r in rows)),
        "illustrations": sum((bool(r["meta"].get("illustrative_only")) for r in rows)),
    }


def read_embeddings(plan, rows, tag):
    chunks, expected_paths = ([], [])
    for start in range(0, len(rows), plan["cache_chunk"]):
        sub = rows[start : start + plan["cache_chunk"]]
        identity = audit.chunk_identity(plan, [r["text"] for r in sub])
        key = hashlib.sha256(audit.js(identity)).hexdigest()
        path = BASE / "embedding_chunks" / f"{tag}_{start:06d}_{key[:16]}.npz"
        chunks.append(audit.load_chunk(path, identity))
        expected_paths.append(path)
    result = np.concatenate(chunks)
    assert result.shape == (len(rows), 1024)
    return (result, expected_paths)


def verify_text_inputs(train, held):
    reconstructed_train = []
    for name in audit.TRAIN_NAMES:
        df = pd.read_parquet(
            ROOT / f"data/processed/{name}.parquet", columns=["id", "source", "text", "lang"]
        )
        assert set(df.lang) == {"de"}
        reconstructed_train += [
            {"id": str(r.id), "source": r.source, "split": name, "text": r.text}
            for r in df.itertuples(index=False)
        ]
    assert train == reconstructed_train
    collected, empties = (defaultdict(list), [])
    for path in audit.heldout_paths():
        df = pd.read_parquet(path, columns=["id", "text", "lang"])
        assert set(df.lang) == {"de"}
        for r in df.itertuples(index=False):
            ref = {"id": str(r.id), "file": audit.util.rel(path)}
            if r.text.strip():
                collected[r.text].append(ref)
            else:
                empties.append(ref)
    assert held == [{"text": text, "references": refs} for text, refs in collected.items()]
    assert len({r["id"] for r in train}) == len(train)
    assert len({r["text"] for r in held}) == len(held)
    return empties


def recompute(a, b, rows, held):
    files = sorted({ref["file"] for h in held for ref in h["references"]})
    indices = {
        f: np.asarray(
            [i for i, h in enumerate(held) if any((r["file"] == f for r in h["references"]))]
        )
        for f in files
    }
    maxima = np.empty(len(a), dtype=np.float32)
    args = np.empty(len(a), dtype=np.int64)
    by_file = {f: np.empty(len(a), dtype=np.float32) for f in files}
    held_max = np.full(len(b), -np.inf, dtype=np.float32)
    retained_max = np.full(len(b), -np.inf, dtype=np.float32)
    retained = np.asarray([not r["proposed_excluded"] for r in rows], dtype=bool)
    saved_args = np.asarray([r["heldout_text_index"] for r in rows])
    saved_neighbor_delta = 0.0
    for start in range(0, len(a), 192):
        end = min(start + 192, len(a))
        sim = a[start:end] @ b.T
        arg = sim.argmax(axis=1)
        maxima[start:end] = sim[np.arange(end - start), arg]
        args[start:end] = arg
        selected = sim[np.arange(end - start), saved_args[start:end]]
        saved_neighbor_delta = max(
            saved_neighbor_delta, float(np.max(np.abs(selected - maxima[start:end])))
        )
        for f, idx in indices.items():
            by_file[f][start:end] = sim[:, idx].max(axis=1)
        held_max = np.maximum(held_max, sim.max(axis=0))
        mask = retained[start:end]
        if mask.any():
            retained_max = np.maximum(retained_max, sim[mask].max(axis=0))
        if start % 1536 == 0 or end == len(a):
            print(f"OFFLINE VECTOR RECHECK: {end}/{len(a)} TRAIN rows", flush=True)
    saved_max = np.asarray([r["max_cosine"] for r in rows], dtype=np.float64)
    delta = float(np.max(np.abs(maxima - saved_max)))
    assert delta < 2e-06 and saved_neighbor_delta < 2e-06
    assert np.array_equal(maxima >= 0.95, saved_max >= 0.95)
    assert np.array_equal(maxima >= 0.98, saved_max >= 0.98)
    return (
        maxima,
        args,
        by_file,
        indices,
        held_max,
        retained_max,
        {
            "max_absolute_cosine_difference": delta,
            "saved_neighbor_max_gap": saved_neighbor_delta,
            "argmax_index_disagreements": int((args != saved_args).sum()),
            "threshold_095_decisions_equal": True,
            "threshold_098_decisions_equal": True,
        },
    )


def main():
    started = time.perf_counter()
    assert not OUT.exists(), "Evidence exists; preserve it instead of overwriting"
    plan = audit.validate()
    assert audit.check_complete()
    report = audit.util.read_json(BASE / "report.json")
    prep = audit.util.read_json(BASE / "preparation.json")
    receipt = audit.util.read_json(BASE / "complete.json")
    train = audit.util.read_lines(BASE / "train_texts.jsonl")
    held = audit.util.read_lines(BASE / "heldout_texts.jsonl")
    rows = audit.util.read_lines(BASE / "nearest_train.jsonl")
    pairs = audit.util.read_lines(BASE / "review_pairs.jsonl")
    paths = [ROOT / p for p in plan["input_hashes"]] + [ROOT / p for p in plan["generated_hashes"]]
    paths += [ROOT / p for k in ("outputs", "embedding_hashes") for p in receipt[k]]
    logs = sorted((BASE / "logs").glob("*.log"))
    paths += [audit.PLAN, BASE / "complete.json", Path(__file__)] + logs
    hashes = {audit.util.rel(p): audit.util.digest(p) for p in sorted(set(paths))}
    assert report["plan_sha256"] == audit.util.digest(audit.PLAN)
    empty = verify_text_inputs(train, held)
    assert empty == prep["empty_heldout_references_not_embedded"]
    assert len(rows) == len(train) and len({r["train_id"] for r in rows}) == len(train)
    for tr, row in zip(train, rows):
        assert (tr["id"], tr["split"]) == (row["train_id"], row["train_split"])
        assert row["nearest_heldout_references"] == held[row["heldout_text_index"]]["references"]
    a, ap = read_embeddings(plan, train, "train")
    b, bp = read_embeddings(plan, held, "heldout")
    assert set(map(audit.util.rel, ap + bp)) == set(receipt["embedding_hashes"])
    assert set(ap + bp) == set((BASE / "embedding_chunks").glob("*.npz"))
    assert not list(BASE.rglob("*.partial_*"))
    maxima, args, by_file, indices, hm, rm, numeric = recompute(a, b, rows, held)
    near = {r["id"] for r, s in zip(train, maxima) if s >= 0.95}
    cheap = set(prep["cheap_excluded_train_ids"])
    historical = set(prep["historical_excluded_train_ids"])
    excluded = near | cheap
    excluded_texts = {audit.util.canonical_text(r["text"]) for r in train if r["id"] in excluded}
    excluded |= {r["id"] for r in train if audit.util.canonical_text(r["text"]) in excluded_texts}
    assert sorted(excluded) == report["proposed_excluded_ids"]
    assert all(
        (
            r["threshold_excluded"] == (r["train_id"] in near)
            and r["proposed_excluded"] == (r["train_id"] in excluded)
            for r in rows
        )
    )
    wanted_pairs = []
    for tr, row in zip(train, rows):
        if row["max_cosine"] >= 0.9495 or row["proposed_excluded"]:
            wanted_pairs.append(
                {
                    **row,
                    "train_text": tr["text"],
                    "heldout_text": held[row["heldout_text_index"]]["text"],
                }
            )
    assert pairs == wanted_pairs
    retained = [r for r in train if r["id"] not in excluded]
    retain_mask = np.asarray([r["id"] not in excluded for r in train])
    new_mask = np.asarray([r["id"] in excluded - cheap for r in train])
    relevant = np.asarray([any(("_eval" in ref["file"] for ref in h["references"])) for h in held])
    coverage = {}
    for f, mx in by_file.items():
        subset_near = {r["id"] for r, s in zip(train, mx) if s >= 0.95}
        coverage[f] = {
            "embedded_texts": len(indices[f]),
            "near_train_rows": len(subset_near),
            "new_beyond_cheap_near_train_rows": int(((mx >= 0.95) & new_mask).sum()),
            "retained_near_train_rows": int(((mx >= 0.95) & retain_mask).sum()),
            "retained_max_cosine": float(mx[retain_mask].max()),
            "near_heldout_texts_before_filter": int((hm[indices[f]] >= 0.95).sum()),
            "near_heldout_texts_after_filter": int((rm[indices[f]] >= 0.95).sum()),
            "near_train_ids": sorted(subset_near),
        }
    assert all((v["retained_near_train_rows"] == 0 for v in coverage.values()))
    kept = [
        r
        for r in audit.util.read_lines(ROOT / "kb/records.jsonl")
        if r["kind"] == "example" and r["lang"] == "de"
    ]
    dropped_small = [r for r in kept if audit.canonical_id(r["id"]) in excluded]
    small = [r for r in kept if r not in dropped_small]
    raw_gold = {}
    for name in audit.TRAIN_NAMES:
        for r in pd.read_parquet(ROOT / f"data/processed/{name}.parquet").to_dict("records"):
            raw_gold[r["id"]] = {
                "source": r["source"],
                "meta": {
                    k: audit.util.clean(r.get(k))
                    for k in ("gate", "target_groups", "hate_types", "severity", "legal")
                },
            }
    pool_counts = {
        "raw_train": len(train),
        "excluded": len(excluded),
        "excluded_percent": 100 * len(excluded) / len(train),
        "near_095": len(near),
        "near_098": int((maxima >= 0.98).sum()),
        "additional_beyond_cheap": len(excluded - cheap),
        "additional_beyond_historical": len(excluded - historical),
        "remaining_train_rows": len(retained),
        "remaining_unique_canonical_texts": len(
            {audit.util.canonical_text(r["text"]) for r in retained}
        ),
        "remaining_small": len(small),
        "dropped_small": len(dropped_small),
        "excluded_sources": dict(Counter((r["source"] for r in train if r["id"] in excluded))),
        "retained_sources": dict(Counter((r["source"] for r in retained))),
        "small_annotations": count_annotations(small),
        "full_row_annotations_before_merging": count_annotations(
            [raw_gold[r["id"]] for r in retained]
        ),
        "dropped_small_ids": [r["id"] for r in dropped_small],
    }
    assert pool_counts["remaining_train_rows"] == report["remaining_train_rows"]
    assert (
        pool_counts["remaining_unique_canonical_texts"]
        == report["remaining_unique_canonical_texts"]
    )
    assert pool_counts["remaining_small"] == report["remaining_small_examples"]
    assert pool_counts["retained_sources"] == report["remaining_sources"]
    assert pool_counts["near_095"] == report["near_train_rows"]
    assert pool_counts["near_098"] == report["very_near_train_rows"]
    assert pool_counts["additional_beyond_cheap"] == report["new_exclusions_beyond_cheap"]
    duplicates = [g for g in prep["duplicate_groups"] if any((i not in excluded for i in g["ids"]))]
    assert all((all((i not in excluded for i in g["ids"])) for g in duplicates))
    held_user = defaultdict(list)
    for i, h in enumerate(held):
        held_user[username_norm(h["text"])].append(i)
    alias_candidates = [
        {
            "train_id": r["id"],
            "heldout_text_indices": held_user[username_norm(r["text"])],
            "max_cosine": row["max_cosine"],
        }
        for r, row in zip(train, rows)
        if r["id"] not in excluded and username_norm(r["text"]) in held_user
    ]
    near_retained = [r for r in rows if not r["proposed_excluded"] and r["max_cosine"] >= 0.94]
    near_retained.sort(key=lambda r: -r["max_cosine"])
    log_summary = []
    for path in logs:
        text = path.read_text(encoding="utf-8")
        log_summary.append(
            {
                "file": audit.util.rel(path),
                "encoded_chunks": len(re.findall("\\(encoded\\)", text)),
                "reused_chunks": len(re.findall("\\(verified cache\\)", text)),
                "success_marker": "GERMAN OVERLAP AUDIT COMPLETE" in text,
                "failure_marker": "Traceback (most recent call last)" in text or "STOPPED." in text,
            }
        )
    assert (
        len(logs) == 1
        and log_summary[0]["success_marker"]
        and (not log_summary[0]["failure_marker"])
    )
    assert log_summary[0]["encoded_chunks"] == len(ap + bp) and log_summary[0]["reused_chunks"] == 0
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Completed German overlap audit; offline verification only",
        "numerical_recheck": numeric,
        "pool_counts": pool_counts,
        "heldout_coverage": coverage,
        "empty_heldout_references": empty,
        "remaining_duplicate_groups": duplicates,
        "borderline_rows": [r for r in rows if abs(r["max_cosine"] - 0.95) <= 0.0005],
        "retained_094_to_095": near_retained,
        "retained_username_normalized_candidates": alias_candidates,
        "retained_eval_texts_with_near_train_before": int((hm[relevant] >= 0.95).sum()),
        "retained_eval_texts_with_near_train_after": int((rm[relevant] >= 0.95).sum()),
        "logs": log_summary,
        "embedding_chunks": len(ap + bp),
        "train_vectors": len(a),
        "heldout_vectors": len(b),
        "audit_resources": report["resources"],
        "audit_report_minutes": report["resources"]["attempt_seconds_before_report"] / 60,
        "verification_seconds": time.perf_counter() - started,
        "limitations": "Frozen threshold decisions verified from saved vectors, not fresh model embeddings. No accuracy inference; no scores or gold labels read for held-out data. Text overlap filtering uses held-out texts and does not create untouched evaluation. Retained near-boundary and username-normalized pairs need a separately documented policy/review before final KB construction; the original 167 exclusions are not changed here. Threshold pass is not proof of no semantic overlap. Original renderer remains unfixed.",
        "input_hashes": hashes,
    }
    assert all((audit.util.digest(ROOT / p) == h for p, h in hashes.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("xb") as f:
        f.write(audit.js(result))
    printable = {
        "pool_counts": pool_counts,
        "numerical_recheck": numeric,
        "coverage": {
            f: {k: v for k, v in d.items() if k != "near_train_ids"} for f, d in coverage.items()
        },
        "remaining_duplicate_groups": len(duplicates),
        "retained_094_to_095": len(near_retained),
        "retained_username_candidates": alias_candidates,
        "embedding_chunks": len(ap + bp),
        "logs": log_summary,
        "audit_minutes": result["audit_report_minutes"],
        "verification_seconds": result["verification_seconds"],
        "verified_input_hashes": len(hashes),
        "evidence": audit.util.rel(OUT),
        "sha256": audit.util.digest(OUT),
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
