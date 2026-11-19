"""Banks english results."""

from src.hsrag.paths import PROJECT_ROOT
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/english_kb_audit_20260914_v1"
OLD = ROOT / "experiments/astra/sq1_kb_size_20260911_v1"
OUT = ROOT / "Thesis_Log/astra_result/evidence/english_kb_results_20260914_v1.json"


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def lines(p):
    return [json.loads(s) for s in Path(p).read_text(encoding="utf-8").splitlines() if s.strip()]


def js(obj):
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def norm(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub("\\s+", " ", text).strip().strip(" .,!?:;\"'")


def body_norm(text):
    return norm(re.sub("(?<![\\w@])@[A-Za-z0-9_]+", "@user", text))


def embeddings(plan, roster, tag):
    chunks, paths, max_norm_error = ([], [], 0.0)
    for start in range(0, len(roster), plan["cache_chunk"]):
        texts = [r["text"] for r in roster[start : start + plan["cache_chunk"]]]
        ident = dict(
            plan_sha256=digest(BASE / "plan.json"),
            model=plan["model"],
            revision=plan["embedding_cache"]["revision"],
            precision=plan["precision"],
            texts_sha256=hashlib.sha256(js(texts)).hexdigest(),
            rows=len(texts),
        )
        key = hashlib.sha256(js(ident)).hexdigest()[:16]
        p = BASE / f"embedding_chunks/{tag}_{start:06d}_{key}.npz"
        with np.load(p, allow_pickle=False) as data:
            assert set(data.files) == {"vectors", "metadata"}
            v = data["vectors"]
            meta = json.loads(str(data["metadata"].item()))
        assert meta.pop("vectors_sha256") == hashlib.sha256(v.tobytes()).hexdigest()
        assert meta == ident and v.dtype == np.float32 and (v.shape == (len(texts), 1024))
        assert np.isfinite(v).all()
        error = float(np.max(np.abs(np.linalg.norm(v, axis=1) - 1)))
        assert error < 0.0002
        max_norm_error = max(max_norm_error, error)
        chunks.append(v)
        paths.append(p)
    return (np.concatenate(chunks), paths, max_norm_error)


def main():
    started = time.perf_counter()
    assert not OUT.exists(), "Preserve existing review; version any amendment."
    plan, receipt, report, prep = [
        read(BASE / f"{n}.json") for n in ("plan", "complete", "report", "preparation")
    ]
    assert receipt["plan_sha256"] == report["plan_sha256"] == digest(BASE / "plan.json")
    hashes = {
        **plan["input_hashes"],
        **plan["generated_hashes"],
        **receipt["outputs"],
        **receipt["embedding_hashes"],
    }
    extras = [
        Path(__file__),
        BASE / "plan.json",
        BASE / "complete.json",
        *sorted((BASE / "logs").glob("*.log")),
        OLD / "retrieval_complete.json",
        OLD / "retrieval_small.json",
        OLD / "retrieval_all.json",
    ]
    hashes.update({rel(p): digest(p) for p in extras})
    assert all((digest(ROOT / p) == h for p, h in hashes.items())), "Input/receipt hash mismatch"
    train, held, rows, pairs = [
        lines(BASE / f"{n}.jsonl")
        for n in ("train_texts", "heldout_texts", "nearest_train", "review_pairs")
    ]
    df = pd.read_parquet(
        ROOT / "data/processed/en_train.parquet", columns=["id", "text", "lang", "source"]
    )
    assert train == df.to_dict("records") and set(df.lang) == {"en"}
    rebuilt = defaultdict(list)
    for name in plan["heldout_files"]:
        df = pd.read_parquet(ROOT / name, columns=["id", "text", "lang"])
        assert set(df.lang) == {"en"} and df.id.nunique() == len(df)
        for r in df.itertuples(index=False):
            assert r.text.strip()
            rebuilt[r.text].append({"id": str(r.id), "file": name})
    assert held == [{"text": t, "references": refs} for t, refs in rebuilt.items()]
    assert len(train) == len(rows) == len({r["id"] for r in train}) == 56190
    assert len(held) == len({r["text"] for r in held}) == 27810
    for t, r in zip(train, rows):
        assert (t["id"], t["source"]) == (r["train_id"], r["source"])
        assert r["nearest_heldout_references"] == held[r["heldout_text_index"]]["references"]
    a, ap, an = embeddings(plan, train, "train")
    b, bp, bn = embeddings(plan, held, "heldout")
    assert set(map(rel, ap + bp)) == set(receipt["embedding_hashes"])
    assert set(ap + bp) == set((BASE / "embedding_chunks").glob("*.npz"))
    assert not list(BASE.rglob("*.partial_*"))
    oldexcluded = set(prep["cheap_excluded_ids"])
    proposed = set(report["proposed_excluded_ids"])
    small = set(prep["previous_small_ids"])
    files = plan["heldout_files"]
    fileidx = {
        f: np.array(
            [i for i, h in enumerate(held) if any((ref["file"] == f for ref in h["references"]))]
        )
        for f in files
    }
    mx = np.empty(len(a), np.float32)
    ix = np.empty(len(a), np.int64)
    byfile = {f: np.empty(len(a), np.float32) for f in files}
    edges = []
    saved_ix = np.array([r["heldout_text_index"] for r in rows])
    maxgap = 0.0
    with threadpool_limits(limits=4):
        for start in range(0, len(a), 173):
            end = min(start + 173, len(a))
            sims = a[start:end] @ b.T
            arg = np.argmax(sims, axis=1)
            ix[start:end] = arg
            mx[start:end] = sims[np.arange(end - start), arg]
            maxgap = max(
                maxgap,
                float(np.max(mx[start:end] - sims[np.arange(end - start), saved_ix[start:end]])),
            )
            for f, idx in fileidx.items():
                byfile[f][start:end] = sims[:, idx].max(axis=1)
            ti, hi = np.where(sims >= 0.94)
            edges.extend(
                (
                    {
                        "train_index": int(start + t),
                        "heldout_text_index": int(h),
                        "cosine": float(sims[t, h]),
                    }
                    for t, h in zip(ti, hi)
                )
            )
            if start % (173 * 64) == 0 or end == len(a):
                print(f"CPU RECHECK {end}/{len(a)}", flush=True)
    saved = np.array([r["max_cosine"] for r in rows], np.float64)
    error = float(np.max(np.abs(mx - saved)))
    assert error < 2e-06 and maxgap < 2e-06
    assert all((np.array_equal(mx.astype(np.float64) >= t, saved >= t) for t in (0.94, 0.95, 0.98)))
    near = {r["id"] for r, s in zip(train, mx) if float(s) >= 0.95}
    assert near | oldexcluded == proposed
    assert all(
        (
            r["threshold_excluded"] == (r["train_id"] in near)
            and r["proposed_excluded"] == (r["train_id"] in proposed)
            for r in rows
        )
    )
    expected_pairs = [
        {**r, "train_text": t["text"], "heldout_text": held[r["heldout_text_index"]]["text"]}
        for t, r in zip(train, rows)
        if r["max_cosine"] >= 0.94 or r["proposed_excluded"]
    ]
    assert pairs == expected_pairs
    canonical = [
        re.sub("\\s+", " ", unicodedata.normalize("NFKC", r["text"]).casefold()).strip()
        for r in train
    ]
    assert len(set(canonical)) == len(train)
    kept = [r for r in train if r["id"] not in proposed]
    summary = dict(
        raw_train=len(train),
        previous_full=len(train) - len(oldexcluded),
        previous_small=len(small),
        threshold_095=len(near),
        threshold_098=int((mx >= 0.98).sum()),
        previous_excluded=len(oldexcluded),
        prior_excluded_below_095=len(oldexcluded - near),
        total_proposed_excluded=len(proposed),
        new_excluded=len(proposed - oldexcluded),
        remaining_full=len(kept),
        remaining_small=len(small - proposed),
        dropped_small_ids=sorted(small & proposed),
        remaining_sources=dict(Counter((r["source"] for r in kept))),
        excluded_sources=dict(Counter((r["source"] for r in train if r["id"] in proposed))),
        review_band_total=int(((mx >= 0.94) & (mx < 0.95)).sum()),
        review_band_retained=sum(
            (0.94 <= r["max_cosine"] < 0.95 and (not r["proposed_excluded"]) for r in rows)
        ),
    )
    for field, value in [
        ("near_train_rows", len(near)),
        ("very_near_train_rows", summary["threshold_098"]),
        ("provisional_remaining_train_rows", len(kept)),
        ("provisional_remaining_texts", len(kept)),
        ("provisional_remaining_previous_small", len(small - proposed)),
        ("remaining_sources", summary["remaining_sources"]),
        ("review_band_rows", summary["review_band_total"]),
        ("boundary_within_0005", int((np.abs(saved - 0.95) <= 0.0005).sum())),
    ]:
        assert report[field] == value, field
    assert sorted(proposed - oldexcluded) == report["new_excluded_ids"]
    coverage = {}
    for f, values in byfile.items():
        hit = {r["id"] for r, s in zip(train, values) if float(s) >= 0.95}
        assert hit <= proposed
        coverage[f] = dict(
            heldout_texts=len(fileidx[f]),
            near_train_ids=sorted(hit),
            near_train_count=len(hit),
            in_previous_full=sorted(hit - oldexcluded),
            in_previous_small=sorted(hit & small),
        )
    ti = {r["id"]: i for i, r in enumerate(train)}
    hi = {ref["id"]: i for i, h in enumerate(held) for ref in h["references"]}
    known = read(BASE / "known_pair_checks.json")
    for k in known:
        s = float(
            np.dot(a[ti[k["train_id"]]].astype(np.float64), b[hi[k["eval_id"]]].astype(np.float64))
        )
        assert abs(s - k["fresh_cosine"]) < 2e-06
        assert (
            k["train_text"] == train[ti[k["train_id"]]]["text"]
            and k["heldout_text"] == held[hi[k["eval_id"]]]["text"]
        )
    candidates = []
    body = defaultdict(list)
    for i, h in enumerate(held):
        body[body_norm(h["text"])].append(i)
    for t, r in zip(train, rows):
        if t["id"] not in proposed and body_norm(t["text"]) in body:
            candidates.append(
                {
                    **r,
                    "train_text": t["text"],
                    "matching_heldout_indices": body[body_norm(t["text"])],
                }
            )
    loginfo = []
    for p in sorted((BASE / "logs").glob("*.log")):
        s = p.read_text(encoding="utf-8")
        info = dict(
            file=rel(p),
            encoded=s.count("(encoded)"),
            reused=s.count("(verified cache)"),
            completed="ENGLISH OVERLAP AUDIT COMPLETE" in s,
            failed="Traceback (most recent call last)" in s or "STOPPED." in s,
        )
        assert info["completed"] and (not info["failed"])
        loginfo.append(info)
    assert (
        len(loginfo) == 1
        and loginfo[0]["encoded"] == len(ap + bp) == 657
        and (loginfo[0]["reused"] == 0)
    )
    oldreceipt = read(OLD / "retrieval_complete.json")
    historical = {}
    for pool in ("small", "all"):
        assert digest(OLD / f"retrieval_{pool}.json") == oldreceipt["hashes"][pool]
        contexts = read(OLD / f"retrieval_{pool}.json")
        bysubset = defaultdict(
            lambda: dict(
                inputs=0,
                contexts_with_new_exclusion=0,
                delivered_new_ids=set(),
                direct_095=[],
                direct_094=[],
            )
        )
        for key, ctx in contexts.items():
            subset, eid = key.split("/", 1)
            stat = bysubset[subset]
            stat["inputs"] += 1
            ids = [x.removeprefix("ex-") for x in ctx["Labelled examples"]]
            new = set(ids) & proposed - oldexcluded
            stat["contexts_with_new_exclusion"] += bool(new)
            stat["delivered_new_ids"].update(new)
            for rid in ids:
                score = float(np.dot(a[ti[rid]].astype(np.float64), b[hi[eid]].astype(np.float64)))
                item = dict(
                    query_id=eid,
                    train_id=rid,
                    cosine=score,
                    newly_excluded=rid in proposed - oldexcluded,
                )
                if score >= 0.95:
                    stat["direct_095"].append(item)
                elif score >= 0.94:
                    stat["direct_094"].append(item)
        historical[pool] = {
            k: {**v, "delivered_new_ids": sorted(v["delivered_new_ids"])}
            for k, v in bysubset.items()
        }
    retained_band = [
        {**r, "train_text": t["text"], "heldout_text": held[r["heldout_text_index"]]["text"]}
        for t, r in zip(train, rows)
        if 0.94 <= r["max_cosine"] < 0.95 and (not r["proposed_excluded"])
    ]
    result = dict(
        id="ASTRA-ENGLISH-OVERLAP-RESULTS-20260914-v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        pool_counts=summary,
        numerical_recheck=dict(
            max_cosine_difference=error,
            saved_neighbor_max_gap=maxgap,
            argmax_disagreements=int((ix != saved_ix).sum()),
            thresholds_094_095_098_equal=True,
            train_max_norm_error=an,
            heldout_max_norm_error=bn,
            compared_pairs=len(a) * len(b),
            block="173 TRAIN x all heldout; 4 BLAS threads",
        ),
        coverage=coverage,
        known_pairs=known,
        retained_review_band=retained_band,
        retained_username_candidates=candidates,
        all_pairs_ge_094=edges,
        historical_context_exposure=historical,
        logs=loginfo,
        audit_resources=report["resources"],
        completed_utc=receipt["completed_utc"],
        review_seconds=time.perf_counter() - started,
        input_hashes=hashes,
        limits="Saved-vector arithmetic independently verified; embeddings not regenerated. Full English text-only scope; no heldout gold/scoring, no final exclusions or KB edits. .95 is a screening policy, not proof all removed rows are literal duplicates or retained rows independent. Additional body/boundary review is diagnostic pending a declared policy.",
    )
    assert all((digest(ROOT / p) == h for p, h in hashes.items())), "Input changed during review"
    with OUT.open("xb") as f:
        f.write(js(result))
    print(
        json.dumps(
            {
                "output": rel(OUT),
                "sha256": digest(OUT),
                "counts": summary,
                "numeric": result["numerical_recheck"],
                "username_candidates": len(candidates),
                "all_pairs_ge_094": len(edges),
                "historical": historical,
                "seconds": result["review_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
