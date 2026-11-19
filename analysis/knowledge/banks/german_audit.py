"""Banks german audit."""

from __future__ import annotations
from src.hsrag.paths import PROJECT_ROOT
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from analysis.knowledge.banks import build as util

ROOT = PROJECT_ROOT
RUN_ID = "german_kb_audit_20260913_v1"
BASE = ROOT / "experiments/astra" / RUN_ID
PLAN = BASE / "plan.json"
TRAIN_NAMES = ("de_train", "de_legal_train")
SELF_FILES = (
    "analysis/knowledge/banks/german_audit.py",
    "tests/knowledge/banks/test_german_audit.py",
    "scripts/validate_german_bank.ps1",
    "analysis/knowledge/banks/build.py",
)


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def owned(path):
    p = Path(path).resolve()
    require(
        p.is_relative_to(BASE.resolve()) and p != BASE.resolve(),
        "Write outside German audit refused",
    )
    return p


def write_once(path, data):
    """Atomic publication; an interrupted temporary file is kept, never reused."""
    p = owned(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        require(p.read_bytes() == data, f"Existing evidence differs: {p}")
        return
    tmp = owned(p.with_name(p.name + f".partial_{time.time_ns()}"))
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    require(not p.exists(), f"Concurrent output appeared: {p}")
    tmp.rename(p)


def js(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def jsonl(rows):
    return "".join(
        (
            json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for r in rows
        )
    ).encode("utf-8")


def canonical_id(value):
    value = str(value)
    if value.startswith("ex-detox-legal-"):
        return "detox-" + value[len("ex-detox-legal-") :]
    return value.removeprefix("ex-")


def heldout_paths():
    return sorted(
        (
            p
            for p in (ROOT / "data/processed").glob("de_*.parquet")
            if "train" not in p.stem and ("dev" in p.stem or "test" in p.stem)
        )
    )


def extract_renderer():
    """Execute just the inspected pure function, with no project imports."""
    path = ROOT / "src/hsrag/prompt.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_render_gold"
        )
    )
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return (
        namespace["_render_gold"],
        ast.get_source_segment(path.read_text(encoding="utf-8"), fn),
    )


def prompt_audit(records):
    render, source = extract_renderer()
    botox = [r for r in records if r.get("source") == "botox-train"]
    require(len(botox) == 45, "Unexpected current BoTox count; inspect before freezing")
    samples = [
        {"id": r["id"], "gold_meta": r["meta"], "actual_render": render(r["meta"])}
        for r in botox
    ]
    originals = sorted((ROOT / "experiments/results").glob("de_dev_kbv3*_live.jsonl"))
    originals += sorted(
        (ROOT / "experiments/results").glob("de_legal_kbv3*_live.jsonl")
    )
    botox_ids = {r["id"] for r in botox}
    exposure = []
    for p in originals:
        rows = [r for r in util.read_lines(p) if "_manifest" not in r]
        for arm in ("few_shot", "rag"):
            rr = [r for r in rows if r["arm"] == arm]
            if not rr:
                continue
            slots = [
                sum(
                    (
                        i in botox_ids
                        for ids in r.get("retrieved", {}).values()
                        for i in ids
                    )
                )
                for r in rr
            ]
            exposure.append(
                {
                    "file": util.rel(p),
                    "arm": arm,
                    "rows": len(rr),
                    "rows_with_botox": sum((n > 0 for n in slots)),
                    "botox_slots": sum(slots),
                    "code_versions": dict(
                        Counter((str(r.get("code_version")) for r in rr))
                    ),
                    "kb_versions": dict(
                        Counter((str(r.get("kb_version")) for r in rr))
                    ),
                }
            )
    return (
        {
            "current_function": source,
            "botox_examples": len(botox),
            "positive_legal_examples": sum(
                (bool(r["meta"].get("legal")) for r in botox)
            ),
            "unique_actual_renders": sorted({s["actual_render"] for s in samples}),
            "reproductions": samples,
            "historical_context_exposure": exposure,
            "limit": "Current renderer defect is reproduced. Historical saved IDs establish BoTox exposure, not exact historical prompt bytes. Most rows lack a code stamp. No historical score is changed; effect on accuracy requires controlled new runs.",
        },
        originals,
    )


def prepare():
    if PLAN.exists():
        validate()
        print("Existing frozen preparation verified. No files replaced.", flush=True)
        return
    import pandas as pd

    inputs = [ROOT / p for p in SELF_FILES]
    inputs += [
        ROOT / f"data/processed/{n}.parquet" for n in TRAIN_NAMES
    ] + heldout_paths()
    inputs += [
        ROOT / p
        for p in (
            "kb/records.jsonl",
            "src/hsrag/prompt.py",
            "src/hsrag/classify.py",
            "scripts/build_kb_records.py",
            "config/config.yaml",
            "config/taxonomy.yaml",
            "prompts/classify_v1.txt",
            "experiments/leakage_train_exclusions.json",
            "experiments/leakage_report.json",
            "experiments/leakage_report_de_legal_train.json",
            "experiments/leakage_report_kb_example.json",
        )
    ]
    inputs += sorted((ROOT / "experiments/results").glob("de_dev_kbv3*_live.jsonl"))
    inputs += sorted((ROOT / "experiments/results").glob("de_legal_kbv3*_live.jsonl"))
    hashes = {util.rel(p): util.digest(p) for p in sorted(set(inputs))}
    records = util.read_lines(ROOT / "kb/records.jsonl")
    train, gold = ([], {})
    for name in TRAIN_NAMES:
        for row in pd.read_parquet(ROOT / f"data/processed/{name}.parquet").to_dict(
            "records"
        ):
            require(
                row["lang"] == "de"
                and isinstance(row["text"], str)
                and row["text"].strip(),
                "Bad German TRAIN row",
            )
            rid = str(row["id"])
            require(rid not in gold, "TRAIN IDs overlap")
            gold[rid] = row
            train.append(
                {
                    "id": rid,
                    "source": str(row["source"]),
                    "split": name,
                    "text": row["text"],
                }
            )
    held, by_text, id_texts, counts, empty_refs = ([], {}, {}, {}, [])
    for p in heldout_paths():
        df = pd.read_parquet(p, columns=["id", "text", "lang"])
        counts[util.rel(p)] = len(df)
        require(df.id.nunique() == len(df), f"Duplicate held-out IDs: {p.name}")
        for row in df.itertuples(index=False):
            rid = str(row.id)
            require(
                row.lang == "de" and isinstance(row.text, str),
                "Bad held-out text type/language",
            )
            require(
                rid not in id_texts or id_texts[rid] == row.text,
                "Held-out ID has conflicting texts",
            )
            id_texts[rid] = row.text
            ref = {"id": rid, "file": util.rel(p)}
            if not row.text.strip():
                empty_refs.append(ref)
                continue
            if row.text not in by_text:
                by_text[row.text] = len(held)
                held.append({"text": row.text, "references": []})
            held[by_text[row.text]]["references"].append(ref)
    require(
        set(counts)
        == {
            f"data/processed/{name}.parquet"
            for name in (
                "de_dev",
                "de_dev_eval",
                "de_test",
                "de_test_eval",
                "de_legal_dev",
                "de_legal_dev_eval",
                "de_legal_test",
                "de_legal_test_eval",
            )
        },
        "German held-out roster changed; review scope",
    )
    historical = util.read_json(ROOT / "experiments/leakage_train_exclusions.json")
    require(historical["threshold"] == 0.95, "Historical threshold changed")
    historical_ids = {canonical_id(i) for i in historical["ids"]["de"]}
    require(
        historical_ids <= gold.keys(),
        "Historical exclusion cannot be mapped to German TRAIN",
    )
    original_de = [r for r in records if r["kind"] == "example" and r["lang"] == "de"]
    require(
        len({canonical_id(r["id"]) for r in original_de}) == len(original_de),
        "Repeated KB alias",
    )
    for r in original_de:
        rid = canonical_id(r["id"])
        require(
            rid in gold and r["text"] == gold[rid]["text"],
            f"KB record differs from TRAIN: {r['id']}",
        )
    exact = set(by_text)
    normalized = {util.norm(t) for t in exact}
    held_ids = set(id_texts)
    exact_ids = {r["id"] for r in train if r["text"] in exact}
    normalized_ids = {r["id"] for r in train if util.norm(r["text"]) in normalized}
    shared_ids = {r["id"] for r in train if r["id"] in held_ids}
    groups = defaultdict(list)
    for r in train:
        groups[util.canonical_text(r["text"])].append(r)
    duplicate_groups = []
    for group in groups.values():
        if len(group) < 2:
            continue
        annotations = {}
        conflicts = []
        for field in ("gate", "target_groups", "hate_types", "severity", "legal"):
            values = [util.clean(gold[r["id"]].get(field)) for r in group]
            observed = [v for v in values if v is not None]
            canonical = {
                json.dumps(sorted(v) if isinstance(v, list) else v, sort_keys=True)
                for v in observed
            }
            annotations[field] = values
            if len(canonical) > 1:
                conflicts.append(field)
        duplicate_groups.append(
            {
                "ids": [r["id"] for r in group],
                "sources": [r["source"] for r in group],
                "byte_identical": len({r["text"] for r in group}) == 1,
                "annotations": annotations,
                "conflicting_annotated_fields": conflicts,
            }
        )
    excluded = historical_ids | normalized_ids | shared_ids
    excluded_texts = {
        util.canonical_text(r["text"]) for r in train if r["id"] in excluded
    }
    cheap_ids = {
        r["id"] for r in train if util.canonical_text(r["text"]) in excluded_texts
    }
    remaining = [r for r in train if r["id"] not in cheap_ids]
    prompt, _ = prompt_audit(records)
    summary = {
        "raw_train_rows": len(train),
        "train_sources": dict(Counter((r["source"] for r in train))),
        "heldout_file_counts": counts,
        "heldout_unique_ids": len(held_ids),
        "heldout_unique_exact_texts": len(held),
        "empty_heldout_unique_ids": len({r["id"] for r in empty_refs}),
        "original_german_examples": len(original_de),
        "duplicate_text_groups": len(duplicate_groups),
        "duplicate_excess_rows": sum((len(g["ids"]) - 1 for g in duplicate_groups)),
        "duplicate_groups_with_annotation_conflict": sum(
            (bool(g["conflicting_annotated_fields"]) for g in duplicate_groups)
        ),
        "historical_exclusion_aliases": len(historical["ids"]["de"]),
        "historical_underlying_exclusions": len(historical_ids),
        "exact_heldout_match_train_rows": len(exact_ids),
        "normalized_heldout_match_train_rows": len(normalized_ids),
        "shared_train_heldout_ids": len(shared_ids),
        "cheap_excluded_train_rows": len(cheap_ids),
        "provisional_remaining_train_rows": len(remaining),
        "provisional_remaining_unique_texts": len(
            {util.canonical_text(r["text"]) for r in remaining}
        ),
        "provisional_small_examples": sum(
            (canonical_id(r["id"]) not in cheap_ids for r in original_de)
        ),
    }
    survey = {
        "summary": summary,
        "prompt_audit": prompt,
        "duplicate_groups": duplicate_groups,
        "historical_excluded_train_ids": sorted(historical_ids),
        "exact_overlap_train_ids": sorted(exact_ids),
        "normalized_overlap_train_ids": sorted(normalized_ids),
        "shared_ids": sorted(shared_ids),
        "cheap_excluded_train_ids": sorted(cheap_ids),
        "empty_heldout_references_not_embedded": empty_refs,
        "limits": "Provisional counts only: semantic scan and final duplicate/annotation merge still pending. Held-out files were read for identity/text/language only. No performance result or active KB build.",
    }
    generated = {
        "train_texts.jsonl": jsonl(train),
        "heldout_texts.jsonl": jsonl(held),
        "preparation.json": js(survey),
    }
    for name, content in generated.items():
        write_once(BASE / name, content)
    plan = {
        "run_id": RUN_ID,
        "created_utc": util.now(),
        "input_hashes": hashes,
        "generated_hashes": {
            util.rel(BASE / n): util.digest(BASE / n) for n in generated
        },
        "environment": util.environment(),
        "embedding_cache": util.cache_info(),
        "summary": summary,
        "model": "BAAI/bge-m3",
        "precision": "float32",
        "batch_size": 4,
        "cache_chunk": 128,
        "train_similarity_chunk": 256,
        "eval_similarity_chunk": 512,
        "threshold": 0.95,
        "very_near": 0.98,
        "policy": "Audit all German TRAIN rows against all German dev/test texts, including full splits. Report every TRAIN row's maximum, not only one TRAIN winner per eval item. Proposed exclusions are union of historical, normalized/ID overlap and cosine >=0.95, propagated across canonical TRAIN text aliases. Never modify original exclusions. Whitespace-only held-out texts are explicitly inventoried and not embedded. Threshold is a conservative filter, not proof each pair is a duplicate. No classifier calls. No held-out gold labels. No KB generation. Stop for review after audit; final pool/renderer experiment is a subsequent stage.",
    }
    require(
        all((util.digest(ROOT / p) == h for p, h in hashes.items())),
        "An input changed during preparation",
    )
    write_once(PLAN, js(plan))
    print(
        json.dumps(
            {
                "status": "prepared_only",
                "summary": summary,
                "plan": util.rel(PLAN),
                "plan_sha256": util.digest(PLAN),
                "prompt_defect": prompt["unique_actual_renders"],
            },
            indent=2,
        ),
        flush=True,
    )


def validate(check_cache=True):
    plan = util.read_json(PLAN)
    require(plan["run_id"] == RUN_ID, "Wrong audit plan")
    for category in ("input_hashes", "generated_hashes"):
        for p, h in plan[category].items():
            require(util.digest(ROOT / p) == h, f"Frozen input changed: {p}")
    require(
        set(map(util.rel, heldout_paths()))
        == set(plan["summary"]["heldout_file_counts"]),
        "Held-out roster changed",
    )
    require(
        util.environment() == plan["environment"], "Python/package versions changed"
    )
    if check_cache:
        require(util.cache_info() == plan["embedding_cache"], "BGE-M3 cache changed")
    print(
        "Frozen inputs, audit files, environment and model cache verified.", flush=True
    )
    return plan


@contextmanager
def lock():
    import msvcrt

    p = owned(BASE / "audit.lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+b") as f:
        if not p.stat().st_size:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError(
                "Another German audit is running. Use only one terminal."
            ) from None
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def chunk_identity(plan, texts):
    return {
        "plan_sha256": util.digest(PLAN),
        "model": plan["model"],
        "revision": plan["embedding_cache"]["revision"],
        "precision": plan["precision"],
        "texts_sha256": hashlib.sha256(js(texts)).hexdigest(),
        "rows": len(texts),
    }


def validate_vectors(vectors, rows):
    import numpy as np

    require(
        vectors.dtype == np.float32 and vectors.shape == (rows, 1024),
        "Bad embedding dtype/shape",
    )
    require(bool(np.isfinite(vectors).all()), "Nonfinite embeddings")
    require(
        bool(np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=0.0002)),
        "Unnormalized embeddings",
    )


def load_chunk(path, identity):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        vectors = data["vectors"]
    checksum = meta.pop("vectors_sha256")
    require(meta == identity, "Embedding cache identity mismatch")
    validate_vectors(vectors, identity["rows"])
    require(
        hashlib.sha256(vectors.tobytes()).hexdigest() == checksum,
        "Embedding cache payload changed",
    )
    return vectors


def save_chunk(path, identity, vectors):
    import numpy as np

    p = owned(path)
    validate_vectors(vectors, identity["rows"])
    p.parent.mkdir(parents=True, exist_ok=True)
    require(not p.exists(), "Existing embedding chunk will not be replaced")
    tmp = owned(p.with_name(p.name + f".partial_{time.time_ns()}"))
    meta = {**identity, "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest()}
    with tmp.open("xb") as f:
        np.savez(
            f, vectors=vectors, metadata=np.asarray(json.dumps(meta, sort_keys=True))
        )
        f.flush()
        os.fsync(f.fileno())
    require(not p.exists(), "Concurrent embedding cache appeared")
    tmp.rename(p)


def encode_rows(plan, texts, tag, get_model):
    import numpy as np

    vectors = []
    paths = []
    for start in range(0, len(texts), plan["cache_chunk"]):
        sub = texts[start : start + plan["cache_chunk"]]
        identity = chunk_identity(plan, sub)
        key = hashlib.sha256(js(identity)).hexdigest()
        path = BASE / "embedding_chunks" / f"{tag}_{start:06d}_{key[:16]}.npz"
        cached = path.exists()
        if not cached:
            model = get_model()
            v = np.asarray(
                model.encode(
                    sub,
                    batch_size=plan["batch_size"],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
            save_chunk(path, identity, v)
        vectors.append(load_chunk(path, identity))
        paths.append(path)
        print(
            f"EMBED {tag}: {min(start + len(sub), len(texts))}/{len(texts)} ({('verified cache' if cached else 'encoded')})",
            flush=True,
        )
    return (np.concatenate(vectors, axis=0), paths)


def nearest(train, held, train_chunk=256, eval_chunk=512):
    """Maximum for EACH training row, retaining all rows over threshold."""
    import numpy as np

    require(len(train) > 0 and len(held) > 0, "Empty scan")
    maxima = np.full(len(train), -np.inf, dtype=np.float32)
    args = np.full(len(train), -1, dtype=np.int64)
    for a in range(0, len(train), train_chunk):
        tr = train[a : a + train_chunk]
        best, arg = (maxima[a : a + len(tr)], args[a : a + len(tr)])
        for b in range(0, len(held), eval_chunk):
            sims = tr @ held[b : b + eval_chunk].T
            ix = sims.argmax(axis=1)
            val = sims[np.arange(len(tr)), ix]
            update = val > best
            best[update], arg[update] = (val[update], ix[update] + b)
        if a % 1024 == 0 or a + len(tr) == len(train):
            print(f"COMPARE: {a + len(tr)}/{len(train)} TRAIN rows", flush=True)
    require(
        bool(np.isfinite(maxima).all()) and bool((args >= 0).all()),
        "Incomplete similarity scan",
    )
    return (maxima, args)


def finalize(plan, train, held, maxima, args, cache_paths, resources):
    survey = util.read_json(BASE / "preparation.json")
    near_ids = {r["id"] for r, s in zip(train, maxima) if float(s) >= plan["threshold"]}
    exclusions = set(survey["cheap_excluded_train_ids"]) | near_ids
    excluded_text = {
        util.canonical_text(r["text"]) for r in train if r["id"] in exclusions
    }
    exclusions |= {
        r["id"] for r in train if util.canonical_text(r["text"]) in excluded_text
    }
    remain = [r for r in train if r["id"] not in exclusions]
    small = [
        r
        for r in util.read_lines(ROOT / "kb/records.jsonl")
        if r["kind"] == "example" and r["lang"] == "de"
    ]
    nearest_rows, pairs = ([], [])
    for r, s, ix in zip(train, maxima, args):
        h = held[int(ix)]
        item = {
            "train_id": r["id"],
            "train_split": r["split"],
            "max_cosine": float(s),
            "nearest_heldout_references": h["references"],
            "heldout_text_index": int(ix),
            "threshold_excluded": r["id"] in near_ids,
            "proposed_excluded": r["id"] in exclusions,
        }
        nearest_rows.append(item)
        if float(s) >= plan["threshold"] - 0.0005 or r["id"] in exclusions:
            pairs.append({**item, "train_text": r["text"], "heldout_text": h["text"]})
    report = {
        "status": "audit_complete_pool_not_built",
        "plan_sha256": util.digest(PLAN),
        "train_rows": len(train),
        "heldout_texts": len(held),
        "near_train_rows": len(near_ids),
        "very_near_train_rows": sum((float(s) >= plan["very_near"] for s in maxima)),
        "borderline_train_rows": sum(
            (abs(float(s) - plan["threshold"]) <= 0.0005 for s in maxima)
        ),
        "proposed_excluded_ids": sorted(exclusions),
        "new_exclusions_beyond_cheap": len(
            exclusions - set(survey["cheap_excluded_train_ids"])
        ),
        "remaining_train_rows": len(remain),
        "remaining_unique_canonical_texts": len(
            {util.canonical_text(r["text"]) for r in remain}
        ),
        "remaining_small_examples": sum(
            (canonical_id(r["id"]) not in exclusions for r in small)
        ),
        "remaining_sources": dict(Counter((r["source"] for r in remain))),
        "resources": resources,
        "limits": "This proposes exclusions, not classifier results or final merged-KB sizes. Review aliases, annotation conflicts, threshold pairs and missing-label rendering before building pools. German-only scan: English overlap clearance is separate. Held-out texts used for filtering, no gold scores.",
    }
    paths = {
        "nearest_train.jsonl": jsonl(nearest_rows),
        "review_pairs.jsonl": jsonl(pairs),
        "report.json": js(report),
    }
    for name, content in paths.items():
        write_once(BASE / name, content)
    completion = {
        "plan_sha256": util.digest(PLAN),
        "outputs": {util.rel(BASE / n): util.digest(BASE / n) for n in paths},
        "embedding_hashes": {util.rel(p): util.digest(p) for p in cache_paths},
    }
    write_once(BASE / "complete.json", js(completion))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "proposed_excluded_ids"}, indent=2
        ),
        flush=True,
    )
    print(
        "GERMAN OVERLAP AUDIT COMPLETE. Tell Astra to inspect the saved report. No classifier run has started.",
        flush=True,
    )


def check_complete():
    p = BASE / "complete.json"
    if not p.exists():
        return False
    receipt = util.read_json(p)
    require(receipt["plan_sha256"] == util.digest(PLAN), "Completion plan differs")
    for group in ("outputs", "embedding_hashes"):
        for name, h in receipt[group].items():
            require(
                util.digest(ROOT / name) == h, f"Completed audit file changed: {name}"
            )
    print(
        "GERMAN OVERLAP AUDIT ALREADY COMPLETE. Saved outputs verified; nothing rerun.",
        flush=True,
    )
    return True


def execute(plan):
    if check_complete():
        return
    require(
        not (BASE / "report.json").exists(),
        "An unreceipted report exists. Preserve it and ask Astra to inspect.",
    )
    import torch

    require(
        torch.cuda.is_available(),
        "CUDA is unavailable. Enable the GPU, then rerun the same command.",
    )
    require(
        shutil.disk_usage(BASE).free >= 2 * 2**30, "Need at least 2 GiB free disk space"
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = plan["embedding_cache"]["hub"]
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    print(
        f"CUDA: {torch.cuda.get_device_name(0)}. Local BGE-M3, FP32, batch {plan['batch_size']}.",
        flush=True,
    )
    model_box = []

    def model():
        if not model_box:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(
                plan["model"],
                revision=plan["embedding_cache"]["revision"],
                local_files_only=True,
                device="cuda",
            )
            m.float()
            m.eval()
            model_box.append(m)
            print(
                f"Loaded pinned model; max sequence length {m.max_seq_length}.",
                flush=True,
            )
        return model_box[0]

    train = util.read_lines(BASE / "train_texts.jsonl")
    held = util.read_lines(BASE / "heldout_texts.jsonl")
    a, ap = encode_rows(plan, [r["text"] for r in train], "train", model)
    b, bp = encode_rows(plan, [r["text"] for r in held], "heldout", model)
    peak = torch.cuda.max_memory_allocated() / 2**30
    model_box.clear()
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    mx, ix = nearest(
        a, b, plan["train_similarity_chunk"], plan["eval_similarity_chunk"]
    )
    validate()
    finalize(
        plan,
        train,
        held,
        mx,
        ix,
        ap + bp,
        {
            "attempt_seconds_before_report": time.perf_counter() - start,
            "peak_allocated_gib": peak,
            "gpu": torch.cuda.get_device_name(0),
            "note": "Attempt timing may include reused chunks; not full fresh-run cost.",
        },
    )


class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "validate", "run", "status"))
    args = parser.parse_args()
    if args.mode == "prepare":
        with lock():
            prepare()
    elif args.mode == "validate":
        validate()
        check_complete()
    elif args.mode == "status":
        print(
            json.dumps(
                {
                    "prepared": PLAN.exists(),
                    "complete": (BASE / "complete.json").exists(),
                    "cached_chunks": len(
                        list((BASE / "embedding_chunks").glob("*.npz"))
                    ),
                },
                indent=2,
            )
        )
    else:
        with lock():
            p = owned(BASE / "logs" / f"audit_{time.time_ns()}.log")
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("x", encoding="utf-8") as f, redirect_stdout(
                Tee(sys.stdout, f)
            ), redirect_stderr(Tee(sys.stderr, f)):
                try:
                    execute(validate())
                except Exception:
                    import traceback

                    traceback.print_exc()
                    print(
                        "STOPPED. Logs and completed embedding chunks are preserved. No original files changed.",
                        flush=True,
                    )
                    raise


if __name__ == "__main__":
    main()
