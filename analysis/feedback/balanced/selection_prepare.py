import os

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "2"
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    ANONYMIZED_TELEMETRY="False",
    TOKENIZERS_PARALLELISM="false",
)
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import socket
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
from analysis.common import feedback_inputs as c
from analysis.common import retrieval as retrieval
from analysis.common import prompt as renderer
from analysis.feedback.source.eligibility import key as norm
from analysis.feedback.source.eligibility import prefixkey
from analysis.feedback.source.eligibility import components

BASE = c.BASE / "benign_selection"
AUDIT = c.ROOT / "experiments/astra/english_kb_audit_20260914_v1"
RET = c.CORE / "retrieval_v1"
SEEDS = [2026100441, 2026100442, 2026100443]
NEAR = 0.9495
PROTOCOL = "# Benign and balanced correction experiment: selection and update rules\n\n## Question and controlled information\n\nDoes correcting false alarms as well as missed hate improve transfer? Earlier SQ3 corrections were all hateful and supplied type labels. This new study supplies only the binary hate/non-hate correction label in every pool. Other source heads remain unannotated; no type, target, severity or legal value is imputed. It is a separate gate-feedback study, not an isolated polarity comparison against historical type-feedback scores.\n\nKeep the full KB, hybrid retrieval, two definitions, two guidelines, five ordinary examples and a single guaranteed extra correction slot. Six matched pool arms: error-positive, error-balanced, correct-positive, correct-balanced, random-positive, random-balanced. Positive pools are entirely hateful; balanced pools contain half hateful and half benign. A fresh no-correction reference is shared. Within an arm/seed, the balanced pool retains the first half of that arm's positive pool and replaces the second half with benign records. The same inspected candidate universe is available to all arms; report labels inspected as well as examples inserted. Random controls may contain mistakes: this is intentional, and identities/statuses must be reported.\n\n## Source/evaluation separation before calls\n\nCandidates come only from the wider English DEV implicit-hate source, with a known binary gate, outside every English core evaluation/correction input. Both polarities use the same data source. No TRAIN example is relabelled as new feedback. Existing 675 taxonomy evaluation inputs (467 hateful, 208 benign) remain evaluation-only. Preserve their known overlap clusters and exposed-DEV status; the benign part mixes source datasets, so give source-stratified diagnostics. Do not interpret its prevalence as natural deployment prevalence.\n\nScreen the complete candidate universe using cached exact-text FP32 BGE-M3 vectors. Exclude candidate components touching any English core input or an English full-KB training example through normalized/prefix-normalized text equality or cosine >=0.9495 (0.95 plus a conservative 0.0005 margin). Candidate-to-candidate edges are transitive; exclude mixed-gate components and choose one stable-hash representative otherwise. Then take at most 512 candidates per polarity in hash order (seed 202610044). Freeze the entire universe, exclusions, retained roster and vector provenance before any new response. No adaptive extra candidate batches are allowed under this version. Read shared old embedding caches for the selected DEV texts only; no new TEST labels/outcomes or neural forward pass.\n\n## Fresh source selection baseline\n\nEvery retained candidate receives the main full-KB prompt, three independent decoder repeats, three votes each. Use an isolated copied full index and original retrieval code with cached query vectors. Keep first-pass messages frozen. Reconstruct prompt/schema/tokens, and compare a cold second retrieval pass before live readiness. A null final prediction in any repeat excludes that candidate from every actual pool, including random controls. Otherwise, majority of the three repeated binary decisions determines error/correct status. Do not use fine-head disagreement or any evaluation result for source selection. This is label-aware correction selection on a dedicated candidate set, not evaluation leakage.\n\nAfter baseline execution, stop for raw-response audit and pool publication. Use the largest predeclared common budget in [40,20] for which EACH of hateful-error, benign-error, hateful-correct, benign-correct has at least that many distinct eligible candidates; otherwise stop as insufficient supply. This conservative rule also supports full-size matched controls. No relaxing the error definition, dropping difficult cases, or searching more DEV for a favourable pool. For seeds 2026100441/2/3, independently stable-hash order each status/polarity list. Pool overlap across seeds is allowed and recorded; these are not three independent datasets.\n\n## Frozen update stage after audited selection\n\nFor each seed, build the six pools above, retrieve one correction using the existing tiny hybrid bucket (dense top10, BM25 positive-score top10, RRF60, deterministic ID ties), with text-only ranking. All selected source gate labels are then rendered. No query gold routes a correction. No selective threshold or extra explanation. Existing 675 evaluation base messages remain identical across arms. Nineteen settings including shared reference ×675×three repeats×three votes = 115,425 ordinary calls, before bounded repairs/retries. This update stage cannot start until its exact pools, prompts and independent checks are frozen. It is not part of the first selection launch.\n\nTwo co-primary outcomes compare balanced-error versus positive-error: benign false-positive-rate change on the 208 benign texts and false-negative-rate change on the 467 hateful texts, each equally averaging three pools and three repeats. Report 97.5% cluster-bootstrap intervals (two-outcome family, 10,000 draws, seeds 202610044 and 202610045). A common mask per population requires all 18 primary outputs. Retain original text clusters, all pools and repeats together. A false-alarm decrease accompanied by more missed hate is a trade-off, not an unqualified improvement. Do not declare non-inferiority from a non-significant recall loss.\n\nSupporting: each arm versus reference, matched already-correct and random controls, balanced accuracy, binary macro F1 at this artificial prevalence, fine-head regressions despite gate-only feedback, per-source benign results, delivered polarity, source-query similarity, overlap of actual pools, annotation/inference costs and all runs. Supporting intervals are 95%, explicitly not new primary discoveries. Keep null outputs; show all-three-valid, cluster-representative, null-as-error and opposite-bound sensitivities. Original SQ3 primaries remain closed. Ordinary insertion competing for five slots, sequential conflicts and new-concept learning remain separate questions.\n"


def put_array(path, array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    c.require(
        path.resolve().is_relative_to(c.BASE.resolve()), "Array outside owned folder"
    )
    if path.exists():
        c.require(path.read_bytes() == buffer.getvalue(), "Preserve vector artifact")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(buffer.getvalue())


def load_vectors(wanted):

    plan = c.read(AUDIT / "plan.json")
    done = c.read(AUDIT / "complete.json")
    c.require(
        done["plan_sha256"] == c.sha(AUDIT / "plan.json"), "Cache receipt differs"
    )
    c.require(
        c.sha(AUDIT / "heldout_texts.jsonl")
        == plan["generated_hashes"][c.rel(AUDIT / "heldout_texts.jsonl")],
        "Cache text roster differs",
    )
    stored = c.rows(AUDIT / "heldout_texts.jsonl")
    found = {}
    provenance = {}
    files = {}
    for start in range(0, len(stored), plan["cache_chunk"]):
        block = stored[start : start + plan["cache_chunk"]]
        picks = [i for i, r in enumerate(block) if r["text"] in wanted]
        if not picks:
            continue
        ident = {
            "plan_sha256": c.sha(AUDIT / "plan.json"),
            "model": plan["model"],
            "revision": plan["embedding_cache"]["revision"],
            "precision": "float32",
            "texts_sha256": c.sig([r["text"] for r in block]),
            "rows": len(block),
        }
        path = AUDIT / f"embedding_chunks/heldout_{start:06d}_{c.sig(ident)[:16]}.npz"
        c.require(
            c.sha(path) == done["embedding_hashes"][c.rel(path)],
            "Cached vector file changed",
        )
        matrix = retrieval.load_chunk(path, ident)
        files[c.rel(path)] = c.sha(path)
        for i in picks:
            text = block[i]["text"]
            c.require(text not in found, "Duplicate exact cache text")
            refs = block[i]["references"]
            c.require(
                any(
                    (
                        r["file"] == "data/processed/en_dev.parquet"
                        and r["id"] in wanted[text]
                        for r in refs
                    )
                ),
                "Candidate has no DEV cache provenance",
            )
            found[text] = matrix[i].copy()
            provenance[text] = {
                "file": c.rel(path),
                "row": i,
                "references": [
                    r for r in refs if r["file"] == "data/processed/en_dev.parquet"
                ],
            }
    c.require(
        set(found) == set(wanted), "Missing candidate cache; no model fallback allowed"
    )
    return (found, provenance, files)


def scope():
    c.require(
        not (BASE / "scope_complete.json").exists(), "Preserve prepared source scope"
    )
    c.write(BASE / "STUDY_PROTOCOL.md", PROTOCOL)
    index = sorted((p for p in (RET / "full/chroma").rglob("*") if p.is_file()))
    c.require(
        len(index) == 6
        and (not any((p.name.endswith(("-wal", "-shm")) for p in index))),
        "Source index not settled",
    )
    paths = [
        __file__,
        c.ROOT / "data/processed/en_dev.parquet",
        c.CORE / "dev_inputs.jsonl",
        c.CORE / "plan.json",
        c.CORE / "kb/full/records.jsonl",
        RET / "query_vectors.npy",
        RET / "full_vectors.npy",
        RET / "plan.json",
        c.OLD_TAX / "roster.jsonl",
        c.OLD_TAX / "prompts.jsonl",
        AUDIT / "plan.json",
        AUDIT / "complete.json",
        AUDIT / "heldout_texts.jsonl",
        BASE / "STUDY_PROTOCOL.md",
        c.ROOT / "analysis/common/retrieval.py",
        c.ROOT / "analysis/feedback/source/eligibility.py",
        c.ROOT / "src/hsrag/retrieve.py",
        c.ROOT / "src/hsrag/kb.py",
        *index,
    ]
    c.write(
        BASE / "preparation_plan.json",
        {
            "created_utc": c.utc(),
            "inputs": c.inputs_for_study(paths),
            "original_bindings": c.preserve_originals(),
            "seed": 202610044,
            "cap_per_polarity": 512,
            "near_exclusion": NEAR,
            "source_index_files": [c.rel(p) for p in index],
            "source_polarity": "gate only, type presence not an eligibility criterion",
            "model_calls": 0,
        },
    )
    df = pd.read_parquet(c.ROOT / "data/processed/en_dev.parquet")
    c.require(
        c.sha(c.ROOT / "data/processed/en_dev.parquet")
        == c.read(AUDIT / "plan.json")["input_hashes"]["data/processed/en_dev.parquet"],
        "DEV changed",
    )
    core = c.rows(c.CORE / "dev_inputs.jsonl")
    protected = [r for r in core if r["lang"] == "en"]
    protected_ids = {r["item_id"] for r in protected}
    universe = []
    for r in df.to_dict("records"):
        if (
            r["source"] == "implicit_hate"
            and type(r["gate"]) is bool
            and (r["id"] not in protected_ids)
        ):
            universe.append(
                {
                    "id": r["id"],
                    "key": "en/" + r["id"],
                    "text": r["text"],
                    "lang": "en",
                    "source": r["source"],
                    "gold": {
                        "hate": r["gate"],
                        "hate_type": (
                            list(r["hate_types"])
                            if r["hate_types"] is not None
                            else None
                        ),
                    },
                    "population": (
                        "hateful_candidate" if r["gate"] else "benign_candidate"
                    ),
                }
            )
    universe.sort(key=lambda r: r["id"])
    c.require(len({r["id"] for r in universe}) == len(universe), "Duplicate candidates")
    c.jsonlines(BASE / "candidate_universe.jsonl", universe)
    wanted = defaultdict(set)
    for r in universe:
        wanted[r["text"]].add(r["id"])
    cache, provenance, files = load_vectors(wanted)
    c.write(BASE / "cache_bindings.json", files)
    uv = np.stack([cache[r["text"]] for r in universe])
    retrieval.validate_vectors(uv, len(universe))
    qv = np.load(RET / "query_vectors.npy", allow_pickle=False)
    pv = np.stack([qv[i] for i, r in enumerate(core) if r["lang"] == "en"])
    all_rows = [
        *universe,
        *[{"id": r["item_id"], "text": r["text"]} for r in protected],
    ]
    av = np.concatenate([uv, pv])
    av = av / np.linalg.norm(av, axis=1, keepdims=True)
    ids = [r["id"] for r in all_rows]
    c.require(len(set(ids)) == len(ids), "Source/evaluation identity leak")
    edges = []
    normal = defaultdict(list)
    for r in all_rows:
        normal[prefixkey(r["text"])].append(r["id"])
    for group in normal.values():
        edges.extend(((group[0], iid) for iid in group[1:]))
    for start in range(0, len(ids), 128):
        sim = av[start : start + 128] @ av.T
        x, y = np.where(sim >= NEAR)
        edges.extend(
            (
                (ids[start + int(i)], ids[int(j)])
                for i, j in zip(x, y)
                if start + int(i) < int(j)
            )
        )
    records = c.rows(c.CORE / "kb/full/records.jsonl")
    fv = np.load(RET / "full_vectors.npy", mmap_mode="r", allow_pickle=False)
    c.verify(c.read(RET / "plan.json")["generated_hashes"])
    ix = [
        i for i, r in enumerate(records) if r["kind"] == "example" and r["lang"] == "en"
    ]
    train = np.asarray(fv[ix], dtype=np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    train_norm = {prefixkey(records[i]["text"]) for i in ix}
    exposed = set()
    near = []
    for start in range(0, len(universe), 128):
        sims = av[start : start + 128] @ train.T
        closest = np.argmax(sims, axis=1)
        for offset in range(min(128, len(universe) - start)):
            j = start + offset
            r = universe[j]
            ti = int(closest[offset])
            value = float(sims[offset, ti])
            same = prefixkey(r["text"]) in train_norm
            if same or value >= NEAR:
                exposed.add(r["id"])
            near.append(
                {
                    "id": r["id"],
                    "train_id": records[ix[ti]]["id"],
                    "cosine": value,
                    "normalized_exact": same,
                    "excluded": same or value >= NEAR,
                }
            )
    c.jsonlines(BASE / "nearest_train.jsonl", near)
    groups = components(ids, edges)
    lookup = {r["id"]: r for r in universe}
    eligible = []
    excluded = {}
    clusters = {}
    for members in groups:
        candidates = [i for i in members if i in lookup]
        if not candidates:
            continue
        cluster = (
            "benignfb-" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        )
        for iid in members:
            clusters[iid] = cluster
        if set(members) & protected_ids:
            reason = "component_touches_English_core"
        elif set(members) & exposed:
            reason = "component_touches_existing_TRAIN"
        elif len({lookup[i]["gold"]["hate"] for i in candidates}) > 1:
            reason = "mixed_gold_candidate_component"
        else:
            chosen = min(candidates, key=lambda i: c.source_order(202610044, i))
            eligible.append(chosen)
            for iid in candidates:
                if iid != chosen:
                    excluded[iid] = "one_representative_per_candidate_component"
            continue
        for iid in candidates:
            excluded[iid] = reason
    chosen = []
    for gate in (False, True):
        eligible_gate = sorted(
            [i for i in eligible if lookup[i]["gold"]["hate"] == gate],
            key=lambda i: c.source_order(202610044, i),
        )
        chosen += eligible_gate[:512]
        for iid in eligible_gate[512:]:
            excluded[iid] = "outcome_blind_cap_512_per_polarity"
    chosen.sort()
    c.require(len(chosen) > 0, "No eligible candidates")
    roster = [{**lookup[i], "cluster": clusters[i]} for i in chosen]
    c.require(
        set(chosen).isdisjoint({r["id"] for r in c.rows(c.OLD_TAX / "roster.jsonl")}),
        "Correction/evaluation overlap",
    )
    c.jsonlines(BASE / "roster.jsonl", roster)
    put_array(BASE / "query_vectors.npy", np.stack([cache[r["text"]] for r in roster]))
    c.jsonlines(
        BASE / "query_vector_sources.jsonl",
        [
            {
                "id": r["id"],
                "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
                **provenance[r["text"]],
            }
            for r in roster
        ],
    )
    c.write(
        BASE / "scope_details.json",
        {
            "components": groups,
            "excluded": excluded,
            "eligible_before_cap": sorted(eligible),
            "protected_ids": sorted(protected_ids),
            "edge_count": len(edges),
            "candidate_TRAIN_exposed": sorted(exposed),
        },
    )
    summary = {
        "universe": len(universe),
        "candidate_counts": dict(Counter((r["population"] for r in roster))),
        "retained": len(roster),
        "excluded": len(excluded),
        "exclusion_reasons": dict(Counter(excluded.values())),
        "cached_chunks": len(files),
        "new_embeddings": 0,
        "cuda_jobs": 0,
        "candidate_ordinary_calls": len(roster) * 9,
        "update_ordinary_calls_after_audited_selection": 115425,
    }
    c.write(BASE / "scope_summary.json", summary)
    c.verify(c.read(BASE / "preparation_plan.json")["inputs"])
    c.verify(files)
    c.preserve_originals()
    c.write(
        BASE / "scope_complete.json",
        {
            "status": "scope_frozen_no_responses",
            "outputs": c.bindings((p for p in BASE.iterdir() if p.is_file())),
            "summary": summary,
        },
    )
    print(json.dumps(summary), flush=True)


def retrieve(check=False):
    import chromadb

    c.verify(c.read(BASE / "scope_complete.json")["outputs"])
    c.verify(c.read(BASE / "preparation_plan.json")["inputs"])
    roster = c.rows(BASE / "roster.jsonl")
    qv = np.load(BASE / "query_vectors.npy", allow_pickle=False)
    index = BASE / "retrieval_index"
    paths = c.read(BASE / "preparation_plan.json")["source_index_files"]
    if not (BASE / "index_copy.json").exists():
        c.require(not check, "Check requires a prepared index")
        for name in paths:
            source = c.ROOT / name
            dest = index / source.relative_to(RET / "full/chroma")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                c.require(c.sha(dest) == c.sha(source), "Partial index copy differs")
            else:
                shutil.copyfile(source, dest)
        c.write(
            BASE / "index_copy.json",
            {
                "source_hashes": c.bindings((c.ROOT / n for n in paths)),
                "copied_before_open": c.bindings(
                    (p for p in index.rglob("*") if p.is_file())
                ),
            },
        )
    client = chromadb.PersistentClient(path=str(index))
    col = client.get_collection("hate_kb")
    records = c.rows(c.CORE / "kb/full/records.jsonl")
    known = {r["id"]: r for r in records}
    fv = np.load(RET / "full_vectors.npy", mmap_mode="r", allow_pickle=False)
    diff = retrieval.verify_collection(col, records, fv)
    obj = retrieval.make_retriever(
        col,
        c.CORE / "kb/full/records.jsonl",
        c.read(c.CORE / "plan.json")["retrieval"],
        retrieval.CachedEncoder([r["text"] for r in roster], qv),
    )
    if not check:
        c.write(
            BASE / "scientific_plan_v1.json",
            {
                "conditions": ["reference"],
                "decoder_repeats": 3,
                "votes_per_prediction": 3,
                "phase": "source_selection_only",
                "primary_outcomes": "none in this selection phase",
                "roster_sha256": c.sha(BASE / "roster.jsonl"),
                "protocol_sha256": c.sha(BASE / "STUDY_PROTOCOL.md"),
                "seed": 202610044,
                "pool_seeds": SEEDS,
                "budget_rule": [40, 20],
                "candidate_selection": "binary gate majority over 3 repeats; any final null excludes",
                "population_counts": dict(Counter((r["population"] for r in roster))),
                "stop_after_audit_for_actual_pool_freeze": True,
                "scope": "exposed DEV; no TRAIN or evaluation correction sources",
            },
        )
    tok = c.tokenizer()
    collected = []
    old = c.rows(BASE / "prompts.jsonl") if check else None
    differences = []
    for i, r in enumerate(roster):
        context = renderer.from_retrieval(obj.retrieve(r["text"], "en"))
        retrieval.validate_context(context, known, "en")
        messages = renderer.render_messages(r["text"], "en", context, "corrected")
        row = {
            "key": r["key"],
            "item_id": r["id"],
            "condition": "reference",
            "population": r["population"],
            "text": r["text"],
            "lang": "en",
            "context": context,
            "messages": messages,
            "context_sha256": c.sig(context),
            "messages_sha256": c.sig(messages),
            "reference_input_tokens": c.token_count(tok, messages),
            "scientific_plan_sha256": c.sha(BASE / "scientific_plan_v1.json"),
        }
        collected.append(row)
        if check and old[i] != row:
            differences.append(
                {
                    "key": r["key"],
                    "messages_changed": old[i]["messages"] != row["messages"],
                    "context_changed": old[i]["context"] != row["context"],
                }
            )
        if (i + 1) % 100 == 0:
            print(
                f"{('COLD CHECK' if check else 'RETRIEVE')} candidates {i + 1}/{len(roster)}",
                flush=True,
            )
    if check:
        c.write(
            BASE / "cold_replay.json",
            {
                "compared": len(roster),
                "differences": differences,
                "max_index_vector_difference": diff,
                "source_index_untouched": True,
                "model_calls": 0,
            },
        )
        c.require(
            not differences,
            "Cold retrieval differs: retain first pass and review before launch",
        )
        c.verify(c.read(BASE / "preparation_plan.json")["inputs"])
        c.preserve_originals()
        c.write(
            BASE / "retrieval_checked.json",
            {
                "status": "cold_retrieval_exactly_reproduced",
                "cold_replay_sha256": c.sha(BASE / "cold_replay.json"),
            },
        )
        summary = {
            **c.read(BASE / "scope_summary.json"),
            "prompt_identities": len(collected),
            "predictions": len(collected) * 3,
            "ordinary_calls": len(collected) * 9,
            "input_tokens_min": min((r["reference_input_tokens"] for r in collected)),
            "input_tokens_max": max((r["reference_input_tokens"] for r in collected)),
            "new_provider_calls": 0,
        }
        c.write(BASE / "summary.json", summary)
        c.write(
            BASE / "prepared.json",
            {
                "status": "prepared_requires_independent_check",
                "outputs": c.bindings((p for p in BASE.iterdir() if p.is_file())),
                "created_utc": c.utc(),
            },
        )
        print(json.dumps(summary), flush=True)
    else:
        c.require(
            not (BASE / "prompts.jsonl").exists(), "Do not replace first-pass prompts"
        )
        c.jsonlines(BASE / "prompts.jsonl", collected)
        c.write(
            BASE / "retrieval_first_pass.json",
            {
                "prompts": len(collected),
                "prompts_sha256": c.sha(BASE / "prompts.jsonl"),
                "max_index_vector_difference": diff,
                "model_calls": 0,
            },
        )
    c.verify(c.read(BASE / "index_copy.json")["source_hashes"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["scope", "retrieve", "cold-check"])
    args = p.parse_args()
    with patch.object(
        socket.socket,
        "connect",
        side_effect=AssertionError("Offline cached preparation"),
    ):
        scope() if args.mode == "scope" else retrieve(args.mode == "cold-check")
