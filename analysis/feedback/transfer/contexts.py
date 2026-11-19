import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import unittest
import numpy as np
from analysis.feedback.transfer import archive as a
from analysis.feedback.transfer import contexts_metrics as k

OUT = a.ROOT / "experiments/astra/sq3_transfer_part5_20261023_v1"
P4 = a.ROOT / "experiments/astra/sq3_transfer_part4_20261023_v1"
SCOPE = a.ROOT / "experiments/astra/sq3_followups_20261012_v1/scope"
SEL = a.ROOT / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection"
CAND = a.ROOT / "experiments/astra/sq3_additions_20261004_v1/benign_selection"
CORE = a.ROOT / "experiments/astra/full_kb_core_20260915_v1"
LATER = a.ROOT / "experiments/astra/later_evaluation_20261008_v1"


def native(x):
    if isinstance(x, np.ndarray):
        return native(x.tolist())
    if isinstance(x, np.generic):
        return native(x.item())
    if isinstance(x, dict):
        return {str(key): native(value) for key, value in x.items()}
    if isinstance(x, (list, tuple)):
        return [native(v) for v in x]
    return x


def save(name, value):
    path = OUT / name
    a.need(path.resolve().is_relative_to(OUT), "Write left new analysis")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value if isinstance(value, bytes) else a.encoded(native(value), pretty=True)
        )


def prepare():
    a.need(not OUT.exists(), "Preserve previous Part 5 work")
    previous = a.read(P4 / "complete.json")
    a.need(previous["status"] == "transfer_part4_complete_reviewed", "Part 4 not ready")
    bindings = {
        **previous["input_bindings"],
        **previous["artifacts"],
        **previous["analysis_sources"],
        a.rel(P4 / "complete.json"): a.sha(P4 / "complete.json"),
    }
    prep = a.read(a.STUDY / "preparation_plan.json")
    done = a.read(a.STUDY / "prepared.json")
    scope = a.read(SCOPE / "plan.json")
    selection = a.read(SEL / "complete.json")
    vectors = a.read(CORE / "retrieval_v1/plan.json")
    extra = [
        a.STUDY / n
        for n in (
            "preparation_plan.json",
            "prepared.json",
            "complete.json",
            "independent_verification.json",
            "choices.jsonl",
            "prompts.jsonl",
        )
    ]
    extra += [
        SCOPE / n
        for n in ("plan.json", "complete.json", "evaluation.jsonl", "vectors.npy")
    ]
    extra += [
        SEL / n
        for n in (
            "complete.json",
            "actual_pools.json",
            "gate_only_records.json",
            "candidate_statuses.json",
        )
    ]
    extra += [
        CAND / "roster.jsonl",
        CAND / "query_vectors.npy",
        CORE / "kb/full/records.jsonl",
        CORE / "retrieval_v1/plan.json",
        CORE / "retrieval_v1/full_vectors.npy",
        CORE / "retrieval_v1/record_vector_sources.jsonl",
        LATER / "retrieval_v3/full/prompts.jsonl",
        a.ROOT / "prompts/classify_v1.txt",
        a.ROOT / "analysis/common/prompt.py",
        a.ROOT / "src/hsrag/prompt.py",
        a.ROOT / "analysis/feedback/source/retrieval.py",
        a.ROOT / "analysis/feedback/transfer/prepare.py",
    ]
    authoritative = {
        **prep["inputs"],
        **done["outputs"],
        **scope["inputs"],
        **scope["prepared"],
        **selection["inputs"],
        **selection["outputs"],
        **vectors["generated_hashes"],
    }
    for path in extra:
        name = a.rel(path)
        digest = a.sha(path)
        if name in authoritative:
            a.need(digest == authoritative[name], "Frozen source changed " + name)
        bindings[name] = digest
    for name, digest in bindings.items():
        a.need(a.sha(a.ROOT / name) == digest, "Previous evidence changed " + name)
    from tests.feedback.transfer import test_contexts as tests

    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    a.need(result.wasSuccessful(), "Context fixtures failed " + log.getvalue())
    sources = {
        a.rel(Path(p)): a.sha(Path(p)) for p in (__file__, k.__file__, tests.__file__)
    }
    contract = dict(
        status="part5_analysis_conventions_recorded_before_new_context_outcome_summaries",
        created_utc=a.utc(),
        input_bindings=bindings,
        analysis_sources=sources,
        scope="Offline review of all fixed broader-transfer corrections, displaced fifth neighbours and unchanged first-four/background context; no new intervention.",
        delivery_units="One query-pool assignment per distinct fixed prompt (47,274). Three decoder repeats and nine ordinary initial vote slots per assignment are repeated exposure, not extra distinct feedback. Null outputs remain in delivery denominators.",
        source_rules="Three frozen 40-record earlier-error pools, 20 per binary polarity. Selection is the saved tiny hybrid bucket. All fine correction fields remain None. Original candidate source labels and screening statuses are retained.",
        structure="Check all 63,032 saved user-message renderings, context/message hashes, single common heading, exactly five examples, first-four order/metadata, unchanged definitions/guidelines/system/query, and exact fifth replacement. Reconcile every one of the 189,096 audited prediction-message bindings.",
        provenance="Reuse completed independent full retrieval/scope checks. Recheck saved RRF rank arithmetic and selected source-query cosine using cached vectors; compare ordinary fifth cosine using identity-bound full-KB vectors. No index replay, encoding or new source retrieval.",
        exposure="Full/common/strict assignment counts by population, source, gold polarity, function, pool and source identity. Retain all 120 pool memberships including zero use; report overlap, same-source selections, concentration and repeats.",
        context_information="Retain per-query ordinary hit metadata including query-dependent ranks. Count displaced known fine fields including known empty sets, remaining annotation coverage, fifth source/polarity, and hateful-example count change. Token lengths are saved local prompt estimates, not measured server cost.",
        similarity=dict(
            fixed_bin_edges=list(k.EDGES),
            bin_names=list(k.BIN_NAMES),
            tie_tolerance=1e-12,
            threshold_role="Descriptive bins, not a selected intervention threshold. Frozen transfer overlap cutoff remains 0.9495.",
        ),
        associations="All reported strata are associations. Use the existing Part 2 common/strict/replay/representative endpoint masks and Part 4 fine-head masks. Pooled conditional rates weight query-pool assignments equally; unequal group sizes do not create equal-pool conditional estimands. Also retain every per-pool rate. Duplicated reference observations are not independent outputs.",
        strata=[
            "all",
            "correction polarity",
            "removed-to-inserted polarity",
            "fixed cosine bin",
            "correction cosine relative to fifth",
            "removed known fine field",
            "baseline number of hateful examples",
        ],
        uncertainty="Only main population/polarity and removed-to-inserted polarity strata get supporting 95% whole-inference-cluster intervals (10,000 PCG64 draws, fixed SHA256-derived seed). Resample each included text cluster with all its selected pool assignments/repeats. No new adjusted discovery family or source-pool uncertainty.",
        finer_diagnostics="Known fine exact-set changes by whether fifth held that head annotation or was the last such example; descriptive only, with fixed Part 4 eligibility. No F1 pooled across repeats.",
        case_rule="One complete context pair per population x removed polarity x inserted polarity, minimum SHA256(A16-part5-context-example|key|condition), selected without outcomes. All examples retained locally for Part 6.",
        causal_limit="No masked-label, fifth-retained or random-message controls in this run. Correction identity, label, replaced content, annotation availability and length change together. Earlier controls belong to a different evaluation.",
        new_Qwen_calls=0,
        new_fits=0,
        new_embeddings=0,
    )
    save("analysis_contract.json", contract)
    save(
        "focused_tests.json",
        dict(passed=True, tests_run=result.testsRun, log=log.getvalue()),
    )
    return contract


def build():
    contract = prepare()
    science = a.read(a.STUDY / "scientific_plan_v1.json")
    conditions = science["conditions"]
    pools = [
        p
        for p in a.read(SEL / "actual_pools.json")["pools"]
        if p["condition"] in conditions
    ]
    a.need([p["condition"] for p in pools] == conditions[1:], "Pool order")
    records = a.read(SEL / "gate_only_records.json")
    statuses = a.read(SEL / "candidate_statuses.json")
    source_rows = list(a.rows(CAND / "roster.jsonl"))
    source_index = {r["id"]: i for i, r in enumerate(source_rows)}
    sv = np.load(CAND / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    qv = np.load(SCOPE / "vectors.npy", mmap_mode="r", allow_pickle=False)
    bv = np.load(
        CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    kb = {}
    vector_origins = a.rows(CORE / "retrieval_v1/record_vector_sources.jsonl")
    for i, (r, origin) in enumerate(
        zip(a.rows(CORE / "kb/full/records.jsonl"), vector_origins, strict=True)
    ):
        a.need(
            r["id"] == origin["id"]
            and hashlib.sha256(r["text"].encode("utf8")).hexdigest()
            == origin["text_sha256"],
            "KB vector identity/text binding",
        )
        a.need(r["id"] not in kb, "Duplicate KB ID")
        kb[r["id"]] = (i, r)
    a.need(
        bv.shape == (len(kb), 1024)
        and sv.shape == (1024, 1024)
        and (qv.shape[1] == 1024),
        "Cached vector shape",
    )
    scope = {r["key"]: r for r in a.rows(SCOPE / "evaluation.jsonl")}
    metadata = list(a.rows(P4 / "metadata.jsonl"))
    N = len(metadata)
    members = Counter((s for p in pools for s in p["ids"]))
    source_catalog = {}
    for sid in members:
        r = records[sid]
        old = source_rows[source_index[sid]]
        status = statuses[sid]
        a.need(
            r["provenance"]["source_id"] == sid
            and r["text"] == old["text"]
            and (r["meta"]["gate"] == old["gold"]["hate"] == status["hate"]),
            "Source attribution",
        )
        a.need(
            status["status"] == "error"
            and status["majority_hate"] != status["hate"]
            and all((r["meta"][f] is None for f in k.FIELDS)),
            "Earlier binary error/fine absence",
        )
        source_catalog[sid] = dict(
            record=r,
            source_vector_row=source_index[sid],
            original_candidate_gold=old["gold"],
            screening=status,
            pool_memberships=[p["condition"] for p in pools if sid in p["ids"]],
        )
    for p in pools:
        a.need(
            len(p["ids"]) == len(set(p["ids"])) == 40
            and Counter((records[s]["meta"]["gate"] for s in p["ids"]))
            == {False: 20, True: 20},
            "Frozen pool composition",
        )
    template = (a.ROOT / "prompts/classify_v1.txt").read_text(encoding="utf8")
    parts = re.split("^=== (\\w+) ===$", template, flags=re.MULTILINE)
    tpl = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
    prompts = a.rows(a.STUDY / "prompts.jsonl")
    choices = a.rows(a.STUDY / "choices.jsonl")
    prompt_index = {}
    reference_index = {}
    system_hashes = set()
    pool_totals = Counter()
    contexts = []
    deliveries = []
    cases = {}
    ordinary_usage = Counter()
    ordinary_store = {}
    counters = Counter()
    source_sets = [set((records[s]["id"] for s in p["ids"])) for p in pools]
    context_file = (OUT / "query_contexts.jsonl").open("xb")
    delivery_file = (OUT / "delivery.jsonl").open("xb")
    try:
        for i, m in enumerate(metadata):
            row = scope[m["key"]]
            q = qv[row["scope_node_index"]]
            group = [next(prompts) for _ in conditions]
            a.need(
                [p["condition"] for p in group] == conditions
                and all(
                    (
                        p["key"] == row["key"]
                        and p["text"] == row["text"]
                        and (p["population"] == m["population"])
                        for p in group
                    )
                ),
                "Prompt roster/order",
            )
            reference = group[0]
            ctx = reference["context"]
            hits = ctx["example_groups"][0][1]
            a.need(
                reference["source_id"] is None and len(hits) == 5,
                "Reference example budget",
            )
            ordinary_cos = []
            for h in [*ctx["definitions"], *ctx["guidelines"], *hits]:
                j, record = kb[h["id"]]
                expected_meta = {
                    **record["meta"],
                    **{
                        key: record[key]
                        for key in ("dimension", "kind", "label", "lang", "source")
                    },
                }
                a.need(
                    h["text"] == record["text"]
                    and {
                        key: value
                        for key, value in h["meta"].items()
                        if key != "_ranks"
                    }
                    == expected_meta,
                    "Ordinary/background metadata origin",
                )
            for h in hits:
                j, record = kb[h["id"]]
                ordinary_cos.append(k.cosine(q, bv[j]))
                ordinary_usage[h["id"]] += 1
                ordinary_store[h["id"]] = dict(vector_row=j, record=record)
            fifth = hits[4]
            fine = {f: fifth["meta"].get(f) is not None for f in k.FIELDS}
            known_counts = {
                f: sum((h["meta"].get(f) is not None for h in hits)) for f in k.FIELDS
            }
            context = dict(
                input_index=i,
                key=m["key"],
                population=m["population"],
                source=m["source"],
                gold_hate=m["gold_hate"],
                inference_cluster=m["inference_cluster"],
                functionality=m["functionality"],
                text=row["text"],
                query_vector_row=row["scope_node_index"],
                reference_messages_sha256=reference["messages_sha256"],
                reference_context_sha256=reference["context_sha256"],
                reference_input_tokens=reference["reference_input_tokens"],
                definitions_ids=[h["id"] for h in ctx["definitions"]],
                guidelines_ids=[h["id"] for h in ctx["guidelines"]],
                ordinary_hits=hits,
                ordinary_cosines=ordinary_cos,
                reference_hateful_examples=sum((h["meta"]["gate"] for h in hits)),
                removed_id=fifth["id"],
                removed_source=fifth["meta"]["source"],
                removed_gate=fifth["meta"]["gate"],
                removed_known_fine=fine,
                reference_known_field_counts=known_counts,
                remaining_known_field_counts={
                    f: known_counts[f] - fine[f] for f in k.FIELDS
                },
            )
            context_file.write(a.encoded(context))
            contexts.append(
                {
                    key: value
                    for key, value in context.items()
                    if key not in ("ordinary_hits", "text")
                }
            )
            reference_index[m["key"]] = (
                reference["messages_sha256"],
                reference["context_sha256"],
            )
            for c, p in enumerate(group):
                a.need(
                    p["context_sha256"] == k.sig(p["context"])
                    and p["messages_sha256"] == k.sig(p["messages"]),
                    "Prompt content checksum",
                )
                a.need(
                    len(p["messages"]) == 2
                    and [x["role"] for x in p["messages"]] == ["system", "user"],
                    "Message roles",
                )
                a.need(
                    p["messages"][0] == reference["messages"][0]
                    and p["messages"][1]["text"]
                    == k.render_user(
                        row["text"], p["context"], tpl["EXAMPLES_CAUTION"]
                    ),
                    "Independent actual user rendering",
                )
                system_hashes.add(k.sig(p["messages"][0]))
                prompt_index[m["key"], p["condition"]] = p["messages_sha256"]
                counters["prompts"] += 1
                if c == 0:
                    continue
                choice = next(choices)
                sid = p["source_id"]
                rec = records[sid]
                a.need(
                    choice["key"] == p["key"]
                    and choice["condition"] == p["condition"]
                    and (choice["source_id"] == sid)
                    and (sid in pools[c - 1]["ids"]),
                    "Choice/prompt/pool identity",
                )
                correction = dict(
                    id=rec["id"], text=rec["text"], meta=rec["meta"], via="rrf"
                )
                removed = k.replacement(ctx, p["context"], correction)
                a.need(
                    removed == fifth and choice["source_gate"] == rec["meta"]["gate"],
                    "Removed/correction label",
                )
                trace = k.fuse(choice["retrieval"], source_sets[c - 1])
                a.need(
                    choice["retrieval"]["selected_id"] == correction["id"],
                    "Choice is not inserted example",
                )
                cos = k.cosine(q, sv[source_index[sid]])
                a.need(
                    abs(cos - choice["cosine"]) < 1e-12 and cos < 0.9495,
                    "Selected cosine/source separation",
                )
                fields = dict(
                    input_index=i,
                    key=m["key"],
                    population=m["population"],
                    source=m["source"],
                    gold_hate=m["gold_hate"],
                    inference_cluster=m["inference_cluster"],
                    functionality=m["functionality"],
                    condition=p["condition"],
                    pool_index=c,
                    source_id=sid,
                    source_gate=rec["meta"]["gate"],
                    source_pool_memberships=members[sid],
                    cosine=cos,
                    cosine_bin=k.similarity_bin(cos),
                    removed_id=fifth["id"],
                    removed_gate=fifth["meta"]["gate"],
                    removed_cosine=ordinary_cos[4],
                    cosine_difference=cos - ordinary_cos[4],
                    similarity_relation=(
                        "higher"
                        if cos - ordinary_cos[4] > 1e-12
                        else "lower" if cos - ordinary_cos[4] < -1e-12 else "tie"
                    ),
                    removed_source=context["removed_source"],
                    removed_known_fine=fine,
                    remaining_known_field_counts=context[
                        "remaining_known_field_counts"
                    ],
                    reference_hateful_examples=context["reference_hateful_examples"],
                    feedback_hateful_examples=context["reference_hateful_examples"]
                    - int(fifth["meta"]["gate"])
                    + int(rec["meta"]["gate"]),
                    hateful_count_change=int(rec["meta"]["gate"])
                    - int(fifth["meta"]["gate"]),
                    messages_sha256=p["messages_sha256"],
                    context_sha256=p["context_sha256"],
                    reference_input_tokens=reference["reference_input_tokens"],
                    feedback_input_tokens=p["reference_input_tokens"],
                    token_difference=p["reference_input_tokens"]
                    - reference["reference_input_tokens"],
                    retrieval_sha256=k.sig(choice["retrieval"]),
                    dense_cosine_regret=trace["dense_best"] - cos,
                    **trace,
                )
                deliveries.append(fields)
                delivery_file.write(a.encoded(fields))
                pool_totals[p["condition"] + "/" + str(rec["meta"]["gate"])] += 1
                case_key = (m["population"], fifth["meta"]["gate"], rec["meta"]["gate"])
                rank = hashlib.sha256(
                    (
                        "A16-part5-context-example|" + m["key"] + "|" + p["condition"]
                    ).encode()
                ).hexdigest()
                if case_key not in cases or rank < cases[case_key]["selection_hash"]:
                    cases[case_key] = dict(
                        selection_hash=rank,
                        delivery=fields,
                        query_context=context,
                        reference_messages=reference["messages"],
                        feedback_messages=p["messages"],
                        correction_record=source_catalog[sid],
                        removed_example=fifth,
                        note="Selected using context identities/polarities only; no prediction outcome used.",
                    )
            if (i + 1) % 3000 == 0:
                print(
                    f"Checked {i + 1}/{N} complete contexts and replacements",
                    flush=True,
                )
    finally:
        context_file.close()
        delivery_file.close()
    a.need(
        next(prompts, None) is None
        and next(choices, None) is None
        and (len(prompt_index) == 63032)
        and (len(deliveries) == 47274),
        "Complete prompt/choice matrix",
    )
    a.need(
        len(system_hashes) == 1
        and dict(pool_totals)
        == a.read(a.STUDY / "independent_verification.json")["delivery"],
        "Existing delivery verification differs",
    )
    matched = 0
    for old in a.rows(LATER / "retrieval_v3/full/prompts.jsonl"):
        if old["key"] in reference_index:
            a.need(
                reference_index[old["key"]]
                == (old["messages_sha256"], old["context_sha256"]),
                "Qualified first-pass reference differs",
            )
            matched += 1
    a.need(matched == N, "All first-pass reference contexts matched")
    seen = set()
    executed = Counter()
    nulls = Counter()
    valids = Counter()
    bindings_by_condition = Counter()
    for path in sorted((a.OUT / "assembly").glob("transfer_*.jsonl")):
        for row in a.rows(path):
            key = (row["key"], row["condition"])
            ident = (*key, row["repeat"])
            a.need(
                ident not in seen
                and row["messages_sha256"] == prompt_index[key]
                and (row["repeat"] in range(3)),
                "Audited actual-output prompt binding",
            )
            seen.add(ident)
            executed[key] += 1
            nulls[key] += row["result"] is None
            valids[key] += row["n_valid"]
            bindings_by_condition[row["condition"]] += 1
    a.need(
        len(seen) == 189096
        and len(executed) == 63032
        and (set(executed.values()) == {3}),
        "Every prompt executed in all three repeats",
    )
    save("source_catalog.json", source_catalog)
    save("ordinary_example_catalog.json", ordinary_store)
    save(
        "context_examples.json",
        {"/".join(map(str, key)): value for key, value in sorted(cases.items())},
    )
    save(
        "execution_binding.json",
        dict(
            status="every_frozen_prompt_linked_to_three_audited_actual_results",
            distinct_prompts=len(prompt_index),
            prediction_bindings=len(seen),
            by_condition=dict(bindings_by_condition),
            source_assignments=len(deliveries),
            source_prediction_slots=len(deliveries) * 3,
            source_ordinary_initial_vote_slots=len(deliveries) * 9,
            system_message_variants=len(system_hashes),
            first_pass_reference_contexts_matched=matched,
            delivery_totals=dict(pool_totals),
            missing_aggregates=sum(nulls.values()),
            valid_votes=sum(valids.values()),
            basis="Part 1 bound audited-result assembly; raw audits reused. Ordinary initial vote-slot exposure excludes repair/retry amplification, which belongs to resources.",
            new_Qwen_calls=0,
        ),
    )
    save(
        "pool_inventory.json",
        dict(
            pools=pools,
            unique_sources=len(members),
            membership_counts=dict(Counter(members.values())),
            pairwise=[
                dict(
                    first=p["condition"],
                    second=q["condition"],
                    intersection=sorted(set(p["ids"]) & set(q["ids"])),
                    union_n=len(set(p["ids"]) | set(q["ids"])),
                )
                for i, p in enumerate(pools)
                for q in pools[i + 1 :]
            ],
            triple_intersection=sorted(
                set.intersection(*(set(p["ids"]) for p in pools))
            ),
            ordinary_examples_used=len(ordinary_store),
            ordinary_occurrences=dict(ordinary_usage),
        ),
    )
    artifacts = {
        a.rel(path): a.sha(path) for path in sorted(OUT.iterdir()) if path.is_file()
    }
    save(
        "assembled.json",
        dict(
            status="part5_context_delivery_assembled_independent_verification_pending",
            created_utc=a.utc(),
            input_bindings=contract["input_bindings"],
            analysis_sources=contract["analysis_sources"],
            artifacts=artifacts,
            queries=N,
            assignments=len(deliveries),
            context_examples=len(cases),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "All saved contexts, inserted corrections, displaced neighbours and execution bindings assembled.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        build()
