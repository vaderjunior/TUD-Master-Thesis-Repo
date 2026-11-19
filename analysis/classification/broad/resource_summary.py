from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import argparse
import math
import numpy as np
from analysis.classification.broad import resources as x

METRICS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "repair_prompt_tokens",
    "repair_completion_tokens",
    "repair_total_tokens",
    "requests",
    "repairs",
    "retries",
    "request_seconds",
    "backoff_seconds",
    "n_valid",
    "raw_chars",
    "raw_utf8_bytes",
    "rationale_chars",
    "rationale_utf8_bytes",
    "ordinary_seconds",
    "repair_seconds",
    "success_seconds",
    "failed_seconds",
)


def distribution(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(
            n=0,
            sum=0.0,
            mean=None,
            min=None,
            p50=None,
            p90=None,
            p95=None,
            p99=None,
            max=None,
        )
    qq = np.quantile(a, [0.5, 0.9, 0.95, 0.99])
    return dict(
        n=len(a),
        sum=float(a.sum()),
        mean=float(a.mean()),
        min=float(a.min()),
        p50=float(qq[0]),
        p90=float(qq[1]),
        p95=float(qq[2]),
        p99=float(qq[3]),
        max=float(a.max()),
    )


def context():
    base = x.c.BASE / "retrieval_v3"
    done = x.c.read(base / "complete.json")
    paths = [
        base / "complete.json",
        base / "plan.json",
        x.ROOT / "analysis/common/cached_retrieval.py",
    ]
    passes = []
    for name, brief in done["passes"].items():
        p = base / name / "complete.json"
        receipt = x.c.read(p)
        paths.append(p)
        x.require(
            x.c.sha(p) == done["outputs"][x.c.rel(p)], "Bound retrieval pass changed"
        )
        x.require(
            receipt["seconds"] == brief["seconds_recorded"]
            and receipt["rows"] == brief["rows"],
            "Retrieval summary differs",
        )
        reused = receipt["checkpoint_resume"]["reused_rows"]
        x.require(reused == brief["checkpoint_reused_rows"], "Resume summary differs")
        passes.append(
            dict(
                pass_name=name,
                rows=receipt["rows"],
                reused_rows=reused,
                new_rows_in_finishing_invocation=receipt["rows"] - reused,
                seconds=receipt["seconds"],
                mean_recorded_seconds_per_roster_input=receipt["seconds"]
                / receipt["rows"],
                complete_pass_timer=reused == 0,
                boundary="Timer begins after index copy/load/vector verification and existing prompt loading; covers retrieval/rendering/chunk gathering, including checkpoint work. Ends before final JSONL serialization. Resumed pass does not recover earlier invocation time.",
            )
        )
    vector_receipt = x.c.read(base / "vectors/complete.json")
    old = x.c.read(x.ROOT / vector_receipt["original_receipt"])
    paths += [
        base / "vectors/complete.json",
        x.ROOT / vector_receipt["original_receipt"],
    ]
    x.require(
        x.c.sha(x.ROOT / vector_receipt["original_receipt"])
        == vector_receipt["original_receipt_sha256"],
        "Vector origin changed",
    )
    empty = x.c.read(base / "vectors/empty_embedding.json")
    paths.append(base / "vectors/empty_embedding.json")
    kb = []
    files = []
    plan = x.c.read(base / "plan.json")
    for pool in ("full", "small"):
        record_path = x.c.CORE / f"kb/{pool}/records.jsonl"
        records = list(x.reader.rows(record_path))
        counts = Counter((a["kind"] for a in records))
        prefix = x.c.rel(x.c.CORE / f"retrieval_v1/{pool}/chroma") + "/"
        indexed = [
            (x.ROOT / k, v)
            for k, v in plan["original_index_hashes"].items()
            if k.startswith(prefix)
        ]
        for p, h in indexed:
            x.require(x.c.sha(p) == h, "Preserved physical index changed")
            files.append(
                dict(
                    category=pool + "_index",
                    path=x.c.rel(p),
                    bytes=p.stat().st_size,
                    sha256=h,
                )
            )
            paths.append(p)
        paths.append(record_path)
        files.append(
            dict(
                category=pool + "_records",
                path=x.c.rel(record_path),
                bytes=record_path.stat().st_size,
                sha256=x.c.sha(record_path),
            )
        )
        kb.append(
            dict(
                pool=pool,
                records=len(records),
                kinds=dict(counts),
                record_bytes=record_path.stat().st_size,
                index_files=len(indexed),
                physical_index_bytes=sum((p.stat().st_size for p, _ in indexed)),
                record_plus_one_index_bytes=record_path.stat().st_size
                + sum((p.stat().st_size for p, _ in indexed)),
                excludes="Model weights, shared embedding caches, separate FP32 vectors, prompts, duplicate index copies, audits and raw HTTP evidence.",
            )
        )
    for category, p in [
        ("full_record_vectors", x.c.CORE / "retrieval_v1/full_vectors.npy"),
        ("broad_query_vectors", base / "vectors/query_vectors.npy"),
        ("full_prompts", base / "full/prompts.jsonl"),
        ("small_prompts", base / "small/prompts.jsonl"),
        ("zero_prompts", base / "zero_shot.jsonl"),
    ]:
        item = dict(
            category=category,
            path=x.c.rel(p),
            bytes=p.stat().st_size,
            sha256=x.c.sha(p),
        )
        if p.suffix == ".npy":
            a = np.load(p, allow_pickle=False, mmap_mode="r")
            item.update(shape=list(a.shape), dtype=str(a.dtype), numeric_bytes=a.nbytes)
            del a
        files.append(item)
        paths.append(p)
    contexts = (
        x.ROOT
        / "experiments/astra/sq2_broad_context_review_20261104_v1/prompt_descriptors.jsonl"
    )
    pc = defaultdict(list)
    pbytes = defaultdict(list)
    cc = defaultdict(list)
    for row in x.reader.rows(contexts):
        for group in (row["condition"], row["phase"] + "/" + row["condition"]):
            pc[group].append(row["message_chars"])
            pbytes[group].append(row["message_utf8_bytes"])
            cc[group].append(row["context_chars"])
    descriptions = [
        dict(
            group=k,
            message_chars=distribution(pc[k]),
            message_utf8_bytes=distribution(pbytes[k]),
            context_chars=distribution(cc[k]),
        )
        for k in sorted(pc)
    ]
    paths.append(contexts)
    x.save(
        x.BASE / "preparation_and_storage.json",
        dict(
            status="preserved_preparation_and_specific_storage_reviewed",
            retrieval_passes=passes,
            kb=kb,
            files=files,
            prompt_sizes=descriptions,
            vectors=dict(
                rows=vector_receipt["rows"],
                cache_rows=vector_receipt["cache_rows"],
                new_embedding_texts_final_v3=vector_receipt["new_embedding_texts"],
                new_embedding_texts_predecessor=old["new_embedding_texts"],
                whitespace_device=empty["device"],
                whitespace_text=empty["text"],
                embedding_model=plan["embedding"],
            ),
            limitations=[
                "Retained source sizes are logical byte sizes, not physical disk allocation or peak RAM. No full raw-output-tree size inventory is repeated.",
                "Index files are retained historical physical snapshots. They are not a minimum fresh-build index-size guarantee.",
                "Cached vector reuse avoids new embedding here but does not make original BGE embedding/index construction free.",
                "No complete human annotation/KB construction timer or matched online retrieval latency is recorded.",
                "The small replay1 finishing invocation reused 9300 rows, so its timer is not a whole-pass compute measurement.",
                "No deployment price, energy, carbon, tokenizer estimate or GPU utilization is inferred.",
            ],
            sources=x.bindings(paths),
        ),
    )
    print("Saved preparation, prompt-size and bounded storage review.", flush=True)


def summarize():
    receipt = x.c.read(x.BASE / "extraction_complete.json")
    x.r.verify(receipt["outputs"])
    ev = np.load(x.BASE / "events.npz", allow_pickle=False)["values"]
    identifiers = []
    metrics = []
    group_ids = defaultdict(list)
    classes = defaultdict(Counter)
    meta_by_identity = {}
    for row in x.reader.rows(x.BASE / "records.jsonl"):
        i = row["index"]
        x.require(i == len(metrics), "Record index differs")
        identifiers.append(
            {
                k: row[k]
                for k in (
                    "key",
                    "phase",
                    "source",
                    "condition",
                    "repeat",
                    "stage",
                    "n_valid",
                )
            }
        )
        metrics.append(
            [
                *[row["usage"][k] for k in x.FIELDS],
                *[row["repair_usage"][k] for k in x.FIELDS],
                row["accounting"]["post_attempts"],
                row["repairs"],
                row["accounting"]["retry_attempts"],
                row["request_seconds"],
                row["backoff_seconds"],
                row["n_valid"],
                row["success_raw_chars"],
                row["success_raw_utf8_bytes"],
                (
                    np.nan
                    if row["delivered_rationale_chars"] is None
                    else row["delivered_rationale_chars"]
                ),
                (
                    np.nan
                    if row["delivered_rationale_utf8_bytes"] is None
                    else row["delivered_rationale_utf8_bytes"]
                ),
                *[
                    row["time_components"].get(k, 0.0)
                    for k in (
                        "ordinary_seconds",
                        "repair_seconds",
                        "success_seconds",
                        "failed_seconds",
                    )
                ],
            ]
        )
        groups = [
            "all",
            "condition/" + row["condition"],
            "phase/" + row["phase"] + "/" + row["condition"],
            "repeat/"
            + row["phase"]
            + "/"
            + row["condition"]
            + "/"
            + str(row["repeat"]),
            "source/" + row["source"] + "/" + row["condition"],
            "stage/" + row["stage"],
            "validity/" + row["condition"] + "/" + str(row["n_valid"]),
        ]
        for group in groups:
            group_ids[group].append(i)
            classes[group].update(row["response_classes"])
        meta_by_identity[row["key"], row["condition"], row["repeat"]] = i
    a = np.array(metrics, float)
    record_indexes = ev[:, 0].astype(int)
    require = x.require
    x.save(x.BASE / "record_index.json", dict(columns=METRICS, records=identifiers))
    with (x.BASE / "record_metrics.npz").open("xb") as f:
        np.savez_compressed(f, values=a, columns=np.asarray(METRICS))
    groups = []
    for name, ids in sorted(group_ids.items()):
        m = np.zeros(len(a), bool)
        m[ids] = True
        e = ev[m[record_indexes]]
        v = a[ids]
        ok = e[:, 4] == 1
        repair = e[:, 2] > 0
        item = dict(
            group=name,
            predictions=len(ids),
            valid_predictions=int((v[:, 11] > 0).sum()),
            null_predictions=int((v[:, 11] == 0).sum()),
            partial_predictions=int(((v[:, 11] > 0) & (v[:, 11] < 3)).sum()),
            repaired_predictions=int((v[:, 7] > 0).sum()),
            response_classes=dict(classes[name]),
            metrics={
                k: distribution(v[:, j])
                for j, k in enumerate(METRICS)
                if k != "n_valid"
            },
            event_slices={},
        )
        for label, mask in [
            ("all", np.ones(len(e), bool)),
            ("ordinary", ~repair),
            ("repair", repair),
            ("successful", ok),
            ("failed", ~ok),
            ("ordinary_successful", ok & ~repair),
            ("repair_successful", ok & repair),
        ]:
            z = e[mask]
            success = z[:, 4] == 1
            item["event_slices"][label] = dict(
                attempts=len(z),
                successes=int(success.sum()),
                request_seconds=distribution(z[:, 5]),
                backoff_seconds=distribution(z[:, 6]),
                prompt_tokens=distribution(z[success, 7]),
                completion_tokens=distribution(z[success, 8]),
                total_tokens=distribution(z[success, 9]),
                raw_chars=distribution(z[success, 10]),
                raw_utf8_bytes=distribution(z[success, 11]),
            )
        item["mean_tokens_per_valid_output_with_all_attempt_cost"] = (
            v[:, 2].sum() / item["valid_predictions"]
            if item["valid_predictions"]
            else None
        )
        item["repair_token_share"] = v[:, 5].sum() / v[:, 2].sum()
        item["retry_attempt_share"] = v[:, 8].sum() / v[:, 6].sum()
        groups.append(item)
    inventory = x.c.read(x.BASE / "execution_inventory.json")
    batches = inventory["stages"]
    time_rows = []
    for j, row in enumerate(batches):
        start = row["attempts"][0]["start"]
        end = row["attempts"][-1]["end"]
        complete = row["generation_complete"]
        audit = row["audit_complete"]
        active = sum((z["seconds"] for z in row["attempts"]))
        span = x.seconds(start, end)
        post = x.seconds(end, complete)
        audit_envelope = x.seconds(complete, audit)
        nextstart = (
            batches[j + 1]["attempts"][0]["start"] if j + 1 < len(batches) else None
        )
        gap = x.seconds(audit, nextstart) if nextstart else None
        require(
            abs(x.seconds(start, complete) - row["audit_report_wall_seconds"]) < 1e-05,
            "Audit wall definition differs",
        )
        time_rows.append(
            dict(
                stage=row["stage"],
                start_utc=start,
                summary_utc=end,
                complete_utc=complete,
                audit_utc=audit,
                generation_attempt_seconds=active,
                between_attempt_seconds=span - active,
                post_summary_commit_seconds=post,
                complete_to_audit_seconds=audit_envelope,
                audit_to_next_start_seconds=gap,
                predictions=row["counts"]["predictions"],
                requests=row["accounting"]["post_attempts"],
                tokens=row["usage"]["total_tokens"],
                retries=row["accounting"]["retry_attempts"],
                model_availability_checks=len(row["attempts"]),
                model_check_seconds=sum(
                    (z["model_check_seconds"] for z in row["attempts"])
                ),
                peak_workers=max((z["peak_workers"] for z in row["attempts"])),
                completed_predictions_per_attempt_second=row["counts"]["predictions"]
                / active,
                reported_tokens_per_attempt_second=row["usage"]["total_tokens"]
                / active,
                summed_vote_seconds_per_attempt_second=row["vote_seconds"] / active,
            )
        )
    timing = dict(
        first_start_utc=time_rows[0]["start_utc"],
        last_summary_utc=time_rows[-1]["summary_utc"],
        last_stage_audit_utc=time_rows[-1]["audit_utc"],
        first_to_last_summary_seconds=x.seconds(
            time_rows[0]["start_utc"], time_rows[-1]["summary_utc"]
        ),
        first_to_last_stage_audit_seconds=x.seconds(
            time_rows[0]["start_utc"], time_rows[-1]["audit_utc"]
        ),
        generation_attempt_seconds=sum(
            (z["generation_attempt_seconds"] for z in time_rows)
        ),
        between_attempt_seconds=sum((z["between_attempt_seconds"] for z in time_rows)),
        post_summary_commit_seconds=sum(
            (z["post_summary_commit_seconds"] for z in time_rows)
        ),
        complete_to_audit_seconds=sum(
            (z["complete_to_audit_seconds"] for z in time_rows)
        ),
        audit_to_next_start_seconds=sum(
            (z["audit_to_next_start_seconds"] or 0 for z in time_rows)
        ),
        model_check_seconds=sum((z["model_check_seconds"] for z in time_rows)),
        final_completion_mtime_utc=inventory["final_completion_mtime_utc"],
        finalization_mtime_proxy_seconds=x.seconds(
            time_rows[-1]["audit_utc"], inventory["final_completion_mtime_utc"]
        ),
        total_mtime_proxy_seconds=x.seconds(
            time_rows[0]["start_utc"], inventory["final_completion_mtime_utc"]
        ),
        batches=time_rows,
    )
    require(
        abs(
            sum(
                (
                    timing[k]
                    for k in (
                        "generation_attempt_seconds",
                        "between_attempt_seconds",
                        "post_summary_commit_seconds",
                        "complete_to_audit_seconds",
                        "audit_to_next_start_seconds",
                    )
                )
            )
            - timing["first_to_last_stage_audit_seconds"]
        )
        < 1e-06,
        "Timeline partition differs",
    )
    quality = []
    paths = []
    for name in x.c.read(x.reader.BASE / "endpoint_setup.json"):
        old, terms = x.prior.endpoint(name)
        settings = [(co, rep) for co in x.CONDITIONS for rep in range(3)]
        common = np.stack([terms[k]["valid"] for k in settings]).all(0)
        ids = np.array(terms[settings[0]]["ids"])[common]
        require(
            len(ids) == old["three_condition_common"]["n_common_valid"],
            "Quality population differs",
        )
        paths.append(
            x.prior.main.BASE / "endpoints" / name.replace("/", "__") / "results.json"
        )
        for co in x.CONDITIONS:
            ix = [meta_by_identity[key, co, rep] for key in ids for rep in range(3)]
            values = a[ix]
            prior_score = old["three_condition_common"]["per_condition"][co]
            quality.append(
                dict(
                    endpoint=name,
                    condition=co,
                    n_common_inputs=len(ids),
                    predictions=len(ix),
                    mean_macro_f1=prior_score["mean_macro_f1"],
                    repeat_macro_f1=prior_score["repeat_macro_f1"],
                    mean_reported_tokens_per_prediction=float(values[:, 2].mean()),
                    mean_prompt_tokens_per_prediction=float(values[:, 0].mean()),
                    mean_completion_tokens_per_prediction=float(values[:, 1].mean()),
                    mean_request_seconds_per_prediction=float(values[:, 9].mean()),
                    mean_backoff_seconds_per_prediction=float(values[:, 10].mean()),
                    mean_repairs_per_prediction=float(values[:, 7].mean()),
                    note="Original descriptive all-nine common-valid quality population; includes all generated fields and all repairs. Per-head cost is not identifiable. These endpoint costs overlap and must not be added.",
                )
            )
    x.save(
        x.BASE / "resource_summary.json",
        dict(
            groups=groups,
            timing=timing,
            quality_cost=quality,
            limitations=[
                "All distributions describe this fixed completed workload; no population intervals or service-level guarantees.",
                "Timers overlap across 128 workers. Requests include queue/network/provider/client time, not pure GPU compute. Three votes are sequential per prediction.",
                "Recorded successful-response usage is not complete billing/compute: timed-out requests may have finished remotely, with no reported tokens.",
                "Stage start includes a GET model availability check and local preparation. Summary-to-complete and complete-to-audit are envelopes, not pure commit/audit CPU timers.",
                "The whole-run completion file mtime is only a proxy; no exact finalization end timestamp was saved.",
                "The generation run uses saved retrieval prompts; it is not an end-to-end online latency benchmark. No standalone per-condition wall time or causal timing comparison is available.",
            ],
            sources=x.bindings(
                [
                    Path(__file__),
                    x.BASE / "extraction_complete.json",
                    x.BASE / "execution_inventory.json",
                    *paths,
                ]
            ),
        ),
    )
    print(
        "Resource groups:",
        len(groups),
        "; aligned cost/quality rows:",
        len(quality),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["context", "summarize"], required=True)
    args = parser.parse_args()
    with x.r.original.offline():
        context() if args.mode == "context" else summarize()
