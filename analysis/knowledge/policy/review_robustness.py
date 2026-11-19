from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
from analysis.knowledge.policy import robustness as p

c, m = (p.c, p.m)


def checked(path, expected):
    raw = c.a.storage_path(path).read_bytes()
    c.require(
        hashlib.sha256(raw).hexdigest() == expected,
        "Saved evidence changed: " + c.ref(path),
    )
    return json.loads(raw)


def process():
    roster, arrays, rows = p.load()
    stats = {n: Counter() for n in p.NAMES}
    models = Counter()
    finishes = Counter()
    errors = Counter()
    metadata = []
    nulls = []
    failed = []
    bindings = {}
    parser = p.reader.tb.bundle(c.ROOT / "config/taxonomy.yaml").parser
    reader_receipt = c.read(p.P3 / "reader_complete.json")
    for i, row in enumerate(rows):
        folder = c.OUTPUT / "live/items" / row["stamp"]
        raw = checked(folder / "result.json", row["result_sha256"])
        receipt = checked(folder / "complete.json", row["receipt_sha256"])
        c.require(
            raw["n_valid"] == row["n_valid"] and raw["repairs"] == row["repairs"],
            "Reader metadata differs",
        )
        cond = row["condition"]
        stats[cond]["predictions"] += 1
        for event in raw["events"]:
            if event["action"] == "ok":
                md = event["model_response_metadata"]
                models[str(md["reported_model"])] += 1
                finishes[str(md["finish_reason"])] += 1
            else:
                stats[cond]["transport_retry_" + event["response_class"]] += 1
        for vote, chain in enumerate(raw["raw_runs"]):
            if raw["run_predictions"][vote] is None:
                parsed, _, error = parser.parse_response(chain[-1])
                c.require(parsed is None and error, "Terminal failure no longer fails")
                errors[cond + "/" + error] += 1
                failed.append(
                    dict(
                        key=row["key"],
                        condition=cond,
                        repeat=row["repeat"],
                        vote=vote,
                        error=error,
                        terminal_text=chain[-1],
                        result_sha256=row["result_sha256"],
                    )
                )
        stats[cond].update(
            dict(
                valid_inner=raw["n_valid"],
                normalisations=raw["normalisations"],
                gate_normalised=raw["gate_normalised"],
            )
        )
        for flag in row["flags"]:
            stats[cond][flag] += 1
        metadata.append(
            dict(
                key=row["key"],
                condition=cond,
                repeat=row["repeat"],
                stage=row["stage"],
                created_utc=receipt["created_utc"],
                n_valid=row["n_valid"],
                repairs=row["repairs"],
                flags=row["flags"],
            )
        )
        if raw["result"] is None:
            nulls.append(
                dict(
                    key=row["key"],
                    condition=cond,
                    repeat=row["repeat"],
                    source=roster[row["key"]]["source"],
                    analysis_endpoints=roster[row["key"]]["analysis_endpoints"],
                    gate=roster[row["key"]]["gate"],
                    hate_types=roster[row["key"]]["hate_types"],
                    target_groups=roster[row["key"]]["target_groups"],
                    severity=roster[row["key"]]["severity"],
                    raw_runs=raw["raw_runs"],
                    messages_sha256=raw["messages_sha256"],
                    result_sha256=row["result_sha256"],
                )
            )
        if (i + 1) % 10000 == 0:
            print(f"Saved process metadata: {i + 1}/62154", flush=True)
    stages = []
    for stage in c.read(c.BASE / "schedule.json")["stages"]:
        name = stage["id"]
        review = c.read(c.OUTPUT / "live/batches" / name / "review.json")
        c.verify(review["availability_hashes"])
        bindings.update(review["availability_hashes"])
        paths = [
            c.ROOT / k
            for k in review["availability_hashes"]
            if k.endswith("/start.json")
        ]
        attempts = []
        for startpath in paths:
            start = c.read(startpath)
            summary = c.read(startpath.with_name("summary.json"))
            c.require(
                start["workers"] == 128
                and (not summary["errors"])
                and (not summary["stopped"])
                and (summary["unsubmitted"] == 0),
                "Execution did not finish cleanly",
            )
            attempts.append(dict(start=start, summary=summary))
        part = [r for r in metadata if r["stage"] == name]
        cells = Counter(((r["condition"], r["repeat"]) for r in part))
        expected = len(stage["jobs"]) // 9
        c.require(
            set(cells.values()) == {expected} and len(cells) == 9, "Unbalanced stage"
        )
        windows = {
            cond: dict(
                first=min((r["created_utc"] for r in part if r["condition"] == cond)),
                last=max((r["created_utc"] for r in part if r["condition"] == cond)),
            )
            for cond in p.NAMES
        }
        stages.append(
            dict(
                stage=name,
                inputs=expected,
                attempts=attempts,
                condition_completion_windows=windows,
                condition_windows_overlap=max((w["first"] for w in windows.values()))
                <= min((w["last"] for w in windows.values())),
            )
        )
    c.require(
        sum(models.values()) == reader_receipt["counts"]["http_success_with_text"],
        "Metadata success count differs",
    )
    p.rowsave("prediction_metadata.jsonl", metadata)
    p.rowsave("terminal_parse_failures.jsonl", failed)
    p.rowsave("null_answer_evidence.jsonl", nulls)
    p.save(
        "process_evidence.json",
        dict(
            status="saved_metadata_census_complete",
            per_condition=stats,
            reported_models=models,
            finish_reasons=finishes,
            terminal_errors=errors,
            missing_source_counts=dict(Counter((r["source"] for r in nulls))),
            missing_unique_keys=sorted({r["key"] for r in nulls}),
            stages=stages,
            bindings=bindings,
            inference_limits="Completion windows overlap but do not prove simultaneous kernel execution or a fixed server weight revision. No server/cache/GPU state was queried.",
        ),
    )


def history():
    roster, arrays, rows = p.load()
    contexts = {r["key"]: r for r in c.rows(p.P4 / "context_features.jsonl")}
    oldroster = {r["key"]: r for r in c.rows(p.OLD_ROSTER) if r["key"] in roster}
    fresh = {(r["key"], r["repeat"]): r for r in rows if r["condition"] == p.NAMES[0]}
    selected = []
    index = {}
    for r in c.rows(p.OLD / "predictions.jsonl"):
        if r["condition"] != "rag_full" or r["key"] not in roster:
            continue
        key = r["key"]
        new = roster[key]
        old = oldroster[key]
        c.require(
            r["messages_sha256"]
            == contexts[key]["settings"]["retrieved_policy"]["messages_sha256"],
            "Historical prompt mismatch",
        )
        for field in (
            "text",
            "phase",
            "gate",
            "hate_types",
            "target_groups",
            "severity",
            "legal",
            "text_cluster",
        ):
            c.require(
                new.get(field) == old.get(field),
                "Historical roster differs: " + key + "/" + field,
            )
        c.require(r["phase"] == new["phase"], "Historical phase mismatch")
        c.require((key, r["repeat"]) not in index, "Duplicate historical identity")
        index[key, r["repeat"]] = r
        selected.append(r)
    c.require(
        len(selected) == 20718 and set(index) == set(fresh),
        "Missing historical counterparts",
    )
    p.rowsave("historical_matched_rows.jsonl", selected)
    joint = {}
    for (key, rep), r in index.items():
        joint[key, "historical", rep] = r
        joint[key, "fresh", rep] = fresh[key, rep]
        joint[key, "duplicate", rep] = fresh[key, rep]
    outputs = {}
    for endpoint, meta in p.ENDPOINTS.items():
        ar = m.endpoint_arrays(
            meta, roster, joint, ("historical", "fresh", "duplicate")
        )
        mask = ar["common"]
        pt = m.point_metrics(meta, ar)
        st = ar["states"][:2, :, mask]
        old_valid = ar["valid"][0].all(axis=0)
        new_valid = ar["valid"][1].all(axis=0)
        metrics = ["macro_f1", "exact"] + (["micro_f1"] if "micro_f1" in pt else [])
        outputs[endpoint] = dict(
            n=int(mask.sum()),
            full_n=len(meta["keys"]),
            old_three_valid_n=int(old_valid.sum()),
            fresh_three_valid_n=int(new_valid.sum()),
            excluded_keys=[k for k, on in zip(meta["keys"], mask) if not on],
            scores={metric: pt[metric][:2] for metric in metrics},
            scores_repeats={metric: pt[metric + "_repeats"][:2] for metric in metrics},
            fresh_minus_historical={
                metric: pt[metric][1] - pt[metric][0] for metric in metrics
            },
            repeat_agreement=pt["repeatability"]["pairwise_agreement"][:2],
            all_nine_cross_period_disagreement=(st[1, :, None] != st[0, None, :]).mean(
                axis=2
            ),
            mean_cross_period_disagreement=float(
                (st[1, :, None] != st[0, None, :]).mean()
            ),
            common_six_sha256=c.signature(
                [k for k, on in zip(meta["keys"], mask) if on]
            ),
        )
        p.npzsave("historical/" + endpoint + "/encoded.npz", ar)
    binding_table = c.read(p.OLD / "item_bindings.json")
    samples = []
    seen = set()
    for endpoint, meta in p.ENDPOINTS.items():
        for rep in range(3):
            key = min(
                meta["keys"],
                key=lambda k: hashlib.sha256(
                    ("policy-part5-history|" + endpoint + "|" + k).encode()
                ).hexdigest(),
            )
            if (key, rep) in seen:
                continue
            seen.add((key, rep))
            r = index[key, rep]
            rawpath = Path(r["result_path"])
            receiptpath = rawpath.with_name("complete.json")
            raw = checked(rawpath, binding_table[c.ref(rawpath)])
            receipt = checked(receiptpath, binding_table[c.ref(receiptpath)])
            jobpath = rawpath.with_name("job.json")
            job = checked(
                jobpath,
                next(
                    (v for k, v in receipt["hashes"].items() if k.endswith("/job.json"))
                ),
            )
            c.require(
                raw["result"] == r["result"]
                and raw["messages_sha256"] == r["messages_sha256"],
                "Historical sample mismatch",
            )
            c.require(
                job["contract"] == c.a.CONTRACT, "Historical request contract differs"
            )
            f = fresh[key, rep]
            fp = c.OUTPUT / "live/items" / f["stamp"]
            freceipt = checked(fp / "complete.json", f["receipt_sha256"])
            fraw = checked(fp / "result.json", f["result_sha256"])
            fjob = checked(
                fp / "job.json",
                next(
                    (
                        v
                        for k, v in freceipt["hashes"].items()
                        if k.endswith("/job.json")
                    )
                ),
            )
            c.require(
                fjob["contract"] == job["contract"], "Fresh request contract differs"
            )
            samples.append(
                dict(
                    endpoint=endpoint,
                    key=key,
                    repeat=rep,
                    historical_utc=receipt["created_utc"],
                    fresh_utc=freceipt["created_utc"],
                    same_request_contract=True,
                    contract=job["contract"],
                    historical_model=raw["active_model"],
                    fresh_model=fraw["active_model"],
                    historical_reported_models=dict(
                        Counter(
                            (
                                str(e["model_response_metadata"]["reported_model"])
                                for e in raw["events"]
                                if e["action"] == "ok"
                            )
                        )
                    ),
                    bindings=c.bindings(
                        [
                            rawpath,
                            receiptpath,
                            jobpath,
                            fp / "result.json",
                            fp / "complete.json",
                            fp / "job.json",
                        ]
                    ),
                )
            )
    p.save(
        "historical_comparison.json",
        dict(
            status="exact_prompt_roster_pairs_verified",
            matched_predictions=20718,
            populations=outputs,
            raw_samples=samples,
            caution="Descriptive old/new baseline comparison only. Three repeat indices are not shared random seeds. No causal identification of server change versus sampling variation.",
        ),
    )
    print("Historical baseline comparison and 18 raw samples verified.", flush=True)


if __name__ == "__main__":
    import argparse

    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["process", "history"])
    args = a.parse_args()
    with c.offline():
        globals()[args.mode]()
