"""Policy reader."""

import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    CUDA_VISIBLE_DEVICES="",
)
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import yaml
from analysis.knowledge.policy import common as c
from analysis.knowledge.german_guidelines import reader as old
from analysis.common import prompt_archive as tb

BASE = c.ROOT / "experiments/astra/complete_policy_part3_20261115_v1"
LIVE = c.OUTPUT / "live"
PLAN_SHA = "4d38e2d42c6d13a19bc727911eb782e8f6eb7d3751260722c1d4316b67fb42b7"
FIELDS = ("hate", "hate_type", "target_group", "severity", "legal")


def save(path, value):
    c.require(c.normal(path).is_relative_to(BASE), "Write outside new Part 3 folder")
    c.save(path, value)


def checked(path, expected):
    data = c.a.storage_path(path).read_bytes()
    c.require(
        hashlib.sha256(data).hexdigest() == expected, "Changed artifact: " + c.ref(path)
    )
    return json.loads(data)


def freeze():
    path = BASE / "analysis_contract.json"
    if path.exists():
        return c.read(path)
    c.require(c.sha(c.PLAN) == PLAN_SHA, "Frozen plan differs")
    plan = c.read(c.PLAN)
    bound = {}
    for field in ("source_hashes", "prepared_hashes"):
        for name, expected in plan[field].items():
            c.require(
                c.sha(c.ROOT / name) == expected, "Changed execution source: " + name
            )
            bound[name] = expected
    for name, expected in plan["input_hashes"].items():
        if name.startswith(c.ref(c.SETUP) + "/") or name == "config/taxonomy.yaml":
            c.require(c.sha(c.ROOT / name) == expected, "Changed setup: " + name)
            bound[name] = expected
    bound.update(
        c.bindings(
            [
                c.PLAN,
                LIVE / "complete.json",
                c.ROOT
                / "Thesis_Log/astra_result/complete_policy_work_plan_20261114_v2.md",
                c.ROOT
                / "Thesis_Log/astra_result/complete_policy_part1_protocol_20261114_v1.md",
                Path(old.__file__),
                Path(tb.__file__),
            ]
        )
    )
    value = dict(
        status="frozen_before_performance_calculation",
        created_utc=c.a.utc(),
        original_bindings=bound,
        conditions=list(c.CONDITIONS),
        plan_sha256=PLAN_SHA,
        source_policy="Only new derived Part 3 artifacts; D and thesis are read-only.",
        mask="All nine final answers valid per frozen endpoint; eligible source strata precede this removal.",
        scoring="Compute each outer repeat, then mean three. Macro primary; micro same frozen multilabel scope.",
        exactness="Fine endpoints require hate=true; legal is independent. Missing is never an empty answer.",
        jaccard="Mean per text; empty/empty=1, with nonempty-gold companion. Closed gate has its delivered empty set.",
        repeatability="Exact endpoint state, including hate gate for fine endpoints; ignore rationale and label order.",
        bootstrap="50000 paired source-membership-stratified normalized-text-cluster draws; seed 20261114; frozen families.",
        limitations="Part 5 will reconcile strict-inner-valid, no-repair, own-valid and failure bounds. No final architecture recommendation yet.",
    )
    save(path, value)
    return value


def replay(value, parser):
    normalizations = normalized_votes = texts = 0
    for i, chain in enumerate(value["raw_runs"]):
        last = None
        for turn, raw in enumerate(chain):
            last, count, error = parser.parse_response(raw)
            normalizations += count
            texts += 1
            c.require((last is None) == bool(error), "Parser error state differs")
            c.require(
                last is None or turn == len(chain) - 1, "Repair after valid answer"
            )
        c.require(
            (last.model_dump() if last is not None else None)
            == value["run_predictions"][i],
            "Saved raw reply does not reproduce inner vote",
        )
        normalized_votes += bool(last is not None and last.gate_normalised)
    c.require(
        normalizations == value["normalisations"]
        and normalized_votes == value["gate_normalised"],
        "Parser normalization accounting changed",
    )
    return texts


def run():
    freeze()
    complete = checked(
        LIVE / "complete.json",
        c.read(BASE / "analysis_contract.json")["original_bindings"][
            c.ref(LIVE / "complete.json")
        ],
    )
    c.require(
        complete["status"] == "all_three_conditions_executed_and_audited"
        and complete["mode"] == "live"
        and (complete["plan_sha256"] == PLAN_SHA),
        "Run is incomplete",
    )
    schedule = c.read(c.BASE / "schedule.json")["stages"]
    roster = {r["key"]: r for r in c.rows(c.SETUP / "roster.jsonl")}
    c.require(
        len(roster) == 6906 and len(schedule) == 47, "Wrong roster or stage count"
    )
    taxonomy = yaml.safe_load(
        (c.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
    )
    parser = tb.bundle(c.ROOT / "config/taxonomy.yaml").parser
    all_rows, seen, bound, totals = ([], set(), {}, Counter())
    cell = {name: Counter() for name in c.CONDITIONS}
    reader_files = []
    implementation_seen = set()
    for stage in schedule:
        sid = stage["id"]
        rp, fp = (
            LIVE / "batches" / sid / "review.json",
            LIVE / "final_checks" / f"{sid}.json",
        )
        review = checked(rp, complete["review_hashes"][c.ref(rp)])
        final = checked(fp, complete["final_integrity_hashes"][c.ref(fp)])
        bound.update(
            {
                c.ref(rp): complete["review_hashes"][c.ref(rp)],
                c.ref(fp): complete["final_integrity_hashes"][c.ref(fp)],
            }
        )
        for obj in (review, final):
            c.require(
                obj["plan_sha256"] == PLAN_SHA
                and obj["stage_sha256"] == c.signature(stage),
                "Stage binding differs",
            )
        c.require(
            review["status"] == "raw_responses_and_votes_verified"
            and review["mode"] == "live"
            and (final["status"] == "all_item_files_rehashed"),
            "Unverified batch",
        )
        expected_stamps = {
            c.signature(old.identity(j, PLAN_SHA)) for j in stage["jobs"]
        }
        c.require(
            set(review["items"]) == set(final["item_receipts"]) == expected_stamps,
            "Batch matrix differs",
        )
        for name, expected in review.get("availability_hashes", {}).items():
            c.require(c.sha(c.ROOT / name) == expected, "Availability receipt differs")
            bound[name] = expected
        rows, counts = ([], Counter())
        for job in stage["jobs"]:
            identity = old.identity(job, PLAN_SHA)
            key = job["key"]
            name = job["condition"]
            repeat = job["repeat"]
            setting = (key, name, repeat)
            c.require(
                setting not in seen
                and key in roster
                and (name in c.CONDITIONS)
                and (repeat in range(3)),
                "Duplicate/unknown identity",
            )
            seen.add(setting)
            stamp = c.signature(identity)
            folder = LIVE / "items" / stamp
            receipt = checked(folder / "complete.json", final["item_receipts"][stamp])
            c.require(
                final["item_receipts"][stamp] == review["items"][stamp]["complete.json"]
                and receipt["status"] == "complete",
                "Item receipt differs",
            )
            saved_job = checked(
                folder / "job.json", receipt["hashes"][f"items/{stamp}/job.json"]
            )
            c.require(
                receipt["job_stamp"] == c.signature(saved_job)
                and saved_job["identity"] == identity
                and (saved_job["messages_sha256"] == job["messages_sha256"])
                and (saved_job["contract"] == c.a.CONTRACT)
                and (saved_job["execution_mode"] == "live"),
                "Complete job binding differs",
            )
            implementation_signature = c.signature(saved_job["implementation"])
            if implementation_signature not in implementation_seen:
                c.verify(saved_job["implementation"])
                implementation_seen.add(implementation_signature)
            result_hash = review["items"][stamp]["result.json"]
            c.require(
                receipt["hashes"][f"items/{stamp}/result.json"] == result_hash,
                "Result receipt differs",
            )
            value = checked(folder / "result.json", result_hash)
            c.require(
                value["identity"] == identity
                and value["messages_sha256"] == job["messages_sha256"],
                "Wrong prompt/result identity",
            )
            details = old.check_value(value, taxonomy)
            counts["raw_texts_reparsed"] += replay(value, parser)
            counts.update(value["accounting"])
            counts["predictions"] += 1
            counts["null_results"] += value["result"] is None
            cell[name].update(
                {
                    "predictions": 1,
                    "null_results": value["result"] is None,
                    "inner_valid_" + str(value["n_valid"]): 1,
                }
            )
            cell[name].update(details["flags"])
            row = dict(
                key=key,
                condition=name,
                repeat=repeat,
                stage=sid,
                stamp=stamp,
                result_sha256=result_hash,
                receipt_sha256=final["item_receipts"][stamp],
                result=(
                    None
                    if value["result"] is None
                    else {f: value["result"][f] for f in FIELDS}
                ),
                n_valid=value["n_valid"],
                repairs=value["repairs"],
                flags=details["flags"],
                duplicate_sensitive_fields=details["duplicate_sensitive_fields"],
                parse_failures=value["parse_failures"],
                gate_normalised=value["gate_normalised"],
            )
            rows.append(row)
        c.require(
            {k: counts[k] for k in review["counts"]} == review["counts"],
            "Batch accounting differs",
        )
        target = BASE / "reader" / f"{sid}.json"
        save(target, rows)
        reader_files.append(target)
        all_rows.extend(rows)
        totals.update(counts)
        print(
            f"{sid}: {len(rows)} identities, saved hashes, raw parser and independent voter verified",
            flush=True,
        )
    c.require(
        seen == {(k, n, r) for k in roster for n in c.CONDITIONS for r in range(3)},
        "Incomplete nine-setting matrix",
    )
    c.require(
        {k: totals[k] for k in complete["counts"]} == complete["counts"],
        "Global accounting differs",
    )
    coverage = {}
    index = {(r["key"], r["condition"], r["repeat"]): r for r in all_rows}
    for endpoint, meta in c.read(c.SETUP / "endpoints.json").items():
        keys = meta["keys"]
        excluded = [
            k
            for k in keys
            if any(
                (
                    index[k, n, r]["result"] is None
                    for n in c.CONDITIONS
                    for r in range(3)
                )
            )
        ]
        coverage[endpoint] = dict(
            eligible=len(keys),
            common_nine=len(keys) - len(excluded),
            excluded_keys=excluded,
        )
    report = dict(
        status="all_saved_results_verified_and_replayed",
        plan_sha256=PLAN_SHA,
        counts=dict(totals),
        per_condition={k: dict(v) for k, v in cell.items()},
        coverage=coverage,
        receipt_bindings=bound,
        reader_files=c.bindings(reader_files),
        reader_sources=c.bindings(
            [Path(__file__), Path(old.__file__), Path(tb.__file__)]
        ),
        audit_scope="Rehashed 62154 item receipts and result files, all 47 batch/final receipts; replayed every saved raw reply and independent vote. The already-complete full physical transport audit is preserved, not redundantly rerun.",
    )
    save(BASE / "reader_complete.json", report)
    print(
        json.dumps(
            {"status": report["status"], "coverage": coverage, "counts": dict(totals)},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with c.offline():
        run()
