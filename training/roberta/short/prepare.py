import argparse
import math
from pathlib import Path
import numpy as np
from training.roberta.short import common as s
from training.roberta.short import audit as audit
from training.roberta import prepare as previous

c = s.c
BASE = s.BASE


def prepare():
    if (BASE / "draft.json").exists():
        validate_draft()
        print("Existing short-schedule draft verified.")
        return
    c.require(
        c.sha(s.OLD / "plan.json") == s.OLD_PLAN_SHA, "Wrong prior scientific plan"
    )
    with s.isolated(s.OLD, expanded=False):
        oldplan = previous.validate()
        prep = previous.validate_preparation()
    data = c.read(s.OLD / "data/train.json")
    inputs = c.read(s.OLD / "data/inference.json")
    oof, receipts, steps = audit.load_old(data)
    bindings = dict(oldplan["inputs"])
    bindings.update(
        s.hashes(
            [
                s.OLD / "plan.json",
                s.OLD / "selection.json",
                s.OLD / "complete.json",
                s.PROTOCOL,
                s.ARCHIVE / "roberta_weights_plan.json",
                s.ARCHIVE / "external_copy_verified.json",
                s.ARCHIVE / "post_offload_verification.json",
                s.ARCHIVE
                / "offload_complete_20261011T094415Z_3637066a701440a3ade913914b9180a5.json",
                c.ROOT
                / "experiments/astra/roberta_tuning_review_20261011_v1/verification.json",
            ]
        )
    )
    held_meta = []
    for job in s.original_search_jobs():
        rp, value, attempt = s.old_paths(job)
        bindings[c.rel(rp)] = c.sha(rp)
        for name, digest in value["products"].items():
            bindings[c.rel(attempt / name)] = digest
        held_meta.append(c.read(attempt / "training.json"))
    archive = s.archive_map()
    reused = []
    with s.isolated():
        for job in s.original_final_jobs():
            _, attempt, missing = s.verify_old_training(job, archive)
            meta = audit.metadata(job, data, attempt)
            rp, value, _ = s.old_paths(job)
            bindings[c.rel(rp)] = c.sha(rp)
            for name, digest in value["products"].items():
                target = attempt / name
                if c.rel(target) not in missing:
                    bindings[c.rel(target)] = digest
            ip = s.OLD / "inference" / job["id"] / "complete.json"
            s.compare_logits(
                ip,
                job,
                s.OLD_PLAN_SHA,
                c.sha(rp),
                inputs,
                data["heads"][job["head"]]["labels"],
            )
            bindings[c.rel(ip)] = c.sha(ip)
            bindings.update(c.read(ip)["artifacts"])
            reused.append(
                dict(
                    job=job,
                    training_receipt_sha256=c.sha(rp),
                    inference_receipt_sha256=c.sha(ip),
                    inference_logits=c.rel(ip.parent / "logits.npy"),
                    missing_archived_weight_files=missing,
                    actual_weight_rehash_today=False if missing else True,
                )
            )
    for name in ["train.json", "inference.json", "scopes.json", "fold_support.json"]:
        with s.isolated():
            c.publish_bytes(BASE / "data" / name, (s.OLD / "data" / name).read_bytes())
    with s.isolated(s.OLD, expanded=False):
        original_assembly = audit.old.assemble(
            c.read(s.OLD / "selection.json")["families"], data, s.OLD_PLAN_SHA, inputs
        )
    c.require(
        original_assembly == c.read(s.OLD / "prediction_audit.json"),
        "Original prediction audit no longer reproduces",
    )
    bindings.update(original_assembly["artifacts"])
    prepared = s.hashes(
        [
            BASE / "data" / name
            for name in [
                "train.json",
                "inference.json",
                "scopes.json",
                "fold_support.json",
            ]
        ]
    )
    work = {}
    estimate = 0.0
    total_steps = 0
    for head in c.HEADS:
        jobs = [j for j in s.search_jobs() if j["head"] == head]
        fit_rows = []
        for f in range(5):
            fit_rows.append(
                sum(
                    data["folds"]["42"][v["feature_row"]] != f
                    for v in data["heads"][head]["rows"]
                )
            )
        counts = [
            math.ceil(fit_rows[j["fold"]] / 16)
            * next(cfg["epochs"] for cfg in s.configs() if cfg["id"] == j["config"])
            for j in jobs
        ]
        prior = [x for x in held_meta if x["job"]["head"] == head]

        step_seconds = float(
            np.median([x["train_seconds"] / x["attempted_steps"] for x in prior])
        )
        overhead = float(
            np.median([x["total_seconds"] - x["train_seconds"] for x in prior])
        )
        seconds = sum(counts) * step_seconds + len(jobs) * overhead
        estimate += seconds
        total_steps += sum(counts)
        work[head] = dict(
            search_jobs=len(jobs),
            attempted_steps=sum(counts),
            fitting_rows_by_fold=fit_rows,
            prior_seconds_per_attempted_step_median=step_seconds,
            prior_overhead_median=overhead,
            rough_search_seconds=seconds,
        )
    c.require(
        len(s.search_jobs()) == 1800 and total_steps == 51678 and len(reused) == 60,
        "Unexpected workload",
    )
    new = dict(
        status="prepared_no_new_scientific_fits",
        utc=c.utc(),
        environment=c.environment(),
        inputs=bindings,
        prepared=prepared,
        protocol_sha256=c.sha(s.PROTOCOL),
        base_cache=prep["base_cache"],
        old_plan_sha256=s.OLD_PLAN_SHA,
        old_search_fits=720,
        new_search_fits=1800,
        union_search_fits=2520,
        old_attempted_steps=steps,
        new_attempted_steps=total_steps,
        configurations=s.configs(),
        new_jobs=s.search_jobs(),
        reusable_finals=reused,
        head_work=work,
        estimated_search_hours_from_prior_runs=estimate / 3600,
        estimate_limits="Prior-run estimate only; 1800 fresh model initializations, per-epoch validation, final fits/inference, hashing and concurrent load can change elapsed time. No runtime promise.",
        max_new_final_models=20,
        max_inference_inputs_per_new_model=len(inputs),
        resources=dict(
            initial_free_disk_GiB=28,
            per_job_free_disk_GiB=12,
            minimum_available_RAM_GiB=3.5,
            free_CUDA_GiB=5,
            maximum_new_weight_GiB=20 * 498628204 / 2**30,
            pilot_weight_allowance_GiB=1.5,
            other_artifact_allowance_GiB=3,
            CV_weights_retained=False,
            per_epoch_held_logits_retained=True,
        ),
        archive_policy="Prior verified copy manifests and all local prediction/receipt chunks checked. F: need not be connected for this saved-prediction reuse; physical archived weights are not rehashed or restored here.",
        original_prediction_audit_reproduced=True,
        evaluation_labels_used_for_selection=False,
        Qwen_calls=0,
        new_scientific_fits=0,
    )
    s.save(BASE / "draft.json", new)
    validate_draft()
    print(
        "Prepared 1800 fresh searches, verified 720 older searches and 60 reusable final prediction jobs.",
        flush=True,
    )
    print(
        c.encoded(
            dict(
                new_steps=total_steps,
                rough_search_hours=estimate / 3600,
                max_new_final_checkpoints=20,
                initial_disk_GiB=28,
            )
        ).decode(),
        flush=True,
    )


def validate_draft():
    draft = c.read(BASE / "draft.json")
    c.verify(draft["inputs"])
    c.verify(draft["prepared"])
    c.require(draft["environment"] == c.environment(), "Package environment changed")
    c.require(
        draft["configurations"] == s.configs() and draft["new_jobs"] == s.search_jobs(),
        "Candidate grid changed",
    )
    for name, digest in draft["base_cache"]["files"].items():
        c.require(
            c.sha(Path(draft["base_cache"]["snapshot"]) / name) == digest,
            "Pretrained cache changed",
        )
    return draft


def seal():
    draft = validate_draft()
    sources = s.hashes(s.source_paths())
    test_path = s.development_receipt("tests")
    pilot_path = s.development_receipt("pilot")
    test = c.read(test_path)
    pilot = c.read(pilot_path)
    c.require(
        test["passed"] and test["sources"] == sources, "Synthetic tests not current"
    )
    c.require(
        pilot["passed"]
        and pilot["sources"] == sources
        and pilot["excluded_from_scientific_results"],
        "CUDA pilot not current",
    )
    c.verify(pilot["artifacts"])
    s.save(
        BASE / "plan.json",
        dict(
            draft,
            status="sealed_ready_no_scientific_fits",
            inputs={
                **draft["inputs"],
                **sources,
                **s.hashes([BASE / "draft.json", test_path, pilot_path]),
            },
            primary=dict(
                left="expanded_tuned_selected",
                right="original_20_60_tuned_selected",
                endpoints=[
                    "en_test/hate",
                    "en_test/hate_type",
                    "larger_dev/target_group",
                    "larger_dev/severity",
                ],
                seeds=list(c.FINAL_SEEDS),
                bootstrap_draws=10000,
                confidence=0.9875,
                exposure="Previously exposed evaluation; follow-up not untouched confirmation",
            ),
            numerical_pilot=pilot["summary"],
        ),
    )
    validate()
    print(
        "Shorter-schedule package sealed; no new scientific fitting started.",
        flush=True,
    )


def validate():
    plan = c.read(BASE / "plan.json")
    c.verify(plan["inputs"])
    c.verify(plan["prepared"])
    c.require(
        plan["environment"] == c.environment()
        and plan["new_jobs"] == s.search_jobs()
        and plan["configurations"] == s.configs(),
        "Sealed environment or grid changed",
    )
    for name, digest in plan["base_cache"]["files"].items():
        c.require(
            c.sha(Path(plan["base_cache"]["snapshot"]) / name) == digest,
            "Pinned pretrained model changed",
        )
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "seal", "validate"])
    args = parser.parse_args()
    with s.isolated():
        globals()[args.mode]()
