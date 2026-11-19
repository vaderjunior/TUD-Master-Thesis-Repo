import argparse
import math
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
from training.roberta.short import common as s
from training.roberta.short import prepare as preparation
from training.roberta.short import train as train
from training.roberta.short import audit as audit
from training.roberta import train as original_train

c = s.c
BASE = s.BASE


def test():
    path = s.development_receipt("tests")
    sources = s.hashes(s.source_paths())
    if path.exists():
        c.require(c.read(path)["sources"] == sources, "Test source changed")
        print("Current tests already passed.")
        return
    c.require(not (BASE / "plan.json").exists(), "Cannot replace tests after seal")
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/training/roberta/short/test_training.py",
        ],
        cwd=c.ROOT,
        check=False,
    )
    c.require(result.returncode == 0, "Short-schedule tests failed")
    s.save(
        path,
        dict(
            passed=True,
            sources=sources,
            command="pytest -q -p no:cacheprovider tests/training/roberta/short/test_training.py",
            Qwen_calls=0,
            scientific_fits=0,
            CUDA_used=False,
        ),
    )


def pilot():
    draft = preparation.validate_draft()
    sources = s.hashes(s.source_paths())
    done = s.development_receipt("pilot")
    if done.exists():
        c.require(c.read(done)["sources"] == sources, "Pilot source changed")
        c.verify(c.read(done)["artifacts"])
        print("Current technical pilot already verified.")
        return
    c.require(not (BASE / "plan.json").exists(), "Cannot change pilot after seal")
    s.resource.resource_guard(3.5, 28)
    train.setup()
    folder = BASE / "development/pilots" / uuid.uuid4().hex[:12]
    data = s.read_data()
    values = []
    with s.isolated():
        for head in c.HEADS:
            for loss in ["unweighted", "balanced"]:
                job = dict(
                    id=f"pilot_{head}_{loss}",
                    mode="search",
                    head=head,
                    config=f"l1_e1_{loss[0]}",
                    seed=987,
                    fold=0,
                )
                directory = folder / job["id"]
                directory.mkdir(parents=True)
                c.save(
                    directory / "started.json",
                    dict(
                        job=job,
                        technical_only=True,
                        sources=sources,
                        draft_sha256=c.sha(BASE / "draft.json"),
                    ),
                )
                print("Excluded one-epoch CUDA pilot:", head, loss, flush=True)
                meta = train.train_attempt(
                    directory,
                    job,
                    data,
                    draft["base_cache"]["snapshot"],
                    pilot=True,
                    save_model=head == "hate_type" and loss == "balanced",
                )
                _, fit, held = train.split(data, job)
                c.require(
                    meta["attempted_steps"] == min(8, math.ceil(len(fit) / 16))
                    and meta["pilot_only"]
                    and meta["successful_steps"] > 0,
                    "Short technical pilot incomplete",
                )
                c.require(
                    np.array_equal(
                        np.load(directory / "epoch_logits/001.npy"),
                        np.load(directory / "held_logits.npy"),
                    ),
                    "Short epoch capture differs",
                )
                values.append(meta)

        job = dict(
            id="pilot_parity_type",
            mode="search",
            head="hate_type",
            config="l1_e5_b",
            seed=988,
            fold=0,
        )
        pair = []
        for label, engine in [("original", original_train), ("new", train)]:
            directory = folder / "parity" / label
            directory.mkdir(parents=True)
            print("Excluded CUDA numerical parity:", label, flush=True)
            pair.append(
                engine.train_attempt(
                    directory,
                    job,
                    data,
                    draft["base_cache"]["snapshot"],
                    pilot=True,
                    save_model=False,
                )
            )
        for field in [
            "job",
            "configuration",
            "fit_ids",
            "held_ids",
            "class_weights",
            "planned_steps",
            "attempted_steps",
            "successful_steps",
            "skipped_steps",
            "steps",
            "epochs",
            "changes",
        ]:
            c.require(
                pair[0][field] == pair[1][field],
                "Training derivative changes numerical procedure: " + field,
            )
        difference = 0.0
        for name in ["held_logits.npy", "probe_logits.npy"]:
            left = np.load(folder / "parity/original" / name)
            right = np.load(folder / "parity/new" / name)
            error = float(np.max(np.abs(left - right)))
            difference = max(difference, error)
            c.require(
                np.array_equal(left, right), "Old/new trainer pilot logits differ"
            )
    artifacts = s.hashes([p for p in folder.rglob("*") if p.is_file()])
    summary = dict(
        excluded_fits=10,
        one_epoch_head_loss_probes=8,
        parity_fits=2,
        attempted_steps=sum(x["attempted_steps"] for x in values + pair),
        skipped_steps=sum(x["skipped_steps"] for x in values + pair),
        max_peak_allocated_GiB=max(x["peak_allocated_gib"] for x in values + pair),
        numerical_parity_logit_max_abs_difference=difference,
        old_new_training_trace_equal=True,
        CPU_export_checked=True,
        gpu=values[0]["gpu"],
    )
    s.save(
        done,
        dict(
            passed=True,
            excluded_from_scientific_results=True,
            sources=sources,
            artifacts=artifacts,
            summary=summary,
            draft_sha256=c.sha(BASE / "draft.json"),
            Qwen_calls=0,
        ),
    )
    print(c.encoded(summary).decode(), flush=True)


def run(stage="all"):
    plan = preparation.validate()
    plan_sha = c.sha(BASE / "plan.json")
    data = s.read_data()
    with s.isolated(), s.locks():
        done = BASE / "complete.json"
        if done.exists():
            rec = c.read(done)
            c.require(rec["plan_sha256"] == plan_sha, "Different final plan")
            c.verify(rec["outputs"])
            selected = audit.select(data, plan_sha)
            final = audit.final_check(data, selected, plan_sha)
            decoded = audit.assemble(data, selected, plan_sha)
            c.require(
                final == c.read(BASE / "final_audit.json")
                and decoded == c.read(BASE / "prediction_audit.json"),
                "Completed audit changed",
            )
            print(
                "VERIFIED COMPLETE. No new training or inference. Tell Astra to inspect.",
                flush=True,
            )
            return
        started = BASE / "execution_started.json"
        s.resource.resource_guard(3.5, 28 if not started.exists() else 15)
        train.setup()
        if not started.exists():
            s.save(
                started,
                dict(
                    utc=c.utc(), plan_sha256=plan_sha, resources=s.resource.resources()
                ),
            )
        else:
            c.require(
                c.read(started)["plan_sha256"] == plan_sha,
                "Started under different plan",
            )
        if stage in ["all", "search"]:
            began = time.perf_counter()
            for i, job in enumerate(s.search_jobs(), 1):
                if not (c.job_dir(job) / "complete.json").exists():
                    s.resource.resource_guard(3.5, 12)
                outcome = train.execute(job, data, plan, plan_sha)
                print(
                    f'SHORT SEARCH {i}/1800: {job["id"]} {outcome}; {(time.perf_counter()-began)/3600:.2f}h this invocation',
                    flush=True,
                )
            selected = audit.select(data, plan_sha)
            print(
                "All 1800 new and 720 old searches checked; expanded TRAIN-only selection locked.",
                flush=True,
            )
            if stage == "search":
                print(
                    "SEARCH COMPLETE. Resume with the normal command for final fitting and predictions.",
                    flush=True,
                )
                return
        else:
            selected = audit.select(data, plan_sha)
        routes = audit.routes(selected)
        fresh = [v["job"] for v in routes if v["origin"] == "new_fit"]
        s.save(
            BASE / "final_routes.json",
            dict(
                plan_sha256=plan_sha,
                routes=routes,
                new_models=len(fresh),
                reused_models=len(routes) - len(fresh),
            ),
        )
        c.require(len(fresh) <= 20, "Unexpected final model expansion")
        s.resource.resource_guard(3.5, s.reserve_for_jobs(fresh))
        for i, job in enumerate(fresh, 1):
            if not (c.job_dir(job) / "complete.json").exists():
                s.resource.resource_guard(3.5, s.reserve_for_jobs(fresh))
            result = train.execute(job, data, plan, plan_sha)
            print(f'NEW FINAL {i}/{len(fresh)}: {job["id"]} {result}', flush=True)
        inputs = c.read(BASE / "data/inference.json")
        for i, job in enumerate(fresh, 1):
            result = train.inference_job(job, data, plan_sha, inputs)
            print(f'NEW INFERENCE {i}/{len(fresh)}: {job["id"]} {result}', flush=True)
        final = audit.final_check(data, selected, plan_sha)
        s.save(BASE / "final_audit.json", final)
        decoded = audit.assemble(data, selected, plan_sha)
        s.save(BASE / "prediction_audit.json", decoded)
        outputs = {
            **final["outputs"],
            **decoded["outputs"],
            **s.hashes(
                [
                    BASE / n
                    for n in [
                        "selection.json",
                        "search_audit.json",
                        "final_routes.json",
                        "final_audit.json",
                        "prediction_audit.json",
                    ]
                ]
            ),
        }
        c.verify(plan["inputs"])
        c.verify(plan["prepared"])
        s.save(
            done,
            dict(
                status="short_schedules_training_predictions_complete_review_pending",
                plan_sha256=plan_sha,
                outputs=outputs,
                new_search_fits=1800,
                reused_search_fits=720,
                new_final_models=len(fresh),
                reused_final_models=len(routes) - len(fresh),
                final_seeds=list(c.FINAL_SEEDS),
                complete_outputs=decoded["complete_outputs"],
                Qwen_calls=0,
                note="All TRAIN selections and text-only predictions checked. Scientific scoring remains for Astra; no next study starts.",
            ),
        )
        print(
            "SHORTER-TRAINING STUDY COMPLETE. All searches, selections, final predictions and automatic audits passed. Tell Astra to inspect.",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["test", "pilot", "run"])
    parser.add_argument("--stage", choices=["all", "search", "finish"], default="all")
    args = parser.parse_args()
    with s.isolated():
        if args.mode == "pilot":
            with s.locks():
                pilot()
        elif args.mode == "test":
            test()
        else:
            run(args.stage)
