import argparse
import math
import shutil
import subprocess
import sys
import time
import uuid
from training.roberta import common as c
from training.roberta import prepare as preparation


def test():
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
            "tests/training/roberta/test_training.py",
        ],
        cwd=c.ROOT,
        check=False,
    )
    c.require(result.returncode == 0, "Tuning fixtures failed")
    path = c.BASE / "development/tests.json"
    value = dict(passed=True, sources=c.source_hashes())
    if path.exists() and c.read(path) != value:
        c.save(
            c.BASE / "development/test_history" / f"{uuid.uuid4().hex[:16]}.json",
            c.read(path),
        )

        c.require(
            not (c.BASE / "plan.json").exists(),
            "Cannot amend tests after execution seal",
        )
        path.unlink()
    c.save(path, value)


def pilot():
    from training.roberta import train as train

    p = preparation.validate_preparation()
    data = c.read(c.BASE / "data/train.json")
    sources = c.source_hashes()
    done = c.BASE / "development/pilot_complete.json"
    if done.exists() and c.read(done)["sources"] == sources:
        c.verify(c.read(done)["artifacts"])
        print("Numerical pilot already verified.")
        return
    c.require(
        not (c.BASE / "plan.json").exists(),
        "Cannot run a changed pilot after execution seal",
    )
    directory = c.BASE / "development/pilots" / uuid.uuid4().hex[:12]
    values = []
    artifacts = {}
    for head in c.HEADS:
        for loss in ("unweighted", "balanced"):
            job = dict(
                id=f"pilot_{head}_{loss}",
                mode="search",
                head=head,
                config=f"l1_e20_{loss[0]}",
                seed=987,
                fold=0,
            )
            path = c.own(directory / job["id"])
            path.mkdir(parents=True)
            c.save(
                path / "started.json",
                dict(
                    job=job,
                    pilot_only=True,
                    sources=sources,
                    preparation_sha256=c.sha(c.BASE / "preparation.json"),
                ),
            )
            print(
                f"Numerical pilot: {head}/{loss}, 8 attempted TRAIN steps only.",
                flush=True,
            )
            value = train.train_attempt(
                path,
                job,
                data,
                p["base_cache"]["snapshot"],
                pilot=True,
                save_model=head == "hate_type" and loss == "balanced",
            )
            c.require(
                value["attempted_steps"] == 8 and value["successful_steps"] > 0,
                "Pilot made no effective progress",
            )
            values.append(value)
            artifacts.update(
                {c.rel(f): c.sha(f) for f in path.rglob("*") if f.is_file()}
            )

    step_cost = {
        h: sum(v["train_seconds"] / 8 for v in values if v["job"]["head"] == h) / 2
        for h in c.HEADS
    }
    init_cost = {
        h: sum(
            max(0, v["total_seconds"] - v["train_seconds"])
            for v in values
            if v["job"]["head"] == h
        )
        / 2
        for h in c.HEADS
    }
    search_steps = 0
    estimate = 0
    for job in c.search_jobs():
        _, fit, _ = train.split(data, job)
        n = math.ceil(len(fit) / 16) * c.config_map()[job["config"]]["epochs"]
        search_steps += n
        estimate += n * step_cost[job["head"]] + init_cost[job["head"]]
    summary = dict(
        pilot_fits=8,
        pilot_attempted_steps=64,
        max_peak_allocated_gib=max(v["peak_allocated_gib"] for v in values),
        skipped_steps=sum(v["skipped_steps"] for v in values),
        cv_planned_steps=search_steps,
        rough_cv_hours=estimate / 3600,
        estimate_caution="Short technical pilot, not a promised runtime; refits, inference, hashing and concurrent workload add time.",
        gpu=values[0]["gpu"],
        final_fits_upper_bound=60,
        inference_inputs_per_final_fit=19082,
    )
    value = dict(
        passed=True,
        sources=sources,
        artifacts=artifacts,
        summary=summary,
        excludes_scientific_results=True,
    )
    if done.exists():
        c.save(
            c.BASE / "development/pilot_history" / f"{uuid.uuid4().hex[:16]}.json",
            c.read(done),
        )
        done.unlink()
    c.save(done, value)
    print(c.encoded(summary).decode("utf8"), flush=True)


def select_search(data, plan_sha):
    from training.roberta import audit as audit

    oof, receipts, steps = audit.load_oof(data, plan_sha)
    selected = c.select(data, oof)
    checks = audit.check_selection(data, oof, selected)
    value = dict(plan_sha256=plan_sha, search_receipts=receipts, families=selected)
    c.save(c.BASE / "selection.json", value)
    c.save(
        c.BASE / "search_audit.json",
        dict(
            search_jobs=len(receipts),
            optimizer_attempted_steps=steps,
            independently_checked_candidates=checks,
            selection_sha256=c.sha(c.BASE / "selection.json"),
            plan_sha256=plan_sha,
        ),
    )
    print(
        f"TRAIN-only selection locked; {checks} candidate scores independently checked. No evaluation gold read.",
        flush=True,
    )
    return selected


def run(stage="all"):
    from training.roberta import train as train
    from training.roberta import audit as audit

    with c.launcher_lock():
        plan = preparation.validate()
        plan_sha = c.sha(c.BASE / "plan.json")
        data = c.read(c.BASE / "data/train.json")
        done = c.BASE / "complete.json"
        if done.exists():
            value = c.read(done)
            c.require(
                value["plan_sha256"] == plan_sha,
                "Final receipt belongs to another plan",
            )
            c.verify(value["artifacts"])

            selected = select_search(data, plan_sha)
            audit.final_check(data, selected, plan_sha)
            result = audit.assemble(
                selected, data, plan_sha, c.read(c.BASE / "data/inference.json")
            )
            c.require(
                result == c.read(c.BASE / "prediction_audit.json"),
                "Completed prediction audit differs",
            )
            print(
                "VERIFIED COMPLETE. No new training or inference. Tell Astra to review the saved evidence."
            )
            return
        train.setup()

        has_refit = (
            next((c.BASE / "refits").glob("*/*/*/complete.json"), None) is not None
            if (c.BASE / "refits").exists()
            else False
        )
        minimum = 8 if has_refit else 45
        c.require(
            shutil.disk_usage(c.ROOT).free >= minimum * 2**30,
            f"Need {minimum}GiB free disk space before continuing",
        )
        if stage in ("all", "search"):
            n = len(plan["jobs"])
            start = time.perf_counter()
            for i, job in enumerate(plan["jobs"], 1):
                result = train.execute(job, data, plan, plan_sha)
                print(
                    f'SEARCH {i}/{n}: {job["id"]} {result}; {(time.perf_counter()-start)/3600:.2f}h this invocation',
                    flush=True,
                )
            selected = select_search(data, plan_sha)
            if stage == "search":
                print("SEARCH COMPLETE; no final refits requested in this invocation.")
                return
        else:
            selected = select_search(data, plan_sha)
        jobs = c.final_jobs(selected)
        for i, job in enumerate(jobs, 1):
            result = train.execute(job, data, plan, plan_sha)
            print(f'REFIT {i}/{len(jobs)}: {job["id"]} {result}', flush=True)
        final = audit.final_check(data, selected, plan_sha)
        c.save(c.BASE / "final_fit_audit.json", final)
        inputs = c.read(c.BASE / "data/inference.json")
        for i, job in enumerate(jobs, 1):
            result = train.inference_job(job, data, plan_sha, inputs)
            print(f'INFERENCE {i}/{len(jobs)}: {job["id"]} {result}', flush=True)
        result = audit.assemble(selected, data, plan_sha, inputs)
        c.save(c.BASE / "prediction_audit.json", result)
        artifacts = {
            c.rel(c.BASE / name): c.sha(c.BASE / name)
            for name in (
                "selection.json",
                "search_audit.json",
                "final_fit_audit.json",
                "prediction_audit.json",
            )
        }
        artifacts.update(result["artifacts"])
        artifacts.update(final["receipts"])
        c.save(
            done,
            dict(
                status="training_and_predictions_complete",
                plan_sha256=plan_sha,
                search_fits=720,
                final_fits=len(jobs),
                final_seeds=c.FINAL_SEEDS,
                decoded_variants=6,
                complete_outputs=result["complete_outputs"],
                artifacts=artifacts,
                Qwen_calls=0,
                result_note="TRAIN selection and text-only inference complete. Evaluation scores and scientific interpretation await independent review.",
            ),
        )
        print(
            "ROBERTA TUNING COMPLETE: all search fits, selected refits, predictions and automatic audits passed. Tell Astra to inspect.",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["test", "pilot", "run"])
    parser.add_argument("--stage", choices=["all", "search", "finish"], default="all")
    args = parser.parse_args()
    if args.mode == "test":
        test()
    elif args.mode == "pilot":
        with c.launcher_lock():
            pilot()
    else:
        run(args.stage)
