import argparse
from pathlib import Path
from training.text_budgets import common as b
from training.text_budgets import prepare as prepare
from training.text_budgets import train as train
from training.text_budgets import features as features
from training.text_budgets import linear as linear
from training.text_budgets import audit as audit

c = b.c


def run(stage="all"):
    c.require(
        (b.BASE / "plan.json").exists(),
        "Stage B is not sealed. Scope/storage decision, checked pilot and tests are required before this command can start training.",
    )
    with b.locks():
        plan = prepare.validate()
        plan_sha = c.sha(b.BASE / "plan.json")
        with b.isolated(plan["grid"], plan["stores"]):
            done = c.BASE / "complete.json"
            if done.exists():
                r = c.read(done)
                c.require(
                    r["plan_sha256"] == plan_sha
                    and r["status"]
                    == "training_predictions_and_automatic_audits_complete",
                    "Completion identity differs",
                )
                c.verify(r["outputs"])
                for pool in b.NAMES:
                    selected = c.read(c.BASE / "selections" / f"{pool}.json")[
                        "families"
                    ]
                    audit.final_check(pool, selected, plan_sha)
                print(
                    "Verified completed Stage B. Scientific result review is a separate next step.",
                    flush=True,
                )
                return
            b.source_guard()
            b.deadline_guard()
            b.ac_guard()
            b.validate_stores(plan["stores"])
            has_final = any((c.BASE / "refits").glob("**/complete.json"))
            all_selected = all(
                (c.BASE / "selections" / f"{pool}.json").exists() for pool in b.NAMES
            )
            c.require(
                not has_final or all_selected,
                "Final fits appeared before all selections",
            )
            reserved_jobs = None
            if all_selected:
                records = {
                    pool: c.read(c.BASE / "selections" / f"{pool}.json")
                    for pool in b.NAMES
                }
                c.require(
                    all(r["plan_sha256"] == plan_sha for r in records.values()),
                    "Wrong completed selection plan",
                )
                reserved_jobs = [
                    j
                    for pool, r in records.items()
                    for j in b.final_jobs(pool, r["families"])
                ]
            b.store_guard(
                plan["stores"],
                remaining=reserved_jobs,
                initial=not (c.BASE / "search").exists(),
            )
            features.run(plan, plan_sha)
            linear.run(plan_sha)
            if stage == "features":
                print(
                    "Frozen-feature and matched linear work complete; scientific scoring deferred.",
                    flush=True,
                )
                return
            selected = {}
            for pool in b.NAMES:
                data = b.data(pool)
                if stage == "finish":
                    c.require(
                        (c.BASE / "selections" / f"{pool}.json").exists(),
                        "Finish requires all search selections",
                    )
                else:
                    for index, job in enumerate(b.jobs(pool, plan["grid"]), 1):
                        if index == 1 or index % 100 == 0:
                            b.store_guard(plan["stores"], remaining=reserved_jobs)
                            print(
                                f'Search {pool}: slot {index}/{len(b.jobs(pool,plan["grid"]))}',
                                flush=True,
                            )
                        train.execute(job, data, plan, plan_sha)
                selected[pool] = audit.select(pool, plan["grid"], plan_sha)
                print(
                    "All candidates/OOF/selection independently verified:",
                    pool,
                    flush=True,
                )
            b.save(
                c.BASE / "all_selections.json",
                dict(
                    plan_sha256=plan_sha,
                    outputs=b.hashes(
                        c.BASE / "selections" / f"{pool}.json" for pool in b.NAMES
                    ),
                    evaluation_gold_used=False,
                ),
            )
            if stage == "search":
                print(
                    "All nine neural selections complete. Resume the normal command for final fits/predictions.",
                    flush=True,
                )
                return
            jobs = [j for pool in b.NAMES for j in b.final_jobs(pool, selected[pool])]
            inputs = c.read(c.BASE / "data/inference.json")
            for index, job in enumerate(jobs, 1):
                b.store_guard(plan["stores"], remaining=jobs)
                print(f'Final model {index}/{len(jobs)}: {job["id"]}', flush=True)
                train.execute(job, b.data(job["pool"]), plan, plan_sha)
                train.inference_job(job, b.data(job["pool"]), plan_sha, inputs)
            outputs = {}
            for pool in b.NAMES:
                outputs.update(
                    audit.final_check(pool, selected[pool], plan_sha)["outputs"]
                )
                outputs.update(
                    audit.assemble(pool, selected[pool], plan_sha)["outputs"]
                )
            for path in (
                c.BASE / "features/complete.json",
                c.BASE / "linear/complete.json",
                c.BASE / "all_selections.json",
            ):
                outputs.update(b.hashes([path]))
                outputs.update(c.read(path)["outputs"])
            outputs.update(b.hashes((c.BASE / "audits").glob("*.json")))
            b.source_guard()
            b.save(
                done,
                dict(
                    status="training_predictions_and_automatic_audits_complete",
                    plan_sha256=plan_sha,
                    outputs=outputs,
                    pools=b.NAMES,
                    distinct_final_models=len(jobs),
                    Qwen_calls=0,
                    scientific_review_pending=True,
                ),
            )
            print(
                "STAGE B COMPLETE. Preserve all outputs and ask Astra for scientific review. No further experiment is launched.",
                flush=True,
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "validate"])
    ap.add_argument(
        "--stage", choices=["all", "features", "search", "finish"], default="all"
    )
    args = ap.parse_args()
    if args.mode == "validate":
        with b.isolated():
            prepare.validate()
            print("Sealed Stage B inputs and sources verified; no training.")
    else:
        run(args.stage)
