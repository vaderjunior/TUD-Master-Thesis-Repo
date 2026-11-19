import argparse
from collections import Counter
from pathlib import Path
from training.text_budgets.extended import common as b

c = b.c
OLD = b.preserved.BASE


def prepare():
    if (b.BASE / "draft.json").exists():
        validate_draft()
        print("Reduced draft verified.")
        return
    old = c.read(OLD / "draft.json")
    c.verify(old["inputs"])
    c.verify(old["prepared"])
    review = c.read(OLD / "development/preparation_review.json")
    c.verify(review["source_hashes"])
    c.verify(review["artifacts"])
    c.require(
        review["synthetic_tests"] == 41 and review["BGE_existing_fits_rechecked"] == 27,
        "Prior checked preparation differs",
    )
    for name, digest in old["base_cache"]["files"].items():
        c.require(
            c.sha(Path(old["base_cache"]["snapshot"]) / name) == digest,
            "Pinned base cache differs",
        )
    inputs = {
        **old["inputs"],
        **old["prepared"],
        **b.hashes(
            [
                OLD / "draft.json",
                OLD / "development/preparation_review.json",
                b.PROTOCOL,
            ]
        ),
    }

    for name in old["prepared"]:
        source = b.ROOT / name
        if source.name in ("jobs_full.json", "jobs_one_rate.json"):
            continue
        target = b.BASE / source.relative_to(OLD)
        c.publish_bytes(target, source.read_bytes())
    previous = c.read(OLD / "data/jobs_one_rate.json")
    wanted = {x["id"] for x in b.configs()}
    values = [r for r in previous if r["config"] in wanted]
    expected = []
    for pool in b.NAMES:
        data = b.data(pool)
        for job in b.jobs(pool):
            status, steps = b.expected_state(data, job)
            spec, fit, held = b.split(data, job)
            expected.append(
                dict(
                    job,
                    state=status,
                    steps=steps,
                    fit_rows=len(fit),
                    held_rows=len(held),
                )
            )
    c.require(
        values == expected,
        "Reduced manifest differs from independent split reconstruction",
    )
    states = dict(Counter(r["state"] for r in values))
    c.require(
        len(values) == 1080
        and states == {"trained": 1068, "empty_held_fold": 12}
        and sum(r["steps"] for r in values) == 792720,
        "Reduced job inventory differs",
    )
    b.save(b.BASE / "data/jobs_fixed20_both_losses.json", values)
    prepared = b.hashes(p for p in (b.BASE / "data").rglob("*") if p.is_file())
    b.save(
        b.BASE / "draft.json",
        dict(
            status="approved_scope_prepared_tests_pilot_capacity_seal_pending",
            utc=c.utc(),
            environment=c.environment(),
            inputs=inputs,
            prepared=prepared,
            base_cache=old["base_cache"],
            policy=b.policy(),
            pools=old["pools"],
            feature_new_rows=old["feature_new_rows"],
            feature_reused_rows=old["feature_reused_rows"],
            grid=b.GRID,
            candidate_configs=b.configs(),
            slots=1080,
            states=states,
            search_steps=792720,
            exact_final_models=360,
            final_steps=330600,
            export_GiB_bound=360 * b.EXPORT_BYTES_MAX / 2**30,
            inherited_draft=c.rel(OLD / "draft.json"),
            scope_authorized=True,
            scientific_fits=0,
            Qwen_calls=0,
        ),
    )
    validate_draft()
    print(
        "Approved reduced draft prepared:1068 fits,12 empty slots,360 final exports; no training."
    )


def validate_draft():
    draft = c.read(b.BASE / "draft.json")
    c.verify(draft["inputs"])
    c.verify(draft["prepared"])
    c.require(
        draft["environment"] == c.environment() and draft["policy"] == b.policy(),
        "Reduced environment/policy changed",
    )
    c.require(
        draft["grid"] == b.GRID and draft["candidate_configs"] == b.configs(),
        "Reduced scope changed",
    )
    for name, digest in draft["base_cache"]["files"].items():
        c.require(
            c.sha(Path(draft["base_cache"]["snapshot"]) / name) == digest,
            "Pinned base cache changed",
        )
    return draft


def seal():
    draft = validate_draft()
    decision = c.read(b.BASE / "decision.json")
    sources = b.hashes(b.sources())
    c.require(
        decision["grid"] == b.GRID
        and decision["scope_authorized"]
        and decision["continuous_power_authorized"],
        "Recorded user scope/power decision required",
    )
    tests = c.read(b.certificate("tests"))
    pilot = c.read(b.certificate("pilot"))
    c.require(
        tests["passed"]
        and pilot["passed"]
        and tests["sources"] == pilot["sources"] == sources,
        "Current reduced-scope tests/pilot required",
    )
    c.verify(tests["artifacts"])
    c.verify(pilot["artifacts"])
    capacity = c.read(b.ROOT / decision["capacity_receipt"])
    c.verify(capacity["source_hashes"])
    c.require(
        capacity["passed"]
        and capacity["planning_hours"] <= 72
        and decision["maximum_elapsed_hours"] == 96,
        "Capacity/budget check incomplete",
    )
    readiness = c.read(b.ROOT / decision["readiness_receipt"])
    c.require(
        readiness["passed"] and readiness["sources"] == sources,
        "Warmed-runtime readiness required",
    )
    b.validate_stores(decision["stores"])
    b.store_guard(decision["stores"], initial=True)
    b.ac_guard()
    b.source_guard()
    paths = [
        b.BASE / "draft.json",
        b.BASE / "decision.json",
        b.certificate("tests"),
        b.certificate("pilot"),
        b.ROOT / decision["capacity_receipt"],
        b.ROOT / decision["readiness_receipt"],
    ]
    b.save(
        b.BASE / "plan.json",
        dict(
            draft,
            status="sealed_ready_no_scientific_fits",
            stores=decision["stores"],
            inputs={
                **draft["inputs"],
                **sources,
                **tests["artifacts"],
                **pilot["artifacts"],
                **b.hashes(paths),
            },
            technical_amendment=c.rel(b.AMENDMENT),
            scope_amendment=c.rel(b.PROTOCOL),
            pilot_receipt=c.rel(b.certificate("pilot")),
            tests_receipt=c.rel(b.certificate("tests")),
            pilot_summary=pilot["summary"],
            maximum_elapsed_hours=96,
        ),
    )
    validate()
    print("Reduced comparison sealed. Scientific execution remains a user action.")


def validate():
    plan = c.read(b.BASE / "plan.json")
    c.verify(plan["inputs"])
    c.verify(plan["prepared"])
    c.require(
        plan["environment"] == c.environment() and plan["policy"] == b.policy(),
        "Sealed environment/policy differs",
    )
    c.require(
        plan["grid"] == b.GRID
        and plan["scope_amendment"] == c.rel(b.PROTOCOL)
        and plan["exact_final_models"] == 360,
        "Sealed reduced scope differs",
    )
    b.validate_stores(plan["stores"])
    for name, digest in plan["base_cache"]["files"].items():
        c.require(
            c.sha(Path(plan["base_cache"]["snapshot"]) / name) == digest,
            "Pinned cache differs",
        )
    return plan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "seal", "validate"])
    args = ap.parse_args()
    with b.isolated():
        globals()[args.mode]()
