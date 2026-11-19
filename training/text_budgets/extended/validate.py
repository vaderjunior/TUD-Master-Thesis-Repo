import argparse
import ast
import gc
import math
from pathlib import Path
import shutil
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from training.text_budgets.extended import common as b
from training.text_budgets.extended import prepare as prepare
from training.text_budgets.extended import train as train
from training.text_budgets.extended import audit as audit

c = b.c


def tests():
    import pytest

    source = b.hashes(b.sources())
    before = b.source_guard()
    path = b.certificate("tests")
    if path.exists():
        c.verify(c.read(path)["artifacts"])
        print("Current reduced tests already certified.")
        return
    root = b.BASE / "development" / ("tests_" + uuid.uuid4().hex[:10])
    root.mkdir(parents=True)
    code = pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            str(b.ROOT / "tests/training/text_budgets/extended/test_training.py"),
            "--junitxml",
            str(root / "results.xml"),
        ]
    )
    c.require(code == 0, "Reduced-scope tests failed; retain all fixture evidence")
    suites = ET.parse(root / "results.xml").getroot().findall("testsuite")
    count = sum(int(v.attrib["tests"]) for v in suites)
    c.require(
        count >= 50
        and all(
            int(v.attrib.get(k, 0)) == 0
            for v in suites
            for k in ("failures", "errors", "skipped")
        ),
        "Incomplete test suite",
    )
    c.require(
        source == b.hashes(b.sources()) and before == b.source_guard(),
        "Sources changed during tests",
    )
    b.save(
        path,
        dict(
            passed=True,
            tests=count,
            sources=source,
            report=c.rel(root / "results.xml"),
            artifacts=b.hashes([root / "results.xml"]),
            Qwen_calls=0,
            scientific_fits=0,
        ),
    )
    print("Reduced-scope tests passed:", count, flush=True)


def kernel_equivalence():
    checked = {}
    for name in ("train", "audit", "features", "linear"):
        old = b.ROOT / f"scratch/analysis/astra_stageb_{name}_v1.py"
        new = b.ROOT / f"scratch/analysis/astra_stageb20_{name}_v1.py"
        original = old.read_text(encoding="utf-8")
        current = new.read_text(encoding="utf-8").replace(
            "astra_stageb20_", "astra_stageb_"
        )
        if name in ("features", "linear"):

            current = "\n".join(
                line
                for line in current.splitlines()
                if line.strip() != "b.deadline_guard()"
            )
        c.require(
            ast.dump(ast.parse(original), include_attributes=False)
            == ast.dump(ast.parse(current), include_attributes=False),
            "Numerical implementation changed beyond routing/deadline: " + name,
        )
        checked[name] = dict(
            original=c.rel(old),
            original_sha256=c.sha(old),
            current=c.rel(new),
            current_sha256=c.sha(new),
            comparison="Exact normalized Python AST; import routing and declared deadline checks only",
        )
    return checked


def numerics():
    source = b.hashes(b.sources())
    before = b.source_guard()
    path = b.certificate("pilot")
    if path.exists():
        c.verify(c.read(path)["artifacts"])
        print("Current reduced numerical evidence certified.")
        return
    draft = prepare.validate_draft()
    equivalence = kernel_equivalence()
    oldpath = b.preserved.certificate("pilot")
    old = c.read(oldpath)
    c.require(
        old["passed"] and old["excluded_from_scientific_results"],
        "Inherited numerical evidence not passing/excluded",
    )
    c.verify(old["sources"])
    c.verify(old["artifacts"])
    root = b.BASE / "development" / ("pilot20_" + uuid.uuid4().hex[:10])
    records = []
    started = time.perf_counter()
    with b.locks(), b.isolated(base=root):
        b.ac_guard()
        initial = b.resource.resource_guard(3.5, 14)
        b.save(
            root / "started.json",
            dict(sources=source, utc=c.utc(), excluded_from_scientific_results=True),
        )
        try:
            jobs = [
                (
                    "sparse20_u",
                    b.job("uniform_compact_s42", "hate_type", "l1_e20_u", 10, None),
                    True,
                ),
                (
                    "sparse20_b",
                    b.job("uniform_compact_s42", "hate_type", "l1_e20_b", 11, None),
                    True,
                ),
                (
                    "gate20_b",
                    b.job("uniform_compact_s43", "hate", "l1_e20_b", 0, 0),
                    False,
                ),
            ]
            for label, job, export in jobs:
                data = b.data(job["pool"])
                attempt = root / label

                meta = train.train_attempt(
                    attempt,
                    job,
                    data,
                    draft["base_cache"]["snapshot"],
                    pilot=False,
                    save_model=export,
                )
                audit.metadata(
                    job,
                    data,
                    attempt,
                    model_directory=meta["model_directory"],
                    pilot=False,
                )
                c.require(
                    len(meta["epochs"]) == 20 and not meta["initialization_only"],
                    "Actual20-epoch numerical check incomplete",
                )
                records.append(
                    dict(
                        label=label,
                        job=job,
                        attempt=c.rel(attempt),
                        steps=meta["planned_steps"],
                        train_seconds=meta["train_seconds"],
                        total_seconds=meta["total_seconds"],
                        export_directory=meta["model_directory"],
                        export_check=meta["export_check"],
                        peak_allocated_GiB=meta["peak_allocated_gib"],
                        peak_reserved_GiB=meta["peak_reserved_gib"],
                    )
                )
                print(
                    "Excluded full20-epoch check passed:",
                    label,
                    round(meta["total_seconds"], 2),
                    "seconds",
                    flush=True,
                )
            b.save(
                root / "measurements.json",
                dict(
                    records=records,
                    source_hashes=source,
                    initial_resources=initial,
                    final_resources=b.resource.resources(),
                    seconds=time.perf_counter() - started,
                    excluded_from_scientific_results=True,
                ),
            )
        except BaseException as exc:
            b.save(
                root / "stopped.json",
                dict(
                    utc=c.utc(),
                    type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    completed_records=records,
                    excluded_from_scientific_results=True,
                ),
            )
            raise
    c.require(
        source == b.hashes(b.sources()) and before == b.source_guard(),
        "Sources changed during numerical checks",
    )
    b.save(
        path,
        dict(
            passed=True,
            sources=source,
            excluded_from_scientific_results=True,
            summary=dict(
                new_full20_epoch_attempts=3,
                new_planned_steps=sum(r["steps"] for r in records),
                inherited_technical_attempts=60,
                inherited_summary=old["summary"],
                exact_kernel_equivalence=True,
                maximum_allocated_GiB=max(r["peak_allocated_GiB"] for r in records),
            ),
            inherited_receipt=c.rel(oldpath),
            kernel_equivalence=equivalence,
            measurements=c.rel(root / "measurements.json"),
            feature_reference=old["feature_reference"],
            artifacts={
                **old["artifacts"],
                **b.hashes([oldpath]),
                **b.hashes(p for p in root.rglob("*") if p.is_file()),
            },
            Qwen_calls=0,
            scientific_fits=0,
        ),
    )
    print(
        "Current scope numerical certificate complete; no scientific queue started.",
        flush=True,
    )


def capacity():
    draft = prepare.validate_draft()
    source = b.hashes(b.sources())
    prior_path = (
        b.ROOT / "experiments/astra/sq2_stage_b_value_review_20261021_v1/estimates.json"
    )
    prior = c.read(prior_path)
    c.verify(prior["source_hashes"])
    choice = prior["options"]["fixed20_both_losses"]
    rows = c.read(b.BASE / "data/jobs_fixed20_both_losses.json")
    total = 0
    files = 0
    for row in rows:
        if row["state"] == "empty_held_fold":
            sizes = [12000, 12000, 12000]
        else:
            job = b.job(
                row["pool"], row["head"], row["config"], row["seed"], row["fold"]
            )
            spec, fit, held = b.split(b.data(row["pool"]), job)
            identity = len(
                c.encoded(
                    dict(
                        fit_ids=[r["id"] for r in fit], held_ids=[r["id"] for r in held]
                    )
                )
            )
            trace = 20000 + identity + row["steps"] * 440 + 20 * 500
            logit = 128 + len(held) * len(spec["labels"]) * 4
            sizes = [trace, 12000, 12000 + 20 * 160, 256, logit] + [logit] * 20
        total += sum(math.ceil(n / 4096) * 4096 + 1024 for n in sizes) + 4 * 4096
        files += len(sizes)
    nonweight = total / 2**30 + 3
    qpath = b.preserved.BASE / "development/capacity/allocation_snapshot.json"
    q = c.read(qpath)
    growth = q["Qwen_remaining_growth_GiB"]
    c_reserve = math.ceil(nonweight + growth + 12)
    root = Path("D:/Thesis_Models/astra_sq2_stage_b20_20261021_v1")
    stores = dict(
        pools={p: str(root) for p in b.NAMES},
        drive_reserve_GiB={"C:\\": c_reserve, "D:\\": 16},
        initial_drive_reserve_GiB={"C:\\": c_reserve, "D:\\": 16},
    )
    b.validate_stores(stores)
    b.store_guard(stores, initial=True)

    c.require(
        root.resolve() == root and root.name == "astra_sq2_stage_b20_20261021_v1",
        "Unexpected export probe target",
    )
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ("preparation_probe_" + uuid.uuid4().hex[:10] + ".json")
    payload = c.encoded(
        dict(
            study=c.rel(b.BASE),
            grid=b.GRID,
            utc=c.utc(),
            purpose="Write/read capability only; not a model",
        )
    )
    with probe.open("xb") as stream:
        stream.write(payload)
    c.require(probe.read_bytes() == payload, "Export drive write/read probe failed")
    required = {"C:\\": c_reserve, "D:\\": 360 * b.EXPORT_BYTES_MAX / 2**30 + 16}
    free = {drive: shutil.disk_usage(drive).free / 2**30 for drive in required}
    value = dict(
        passed=all(free[d] >= v for d, v in required.items()),
        utc=c.utc(),
        sources=source,
        stores=stores,
        required_free_GiB=required,
        current_free_GiB=free,
        nonweight_GiB_upper=nonweight,
        search_files_upper=files,
        Qwen_growth_GiB_allowance=growth,
        Qwen_growth_note="Retain the earlier conservative allowance; do not reduce it using unreviewed live progress.",
        final_exports=360,
        export_GiB_bound=360 * b.EXPORT_BYTES_MAX / 2**30,
        planning_hours=choice["planning_hours_with_30percent_and_8hours"],
        component_hours=choice["measured_component_hours"],
        maximum_elapsed_hours=96,
        external_F_needed=False,
        export_probe=dict(path=str(probe), sha256=c.sha(probe)),
        source_hashes=b.hashes(
            [
                prior_path,
                qpath,
                b.BASE / "draft.json",
                b.BASE / "data/jobs_fixed20_both_losses.json",
                Path(__file__),
            ]
        ),
        caveat="Snapshots are not reservations; launch/runtime guards remain. Growth allowances are conservative planning, not hard bounds on unlimited retries.",
        Qwen_calls=0,
    )
    path = b.BASE / "development" / ("capacity_" + c.signature(source) + ".json")
    b.save(path, value)
    print(
        dict(
            receipt=c.rel(path),
            **{
                k: value[k]
                for k in [
                    "passed",
                    "current_free_GiB",
                    "required_free_GiB",
                    "nonweight_GiB_upper",
                    "planning_hours",
                    "external_F_needed",
                ]
            },
        )
    )


def ready():
    source = b.hashes(b.sources())
    before = b.source_guard()
    started = c.utc()

    torch = train.setup()
    from transformers import RobertaForSequenceClassification

    gc.collect()
    torch.cuda.empty_cache()
    b.ac_guard()
    snapshot = b.resource.resources()
    free, total = torch.cuda.mem_get_info()
    path = b.BASE / "development" / ("readiness_" + uuid.uuid4().hex[:10] + ".json")
    passed = snapshot["available_GiB"] >= 3.5 and free >= 5 * 2**30
    b.save(
        path,
        dict(
            passed=passed,
            started_utc=started,
            utc=c.utc(),
            sources=source,
            resources=snapshot,
            free_GPU_GiB=free / 2**30,
            gpu=torch.cuda.get_device_name(0),
            AC_connected=True,
            imports_warmed=True,
            requirements=dict(available_RAM_GiB=3.5, free_GPU_GiB=5),
            Qwen_calls=0,
            model_inference=0,
            scientific_fits=0,
        ),
    )
    c.require(
        before == b.source_guard() and source == b.hashes(b.sources()),
        "Sources changed during readiness",
    )
    print(
        dict(
            passed=passed,
            receipt=c.rel(path),
            resources=snapshot,
            free_GPU_GiB=free / 2**30,
        ),
        flush=True,
    )
    c.require(
        passed,
        "Warmed runtime needs3.5 GiB available RAM and5 GiB free GPU; preserve evidence and close unused apps before launch",
    )


def decision():
    source = b.hashes(b.sources())
    cap_path = b.BASE / "development" / ("capacity_" + c.signature(source) + ".json")
    cap = c.read(cap_path)
    c.require(
        cap["passed"] and cap["sources"] == source, "Current capacity record required"
    )
    candidates = [
        p
        for p in (b.BASE / "development").glob("readiness_*.json")
        if c.read(p)["passed"] and c.read(p)["sources"] == source
    ]
    c.require(bool(candidates), "Current passed warmed readiness required")
    rp = max(candidates, key=lambda p: c.read(p)["utc"])
    c.require(
        c.sha(Path(cap["export_probe"]["path"])) == cap["export_probe"]["sha256"],
        "Export probe changed",
    )
    b.save(
        b.BASE / "decision.json",
        dict(
            date="2026-10-21",
            grid=b.GRID,
            scope_authorized=True,
            continuous_power_authorized=True,
            user_scope="yeah 2-3 days sound reasonable, i can keep the lap plugged in throught; take your time to set it up properly, be thorough; do the important stuff with the 2-3 day budget we have, 4 days is also fine if it helps that much",
            previous_one_rate_authorized=True,
            adopted_epochs=[20],
            adopted_learning_rates=[2e-5],
            adopted_losses=["unweighted", "balanced"],
            retained_pools=9,
            retained_heads=4,
            search_seeds=[0, 1, 2],
            final_seeds=[10, 11, 12, 13, 14],
            scope_reason="Use the agreed resource-bounded design. Preserve replication and loss/cutoff controls; no further duration search.",
            expected_continuous_hours=cap["planning_hours"],
            maximum_elapsed_hours=96,
            stores=cap["stores"],
            external_F_needed=False,
            capacity_receipt=c.rel(cap_path),
            readiness_receipt=c.rel(rp),
            scope_amendment=c.rel(b.PROTOCOL),
            scientific_launch_by_user=True,
        ),
    )
    print(
        "Explicit user decision and checked internal storage recorded; ready for seal."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "mode", choices=["tests", "numerics", "capacity", "ready", "decision"]
    )
    args = ap.parse_args()
    with b.isolated():
        globals()[args.mode]()
