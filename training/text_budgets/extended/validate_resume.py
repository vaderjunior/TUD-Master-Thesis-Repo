import argparse
import ctypes
import gc
from pathlib import Path
import time
import traceback
from unittest.mock import patch
import uuid
from training.text_budgets.extended import common as b
from training.text_budgets.extended import prepare as prepare
from training.text_budgets.extended import train as train
from training.text_budgets.extended import audit as audit
from training.text_budgets.extended import resume as r

c = b.c


def resources():
    values = b.resource.resources()

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "working",
                "qpp",
                "qpa",
                "qnp",
                "qna",
                "private",
                "private_peak",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    c.require(
        query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb),
        "Memory query failed",
    )
    return dict(
        values,
        process_working_GiB=counters.working / 2**30,
        process_private_GiB=counters.private / 2**30,
    )


def science_paths():
    for name in (
        "search",
        "refits",
        "features",
        "linear",
        "selections",
        "inference",
        "predictions",
        "audits",
    ):
        yield from (p for p in (b.BASE / name).rglob("*") if p.is_file())
    for name in (
        "execution_budget.json",
        "plan.json",
        "all_selections.json",
        "complete.json",
    ):
        if (b.BASE / name).exists():
            yield b.BASE / name


def checkpoint():
    with b.locks(), b.isolated():
        plan = prepare.validate()
        with b.isolated(stores=plan["stores"]):
            state = r.preflight(plan, r.PLAN_SHA)
            record = dict(
                utc=c.utc(),
                state=state,
                original_sources=b.hashes(b.sources()),
                Qwen_sources=b.source_guard(),
                science=b.hashes(science_paths()),
                clock=c.read(b.BASE / "execution_budget.json"),
                resources=resources(),
                stopped_attempts=len(list((b.BASE / "search").rglob("stopped.json"))),
                excluded_from_scientific_results=True,
                Qwen_calls=0,
            )
            r.write_new(r.RECOVERY / "checkpoint.json", record)
            print(
                dict(
                    completed_search=sum(state["completed_search"].values()),
                    science_files=len(record["science"]),
                    resources=record["resources"],
                ),
                flush=True,
            )


def tests():
    import pytest
    import xml.etree.ElementTree as ET

    folder = r.RECOVERY / "development" / ("tests_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    paths = [
        b.ROOT / "tests/training/text_budgets/extended/test_resume.py",
        b.ROOT / "tests/training/text_budgets/extended/test_training.py",
    ]
    code = pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            *[str(p) for p in paths],
            "--basetemp",
            str(folder / "fixtures"),
            "--junitxml",
            str(folder / "results.xml"),
        ]
    )
    c.require(code == 0, "Recovery tests failed; preserve all test evidence")
    count = sum(
        int(x.attrib["tests"])
        for x in ET.parse(folder / "results.xml").getroot().findall("testsuite")
    )
    r.write_new(
        r.RECOVERY / "tests.json",
        dict(
            passed=True,
            tests=count,
            artifacts=b.hashes([folder / "results.xml"]),
            sources=b.hashes(paths + [Path(r.__file__), Path(__file__)]),
        ),
    )
    print(dict(passed=True, tests=count), flush=True)


def pilot(kind):
    import numpy as np

    c.require(kind in ("search", "final"), "Unknown technical check")
    plan = prepare.validate()
    reference = c.read(r.RECOVERY / "checkpoint.json")
    c.verify(reference["original_sources"])
    c.verify(reference["science"])
    root = r.RECOVERY / "development" / ("pilot_" + kind + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    technical_sha = "EXCLUDED_MEMORY_RECOVERY_" + kind
    job = (
        b.job("uniform_compact_s43", "hate", "l1_e20_b", 0, 0)
        if kind == "search"
        else b.job("uniform_compact_s42", "hate_type", "l1_e20_u", 10, None)
    )
    data = b.data(job["pool"])
    started = time.perf_counter()
    old_measure = c.read(b.ROOT / c.read(b.certificate("pilot"))["measurements"])
    label = "gate20_b" if kind == "search" else "sparse20_u"
    original = b.ROOT / next(
        x["attempt"] for x in old_measure["records"] if x["label"] == label
    )
    try:
        with b.locks(), b.isolated(base=root, stores=None):
            before = resources()
            b.ac_guard()
            b.resource.resource_guard(3.5, 12)
            technical_plan = dict(plan, stores=None)
            if kind == "final":
                b.save(
                    root / "all_selections.json",
                    dict(technical_fixture=True, excluded=True),
                )
                inputs = [
                    dict(id="TRAIN_" + str(i), text=text)
                    for i, text in enumerate(data["texts"][:32])
                ]
                b.save(root / "data/inference.json", inputs)
                with patch.object(r, "final_jobs", lambda *args: ({}, [job])):
                    result = r.perform(
                        dict(op=kind, job=job), technical_plan, technical_sha
                    )
            else:
                result = r.perform(
                    dict(op=kind, job=job), technical_plan, technical_sha
                )
            after = resources()
            receipt, attempt = b.receipt(job, technical_sha)
            meta = c.read(attempt / "training.json")
            old = c.read(original / "training.json")
            audit.metadata(job, data, attempt, model_directory=meta["model_directory"])
            fields = [
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
                "training_precision",
                "inference_precision",
            ]
            c.require(
                all(meta[k] == old[k] for k in fields),
                "Technical training trace differs from preserved original",
            )
            arrays = list(original.glob("*.npy")) + list(
                (original / "epoch_logits").glob("*.npy")
            )
            c.require(
                all(
                    np.array_equal(
                        np.load(p, allow_pickle=False),
                        np.load(attempt / p.relative_to(original), allow_pickle=False),
                    )
                    for p in arrays
                ),
                "Fresh-process numerical outputs differ from preserved original",
            )

            with patch.object(
                train,
                "train_attempt",
                side_effect=AssertionError("Completed fit must not rerun"),
            ):
                c.require(
                    train.execute(job, data, technical_plan, technical_sha)
                    == "verified",
                    "Completed fit not reused",
                )
            if kind == "final":
                c.require(
                    train.inference_job(job, data, technical_sha, inputs) == "verified",
                    "Completed inference not reused",
                )
            try:
                b.resource.resource_guard(3.5, 12)
                would_continue = True
                stop_message = None
            except RuntimeError as exc:
                would_continue = False
                stop_message = str(exc)
            gc.collect()
            import torch

            torch.cuda.empty_cache()
            after_gc = resources()
            record = dict(
                passed=True,
                kind=kind,
                job=job,
                root=c.rel(root),
                original_reference=c.rel(original),
                result=result,
                full20_epoch=True,
                planned_steps=meta["planned_steps"],
                skipped_steps=meta["skipped_steps"],
                exact_training_trace_fields=fields,
                identical_numpy_files=len(arrays),
                export_check=meta["export_check"],
                resources_before=before,
                resources_after=after,
                resources_after_gc=after_gc,
                original_between_fit_guard_would_pass=would_continue,
                guard_message=stop_message,
                seconds=time.perf_counter() - started,
                excluded_from_scientific_results=True,
                Qwen_calls=0,
                scientific_fits=0,
                technical_fits=1,
                artifacts=b.hashes(p for p in root.rglob("*") if p.is_file()),
            )
        c.verify(reference["science"])
        c.verify(reference["original_sources"])
        c.verify(reference["Qwen_sources"])
        r.write_new(r.RECOVERY / ("pilot_" + kind + ".json"), record)
        print(
            {
                k: record[k]
                for k in (
                    "passed",
                    "kind",
                    "planned_steps",
                    "identical_numpy_files",
                    "seconds",
                    "resources_before",
                    "resources_after",
                    "resources_after_gc",
                    "original_between_fit_guard_would_pass",
                )
            },
            flush=True,
        )
    except BaseException as exc:
        r.write_new(
            root / "stopped_check.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["checkpoint", "tests", "pilot_search", "pilot_final"]
    )
    args = parser.parse_args()
    if args.mode.startswith("pilot_"):
        pilot(args.mode.split("_")[1])
    else:
        globals()[args.mode]()
