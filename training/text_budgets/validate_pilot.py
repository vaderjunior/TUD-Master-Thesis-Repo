import argparse
import gc
from pathlib import Path
import time
import uuid
import numpy as np
from training.text_budgets import common as b
from training.text_budgets import prepare as prepare
from training.text_budgets import train as train
from training.roberta.short import train as preserved
from training.text_budgets import audit as audit
from training.frozen_features import roberta as features

c = b.c


def tests():
    import pytest

    source = b.hashes(b.sources())
    before = b.source_guard()
    certificate = b.certificate("tests")
    if certificate.exists():
        c.verify(c.read(certificate)["artifacts"])
        print("Current source-bound tests already pass.")
        return
    folder = b.BASE / "development" / ("test_run_" + uuid.uuid4().hex[:10])
    folder.mkdir(parents=True)
    code = pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            str(b.ROOT / "tests/training/text_budgets/test_training.py"),
            "--junitxml",
            str(folder / "results.xml"),
        ]
    )
    c.require(code == 0, "Stage B synthetic tests failed; preserve the report")
    c.require(
        b.hashes(b.sources()) == source and b.source_guard() == before,
        "Sources changed during tests",
    )
    import xml.etree.ElementTree as ET

    suites = ET.parse(folder / "results.xml").getroot().findall("testsuite")
    count = sum(int(v.attrib["tests"]) for v in suites)
    c.require(
        count >= 30
        and all(
            int(v.attrib.get(k, 0)) == 0
            for v in suites
            for k in ("failures", "errors", "skipped")
        ),
        "Incomplete synthetic test run",
    )
    with b.isolated():
        b.save(
            certificate,
            dict(
                passed=True,
                tests=count,
                sources=source,
                report=c.rel(folder / "results.xml"),
                artifacts=b.hashes([folder / "results.xml"]),
                Qwen_calls=0,
            ),
        )
    print("Stage B synthetic tests certified:", count, flush=True)


def run():
    certificate = b.certificate("pilot")
    if certificate.exists():
        c.verify(c.read(certificate)["artifacts"])
        print("Current source-bound pilot already passes.")
        return
    root = b.BASE / "development" / ("pilot_" + uuid.uuid4().hex[:10])
    root.mkdir(parents=True)
    source = b.hashes(b.sources())
    before = b.source_guard()
    plan = prepare.validate_draft()
    records = []
    summary = {}
    started = time.perf_counter()
    with b.locks(), b.isolated(base=root):
        b.ac_guard()
        initial = b.resource.resource_guard(3.5, 14)

        def fit(label, job, *, export=False, old=False, pilot=False):
            path = root / label
            data = b.data(job["pool"])
            engine = preserved if old else train
            m = engine.train_attempt(
                path,
                job,
                data,
                plan["base_cache"]["snapshot"],
                pilot=pilot,
                save_model=export,
            )
            if not old:
                audit.metadata(
                    job, data, path, model_directory=m["model_directory"], pilot=pilot
                )
            records.append(
                dict(
                    label=label,
                    job=job,
                    attempt=c.rel(path),
                    planned_steps=m["planned_steps"],
                    train_seconds=m["train_seconds"],
                    total_seconds=m["total_seconds"],
                    initialization_only=m.get("initialization_only", False),
                    skipped=m["skipped_steps"],
                    peak_allocated_GiB=m["peak_allocated_gib"],
                    peak_reserved_GiB=m["peak_reserved_gib"],
                )
            )
            print(
                "Excluded technical pilot:",
                label,
                round(m["total_seconds"], 2),
                "seconds",
                flush=True,
            )
            return m, path

        j = b.job("uniform_compact_s43", "hate", "l1_e2_b", 0, 0)
        original, op = fit("preserved_parity", j, old=True)
        current, npth = fit("current_parity", j)
        keys = [
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
            "token_lengths",
            "loading",
        ]
        c.require(
            all(original[k] == current[k] for k in keys),
            "Numerical metadata differs from preserved trainer",
        )
        for name in [
            "held_logits.npy",
            "probe_logits.npy",
            "epoch_logits/001.npy",
            "epoch_logits/002.npy",
        ]:
            c.require(
                np.array_equal(
                    np.load(op / name, allow_pickle=False),
                    np.load(npth / name, allow_pickle=False),
                ),
                "Numerical logits differ from preserved trainer",
            )
        summary["preserved_trainer_exact_parity"] = True

        manifest = c.read(b.BASE / "data/jobs_one_rate.json")
        cases = {}
        for row in manifest:
            if row["state"] == "initialization_only" and row["seed"] == 0:
                j = b.job(
                    row["pool"], row["head"], row["config"], row["seed"], row["fold"]
                )
                cases[j["id"]] = j
        for seed in (1, 2):
            for loss in ("u", "b"):
                j = b.job("uniform_compact_s42", "hate_type", f"l1_e1_{loss}", seed, 0)
                cases[j["id"]] = j
        for i, j in enumerate(cases.values()):
            fit(f"sparse_{i:02d}", j)
        for seed in (0, 1, 2):
            for loss in ("u", "b"):
                fit(
                    f"two_step_s{seed}_{loss}",
                    b.job("uniform_compact_s42", "hate_type", f"l1_e2_{loss}", seed, 0),
                )
        summary["one_step_cases"] = len(cases)
        summary["two_step_cases"] = 6

        for pool in [f"uniform_5000_s{s}" for s in (42, 43, 44)]:
            for head in c.HEADS:
                for loss in (("u", "b") if pool.endswith("s42") else ("u",)):
                    fit(
                        f"benchmark_{pool}_{head}_{loss}",
                        b.job(pool, head, f"l1_e1_{loss}", 0, 0),
                    )

        exported = []
        for label, j in [
            (
                "export_trained",
                b.job("uniform_compact_s43", "hate", "l1_e2_b", 10, None),
            ),
            (
                "export_initialization",
                b.job("uniform_compact_s42", "hate_type", "l1_e1_u", 10, None),
            ),
        ]:
            m, path = fit(label, j, export=True)
            exported.append((j, m, path))
        summary["export_bytes"] = [
            sum(
                p.stat().st_size
                for p in Path(m["model_directory"]).rglob("*")
                if p.is_file()
            )
            for _, m, _ in exported
        ]

        evaluation = c.read(b.BASE / "data/inference.json")
        positions = np.linspace(0, len(evaluation) - 1, 1024, dtype=int).tolist()
        texts = [evaluation[i]["text"] for i in positions]
        j, m, path = exported[0]
        torch = train.setup()
        tok, model, _ = train.load(
            Path(m["model_directory"]),
            b.data(j["pool"])["heads"][j["head"]],
            j["seed"],
            trained=True,
        )
        try:
            encoded = train.tokens(tok, texts)
            train.infer(model, encoded[:32])
            torch.cuda.synchronize()
            start = time.perf_counter()
            z = train.infer(model, encoded)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            b.array(root / "inference_probe.npy", z)
            summary["inference"] = dict(
                rows=len(texts),
                seconds=seconds,
                rows_per_second=len(texts) / seconds,
                input_positions=positions,
                mean_tokens=float(np.mean([len(x["input_ids"]) for x in encoded])),
                at_128=sum(len(x["input_ids"]) == 128 for x in encoded),
            )
        finally:
            del tok, model
            gc.collect()
            torch.cuda.empty_cache()

        tok, model, info = features.load("cpu", plan)
        cpu = features.infer(tok, model, features.PROBES, "cpu", True)["mean"]
        del model, tok
        gc.collect()
        tok, model, info = features.load("cuda", plan)
        try:
            gpu = features.infer(tok, model, features.PROBES, "cuda", True)["mean"]
            split = np.concatenate(
                [
                    features.infer(
                        tok, model, features.PROBES[i : i + 2], "cuda", True
                    )["mean"]
                    for i in range(0, len(features.PROBES), 2)
                ]
            )
            c.require(
                np.allclose(cpu, gpu, rtol=0, atol=1e-4)
                and np.allclose(gpu, split, rtol=0, atol=1e-4),
                "Feature device/padding equivalence failed",
            )
            old_rows = c.read(features.BASE / "feature_inputs.json")
            old = np.load(
                features.BASE / "features_mean.npy", mmap_mode="r", allow_pickle=False
            )
            cached = features.infer(
                tok, model, [v["text"] for v in old_rows[:16]], "cuda", True
            )["mean"]
            c.require(
                np.allclose(cached, old[:16], rtol=0, atol=1e-4),
                "Cached frozen representation procedure differs",
            )
            torch.cuda.synchronize()
            start = time.perf_counter()
            for i in range(0, len(texts), 16):
                features.infer(tok, model, texts[i : i + 16], "cuda")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            summary["feature_throughput"] = dict(
                rows=len(texts), seconds=seconds, rows_per_second=len(texts) / seconds
            )
        finally:
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
        b.array(root / "cpu_feature_reference.npy", cpu)
        summary["feature_equivalence"] = dict(
            cpu_gpu_max=float(np.abs(cpu - gpu).max()),
            padding_max=float(np.abs(gpu - split).max()),
            old_cache_max=float(np.abs(cached - old[:16]).max()),
        )
        summary.update(
            initial_resources=initial,
            final_resources=b.resource.resources(),
            total_seconds=time.perf_counter() - started,
            maximum_allocated_GiB=max(v["peak_allocated_GiB"] for v in records),
            maximum_reserved_GiB=max(v["peak_reserved_GiB"] for v in records),
        )
        b.save(
            root / "measurements.json",
            dict(
                summary=summary,
                records=records,
                sources=source,
                excluded_from_scientific_results=True,
            ),
        )
    c.require(
        b.hashes(b.sources()) == source and b.source_guard() == before,
        "Source changed during pilot",
    )
    with b.isolated():
        b.save(
            certificate,
            dict(
                passed=True,
                sources=source,
                summary=summary,
                measurements=c.rel(root / "measurements.json"),
                excluded_from_scientific_results=True,
                feature_reference=dict(
                    cpu_path=c.rel(root / "cpu_feature_reference.npy")
                ),
                artifacts=b.hashes(p for p in root.rglob("*") if p.is_file()),
                Qwen_calls=0,
                scientific_fits=0,
            ),
        )
    print("Excluded Stage B pilot passed. No scientific queue started.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tests", "run"])
    args = ap.parse_args()
    with b.isolated():
        globals()[args.mode]()
