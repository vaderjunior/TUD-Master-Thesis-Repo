from src.hsrag.paths import PROJECT_ROOT
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = PROJECT_ROOT
OUT = ROOT / "experiments/astra/sq2_cpu_memory_20261012_v1"
PLAN_SHA = "916e7266b2238661b7b098b2a19cd9b2457b140fa91a6ed47fc8cb6cf6dc0ac7"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def child(kind, folder):
    from training.cpu import common as c
    from training.cpu import run as run
    import numpy as np
    from scipy import sparse
    from threadpoolctl import threadpool_limits

    with c.offline(), threadpool_limits(limits=1):
        c.require(
            c.sha(c.BASE / "plan.json") == PLAN_SHA, "Unexpected sealed source plan"
        )
        plan = c.read(c.BASE / "plan.json")
        c.verify(plan["inputs"])
        c.verify(plan["prepared"])
        c.current_cuda_guard()

        before = c.resource_guard(3.0, 15.0)
        spec = c.read(c.BASE / "data/full_heads.json")["hate"]
        rows = spec["train"] + spec["val"]
        c.require(len(rows) == 55753, "Largest permitted TRAIN roster differs")
        texts = run.load_texts()

        evaluation_inputs = c.read(c.BASE / "data/evaluation.json")
        yy = c.target_bits("hate", spec, rows)
        begin = time.perf_counter()
        if kind == "bge":
            x = run.dense_rows(rows)
        else:
            x, vocabulary, idf = c.fit_text_features([texts[r["id"]] for r in rows])
        feature_seconds = time.perf_counter() - begin
        feature_resource = c.resources()
        print(
            kind,
            "features built; peak GiB",
            round(feature_resource["process_peak_working_set_GiB"], 3),
            flush=True,
        )

        with io.BytesIO() as buffer:
            if kind == "bge":
                np.save(buffer, x, allow_pickle=False)
            else:
                sparse.save_npz(buffer, x, compressed=True)
            payload = buffer.getvalue()
            serialization_sha = hashlib.sha256(payload).hexdigest()
            serialized_bytes = len(payload)
            del payload
        serialization_resource = c.resources()
        if kind == "tfidf":
            del vocabulary, idf
        gc.collect()
        begin = time.perf_counter()

        fitted = c.fit_linear(x, yy, 100.0, 42)
        fit_seconds = time.perf_counter() - begin
        after = c.resources()
        result = dict(
            status="resource_probe_completed",
            representation=kind,
            utc=c.utc(),
            source_plan_sha256=PLAN_SHA,
            train_rows=len(rows),
            feature_dimensions=x.shape[1],
            evaluation_metadata_rows=len(evaluation_inputs),
            sparse_nonzeros=x.nnz if sparse.issparse(x) else None,
            feature_seconds=feature_seconds,
            fit_seconds=fit_seconds,
            iterations=fitted["iterations"].tolist(),
            serialized_feature_bytes=serialized_bytes,
            serialization_sha256=serialization_sha,
            before=before,
            after_features=feature_resource,
            after_serialization=serialization_resource,
            after=after,
            C=100.0,
            Qwen_calls=0,
            CUDA_used=False,
            evaluation_gold_read=False,
            F1_computed=False,
            scientific_result=False,
            weights_retained=False,
            limitation="Largest TRAIN head, one C, one process per representation. Does not guarantee all future peaks or background-app memory.",
        )
        write(folder / (kind + "_child.json"), result)
        print(
            kind,
            "probe finished;",
            round(after["process_peak_working_set_GiB"], 3),
            "GiB peak",
            flush=True,
        )


def monitor(kind, folder):
    import psutil

    if psutil.virtual_memory().available / 2**30 < 3.5:
        raise RuntimeError(
            "Resource probe needs 3.5 GiB available before its child starts."
        )
    log_path = folder / (kind + ".log")
    reason = None
    minimum = float("inf")
    maximum_rss = 0
    maximum_private = 0
    observed = set()
    samples = 0
    start = time.monotonic()
    with log_path.open("x", encoding="utf8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-X",
                "utf8",
                "-B",
                "-u",
                "-m",
                __spec__.name,
                "child",
                "--kind",
                kind,
                "--folder",
                str(folder),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        parent = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                available = psutil.virtual_memory().available / 2**30
                minimum = min(minimum, available)
                rss = private = 0
                try:
                    tree = [parent] + parent.children(recursive=True)
                except psutil.NoSuchProcess:
                    tree = []
                for p in tree:
                    try:
                        mem = p.memory_info()
                        rss += mem.rss
                        private += mem.private
                        observed.add(p.pid)
                    except psutil.NoSuchProcess:
                        pass
                maximum_rss = max(maximum_rss, rss / 2**30)
                maximum_private = max(maximum_private, private / 2**30)
                samples += 1
                if available < 1.25:
                    reason = "available_memory_below_1.25_GiB"
                elif private / 2**30 > 2.6:
                    reason = "probe_private_memory_above_2.6_GiB"
                elif time.monotonic() - start > 600:
                    reason = "ten_minute_probe_time_limit"
                if reason:
                    break
                time.sleep(0.1)
        finally:
            if process.poll() is None:

                try:
                    descendants = parent.children(recursive=True)
                except psutil.NoSuchProcess:
                    descendants = []
                for descendant in reversed(descendants):
                    try:
                        descendant.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                process.wait(timeout=20)
    status = dict(
        representation=kind,
        exit_code=process.returncode,
        stopped_reason=reason,
        seconds=time.monotonic() - start,
        sampled_min_available_GiB=minimum,
        sampled_peak_tree_working_set_GiB=maximum_rss,
        sampled_peak_tree_private_GiB=maximum_private,
        samples=samples,
        probe_pids=sorted(observed),
        log=str(log_path.relative_to(ROOT)),
    )
    write(folder / (kind + "_monitor.json"), status)
    print(json.dumps(status), flush=True)
    if reason or process.returncode != 0:
        raise RuntimeError(
            "Resource probe stopped; original CPU guard remains unchanged. Inspect the preserved log."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "child"])
    parser.add_argument("--kind", choices=["bge", "tfidf"])
    parser.add_argument("--folder")
    args = parser.parse_args()
    if args.mode == "child":
        folder = Path(args.folder).resolve()
        if not folder.is_relative_to(OUT.resolve()) or not args.kind:
            raise RuntimeError("Invalid probe output location")
        child(args.kind, folder)
        return
    folder = OUT / ("attempt_" + str(time.time_ns()) + "_" + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    print(
        "TRAIN-only memory probe. Original study and running Qwen processes are untouched.",
        flush=True,
    )
    print(folder.relative_to(ROOT), flush=True)
    for kind in ("bge", "tfidf"):
        monitor(kind, folder)
    write(
        folder / "complete.json",
        dict(
            status="both_resource_probes_completed",
            source_plan_sha256=PLAN_SHA,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scientific_fits_accepted=0,
            Qwen_calls=0,
            CUDA_used=False,
        ),
    )


if __name__ == "__main__":
    main()
