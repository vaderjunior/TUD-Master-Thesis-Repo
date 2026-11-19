from training.recovery.state import (
    reviewed_stop,
    completed_state,
    expected_saved,
    policy_sha256,
)
from collections import Counter
import gc
import math
from pathlib import Path
import traceback
from types import SimpleNamespace
import uuid
import numpy as np
from scipy.special import expit, softmax
from training.full_data import common as c
from training.full_data import plan as p
from training.full_data import runtime as r
from training.full_data import audit as audit
from training.full_data import reference as reference
from training.full_data import selection as selection


def identity(job):
    return dict(
        job=job,
        plan_sha256=c.sha(p.PACKAGE / "plan.json"),
        package_sha256=c.sha(p.PACKAGE / "manifest.json"),
        gpu_profile="search_8x10" if job["mode"] == "search" else "final_1x80",
    )


def job_dir(job):
    return r.inside(r.OUTPUT, "jobs/" + job["id"])


def model_path(job, attempt):
    return r.inside(r.OUTPUT, "models/" + job["id"] + "/" + attempt)


def receipt(job, *, weights=True):
    value = c.read(job_dir(job) / "complete.json")
    c.require(
        value["identity"] == identity(job) and value["status"] == "complete",
        "Fit identity changed.",
    )
    attempt = r.inside(r.OUTPUT, value["attempt"])
    c.require(attempt.parent == job_dir(job) / "attempts", "Attempt path changed.")
    r.verify_tree(attempt, value["products"])
    if job["mode"] == "final" and weights:
        model = model_path(job, attempt.name)
        c.require(
            value["model_directory"] == str(model)
            and {"model.safetensors", "config.json", "study_meta.json"}
            <= set(value["model_products"]),
            "Invalid model export.",
        )
        r.verify_tree(model, value["model_products"])
    return value, attempt


def execute(job, data, session):
    from training.full_data import train as train

    done = job_dir(job) / "complete.json"
    if done.exists():
        receipt(job)
        return
    for old in (job_dir(job) / "attempts").glob("*/stopped.json"):
        c.require(
            reviewed_stop(old),
            "A preserved numerical/integrity failure needs review: " + str(old),
        )
    state, steps = p.expected_state(data, job)
    attempt = job_dir(job) / "attempts" / uuid.uuid4().hex
    c.save(
        attempt / "started.json",
        dict(identity=identity(job), utc=c.utc(), session_sha256=c.signature(session)),
    )
    try:
        r.deadline_guard(session)
        if state == "empty_held_fold":
            spec, fit, held = p.split(data, job)
            c.save(
                attempt / "empty_held_fold.json",
                dict(
                    identity=identity(job),
                    fit_ids=[x["id"] for x in fit],
                    held_ids=[],
                    fits=0,
                    steps=0,
                ),
            )
            model = None
        else:
            train.b = SimpleNamespace(
                split=p.split,
                deadline_guard=lambda: r.deadline_guard(session),
                model_path=model_path,
                EXPORT_BYTES_MAX=p.EXPORT_BYTES_MAX,
            )
            info = train.train_attempt(
                attempt,
                job,
                data,
                p.PACKAGE / "data/model",
                save_model=job["mode"] == "final",
            )
            model = Path(info["model_directory"]) if info["model_directory"] else None
            audit.metadata(job, data, attempt, model_directory=model)
        state = completed_state(job, data, attempt, state)
        products = r.manifest(attempt)
        c.save(
            done,
            dict(
                status="complete",
                verification_amendment_sha256=policy_sha256(),
                identity=identity(job),
                attempt=attempt.relative_to(r.OUTPUT).as_posix(),
                state=state,
                steps=steps,
                products=products,
                model_directory=str(model) if model else None,
                model_products=r.manifest(model) if model else {},
            ),
        )
        receipt(job)
    except BaseException as exc:
        c.save(
            attempt / "stopped.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                utc=c.utc(),
                traceback=traceback.format_exc(),
                resumable=isinstance(exc, (r.Pause, KeyboardInterrupt)),
            ),
        )
        raise


def select_pool(pool):
    data = p.data(pool)
    oof = {
        h: {
            cfg["id"]: np.full(
                (
                    len(p.CV_SEEDS),
                    len(data["heads"][h]["rows"]),
                    len(data["heads"][h]["labels"]),
                ),
                np.nan,
            )
            for cfg in p.configs()
        }
        for h in p.HEADS
    }
    receipts = {}
    counts = Counter()
    steps = 0
    for job in p.jobs(pool):
        saved, attempt = receipt(job)
        state, total = expected_saved(data, job, saved, attempt)
        c.require(
            saved["state"] == state and saved["steps"] == total,
            "Fit classification changed.",
        )
        if state == "empty_held_fold":
            spec, fit, held = p.split(data, job)
            c.require(
                c.read(attempt / "empty_held_fold.json")
                == dict(
                    identity=identity(job),
                    fit_ids=[x["id"] for x in fit],
                    held_ids=[],
                    fits=0,
                    steps=0,
                )
                and not held
                and not (attempt / "held_logits.npy").exists()
                and not (attempt / "training.json").exists(),
                "Empty fold differs.",
            )
        else:
            audit.metadata(job, data, attempt)
            audit.place_oof(
                oof[job["head"]][job["config"]],
                data,
                job,
                np.load(attempt / "held_logits.npy", allow_pickle=False),
            )
        receipts[(job_dir(job) / "complete.json").relative_to(r.OUTPUT).as_posix()] = (
            c.sha(job_dir(job) / "complete.json")
        )
        counts[state] += 1
        steps += total
    c.require(
        all(np.isfinite(v).all() for head in oof.values() for v in head.values()),
        "Missing OOF entries.",
    )
    selected = selection.select(data, oof)
    checked = reference.check_selection(data, oof, selected)
    path = r.OUTPUT / "selections" / f"{pool}.json"
    c.save(
        path,
        dict(
            plan_sha256=c.sha(p.PACKAGE / "plan.json"),
            pool=pool,
            families=selected,
            search_receipts=receipts,
            states=dict(counts),
            attempted_steps=steps,
            independent_candidate_checks=checked,
            evaluation_gold_used=False,
        ),
    )
    return selected


def check_inference(job, inputs, labels):
    root = r.inside(r.OUTPUT, "inference/" + job["id"])
    value = c.read(root / "complete.json")
    expected = dict(
        identity=identity(job),
        training_receipt_sha256=c.sha(job_dir(job) / "complete.json"),
        input_sha256=c.signature(inputs),
        labels=labels,
        rows=len(inputs),
    )
    c.require(value["binding"] == expected, "Inference identity changed.")
    r.verify_tree(root, value["products"])
    pieces = []
    for start in range(0, len(inputs), 512):
        stop = min(start + 512, len(inputs))
        chunk = root / "chunks" / f"{start:06d}_{stop:06d}"
        saved = c.read(chunk.with_suffix(".json"))
        ident = dict(
            identity=identity(job),
            training_receipt_sha256=expected["training_receipt_sha256"],
            start=start,
            stop=stop,
            input_sha256=c.signature(inputs[start:stop]),
        )
        c.require(
            saved["identity"] == ident
            and c.sha(chunk.with_suffix(".npy")) == saved["sha256"],
            "Inference chunk changed.",
        )
        z = np.load(chunk.with_suffix(".npy"), allow_pickle=False)
        c.require(
            z.dtype == np.float32
            and z.shape == (stop - start, len(labels))
            and np.isfinite(z).all(),
            "Invalid inference chunk.",
        )
        pieces.append(z)
    full = np.load(root / "logits.npy", allow_pickle=False)
    c.require(
        np.array_equal(full, np.concatenate(pieces)),
        "Assembled logits differ from chunks.",
    )
    expected_files = {"logits.npy"} | {
        f"chunks/{start:06d}_{min(start+512,len(inputs)):06d}.{ext}"
        for start in range(0, len(inputs), 512)
        for ext in ("npy", "json")
    }
    c.require(
        set(value["products"]) == expected_files, "Inference product roster differs."
    )
    return full


def inference(job, data, session):
    from training.full_data import train as train

    root = r.inside(r.OUTPUT, "inference/" + job["id"])
    inputs = c.read(p.PACKAGE / "data/inference.json")
    spec = data["heads"][job["head"]]
    if (root / "complete.json").exists():
        check_inference(job, inputs, spec["labels"])
        return
    source, attempt = receipt(job)
    source_sha = c.sha(job_dir(job) / "complete.json")
    output = np.empty((len(inputs), len(spec["labels"])), dtype=np.float32)
    pending = []
    for start in range(0, len(inputs), 512):
        stop = min(start + 512, len(inputs))
        chunk = root / "chunks" / f"{start:06d}_{stop:06d}"
        ident = dict(
            identity=identity(job),
            training_receipt_sha256=source_sha,
            start=start,
            stop=stop,
            input_sha256=c.signature(inputs[start:stop]),
        )
        if chunk.with_suffix(".json").exists():
            saved = c.read(chunk.with_suffix(".json"))
            z = np.load(chunk.with_suffix(".npy"), allow_pickle=False)
            c.require(
                saved["identity"] == ident
                and saved["sha256"] == c.sha(chunk.with_suffix(".npy")),
                "Changed inference commit.",
            )
            c.require(
                z.dtype == np.float32
                and z.shape == (stop - start, len(spec["labels"]))
                and np.isfinite(z).all(),
                "Invalid inference values.",
            )
            output[start:stop] = z
        else:
            pending.append((start, stop, chunk, ident))
    if pending:
        torch = train.setup()
        tok, model, _ = train.load(
            Path(source["model_directory"]), spec, job["seed"], trained=True
        )
        try:
            for start, stop, chunk, ident in pending:
                r.deadline_guard(session)
                values = train.infer(
                    model, train.tokens(tok, [x["text"] for x in inputs[start:stop]])
                )
                dest = chunk.with_suffix(".npy")
                if dest.exists():
                    old = np.load(dest, allow_pickle=False)
                    c.require(
                        old.dtype == np.float32
                        and old.shape == values.shape
                        and np.allclose(old, values, rtol=0, atol=1e-4),
                        "An interrupted inference chunk differs.",
                    )
                    values = old
                else:
                    c.array(dest, values)
                c.save(
                    chunk.with_suffix(".json"), dict(identity=ident, sha256=c.sha(dest))
                )
                output[start:stop] = values
        finally:
            del tok, model
            gc.collect()
            torch.cuda.empty_cache()
    c.array(root / "logits.npy", output)
    products = {
        f"chunks/{start:06d}_{min(start+512,len(inputs)):06d}.{ext}": c.sha(
            root / "chunks" / f"{start:06d}_{min(start+512,len(inputs)):06d}.{ext}"
        )
        for start in range(0, len(inputs), 512)
        for ext in ("npy", "json")
    }
    products["logits.npy"] = c.sha(root / "logits.npy")
    c.save(
        root / "complete.json",
        dict(
            binding=dict(
                identity=identity(job),
                training_receipt_sha256=source_sha,
                input_sha256=c.signature(inputs),
                labels=spec["labels"],
                rows=len(inputs),
            ),
            products=products,
        ),
    )
    check_inference(job, inputs, spec["labels"])


def assemble(pool):
    chosen = c.read(r.OUTPUT / "selections" / f"{pool}.json")
    selected = chosen["families"]
    r.verify_tree(r.OUTPUT, chosen["search_receipts"])
    data = p.data(pool)
    inputs = c.read(p.PACKAGE / "data/inference.json")
    output = {}
    checks = 0
    for job in p.final_jobs(pool, selected):
        saved, attempt = receipt(job)
        audit.metadata(job, data, attempt, model_directory=saved["model_directory"])
        expected_saved(data, job, saved, attempt)
        for path in (
            job_dir(job) / "complete.json",
            r.OUTPUT / "inference" / job["id"] / "complete.json",
        ):
            output[path.relative_to(r.OUTPUT).as_posix()] = c.sha(path)
    for family, heads in selected.items():
        for seed in p.FINAL_SEEDS:
            probs = {}
            for head in p.HEADS:
                job = p.job(pool, head, heads[head]["selected"]["config"], seed, None)
                z = check_inference(job, inputs, data["heads"][head]["labels"])
                probs[head] = c.probabilities(head, z)
                other = (
                    expit(z.astype(float))
                    if head in ("hate_type", "target_group")
                    else softmax(z.astype(float), axis=1)
                )
                c.require(
                    np.allclose(probs[head], other, rtol=0, atol=1e-14),
                    "Independent probabilities differ.",
                )
            for cutoff in ("selected", "default"):
                path = (
                    r.OUTPUT / "predictions" / pool / f"{family}_{cutoff}_s{seed}.npz"
                )
                values = audit.decoded(probs, heads, cutoff)
                c.arrays(path, values)
                output[path.relative_to(r.OUTPUT).as_posix()] = c.sha(path)
                checks += sum(x.size for x in values.values())
    value = dict(
        status="pass",
        pool=pool,
        selection_sha256=c.sha(r.OUTPUT / "selections" / f"{pool}.json"),
        final_models=len(p.final_jobs(pool, selected)),
        text_predictions=len(inputs) * 30,
        decision_bits=checks,
        outputs=output,
        evaluation_gold_used=False,
    )
    c.save(r.OUTPUT / "audits" / f"{pool}_complete.json", value)
    return value
