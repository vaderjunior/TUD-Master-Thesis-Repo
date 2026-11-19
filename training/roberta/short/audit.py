import math
import numpy as np
from unittest.mock import patch
from scipy.special import expit, softmax
from training.roberta.short import common as s
from training.roberta import audit as old
from training.roberta.short import train as train

c = s.c
BASE = s.BASE


def metadata(job, data, attempt, epochs_required=False):
    meta = c.read(attempt / "training.json")
    spec, fit, held = train.split(data, job)
    cfg = c.config_map()[job["config"]]
    total = math.ceil(len(fit) / 16) * cfg["epochs"]
    c.require(
        meta["job"] == job and meta["configuration"] == cfg and not meta["pilot_only"],
        "Wrong training identity",
    )
    c.require(
        meta["fit_ids"] == [v["id"] for v in fit]
        and meta["held_ids"] == [v["id"] for v in held],
        "TRAIN membership mismatch",
    )
    c.require(
        meta["planned_steps"] == meta["attempted_steps"] == len(meta["steps"]) == total,
        "Training duration mismatch",
    )
    c.require(
        meta["successful_steps"] > 0
        and meta["successful_steps"] + meta["skipped_steps"] == total,
        "AMP step accounting",
    )
    c.require(
        sum(v["skipped"] for v in meta["steps"]) == meta["skipped_steps"],
        "AMP trace mismatch",
    )
    c.require(
        meta["class_weights"]
        == c.class_weights(
            job["head"],
            [v["target"] for v in fit],
            len(spec["labels"]),
            cfg["loss"] == "balanced",
        ),
        "Fold-only loss weights mismatch",
    )
    old.require_learning_rate_trace(meta)
    c.require(
        len(meta["epochs"]) == cfg["epochs"]
        and [x["epoch"] for x in meta["epochs"]] == list(range(1, cfg["epochs"] + 1)),
        "Incomplete epoch trace",
    )
    c.require(
        all(float(v) > 0 for v in meta["changes"].values()),
        "Encoder/head did not update",
    )
    if held:
        z = np.load(attempt / "held_logits.npy", allow_pickle=False)
        c.require(
            z.shape == (len(held), len(spec["labels"])) and np.isfinite(z).all(),
            "Held logit matrix invalid",
        )
        if epochs_required:
            paths = list((attempt / "epoch_logits").glob("*.npy"))
            c.require(len(paths) == cfg["epochs"], "Missing per-epoch held logits")
            y = np.asarray([v["target"] for v in held])
            errors = []
            for e in meta["epochs"]:
                q = np.load(
                    attempt / "epoch_logits" / f"{e['epoch']:03d}.npy",
                    allow_pickle=False,
                )
                c.require(
                    q.shape == z.shape and np.isfinite(q).all(),
                    "Epoch logit matrix invalid",
                )

                if job["head"] in ("hate_type", "target_group"):
                    loss = float(np.mean(np.logaddexp(0, q.astype(float)) - y * q))
                else:
                    q64 = q.astype(float)
                    maximum = q64.max(1)
                    lse = maximum + np.log(np.exp(q64 - maximum[:, None]).sum(1))
                    loss = float(np.mean(lse - q64[np.arange(len(y)), y]))
                error = abs(loss - e["held_unweighted_loss"])
                c.require(error < 2e-5, "Epoch saved loss/logits disagree")
                errors.append(error)
            c.require(
                np.array_equal(
                    z,
                    np.load(
                        attempt / "epoch_logits" / f"{cfg['epochs']:03d}.npy",
                        allow_pickle=False,
                    ),
                ),
                "Terminal logits differ from final epoch",
            )
    else:
        mm = c.read(attempt / "model/study_meta.json")
        c.require(
            mm
            == dict(
                job=job,
                labels=spec["labels"],
                kind=spec["kind"],
                config=cfg,
                pilot_only=False,
            ),
            "Final export identity mismatch",
        )
        c.require(
            meta["export_check"]["clean_reload"]
            and meta["export_check"]["max_abs_logit_difference"] <= 1e-4,
            "Final CPU export probe failed",
        )
    return meta


def load_old(data):
    with s.isolated(s.OLD, expanded=False):
        oof, receipts, steps = old.load_oof(data, s.OLD_PLAN_SHA)
        selection = c.select(data, oof)
        n = old.check_selection(data, oof, selection)
        c.require(
            selection == c.read(s.OLD / "selection.json")["families"],
            "Original selection no longer reconstructs",
        )
    c.require(len(receipts) == 720 and n == 392, "Original search coverage differs")
    return oof, receipts, steps


def load_search(data, plan_sha):
    previous, old_receipts, old_steps = load_old(data)
    with s.isolated(), patch.object(
        c, "configs", lambda: s.configs(s.SHORT)
    ), patch.object(c, "search_jobs", s.search_jobs):
        fresh, new_receipts, new_steps = old.load_oof(data, plan_sha)
    c.require(len(new_receipts) == 1800, "Short grid incomplete")
    with s.isolated():
        for job in s.search_jobs():
            _, attempt = c.verify_receipt(
                c.job_dir(job) / "complete.json", job, plan_sha
            )
            metadata(job, data, attempt, epochs_required=True)
    combined = {h: {**previous[h], **fresh[h]} for h in c.HEADS}
    c.require(
        all(set(v) == {x["id"] for x in s.configs()} for v in combined.values()),
        "Union grid incomplete",
    )
    return combined, dict(
        previous=old_receipts,
        new=new_receipts,
        old_attempted_steps=old_steps,
        new_attempted_steps=new_steps,
    )


def select(data, plan_sha):
    oof, records = load_search(data, plan_sha)
    with s.isolated():
        selected = c.select(data, oof)
        checks = old.check_selection(data, oof, selected)
    c.require(checks == 1232, "Expanded selection coverage differs")
    original = c.read(s.OLD / "selection.json")["families"]
    for family in ["fixed_unweighted", "fixed_balanced"]:
        c.require(
            selected[family] == original[family],
            "Fixed-family historical controls changed",
        )
    s.save(
        BASE / "selection.json",
        dict(plan_sha256=plan_sha, families=selected, search_receipts=records),
    )
    s.save(
        BASE / "search_audit.json",
        dict(
            status="pass",
            plan_sha256=plan_sha,
            new_fits=1800,
            reused_fits=720,
            candidate_scores=checks,
            new_attempted_steps=records["new_attempted_steps"],
            old_attempted_steps=records["old_attempted_steps"],
            selection_sha256=c.sha(BASE / "selection.json"),
            fixed_families_reproduce=True,
            head_native_evaluation_gold_used=False,
        ),
    )
    return selected


def routes(selected):
    oldids = {j["id"] for j in s.original_final_jobs()}
    return [
        dict(
            job=j,
            origin="historical_saved_predictions" if j["id"] in oldids else "new_fit",
        )
        for j in s.all_jobs(selected)
    ]


def final_check(data, selected, plan_sha):
    outputs = {}
    archive = s.archive_map()
    reused = []
    new = []
    inputs = c.read(BASE / "data/inference.json")
    with s.isolated():
        for route in routes(selected):
            job = route["job"]
            if route["origin"] == "historical_saved_predictions":
                _, attempt, missing = s.verify_old_training(job, archive)
                rp = s.OLD / "refits" / job["id"] / "complete.json"
                ip = s.OLD / "inference" / job["id"] / "complete.json"
                source_plan = s.OLD_PLAN_SHA
                reused.append(dict(job=job, archived_weights_not_rehashed_now=missing))
            else:
                rp = BASE / "refits" / job["id"] / "complete.json"
                _, attempt = c.verify_receipt(rp, job, plan_sha)
                ip = BASE / "inference" / job["id"] / "complete.json"
                source_plan = plan_sha
                new.append(job)
            metadata(job, data, attempt)
            s.compare_logits(
                ip,
                job,
                source_plan,
                c.sha(rp),
                inputs,
                data["heads"][job["head"]]["labels"],
            )
            outputs.update(s.hashes([rp, ip]))
            outputs.update(c.read(ip)["artifacts"])
    return dict(
        status="pass",
        plan_sha256=plan_sha,
        reused=reused,
        new=new,
        outputs=outputs,
        note="Archived weights are not needed or asserted present: only provenance-verified saved predictions reused. Restore/recheck archive before future fresh inference.",
    )


def assemble(data, selection, plan_sha):
    inputs = c.read(BASE / "data/inference.json")
    route = {x["job"]["id"]: x for x in routes(selection)}
    outputs = {}
    checks = 0
    shared = c.read(BASE / "data/scopes.json")
    raw_checks = 0
    for family, heads in selection.items():
        for seed in c.FINAL_SEEDS:
            probs = {}
            for head in c.HEADS:
                ident = f"{head}/{heads[head]['selected']['config']}/s{seed}"
                entry = route[ident]
                folder = (
                    s.OLD if entry["origin"] == "historical_saved_predictions" else BASE
                )
                logits = np.load(
                    folder / "inference" / ident / "logits.npy", allow_pickle=False
                )
                probs[head] = c.probabilities(head, logits)
                other = (
                    expit(logits.astype(float))
                    if head in ("hate_type", "target_group")
                    else softmax(logits.astype(float), axis=1)
                )
                c.require(
                    np.allclose(probs[head], other, rtol=0, atol=1e-14),
                    "Independent probability conversion differs",
                )
                raw_checks += logits.size
            for policy in ["selected", "default"]:
                decoded = {}
                gate = None
                for head in c.HEADS:
                    cutoff = (
                        None
                        if head == "severity"
                        else (
                            0.5
                            if policy == "default"
                            else heads[head]["selected"]["threshold"]
                        )
                    )
                    raw = c.decisions(head, probs[head], cutoff)
                    given = raw.copy()
                    if head == "hate":
                        gate = raw[:, 1].astype(bool)
                    else:
                        given[~gate] = 0

                    if head == "hate":
                        x = probs[head][:, 1] > cutoff
                        expected = np.column_stack((~x, x))
                    elif head == "severity":
                        expected = np.eye(probs[head].shape[1], dtype=bool)[
                            probs[head].argmax(1)
                        ]
                    else:
                        expected = probs[head] >= cutoff
                    if head != "hate":
                        expected[~gate] = False
                    c.require(np.array_equal(given, expected), "Decoded output differs")
                    decoded[head] = given.astype(np.uint8)
                    decoded[head + "_ungated"] = raw.astype(np.uint8)
                    checks += given.size
                path = BASE / "predictions" / f"{family}_{policy}_s{seed}.npz"
                s.arrays(path, decoded)
                outputs.update(s.hashes([path]))
                if family != "tuned":
                    with np.load(
                        s.OLD / "predictions" / path.name, allow_pickle=False
                    ) as oldz:
                        c.require(
                            all(np.array_equal(oldz[h], decoded[h]) for h in c.HEADS),
                            "Fixed historical predictions changed",
                        )
    return dict(
        status="pass",
        plan_sha256=plan_sha,
        delivered_decision_bits_checked=checks,
        raw_logit_values_checked=raw_checks,
        variants=6,
        seeds=list(c.FINAL_SEEDS),
        complete_outputs=len(inputs) * 30,
        outputs=outputs,
        fixed_historical_predictions_reproduce=True,
    )
