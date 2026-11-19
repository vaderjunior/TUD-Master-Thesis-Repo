from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from sklearn.metrics import f1_score
from analysis.classification.supervised import archive as r

BASE = r.ROOT / "experiments/astra/sq2_gpu_training_review_20261103_v1"
RECOVERY = r.ROOT / "artifacts/sq2_recovery_20261030_v1"
CV_SEEDS = (0, 1, 2)
CUTOFFS = tuple(i / 10 for i in range(1, 10))
CS = (0.1, 1.0, 10.0, 100.0)
WATCHED = {
    "roberta.encoder.layer.0.attention.self.query.weight",
    "roberta.encoder.layer.11.attention.self.query.weight",
    "classifier.out_proj.weight",
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        out.write(r.encoded(value))


def require(ok, message):
    if not ok:
        raise r.IntegrityError(message)


def truth(head, spec):
    y = np.asarray([x["target"] for x in spec["rows"]], dtype=np.int64)
    result = (
        y
        if head in ("hate_type", "target_group")
        else np.eye(len(spec["labels"]), dtype=np.int64)[y]
    )
    require(
        result.shape == (len(spec["rows"]), len(spec["labels"]))
        and ((result == 0) | (result == 1)).all(),
        "Malformed TRAIN target.",
    )
    return result


def supported_columns(y):
    return np.flatnonzero((y.sum(0) >= 2) & ((1 - y).sum(0) >= 2)).tolist()


def macro_score(y, pred, supported):
    require(
        y.shape == pred.shape
        and len(supported) > 0
        and ((pred == 0) | (pred == 1)).all(),
        "Invalid TRAIN score inputs.",
    )
    width = len(supported)
    codes = (2 * y[:, supported] + pred[:, supported] + 4 * np.arange(width)).astype(
        np.int64
    )
    counts = np.bincount(codes.ravel(), minlength=4 * width).reshape(width, 4)
    denom = 2 * counts[:, 3] + counts[:, 1] + counts[:, 2]
    per_label = np.divide(2 * counts[:, 3], denom, out=np.zeros(width), where=denom > 0)
    return float(per_label.mean())


def decisions(head, probs, threshold, gate=None):
    require(
        probs.ndim == 2
        and np.isfinite(probs).all()
        and ((probs >= 0) & (probs <= 1)).all(),
        "Invalid TRAIN probabilities.",
    )
    if head == "hate":
        p = probs[:, -1] > threshold
        result = np.column_stack((~p, p)).astype(np.int64)
    elif head == "severity":
        result = np.zeros(probs.shape, dtype=np.int64)
        result[np.arange(len(probs)), probs.argmax(1)] = 1
    else:
        result = (probs >= threshold).astype(np.int64)
    if gate is not None:
        require(
            gate.shape == (len(result),), "Gate rows differ from held fine-head rows."
        )
        result[~gate] = 0
    return result


def neural_rank(candidate, configs):
    cfg = configs[candidate["config"]]
    cutoff = 0.5 if candidate["threshold"] is None else candidate["threshold"]
    return (
        -candidate["mean_score"],
        cfg["epochs"],
        cfg["learning_rate"],
        cfg["loss"] != "unweighted",
        abs(cutoff - 0.5),
        -cutoff,
    )


def linear_rank(candidate):
    cutoff = 0.5 if candidate["threshold"] is None else candidate["threshold"]
    return (-candidate["score"], candidate["C"], abs(cutoff - 0.5), -cutoff)


def class_weights(head, y, balanced):
    if head in ("hate_type", "target_group"):
        pos = y.sum(0)
        neg = len(y) - pos
        counts = np.stack((neg, pos), axis=1)
        weights = np.ones(counts.shape)
        active = (pos > 0) & (neg > 0)
        if balanced:
            weights[active] = len(y) / (2 * counts[active])
    else:
        counts = y.sum(0)
        weights = np.ones(len(counts))
        if balanced:
            active = counts > 0
            weights[active] = len(y) / (active.sum() * counts[active])
    return dict(counts=counts.tolist(), weights=weights.tolist(), balanced=balanced)


def data_review(data, input_rows):
    n = len(data["texts"])
    require(
        n == len(data["gate_ids"]) == len(set(data["gate_ids"])),
        "TRAIN identities repeated/missing.",
    )
    require(set(data["folds"]) == {"42", "43", "44"}, "TRAIN group assignments differ.")
    clusters = [r.signature(r.normalized(x)) for x in data["texts"]]
    eval_clusters = {r.signature(r.normalized(x["text"])) for x in input_rows}
    require(
        not set(clusters) & eval_clusters, "Normalized TRAIN/evaluation text overlap."
    )
    for fold_id, folds in data["folds"].items():
        require(
            len(folds) == n and all(type(x) is int and 0 <= x < 5 for x in folds),
            "Bad TRAIN fold.",
        )
        membership = defaultdict(set)
        for cluster, fold in zip(clusters, folds):
            membership[cluster].add(fold)
        require(
            all(len(x) == 1 for x in membership.values()),
            "TRAIN text group crosses folds.",
        )
    result = dict(
        texts=n, unique_text_clusters=len(set(clusters)), evaluation_overlap=0, heads={}
    )
    splits = {}
    for head in r.HEADS:
        spec = data["heads"][head]
        y = truth(head, spec)
        require(
            len(spec["rows"]) == len({x["id"] for x in spec["rows"]})
            and len(set(spec["labels"])) == len(spec["labels"]),
            "Repeated annotated identity/label.",
        )
        for row in spec["rows"]:
            ix = row["feature_row"]
            require(
                type(ix) is int
                and 0 <= ix < n
                and row["id"] == data["gate_ids"][ix]
                and row["text_hash"] == r.signature(data["texts"][ix])
                and row["cluster"] == clusters[ix],
                "TRAIN row/text/feature identity differs.",
            )
        if head == "hate":
            require(
                [x["id"] for x in spec["rows"]] == data["gate_ids"],
                "Hate gate does not cover the complete permitted pool.",
            )
        supported = supported_columns(y)
        require(bool(supported), "No supported TRAIN selection label.")
        fold_table = []
        for fold in (*range(5), None):
            held_ix = [
                i
                for i, x in enumerate(spec["rows"])
                if fold is not None and data["folds"]["42"][x["feature_row"]] == fold
            ]
            held_set = set(held_ix)
            fit_ix = [i for i, x in enumerate(spec["rows"]) if i not in held_set]
            fit = [spec["rows"][i] for i in fit_ix]
            held = [spec["rows"][i] for i in held_ix]
            require(
                fit and not {x["cluster"] for x in fit} & {x["cluster"] for x in held},
                "Fitting/held groups overlap or fitting set empty.",
            )
            splits[head, fold] = dict(
                spec=spec,
                fit=fit,
                held=held,
                fit_indices=fit_ix,
                held_indices=held_ix,
                weights={
                    loss: class_weights(head, y[fit_ix], loss == "balanced")
                    for loss in ("unweighted", "balanced")
                },
            )
            if fold is not None:
                fold_table.append(
                    dict(
                        fold=fold,
                        fit=len(fit),
                        held=len(held),
                        fit_positive=y[fit_ix].sum(0).tolist(),
                        held_positive=y[held_ix].sum(0).tolist(),
                    )
                )
        result["heads"][head] = dict(
            rows=len(y),
            labels=spec["labels"],
            positives=y.sum(0).tolist(),
            selection_indices=supported,
            selection_labels=[spec["labels"][i] for i in supported],
            unsupported_labels=[
                x for i, x in enumerate(spec["labels"]) if i not in supported
            ],
            sources=dict(Counter(x["source"] for x in spec["rows"])),
            folds=fold_table,
        )
    return result, splits


def trace_review(meta, job, split, cfg, update_rules, equivalence=None):
    fit, held = split["fit"], split["held"]
    total = math.ceil(len(fit) / 16) * cfg["epochs"]
    steps = meta["steps"]
    require(
        meta["job"] == job
        and meta["configuration"] == cfg
        and meta["pilot_only"] is False,
        "Training identity/configuration/pilot differs.",
    )
    require(
        meta["fit_ids"] == [x["id"] for x in fit]
        and meta["held_ids"] == [x["id"] for x in held],
        "Training partition order differs.",
    )
    require(
        meta["class_weights"] == split["weights"][cfg["loss"]],
        "Class weights did not use fitting rows only.",
    )
    require(
        meta["planned_steps"] == meta["attempted_steps"] == len(steps) == total,
        "Training horizon differs.",
    )
    require(
        meta["optimizer"]
        == "AdamW; correct bias correction; no bias/LayerNorm weight decay; linear warmup/decay"
        and meta["training_precision"] == "FP16 autocast, FP32 parameters"
        and meta["inference_precision"] == "FP32; no autocast",
        "Numerical recipe differs.",
    )
    load = meta["loading"]
    require(
        not load["error_msgs"]
        and not load["mismatched_keys"]
        and set(load["missing_keys"])
        == {
            "classifier.dense.weight",
            "classifier.dense.bias",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
        }
        and all(
            x.startswith("lm_head.") or x.endswith(("position_ids", "token_type_ids"))
            for x in load["unexpected_keys"]
        ),
        "Pretrained/model-head initialization differs.",
    )
    success = skipped = positive = zero = consecutive = longest = nonfinite = 0
    warm = math.ceil(0.1 * total)
    previous_scale = 65536.0
    epoch_success = []
    epoch_attempted = []
    batches = math.ceil(len(fit) / 16)
    for i, step in enumerate(steps, 1):
        require(
            step["step"] == i
            and step["epoch"] == (i - 1) // batches + 1
            and type(step["skipped"]) is bool,
            "Step/epoch order differs.",
        )
        factor = (
            success / max(1, warm)
            if success < warm
            else max(0.0, (total - success) / max(1, total - warm))
        )
        require(
            abs(step["lr"] - cfg["learning_rate"] * factor) < 1e-14
            and math.isfinite(step["lr"]),
            "Learning-rate schedule differs.",
        )
        before, after = step["scale_before"], step["scale_after"]
        require(
            before == previous_scale
            and math.isfinite(after)
            and after > 0
            and step["skipped"] == (after < before),
            "AMP scale/skip trace differs.",
        )
        require(
            after == before / 2 if step["skipped"] else after in (before, 2 * before),
            "Unexpected AMP scale transition.",
        )
        norm = float(step["grad_norm"])
        if not math.isfinite(norm):
            require(step["skipped"], "Nonfinite gradient was accepted.")
            nonfinite += 1
        if step["skipped"]:
            skipped += 1
            consecutive += 1
            longest = max(longest, consecutive)
        else:
            success += 1
            consecutive = 0
            if step["lr"] > 0:
                positive += 1
            else:
                zero += 1
        require(consecutive <= 20, "Consecutive AMP skip guard exceeded.")
        previous_scale = after
        if i % batches == 0 or i == total:
            epoch_success.append(success)
            epoch_attempted.append(i)
    require(
        meta["successful_steps"] == success
        and meta["skipped_steps"] == skipped
        and success + skipped == total,
        "Recorded update counts differ.",
    )
    require(len(meta["epochs"]) == cfg["epochs"], "Missing/extra epochs.")
    for i, epoch in enumerate(meta["epochs"]):
        require(
            epoch["epoch"] == i + 1
            and epoch["attempted_steps"] == epoch_attempted[i]
            and epoch["successful_steps"] == epoch_success[i]
            and math.isfinite(epoch["training_objective"]),
            "Epoch trace differs.",
        )
        require(
            ("held_unweighted_loss" in epoch) == bool(held), "Held-loss scope differs."
        )
        if held:
            require(
                math.isfinite(epoch["held_unweighted_loss"]), "Nonfinite held loss."
            )
    state = update_rules.verify(meta, equivalence)
    require(
        set(meta["changes"]) == WATCHED
        and all(math.isfinite(v) and v >= 0 for v in meta["changes"].values()),
        "Invalid weight-change proof.",
    )
    if job["mode"] == "final":
        check = meta["export_check"]
        require(
            check["clean_reload"] is True
            and check["tolerance"] == 1e-4
            and 0 <= check["max_abs_logit_difference"] <= 1e-4,
            "Final export/reload proof differs.",
        )
    else:
        require(
            meta["model_directory"] is None and meta["export_check"] is None,
            "Search fit unexpectedly exported as a final model.",
        )
    return dict(
        state=state,
        planned_steps=total,
        successful_steps=success,
        skipped_steps=skipped,
        positive_updates=positive,
        accepted_zero_rate_steps=zero,
        longest_skip_streak=longest,
        nonfinite_skipped_gradients=nonfinite,
        fit_rows=len(fit),
        held_rows=len(held),
        first_objective=meta["epochs"][0]["training_objective"],
        last_objective=meta["epochs"][-1]["training_objective"],
        first_held_loss=meta["epochs"][0].get("held_unweighted_loss"),
        last_held_loss=meta["epochs"][-1].get("held_unweighted_loss"),
        train_seconds=meta["train_seconds"],
        total_seconds=meta["total_seconds"],
        fit_token_at128=meta["token_lengths"]["fit_at128"],
        fit_token_max=meta["token_lengths"]["fit_max"],
        export_max_abs_logit_difference=(
            None
            if meta["export_check"] is None
            else meta["export_check"]["max_abs_logit_difference"]
        ),
    )


def fill_oof(dest, seed_index, held_indices, logits):
    require(
        logits.shape == dest[seed_index, held_indices].shape
        and np.isfinite(logits).all(),
        "Held probability dimensions/values differ.",
    )
    require(np.isnan(dest[seed_index, held_indices]).all(), "OOF row assigned twice.")
    dest[seed_index, held_indices] = logits


def review_selection(data, families, oof, configs, states):
    result = []
    candidate_rows = []
    checks = 0
    max_error = 0.0
    sklearn_checks = 0
    for family in r.FAMILIES:
        gate = None
        for head in r.HEADS:
            spec = data["heads"][head]
            y = truth(head, spec)
            saved = families[family][head]
            supported = supported_columns(y)
            require(
                supported == saved["selection_indices"], "Wrong TRAIN label support."
            )
            allowed = (
                list(configs)
                if family == "tuned"
                else ["l1_e20_u" if family == "fixed_unweighted" else "l1_e20_b"]
            )
            expected = {
                (cfg, t)
                for cfg in allowed
                for t in ((None,) if head == "severity" else CUTOFFS)
            }
            candidates = saved["candidates"]
            require(
                len(candidates) == len(expected)
                and {(x["config"], x["threshold"]) for x in candidates} == expected,
                "Candidate grid differs.",
            )
            recomputed = []
            indices = [x["feature_row"] for x in spec["rows"]]
            for candidate in candidates:
                scores = []
                for si, seed in enumerate(CV_SEEDS):
                    z = decisions(
                        head,
                        oof[head][candidate["config"]][si],
                        candidate["threshold"],
                        None if gate is None else gate[si, indices],
                    )
                    scores.append(macro_score(y, z, supported))
                    if candidate == saved["selected"]:
                        reference = float(
                            np.mean(
                                [
                                    f1_score(y[:, j], z[:, j], zero_division=0)
                                    for j in supported
                                ]
                            )
                        )
                        require(
                            abs(reference - scores[-1]) < 1e-12,
                            "Independent sklearn selected score differs.",
                        )
                        sklearn_checks += 1
                mean = float(np.mean(scores))
                error = max(
                    abs(mean - candidate["mean_score"]),
                    max(abs(a - b) for a, b in zip(scores, candidate["seed_scores"])),
                )
                require(
                    len(candidate["seed_scores"]) == 3 and error < 1e-12,
                    "Saved TRAIN candidate score differs.",
                )
                max_error = max(max_error, error)
                checks += 1
                recomputed.append(
                    dict(
                        config=candidate["config"],
                        threshold=candidate["threshold"],
                        seed_scores=scores,
                        mean_score=mean,
                    )
                )
            original_order = sorted(candidates, key=lambda x: neural_rank(x, configs))
            require(
                original_order[0] == saved["selected"],
                "TRAIN winner or tie ordering differs.",
            )
            computed_order = sorted(recomputed, key=lambda x: neural_rank(x, configs))
            require(
                (computed_order[0]["config"], computed_order[0]["threshold"])
                == (saved["selected"]["config"], saved["selected"]["threshold"]),
                "Recomputed winner differs.",
            )
            chosen = saved["selected"]
            cfg = chosen["config"]
            maximum = chosen["mean_score"]
            ties = [v for v in candidates if v["mean_score"] == maximum]
            other = [v for v in candidates if v["config"] != cfg]
            below = [v["mean_score"] for v in candidates if v["mean_score"] < maximum]
            state_counts = Counter(
                states[head, cfg, s, f] for s in CV_SEEDS for f in range(5)
            )
            held_state_counts = Counter()
            for s in CV_SEEDS:
                for f in range(5):
                    count = sum(
                        data["folds"]["42"][row["feature_row"]] == f
                        for row in spec["rows"]
                    )
                    held_state_counts[states[head, cfg, s, f]] += count
            seed_choices = []
            for si, seed in enumerate(CV_SEEDS):
                ranked = sorted(
                    candidates,
                    key=lambda x: neural_rank(
                        x | {"mean_score": x["seed_scores"][si]}, configs
                    ),
                )
                seed_choices.append(
                    dict(
                        seed=seed,
                        config=ranked[0]["config"],
                        threshold=ranked[0]["threshold"],
                        score=ranked[0]["seed_scores"][si],
                    )
                )
            chosen_prob = oof[head][cfg]
            row = dict(
                family=family,
                head=head,
                annotations=len(y),
                selection_labels=[spec["labels"][i] for i in supported],
                selected=chosen,
                configuration=configs[cfg],
                candidate_count=len(candidates),
                exact_best_ties=len(ties),
                tied_configs=sorted({x["config"] for x in ties}),
                margin_to_next_distinct_score=(
                    None if not below else maximum - max(below)
                ),
                margin_to_best_other_config=(
                    None if not other else maximum - max(x["mean_score"] for x in other)
                ),
                selected_search_states=dict(state_counts),
                selected_held_row_states_across_three_seeds=dict(held_state_counts),
                seedwise_choices_conditioned_on_same_family_gate=seed_choices,
                oof_mean_score_is_selection_statistic_not_test_result=True,
            )
            if head == "hate":
                gate = chosen_prob[:, :, 1] > chosen["threshold"]
                row["oof_gate_counts"] = [
                    dict(seed=s, predicted_hate=int(gate[i].sum()), rows=len(y))
                    for i, s in enumerate(CV_SEEDS)
                ]
            else:
                row["oof_gate_counts"] = [
                    dict(seed=s, rows=len(y), gate_positive=int(gate[i, indices].sum()))
                    for i, s in enumerate(CV_SEEDS)
                ]
            result.append(row)
            candidate_rows.append(dict(family=family, head=head, candidates=recomputed))
    return (
        dict(
            heads=result,
            candidate_checks=checks,
            selected_sklearn_score_checks=sklearn_checks,
            maximum_candidate_score_error=max_error,
        ),
        candidate_rows,
    )


def job_review(pool, job, split, rules):
    store = pool.store
    ident = job["id"]
    source = f"jobs/{ident}/complete.json"
    rec = store.json(source)
    cfg = pool.study.configs[job["config"]]
    identity = dict(
        job=job,
        plan_sha256=r.PLAN_SHA,
        package_sha256=r.PACKAGE_SHA,
        gpu_profile="search_8x10" if job["mode"] == "search" else "final_1x80",
    )
    require(
        rec["status"] == "complete" and rec["identity"] == identity,
        "Canonical training receipt identity differs.",
    )
    attempt = r.relative(rec["attempt"])
    require(
        attempt.rsplit("/", 1)[0] == f"jobs/{ident}/attempts",
        "Canonical attempt path differs.",
    )
    if "verification_amendment_sha256" in rec:
        require(
            rec["verification_amendment_sha256"] == r.RECOVERY_SHA,
            "Unknown numerical amendment.",
        )
    require(
        attempt + "/stopped.json" not in store.entries,
        "Stopped attempt was selected as complete.",
    )
    for name, expected in rec["products"].items():
        require(
            store.expected(attempt + "/" + r.relative(name)) == expected,
            "Saved training-product binding differs.",
        )
    began = store.json(attempt + "/started.json")
    require(
        began["identity"] == identity and len(began["session_sha256"]) == 64,
        "Training start/session binding differs.",
    )
    row = dict(
        job=job,
        attempt=attempt,
        receipt_sha256=store.expected(source),
        started_utc=began["utc"],
        session_sha256=began["session_sha256"],
        recovery_amendment=rec.get("verification_amendment_sha256"),
    )
    empty = job["mode"] == "search" and not split["held"]
    if empty:
        require(
            rec["state"] == "empty_held_fold"
            and rec["steps"] == 0
            and rec["model_directory"] is None
            and not rec["model_products"],
            "Empty held fold was fitted/exported.",
        )
        require(
            set(rec["products"]) == {"started.json", "empty_held_fold.json"},
            "Empty held fold has invented fit products.",
        )
        require(
            store.json(attempt + "/empty_held_fold.json")
            == dict(
                identity=identity,
                fit_ids=[x["id"] for x in split["fit"]],
                held_ids=[],
                fits=0,
                steps=0,
            ),
            "Empty-fold proof differs.",
        )
        row.update(
            state="empty_held_fold",
            planned_steps=0,
            successful_steps=0,
            skipped_steps=0,
            positive_updates=0,
            accepted_zero_rate_steps=0,
            longest_skip_streak=0,
            nonfinite_skipped_gradients=0,
            fit_rows=len(split["fit"]),
            held_rows=0,
        )
        return row, None
    meta = store.json(attempt + "/training.json")
    eq = (
        store.json(attempt + "/parameter_equivalence.json")
        if "parameter_equivalence.json" in rec["products"]
        else None
    )
    detail = trace_review(meta, job, split, cfg, rules, eq)
    require(
        rec["state"] == detail["state"] and rec["steps"] == detail["planned_steps"],
        "Receipt and actual update state differ.",
    )
    if detail["state"] == "amp_initialization_only":
        require(
            rec.get("verification_amendment_sha256") == r.RECOVERY_SHA,
            "Two-step state lacks its amendment.",
        )
    expected = {"started.json", "training.json", "probe_logits.npy"}
    if detail["state"] != "trained":
        expected.add("parameter_equivalence.json")
    if split["held"]:
        expected.add("held_logits.npy")
        expected |= {f"epoch_logits/{i:03d}.npy" for i in range(1, cfg["epochs"] + 1)}
    require(set(rec["products"]) == expected, "Canonical fit product coverage differs.")
    logits = None
    if split["held"]:
        logits = store.array(attempt + "/held_logits.npy")
        require(
            logits.dtype == np.float32
            and logits.shape == (len(split["held"]), len(split["spec"]["labels"])),
            "Held logits shape/type differs.",
        )
        terminal = attempt + f"/epoch_logits/{cfg['epochs']:03d}.npy"
        if store.expected(terminal) != store.expected(attempt + "/held_logits.npy"):
            require(
                np.array_equal(logits, store.array(terminal)),
                "Held logits are not the terminal-epoch logits.",
            )
        z = logits.astype(np.float64)
        targets = np.asarray([x["target"] for x in split["held"]])
        if job["head"] in ("hate_type", "target_group"):
            loss = float(np.mean(np.logaddexp(0, z) - targets * z))
        else:
            from scipy.special import logsumexp

            loss = float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(z)), targets]))
        detail["terminal_held_loss_error"] = abs(loss - detail["last_held_loss"])
        require(
            detail["terminal_held_loss_error"] < 2e-5,
            "Terminal held loss does not match saved logits.",
        )
    if job["mode"] == "final":
        model = "models/" + ident + "/" + attempt.rsplit("/", 1)[1]
        require(
            meta["model_directory"]
            == rec["model_directory"]
            == "/results/sq2_full_20261029_v1/" + model,
            "Final training/export attempts differ.",
        )
        require(
            {"model.safetensors", "config.json", "study_meta.json"}
            <= set(rec["model_products"]),
            "Incomplete final export.",
        )
        for name, expected_sha in rec["model_products"].items():
            require(
                store.expected(model + "/" + r.relative(name)) == expected_sha,
                "Final model-product binding differs.",
            )
        require(
            store.json(model + "/study_meta.json")
            == dict(
                job=job,
                labels=split["spec"]["labels"],
                kind=split["spec"]["kind"],
                config=cfg,
                pilot_only=False,
            ),
            "Final exported model identity differs.",
        )
        prior = r.read(
            r.ROOT
            / "experiments/astra/sq2_gpu_review_20261103_v1/reader_runs/1ed035a64773/pools"
            / f"{pool.pool}.json"
        )
        require(
            prior["canonical_final_attempts"][ident]["receipt_sha256"]
            == store.expected(source),
            "Final fit changed since the reader checkpoint.",
        )
    else:
        require(
            rec["model_directory"] is None and rec["model_products"] == {},
            "Search model entered the final export roster.",
        )
    row.update(detail)
    if eq is not None:
        row["parameter_equivalence"] = dict(
            proof_sha256=store.expected(attempt + "/parameter_equivalence.json"),
            whole_model_numerical_sha256=eq["before"]["zero_canonical_sha256"],
            raw_hash_equal=eq["before"]["raw_sha256"] == eq["after"]["raw_sha256"],
            negative_zeros_before=eq["before"]["negative_zero_count"],
            negative_zeros_after=eq["after"]["negative_zero_count"],
        )
    return row, logits


def historical_attempts(pool, canonical):
    attempts = defaultdict(list)
    for name in pool.store.entries:
        if name.startswith("jobs/") and "/attempts/" in name:
            prefix, tail = name.split("/attempts/", 1)
            attempts[prefix + "/attempts/" + tail.split("/")[0]].append(name)
    result = []
    for path, files in sorted(attempts.items()):
        if path in canonical:
            continue
        start = (
            pool.store.json(path + "/started.json")
            if path + "/started.json" in files
            else None
        )
        stop = (
            pool.store.json(path + "/stopped.json")
            if path + "/stopped.json" in files
            else None
        )
        if start:
            identity = start["identity"]
            require(
                identity["plan_sha256"] == r.PLAN_SHA
                and identity["package_sha256"] == r.PACKAGE_SHA
                and identity["job"]["pool"] == pool.pool,
                "Historical attempt belongs to another study.",
            )
        if stop and not stop["resumable"]:
            policy = r.read(RECOVERY / "recovery_policy.json")["reviewed_stop"]
            require(
                path + "/stopped.json" == policy["path"]
                and pool.store.expected(path + "/stopped.json") == policy["sha256"],
                "Unexpected non-resumable historical stop.",
            )
        result.append(
            dict(
                attempt=path,
                files=len(files),
                started_utc=None if start is None else start["utc"],
                job=None if start is None else start["identity"]["job"],
                stop=(
                    None
                    if stop is None
                    else {
                        k: stop.get(k) for k in ("type", "message", "utc", "resumable")
                    }
                ),
                started_sha256=(
                    None
                    if start is None
                    else pool.store.expected(path + "/started.json")
                ),
                stopped_sha256=(
                    None
                    if stop is None
                    else pool.store.expected(path + "/stopped.json")
                ),
                used_in_scientific_OOF_or_final_prediction=False,
            )
        )
    return result


def linear_counts(data, fold_id):
    counts = Counter(fits=0, constant_tasks=0, empty_fit_tasks=0, empty_held_folds=0)
    folds = np.asarray(data["folds"][str(fold_id)])
    for head in r.HEADS:
        spec = data["heads"][head]
        yy = truth(head, spec)
        if head == "hate":
            yy = yy[:, 1:2]
        ix = [x["feature_row"] for x in spec["rows"]]
        for _ in CS:
            for k in range(5):
                fit = folds[ix] != k
                held = ~fit
                if not held.any():
                    counts["empty_held_folds"] += 1
                    continue
                for column in range(yy.shape[1]):
                    if not fit.any():
                        counts["empty_fit_tasks"] += 1
                    elif len(np.unique(yy[fit, column])) < 2:
                        counts["constant_tasks"] += 1
                    else:
                        counts["fits"] += 1
    return dict(counts)


def linear_review(linear, pool, data):
    rows = []
    candidates_checked = selected_checked = historical_checks = 0
    max_error = 0.0
    for method, fold in [("frozen_roberta", 42), ("bge", 42), ("bge", 43), ("bge", 44)]:
        prefix = (r.LINEAR / "linear" / pool).relative_to(r.ROOT).as_posix()
        source_name = prefix + (
            "/fit_complete.json"
            if method == "frozen_roberta"
            else f"/bge_fold{fold}_reuse.json"
        )
        source = r.read(linear.load(source_name))
        if method == "frozen_roberta":
            require(
                source["identity"]["pool"] == pool
                and source["identity"]["fold_assignment"] == fold
                and source["identity"]["plan_sha256"] == r.LINEAR_PLAN_SHA,
                "Linear source identity differs.",
            )
            fit = r.read(linear.load(source["fit_path"]))
            cached = r.arrays_from_bytes(linear.load(source["oof_path"]).read_bytes())
            require(
                set(cached) == {h + "__" + str(v) for h in r.HEADS for v in CS},
                "Incomplete frozen-encoder OOF candidate cache.",
            )
            counts = fit["counts"]
        else:
            require(
                source["pool"] == pool
                and source["fold_assignment"] == fold
                and source["plan_sha256"] == r.LINEAR_PLAN_SHA
                and source["historical_all_candidate_checks"] == 112
                and not source["all_candidate_OOF_recomputed"],
                "BGE reuse scope differs.",
            )
            fit_name = f"experiments/astra/compact_pools_20261008_v1/jobs/{pool}_fold{fold}/fit.json"
            require(
                source["outputs"][fit_name] == linear.outputs[fit_name],
                "BGE source fit binding differs.",
            )
            fit = r.read(linear.load(fit_name))
            cached = None
            require(
                fit["job"]["pool"] == pool
                and fit["job"]["seed"] == fold
                and fit["checks"]["candidate_score_checks"] == 112,
                "Historical BGE fit/audit identity differs.",
            )
            counts = fit["checks"]["oof_counts"]
            historical_checks += 112
        require(
            counts == linear_counts(data, fold),
            "Linear sparse-fold task accounting differs.",
        )
        gate = None
        for head in r.HEADS:
            spec = data["heads"][head]
            saved = fit["heads"][head]
            y = truth(head, spec)
            support = supported_columns(y)
            require(
                saved["labels"] == spec["labels"]
                and saved["training_ids"] == [x["id"] for x in spec["rows"]]
                and saved["selection_indices"] == support
                and not saved["untrained"]
                and not saved["selection_fallback"],
                "Linear TRAIN membership/support differs.",
            )
            candidates = saved["candidates"]
            expected = {
                (v, t) for v in CS for t in ((None,) if head == "severity" else CUTOFFS)
            }
            require(
                len(candidates) == len(expected)
                and {(x["C"], x["threshold"]) for x in candidates} == expected
                and min(candidates, key=linear_rank) == saved["selected"],
                "Linear candidate grid/winner differs.",
            )
            indices = [x["feature_row"] for x in spec["rows"]]
            selected_prob = np.asarray(saved["oof_probabilities"], dtype=float)
            width = 1 if head == "hate" else len(spec["labels"])
            require(
                selected_prob.shape == (len(y), width),
                "Linear selected OOF shape differs.",
            )
            selected = saved["selected"]
            evaluated = candidates if cached is not None else [selected]
            for candidate in evaluated:
                prob = (
                    selected_prob
                    if cached is None
                    else cached[head + "__" + str(candidate["C"])]
                )
                require(
                    prob.shape == (len(y), width),
                    "Linear all-candidate OOF shape differs.",
                )
                z = decisions(
                    head,
                    prob,
                    candidate["threshold"],
                    None if gate is None else gate[indices],
                )
                value = macro_score(y, z, support)
                error = abs(value - candidate["score"])
                require(error < 1e-12, "Linear TRAIN score differs.")
                max_error = max(max_error, error)
                if cached is not None:
                    candidates_checked += 1
                if candidate == selected:
                    require(
                        np.array_equal(prob, selected_prob),
                        "Selected linear OOF cache differs.",
                    )
                    check = float(
                        np.mean(
                            [
                                f1_score(y[:, j], z[:, j], zero_division=0)
                                for j in support
                            ]
                        )
                    )
                    require(
                        abs(check - value) < 1e-12,
                        "Independent sklearn linear score differs.",
                    )
                    selected_checked += 1
            if head == "hate":
                gate = selected_prob[:, 0] > selected["threshold"]
            model_labels = ["true"] if head == "hate" else saved["labels"]
            require(
                len(saved["models"]) == len(model_labels),
                "Linear final head dimension differs.",
            )
            constants = {
                lab: m["constant"]
                for lab, m in zip(model_labels, saved["models"])
                if "constant" in m
            }
            require(
                all(x in (0, 1) for x in constants.values()),
                "Invalid constant label model.",
            )
            iterations = [
                m["iterations"] for m in saved["models"] if "constant" not in m
            ]
            require(
                all(0 < v <= 2000 for v in iterations),
                "Linear solver iteration bound differs.",
            )
            ties = sum(v["score"] == selected["score"] for v in candidates)
            rows.append(
                dict(
                    pool=pool,
                    method=method,
                    fold=fold,
                    head=head,
                    annotations=len(y),
                    selection_labels=[spec["labels"][i] for i in support],
                    selected=selected,
                    exact_best_ties=ties,
                    constant_final_labels=constants,
                    maximum_solver_iterations=max(iterations, default=0),
                    oof_task_counts=counts,
                    all_candidate_scores_recomputed=cached is not None,
                    selected_score_recomputed=True,
                    OOF_gate_source="same method, pool and fold assignment",
                )
            )
    return dict(
        heads=rows,
        new_all_candidate_checks=candidates_checked,
        new_selected_score_checks=selected_checked,
        historical_BGE_all_candidate_checks_bound=historical_checks,
        maximum_score_error=max_error,
    )


def load_rules(study):
    r.checked_file(RECOVERY / "recovery_policy.json", r.RECOVERY_SHA, study.bindings)
    from training.recovery import update_rules

    path = Path(update_rules.__file__).resolve()
    study.bindings[str(path)] = r.sha(path)
    return update_rules


def run_pool(study, linear, name, folder, rules):
    folder.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    with study.pool(name) as pool:
        data, splits = data_review(pool.data, study.inputs)
        save(folder / "data_review.json", data)
        oof = {
            h: {
                cfg: np.full(
                    (3, len(pool.data["heads"][h]["rows"]), len(pool.labels[h])), np.nan
                )
                for cfg in study.configs
            }
            for h in r.HEADS
        }
        states = {}
        records = []
        canonical = set()
        paths = sorted(pool.selection["search_receipts"]) + [
            f"jobs/{x}/complete.json" for x in pool.jobs
        ]
        for number, path in enumerate(paths, 1):
            ident = path[len("jobs/") : -len("/complete.json")]
            pp, head, cfg, last = ident.split("/")
            seed_text, _, fold_text = last.partition("_f")
            seed = int(seed_text[1:])
            fold = int(fold_text) if fold_text else None
            require(
                pp == name and seed in (r.SEEDS if fold is None else CV_SEEDS),
                "Unknown canonical job.",
            )
            job = dict(
                id=ident,
                pool=name,
                head=head,
                config=cfg,
                seed=seed,
                fold=fold,
                mode="final" if fold is None else "search",
            )
            record, logits = job_review(pool, job, splits[head, fold], rules)
            canonical.add(record["attempt"])
            records.append(record)
            if fold is not None:
                states[head, cfg, seed, fold] = record["state"]
                if logits is not None:
                    fill_oof(
                        oof[head][cfg],
                        CV_SEEDS.index(seed),
                        splits[head, fold]["held_indices"],
                        r.probabilities(head, logits),
                    )
            if number % 420 == 0:
                print(
                    name,
                    number,
                    "/",
                    len(paths),
                    "training receipts/traces checked",
                    flush=True,
                )
        require(
            all(np.isfinite(x).all() for hh in oof.values() for x in hh.values()),
            "Incomplete reconstructed TRAIN OOF matrix.",
        )
        search = [x for x in records if x["job"]["mode"] == "search"]
        final = [x for x in records if x["job"]["mode"] == "final"]
        require(
            dict(Counter(x["state"] for x in search)) == pool.selection["states"]
            and sum(x["planned_steps"] for x in search)
            == pool.selection["attempted_steps"],
            "Selection trace totals differ.",
        )
        chosen, candidates = review_selection(
            pool.data, pool.families, oof, study.configs, states
        )
        require(
            chosen["candidate_checks"]
            == pool.selection["independent_candidate_checks"]
            == 1232,
            "Candidate check count differs.",
        )
        by_id = {x["job"]["id"]: x for x in final}
        for row in chosen["heads"]:
            fits = [
                by_id[pool.job(row["head"], row["family"], seed)["id"]]
                for seed in r.SEEDS
            ]
            row["final_states"] = dict(Counter(x["state"] for x in fits))
            row["final_positive_updates"] = [x["positive_updates"] for x in fits]
            row["final_job_ids"] = [x["job"]["id"] for x in fits]
        previous = historical_attempts(pool, canonical)
        save(folder / "canonical_jobs.json", records)
        save(folder / "historical_attempts.json", previous)
        save(folder / "selection_review.json", chosen)
        save(folder / "candidate_scores.json", candidates)
        with (folder / "train_oof.npz").open("xb") as out:
            np.savez_compressed(
                out,
                **{
                    h + "__" + cfg: value
                    for h, hh in oof.items()
                    for cfg, value in hh.items()
                },
            )
        baseline = linear_review(linear, name, pool.data)
        save(folder / "linear_selection_review.json", baseline)
        groups = {}
        for mode, items in [("search", search), ("final", final)]:
            groups[mode] = dict(
                jobs=len(items),
                states=dict(Counter(x["state"] for x in items)),
                attempted_steps=sum(x["planned_steps"] for x in items),
                successful_steps=sum(x["successful_steps"] for x in items),
                skipped_steps=sum(x["skipped_steps"] for x in items),
                positive_updates=sum(x["positive_updates"] for x in items),
                accepted_zero_rate_steps=sum(
                    x["accepted_zero_rate_steps"] for x in items
                ),
                maximum_skip_streak=max(x["longest_skip_streak"] for x in items),
            )
        save(folder / "archive_bindings.json", pool.store.read_bindings)
        summary = dict(
            pool=name,
            groups=groups,
            annotated_rows={h: v["rows"] for h, v in data["heads"].items()},
            neural_candidate_checks=chosen["candidate_checks"],
            linear_all_candidate_checks=baseline["new_all_candidate_checks"],
            linear_selected_checks=baseline["new_selected_score_checks"],
            historical_BGE_candidate_checks=baseline[
                "historical_BGE_all_candidate_checks_bound"
            ],
            previous_attempts=len(previous),
            previous_stops=sum(x["stop"] is not None for x in previous),
            maximum_candidate_score_error=chosen["maximum_candidate_score_error"],
            maximum_terminal_loss_error=max(
                x.get("terminal_held_loss_error", 0) for x in records
            ),
            archive_members_read=len(pool.store.read_bindings),
            archive_payload_bytes_read=pool.store.bytes_read,
            seconds=time.perf_counter() - begin,
            outputs={x.name: r.sha(x) for x in folder.iterdir() if x.is_file()},
        )
        save(folder / "complete.json", summary)
    print(name, "TRAIN REVIEW PASSED", groups, flush=True)
    gc.collect()
    return summary


def main():
    run = BASE / "runs" / uuid.uuid4().hex[:12]
    begin = time.perf_counter()
    sources = [
        Path(__file__),
        Path(r.__file__),
        r.ROOT / "tests/classification/supervised/test_training_audit.py",
    ]
    hashes = {str(x): r.sha(x) for x in sources}
    save(
        run / "started.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            sources=hashes,
            scope="TRAIN traces, OOF selection and final fit validity only",
            test_or_DEV_scores=False,
            training_calls=0,
            Qwen_calls=0,
        ),
    )
    try:
        with r.offline():
            ready = r.read(
                r.ROOT
                / "experiments/astra/sq2_gpu_review_20261103_v1/reader_ready.json"
            )
            for name, expected in ready["bindings"].items():
                r.checked_file(r.ROOT / name, expected)
            study = r.GpuStudy()
            linear = r.LinearStudy(study)
            rules = load_rules(study)
            for name in (
                "app/selection.py",
                "app/reference.py",
                "app/audit_kernels.py",
                "app/training_kernels.py",
                "app/plan.py",
            ):
                r.checked_file(
                    r.PACKAGE / name,
                    study.package_manifest["files"][name],
                    study.bindings,
                )
            summaries = []
            for pool in r.POOLS:
                print("Starting TRAIN review:", pool, flush=True)
                summaries.append(run_pool(study, linear, pool, run / pool, rules))
            require(
                sum(x["groups"]["search"]["jobs"] for x in summaries) == 22680
                and sum(x["groups"]["final"]["jobs"] for x in summaries) == 530
                and sum(x["groups"]["search"]["attempted_steps"] for x in summaries)
                == study.plan["search_steps"],
                "Full study coverage differs.",
            )
            bindings = study.bindings | linear.bindings | hashes
            for name, expected in bindings.items():
                r.checked_file(name, expected)
            save(run / "source_bindings.json", bindings)
            require("torch" not in sys.modules, "Training runtime imported.")
            report = dict(
                status="TRAIN_validity_and_selection_pass",
                utc=datetime.now(timezone.utc).isoformat(),
                seconds=time.perf_counter() - begin,
                pools=summaries,
                source_bindings=len(bindings),
                training_calls=0,
                Qwen_calls=0,
                network_calls=0,
                model_weights_loaded=0,
                evaluation_predictions_read=0,
                test_or_DEV_scores_computed=False,
                new_bootstrap_draws=0,
                selection_changes=0,
                original_evidence_changed=False,
                outputs={
                    x.relative_to(run).as_posix(): r.sha(x)
                    for x in run.rglob("*")
                    if x.is_file()
                },
            )
            save(run / "complete.json", report)
            print(
                "ALL NINE TRAINING AND SELECTION REVIEWS PASSED",
                run.relative_to(r.ROOT),
                flush=True,
            )
    except BaseException as exc:
        import traceback

        save(
            run / "stopped.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
                original_evidence_changed=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
