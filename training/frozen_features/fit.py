import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import io
import time
from pathlib import Path
import numpy as np
from training.frozen_features import compact as old
from analysis.common import study as c
from analysis.common.api import save_new

HEADS = old.HEADS
CS = old.CS
THRESHOLDS = old.THRESHOLDS


def save_array(path, values):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    if dest.exists():
        c.require(dest.read_bytes() == buffer.getvalue(), "Saved array differs")
        return
    with dest.open("xb") as f:
        f.write(buffer.getvalue())
        f.flush()
        os.fsync(f.fileno())


def save_rows(path, rows):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    import json

    payload = "".join(
        (
            json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for r in rows
        )
    ).encode("utf8")
    if dest.exists():
        c.require(dest.read_bytes() == payload, "Saved rows differ")
        return
    temporary = dest.with_suffix(dest.suffix + ".writing")
    c.require(not temporary.exists(), "Unfinished publication; inspect before retry")
    with temporary.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    c.require(not dest.exists(), "Destination appeared while publishing")
    temporary.rename(dest)


def truth_bits(head, spec):
    if not spec["rows"]:
        return np.empty((0, 1 if head == "hate" else len(spec["labels"])), dtype=int)
    return old.target_bits(head, spec)


def make_oof(data, x, seed):

    folds = np.asarray(data["folds"][str(seed)], dtype=int)
    c.require(
        len(folds) == len(x) and set(folds) <= set(range(5)), "Wrong fold assignment"
    )
    cached = {}
    counts = dict(fits=0, constant_tasks=0, empty_fit_tasks=0, empty_held_folds=0)
    for head in HEADS:
        spec = data["heads"][head]
        yy = truth_bits(head, spec)
        ix = np.asarray([r["feature_row"] for r in spec["rows"]], dtype=int)
        cached[head] = {}
        for strength in CS:
            prob = np.full(yy.shape, np.nan, dtype=float)
            for k in range(5):
                train = folds[ix] != k
                held = ~train
                if not held.any():
                    counts["empty_held_folds"] += 1
                    continue
                for j in range(yy.shape[1]):
                    if not train.any():
                        model = {"constant": 0}
                        counts["empty_fit_tasks"] += 1
                    else:
                        model = old.binary_fit(
                            x[ix[train]], yy[train, j], strength, seed
                        )
                        counts["constant_tasks" if "constant" in model else "fits"] += 1
                    prob[held, j] = old.binary_predict(model, x[ix[held]])
            c.require(np.isfinite(prob).all(), "Incomplete OOF scores")
            cached[head][str(strength)] = prob
    return (cached, counts)


def fit_recipe(
    data,
    x,
    seed,
    cached,
    *,
    tune_c=True,
    tune_threshold=True,
    original_fit=False,
    locked_selection=None
):
    heads = {}
    gate = None
    for head in HEADS:
        spec = data["heads"][head]
        yy = truth_bits(head, spec)
        ix = np.asarray([r["feature_row"] for r in spec["rows"]], dtype=int)
        truth = np.column_stack((1 - yy[:, 0], yy[:, 0])) if head == "hate" else yy
        supported = np.flatnonzero(
            (truth.sum(axis=0) >= 2) & ((1 - truth).sum(axis=0) >= 2)
        )
        candidates = []
        for strength in CS if tune_c else (1.0,):
            probs = cached[head][str(strength)]
            cutoffs = (
                (None,)
                if head == "severity"
                else THRESHOLDS if tune_threshold else (0.5,)
            )
            for cutoff in cutoffs:
                decision = old.decisions(
                    head, probs, cutoff, None if head == "hate" else gate[ix]
                )
                score = (
                    old.macro(truth, decision, supported) if len(supported) else None
                )
                candidates.append({"C": strength, "threshold": cutoff, "score": score})
        if locked_selection is not None:
            selected = dict(locked_selection[head]["selected"])
        elif len(supported):
            selected = old.choose(candidates)
        else:
            selected = {
                "C": 1.0,
                "threshold": None if head == "severity" else 0.5,
                "score": None,
            }
        oof = cached[head][str(selected["C"])]
        if head == "hate":
            gate = old.decisions(head, oof, selected["threshold"])[:, 1].astype(bool)
        fitmask = np.asarray(
            [
                r.get("original_partition") == "train" if original_fit else True
                for r in spec["rows"]
            ],
            dtype=bool,
        )
        models = [
            (
                old.binary_fit(x[ix[fitmask]], yy[fitmask, j], selected["C"], seed)
                if fitmask.any()
                else {"constant": 0}
            )
            for j in range(yy.shape[1])
        ]
        heads[head] = {
            "labels": spec["labels"],
            "selected": selected,
            "candidates": candidates,
            "selection_indices": supported.tolist(),
            "models": models,
            "training_ids": [r["id"] for r, on in zip(spec["rows"], fitmask) if on],
            "permitted_annotation_rows": len(spec["rows"]),
            "untrained": not fitmask.any(),
            "selection_fallback": not len(supported),
            "oof_probabilities": oof.tolist(),
        }
    return heads


def probabilities(heads, x):
    return {
        h: np.column_stack([old.binary_predict(m, x) for m in heads[h]["models"]])
        for h in HEADS
    }


def decode(heads, probs, *, ungated=False):
    n = len(probs["hate"])
    values = {}
    for head in HEADS:
        values[head] = old.decisions(
            head,
            probs[head],
            heads[head]["selected"]["threshold"],
            None if head == "hate" or ungated else values["hate"][:, 1],
        )
        if heads[head].get("untrained") and head != "hate":
            values[head][:] = 0
    output = []
    for i in range(n):
        row = dict(
            hate=bool(values["hate"][i, 1]),
            hate_type=[],
            target_group=[],
            severity=None,
            legal=[],
        )
        for head in HEADS[1:]:
            labs = [
                lab for j, lab in enumerate(heads[head]["labels"]) if values[head][i, j]
            ]
            row[head] = (labs[0] if labs else None) if head == "severity" else labs
        output.append(row)
    return output


def check_selection(data, seed, heads, cached):

    from sklearn.metrics import f1_score

    gate = None
    checks = 0
    for head in HEADS:
        spec = data["heads"][head]
        yy = truth_bits(head, spec)
        truth = np.column_stack((1 - yy[:, 0], yy[:, 0])) if head == "hate" else yy
        idx = heads[head]["selection_indices"]
        ix = [r["feature_row"] for r in spec["rows"]]
        for candidate in heads[head]["candidates"]:
            p = cached[head][str(candidate["C"])]
            if head == "hate":
                z = p[:, 0] > candidate["threshold"]
                pred = np.column_stack((~z, z))
            elif head == "severity":
                pred = np.eye(p.shape[1], dtype=int)[p.argmax(axis=1)]
            else:
                pred = (p >= candidate["threshold"]).astype(int)
            if head != "hate":
                pred[~gate[ix]] = 0
            expected = (
                float(
                    np.mean(
                        [
                            f1_score(truth[:, j], pred[:, j], zero_division=0)
                            for j in idx
                        ]
                    )
                )
                if idx
                else None
            )
            c.require(
                expected == candidate["score"]
                or (
                    expected is not None and abs(expected - candidate["score"]) < 1e-12
                ),
                "Independent tuning score mismatch",
            )
            checks += 1
        if head == "hate":
            choice = heads[head]["selected"]
            gate = cached[head][str(choice["C"])][:, 0] > choice["threshold"]
    return checks
