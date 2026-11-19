"""Common micro f1 resampling."""

from analysis.common import micro_f1_files as c
from analysis.common import micro_f1 as m
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = c.ROOT
BASE = ROOT / "experiments/astra/micro_f1_sq2_retrieval_20261113_v1"
SETUP = c.BASE
METHODS = ("macro", "micro", "micro_macro_labels")
CHECKS = defaultdict(int)
ERRORS = defaultdict(float)


def equal(a, b, name, tolerance=1e-12):
    if a is None or b is None:
        c.require(a is None and b is None, "Unavailable mismatch: " + name)
    else:
        a, b = (np.asarray(a, float), np.asarray(b, float))
        c.require(
            a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            "Shape/nonfinite: " + name,
        )
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        ERRORS[name] = max(ERRORS[name], error)
        c.require(error < tolerance, name + ": " + str(error))
    CHECKS[name] += 1


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Output outside new evidence folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            value, f, indent=2, ensure_ascii=False, allow_nan=False, default=default
        )
        f.write("\n")


def save_arrays(path, **arrays):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Array outside new evidence folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)


def slug(value):
    return value.replace("/", "__").replace("|", "_")


def read_arrays(path):
    with np.load(c.bind(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def freeze():
    c.require(not BASE.exists(), "Preserve existing Part 2 folder")
    BASE.mkdir(parents=True)
    reg = c.read(SETUP / "comparison_registry_v2.json")
    pairs = c.read(SETUP / "paired_contrast_manifest.json")
    complete = c.read(SETUP / "complete.json")
    c.require(complete["status"] == "Part 1 complete", "Setup not complete")
    for path in (
        SETUP / "comparison_registry_v2.json",
        SETUP / "paired_contrast_manifest.json",
        SETUP / "scoring_contract.json",
    ):
        c.require(
            c.sha(path) == complete["artifact_sha256"][c.rel(path)],
            "Frozen setup changed",
        )
    save(
        BASE / "preservation_before.json",
        dict(created_utc=datetime.now(timezone.utc).isoformat(), sources=c.BINDINGS),
    )
    save(
        BASE / "analysis_plan.json",
        dict(
            status="Part 2 scope fixed before new micro results",
            part=2,
            studies=[s for s in reg["studies"] if 2 in s["parts"]],
            adjusted_families={
                key: reg["new_micro_families"][key] for key in ("broad", "core")
            },
            main_pairs=[x for x in pairs["rows"] if x["family"] in ("broad", "core")],
            scope="All registered broad and original core multilabel contrasts, earlier neighbour-label voting, response voting and pooled-target controls. Core 103-type sensitivity remains supporting. Include own-valid and common-nine descriptive tables and native full-data references on those same texts.",
            combined_display="Only Qwen, original historical references and the three native full-data selected procedures here. Equal-budget GPU and other training/cutoff analyses remain Part 3; no prematurely scored GPU variants.",
            inference="Original 10000 paired bootstrap algorithm and seed; full-label and macro-label micro. New broad family m=10, core m=9; remaining original contrasts supporting95. Compact/no contrast descriptive only, as frozen. No interval claim for unsupported German target macro.",
            required_diagnostics="Original macro reproduction; nominal95 comparisons; label-scope decomposition; per-label counts and micro weights; exact sets and gate; full-roster missing-output bounds; natural source-stratified and HateCheck item-resampling sensitivities for the ten main broad questions. These are specified before scoring, not chosen by favourable direction.",
            future_robustness="Detailed selected-validity and cross-study robustness remain Part 8. Any extra diagnostic needed to explain a discrepancy must be recorded before calculation.",
            original_primary_metrics_unchanged=True,
            new_calls=0,
            new_training=0,
            new_retrieval=0,
            latex_edits=0,
        ),
    )
    print(
        "Part 2 frozen: broad/core comparisons and the registered supporting controls.",
        flush=True,
    )


def encode(gold, predictions, valid):
    return m.contributions(gold, predictions, valid)


def values(counts, indices):
    shape = np.asarray(counts).shape[:-2]
    macro = m.macro(counts, indices)
    restricted = m.micro(counts, indices)
    return np.stack(
        [
            np.full(shape, np.nan) if macro is None else macro,
            m.micro(counts),
            np.full(shape, np.nan) if restricted is None else restricted,
        ],
        axis=-1,
    )


def scalar_value(x):
    return float(x) if np.isfinite(x) else None


def setting_summary(gold, pred, valid, labels, macro_labels, gate=None):
    count = encode(gold, pred[None], valid[None])[0].sum(0)
    n = int(valid.sum())
    idx = [labels.index(x) for x in macro_labels]
    score = values(count, idx)
    tp, fp, fn = count.T
    den = 2 * tp + fp + fn
    precision = int(tp.sum()) / int((tp + fp).sum()) if (tp + fp).sum() else 0.0
    recall = int(tp.sum()) / int((tp + fn).sum()) if (tp + fn).sum() else 0.0
    full_weights = den / den.sum() if den.sum() else np.zeros(len(labels))
    restricted_weights = np.zeros(len(labels))
    rd = den[idx].sum()
    if rd:
        restricted_weights[idx] = den[idx] / rd
    macro_weights = np.zeros(len(labels))
    macro_weights[idx] = 1 / len(idx) if idx else 0
    label_f = m.label_f1(count)
    exact = np.all(gold == pred, axis=-1)
    per = []
    for i, label in enumerate(labels):
        per.append(
            dict(
                label=label,
                tp=int(tp[i]),
                fp=int(fp[i]),
                fn=int(fn[i]),
                gold_positive=int(gold[valid, i].sum()),
                predicted_positive=int(pred[valid, i].sum()),
                f1=float(label_f[i]),
                micro_weight=float(full_weights[i]),
                macro_weight=float(macro_weights[i]),
                restricted_micro_weight=float(restricted_weights[i]),
                full_micro_contribution=float(full_weights[i] * label_f[i]),
                restricted_minus_macro_contribution=(
                    float((restricted_weights[i] - macro_weights[i]) * label_f[i])
                    if idx
                    else None
                ),
            )
        )
    oldlow = count.copy()
    oldhigh = count.copy()
    oldhigh[:, 0] += gold[~valid].sum(0)
    oldlow[:, 1] += (~gold[~valid].astype(bool)).sum(0)
    oldlow[:, 2] += gold[~valid].sum(0)
    low, high = (values(oldlow, idx), values(oldhigh, idx))
    if idx:
        equal(
            sum((x["restricted_minus_macro_contribution"] for x in per)),
            score[2] - score[0],
            "label weight decomposition",
        )
    return dict(
        n_eligible=len(valid),
        n_valid=n,
        n_invalid=len(valid) - n,
        scores={
            name: scalar_value(score[j]) if n else None
            for j, name in enumerate(METHODS)
        },
        tp_fp_fn=count,
        micro_precision=precision if n else None,
        micro_recall=recall if n else None,
        set_only_exact=float(exact[valid].mean()) if n else None,
        complete_field_exact=(
            float((exact & gate.astype(bool))[valid].mean())
            if gate is not None and n
            else None
        ),
        predicted_gate_miss=(
            float((~gate.astype(bool))[valid].mean())
            if gate is not None and n
            else None
        ),
        mean_gold_labels=float(gold[valid].sum(1).mean()) if n else None,
        mean_predicted_labels=float(pred[valid].sum(1).mean()) if n else None,
        per_label=per,
        full_roster_missing_bounds={
            name: [scalar_value(low[j]), scalar_value(high[j])]
            for j, name in enumerate(METHODS)
        },
        zero_information_denominator=int(den.sum()) == 0,
    )


def average(summaries):
    names = (
        "micro_precision",
        "micro_recall",
        "set_only_exact",
        "complete_field_exact",
        "predicted_gate_miss",
        "mean_gold_labels",
        "mean_predicted_labels",
    )
    out = {
        key: (
            float(np.mean([x[key] for x in summaries]))
            if all((x[key] is not None for x in summaries))
            else None
        )
        for key in names
    }
    out["scores"] = {
        key: (
            float(np.mean([x["scores"][key] for x in summaries]))
            if all((x["scores"][key] is not None for x in summaries))
            else None
        )
        for key in METHODS
    }
    out["repeat_scores"] = {
        key: [x["scores"][key] for x in summaries] for key in METHODS
    }
    out["n_valid_by_setting"] = [x["n_valid"] for x in summaries]
    return out


def bootstrap(
    gold,
    pred,
    valid,
    indices,
    clusters,
    nleft,
    seed,
    *,
    draws=10000,
    item_order=False,
    strata_sources=None,
    delta_mode="pairs"
):
    mask = valid.all(0)
    positions = np.flatnonzero(mask)
    cube = encode(gold, pred, valid)[:, mask]
    c.require(len(positions) > 0, "Empty paired population")
    if item_order:
        kept = list(range(len(positions)))
        names = kept
    else:
        kept = [clusters[i] for i in positions]
        names = sorted(set(kept))
    lookup = {k: i for i, k in enumerate(names)}
    grouped = np.zeros((len(pred), len(names), len(gold[0]), 3), np.int64)
    for i, k in enumerate(kept):
        grouped[:, lookup[k]] += cube[:, i]
    strata = defaultdict(list)
    memberships = defaultdict(set)
    if strata_sources is not None:
        for cluster, source in zip(clusters, strata_sources):
            memberships[cluster].add(source)
    for j, k in enumerate(names):
        strata[
            tuple(sorted(memberships[k])) if strata_sources is not None else ("all",)
        ].append(j)
    point = values(cube.sum(1), indices)

    def difference(v):
        if delta_mode == "pairs":
            return (v[..., :nleft, None, :] - v[..., None, nleft:, :]).mean(
                axis=(-3, -2)
            )
        return v[..., :nleft, :].mean(-2) - v[..., nleft:, :].mean(-2)

    rng = np.random.default_rng(seed)
    saved = []
    checks = 0
    max_error = 0.0
    flat = grouped.transpose(1, 0, 2, 3).reshape(len(names), -1).astype(float)
    alternative = np.stack(
        [
            grouped[..., 0],
            grouped[..., 0] + grouped[..., 1],
            grouped[..., 0] + grouped[..., 2],
        ],
        axis=-1,
    )
    altflat = alternative.transpose(1, 0, 2, 3).reshape(len(names), -1).astype(float)
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        picks = np.concatenate(
            [
                rng.choice(group, (size, len(group)), replace=True)
                for _, group in sorted(strata.items())
            ],
            axis=1,
        )
        weights = np.stack([np.bincount(row, minlength=len(names)) for row in picks])
        total = (
            (weights @ flat).reshape(size, len(pred), len(gold[0]), 3).astype(np.int64)
        )
        c.require(
            np.array_equal(total[0], grouped[:, picks[0]].sum(1)),
            "Direct expansion mismatch",
        )
        checks += 1
        score = values(total, indices)
        alt = (weights @ altflat).reshape(size, len(pred), len(gold[0]), 3)
        tp, pp, aa = (alt[..., j] for j in range(3))
        den = pp + aa
        lf = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den != 0)
        indmacro = (
            lf[..., indices].mean(-1) if indices else np.full(tp.shape[:-1], np.nan)
        )
        summed = den.sum(-1)
        indmicro = np.divide(
            2 * tp.sum(-1), summed, out=np.zeros_like(summed), where=summed != 0
        )
        rd = den[..., indices].sum(-1)
        indrest = (
            np.divide(
                2 * tp[..., indices].sum(-1), rd, out=np.zeros_like(rd), where=rd != 0
            )
            if indices
            else np.full_like(indmicro, np.nan)
        )
        independent = np.stack([indmacro, indmicro, indrest], axis=-1)
        c.require(
            np.array_equal(np.isnan(score), np.isnan(independent)),
            "Independent availability differs",
        )
        error = float(np.nanmax(np.abs(score - independent)))
        max_error = max(max_error, error)
        c.require(error < 1e-14, "Independent bootstrap metric mismatch")
        saved.append(
            np.stack(
                [score[:, :nleft].mean(1), score[:, nleft:].mean(1), difference(score)],
                axis=1,
            )
        )
    return dict(
        mask=mask,
        point=point,
        delta=difference(point),
        draws=np.concatenate(saved),
        n_clusters=len(names),
        cluster_strata={"|".join(k): len(v) for k, v in strata.items()},
        seed=seed,
        checks=dict(
            direct_expansions=checks,
            independent_setting_metric_values=draws * len(pred) * (3 if indices else 1),
            max_error=max_error,
        ),
    )


def interval(draws, confidence=0.95):
    if np.isnan(draws).all():
        return None
    c.require(np.isfinite(draws).all(), "Undefined bootstrap values")
    return np.quantile(draws, [(1 - confidence) / 2, 1 - (1 - confidence) / 2]).tolist()


def contrast_summary(boot, left, right, family_size=None):
    nleft = len(left)
    d = boot["draws"]
    out = {}
    for j, name in enumerate(METHODS):
        out[name] = dict(
            left=scalar_value(boot["point"][:nleft, j].mean()),
            right=scalar_value(boot["point"][nleft:, j].mean()),
            difference=scalar_value(boot["delta"][j]),
            ci95=interval(d[:, 2, j]),
            adjusted_interval=(
                interval(d[:, 2, j], 1 - 0.05 / family_size)
                if family_size and name != "macro"
                else None
            ),
            adjusted_confidence=(
                1 - 0.05 / family_size if family_size and name != "macro" else None
            ),
        )
    return dict(
        left_settings=left,
        right_settings=right,
        n_common=int(boot["mask"].sum()),
        n_clusters=boot["n_clusters"],
        metrics=out,
        seed=boot["seed"],
        bootstrap_draws=len(d),
        family_size=family_size,
        cluster_strata=boot["cluster_strata"],
        verification=boot["checks"],
    )


if __name__ == "__main__":
    freeze()
