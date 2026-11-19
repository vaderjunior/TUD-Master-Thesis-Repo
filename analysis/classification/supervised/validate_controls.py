import hashlib
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from threadpoolctl import threadpool_limits
from analysis.classification.supervised import controls as c
from analysis.classification.supervised import archive as r


def load_bootstrap(path, models):
    "Read only our sealed analysis matrix, with an exact expected shape."
    r.require(1 <= models <= 846, "Unexpected bootstrap model count.")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        r.require(
            len(entries) == 1
            and entries[0].filename == "scores.npy"
            and entries[0].file_size <= 10000 * models * 8 + 1024,
            "Unexpected bootstrap array members or size.",
        )
    with np.load(path, allow_pickle=False) as arrays:
        value = arrays["scores"]
    r.require(
        value.shape == (10000, models)
        and value.dtype == np.float64
        and np.isfinite(value).all()
        and ((value >= 0) & (value <= 1)).all(),
        "Invalid bootstrap matrix.",
    )
    return value


def means(meta, values):
    result = {}
    keys = sorted({(x["method"], x["variant"]) for x in meta["models"]})
    for budget in (147, 1000, 5000):
        for method, variant in keys:
            pp = []
            for pool in r.POOLS:
                if (147 if "compact" in pool else int(pool.split("_")[1])) != budget:
                    continue
                ix = [
                    j
                    for j, x in enumerate(meta["models"])
                    if x["pool"] == pool
                    and x["method"] == method
                    and x["variant"] == variant
                ]
                expected = 5 if method in r.FAMILIES else 1
                r.require(len(ix) == expected, "Independent pool roster differs.")
                pp.append(np.sum(values[..., ix], axis=-1) / expected)
            r.require(len(pp) == 3, "Independent budget roster differs.")
            result[f"{budget}/{method}/{variant}"] = (pp[0] + pp[1] + pp[2]) / 3
    return result


def quantile(values, p):
    ordered = np.sort(values)
    q = (len(ordered) - 1) * p
    j = int(np.floor(q))
    k = min(j + 1, len(ordered) - 1)
    return float(ordered[j] + (q - j) * (ordered[k] - ordered[j]))


def verify(folder):
    r.offline()
    started = time.monotonic()
    complete = r.read(folder / "complete.json")
    for name, digest in complete["outputs"].items():
        r.checked_file(folder / name, digest)

    for prior in (c.MAIN, c.TRAIN):
        receipt = r.read(prior / "complete.json")
        wanted = (
            (
                lambda n: n.endswith(
                    (
                        "selection_review.json",
                        "candidate_scores.json",
                        "canonical_jobs.json",
                    )
                )
            )
            if prior == c.TRAIN
            else (
                lambda n: any(
                    n.endswith(s)
                    for s in (
                        "_metadata.json",
                        "_predictions.npz",
                        "_point_counts.npz",
                        "_bootstrap.npz",
                    )
                )
            )
        )
        for name, digest in receipt["outputs"].items():
            if wanted(name):
                r.checked_file(prior / name, digest)
    summary = {}
    with threadpool_limits(limits=2):
        for endpoint in r.PRIMARY:
            slug = endpoint.replace("/", "__")
            meta = r.read(folder / f"{slug}_metadata.json")
            z = r.arrays_from_bytes((folder / f"{slug}_arrays.npz").read_bytes())
            q = r.read(folder / f"{slug}_quality.json")
            saved = load_bootstrap(
                folder / f"{slug}_bootstrap.npz", len(meta["models"])
            )
            results = r.read(folder / f"{slug}_results.json")
            y = z["truth"].astype(bool)
            pred = z["predictions"].astype(bool)
            points = []
            quality_checks = 0
            decision_count = 0
            for j, model in enumerate(meta["models"]):
                raw = meta["raw_models"][model["raw_index"]]
                pp = z["probabilities"][model["raw_index"]]
                gp = z["gate_probabilities"][model["raw_index"]]
                v = model["variant"]
                head = meta["head"]
                use_selected_gate = v in ("selected", "gate_only")
                gate = gp > (raw["hate_cutoff"] if use_selected_gate else 0.5)
                if head == "hate":
                    expected = np.stack([~gate, gate], axis=1)
                else:
                    if head == "severity":
                        expected = (
                            np.arange(pp.shape[1])[None, :]
                            == np.argmax(pp, axis=1)[:, None]
                        )
                    else:
                        cutoff = (
                            raw["head_cutoff"]
                            if v in ("selected", "head_only", "ungated_selected")
                            else 0.5
                        )
                        expected = pp >= cutoff
                    if v.startswith("ungated"):
                        gate = np.ones(len(y), dtype=bool)
                    expected = expected & gate[:, None]
                r.require(
                    np.array_equal(pred[j], expected)
                    and np.array_equal(gate, z["gates"][j]),
                    "Independent cutoff decoding differs.",
                )
                decision_count += expected.size
                f = float(f1_score(y, pred[j], average="macro", zero_division=0))
                points.append(f)
                reference = dict(
                    macro_f1=f,
                    micro_f1=f1_score(y, pred[j], average="micro", zero_division=0),
                    micro_precision=precision_score(
                        y, pred[j], average="micro", zero_division=0
                    ),
                    micro_recall=recall_score(
                        y, pred[j], average="micro", zero_division=0
                    ),
                    exact_set=accuracy_score(y, pred[j]),
                    hamming_error=np.count_nonzero(y != pred[j]) / y.size,
                    predicted_labels=np.count_nonzero(pred[j]) / len(y),
                    empty_rate=np.mean(~np.any(pred[j], axis=1)),
                    predicted_hate=np.mean(gate),
                )
                for name, value in reference.items():
                    r.require(
                        abs(float(q[name][j]) - float(value)) < 1e-14,
                        "Independent quality differs: " + name,
                    )
                    quality_checks += 1
                for name, mask in [
                    ("tp", y & pred[j]),
                    ("fp", ~y & pred[j]),
                    ("fn", y & ~pred[j]),
                ]:
                    r.require(
                        np.array_equal(mask.sum(0), q["counts"][name][j]),
                        "Independent confusion counts differ.",
                    )
            points = np.asarray(points)
            r.require(np.isfinite(points).all(), "Invalid independent point scores.")

            unique, inverse = np.unique(
                np.asarray(meta["clusters"]), return_inverse=True
            )
            models, _, labels = pred.shape
            terms = np.zeros((len(unique), 3 * models * labels))
            for j in range(models):
                for k, mask in enumerate((y & pred[j], ~y & pred[j], y & ~pred[j])):
                    for label in range(labels):
                        terms[:, k * models * labels + j * labels + label] = (
                            np.bincount(
                                inverse, weights=mask[:, label], minlength=len(unique)
                            )
                        )
            rng = np.random.default_rng(20261020)
            hasher = hashlib.sha256()
            maxerr = 0.0
            weighted_checks = 0
            for first in range(0, 10000, 32):
                n = min(32, 10000 - first)
                picks = rng.choice(len(unique), size=(n, len(unique)), replace=True)
                hasher.update(picks.astype("<i8", copy=False).tobytes())
                weights = np.array(
                    [np.bincount(x, minlength=len(unique)) for x in picks], dtype=float
                )
                totals = (weights @ terms).reshape(n, 3, models, labels)
                den = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
                f = np.divide(
                    2 * totals[:, 0], den, out=np.zeros_like(den), where=den > 0
                ).mean(-1)
                error = float(np.max(np.abs(f - saved[first : first + n])))
                maxerr = max(maxerr, error)
                r.require(error < 1e-14, "Independent bootstrap differs.")
                for target in (0, 17, 999, 9999):
                    if first <= target < first + n:
                        row_weights = weights[target - first, inverse]
                        for j in range(models):
                            check = f1_score(
                                y,
                                pred[j],
                                sample_weight=row_weights,
                                average="macro",
                                zero_division=0,
                            )
                            r.require(
                                abs(check - f[target - first, j]) < 1e-14,
                                "Weighted sklearn bootstrap check differs.",
                            )
                            weighted_checks += 1
                if first + n in (2048, 4096, 6144, 8192, 10000):
                    print(
                        endpoint + f": independent resamples {first+n}/10000",
                        flush=True,
                    )
            r.require(
                hasher.hexdigest() == results["bootstrap"]["picks_sha256"],
                "Independent paired draws differ.",
            )
            budget_points = means(meta, points)
            budget_draws = means(meta, saved)
            numerical_rows = 0
            for key, row in results["absolute"].items():
                r.require(
                    abs(row["value"] - budget_points[key]) < 1e-14
                    and abs(row["low"] - quantile(budget_draws[key], 0.025)) < 1e-14
                    and abs(row["high"] - quantile(budget_draws[key], 0.975)) < 1e-14,
                    "Absolute interval differs.",
                )
                numerical_rows += 1
            for row in results["comparisons"]:
                a, b = (f"{row['budget']}/{row[k]}" for k in ("after", "before"))
                expected = budget_points[a] - budget_points[b]
                dd = budget_draws[a] - budget_draws[b]
                r.require(
                    abs(expected - row["value"]) < 1e-14
                    and abs(quantile(dd, 0.025) - row["low"]) < 1e-14
                    and abs(quantile(dd, 0.975) - row["high"]) < 1e-14,
                    "Comparison differs.",
                )
                numerical_rows += 1
            for row in results["cutoff_components"]:
                keys = [
                    f"{row['budget']}/{row['family']}/{v}"
                    for v in ("default", "gate_only", "head_only", "selected")
                ]
                aa, gg, hh, zz = (budget_points[k] for k in keys)
                av, gv, hv, zv = (budget_draws[k] for k in keys)
                reference = {
                    "total": (zz - aa, zv - av),
                    "head": ((hh - aa + zz - gg) / 2, (hv - av + zv - gv) / 2),
                    "gate": ((gg - aa + zz - hh) / 2, (gv - av + zv - hv) / 2),
                    "interaction": (zz - hh - gg + aa, zv - hv - gv + av),
                    "head_first": (hh - aa, hv - av),
                    "head_second": (zz - gg, zv - gv),
                    "gate_first": (gg - aa, gv - av),
                    "gate_second": (zz - hh, zv - hv),
                }
                for name, (value, dd) in reference.items():
                    r.require(
                        abs(value - row[name]["value"]) < 1e-14
                        and abs(quantile(dd, 0.025) - row[name]["low"]) < 1e-14
                        and abs(quantile(dd, 0.975) - row[name]["high"]) < 1e-14,
                        "Factorial decomposition differs.",
                    )
                    numerical_rows += 1
            for metric in c.SCALARS:
                for key, value in means(meta, np.asarray(q[metric])).items():
                    r.require(
                        abs(value - results["quality_budget"][metric][key]) < 1e-14,
                        "Quality budget aggregation differs.",
                    )
            summary[endpoint] = dict(
                model_views=models,
                decoded_decisions=decision_count,
                quality_checks=quality_checks,
                bootstrap_values=saved.size,
                bootstrap_maximum_absolute_error=maxerr,
                weighted_sklearn_checks=weighted_checks,
                interval_rows=numerical_rows,
                all_previous_primary_results_preserved=63,
            )
            print(endpoint + ": independent review passed.", flush=True)
    c.save(
        folder / "independent_review.json",
        dict(
            status="pass",
            seconds=time.monotonic() - started,
            endpoints=summary,
            sources_preserved=True,
            code_sha256=r.sha(Path(__file__)),
            run_complete_sha256=r.sha(folder / "complete.json"),
        ),
    )


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
