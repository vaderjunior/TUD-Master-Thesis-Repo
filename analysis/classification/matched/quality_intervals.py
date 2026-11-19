from analysis.classification.matched import common as m
from analysis.classification.matched import validate as independent
from pathlib import Path
import numpy as np


def item_value(row, pred, metric):
    if metric == "known_exact":
        known = {
            d: v for d in m.DIMENSIONS if (v := independent.truth(row, d)) is not None
        }
        return float(
            all(
                pred[d] == (sorted(v) if isinstance(v, list) else v)
                for d, v in known.items()
            )
        )
    if metric in ["fpr", "fnr"]:
        if row["gate"] is not (metric == "fnr"):
            return None
        return float(pred["hate"] != row["gate"])
    dim, kind = metric.rsplit(":", 1)
    gold = independent.truth(row, dim)
    if gold is None:
        return None
    if kind == "exact":
        return float(pred[dim] == (sorted(gold) if isinstance(gold, list) else gold))
    y = set(gold)
    x = set(pred[dim])
    return float(len(x - y) if kind == "extra" else len(y - x))


def run():
    with m.runtime.offline():
        groups, outputs, meta = m.load()
        result = {}
        bindings = {}
        verified = 0
        for phase, rr in groups.items():
            dim = m.scoring.MATCHED_DIMS[phase]
            metrics = ["known_exact"] + (
                ["fpr", "fnr", "target_group:exact", "severity:exact"]
                if dim == "hate"
                else (
                    ["fnr", "hate_type:extra", "hate_type:missing"]
                    if dim == "hate_type"
                    else ["legal:extra", "legal:missing"]
                )
            )
            for family in m.PAIRS[:2]:
                lhs = m.names(outputs, "retrieved_complete")
                rhs = m.names(outputs, family)
                for metric in metrics:
                    kept = [
                        r
                        for r in rr
                        if all(outputs[k][r["id"]] is not None for k in lhs + rhs)
                        and item_value(r, outputs[lhs[0]][r["id"]], metric) is not None
                    ]
                    m.a.require(
                        len({r["text_cluster"] for r in kept}) == len(kept),
                        "Quality interval requires distinct text clusters",
                    )
                    left = np.array(
                        [
                            [item_value(r, outputs[k][r["id"]], metric) for r in kept]
                            for k in lhs
                        ]
                    )
                    right = np.array(
                        [
                            [item_value(r, outputs[k][r["id"]], metric) for r in kept]
                            for k in rhs
                        ]
                    )
                    delta = left.mean(axis=0) - right.mean(axis=0)
                    draws = []
                    rng = np.random.default_rng(20261030)
                    n = len(kept)
                    error = 0.0
                    for begin in range(0, 10000, 64):
                        picks = rng.integers(0, n, size=(min(64, 10000 - begin), n))
                        direct = delta[picks].mean(axis=1)
                        weights = np.array([np.bincount(p, minlength=n) for p in picks])
                        check = weights @ delta / n
                        error = max(error, float(np.abs(direct - check).max()))
                        draws.extend(direct.tolist())
                        verified += len(picks)
                    m.a.require(
                        error < 1e-12, "Independent quality resample arithmetic differs"
                    )
                    path = (
                        m.BASE
                        / "quality_bootstrap"
                        / (m.slug(phase, family, metric.replace(":", "_")) + ".npy")
                    )
                    m.array(path, draws)
                    bindings[m.runtime.rel(path)] = m.c.sha(path)
                    result[phase + "/" + family + "/" + metric] = dict(
                        n=n,
                        ids=[r["id"] for r in kept],
                        left_mean=float(left.mean()),
                        right_mean=float(right.mean()),
                        difference=float(delta.mean()),
                        ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                        bootstrap_file=m.runtime.rel(path),
                        maximum_independent_sample_difference=error,
                        draws=10000,
                        seed=20261030,
                    )
        m.save(
            m.BASE / "quality_intervals.json",
            dict(
                scope="Additional descriptive 95% intervals for both primary-control families on all three populations; no multiplicity-adjusted claim. Prepared after primary F1 results were inspected. Conditional paired text resampling within each eligible gold subset; fixed decoder/selection variants.",
                results=result,
                independently_checked_bootstrap_values=verified,
                outputs=bindings,
                sources=m.bindings([Path(__file__), Path(independent.__file__)]),
                Qwen_calls=0,
                training_fits=0,
            ),
        )
        print(
            "Descriptive quality intervals complete:",
            verified,
            "resample values checked by two reduction paths.",
            flush=True,
        )


if __name__ == "__main__":
    run()
