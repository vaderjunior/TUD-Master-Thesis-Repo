"""German guidelines quantile precision."""

from pathlib import Path
import json
import math
import numpy as np
from scipy.stats import binom
from analysis.knowledge.german_guidelines import robustness as r
from analysis.knowledge.german_guidelines.verify_hate import bootstrap_metrics

c = r.c


def binomial_cdf(k, n, p):
    return math.fsum(
        (
            math.exp(
                math.lgamma(n + 1)
                - math.lgamma(j + 1)
                - math.lgamma(n - j + 1)
                + j * math.log(p)
                + (n - j) * math.log1p(-p)
            )
            for j in range(k + 1)
        )
    )


def quantile_band(values, q):
    z = np.sort(np.asarray(values, dtype=float))
    n = len(z)
    lo = int(binom.ppf(0.025, n, q))
    hi = int(binom.ppf(0.975, n, q)) + 1
    r.require(1 <= lo <= hi <= n, "Quantile rank band touches an unsupported infinite endpoint")
    left = binomial_cdf(lo - 1, n, q)
    right = 1 - binomial_cdf(hi - 1, n, q)
    r.require(
        left <= 0.025 + 1e-09 and right <= 0.025 + 1e-09, "Order-statistic tail coverage failed"
    )
    return {
        "quantile": q,
        "saved_percentile": float(np.quantile(z, q)),
        "order_statistic_ranks_one_based": [lo, hi],
        "monte_carlo_95_rank_band": [float(z[lo - 1]), float(z[hi - 1])],
        "independently_checked_binomial_tail_probabilities": [left, right],
        "expected_draws_beyond_tail": n * min(q, 1 - q),
    }


def main():
    with r.offline():
        _, out = r.verified()
        _, hp = r.h.verified()
        report = r.BASE / "report"
        paths = [out / "exclusions" / name / "bootstrap.npz" for name in r.SEEDS] + [
            hp / "de_test/arrays.npz",
            Path(__file__),
        ]
        r.save(
            report / "precision_scope.json",
            {
                "status": "additional_numerical_precision_check_before_calculation",
                "created_utc": c.a.utc(),
                "reason": "The combined eight-key sensitivity has a 98.75% upper endpoint only 0.00003956 above zero. Separate datasets use frozen different draw seeds, so cutoff movement need not be solely caused by exclusions.",
                "method": "For each existing general-only-minus-shared hate TEST draw vector, show 95% binomial order-statistic Monte Carlo bands for the .00625 and .99375 percentiles. Independently sum the binomial probabilities to verify rank coverage.",
                "scope": "Numerical precision of saved percentile endpoints conditional on the bootstrap distribution; not another confidence interval for the true treatment effect. Discreteness can make bands conservative.",
                "limits": "No new draws, seed changes, replacement intervals, main test changes, significance selection or recalculation of scores.",
                "sources": c.bindings(paths),
                "new_bootstrap_draws": 0,
                "actual_Qwen_calls": 0,
            },
        )
        results = {}
        for name in ["original", *r.SEEDS]:
            if name == "original":
                with np.load(hp / "de_test/arrays.npz", allow_pickle=False) as f:
                    z = f["bootstrap_confusions"]
            else:
                with np.load(out / "exclusions" / name / "bootstrap.npz", allow_pickle=False) as f:
                    z = f["counts"]
            v = bootstrap_metrics(z)["macro_f1"]
            delta = v[:, 1] - v[:, 0]
            results[name] = {
                "draws": len(delta),
                "lower": quantile_band(delta, 0.00625),
                "upper": quantile_band(delta, 0.99375),
            }
        r.save(
            report / "precision_check.json",
            {
                "status": "saved_draw_precision_checked",
                "results": results,
                "source_scope": c.ref(report / "precision_scope.json"),
                "bindings": c.bindings(paths + [report / "precision_scope.json"]),
                "new_bootstrap_draws": 0,
                "actual_Qwen_calls": 0,
            },
        )
        print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
