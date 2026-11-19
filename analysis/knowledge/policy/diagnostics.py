from pathlib import Path
import numpy as np
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import reader as r


def main():
    base = r.BASE
    root = c.ROOT / c.read(base / "analysis_complete.json")["run"]
    c.require(
        c.read(base / "final_independent_checks.json")["status"] == "passed",
        "Independent review is incomplete",
    )
    result = {}
    for endpoint in ("de_legal", "en_severity"):
        v = c.read(root / endpoint / "results.json")
        point = v["point"]
        with np.load(root / endpoint / "encoded.npz", allow_pickle=False) as z:
            mask = z["common"]
            y = z["gold"][mask]
            p = z["predicted"][:, :, mask]
            correct = z["correct"][:, :, mask]
        if endpoint == "de_legal":
            cohorts = {}
            for name, subset in [
                ("empty_gold", ~y.any(axis=1)),
                ("nonempty_gold", y.any(axis=1)),
            ]:
                exact = correct[:, :, subset].mean(axis=2)
                extra = (p[:, :, subset] & ~y[subset]).sum(axis=(2, 3))
                missing = (~p[:, :, subset] & y[subset]).sum(axis=(2, 3))
                cohorts[name] = dict(
                    n=int(subset.sum()),
                    exact_repeats=exact.tolist(),
                    exact=exact.mean(axis=1).tolist(),
                    extra_label_counts_repeats=extra.tolist(),
                    missing_label_counts_repeats=missing.tolist(),
                    net_exact_matches_C_minus_A_mean=float(
                        (exact[2] - exact[0]).mean() * subset.sum()
                    ),
                )
            counts = np.array(point["counts"]).sum(axis=2)
            precision = counts[:, :, 0] / (counts[:, :, 0] + counts[:, :, 1])
            recall = counts[:, :, 0] / (counts[:, :, 0] + counts[:, :, 2])
            result[endpoint] = dict(
                cohorts=cohorts,
                label_precision=precision.mean(axis=1).tolist(),
                label_recall=recall.mean(axis=1).tolist(),
            )
            c.require(
                np.allclose(cohorts["empty_gold"]["exact"], point["empty_gold_exact"]),
                "Empty-gold exactness differs",
            )
            c.require(
                abs(
                    sum(
                        (
                            x["net_exact_matches_C_minus_A_mean"]
                            for x in cohorts.values()
                        )
                    )
                    - point["transitions"][0]["net_correct_mean"]
                )
                < 1e-10,
                "Legal cohort gains do not reconcile",
            )
        else:
            s = point["severity"]
            error = s["signed_error_counts_repeats"]
            result[endpoint] = dict(
                mean_signed_error_counts={
                    k: np.mean(value, axis=1).tolist() for k, value in error.items()
                },
                mean_gate_closed_counts=np.mean(
                    point["gate_closed_counts"], axis=1
                ).tolist(),
                common_emitted_exact=correct[:, :, p.any(axis=-1).all(axis=(0, 1))]
                .mean(axis=(1, 2))
                .tolist(),
            )
    rr = c.read(base / "reader_complete.json")
    nulls = [
        entry
        for name in rr["reader_files"]
        for entry in c.read(c.ROOT / name)
        if entry["result"] is None
    ]
    result["missing"] = dict(
        predictions=len(nulls),
        unique_keys=len({x["key"] for x in nulls}),
        conditions=sorted({x["condition"] for x in nulls}),
    )
    contrasts = c.read(root / "contrasts.json")
    for entry in contrasts:
        lo, hi = entry["interval95"]
        c.require(-1 <= lo <= hi <= 1, "Invalid descriptive interval")
        if entry["family_interval"]:
            low, high = entry["family_interval"]
            c.require(
                -1 <= low <= lo <= hi <= high <= 1,
                "Family interval must contain ordinary interval",
            )
    result["interval_checks"] = dict(
        registered_rows=len(contrasts),
        status="all_bounds_finite_ordered_and_family_intervals_contain_95_percent",
    )
    r.save(base / "interpretation_checks.json", result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "en_hate": "English hate",
        "en_hate_type": "English hate types",
        "de_hate": "German hate",
        "de_legal": "German legal labels",
        "en_target_group": "English targets",
        "en_severity": "English severity",
    }
    groups = [
        ("main_macro_4", "Macro F1: four main comparisons", "98.75% intervals"),
        (
            "complete_policy_agreement_6",
            "Agreement between repeated answers",
            "99.17% intervals",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, (family, title, subtitle) in zip(axes, groups):
        entries = [v for v in contrasts if v["family"] == family]
        for i, entry in enumerate(entries):
            est = 100 * entry["estimate"]
            lo, hi = [100 * x for x in entry["family_interval"]]
            color = "#176d58" if lo > 0 else "#b04739" if hi < 0 else "#596779"
            ax.plot([lo, hi], [i, i], color=color, lw=2.2)
            ax.plot([lo, hi], [i, i], "|", color=color, markersize=9)
            ax.plot(est, i, "o", color=color, ms=6)
        ax.axvline(0, color="#9ca3af", lw=1, ls="--")
        ax.set_yticks(range(len(entries)), [titles[e["endpoint"]] for e in entries])
        ax.invert_yaxis()
        ax.set_xlabel("Full policy minus retrieved policy (percentage points)")
        ax.set_title(title + "\n" + subtitle, fontsize=12, pad=12)
        ax.grid(axis="x", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "More policy helps legal scores, but does not improve repeat agreement overall",
        fontsize=14,
    )
    figures = base / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        target = figures / f"primary_effects.{suffix}"
        c.require(not target.exists(), "Preserve existing figure")
        fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)
    r.save(
        base / "figure_and_diagnostics_complete.json",
        dict(
            status="complete",
            sources=c.bindings(
                [
                    Path(__file__),
                    root / "contrasts.json",
                    base / "interpretation_checks.json",
                ]
            ),
            figures=c.bindings(sorted(figures.iterdir())),
            caption="Common-nine texts; same five examples; 50000 paired source-stratified text-cluster bootstrap draws. Within-family intervals only, conditional on selected data and three repeats. Right panel measures label-state agreement, not correctness.",
        ),
    )
    print(result)


if __name__ == "__main__":
    with c.offline():
        main()
