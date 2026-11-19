from analysis.knowledge.german_guidelines import hate as h
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(h.BASE / "report/matplotlib_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

NAMES = {
    "mixed_2": "Current shared slots",
    "general_2": "General guidance only",
    "general_2_legal_1": "General + separate legal",
}
COLORS = {"mixed_2": "#315B7C", "general_2": "#B86524", "general_2_legal_1": "#237B70"}
PAIR_NAMES = [
    "General only − shared",
    "General + legal − shared",
    "General + legal − general",
]


def publish(fig, folder, stem):
    files = []
    for suffix in ("png", "svg"):
        path = folder / f"{stem}.{suffix}"
        h.require(not path.exists(), "Preserve existing figure: " + str(path))
        fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        files.append(path)
    plt.close(fig)
    return files


def main():
    with h.offline():
        _, run = h.verified()
        d = h.c.read(run / "de_test/results.json")
        folder = h.BASE / "report/figures"
        folder.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 11,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "svg.fonttype": "none",
                "axes.titleweight": "bold",
            }
        )
        files, plotted = ([], {})
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), sharey=True)
        fig.suptitle(
            "German hate TEST: false alarms and missed hateful texts",
            fontsize=15,
            fontweight="bold",
            y=0.99,
        )
        fig.subplots_adjust(left=0.21, right=0.99, bottom=0.23, top=0.81, wspace=0.12)
        rows = np.arange(3)[::-1]
        for ax, field, count, denominator, title in zip(
            axes,
            ("fpr", "fnr"),
            ("fp", "fn"),
            (2306, 861),
            ("False alarms among non-hateful texts", "Misses among hateful texts"),
        ):
            values = []
            for ci, cond in enumerate(h.CONDITIONS):
                mean = d["common"][cond]["mean"][field] * 100
                interval = (
                    np.asarray(
                        d["condition_descriptive_intervals"][cond][field]["ci95"]
                    )
                    * 100
                )
                repeats = (
                    np.asarray([v[field] for v in d["common"][cond]["repeats"]]) * 100
                )
                ax.barh(rows[ci], mean, color=COLORS[cond], height=0.57, alpha=0.92)
                ax.errorbar(
                    mean,
                    rows[ci],
                    xerr=[[mean - interval[0]], [interval[1] - mean]],
                    color="#202C36",
                    capsize=4,
                    linewidth=1.1,
                    fmt="none",
                )
                ax.scatter(
                    repeats,
                    rows[ci] + np.array([-0.1, 0, 0.1]),
                    s=28,
                    facecolor="white",
                    edgecolor="#202C36",
                    linewidth=0.8,
                    zorder=4,
                )
                errors = d["common"][cond]["mean"][count]
                ax.text(
                    25.3,
                    rows[ci],
                    f"{mean:.2f}%\n{errors:.1f} / {denominator:,}",
                    ha="right",
                    va="center",
                    fontsize=10,
                )
                values.append(
                    {
                        "condition": cond,
                        "mean_percent": mean,
                        "ci95_percent": interval.tolist(),
                        "repeat_percent": repeats.tolist(),
                        "mean_count": errors,
                        "denominator": denominator,
                    }
                )
            ax.set_title(title, fontsize=11, pad=12)
            ax.set_xlim(0, 26)
            ax.set_xticks([0, 5, 10, 15, 20, 25])
            ax.set_xlabel("Error rate (%) — lower is better")
            ax.grid(axis="x", alpha=0.17)
            ax.set_axisbelow(True)
            ax.set_yticks(rows, [NAMES[x] for x in h.CONDITIONS])
            plotted[field] = values
        fig.text(
            0.21,
            0.1,
            "Bars: mean of three repeat rates. White dots: individual repeats. Whiskers: descriptive 95% intervals.",
            fontsize=9,
        )
        fig.text(
            0.21,
            0.045,
            "Same 3,167 texts in every setting; 10,000 paired, source-stratified text-cluster bootstrap draws.",
            fontsize=9,
        )
        files += publish(fig, folder, "german_hate_test_error_rates")
        fig, ax = plt.subplots(figsize=(10.3, 4.8))
        fig.subplots_adjust(left=0.31, right=0.98, bottom=0.29, top=0.8)
        fig.suptitle(
            "German hate TEST: changes in macro-F1",
            fontsize=15,
            fontweight="bold",
            y=0.97,
        )
        ax.axvline(0, color="#707980", linestyle="--", linewidth=1)
        points = []
        for i, value in enumerate(d["contrasts"]):
            m = value["metrics"]["macro_f1"]
            y = 2 - i
            estimate = 100 * m["difference"]
            ci = np.asarray(m["ci95"]) * 100
            family = (
                np.asarray(m["family_interval"]) * 100
                if m["family_interval"] is not None
                else None
            )
            color = COLORS[value["left"]]
            if family is not None:
                ax.hlines(y, *family, color="#8997A1", linewidth=2.3)
                ax.plot(family, [y, y], "|", color="#8997A1", markersize=13)
            ax.hlines(y, *ci, color=color, linewidth=6)
            ax.plot(estimate, y, "o", color="#17242E", markersize=7)
            ax.annotate(
                f"{estimate:+.3f}",
                (estimate, y),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
            points.append(
                {
                    "left": value["left"],
                    "right": value["right"],
                    "difference_points": estimate,
                    "ci95_points": ci.tolist(),
                    "family_interval_points": (
                        None if family is None else family.tolist()
                    ),
                }
            )
        ax.set_yticks([2, 1, 0], PAIR_NAMES)
        ax.set_ylim(-0.48, 2.5)
        ax.set_xlim(-2.05, 1.0)
        ax.set_xlabel("Change in macro-F1 points (1 point = 0.01 F1)")
        ax.grid(axis="x", alpha=0.15)
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#237B70",
                    linewidth=6,
                    label="Descriptive 95% interval",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#8997A1",
                    linewidth=2.3,
                    marker="|",
                    markersize=10,
                    label="98.75% family interval for the two baseline contrasts",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.45, -0.23),
            frameon=False,
            fontsize=9,
        )
        fig.text(
            0.31,
            0.04,
            "Four-comparison family retained; legal contrasts are evaluated separately in Part 3.",
            fontsize=9,
        )
        plotted["macro_f1_contrasts"] = points
        files += publish(fig, folder, "german_hate_test_macro_contrasts")
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), sharey=True)
        fig.subplots_adjust(left=0.22, right=0.98, top=0.8, bottom=0.24, wspace=0.13)
        fig.suptitle(
            "German hate TEST: errors avoided and new errors",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )
        flow_data = {}
        for ax, error, title, limit in zip(
            axes, ("fp", "fn"), ("False alarms", "Missed hateful texts"), (116, 43)
        ):
            records = []
            for i, value in enumerate(d["contrasts"]):
                flow = value["transitions"]["mean_events"]
                fixed, new = (flow[error + "_fixed"], flow[error + "_introduced"])
                ax.barh(2 - i + 0.16, fixed, height=0.29, color="#237B70")
                ax.barh(2 - i - 0.16, new, height=0.29, color="#B34F43")
                for number, y in ((fixed, 2 - i + 0.16), (new, 2 - i - 0.16)):
                    ax.text(
                        number + limit * 0.018,
                        y,
                        f"{number:.1f}",
                        va="center",
                        fontsize=10,
                    )
                records.append(
                    {"comparison": PAIR_NAMES[i], "avoided": fixed, "introduced": new}
                )
            ax.set_title(title, fontsize=12, pad=12)
            ax.set_yticks([2, 1, 0], PAIR_NAMES)
            ax.set_xlim(0, limit)
            ax.set_xlabel("Mean number of texts per comparison")
            ax.grid(axis="x", alpha=0.17)
            ax.set_axisbelow(True)
            flow_data[error] = records
        fig.legend(
            handles=[
                Patch(color="#237B70", label="Reference error avoided"),
                Patch(color="#B34F43", label="New error"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.59, 0.075),
            ncol=2,
            frameon=False,
            fontsize=10,
        )
        fig.text(
            0.22,
            0.035,
            "Averages over all nine cross-repeat pairings on the same texts; these are not nine independent samples.",
            fontsize=9,
        )
        plotted["error_flows"] = flow_data
        files += publish(fig, folder, "german_hate_test_error_flows")
        h.save(h.BASE / "report/figure_data.json", plotted)
        h.save(
            h.BASE / "report/figure_manifest.json",
            {
                "status": "rendered_from_verified_part2_results",
                "inputs": h.c.bindings(
                    [
                        Path(__file__),
                        h.BASE / "complete.json",
                        run / "de_test/results.json",
                    ]
                ),
                "outputs": h.c.bindings([*files, h.BASE / "report/figure_data.json"]),
                "matplotlib_version": matplotlib.__version__,
                "actual_Qwen_calls": 0,
                "latex_edits": 0,
            },
        )
        print("\n".join((h.c.ref(path) for path in files)), flush=True)


if __name__ == "__main__":
    main()
