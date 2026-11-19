from pathlib import Path
import os
import json
import numpy as np
from analysis.knowledge.german_guidelines import legal as l


def main():
    with l.offline():
        _, run = l.verified()
        out = l.BASE / "report"
        figures = out / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(out / "matplotlib_cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "savefig.dpi": 180,
                "svg.fonttype": "none",
            }
        )
        r = l.c.read(run / "de_legal_test/results.json")
        labels = [
            "Current shared slots",
            "Two general guidelines",
            "Two general + one legal",
        ]
        colors = ["#276886", "#e6a23c", "#7b5aa6"]
        data = {
            "population": "German legal TEST",
            "n_common": r["common_inputs"],
            "gold_support": r["common_gold_support"],
            "conditions": list(l.CONDITIONS),
            "confidence": "Descriptive 95% paired whole-cluster intervals. Use paired contrast intervals for setting differences.",
            "score_panels": {},
            "category_effects": {},
            "set_partition": {},
        }
        outputs = []

        def export(fig, stem):
            for suffix in ("png", "svg"):
                path = figures / (stem + "." + suffix)
                l.require(not path.exists(), "Figure already exists")
                fig.savefig(path, bbox_inches="tight", facecolor="white")
                outputs.append(path)
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))
        for ax, name, title, scale, xlim in zip(
            axes,
            ("macro_f1", "exact_set_accuracy"),
            ("Legal macro-F1", "Entire legal label set correct"),
            (1.0, 100.0),
            ((0.48, 0.81), (40, 80)),
        ):
            rows = []
            for i, cond in enumerate(l.CONDITIONS):
                mean = r["common"][cond]["mean"][name]
                ci = r["condition_descriptive_intervals"][cond][name]["ci95"]
                repeats = [row[name] for row in r["common"][cond]["repeats"]]
                ax.hlines(i, ci[0] * scale, ci[1] * scale, color=colors[i], lw=2.3)
                ax.scatter(
                    [v * scale for v in repeats],
                    [i] * 3,
                    facecolors="white",
                    edgecolors=colors[i],
                    s=33,
                    linewidths=1.2,
                    zorder=3,
                )
                ax.scatter(mean * scale, i, color=colors[i], marker="D", s=42, zorder=4)
                ax.annotate(
                    f"{mean:.4f}" if scale == 1 else f"{mean * 100:.2f}%",
                    (mean * scale, i),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                )
                rows.append(
                    {"condition": cond, "mean": mean, "ci95": ci, "repeats": repeats}
                )
            data["score_panels"][name] = rows
            ax.set(
                yticks=range(3),
                yticklabels=labels,
                ylim=(2.5, -0.6),
                xlim=xlim,
                xlabel=(
                    "F1 (higher is better)"
                    if scale == 1
                    else "Percentage of texts (higher is better)"
                ),
                title=title,
            )
            ax.grid(axis="x", alpha=0.18)
        fig.suptitle(
            "Separate legal guidance did not improve average overall legal scores",
            fontsize=14,
            y=1.01,
        )
        fig.text(
            0.5,
            -0.015,
            "174 shared valid TEST texts. Diamonds: mean of three repeats; circles: each repeat; lines: descriptive 95% intervals.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(w_pad=3)
        export(fig, "legal_test_scores")
        added = next(
            (
                v
                for v in r["contrasts"]
                if v["left"] == "general_2_legal_1" and v["right"] == "general_2"
            )
        )
        names = ["Propaganda", "Incitement / public peace", "Insult / defamation"]
        fig, ax = plt.subplots(figsize=(10.5, 4.3))
        for i, label in enumerate(l.LABELS):
            v = added["per_label"][label]["f1"]
            ci = np.asarray(v["ci95"]) * 100
            value = v["difference"] * 100
            ax.hlines(i, ci[0], ci[1], color="#276886", lw=3)
            ax.scatter(value, i, color="#276886", marker="D", s=48, zorder=3)
            ax.annotate(
                f"{value:+.2f}",
                (value, i),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
            )
            data["category_effects"][label] = v
        ax.axvline(0, color="#555555", ls="--", lw=1)
        ax.set(
            yticks=range(3),
            yticklabels=names,
            ylim=(2.5, -0.6),
            xlim=(-10.5, 8),
            xlabel="Change in category F1 points: separate legal slot minus general-only",
            title="The added legal rule had different effects across categories",
        )
        ax.grid(axis="x", alpha=0.18)
        fig.text(
            0.5,
            -0.015,
            "Same 174 TEST texts and three fixed repeats. Descriptive 95% intervals; category comparisons are exploratory.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout()
        export(fig, "legal_test_added_rule_categories")
        fig, ax = plt.subplots(figsize=(12.2, 4.7))
        fields = ["exact_correct", "only_extra", "only_missing", "mixed_error"]
        titles = [
            "Entire set correct",
            "Extra labels only",
            "Missing labels only",
            "Extra and missing labels",
        ]
        palette = ["#3f8b75", "#dc9a3e", "#aa596c", "#79558b"]
        left = np.zeros(3)
        for field, title, color in zip(fields, titles, palette):
            values = np.array(
                [r["common"][cond]["mean"][field] for cond in l.CONDITIONS]
            )
            ax.barh(
                np.arange(3), values, left=left, label=title, color=color, height=0.56
            )
            for i, value in enumerate(values):
                ax.text(
                    left[i] + value / 2,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="white" if field != "only_extra" else "#222222",
                    fontsize=9,
                )
            data["set_partition"][field] = values.tolist()
            left += values
        l.require(np.allclose(left, 174), "Set chart does not partition texts")
        ax.set(
            yticks=range(3),
            yticklabels=labels,
            ylim=(2.65, -0.55),
            xlim=(0, 174),
            xlabel="Number of texts, averaged across three repeats",
            title="Extra labels dominate the incorrect legal answers",
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        fig.text(
            0.5,
            -0.06,
            "Each text is counted once. A single text can contain multiple incorrect labels; this chart counts texts, not labels.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout()
        export(fig, "legal_test_set_errors")
        l.save(out / "figure_data.json", data)
        l.save(
            out / "figure_manifest.json",
            {
                "status": "part3_figures_generated",
                "matplotlib_version": matplotlib.__version__,
                "inputs": l.c.bindings(
                    [
                        Path(__file__),
                        l.BASE / "complete.json",
                        run / "de_legal_test/results.json",
                    ]
                ),
                "outputs": l.c.bindings([*outputs, out / "figure_data.json"]),
                "actual_Qwen_calls": 0,
                "latex_edits": 0,
            },
        )
        print(json.dumps({"figures": [str(p) for p in outputs if p.suffix == ".png"]}))


if __name__ == "__main__":
    main()
