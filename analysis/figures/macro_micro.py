from src.hsrag.paths import PROJECT_ROOT
import hashlib
import json
from pathlib import Path

ROOT = PROJECT_ROOT
MACRO_SHA256 = "7b57949d2f5ff20cc33f7ce5f2bfcc832855240f04c91c2739437ee3838315b2"
REFERENCE_SHA256 = "f665da24fd8a1f08298055c3ac7f9d403abe4de443de453f6a23a170222d83e2"
MICRO_SHA256 = {
    "de_legal_dev_eval/results.json": "f34f85bd298ee24902eec4eb0c802e7380e97d7df8e8c7161966b4ec8bfc2fde",
    "en_dev_eval_sq3_types/results.json": "8c46963163a2eb5d398fdddf3afcc8b209f4df00a149e8eed249ca420d88b945",
    "en_dev_eval_targets/results.json": "e3b6bc3a70ccf6b6c791c260dab1669a745f048161b87a604be6de2defc6f2cd",
    "en_dev_eval_types/results.json": "7bc162dcd16083f91284c7950a8a1ea07f73669ad686ee0ccd7a555f354b452b",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


DEFAULT_MACRO = (
    ROOT / "Thesis_Log/astra_result/evidence/core_completion_summary_20260920_v1.json"
)
DEFAULT_MICRO = (
    ROOT
    / "experiments/astra/micro_f1_sq2_retrieval_20261113_v1/attempts/20261113T145217828441_core"
)
DEFAULT_OUTPUT = ROOT / "results/figures/macro_micro"
ORDER = [
    ("en_dev_eval_main", "English hate"),
    ("de_legal_dev_eval", "German legal categories"),
    ("en_dev_eval_sq3_types", "English types: main"),
    ("en_dev_eval_targets", "English targets"),
    ("en_dev_eval_severity", "English severity"),
    ("de_dev_eval", "German hate"),
    ("en_dev_eval_types", "English types: sensitivity"),
]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Draw macro and supplementary micro F1 from the matching saved scores."
    )
    parser.add_argument("--macro-json", type=Path, default=DEFAULT_MACRO)
    parser.add_argument("--micro-directory", type=Path, default=DEFAULT_MICRO)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-pdf",
        type=Path,
        help="Optionally verify the original figure PDF hash",
    )
    args = parser.parse_args()
    MACRO, MICRO, OUT = (args.macro_json, args.micro_directory, args.output_directory)
    EXPECTED = {
        MACRO: MACRO_SHA256,
        **{MICRO / name: value for name, value in MICRO_SHA256.items()},
    }
    if args.reference_pdf is not None:
        EXPECTED[args.reference_pdf] = REFERENCE_SHA256
    for path, expected in EXPECTED.items():
        assert sha(path) == expected, f"Source hash differs: {path}"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    original = read(MACRO)["endpoints"]
    rows = []
    for endpoint, label in ORDER:
        m = original[endpoint]["full_minus_small"]
        assert (m["left"], m["right"]) == ("rag_full", "rag_small")
        record = {
            "endpoint": endpoint,
            "label": label,
            "n_shared_valid": m["n_common_valid"],
            "macro_primary": m["primary"],
            "macro": {
                "full": m["left_mean_on_common_valid"],
                "compact": m["right_mean_on_common_valid"],
                "difference": m["mean_difference"],
                "interval": m["family_ci"] if m["primary"] else m["ci95"],
                "confidence": m["family_confidence"] if m["primary"] else 0.95,
            },
            "micro": None,
        }
        micro_path = MICRO / endpoint / "results.json"
        if micro_path in EXPECTED:
            d = read(micro_path)
            matches = [
                v for v in d["contrasts"] if v["name"] == "rag_full_minus_rag_small"
            ]
            assert len(matches) == 1
            c = matches[0]
            assert c["n_common"] == record["n_shared_valid"]
            assert c["left_settings"] == [["rag_full", i] for i in range(3)]
            assert c["right_settings"] == [["rag_small", i] for i in range(3)]
            assert c["label_scope"]["full"] == c["label_scope"]["macro"]
            for key, original_key in [
                ("left", "left_mean_on_common_valid"),
                ("right", "right_mean_on_common_valid"),
                ("difference", "mean_difference"),
            ]:
                assert abs(c["metrics"]["macro"][key] - m[original_key]) < 1e-12
            u = c["metrics"]["micro"]
            adjusted = u["adjusted_interval"] is not None
            assert adjusted == (endpoint != "en_dev_eval_types")
            if adjusted:
                assert c["family_size"] == 9
                assert abs(u["adjusted_confidence"] - (1 - 0.05 / 9)) < 1e-12
            record["micro"] = {
                "full": u["left"],
                "compact": u["right"],
                "difference": u["difference"],
                "interval": u["adjusted_interval"] if adjusted else u["ci95"],
                "confidence": u["adjusted_confidence"] if adjusted else 0.95,
                "nominal_95_interval": u["ci95"],
            }
        rows.append(record)
    assert len(rows) == 7 and sum((r["micro"] is not None for r in rows)) == 4
    for row in rows:
        for metric in ["macro", "micro"]:
            v = row[metric]
            if v is not None:
                assert abs(v["full"] - v["compact"] - v["difference"]) < 1e-12
                assert v["interval"][0] <= v["difference"] <= v["interval"][1]
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": "#253447",
            "text.color": "#253447",
            "xtick.color": "#253447",
            "ytick.color": "#253447",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(11.7, 7.4), facecolor="white")
    ax = fig.add_axes([0.315, 0.265, 0.52, 0.535])
    blue, teal, orange = ("#16529A", "#22828A", "#C66A13")
    for i in range(len(rows)):
        if i % 2 == 1:
            ax.axhspan(i - 0.45, i + 0.45, facecolor="#F4F6F8", zorder=0)
    ax.axvline(0, color="#6F7883", lw=1.05, ls=(0, (3, 3)), zorder=1)
    for i, row in enumerate(rows):
        for name in ["macro", "micro"]:
            v = row[name]
            if v is None:
                continue
            if name == "macro":
                color = blue if row["macro_primary"] else teal
                marker = "s" if row["macro_primary"] else "o"
                y = i - 0.14 if row["micro"] else i
            else:
                color, marker, y = (orange, "D", i + 0.14)
            point = 100 * v["difference"]
            lo, hi = [100 * x for x in v["interval"]]
            ax.errorbar(
                point,
                y,
                xerr=[[point - lo], [hi - point]],
                fmt=marker,
                color=color,
                markersize=6.4,
                capsize=3.3,
                elinewidth=1.7,
                capthick=1.2,
                zorder=3,
            )
        macro_color = blue if row["macro_primary"] else teal
        ax.text(
            1.075,
            i,
            f"{100 * row['macro']['difference']:+.2f}",
            transform=ax.get_yaxis_transform(),
            color=macro_color,
            va="center",
            ha="center",
            fontsize=11,
        )
        micro_text = f"{100 * row['micro']['difference']:+.2f}" if row["micro"] else "—"
        ax.text(
            1.215,
            i,
            micro_text,
            transform=ax.get_yaxis_transform(),
            color=orange if row["micro"] else "#A0A7AF",
            va="center",
            ha="center",
            fontsize=11,
        )
    ax.set_yticks(
        range(len(rows)), [f"{r['label']}  (n = {r['n_shared_valid']})" for r in rows]
    )
    ax.set_ylim(6.55, -0.55)
    ax.set_xlim(-25, 15)
    ax.set_xticks([-20, -10, 0, 10], ["−20", "−10", "0", "+10"])
    ax.set_xlabel("Full minus compact F1 (percentage points)", labelpad=10)
    ax.grid(axis="x", color="#D9DFE6", lw=0.6, zorder=0)
    ax.tick_params(axis="y", length=0, pad=11)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#98A1AB")
    ax.text(
        1.075,
        1.035,
        "Macro",
        transform=ax.transAxes,
        ha="center",
        weight="bold",
        fontsize=10,
    )
    ax.text(
        1.215,
        1.035,
        "Micro",
        transform=ax.transAxes,
        ha="center",
        weight="bold",
        fontsize=10,
    )
    fig.text(
        0.04,
        0.955,
        "Full versus compact: macro and micro F1",
        fontsize=17,
        weight="bold",
    )
    fig.text(
        0.04,
        0.915,
        "Positive differences favour the full knowledge bank.",
        fontsize=11.5,
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=blue,
                marker="s",
                lw=1.7,
                markersize=6,
                label="Macro: primary",
            ),
            Line2D(
                [],
                [],
                color=teal,
                marker="o",
                lw=1.7,
                markersize=6,
                label="Macro: supporting",
            ),
            Line2D(
                [],
                [],
                color=orange,
                marker="D",
                lw=1.7,
                markersize=6,
                label="Micro: supplementary",
            ),
        ],
        loc="center left",
        bbox_to_anchor=(0.035, 0.862),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        columnspacing=2.6,
    )
    fig.text(
        0.04,
        0.14,
        "Macro intervals: 98.75% for the two primary comparisons; 95% for the others.",
        fontsize=10,
    )
    fig.text(
        0.04,
        0.108,
        "Micro intervals: 99.4444%, except the 103-text sensitivity check (95%).",
        fontsize=10,
    )
    fig.text(
        0.04,
        0.076,
        "The adjustments cover different comparison families. Micro is added only for multilabel outputs.",
        fontsize=10,
    )
    fig.text(
        0.04,
        0.036,
        "Same saved responses and shared-valid texts; paired item bootstrap. No new experiment.",
        fontsize=9.5,
        color="#626D7B",
    )
    stem = OUT / "figure3_macro_micro_preview"
    fig.savefig(stem.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(
        stem.with_suffix(".pdf"),
        facecolor="white",
        metadata={"Title": "Figure 3 preview: macro and micro F1"},
    )
    plt.close(fig)
    provenance = {
        "status": "preview_only_thesis_unchanged",
        "sources_sha256": {str(p): digest for p, digest in EXPECTED.items()},
        "units_in_saved_data": "F1 on the 0–1 scale; figure multiplies by 100",
        "comparison": "Mean of three repeat scores for full minus compact, on the original common valid texts",
        "interval_policy": "Preserve original macro family intervals; preserve separate nine-question micro family intervals; nominal 95% for sensitivity",
        "rows": rows,
    }
    stem.with_suffix(".json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path, expected in EXPECTED.items():
        assert sha(path) == expected, f"Source changed during rendering: {path}"
    for r in rows:
        print(
            r["endpoint"],
            "n=",
            r["n_shared_valid"],
            "macro_pp=",
            round(100 * r["macro"]["difference"], 4),
            "micro_pp=",
            round(100 * r["micro"]["difference"], 4) if r["micro"] else None,
        )
    print("Numerical source hashes unchanged.")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
