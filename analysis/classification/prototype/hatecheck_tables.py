import hashlib
import json
from pathlib import Path

SOURCE = Path("Thesis_Log/astra_result/evidence/hatecheck_statistics_20260910_v1.json")
DEST = Path(
    "Thesis_Log/astra_result/evidence/hatecheck_statistics_20260910_v1_tables.md"
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    if DEST.exists():
        raise ValueError(f"Refusing to overwrite {DEST}")
    e = json.loads(SOURCE.read_text(encoding="utf-8"))
    for entry in e["inputs"]:
        assert sha(entry["path"]) == entry["sha256"], entry["path"]
    lines = [
        "# Astra HateCheck: complete numerical tables",
        "",
        "Checkpoint: `ASTRA-SQ2-P2-C4-20260910-v1`.",
        "",
        f"Source evidence: `{SOURCE.as_posix()}`. SHA256: `{sha(SOURCE)}`.",
        "",
        "These tables render the completed audit output. They are not another statistical run. "
        "All intervals are nominal 95% percentile intervals from 3,000 template-cluster resamples. "
        "P-values use the inclusive bootstrap sign-tail convention described in the JSON. "
        "A displayed zero is finite-resampling output, not a population probability of zero.",
        "",
        "## Source runs",
        "",
        "| Arm | Replicate | Source stem | Rows | Unscored | Missing rows |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in e["source_inventory"]:
        lines.append(
            f"| {r['arm']} | {r['replicate']} | {r['stem']} | {r['rows']} | {len(r['unscored_ids'])} | {len(r['missing_ids'])} |"
        )
    lines += [
        "",
        "Exact excluded item IDs, manifests, stamps, and input hashes are retained in the JSON.",
        "",
        "## Primary three-arm gate",
        "",
        "| Arm | Rep 1 | Rep 2 | Rep 3 | Mean | Range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, r in e["primary_gate_scores"].items():
        vals = " | ".join(f"{v:.6f}" for v in r["replicates"])
        lines.append(f"| {arm} | {vals} | {r['mean']:.6f} | {r['spread']:.6f} |")
    lines += [
        "",
        "| Contrast | Pair | Delta | 95% interval | p |",
        "|---|---:|---:|---|---:|",
    ]
    for c in e["primary_gate_comparisons"]:
        for r in c["pairs"]:
            lines.append(
                f"| {c['arm_a']} - {c['arm_b']} | {r['replicate_pair']} | {r['delta']:+.6f} | "
                f"[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | {r['p_value']:.6f} |"
            )
    lines += [
        "",
        "| Contrast | Mean delta | Floor | Ratio | Intervals excluding zero |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in e["primary_gate_comparisons"]:
        lines.append(
            f"| {r['arm_a']} - {r['arm_b']} | {r['delta']:+.6f} | {r['floor']:.6f} | "
            f"{r['magnitude_over_floor']:.2f} | {r['intervals_excluding_zero']}/3 |"
        )
    for scope, record in e["seven_arm_results"].items():
        lines += [
            "",
            f"## Seven-arm scope: {scope}",
            "",
            f"Items: {record['n_items']}; templates: {record['n_templates']}; metric: `{record['metric']}`.",
            "",
            "Intervals below use one shared seed-42 resample stream for all arms and fixed replicates within this scope. "
            "They are exploratory supplementary intervals; they are distinct from the legacy RNG reproduction below.",
            "",
            "| Arm | Rep 1 | Rep 2 | Rep 3 | Mean | Range |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for arm, v in record["replicate_scores"].items():
            lines.append(
                f"| {arm} | "
                + " | ".join(f"{x:.6f}" for x in v)
                + f" | {sum(v)/len(v):.6f} | {max(v)-min(v):.6f} |"
            )
        lines += [
            "",
            "| Contrast | Delta | Floor | Ratio | 95% interval | p |",
            "|---|---:|---:|---:|---|---:|",
        ]
        for r in record["contrasts"]:
            lines.append(
                f"| {r['arm_a']} - {r['arm_b']} | {r['delta']:+.6f} | {r['floor']:.6f} | {r['magnitude_over_floor']:.2f} | "
                f"[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | {r['p_value']:.6f} |"
            )
        lines += [
            "",
            "| Composition residual | Delta | Floor | Ratio | 95% interval | p |",
            "|---|---:|---:|---:|---|---:|",
        ]
        for r in record["composition_residuals"]:
            lines.append(
                f"| {r['name']} | {r['delta']:+.6f} | {r['floor']:.6f} | {r['magnitude_over_floor']:.2f} | "
                f"[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | {r['p_value']:.6f} |"
            )
        if record["metric"] == "gate":
            lines += ["", "| Arm | FP r1/r2/r3 | FN r1/r2/r3 |", "|---|---|---|"]
            for arm, ss in record["scores"].items():
                lines.append(
                    f"| {arm} | "
                    + " / ".join(str(r["fp"]) for r in ss)
                    + " | "
                    + " / ".join(str(r["fn"]) for r in ss)
                    + " |"
                )
        if record["metric"] == "target_group":
            lines += [
                "",
                "| Label | Gold support | Definitions-minus-zero-shot mean F1 |",
                "|---|---:|---:|",
            ]
            for label in record["labels"]:
                lines.append(
                    f"| {label} | {record['gold_support'][label]['support']} | {record['definitions_minus_zero_shot_per_label_delta'][label]:+.6f} |"
                )
            lines += [
                "",
                f"Gold cardinality: {record['gold_cardinality']:.6f}. Sexual-orientation share of net definition gain: "
                f"{record['sexual_orientation_share_of_definition_gain']:.6f}.",
                "",
            ]
    lines += [
        "",
        "## Original FP random-stream reproduction",
        "",
        "Matches the scope and contrast order in `analysis/classification/prototype/six_arms.py`. "
        "Do not splice these interval endpoints into a claim that all contrasts used the same random draws.",
        "",
        "| Scope | Contrast | Delta | 95% interval | p |",
        "|---|---|---:|---|---:|",
    ]
    for r in e["legacy_fp_bootstrap"]:
        lines.append(
            f"| {r['scope']} | {r['arm_a']} - {r['arm_b']} | {r['delta']:+.6f} | "
            f"[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | {r['p_value']:.6f} |"
        )
    lines += [
        "",
        "The primary/legacy console helper printed a zero floor placeholder on rows without a floor field. "
        "No zero floor was used in the calculations. Use the actual mean-contrast floors in these tables and the JSON.",
        "",
        "Interpretation and P2 status: `Thesis_Log/astra_result/astra_sq2.md`, checkpoint C4. Priority 3 was not started.",
        "",
    ]
    with DEST.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print(f"WROTE NEW TABLES: {DEST}")
    print(
        "Source hashes verified; 9 primary pair intervals, 24 legacy FP intervals, 72 supplementary contrasts, 12 residuals rendered."
    )


if __name__ == "__main__":
    main()
