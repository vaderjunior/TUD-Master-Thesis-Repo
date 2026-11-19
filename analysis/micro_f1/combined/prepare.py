from analysis.common import micro_f1_files as c
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib

BASE = c.ROOT / "experiments/astra/micro_f1_combined_20261113_v1"
NAMES = {
    1: "micro_f1_setup_20261113_v1",
    2: "micro_f1_sq2_retrieval_20261113_v1",
    3: "micro_f1_sq2_supervised_20261113_v1",
    4: "micro_f1_sq2_matched_20261113_v1",
    5: "micro_f1_sq1_banks_strategies_20261113_v1",
    6: "micro_f1_sq1_components_guidelines_20261113_v1",
    7: "micro_f1_sq3_20261113_v1",
}
ROOTS = {k: c.ROOT / "experiments/astra" / v for k, v in NAMES.items()}


def save(name, value):
    path = (BASE / name).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Write outside new Part 8 evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def freeze():
    c.require(not BASE.exists(), "Preserve existing combined analysis")
    prior = c.read(ROOTS[7] / "source_preservation.json")
    complete = c.read(ROOTS[7] / "complete.json")
    c.require(complete["status"] == "Part 7 complete", "Part 7 incomplete")
    for i, (name, digest) in enumerate(prior["source_sha256"].items(), 1):
        c.require(c.sha(c.ROOT / name) == digest, "Earlier source changed: " + name)
        if i % 500 == 0:
            print("Checking prior source hashes", i, flush=True)
    changes = dict()
    BASE.mkdir(parents=True)
    registry = c.read(ROOTS[1] / "comparison_registry_v2.json")
    save(
        "preservation_before.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            old_sources=prior["source_sha256"],
        ),
    )
    save(
        "analysis_plan.json",
        dict(
            part=8,
            status="scope fixed before combined numerical diagnostics",
            created_utc=datetime.now(timezone.utc).isoformat(),
            scope="All 77 prelisted supplementary questions in 17 separate families, plus complete existing supporting-change inventories. Reuse Parts 2–7; Part 5 core reuse is not a second result.",
            families=registry["new_micro_families"],
            contract="No new predictions, inference, fitting, cutoff or retrieval. Preserve original macro, full micro and macro-label-matched micro, annotation/gate/validity masks, repeat/pool hierarchy, effect direction and original inference roles.",
            matrix="Claim-by-claim SQ1/SQ2/SQ3 conclusions, distinguishing stable findings, metric-specific support, uncertain point reversals, supported opposite signs, confidence-family differences and descriptive-only outcomes. Include adverse and null findings. No global F1, pooled study effect, thesis-wide significance guarantee or independent win count.",
            robustness="Audit existing strict validity, missing-output bounds, source/cluster resampling, pool/seed/repeat variation, gate and exactness diagnostics. Do not invent an unperformed sensitivity. Label full-roster bounds separately from paired confidence intervals.",
            planned_derived_checks=[
                "Reconcile all registered question IDs and families without duplicates or omissions.",
                "Recompute interval classifications from full-precision endpoints at matched 95%, retaining separate original-macro and supplementary-micro adjusted confidence.",
                "Decompose point differences into macro-to-restricted-micro weighting and restricted-to-full-micro label-scope terms where defined.",
                "Inspect exact zero-boundary distances and effective tail ranks for existing 10000-draw intervals. Flag adjusted endpoints within 0.25 percentage points of zero for numerical wording review; this descriptive threshold is not a new test.",
                "Compare pre-existing same-confidence and label-scope diagnostics; no new bootstrap sample or chosen alternative interval.",
                "Verify known reused common-cohort and source populations; quantify selected same-ID intersections only where original ID namespaces agree. Do not combine scores across those studies.",
                "Read live thesis conclusions/discussion to identify precise qualifications without editing LaTeX.",
            ],
            new_diagnostics="If a discrepancy requires a new targeted computation, record an addendum first. Source scores and original inference are never silently replaced.",
            outputs="Combined numeric ledger, complete change inventory, robustness matrix, scientific conclusion matrix, detailed report, append-only writer actions M77 onward, session/roadmap/preservation receipts. Part 9 consolidated implementation package remains separate.",
            authority="User authorises offline analysis and log append only. Macro primary; plain writing; useful modest growth above 15000 allowed; no new trimming.",
            new_qwen_calls=0,
            new_training=0,
            new_retrieval=0,
            latex_edits=0,
        ),
    )


if __name__ == "__main__":
    freeze()
