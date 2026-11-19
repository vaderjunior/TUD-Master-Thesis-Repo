import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import argparse
import json
from analysis.feedback.transfer import archive as a

ROOT = a.ROOT
OUT = ROOT / "experiments/astra/sq3_transfer_part10_20261023_v1"
LOG = ROOT / "Thesis_Log"
RESULT = LOG / "astra_result"
PARTS = {
    i: ROOT / f"experiments/astra/sq3_transfer_part{i}_20261023_v1"
    for i in range(1, 10)
}
REPORT = RESULT / "astra_sq3_transfer_final_results_20261023_v1.md"
HANDOFF = RESULT / "astra_sq3_transfer_writing_handoff_20261023_v1.md"
VERIFY_REPORT = RESULT / "astra_sq3_transfer_part10_verification_20261023_v1.md"
RESOURCE = RESULT / "astra_resource_evidence_20261023_v7.md"
OLDRESOURCE = RESULT / "astra_resource_evidence_20261020_v6.md"
JOURNAL = LOG / "astra_master_research_journal.md"
JRECEIPT = RESULT / "evidence/astra_master_research_journal_20261023_v16.json"
JQA = RESULT / "evidence/astra_master_journal_qa_20261023_v16/verification.json"
UID = "WR-20261023-SQ3-TRANSFER-FINAL-v1"
MARK = "ASTRA-TRANSFER-FINAL-20261023-v1"
PROOFS = {
    1: ["independent_matrix_check.json"],
    2: ["independent_check.json", "interpretation_check.json", "verified.json"],
    3: [
        "independent_check.json",
        "interpretation_check.json",
        "final_publication_check.json",
    ],
    4: [
        "independent_numerical_check.json",
        "verified.json",
        "publication/final_check.json",
    ],
    5: [
        "independent_context_check.json",
        "verified.json",
        "publication/final_check.json",
    ],
    6: ["verified.json", "cases_verified.json", "publication/final_check.json"],
    7: [
        "verified.json",
        "failure_details_verified.json",
        "publication/final_check.json",
    ],
    8: [
        "verified.json",
        "figure_receipt_checked_v1.json",
        "publication/final_check.json",
    ],
    9: ["verified.json", "repair_depth.json", "publication/final_check.json"],
}
DRAW_PROOFS = {
    2: ("independent_check.json", "bootstrap_sets", "bootstrap_scalars_checked"),
    3: ("independent_check.json", "new_bootstrap_sets", "new_scalar_values"),
    4: ("independent_numerical_check.json", "new_bootstrap_sets", "new_scalar_values"),
    5: ("verified.json", "new_bootstrap_sets", "new_scalar_values"),
    7: ("verified.json", "new_bootstrap_sets", "verified_bootstrap_scalars"),
    8: ("verified.json", "bootstrap_sets", "verified_bootstrap_scalars"),
    9: ("verified.json", "paired_bootstrap_sets", "verified_bootstrap_scalars"),
}
CURRENT_FIGURES = {
    2: "figures_v2",
    3: "figures_v2",
    4: "figures_v1",
    5: "figures_v1",
    6: "figures_v2",
    7: "figures_v1",
    8: "figures_v2",
    9: "figures_v2",
}
ENDPOINTS = (
    "en_test_FPR",
    "en_test_FNR",
    "en_hatecheck_eval_FPR",
    "en_hatecheck_eval_FNR",
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value
            if isinstance(value, bytes)
            else (
                value.encode("utf8")
                if isinstance(value, str)
                else a.encoded(value, pretty=True)
            )
        )


def bindings_check(values):
    for name, sha in values.items():
        a.need(a.sha(ROOT / name) == sha, "Evidence changed: " + name)


def prepare():
    a.need(not OUT.exists(), "Preserve previous synthesis")
    OUT.mkdir()
    bound = {}
    catalog = []
    draws = []
    figures = []

    def bind(name, digest):
        a.need(
            name not in bound or bound[name] == digest,
            "Conflicting immutable binding " + name,
        )
        bound[name] = digest

    for i, path in PARTS.items():
        c = a.read(path / "complete.json")
        a.need(c["new_Qwen_calls"] == 0 and c["new_fits"] == 0, "Offline prior review")
        for group in ("input_bindings", "analysis_sources", "artifacts"):
            for name, sha in c.get(group, {}).items():
                bind(name, sha)
        for file in [path / "complete.json", *[path / name for name in PROOFS[i]]]:
            bind(a.rel(file), a.sha(file))
        catalog.append(
            dict(
                part=i,
                complete=a.rel(path / "complete.json"),
                complete_sha256=a.sha(path / "complete.json"),
                report=c.get("report"),
                proofs=[
                    dict(
                        path=a.rel(path / name),
                        sha256=a.sha(path / name),
                        status=a.read(path / name).get("status"),
                    )
                    for name in PROOFS[i]
                ],
                historical_remaining_markers=True,
            )
        )
        if i in CURRENT_FIGURES:
            folder = path / CURRENT_FIGURES[i]
            for png in sorted(folder.glob("*.png")):
                svg = png.with_suffix(".svg")
                a.need(svg.exists(), "Editable scientific figure")
                figures.append(
                    dict(
                        part=i,
                        name=png.stem,
                        png=a.rel(png),
                        png_sha256=a.sha(png),
                        svg=a.rel(svg),
                        svg_sha256=a.sha(svg),
                        existing_visual_review=True,
                    )
                )
        if i in DRAW_PROOFS:
            proof, sets, scalars = DRAW_PROOFS[i]
            q = a.read(path / proof)
            draws.append(
                dict(
                    part=i,
                    proof=a.rel(path / proof),
                    new_sets_in_original_part=q[sets],
                    independently_verified_scalars=q[scalars],
                    regenerated_in_part10=False,
                )
            )
    for path in (OLDRESOURCE,):
        bind(a.rel(path), a.sha(path))
    for name in (
        "astra_sq3_reserved_final_results_20261020_v1.md",
        "astra_sq3_human_feedback_results_20261020_v1.md",
        "astra_sq3_linear_dev_results_20261020_v1.md",
        "astra_sq3_available_synthesis_20261020_v1.md",
    ):
        path = RESULT / name
        bind(a.rel(path), a.sha(path))
    bindings_check(bound)
    a.need(len(figures) == 25, "Current figures inventory")
    a.need(
        sum((q["new_sets_in_original_part"] for q in draws)) == 335
        and sum((q["independently_verified_scalars"] for q in draws)) == 77220000,
        "Unique numerical proof scope",
    )
    save(
        OUT / "analysis_contract.json",
        dict(
            created_utc=a.utc(),
            scope="Final synthesis of Parts 1-9 and principal static classifier comparison, with current immutable proof bindings, fresh direct main-point arithmetic, saved-interval transcription checks and exact-once remote-ledger extension. No new model evaluation or statistical family.",
            input_bindings=bound,
            prior_parts=catalog,
            current_figures=figures,
            reused_resampling_proofs=draws,
            unique_prior_verified_bootstrap_sets=335,
            unique_prior_verified_bootstrap_scalars=77220000,
            new_bootstrap_draws=0,
            publication_boundaries=dict(
                close_broader_transfer=True,
                principal_static_classifier_complete=True,
                cumulative_classifier_pending=True,
                all_SQ3_complete=False,
                experimental_complete_after=4,
                experimental_open_after=5,
                inventory_rows=16,
            ),
            plan=[
                "Verify prior receipts and immutable source/display bindings; preserve historical versions and qualifications",
                "Reconstruct four primary points and four classifier contrasts directly from saved decisions, plus main binary/known-field/set quality",
                "Reuse the 335 independently regenerated prior draw sets; recheck the eight headline intervals from saved sorted values",
                "Publish synthesis, exact display tables, claim map and current writer integration plan",
                "Extend resource ledger once, journal v16 and roadmap v85; append a source-bound writer update without acknowledging it",
                "Verify links, embedded figures, original evidence, additive history and writer ownership",
            ],
            script_sha256=a.sha(Path(__file__)),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_annotations=0,
            new_primary_families=0,
        ),
    )
    print(
        "Part 10 contract prepared:",
        len(bound),
        "immutable bindings, 25 figures, 335 previously verified draw sets. No experiments launched.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare"])
    args = parser.parse_args()
    with a.offline():
        prepare()
