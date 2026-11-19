import argparse
from functools import lru_cache
from pathlib import Path
import re
from analysis.classification.broad import resources as x

ROOT = x.ROOT
BASE = ROOT / "experiments/astra/sq2_broad_synthesis_20261104_v1"
REPORT = ROOT / "Thesis_Log/astra_result/astra_sq2_broad_final_synthesis_20261104_v1.md"
PARTS = (
    "reader",
    "main",
    "errors",
    "quality",
    "sources",
    "context",
    "robustness",
    "resources",
)
REPORT_NAMES = (
    "reader_setup_20261104_v1",
    "main_comparisons_20261104_v1",
    "errors_and_gate_20261104_v1",
    "answer_quality_20261104_v1",
    "sources_functions_20261104_v2",
    "retrieval_contexts_20261104_v2",
    "robustness_20261104_v2",
    "resources_20261104_v1",
)
MAIN = ROOT / "experiments/astra/sq2_broad_main_review_20261104_v1"
PRIMARY = (
    "en_test/hate",
    "larger_dev/target_group",
    "larger_dev/severity",
    "de_legal_test/legal",
)
ORDER = (
    "en_test/hate",
    "en_test/hate_type",
    "en_test/target_group",
    "en_test/severity",
    "larger_dev/target_group",
    "larger_dev/severity",
    "de_test/hate",
    "de_test/target_group",
    "de_legal_test/legal",
    "en_hatecheck_eval/hate",
    "en_hatecheck_eval/target_group",
)
LABELS = (
    "English TEST hate",
    "English TEST types",
    "English TEST targets",
    "English TEST severity",
    "English DEV targets",
    "English DEV severity",
    "German TEST hate",
    "German TEST targets",
    "German legal TEST",
    "HateCheck hate",
    "HateCheck targets",
)
NAMES = dict(zip(ORDER, LABELS))


def part(name):
    return ROOT / f"experiments/astra/sq2_broad_{name}_review_20261104_v1"


def rel(path):
    path = Path(path).resolve()
    return (
        path.relative_to(ROOT.resolve()).as_posix()
        if path.is_relative_to(ROOT.resolve())
        else path.as_posix()
    )


def bind(paths):
    return {rel(path): x.c.sha(Path(path)) for path in paths}


def read(path):
    return x.c.read(Path(path))


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        f.write(text)


def save(path, value):
    path = Path(path)
    x.require(path.resolve().is_relative_to(BASE.resolve()), "Output outside synthesis")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(x.c.encoded(x.prior.safe(value)))


def verify_map(values):
    for path, h in values.items():
        x.require(
            x.c.sha(Path(path) if Path(path).is_absolute() else ROOT / path) == h,
            "Changed bound source " + path,
        )


def freeze():
    x.require(not (BASE / "analysis_contract.json").exists(), "Scope already recorded")
    bindings = {}
    reviews = []
    seals = []
    for i, name in enumerate(PARTS, 1):
        directory = part(name)
        sp = directory / f"part{i}_complete.json"
        fp = directory / "final_review.json"
        s = read(sp)
        v = read(fp)
        x.require(
            "independently_verified" in s["status"] and v["status"].endswith("_passed"),
            "Prior part not complete",
        )
        x.require(
            s["new_Qwen_calls"] == 0 and s["new_training"] == 0,
            "Unexpected prior review generation",
        )
        for path, h in s["outputs"].items():
            x.require(
                path not in bindings or bindings[path] == h,
                "Conflicting immutable prior output",
            )
            bindings[path] = h
        seals.extend([sp, fp])
        reviews.append(
            dict(
                part=i,
                root=rel(directory),
                seal_sha256=x.c.sha(sp),
                status=s["status"],
                review=v["status"],
                outputs=len(s["outputs"]),
            )
        )
    verify_map(bindings)
    sources = (
        seals
        + [
            ROOT / "Thesis_Log/astra_result" / ("astra_sq2_broad_" + name + ".md")
            for name in REPORT_NAMES
        ]
        + [
            ROOT
            / "Thesis_Log/astra_result/astra_sq2_broad_analysis_plan_20261104_v1.md",
            ROOT
            / "Thesis_Log/astra_result/astra_remaining_work_before_writing_20261104_v151.md",
            Path(__file__),
            ROOT
            / "Thesis_Log/astra_result/astra_sq2_matched_writing_handover_20261030_v1.md",
            ROOT
            / "Thesis_Log/astra_result/astra_sq2_gpu_writing_handover_20261104_v1.md",
        ]
    )
    sources += [] + [
        writer / "sections/Thesis_Chapters" / name
        for name in (
            "Introduction.tex",
            "Results_SQ2.tex",
            "Discussion.tex",
            "Conclusion.tex",
        )
    ]
    save(
        BASE / "analysis_contract.json",
        dict(
            status="part9_integration_scope_recorded",
            created_utc=x.c.utc(),
            purpose="Standalone synthesis of Parts 1–8; close the broader experimental work item after integration checks. Combined SQ2 and final thesis synthesis remain separate.",
            prior_knowledge="All eight outcomes already reviewed; this is integration, not a new preregistration or confirmatory analysis.",
            fixed="All 11 endpoints, 40 original contrasts (38 scorable), four primary full/small questions, conditional adjusted intervals, original masks, gold, repeats and failure bounds remain.",
            verification="Check prior immutable seals; reconstruct main points and masks from saved encoded predictions; independently interpolate all existing contrast intervals; reconcile binary error and gate-aware exact/joint measures across parts; bind copied tables, new claims, figure and writing handover to original evidence. No new bootstrap draws.",
            sources=bind(sources),
            prior_output_bindings=bindings,
            prior_reviews=reviews,
            limits=[
                "No new model calls, retrieval, training, inference, server action, deletion or writer edit.",
                "No new comparison family, optimal model choice, gold changes or cross-study pooled score.",
                "No full raw-tree audit is repeated.",
                "A source-bound synthesis is not an independent expert validation of annotations or rationales.",
            ],
            deferred=[
                "Combined SQ2 synthesis and cross-study comparison alignment",
                "Global resource/journal reconciliation and final experiment-stop record",
                "Supplementary weights E: backup, whole-thesis backup, server retention/release",
                "Writer integration and compilation",
            ],
        ),
    )
    print(
        "Part 9 scope saved; all eight prior seals verified:",
        len(bindings),
        "unique immutable outputs.",
        flush=True,
    )


def blocks(path):
    lines = path.read_text(encoding="utf8").splitlines()
    out = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("|"):
            start = i
            while i < len(lines) and lines[i].startswith("|"):
                i += 1
            if i - start >= 2:
                out.append(
                    dict(
                        first_line=start + 1,
                        markdown="\n".join(lines[start:i]),
                        rows=i - start - 2,
                    )
                )
        else:
            i += 1
    return out


def consolidate():
    contract = read(BASE / "analysis_contract.json")
    verify_map(contract["sources"])
    verify_map(contract["prior_output_bindings"])
    result = {}
    paths = []
    for name in ORDER:
        p = MAIN / "endpoints" / name.replace("/", "__") / "results.json"
        v = read(p)
        paths.append(p)
        result[name] = dict(
            endpoint=name,
            eligible=len(v["endpoint"]["scorable_ids"]),
            macro_labels=v["endpoint"]["macro_labels"],
            shared={
                k: w
                for k, w in v["three_condition_common"].items()
                if k != "excluded_ids"
            },
            contrasts=[
                {
                    k: w
                    for k, w in a.items()
                    if k not in ("common_valid_ids", "excluded_ids")
                }
                for a in v["contrasts"]
            ],
        )
    save(BASE / "main_evidence.json", dict(endpoints=result, sources=bind(paths)))
    specs = [
        (
            "primary",
            "main_comparisons_20261104_v1",
            0,
            "Four primary full-minus-small questions",
            "Pair-specific six-setting masks; original 98.75% family intervals.",
        ),
        (
            "shared",
            "main_comparisons_20261104_v1",
            1,
            "Three-condition scores on identical texts",
            "All-nine common-valid masks; small-minus-zero is descriptive only.",
        ),
        (
            "full_zero",
            "main_comparisons_20261104_v1",
            2,
            "All supporting full-minus-no-retrieval comparisons",
            "Pointwise 95% intervals; no new primary family.",
        ),
        (
            "other_small",
            "main_comparisons_20261104_v1",
            3,
            "Remaining full-minus-small comparisons",
            "All scorable nonprimary endpoints, including the adverse target result.",
        ),
        (
            "encoders",
            "main_comparisons_20261104_v1",
            4,
            "All historical encoder comparisons",
            "Original aligned checkpoints; these are not the later equal-budget tuned models.",
        ),
        (
            "bounds",
            "main_comparisons_20261104_v1",
            5,
            "All original full-roster failure bounds",
            "Bounds for fixed unknown answers, not sampling confidence intervals.",
        ),
        (
            "errors",
            "errors_and_gate_20261104_v1",
            0,
            "Absolute binary errors on shared texts",
            "All-nine masks; counts and rates are repeat means.",
        ),
        (
            "error_pairs",
            "errors_and_gate_20261104_v1",
            1,
            "All paired binary error changes",
            "Each comparison retains its own six-setting population; fewer FP can coexist with more FN.",
        ),
        (
            "error_intervals",
            "errors_and_gate_20261104_v1",
            2,
            "Uncertainty for binary error changes",
            "Supporting 95% diagnostic intervals, separate from the primary F1 family.",
        ),
        (
            "exact",
            "answer_quality_20261104_v1",
            0,
            "Complete annotated fields",
            "Correct hate gate required except for independent legal labels; no all-field gold score.",
        ),
        (
            "exact_pairs",
            "answer_quality_20261104_v1",
            1,
            "Exact and joint answer differences",
            "Supporting diagnostic intervals with original paired masks.",
        ),
        (
            "legal",
            "answer_quality_20261104_v1",
            3,
            "Legal-set false flags and positive-gold agreement",
            "Legal annotations are not determinations of criminal liability.",
        ),
        (
            "joint",
            "answer_quality_20261104_v1",
            4,
            "Genuinely co-annotated joint answers",
            "MHS hate/targets/severity only; gold-hate cohorts do not measure false alarms.",
        ),
        (
            "sources",
            "sources_tables_20261104_v1",
            1,
            "All natural-source binary results",
            "Source definitions and populations differ; no language ranking.",
        ),
        (
            "contexts",
            "context_tables_20261104_v1",
            1,
            "Delivered context composition",
            "Query-dependent retrieval; example annotations and composition were not independently randomized.",
        ),
        (
            "robust_primary",
            "robustness_20261104_v2",
            0,
            "Primary robustness summary",
            "Original primary findings are retained; selected subsets do not replace them.",
        ),
        (
            "resources",
            "resources_20261104_v1",
            0,
            "Cost of the measured three-vote answer",
            "Includes repairs; reported tokens are not monetary cost or complete backend usage.",
        ),
        (
            "timers",
            "resources_tables_20261104_v1",
            3,
            "Recorded client request distributions",
            "Overlapping request-time sums, not isolated production latency.",
        ),
    ]
    tables = []
    for ident, suffix, index, title, note in specs:
        p = ROOT / "Thesis_Log/astra_result" / ("astra_sq2_broad_" + suffix + ".md")
        block = blocks(p)[index]
        tables.append(
            dict(
                id=ident,
                title=title,
                note=note,
                source=rel(p),
                source_sha256=x.c.sha(p),
                source_block=index,
                **block,
            )
        )
        paths.append(p)
    save(BASE / "consolidated_tables.json", tables)
    write(
        BASE / "tables.md",
        "# Broader SQ2 synthesis — consolidated evidence tables\n\nEvery table is reproduced from its checked source. Populations and interval families remain separate.\n\n"
        + "\n\n".join(
            (
                "## "
                + str(i + 1)
                + ". "
                + z["title"]
                + "\n\n"
                + z["markdown"]
                + "\n\n"
                + z["note"]
                + " [Original source](../../../"
                + z["source"]
                + "). "
                for i, z in enumerate(tables)
            )
        ),
    )
    save(
        BASE / "consolidation_complete.json",
        dict(
            status="evidence_and_tables_consolidated_for_integration_check",
            sources=bind([Path(__file__), BASE / "analysis_contract.json", *paths]),
            outputs=bind(
                [
                    BASE / "main_evidence.json",
                    BASE / "consolidated_tables.json",
                    BASE / "tables.md",
                ]
            ),
            tables=len(tables),
            table_rows=sum((z["rows"] for z in tables)),
        ),
    )
    print(
        "Consolidated",
        len(tables),
        "tables and",
        sum((z["rows"] for z in tables)),
        "rows; no new scoring.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["freeze", "consolidate"], required=True)
    args = parser.parse_args()
    with x.r.original.offline():
        freeze() if args.mode == "freeze" else consolidate()
