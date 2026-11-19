from __future__ import annotations
from src.hsrag.paths import PROJECT_ROOT
import contextlib
import datetime as dt
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = PROJECT_ROOT
OUT = ROOT / "experiments/astra/sq3_final_synthesis_20261027_v1"
LOG = ROOT / "Thesis_Log/astra_result"
REPORTS = {
    "historical": "astra_sq3_p3_historical_audit_20260928_v1.md",
    "fixed_feedback": "astra_sq3_feedback_final_results_20260930_v1.md",
    "relevance": "astra_sq3_relevance_final_results_20261001_v1.md",
    "explanation_selectivity": "astra_sq3_explanation_selectivity_final_results_20261004_v1.md",
    "taxonomy": "astra_sq3_taxonomy_final_results_20261004_v1.md",
    "early_family": "astra_sq3_combined_final_summary_20261004_v1.md",
    "factorial": "astra_sq3_factorial_results_20261006_v1.md",
    "dataset_meaning": "astra_sq3_human_meaning_results_20261006_v1.md",
    "candidate_screening": "astra_sq3_benign_selection_results_20261006_v1.md",
    "additions": "astra_sq3_additions_final_summary_20261006_v1.md",
    "balanced_sixth": "astra_sq3_balanced_final_results_20261010_v1.md",
    "human_annotation": "astra_sq3_human_review_results_20261010_v2.md",
    "ordinary_insertion": "astra_sq3_five_slot_results_20261012_v2.md",
    "reserved": "astra_sq3_reserved_final_results_20261020_v1.md",
    "human_feedback": "astra_sq3_human_feedback_results_20261020_v1.md",
    "classifier_dev": "astra_sq3_linear_dev_results_20261020_v1.md",
    "broader": "astra_sq3_transfer_final_results_20261023_v1.md",
    "classifier_broad": "astra_sq3_transfer_part8_classifier_20261023_v1.md",
    "successive": "astra_sq3_iterative_final_results_20261027_v1.md",
    "classifier_cumulative": "astra_sq3_cumulative_classifier_results_20261027_v1.md",
}
MACHINE = {
    "fixed": "sq3_full_kb_feedback_20260928_v1/results_v1/transfer_b04.json",
    "relevance": "sq3_full_kb_relevance_20260930_v1/results_v1/primary.json",
    "explanation": "sq3_combined_results_20261004_v1/explanations/primary_explanation.json",
    "selectivity": "sq3_combined_results_20261004_v1/explanations/primary_application.json",
    "taxonomy": "sq3_combined_results_20261004_v1/taxonomy/primary.json",
    "factorial": "sq3_additions_results_20261006_v1/factorial/primary.json",
    "meaning": "sq3_additions_results_20261006_v1/human_meaning/primary.json",
    "reserved": "sq3_reserved_part10_results_20261020_v1/display_values.json",
    "broad": "sq3_transfer_part10_20261023_v1/display_values.json",
    "iterative": "sq3_iterative_review_20261026_v1/part2/errors_v1/results.json",
    "cumulative": "sq3_cumulative_results_20261027_v1/analysis.json",
    "cumulative_resources": "sq3_cumulative_results_20261027_v1/resources.json",
    "remote_ledger": "sq3_iterative_review_20261026_v1/part8/partial_programme_ledger.json",
}
PROOFS = [
    "sq3_full_kb_feedback_20260928_v1/final_results_publication_v1/independent_verification.json",
    "sq3_full_kb_relevance_20260930_v1/results_v1/independent_verification.json",
    "sq3_combined_results_20261004_v1/independent_verification.json",
    "sq3_additions_results_20261006_v1/final_verification.json",
    "sq3_reserved_part9_results_20261020_v1/complete.json",
    "sq3_reserved_part10_results_20261020_v1/complete.json",
    "sq3_human_feedback_results_20261020_v1/complete.json",
    "sq3_human_feedback_results_20261020_v1/independent_verification.json",
    "sq3_transfer_part10_20261023_v1/complete.json",
    "sq3_iterative_review_20261026_v1/part2/errors_v1/independent_verification.json",
    "sq3_iterative_review_20261026_v1/part8/final_verification.json",
    "sq3_cumulative_results_20261027_v1/final_verification.json",
    "sq3_cumulative_results_20261027_v1/independent_verification.json",
    "sq3_cumulative_results_20261027_v1/exports_v2.json",
    "sq3_iterative_review_20261026_v1/part8/study_figure_index.json",
    "sq3_transfer_part10_20261023_v1/figure_inventory.json",
    "sq3_reserved_part10_results_20261020_v1/writer_artifacts.json",
]


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (
        value
        if isinstance(value, str)
        else json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    if p.exists():
        need(
            p.read_text(encoding="utf8") == data,
            f"Preserve non-identical existing artifact: {p}",
        )
        return
    with p.open("x", encoding="utf8", newline="\n") as f:
        f.write(data)


@contextlib.contextmanager
def offline():

    def blocked(*args, **kwargs):
        raise RuntimeError("This synthesis is offline. Network access is blocked.")

    original = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.create_connection,
    )
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = (
        blocked
    )
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = (
            original
        )


def machine(key):
    return ROOT / "experiments/astra" / MACHINE[key]


def prepare():
    need(
        not (OUT / "analysis_contract.json").exists(),
        "Preserve the existing synthesis contract.",
    )
    sources = {rel(LOG / name): sha(LOG / name) for name in REPORTS.values()}
    for path in [machine(k) for k in MACHINE] + [
        ROOT / "experiments/astra" / p for p in PROOFS
    ]:
        need(path.is_file(), f"Missing authoritative source: {path}")
        sources[rel(path)] = sha(path)
    save(
        "analysis_contract.json",
        {
            "status": "frozen_before_combined_extraction",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "request": "Final combined SQ3 synthesis, log findings and prepare the writing handover.",
            "scope": "Interpretation and exact transcription of completed studies; no new scientific estimates or inferential family.",
            "preserve": [
                "all raw outputs, protocols, original labels, models, reports and audit history",
                "writer-owned files and acknowledgements",
                "six completed / three open SQ2 experiment items and all sixteen inventory entries",
            ],
            "prohibited": [
                "Qwen calls including availability GETs",
                "training",
                "embeddings",
                "new annotation",
                "TEST-driven tuning",
                "new bootstrap tests",
                "cross-study pooled effect",
                "writer-owned edits or messages",
            ],
            "verification": "Fresh source hashes, direct machine-value transcription, rational headline arithmetic, interval roles, resource attribution, status and links. Reuse prior independent numerical proofs; do not claim a new universal raw-response audit.",
            "rounding_decision": "English broad FPR uses exact 2374/74619 = 3.181495... pp: +3.181 at three decimals, +3.18 at two. Earlier +3.182 prose is a display error only; preserve those historical documents.",
            "sources": sources,
            "no_claim_of_writer_integration": True,
        },
    )
    print(f"Frozen synthesis contract: {len(sources)} evidence files.")


def value_at(value, pointer):
    for k in pointer:
        value = value[k]
    return value


def build():
    contract = read(OUT / "analysis_contract.json")
    for p, expected in contract["sources"].items():
        need(
            sha(ROOT / p) == expected,
            f"Source changed since synthesis preparation: {p}",
        )
    records = []
    evidence = []

    def get(key, pointer):
        path = machine(key)
        v = value_at(read(path), pointer)
        evidence.append(
            {"source": rel(path), "sha256": sha(path), "pointer": pointer, "value": v}
        )
        return v

    def add(
        code,
        study,
        endpoint,
        contrast,
        unit,
        confidence,
        n,
        point,
        interval,
        role="prespecified primary",
        **extra,
    ):
        records.append(
            dict(
                id=code,
                study=study,
                endpoint=endpoint,
                contrast=contrast,
                unit=unit,
                confidence=confidence,
                n=n,
                point=point,
                interval=interval,
                role=role,
                **extra,
            )
        )

    f = ["contrasts", "error_minus_correct"]
    add(
        "fixed_40",
        "Fixed full-KB correction pools",
        "seven-type macro F1",
        "mistake-selected − already-correct",
        "F1 (0–1)",
        0.95,
        get("fixed", ["common_valid_n"]),
        get("fixed", f + ["mean_difference"]),
        get("fixed", f + ["macro_f1_ci95"]),
    )
    for key, contrast, title in [
        (
            "relevance",
            "relevant_full_minus_natural",
            "Gold-assisted correction relevance",
        ),
        (
            "explanation",
            "specific_always_minus_plain_always",
            "Source-specific assistant explanation",
        ),
        (
            "selectivity",
            "plain_selective_minus_plain_always",
            "Fixed selective application",
        ),
    ]:
        pre = ["contrasts", contrast]
        add(
            key,
            title,
            "seven-type macro F1",
            contrast.replace("_", " "),
            "F1 (0–1)",
            0.9875,
            get(key, ["common_valid_n"]),
            get(key, pre + ["difference"]),
            get(key, pre + ["ci9875"]),
            qualification=(
                "Gold-assisted relevance is diagnostic, not a deployable selector."
                if key == "relevance"
                else "Conditional on exposed DEV and frozen policy."
            ),
        )
    add(
        "taxonomy",
        "Familiar-category restoration",
        "mean restored-target F1",
        "full package − schema-only",
        "F1 (0–1)",
        0.9875,
        get("taxonomy", ["common_valid_n"]),
        get("taxonomy", ["primary", "difference"]),
        get("taxonomy", ["primary", "ci9875"]),
        reference=get("taxonomy", ["primary", "schema_only_target_f1"]),
        updated=get("taxonomy", ["primary", "full_target_f1"]),
        qualification="Average across seven separate category contrasts. Full condition is the existing full KB, not an improved model above it.",
    )
    for term in ["example", "definition"]:
        pre = ["effects", f"marginal_{term}_block_effect"]
        add(
            "factorial_" + term,
            "Examples × definitions supplement",
            "mean restored-target F1",
            term + " block main effect",
            "F1 (0–1)",
            0.975,
            get("factorial", ["common_n"]),
            get("factorial", pre + ["difference"]),
            get("factorial", pre + ["family_interval"]),
            clusters=get("factorial", ["cluster_count"]),
            qualification="193 selected input/category pairs in 155 text clusters, not 193 independent texts.",
        )
    add(
        "dataset_meaning",
        "Existing dataset human implied meanings",
        "seven-type macro F1",
        "human meaning − plain correction",
        "F1 (0–1)",
        0.95,
        get("meaning", ["common_n"]),
        get("meaning", ["contrasts", "human_minus_plain", "difference"]),
        get("meaning", ["contrasts", "human_minus_plain", "ci95"]),
    )
    for i in range(2):
        r = get("reserved", ["primary", i])
        add(
            "reserved_" + r["endpoint"],
            "Reserved correction, exposed DEV",
            r["endpoint"],
            "4 ordinary + correction − 5 ordinary",
            "rate (0–1)",
            0.975,
            r["common_n"],
            float(Fraction(r["source"]["difference_exact"])),
            [float(Fraction(x)) for x in r["source"]["interval_exact"]],
            reference=float(Fraction(r["source"]["rates"][1])),
            updated=float(Fraction(r["source"]["rates"][0])),
            exact_difference=r["source"]["difference_exact"],
        )
    for i in range(4):
        r = get("broad", ["primary", i])
        need(
            abs(float(Fraction(r["difference_exact"])) - r["difference"]) < 1e-15,
            "Broad rational point mismatch",
        )
        add(
            "broad_" + r["endpoint"],
            "Broader fixed-policy transfer",
            r["endpoint"],
            "4 ordinary + correction − 5 ordinary",
            "rate (0–1)",
            0.9875,
            r["common_n"],
            r["difference"],
            r["interval98_75"],
            clusters=r["clusters"],
            reference=r["reference"],
            updated=r["feedback"],
            exact_difference=r["difference_exact"],
        )
        z = get("broad", ["classifier", i])
        add(
            "static_" + z["endpoint"],
            "Same-feedback static classifier",
            z["endpoint"],
            "(linear update − linear base) − (Qwen feedback − Qwen reference)",
            "rate (0–1)",
            0.9875,
            z["common_n"],
            z["estimates"]["difference_in_change"],
            z["primary_interval98_75"],
            estimates=z["estimates"],
        )
    original = read(machine("iterative"))["records"]
    for key in [k for k in original if k.startswith("primary_")]:
        r = get("iterative", ["records", key])
        column = r["column_order"][0]
        need(
            abs(
                float(
                    Fraction(
                        r["integer_error_numerators"][column], r["integer_denominator"]
                    )
                )
                - r["points"][column]
            )
            < 1e-15,
            "Iterative rational point mismatch",
        )
        add(
            key,
            "Four successive rounds",
            r["endpoint"],
            column.replace("r4_", "").replace("-", " − "),
            "rate (0–1)",
            r["interval_level"],
            r["inputs"],
            r["points"][column],
            r["intervals"][column],
            clusters=r["inference_clusters"],
        )
    c = read(machine("cumulative"))
    need(
        c["settings"][0] == ["all_permitted", 0.6] and c["policies"][0] == "adaptive",
        "Cumulative recipe order changed",
    )
    for idx, g in enumerate(c["groups"]):
        if g["cohort"] != "round_4":
            continue
        p = get("cumulative", ["groups", idx, "point", 0, 0, 0])
        ci = get("cumulative", ["groups", idx, "ci95", 0, 0, 0])
        estimates = dict(zip(c["metrics"], p))
        intervals = dict(zip(c["metrics"], ci))
        add(
            "cumulative_" + g["population"] + "_" + g["endpoint"],
            "Same-feedback cumulative classifier, round 4",
            g["population"] + "_" + g["endpoint"],
            "(linear update − linear base) − (Qwen feedback − Qwen reference)",
            "rate (0–1)",
            0.95,
            get("cumulative", ["groups", idx, "inputs"]),
            estimates["difference_in_change"],
            intervals["difference_in_change"],
            role="supporting, nominal 95%; no new primary family",
            clusters=get("cumulative", ["groups", idx, "clusters"]),
            estimates=estimates,
            metric_intervals=intervals,
        )
    save(
        "display_values.json",
        {
            "records": records,
            "source_extractions": evidence,
            "new_scientific_estimates": 0,
            "cross_study_pooling": False,
        },
    )
    lines = [
        "# Exact values for the final SQ3 synthesis",
        "",
        "These are transcribed from completed reviews. Different rows have different comparators, cohorts and prespecified interval families. They are not a meta-analysis. F1 changes use the 0–1 scale; error changes below use percentage points (100 × rate difference).",
        "",
        "| ID / study | Endpoint and contrast | N | Difference | Interval | Confidence / role |",
        "|---|---|---:|---:|---|---|",
    ]
    for r in records:
        scale = 100 if r["unit"] == "rate (0–1)" else 1
        digits = 3 if scale == 100 else 6
        suffix = " pp" if scale == 100 else ""
        r["display_point"] = f"{r['point'] * scale:+.{digits}f}{suffix}"
        r["display_interval"] = (
            "["
            + ", ".join((f"{x * scale:+.{digits}f}" for x in r["interval"]))
            + "]"
            + suffix
        )
        lines.append(
            f"| {r['id']} / {r['study']} | {r['endpoint']}; {r['contrast']} | {r['n']} | {r['display_point']} | {r['display_interval']} | {r['confidence'] * 100:g}%; {r['role']} |"
        )
    save("display_tables.md", "\n".join(lines) + "\n")
    save("display_rendering.json", {"records": records})
    old = get("remote_ledger", [])
    need(len(old["programmes"]) == 13, "Remote ledger scope changed")
    totals = {k: sum((p[k] for p in old["programmes"])) for k in old["totals"]}
    need(totals == old["totals"], "Remote ledger arithmetic mismatch")
    sq3rows = [p for p in old["programmes"] if p["programme"].startswith("SQ3")]
    need(len(sq3rows) == 8, "Unexpected SQ3 programme attribution")
    local = get("cumulative_resources", [])
    historical_tokens = {"fixed_feedback": 107971589, "relevance": 49986185}
    for key, n in historical_tokens.items():
        source_text = (LOG / REPORTS[key]).read_text(encoding="utf8")
        if key == "relevance":
            need(
                "42,760,950 prompt tokens" in source_text
                and "7,225,235 completion tokens" in source_text
                and (n == 42760950 + 7225235),
                "Earlier relevance usage sum mismatch",
            )
        else:
            need(f"{n:,}" in source_text, "Earlier token total not in source report")
    resource = {
        "status": "SQ3 resource reconciliation; inherited thirteen remote rows unchanged",
        "source": rel(machine("remote_ledger")),
        "source_sha256": sha(machine("remote_ledger")),
        "programmes": old["programmes"],
        "inherited_thirteen_programme_totals": totals,
        "included_sq3_programmes": sq3rows,
        "included_sq3_reported_tokens": sum((p["reported_tokens"] for p in sq3rows)),
        "two_additional_previously_omitted_sq3_programmes_tokens": historical_tokens,
        "ten_named_sq3_programmes_reported_tokens": sum(
            (p["reported_tokens"] for p in sq3rows)
        )
        + sum(historical_tokens.values()),
        "historical_resource_source_scope": "The two earlier token totals are report-bound; their full call-attempt ledgers are not silently appended to the thirteen-programme machine ledger. They are separate fresh campaigns, not nested taxonomy/screening rows.",
        "new_fits_this_synthesis": 0,
        "static_fits_previously_run": 38,
        "cumulative_new_fits_previously_run": local["new_fits"],
        "distinct_classifier_fits": local[
            "all_distinct_classifier_fits_static_plus_cumulative"
        ],
        "cumulative_resources": local,
        "human_judgments": 96,
        "participant_minutes_reported": 100,
        "new_human_judgments": 0,
        "source_candidate_labels": 1024,
        "source_candidate_label_status": "Inherited labels, reused across later studies; successive screening still incurs fresh calls.",
        "excluded_capacity_probe_reported_tokens": 12073826,
        "limitations": [
            "not whole-thesis or complete historical SQ3 cost",
            "not a bill, energy or GPU-hour estimate",
            "unknown remote work behind failed transports not fabricated as zero",
            "taxonomy nested inside combined programme",
            "older source screening nested inside additions",
            "2 reused classifier baselines and overlapping aliases are not new fits",
            "representation, acquisition, preparation and author labour incompletely measured",
        ],
    }
    need(
        resource["ten_named_sq3_programmes_reported_tokens"] == 2363308140,
        "SQ3 token boundary differs",
    )
    need(
        resource["distinct_classifier_fits"] == 102 and local["new_fits"] == 64,
        "Local reuse accounting changed",
    )
    save("resource_reconciliation.json", resource)
    exact = Fraction(2374, 74619) * 100
    save(
        "reporting_corrections.json",
        {
            "broader_English_FPR": {
                "source": rel(machine("broad")),
                "pointer": ["primary", 0, "difference_exact"],
                "exact_fraction_pp": str(exact),
                "unrounded_pp": float(exact),
                "correct_3dp": f"{float(exact):+.3f}",
                "correct_2dp": f"{float(exact):+.2f}",
                "historical_inconsistent_prose": "+3.182 pp",
                "cause": "Rounding/transcription inconsistency in later prose; the exact stored estimate and original table are unchanged.",
                "action": "Use +3.181 pp (or +3.18 pp) in final writing; preserve old reports.",
                "scientific_conclusion_changed": False,
            },
            "resource_status": "Resource v8 retained a pre-cumulative pending sentence. v9 records the already-completed 64 fits without new remote usage.",
            "historical_pending_text": "Dated reports remain historical snapshots. Final synthesis releases the writing hold; it is not a writer acknowledgement.",
        },
    )
    save(
        "build_complete.json",
        {
            "status": "combined_transcription_and_accounting_complete",
            "records": len(records),
            "source_extractions": len(evidence),
            "source_contract_sha256": sha(OUT / "analysis_contract.json"),
            "new_calls": 0,
            "new_fits": 0,
            "new_bootstraps": 0,
            "outputs": {rel(p): sha(p) for p in OUT.glob("*") if p.is_file()},
        },
    )
    print(
        f"Combined transcription: {len(records)} rows; {len(evidence)} direct source extractions. No new estimates."
    )


def verify():
    contract = read(OUT / "analysis_contract.json")
    for p, s in contract["sources"].items():
        need(sha(ROOT / p) == s, f"Preserved evidence changed: {p}")
    v = read(OUT / "display_values.json")
    checks = 0
    for e in v["source_extractions"]:
        current = read(ROOT / e["source"])
        for part in e["pointer"]:
            current = current[part]
        need(
            current == e["value"],
            f"Transcription mismatch: {e['source']} {e['pointer']}",
        )
        checks += 1
    broad = next((r for r in v["records"] if r["id"] == "broad_en_test_FPR"))
    need(
        Fraction(broad["exact_difference"])
        == Fraction(29050, 74619) - Fraction(8892, 24873),
        "Broad FPR numerator/denominator mismatch",
    )
    need(
        f"{100 * broad['point']:+.3f}" + " pp" == "+3.181 pp",
        "Broader FPR rounding failed",
    )
    it = [r for r in v["records"] if r["id"].startswith("primary_")]
    need(
        len(it) == 4
        and all(
            (
                r["confidence"] == 0.9875 and r["interval"][0] <= 0 <= r["interval"][1]
                for r in it
            )
        ),
        "Original iterative primary role/conclusion changed",
    )
    cu = [r for r in v["records"] if r["id"].startswith("cumulative_")]
    need(
        len(cu) == 4
        and all(
            (r["confidence"] == 0.95 and r["role"].startswith("supporting") for r in cu)
        ),
        "Cumulative inference promoted",
    )
    need(
        all(
            (
                r["estimates"]["linear_updated"] > r["estimates"]["qwen_updated"]
                for r in cu
                if r["endpoint"].endswith("FNR")
            )
        ),
        "Absolute missed-hate boundary lost",
    )
    data = read(OUT / "resource_reconciliation.json")
    prior = read(machine("remote_ledger"))
    need(
        data["programmes"] == prior["programmes"], "Existing programme objects changed"
    )
    need(
        data["included_sq3_reported_tokens"] == 2205350366, "Included subtotal mismatch"
    )
    save(
        "independent_readback.json",
        {
            "status": "pass",
            "kind": "Direct JSON-path readback, rational headline arithmetic, interval-role and resource reconciliation; not a new raw-response or bootstrap audit.",
            "direct_extractions_checked": checks,
            "preserved_source_files": len(contract["sources"]),
            "primary_rows": len(it),
            "cumulative_supporting_rows": len(cu),
            "source_files_unchanged": True,
            "rounding_correction_verified": True,
            "thirteen_programme_objects_unchanged": True,
            "new_calls": 0,
            "new_fits": 0,
            "new_inferential_tests": 0,
        },
    )
    print(
        "Direct source readback, rational arithmetic, interval roles and resource accounting pass."
    )


if __name__ == "__main__":
    with offline():
        {"prepare": prepare, "build": build, "verify": verify}[sys.argv[1]]()
