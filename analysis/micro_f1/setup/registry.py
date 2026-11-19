from analysis.common import micro_f1_files as c
from datetime import datetime, timezone
import re

B = c.ROOT / "experiments/astra"
R = c.ROOT / "Thesis_Log/astra_result"
MULTI = {"hate_type", "target_group", "legal"}
STUDIES, UNITS, FAMILIES = ([], [], {})
TYPE = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
TARGET = [
    "race",
    "religion",
    "gender",
    "sexual_orientation",
    "disability",
    "national_origin",
    "other",
]
LEGAL = ["propaganda", "incitement_public_peace", "insult_defamation"]


def study(key, name, parts, report, location, note=""):
    path = R / report
    c.bind(path)
    value = dict(
        id=key,
        name=name,
        parts=parts,
        report=c.rel(path),
        thesis_location="sections/Thesis_Chapters/" + location,
        boundary=note,
    )
    c.require((c.THESIS / value["thesis_location"]).exists(), "Missing thesis location")
    STUDIES.append(value)


def unit(
    study_id, key, path, pointer, endpoint, conditions, contrasts, reader, extra=None
):
    path = c.bind(path)
    dimension = endpoint.get("dimension", key.rsplit("/", 1)[-1])
    labels = endpoint.get("all_labels", endpoint.get("labels", []))
    value = dict(
        id=study_id + "/" + key,
        study=study_id,
        endpoint=key,
        dimension=dimension,
        source=c.rel(path),
        source_sha256=c.sha(path),
        json_pointer=pointer,
        reader=reader,
        labels=labels,
        macro_labels=endpoint.get("macro_labels", []),
        gold_support=endpoint.get("gold_support"),
        original_mask=endpoint.get("mask"),
        original_rows=endpoint.get("rows", endpoint.get("scorable_rows")),
        scorable_rows=endpoint.get(
            "scorable_rows", len(endpoint.get("scorable_ids", []))
        ),
        scorable_ids=endpoint.get("scorable_ids"),
        conditions=conditions,
        original_contrasts=contrasts,
    )
    if extra:
        value.update(extra)
    UNITS.append(value)
    return value


def family(name, members, rationale):
    c.require(members and len(members) == len(set(members)), "Bad family " + name)
    FAMILIES[name] = dict(
        m=len(members),
        confidence=1 - 0.05 / len(members),
        members=members,
        rationale=rationale,
        role="new_post_hoc_supporting_micro_family",
        original_macro_family_changed=False,
    )


def legacy_units(key, paths):
    main_members = []
    for path in paths:
        data = c.read(path)
        endpoint_data = data.get("endpoints") or {data["endpoint"]["subset"]: data}
        for subset, result in endpoint_data.items():
            ep = result["endpoint"]
            if ep["dimension"] not in MULTI:
                continue
            views = result.get("paired_comparison")
            if views is None:
                views = {"paired": {"contrasts": result.get("contrasts", {})}}
            elif "contrasts" in views:
                views = {"paired": views}
            conditions = sorted(result["conditions"])
            contrasts = []
            for view, record in views.items():
                for name, old in record["contrasts"].items():
                    cid = subset + "|" + view + "|" + name
                    original_primary = old.get("family_confidence") is not None
                    is_main = original_primary and subset != "en_dev_eval_types"
                    if key == "core" and subset != "en_dev_eval_types":
                        is_main = name in {
                            "rag_full_minus_rag_small",
                            "rag_full_minus_zero_shot",
                            "rag_full_minus_eligible_fewshot_mean",
                        }
                    contrasts.append(
                        dict(
                            id=cid,
                            view=view,
                            name=name,
                            original_role=(
                                "adjusted" if original_primary else "supporting"
                            ),
                            new_micro_role="family" if is_main else "descriptive",
                            weights=old.get("weights"),
                            n=old.get("n_common_valid", record.get("n_common_valid")),
                            original_seed=old.get(
                                "bootstrap_seed", record.get("bootstrap_seed")
                            ),
                            original_draws=old.get(
                                "bootstrap_draws", record.get("bootstrap_draws")
                            ),
                            original_common_ids=old.get(
                                "common_valid_ids", record.get("common_valid_ids")
                            ),
                        )
                    )
                    if is_main:
                        main_members.append(key + "/" + cid)
            unit(
                key,
                subset,
                path,
                "/endpoints/" + subset if "endpoints" in data else "",
                ep,
                conditions,
                contrasts,
                "legacy_assembled_parquet",
                {
                    "sensitivity_only": subset == "en_dev_eval_types",
                    "resampling": "original paired individual-text bootstrap",
                    "averaging": "repeat mean within each fixed context/pool, then equal pool mean; shared reference once",
                    "assembled": {
                        p: h
                        for p, h in data.get("assembled", {}).items()
                        if "endpoints" not in data or "/" + subset + "/" in p
                    },
                    "gold_path": "data/processed/" + subset + ".parquet",
                    "original_views": {
                        v: {
                            k: x
                            for k, x in rec.items()
                            if k not in ("contrasts", "conditions")
                        }
                        for v, rec in views.items()
                    },
                },
            )
    if main_members:
        family(
            key,
            main_members,
            "Multilabel members of the original adjusted family; core also includes the central type/target retrieval questions.",
        )


def generic_cells(key, path, stat_path=None):
    data = c.read(path)
    stats = c.read(stat_path) if stat_path else None
    cells = data.get("cells", data.get("endpoints", {}))
    for name, record in cells.items():
        ep = record.get("endpoint", {})
        dimension = ep.get("dimension", name.rsplit("/", 1)[-1])
        if dimension not in MULTI:
            continue
        if not ep:
            ep = dict(
                dimension=dimension,
                all_labels=(
                    TYPE
                    if dimension == "hate_type"
                    else TARGET if dimension == "target_group" else LEGAL
                ),
                macro_labels=record.get("macro_labels", []),
                gold_support=record.get("gold_support"),
            )
        unit(
            key,
            name,
            path,
            "/cells/" + name,
            ep,
            sorted(
                record.get("runs", record.get("means", record.get("group_means", {})))
            ),
            [],
            "saved_setting_counts",
            {
                "scorable_rows": record.get("n", record.get("rows")),
                "statistics_source": c.rel(stat_path) if stat_path else None,
                "supporting_scope": "All saved per-setting policies for this endpoint, preserving original aggregation and primary versus supporting roles.",
            },
        )
    return stats


def append_source_unit(key, name, path, dimension, labels, extra=None):
    data = c.read(path)
    unit(
        key,
        name,
        path,
        "",
        dict(
            dimension=dimension,
            all_labels=labels,
            macro_labels=data.get("labels", labels),
        ),
        sorted(data.get("groups", data.get("conditions", {}))),
        [],
        "saved_setting_counts",
        extra,
    )
    return data


def build():
    study(
        "core",
        "Original aligned retrieval and fixed examples",
        [2, 5],
        "astra_core_results_summary_20260920_v1.md",
        "Experiments/01_full_versus_compact_knowledge.tex",
        "Earlier DEV responses; no merging with the later broad study.",
    )
    legacy_units(
        "core",
        [
            B / "full_kb_core_20260915_v1/primary_results_v1/results.json",
            B
            / "full_kb_core_20260915_v1/secondary_results_v4/en_dev_eval_sq3_types/results.json",
            B
            / "full_kb_core_20260915_v1/secondary_results_v2/en_dev_eval_targets/results.json",
            B
            / "full_kb_core_20260915_v1/secondary_results_v4/en_dev_eval_types/results.json",
        ],
    )
    configs = [
        (
            "strategy",
            "Retrieval strategies",
            5,
            "astra_sq1_strategy_final_results_20260921_v1.md",
            "Experiments/02_retrieval_strategies.tex",
            "sq1_full_kb_strategy_20260920_v1",
            ["strategy_results_v2", "strategy_results_v1"],
        ),
        (
            "components",
            "Examples definitions and guidelines",
            6,
            "astra_sq1_components_final_summary_20260924_v1.md",
            "Experiments/03_examples_definitions_and_guidelines.tex",
            "sq1_full_kb_components_20260921_v1",
            ["component_results_v1"],
        ),
        (
            "allocation",
            "Definition allocation",
            6,
            "astra_sq1_allocation_final_summary_20260925_v1.md",
            "Experiments/04_definition_coverage_and_deliberate_guideline_delivery.tex",
            "sq1_full_kb_allocation_20260924_v1",
            ["allocation_results_v1"],
        ),
        (
            "pools",
            "Knowledge bank composition",
            5,
            "astra_sq1_pool_final_summary_20260928_v1.md",
            "Experiments/05_pool_selection_and_supplied_annotations.tex",
            "sq1_pool_size_composition_20260926_v1",
            ["pool_results_v1"],
        ),
    ]
    for key, name, part, report, location, root, versions in configs:
        study(key, name, [part], report, location)
        chosen = {}
        for version in versions:
            for path in sorted((B / root / version).glob("*/results.json")):
                chosen.setdefault(path.parent.name, path)
        legacy_units(key, list(chosen.values()))
    study(
        "broad",
        "Broader full compact and no retrieval",
        [2],
        "astra_sq2_broad_final_synthesis_20261104_v2.md",
        "Results_SQ2/01_retrieval_versus_no_retrieval.tex",
    )
    members = []
    for path in sorted(
        (B / "sq2_broad_main_review_20261104_v1/endpoints").glob("*/results.json")
    ):
        data = c.read(path)
        ep = data["endpoint"]
        if ep["dimension"] not in MULTI:
            continue
        name = ep.get("id", path.parent.name.replace("__", "/"))
        contrasts = []
        for i, x in enumerate(data["contrasts"]):
            main = (
                x["left"] == "rag_full"
                and x["right"] in ("rag_small", "zero_shot")
                and bool(ep["macro_labels"])
            )
            cid = name + "|" + x["left"] + "-" + x["right"]
            contrasts.append(
                dict(
                    id=cid,
                    pointer="/contrasts/" + str(i),
                    left=x["left"],
                    right=x["right"],
                    new_micro_role="family" if main else "descriptive",
                    original_common_ids=x.get("common_valid_ids"),
                    original_seed=x.get("seed"),
                    original_draws=x.get("draws"),
                    original_interval_role=x.get("role"),
                )
            )
            if main:
                members.append("broad/" + cid)
        if ep["macro_labels"]:
            contrasts.append(
                dict(
                    id=name + "|rag_small-zero_shot",
                    new_micro_role="descriptive",
                    population="original three-condition common-valid mask",
                    note="Existing descriptive compact/no comparison; no original paired CI to borrow.",
                )
            )
        unit(
            "broad",
            name,
            path,
            "",
            ep,
            data["settings"],
            contrasts,
            "broad_encoded",
            {
                "encoded_path": data["encoded_path"],
                "encoded_sha256": data["encoded_sha256"],
                "shared_table_mask": "all nine Qwen settings valid; native classifiers use the same rows",
                "resampling": "original normalized-text clusters; HateCheck connected functional clusters",
                "averaging": "three outer-repeat F1 values, then mean",
                "sparse_diagnostic_only": not bool(ep["macro_labels"]),
            },
        )
    family(
        "broad",
        members,
        "Two retrieval contrasts on each of the five macro-supported multilabel populations. Sparse German targets remain descriptive.",
    )
    study(
        "gpu",
        "Equal total text budget classifiers",
        [3],
        "astra_sq2_gpu_final_synthesis_20261104_v1.md",
        "Results_SQ2/03_comparison_with_stronger_supervised_procedures.tex",
    )
    root = B / "sq2_gpu_main_review_20261103_v1/runs/9deace7a4235"
    primary = c.read(root / "primary_comparisons.json")["comparisons"]
    members = []
    for stem in ("en_test__hate_type", "larger_dev__target_group"):
        meta = c.read(root / (stem + "_metadata.json"))
        path = root / (stem + "_results.json")
        result = c.read(path)
        name = meta["endpoint"]
        contrasts = []
        for x in primary:
            if x["endpoint"] != name:
                continue
            cid = f"{name}|{x['budget']}|{x['contrast']}"
            members.append("gpu/" + cid)
            contrasts.append(
                dict(
                    id=cid,
                    budget=x["budget"],
                    name=x["contrast"],
                    new_micro_role="family",
                )
            )
        unit(
            "gpu",
            name,
            path,
            "",
            dict(meta, all_labels=meta["labels"], scorable_ids=meta["ids"]),
            meta["models"],
            contrasts,
            "gpu_arrays",
            {
                "metadata_path": c.rel(root / (stem + "_metadata.json")),
                "prediction_path": c.rel(root / (stem + "_predictions.npz")),
                "count_path": c.rel(root / (stem + "_point_counts.npz")),
                "resampling": "original normalized-text clusters, conditional on fixed fits and pools",
                "averaging": "five final neural seed scores within pool, then three equally weighted pools; one fold-42 linear fit per pool",
                "original_family_size": 24,
            },
        )
    family(
        "gpu",
        members,
        "The twelve type/target members of the original 24-contrast family; other original heads remain in their own metrics.",
    )
    study(
        "gpu_controls",
        "Training recipe and cutoff controls",
        [3],
        "astra_sq2_gpu_final_synthesis_20261104_v1.md",
        "Results_SQ2/03_comparison_with_stronger_supervised_procedures.tex",
        "Supporting controls; no re-selection under micro.",
    )
    root = B / "sq2_gpu_controls_review_20261103_v1/runs/b32f91704667"
    for stem in ("en_test__hate_type", "larger_dev__target_group"):
        meta = c.read(root / (stem + "_metadata.json"))
        unit(
            "gpu_controls",
            stem.replace("__", "/"),
            root / (stem + "_results.json"),
            "",
            dict(meta, all_labels=meta.get("labels", [])),
            meta.get("models", []),
            [],
            "gpu_control_arrays",
            {
                "metadata_path": c.rel(root / (stem + "_metadata.json")),
                "prediction_path": c.rel(root / (stem + "_arrays.npz")),
                "new_micro_role": "descriptive",
                "scope": "All original fixed/tuned and selected/default policies and declared controls in the bound source.",
            },
        )
    study(
        "five_fit_update",
        "Separate five-fit update sensitivity",
        [3],
        "astra_sq2_type_update_results_20261103_v1.md",
        "Results_SQ2/03_comparison_with_stronger_supervised_procedures.tex",
        "Never replaces the original initialization-only fits.",
    )
    append_source_unit(
        "five_fit_update",
        "en_test/hate_type",
        B / "sq2_type_sensitivity_20261103_v1/runs/2b1cee266dcb/main_comparison.json",
        "hate_type",
        TYPE,
        {
            "new_micro_role": "sensitivity",
            "metadata_path": "experiments/astra/sq2_type_sensitivity_20261103_v1/runs/2b1cee266dcb/primary_metadata.json",
            "prediction_path": "experiments/astra/sq2_type_sensitivity_20261103_v1/runs/2b1cee266dcb/primary_paired.npz",
        },
    )
    historical = [
        (
            "compact_tuning",
            "Original curated-pool training selection",
            "roberta_tuning_review_20261011_v1",
            "astra_sq2_roberta_tuning_results_20261011_v1.md",
        ),
        (
            "shorter",
            "Shorter training schedules",
            "roberta_short_review_20261020_v1",
            "astra_sq2_short_training_results_20261020_v1.md",
        ),
        (
            "matched_type",
            "Matched type-annotation pools",
            "matched_types_review_20261012_v1",
            "astra_sq2_matched_type_results_20261012_v1.md",
        ),
        (
            "native_cpu",
            "Native full-data BGE TF-IDF and neural references",
            "sq2_cpu_review_20261012_v1",
            "astra_sq2_cpu_results_20261012_v1.md",
        ),
    ]
    for key, name, folder, report in historical:
        study(
            key,
            name,
            [3],
            report,
            "Results_SQ2/03_comparison_with_stronger_supervised_procedures.tex",
        )
        stats = generic_cells(
            key, B / folder / "scores.json", B / folder / "statistics.json"
        )
        if key == "matched_type":
            members = [
                f"{key}/en_test/hate_type|{g}|{contrast}"
                for g, r in stats["groups"].items()
                for contrast in r["primary"]
            ]
        else:
            members = [
                key + "/" + ep
                for ep in stats.get("primary_family", stats.get("family", []))
                if ep.rsplit("/", 1)[-1] in MULTI
            ]
        family(
            key,
            members,
            "Multilabel members of the study's original primary comparison list; other policies and population views remain supporting.",
        )
    study(
        "combined_context",
        "Common-population Qwen and classifier display",
        [2, 3],
        "astra_sq2_combined_performance_errors_20261104_v2.md",
        "Results_SQ2/03_comparison_with_stronger_supervised_procedures.tex",
        "Descriptive evaluation alignment only; not an equal-training-budget comparison.",
    )
    for folder in sorted(
        (B / "sq2_combined_performance_20261104_v1/endpoints").iterdir()
    ):
        dim = folder.name.split("__")[-1]
        if dim not in MULTI:
            continue
        files = sorted((p for p in folder.iterdir() if p.suffix in {".json", ".npz"}))
        for p in files:
            c.bind(p)
        unit(
            "combined_context",
            folder.name.replace("__", "/"),
            files[0],
            "",
            dict(
                dimension=dim,
                all_labels=(
                    TYPE
                    if dim == "hate_type"
                    else TARGET if dim == "target_group" else LEGAL
                ),
            ),
            [],
            [],
            "combined_aligned_arrays",
            {
                "files": [c.rel(p) for p in files],
                "new_micro_role": "descriptive",
                "family": "none; reuse original within-study contrasts separately",
            },
        )
    study(
        "matched",
        "Example information controls",
        [4],
        "astra_sq2_matched_results_20261030_v1.md",
        "Results_SQ2/02_what_information_in_examples_helps.tex",
    )
    path = B / "sq2_matched_review_20261030_v1/metrics.json"
    data = c.read(path)
    members = []
    for name, record in data["results"].items():
        ep = record["endpoint"]
        if ep["dimension"] not in MULTI:
            continue
        contrasts = []
        for contrast in record["contrasts"]:
            main = name in (
                "en_dev_eval_sq3_types/hate_type",
                "de_legal_dev_eval/legal",
            ) and contrast in ("coverage_matched_", "permuted_")
            cid = name + "|" + contrast
            contrasts.append(
                dict(
                    id=cid,
                    name=contrast,
                    new_micro_role="family" if main else "descriptive",
                )
            )
            if main:
                members.append("matched/" + cid)
        unit(
            "matched",
            name,
            path,
            "/results/" + name,
            ep,
            sorted(record["per_condition_repeat"]),
            contrasts,
            "matched_jsonl",
            {
                "gold_path": "experiments/astra/sq2_matched_review_20261030_v1/gold_rosters.jsonl",
                "prediction_path": "experiments/astra/sq2_matched_review_20261030_v1/predictions.jsonl",
                "resampling": "original text clusters and separate source-stratified sensitivity",
                "sparse_diagnostic_only": not bool(ep["macro_labels"]),
            },
        )
    family(
        "matched",
        members,
        "Four type/legal members of the original six-question primary family.",
    )
    study(
        "german_guidelines",
        "German guideline legal allocation",
        [6],
        "german_guidelines_writing_defence_handover_20261113_v1.md",
        "Experiments/04_definition_coverage_and_deliberate_guideline_delivery.tex",
    )
    root = (
        B / "german_guidelines_legal_analysis_20261113_v1/runs/run_1794526174997253700"
    )
    members = []
    for name in ("de_legal_test", "de_legal_dev_eval"):
        path = root / name / "results.json"
        data = c.read(path)
        contrasts = []
        for i, x in enumerate(data["contrasts"]):
            cid = f"{name}/legal|pair{i + 1}"
            contrasts.append(
                dict(
                    id=cid,
                    pointer="/contrasts/" + str(i),
                    original_descriptor={
                        k: v
                        for k, v in x.items()
                        if k
                        in (
                            "id",
                            "name",
                            "left",
                            "right",
                            "contrast",
                            "treatment",
                            "control",
                        )
                    },
                    new_micro_role=(
                        "family" if name == "de_legal_test" else "descriptive"
                    ),
                )
            )
            if name == "de_legal_test":
                members.append("german_guidelines/" + cid)
        unit(
            "german_guidelines",
            name + "/legal",
            path,
            "",
            dict(
                dimension="legal",
                all_labels=data["labels"],
                macro_labels=data["macro_labels"],
                gold_support=data["full_gold_support"],
                rows=data["full_inputs"],
            ),
            sorted(data["common"]),
            contrasts,
            "german_legal_arrays",
            {
                "scorable_rows": data["full_inputs"],
                "common_rows": data["common_inputs"],
                "prediction_path": c.rel(root / name / "arrays.npz"),
                "metadata_path": c.rel(root / name / "input_roster.json"),
                "resampling_source": c.rel(root / name / "bootstrap_design.json"),
                "legal_gate": "independent of predicted hate",
            },
        )
    family(
        "german_guidelines",
        members,
        "Three legal TEST pairs; DEV remains supporting and the original binary hate family is unchanged.",
    )
    study(
        "feedback",
        "Correction uptake and fixed-pool transfer",
        [7],
        "astra_sq3_feedback_final_results_20260930_v1.md",
        "Results_SQ3/01_correction_uptake_transfer_and_relevance.tex",
    )
    root = B / "sq3_full_kb_feedback_20260928_v1/results_v1"
    for path in sorted(root.glob("*_b0[1-4].json")):
        append_source_unit(
            "feedback",
            path.stem,
            path,
            "hate_type",
            TYPE,
            {
                "resampling": "original paired text resampling and study-specific pool/repeat hierarchy",
                "new_micro_role": (
                    "family" if path.stem == "transfer_b04" else "descriptive"
                ),
            },
        )
    c.bind(root / "candidate_diagnostics.json")
    c.bind(root / "all_run_metrics.json")
    family(
        "feedback",
        ["feedback/transfer_b04|error_minus_correct"],
        "The original forty-correction transfer contrast; source uptake, other budgets and controls remain supporting.",
    )
    study(
        "relevance",
        "Gold-assisted correction relevance",
        [7],
        "astra_sq3_relevance_final_results_20261001_v1.md",
        "Results_SQ3/01_correction_uptake_transfer_and_relevance.tex",
    )
    for name in ("primary", "primary_controls", "all_transfer", "partial_cover_seven"):
        append_source_unit(
            "relevance",
            name,
            B / "sq3_full_kb_relevance_20260930_v1/results_v1" / (name + ".json"),
            "hate_type",
            TYPE,
        )
    study(
        "explanations",
        "Correction explanations and selectivity",
        [7],
        "astra_sq3_explanation_selectivity_final_results_20261004_v1.md",
        "Results_SQ3/02_explanations_and_selective_delivery.tex",
    )
    for name in ("primary_explanation", "primary_application", "all_transfer"):
        append_source_unit(
            "explanations",
            name,
            B / "sq3_combined_results_20261004_v1/explanations" / (name + ".json"),
            "hate_type",
            TYPE,
        )
    family(
        "sq3_followups",
        [
            "relevance/primary|relevant_full_minus_reference",
            "explanations/primary_explanation|specific_always_minus_plain_always",
            "explanations/primary_application|plain_selective_minus_plain_always",
        ],
        "Three ordinary multilabel questions from the original four-question follow-up family. Focal-category restoration is a different estimand.",
    )
    supplemental = [
        (
            "balanced",
            "Balanced correction sources",
            "astra_sq3_balanced_final_results_20261010_v1.md",
            "Results_SQ3/04_balanced_corrections_ordinary_insertion_and_delivery.tex",
            "sq3_balanced_results_20261010_v1/types/all_policies.json",
        ),
        (
            "ordinary_insertion",
            "Ordinary five-slot insertion",
            "astra_sq3_additions_final_summary_20261006_v1.md",
            "Results_SQ3/04_balanced_corrections_ordinary_insertion_and_delivery.tex",
            "sq3_five_slot_results_20261012_v1/full/types/all_policies.json",
        ),
        (
            "reserved",
            "Reserved-slot correction label and context controls",
            "astra_sq3_reserved_part5_quality_20261020_v1.md",
            "Results_SQ3/05_reserved_delivery_and_broader_transfer.tex",
            "sq3_reserved_part5_results_20261020_v1/paired_quality.json",
        ),
    ]
    for key, name, report, loc, source in supplemental:
        study(
            key,
            name,
            [7],
            report,
            loc,
            "Fine-label results remain supporting to the original binary primary questions.",
        )
        append_source_unit(
            key,
            "hate_type",
            B / source,
            "hate_type",
            TYPE,
            {
                "new_micro_role": "descriptive",
                "scope": "All original type comparisons in this bound result; exclude alternate-mask duplicates from main tables, retain them as sensitivity.",
            },
        )
    study(
        "transfer",
        "Broader fixed-policy correction transfer",
        [7],
        "astra_sq3_transfer_part4_quality_20261023_v1.md",
        "Results_SQ3/05_reserved_delivery_and_broader_transfer.tex",
    )
    path = B / "sq3_transfer_part4_20261023_v1/quality_results.json"
    data = c.read(path)
    members = []
    for name, record in data.items():
        if not name.startswith("population/") or record["head"] not in (
            "hate_type",
            "target_group",
        ):
            continue
        if record.get("eligible_n", 0) == 0:
            continue
        dim = record["head"]
        unit(
            "transfer",
            name,
            path,
            "/" + name,
            dict(
                dimension=dim,
                all_labels=TYPE if dim == "hate_type" else TARGET,
                macro_labels=record["population_macro_labels"],
                scorable_rows=record["eligible_n"],
            ),
            sorted(record.get("condition_means", {})),
            [],
            "quality_features",
            {
                "common_rows": record["common_n"],
                "mode": record.get("mode"),
                "view": record.get("view"),
                "new_micro_role": "family" if name.endswith("/main") else "sensitivity",
                "prediction_path": "experiments/astra/sq3_transfer_part4_20261023_v1/prediction_codes.npz",
                "feature_path": "experiments/astra/sq3_transfer_part4_20261023_v1/features.npz",
                "schema_path": "experiments/astra/sq3_transfer_part4_20261023_v1/feature_schema.json",
            },
        )
        if name.endswith("/main"):
            members.append("transfer/" + name + "|feedback-reference")
    family(
        "transfer",
        members,
        "Three scorable global type/target outcomes; source/function and strict views remain supporting.",
    )
    study(
        "rounds",
        "Four successive feedback rounds",
        [7],
        "astra_sq3_iterative_part6_quality_20261027_v1.md",
        "Results_SQ3/06_four_successive_simulated_rounds.tex",
    )
    path = B / "sq3_iterative_review_20261026_v1/part6/analysis.json"
    data = c.read(path)
    members = []
    for name, record in data["records"].items():
        if not name.startswith("population/") or record["head"] not in (
            "hate_type",
            "target_group",
        ):
            continue
        if record.get("eligible_n", 0) == 0:
            continue
        dim = record["head"]
        unit(
            "rounds",
            name,
            path,
            "/records/" + name,
            dict(
                dimension=dim,
                all_labels=TYPE if dim == "hate_type" else TARGET,
                macro_labels=record.get("macro_labels", []),
                scorable_rows=record["eligible_n"],
            ),
            list(record.get("columns", {})),
            [],
            "round_features",
            {
                "common_rows": record.get("n"),
                "view": record["view"],
                "original_columns": record.get("columns"),
                "prediction_path": "experiments/astra/sq3_iterative_review_20261026_v1/part6/prediction_codes.npz",
                "feature_path": "experiments/astra/sq3_iterative_review_20261026_v1/part6/features.npz",
                "schema_path": "experiments/astra/sq3_iterative_review_20261026_v1/part6/feature_schema.json",
            },
        )
        if record["view"] == "final":
            for contrast in (
                "r4_adaptive-reference",
                "r4_adaptive-static",
                "r4_adaptive-random",
            ):
                c.require(contrast in record["columns"], "Missing final-round contrast")
                members.append("rounds/" + name + "|" + contrast)
    family(
        "rounds",
        members,
        "Final current-error versus fresh reference and both selection controls on the three scorable global type/target populations. Earlier-round and fixed-cohort trajectories are supporting.",
    )
    study(
        "human",
        "Two-person human feedback",
        [7],
        "astra_sq3_human_feedback_results_20261020_v1.md",
        "Results_SQ3/07_human_interpretation_and_feedback.tex",
        "Descriptive only; no new hypothesis tests.",
    )
    append_source_unit(
        "human",
        "hate_type",
        B / "sq3_human_feedback_results_20261020_v1/frozen_descriptive_analysis.json",
        "hate_type",
        TYPE,
        {
            "new_micro_role": "descriptive_no_intervals",
            "scope": "Original dataset, reviewer A and reviewer B separately; source uptake versus transfer; preserve unresolved-annotation masks and message-change controls.",
            "prediction_path": "experiments/astra/sq3_human_feedback_results_20261020_v1/predictions.json",
            "gold_and_cases_path": "experiments/astra/sq3_human_feedback_results_20261020_v1/all_cases.json",
        },
    )
    for key, name, report, source in [
        (
            "label_vote",
            "Voting from the same neighbours",
            "astra_sq2_label_vote_results_20261001_v1.md",
            "sq2_retrieved_label_vote_20261001_v1/results.json",
        ),
        (
            "pooled",
            "Pooled existing fine-field evidence",
            "astra_pooled_fineheads_results_20260930_v1.md",
            "pooled_fineheads_20260930_v1/attempts/1790803999301263800/results.json",
        ),
        (
            "response_vote",
            "Single response versus response voting",
            "astra_single_three_vote_results_20261006_v1.md",
            "offline_closure_20261006_v1/votes/results.json",
        ),
    ]:
        study(
            key,
            name,
            [2],
            report,
            "Results_SQ2/05_context_explanations_and_reliability.tex",
            "Earlier supporting evidence, separate from repeated-answer consistency and broader Qwen.",
        )
        data = c.read(B / source)
        unit(
            key,
            "multilabel_supporting",
            B / source,
            "",
            dict(dimension="multiple"),
            [],
            [],
            "earlier_supporting_result",
            {
                "scope": "All original type/target/legal comparisons in this bound source; retain original masks, abstentions and ordering. No new family for these supporting controls."
            },
        )
    exclusions = [
        dict(
            id="binary",
            reason="Hate is binary; retain its original macro F1, FPR/FNR and accuracy. No relabelling as multilabel micro.",
        ),
        dict(
            id="severity",
            reason="Ordinal output; retain original scoring and gate handling. Do not assume an empty prediction is an ordinary class.",
        ),
        dict(
            id="restoration",
            reason="The principal restoration statistic averages focal-category F1 across seven different interventions, not one model's full label set. No ordinary pooled micro counterpart.",
            thesis_location="sections/Thesis_Chapters/Results_SQ3/03_restoration_of_familiar_categories.tex",
        ),
        dict(
            id="restoration_factorial",
            reason="Focal text/category intervention pairs and their example/definition effects retain their original estimand, rather than combining different interventions into a synthetic multilabel predictor.",
        ),
        dict(
            id="sq3_linear",
            reason="Same-feedback classifiers only predict binary hate; no type/target/legal predictions exist.",
            thesis_location="sections/Thesis_Chapters/Results_SQ3/08_using_the_same_labels_to_refit_a_classifier.tex",
        ),
        dict(
            id="benign_guidelines",
            reason="Selected benign-only English HateCheck experiment estimates false alarms; no new multilabel F1 claim.",
        ),
        dict(
            id="repeat_agreement",
            reason="Agreement between model repeats is not F1 against gold and already has its own completed analysis.",
        ),
        dict(
            id="retired_trials",
            reason="Abandoned, superseded or purely operational trials are outside the current central comparisons. They remain preserved.",
        ),
        dict(
            id="unsupported_fine_gold",
            reason="HateCheck type outputs and unannotated fields are unscorable. Empty observed target sets remain distinct from missing fields.",
        ),
    ]
    c.require(len({u["id"] for u in UNITS}) == len(UNITS), "Duplicate registry unit")
    for u in UNITS:
        u["full_micro_label_policy"] = (
            "All scoreable columns, including zero/low gold support; frozen taxonomy only."
        )
        u["restricted_micro_required"] = (
            bool(u["labels"]) and u["labels"] != u["macro_labels"]
        )
        u["macro_role_preserved"] = True
    c.save(
        "comparison_registry.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            status="frozen_before_wider_micro_estimates",
            studies=STUDIES,
            units=UNITS,
            new_micro_families=FAMILIES,
            exclusions=exclusions,
            wider_micro_estimates_computed=0,
            rule="Main families supplement the original macro tests. All other registered contrasts are nominal supporting, sensitivity, or descriptive only. Do not reselect contrasts after seeing micro.",
        ),
    )
    c.save("registry_source_bindings.json", c.BINDINGS)
    print(
        "Frozen registry:",
        len(STUDIES),
        "study groups;",
        len(UNITS),
        "endpoint/view units;",
        sum((v["m"] for v in FAMILIES.values())),
        "adjusted supporting micro contrasts.",
        flush=True,
    )


if __name__ == "__main__":
    build()
