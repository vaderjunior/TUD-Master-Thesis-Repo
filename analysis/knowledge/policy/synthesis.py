from pathlib import Path
import argparse
import json
from collections import Counter
from itertools import combinations
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import reader as reader

BASE = c.ROOT / "experiments/astra/complete_policy_part7_20261115_v1"
RESULT = c.ROOT / "Thesis_Log/astra_result"
P3 = reader.BASE
RUN = None
P4 = c.ROOT / "experiments/astra/complete_policy_part4_20261115_v1"
P5 = c.ROOT / "experiments/astra/complete_policy_part5_20261115_v1"
P6 = c.ROOT / "experiments/astra/complete_policy_part6_20261115_v1"
BANKS = c.ROOT / "experiments/astra/micro_f1_sq1_banks_strategies_20261113_v1"
COMPONENTS = c.ROOT / "experiments/astra/micro_f1_sq1_components_guidelines_20261113_v1"
GERMAN = c.ROOT / "experiments/astra/german_guidelines_synthesis_20261113_v1"
REPORT_NAMES = [
    "complete_policy_part1_protocol_20261114_v1.md",
    "complete_policy_inventory_20261114_v1.md",
    "complete_policy_work_plan_20261114_v2.md",
    "complete_policy_part3_performance_20261115_v1.md",
    "complete_policy_part4_context_cases_20261115_v1.md",
    "complete_policy_part5_robustness_20261115_v1.md",
    "complete_policy_part6_resources_20261115_v1.md",
    "astra_core_results_summary_20260920_v1.md",
    "astra_sq1_strategy_final_results_20260921_v1.md",
    "astra_sq1_components_final_summary_20260924_v1.md",
    "astra_sq1_allocation_final_summary_20260925_v1.md",
    "astra_sq1_guideline_final_results_20260926_v1.md",
    "astra_sq1_pool_final_summary_20260928_v1.md",
    "astra_sq1_annotation_coverage_20260926_v1.md",
    "micro_f1_sq1_banks_strategies_analysis_20261113_v1.md",
    "micro_f1_sq1_components_guidelines_analysis_20261113_v1.md",
    "german_guidelines_part7_synthesis_20261113_v2.md",
    "german_guidelines_part6_robustness_20261113_v1.md",
    "astra_sq2_broad_final_synthesis_20261104_v2.md",
    "astra_sq2_combined_final_synthesis_20261104_v1.md",
    "astra_sq2_matched_writing_handover_20261030_v1.md",
    "astra_remaining_work_before_writing_20261115_v187.md",
]
LEGACY_EVIDENCE = [
    "core_completion_summary_20260920_v1.json",
    "sq1_strategy_completion_20260921_v1.json",
    "sq1_components_completion_20260924_v1.json",
    "sq1_allocation_completion_20260925_v1.json",
    "sq1_guideline_final_results_20260926_v1.json",
    "sq1_guideline_final_verification_20260926_v1.json",
    "sq1_pool_completion_20260928_v1.json",
]


def run_root():
    global RUN
    if RUN is None:
        RUN = c.ROOT / c.read(P3 / "analysis_complete.json")["run"]
    return RUN


def save(name, value):
    path = BASE / name
    c.require(c.normal(path).is_relative_to(BASE), "Write outside Part 7")
    c.save(path, value)


def prepare():
    assert not (BASE / "analysis_contract.json").exists(), "Part 7 already prepared"
    bindings = {}
    receipt_checks = []
    for folder, part in ((P3, 3), (P4, 4), (P5, 5), (P6, 6)):
        path = folder / f"part{part}_complete.json"
        value = c.read(path)
        for field in ("artifacts", "scripts", "reports"):
            c.verify(value.get(field, {}))
            bindings.update(value.get(field, {}))
        bindings.update(c.bindings([path]))
        receipt_checks.append(
            dict(
                part=part,
                status=value.get("status"),
                bindings=len(value.get("artifacts", {})),
            )
        )
    legacy = []
    for folder in (BANKS, COMPONENTS):
        receipt = c.read(folder / "complete.json")
        index = c.read(folder / "accepted_analysis_index.json")
        paths = [
            folder / "complete.json",
            folder / "accepted_analysis_index.json",
            folder / "independent_verification.json",
            folder / "publication_checks.json",
        ]
        for item in index["outputs"]:
            path = c.ROOT / item
            expected = receipt["artifact_sha256"][item]
            assert c.sha(path) == expected, item
            paths.append(path)
        report_path = RESULT / (
            "micro_f1_sq1_banks_strategies_analysis_20261113_v1.md"
            if folder == BANKS
            else "micro_f1_sq1_components_guidelines_analysis_20261113_v1.md"
        )
        assert c.sha(report_path) == receipt["artifact_sha256"][c.ref(report_path)]
        bindings.update(c.bindings(paths))
        legacy.append(
            dict(
                root=c.ref(folder),
                accepted_status=index["status"],
                checked_outputs=len(index["outputs"]),
            )
        )
    paths = [RESULT / name for name in REPORT_NAMES] + [
        RESULT / "evidence" / name for name in LEGACY_EVIDENCE
    ]
    paths += [
        c.SETUP / name
        for name in (
            "protocol.json",
            "policy_inventory.json",
            "roster.jsonl",
            "endpoints.json",
            "comparison_registry.json",
            "budget.json",
        )
    ]
    paths += [
        c.PLAN,
        GERMAN / "report_v2/cross_study_evidence.json",
        GERMAN / "report_v2/publication_tables.json",
        GERMAN / "runs/run_1794534112318762800/synthesis_scores.json",
        c.ROOT / "config/taxonomy.yaml",
        c.ROOT / "src/hsrag/retrieve.py",
        c.ROOT / "src/hsrag/prompt.py",
    ]
    paths = [
        path
        for path in paths
        if path != c.ROOT / "src/hsrag/prompt.py" or path.exists()
    ]
    bindings.update(c.bindings(paths))
    before = reader.snapshot_thesis()
    save(
        "analysis_contract.json",
        dict(
            status="Part 7 synthesis scope before new evidence tables",
            created_utc=c.a.utc(),
            inputs=bindings,
            previous_part_receipt_checks=receipt_checks,
            legacy_accepted_output_checks=legacy,
            thesis_before=before,
            aim="Answer SQ1 by integrating available example banks, retrieval procedures, naturally selected components, deliberate rule delivery, definition allocation, German guideline separation and complete policy.",
            interpretation="Retrospective scientific synthesis of already analysed studies. Preserve each intervention, population, mask, estimator, uncertainty family and exposure status. No new F1, bootstrap, pooled estimate, cross-study subtraction or multiple-test selection.",
            evidence_order="Frozen within-study comparisons first; supporting error/coverage/context evidence second; descriptive cross-study compatibility third. Similar directions do not create independent replications.",
            architecture="Distinguish example retrieval from selection of short policy records. A/B/C retain the same five examples. No automatic baseline replacement, untested hybrid configuration, guideline removal or new experiment.",
            limits="No causal attribution to individual rules, pure length/language/ordering effects or formal interaction from differently significant studies. Retain legal recall/strict-inner cautions, type consistency decline and output-validity versus correctness.",
            task_boundary="Scientific synthesis only. Separate writer/defence implementation handover remains Part 8. No changes to LaTeX, user data, raw predictions, earlier analyses, model/training/server state or Git.",
            new_calls=0,
            no_network=True,
            no_new_training=True,
        ),
    )
    with (BASE / "session_before.md").open("xb") as f:
        f.write((c.ROOT / "Thesis_Log/astra_session.md").read_bytes())
    print(
        f"Part 7 scope recorded; {len(bindings)} source bindings, prior Parts 3–6 and 22 accepted legacy views verified.",
        flush=True,
    )


def token(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def pointer(value, path):
    for key in path.lstrip("/").split("/") if path else []:
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def catalogue():
    contract = c.read(BASE / "analysis_contract.json")
    c.verify(contract["inputs"])
    records = []
    studies = []

    def add(path, identity, metadata, fields):
        source = c.read(path)
        reference = c.ref(path)
        c.require(reference in contract["inputs"], "Unbound synthesis source")
        row = dict(
            id=identity,
            **metadata,
            source=reference,
            source_sha256=c.sha(path),
            source_fields=fields,
        )
        for field, ptr in fields.items():
            row[field] = pointer(source, ptr)
        c.require(
            row["metric"]
            in (
                "macro_f1",
                "micro_f1",
                "false_positive_rate",
                "mean_pairwise_repeat_agreement",
            ),
            "Unknown metric",
        )
        c.require(
            row["n"] > 0 and len(row["ci95"]) == 2,
            "Missing paired cohort or uncertainty",
        )
        c.require(
            (row["family_interval"] is None) == (row["family_confidence"] is None),
            "Incomplete family interval",
        )
        row["preferred_interval"] = (
            row["family_interval"]
            if row["family_interval"] is not None
            else row["ci95"]
        )
        row["preferred_confidence"] = (
            row["family_confidence"] if row["family_confidence"] is not None else 0.95
        )
        row["interval_scope"] = (
            "original comparison family"
            if row["family_interval"] is not None
            else "descriptive 95%"
        )
        row["better_direction"] = (
            "lower" if row["metric"] == "false_positive_rate" else "higher"
        )
        records.append(row)

    path = RESULT / "evidence/core_completion_summary_20260920_v1.json"
    for ep, info in c.read(path)["endpoints"].items():
        start = "/endpoints/" + token(ep) + "/full_minus_small"
        fields = {
            k: start + "/" + v
            for k, v in dict(
                difference="mean_difference",
                n="n_common_valid",
                ci95="ci95",
                family_interval="family_ci",
                family_confidence="family_confidence",
                left="left",
                right="right",
                left_value="left_mean_on_common_valid",
                right_value="right_mean_on_common_valid",
            ).items()
        }
        add(
            path,
            f"core:{ep}:full_minus_small:macro_f1",
            dict(
                study="core",
                endpoint=ep,
                view="paired",
                metric="macro_f1",
                source_role=info["endpoint"]["role"],
                primary=info["full_minus_small"]["primary"],
            ),
            fields,
        )
    for study, name in [
        ("strategy", "sq1_strategy_completion_20260921_v1.json"),
        ("components", "sq1_components_completion_20260924_v1.json"),
        ("allocation", "sq1_allocation_completion_20260925_v1.json"),
        ("pools", "sq1_pool_completion_20260928_v1.json"),
    ]:
        path = RESULT / "evidence" / name
        for ep, info in c.read(path)["endpoints"].items():
            for name, con in info["contrasts"].items():
                start = "/endpoints/" + token(ep) + "/contrasts/" + token(name)
                fields = {
                    k: start + "/" + v
                    for k, v in dict(
                        difference="mean_difference",
                        n="n_common_valid",
                        ci95="ci95",
                        family_interval="family_ci",
                        family_confidence="family_confidence",
                    ).items()
                }
                for field in (
                    "left",
                    "right",
                    "weights",
                    "mask",
                    "group",
                    "primary_family_member",
                ):
                    if field in con:
                        fields[field] = start + "/" + field
                add(
                    path,
                    f"{study}:{ep}:{name}:macro_f1",
                    dict(
                        study=study,
                        endpoint=ep,
                        view=con.get("mask", "paired"),
                        metric="macro_f1",
                        source_role=info.get(
                            "role", info.get("endpoint", {}).get("role")
                        ),
                        contrast=name,
                    ),
                    fields,
                )
    for folder in (BANKS, COMPONENTS):
        for reference in c.read(folder / "accepted_analysis_index.json")["outputs"]:
            path = c.ROOT / reference
            value = c.read(path)
            unit = value["unit"]
            study = unit["id"].split("/")[0]
            ep = unit["endpoint"]
            studies.append(
                dict(
                    source=reference,
                    source_sha256=c.sha(path),
                    unit=unit,
                    view=value.get("view", "paired"),
                    n_eligible=value["n_eligible"],
                    n_common=value["n_common"],
                    labels=value["labels"],
                    macro_labels=value["macro_labels"],
                )
            )
            for name, con in value["contrasts"].items():
                start = "/contrasts/" + token(name)
                m = start + "/metrics/micro"
                fields = {
                    k: m + "/" + v
                    for k, v in dict(
                        difference="difference",
                        ci95="ci95",
                        family_interval="adjusted_interval",
                        family_confidence="adjusted_confidence",
                        original_confidence_diagnostic="original_confidence_diagnostic",
                    ).items()
                }
                fields["n"] = start + "/n_common"
                for field in ("left", "right", "weights", "family_size", "question_id"):
                    if field in con:
                        fields[field] = start + "/" + field
                add(
                    path,
                    f"{study}:{ep}:{name}:micro_f1",
                    dict(
                        study=study,
                        endpoint=ep,
                        view=value.get("view", "paired"),
                        metric="micro_f1",
                        source_role="complementary micro; original macro remains primary",
                        contrast=name,
                    ),
                    fields,
                )
    path = RESULT / "evidence/sq1_guideline_final_results_20260926_v1.json"
    for name, con in c.read(path)["scopes"]["all"]["contrasts"].items():
        start = "/scopes/all/contrasts/" + token(name)
        fields = {
            k: start + "/" + v
            for k, v in dict(
                difference="mean_difference",
                ci95="descriptive_95_interval",
                family_interval="primary_family_interval",
                family_confidence="primary_family_confidence",
                left="left",
                right="right",
            ).items()
        }
        fields["n"] = "/scopes/all/common_items"
        add(
            path,
            f"guideline_oracle:en_benign:{name}:false_positive_rate",
            dict(
                study="guideline_oracle",
                endpoint="en_benign",
                view="all",
                metric="false_positive_rate",
                contrast=name,
                source_role=(
                    "primary"
                    if con["primary_family_interval"] is not None
                    else "supporting"
                ),
            ),
            fields,
        )
    path = GERMAN / "runs/run_1794534112318762800/synthesis_scores.json"
    for ep, info in c.read(path).items():
        for i, con in enumerate(info["contrasts"]):
            start = "/" + token(ep) + "/contrasts/" + str(i)
            m = start + "/metrics/macro_f1"
            fields = {
                k: m + "/" + k
                for k in ("difference", "ci95", "family_interval", "family_confidence")
            }
            fields.update(
                n="/" + token(ep) + "/common_inputs",
                left=start + "/left",
                right=start + "/right",
            )
            add(
                path,
                f"german_guidelines:{ep}:{con['left']}_minus_{con['right']}:macro_f1",
                dict(
                    study="german_guidelines",
                    endpoint=ep,
                    view="paired",
                    metric="macro_f1",
                    source_role=con["role"],
                ),
                fields,
            )
    path = run_root() / "contrasts.json"
    for i, con in enumerate(c.read(path)):
        start = "/" + str(i)
        fields = {
            k: start + "/" + v
            for k, v in dict(
                difference="estimate",
                ci95="interval95",
                family_interval="family_interval",
                family_confidence="family_confidence",
                n="n",
                left="left",
                right="right",
                left_value="left_value",
                right_value="right_value",
                family="family",
            ).items()
        }
        metric = "micro_f1" if con["metric"] == "micro_f1_all_labels" else con["metric"]
        add(
            path,
            f"complete_policy:{con['endpoint']}:{con['left']}_minus_{con['right']}:{metric}",
            dict(
                study="complete_policy",
                endpoint=con["endpoint"],
                view="common_nine",
                metric=metric,
                source_role=con["role"],
            ),
            fields,
        )
    c.require(
        len({r["id"] for r in records}) == len(records),
        "Duplicated comparison identity",
    )
    save(
        "comparison_catalogue.json",
        dict(
            status="Copied from accepted sources; not rescored",
            created_utc=c.a.utc(),
            records=records,
            by_study=dict(Counter((r["study"] for r in records))),
            by_metric=dict(Counter((r["metric"] for r in records))),
            scope="Original core macro, all original SQ1 strategy/pool/component/allocation macro contrasts, their accepted multilabel micro additions, English benign guideline contrasts, German guideline macro/micro and complete-policy contrasts. Historical core micro and SQ2 bridges remain in their linked accepted reports.",
            caution="No cross-study subtraction, pooled effect, newly adjusted family or extra test. A family interval is local to its original comparison family, not global multiplicity control for this whole catalogue. Descriptive intervals do not establish equivalence.",
        ),
    )
    save("legacy_view_metadata.json", studies)
    print(
        f"{len(records)} accepted contrasts copied with exact source pointers; {len(studies)} legacy views.",
        flush=True,
    )


def cohorts():
    import numpy as np

    groups = []

    def normal_id(key, lang):
        if key.startswith(("en/", "de/")):
            c.require(key.startswith(lang + "/"), "Unexpected ID language")
            return key
        c.require("/" not in key, "Unrecognised identifier shape")
        return lang + "/" + key

    for folder in (BANKS, COMPONENTS):
        for reference in c.read(folder / "accepted_analysis_index.json")["outputs"]:
            value = c.read(c.ROOT / reference)
            ep = value["unit"]["endpoint"]
            lang = ep.split("_")[0]
            c.require(lang in ("en", "de"), "Unknown cohort language")
            full = [normal_id(k, lang) for k in value["ids"]]
            common = [normal_id(k, lang) for k in value["common_ids"]]
            c.require(
                len(full) == len(set(full)) == value["n_eligible"],
                "Duplicate or missing legacy roster IDs",
            )
            c.require(
                len(common) == len(set(common)) == value["n_common"]
                and set(common) <= set(full),
                "Legacy common mask mismatch",
            )
            groups.append(
                dict(
                    id=value["unit"]["id"] + "/" + value.get("view", "paired"),
                    study=value["unit"]["id"].split("/")[0],
                    endpoint=ep,
                    dimension=value["unit"]["dimension"],
                    source=reference,
                    source_sha256=c.sha(c.ROOT / reference),
                    roster_ids=sorted(full),
                    common_ids=sorted(common),
                )
            )
    for ep, meta in c.read(c.SETUP / "endpoints.json").items():
        mask = np.load(run_root() / ep / "encoded.npz", allow_pickle=False)["common"]
        full = meta["keys"]
        common = [key for key, on in zip(full, mask, strict=True) if on]
        c.require(
            len(common) == c.read(run_root() / ep / "results.json")["common_nine"],
            "Complete-policy common mask differs",
        )
        groups.append(
            dict(
                id="complete_policy/" + ep,
                study="complete_policy",
                endpoint=ep,
                dimension=meta["dimension"],
                source=c.ref(c.SETUP / "endpoints.json"),
                source_sha256=c.sha(c.SETUP / "endpoints.json"),
                common_mask_source=c.ref(run_root() / ep / "encoded.npz"),
                roster_ids=sorted(full),
                common_ids=sorted(common),
            )
        )
    pairs = []
    for a, b in combinations(groups, 2):
        if a["dimension"] != b["dimension"] or a["endpoint"][:2] != b["endpoint"][:2]:
            continue
        pairs.append(
            dict(
                left=a["id"],
                right=b["id"],
                dimension=a["dimension"],
                roster_n=[len(a["roster_ids"]), len(b["roster_ids"])],
                shared_roster=len(set(a["roster_ids"]) & set(b["roster_ids"])),
                common_n=[len(a["common_ids"]), len(b["common_ids"])],
                shared_common=len(set(a["common_ids"]) & set(b["common_ids"])),
            )
        )
    save(
        "cohort_relationships.json",
        dict(
            created_utc=c.a.utc(),
            groups=groups,
            pairs=pairs,
            method="Read saved evaluation identities and masks only. Add the recorded endpoint language prefix when old IDs omit it; no fuzzy matching. Do not compare outcomes across different studies.",
            limits="ID equality documents reuse of texts, not independent replication or identical experimental conditions. Nonmatching DEV/TEST IDs do not establish semantic independence, lack of pretraining exposure or an untouched test set. HateCheck template dependence and GAHD partners remain as previously reported.",
        ),
    )
    print(
        f"{len(groups)} saved cohort rosters and {len(pairs)} same-dimension identity overlaps recorded.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "catalogue", "cohorts"])
    args = parser.parse_args()
    with c.offline():
        globals()[args.mode]()
