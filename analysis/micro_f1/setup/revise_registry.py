from analysis.common import micro_f1_files as c
from datetime import datetime, timezone
import copy

B = c.ROOT / "experiments/astra"
MULTI = {"hate_type", "target_group", "legal"}


def get_source(u):
    path = c.ROOT / u["source"]
    if u["study"] == "combined_context":
        path = path.parent / "scores.json"
    d = c.read(path)
    pointer = u.get("json_pointer", "")
    if pointer.startswith("/cells/"):
        return (d["cells"][pointer[7:]], ["cells", pointer[7:]])
    if pointer.startswith("/endpoints/"):
        return (d["endpoints"][pointer[11:]], ["endpoints", pointer[11:]])
    if pointer.startswith("/results/"):
        return (d["results"][pointer[9:]], ["results", pointer[9:]])
    if pointer.startswith("/records/"):
        return (d["records"][pointer[9:]], ["records", pointer[9:]])
    if pointer.startswith("/population/"):
        return (d[pointer[1:]], [pointer[1:]])
    return (d, [])


def scope(ep):
    labels = ep.get("all_labels", ep.get("labels", []))
    macro = ep.get("macro_labels")
    if macro is None and "supported_indices" in ep:
        macro = [labels[i] for i in ep["supported_indices"]]
    return (labels, macro)


def micro_paths(d, path=()):
    result = []
    if isinstance(d, dict):
        for k, v in d.items():
            if "micro" in k.lower():
                result.append(list(path + (k,)))
            else:
                result.extend(micro_paths(v, path + (k,)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            result.extend(micro_paths(v, path + (i,)))
    return result


def main():
    original = c.read(c.BASE / "comparison_registry.json")
    reg = copy.deepcopy(original)
    units = []
    availability = {}
    for old in reg["units"]:
        u = copy.deepcopy(old)
        sid = u["study"]
        data, selector = get_source(u)
        if sid in {"label_vote", "response_vote", "pooled"}:
            for name, record in data["endpoints"].items():
                ep = record.get("endpoint", {})
                if sid == "pooled":
                    if name != "target_group":
                        continue
                    ep = dict(
                        record["roster"],
                        dimension=name,
                        all_labels=record["roster"]["labels"],
                    )
                    ep["macro_labels"] = [
                        x for x in ep["all_labels"] if ep["gold_support"][x] >= 10
                    ]
                if ep.get("dimension") not in MULTI:
                    continue
                v = copy.deepcopy(u)
                labels, macro = scope(ep)
                v.update(
                    id=sid + "/" + name,
                    endpoint=name,
                    dimension=ep["dimension"],
                    labels=labels,
                    macro_labels=macro,
                    gold_support=ep.get("gold_support"),
                    original_mask=ep.get("mask"),
                    scorable_rows=ep.get("scorable_rows", ep.get("rows")),
                    scorable_ids=ep.get("scorable_ids", ep.get("original_ids")),
                    source_selector=["endpoints", name],
                    reader="earlier_supporting_counts",
                    new_micro_role="descriptive",
                    conditions=sorted(
                        record.get("conditions", record.get("pools", {}))
                    ),
                    original_contrasts=list(record.get("contrasts", {})),
                )
                if sid == "label_vote":
                    v["comparison_rule"] = (
                        "RAG repeat mean minus one deterministic neighbour-label baseline, separately for full and compact retrieval; preserve annotation availability and abstention."
                    )
                if sid == "response_vote":
                    v["comparison_rule"] = (
                        "Three-response majority minus each saved single response; preserve inner versus outer repeat roles."
                    )
                v["original_endpoint_descriptor"] = ep
                units.append(v)
                availability[v["id"]] = micro_paths(record)
            continue
        if sid == "relevance" and u["endpoint"] == "partial_cover_seven":
            reg["exclusions"].append(
                dict(
                    id="relevance_partial_cover_as_primary",
                    reason="Seven partial-cover examples are a descriptive coverage diagnostic, not a new primary F1 population.",
                    source=u["source"],
                )
            )
            continue
        if sid in {"gpu_controls", "five_fit_update", "combined_context"}:
            parent = (c.ROOT / u["source"]).parent
            meta = (
                c.read(parent / "metadata.json")
                if sid == "combined_context"
                else c.read(c.ROOT / u["metadata_path"])
            )
            labels, macro = scope(meta)
            if macro is None:
                macro = [
                    labels[i] for i in data.get("supported_indices", range(len(labels)))
                ]
            u["labels"] = labels
            u["macro_labels"] = macro
            u["scorable_ids"] = meta.get("ids")
            u["scorable_rows"] = len(meta.get("ids", []))
            u["gold_support"] = meta.get("gold_support")
            if sid == "combined_context":
                u["source"] = c.rel(parent / "scores.json")
                u["source_sha256"] = c.sha(parent / "scores.json")
                data = c.read(parent / "scores.json")
                u["metadata_path"] = c.rel(parent / "metadata.json")
                u["prediction_path"] = c.rel(parent / "aligned.npz")
                u["conditions"] = meta["models"]
                u["original_contrasts"] = meta["original_contrasts"]
        if sid == "core":
            for x in u["original_contrasts"]:
                if (
                    x["name"] == "rag_full_minus_fewshot_eligible"
                    and u["endpoint"] != "en_dev_eval_types"
                ):
                    x["new_micro_role"] = "family"
        if sid == "relevance":
            u["count_source"] = c.rel(
                B
                / "sq3_full_kb_relevance_20260930_v1/results_v1/label_count_diagnostics.json"
            )
        if sid == "reserved":
            u["count_source"] = c.rel(
                B / "sq3_reserved_part5_results_20261020_v1/paired_run_details.json"
            )
        if sid == "human":
            u["gold_targets"] = sorted(data["transfer"]["other_all"])
            u["cohorts"] = sorted(data["transfer"])
            u["conditions"] = sorted(
                data["transfer"]["other_all"]["dataset"]["labels"]["arms"]
            )
            u["macro_labels"] = u["labels"]
        if sid in {
            "feedback",
            "relevance",
            "explanations",
            "balanced",
            "ordinary_insertion",
        }:
            u["original_contrasts"] = list(data.get("contrasts", {}))
            u["original_group_descriptors"] = data.get("groups", {})
            u["macro_labels"] = u["labels"]
        if sid in {"compact_tuning", "shorter", "matched_type", "native_cpu"}:
            labels, macro = scope(data.get("endpoint", data))
            if labels:
                u["labels"] = labels
            if macro is not None:
                u["macro_labels"] = macro
            stats = c.read(c.ROOT / u["statistics_source"])
            u["frozen_statistics_selectors"] = list(
                stats.get("contrasts", stats.get("groups", {}))
            )
        u["source_selector"] = selector
        units.append(u)
        availability[u["id"]] = micro_paths(data)
    reg["units"] = units
    for study in reg["studies"]:
        if study["id"] == "ordinary_insertion":
            study["report"] = (
                "Thesis_Log/astra_result/astra_sq3_five_slot_results_20261012_v2.md"
            )
        c.bind(c.ROOT / study["report"])
    family = reg["new_micro_families"]["core"]
    family["members"] = [
        "core/" + x["id"]
        for u in units
        if u["study"] == "core"
        for x in u["original_contrasts"]
        if x["new_micro_role"] == "family"
    ]
    family["m"] = len(family["members"])
    family["confidence"] = 1 - 0.05 / family["m"]
    family = reg["new_micro_families"]["sq3_followups"]
    family["members"][0] = "relevance/primary|relevant_full_minus_natural"
    for u in units:
        u["json_pointer"] = "".join(
            (
                "/" + str(k).replace("~", "~0").replace("/", "~1")
                for k in u["source_selector"]
            )
        )
        u["restricted_micro_required"] = set(u["labels"]) != set(u["macro_labels"])
        c.require(bool(u["labels"]), "Unspecified label set: " + u["id"])
        c.require(set(u["macro_labels"]).issubset(u["labels"]), "Unknown macro label")
        c.require(u["source_sha256"] == c.sha(c.ROOT / u["source"]), "Source changed")
        u["available_saved_micro_fields"] = len(availability[u["id"]])
        u["saved_micro_reuse_rule"] = (
            "Candidate only. Verify IDs, validity, label scope and aggregation before reusing; old headline prose and intervals are not automatically a match."
        )
        if "count_source" in u:
            c.bind(c.ROOT / u["count_source"])
    c.require(len({u["id"] for u in units}) == len(units), "Duplicate unit")
    for name, f in reg["new_micro_families"].items():
        c.require(
            len(f["members"]) == f["m"] == len(set(f["members"])), "Bad family count"
        )
    for member in reg["new_micro_families"]["sq3_followups"]["members"]:
        uid, contrast = member.split("|")
        u = next((x for x in units if x["id"] == uid))
        c.require(
            contrast in c.read(c.ROOT / u["source"])["contrasts"],
            "Missing follow-up question",
        )
    reg.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        version=2,
        supersedes="comparison_registry.json",
        wider_micro_estimates_computed=0,
        amendments=[
            "The core secondary files call the fixed eligible-example contrast rag_full_minus_fewshot_eligible, unlike the primary legal file. Include both type/target questions; core family 7 to 9.",
            "Correct the relevance primary comparator to natural retrieval. Reference is a supporting control, not the declared primary comparator.",
            "Bind ordinary five-slot insertion to its current v2 report rather than the different additions programme.",
            "Expand label voting, pooled targets and response voting to nine explicit multilabel endpoint units rather than three placeholder groups.",
            "Read original supported labels and model identities for control/aligned displays; point the aligned source at scores.json rather than a binary array.",
            "Move the seven partial-coverage illustrations out of the F1 scoring units; preserve their diagnostic source.",
            "Use explicit source selectors and escaped JSON pointers. No original evidence or numerical estimate is changed.",
        ],
    )
    c.save("comparison_registry_v2.json", reg)
    c.save(
        "saved_micro_availability.json",
        dict(
            status="source locations only; wider values not recalculated",
            locations=availability,
        ),
    )
    c.save("registry_v2_source_bindings.json", c.BINDINGS)
    print(
        len(reg["studies"]),
        "study groups",
        len(units),
        "units",
        sum((f["m"] for f in reg["new_micro_families"].values())),
        "supporting micro contrasts",
    )


if __name__ == "__main__":
    main()
