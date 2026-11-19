from analysis.common import micro_f1_files as c
from analysis.common import micro_f1_reader as r
from pathlib import Path

B = c.ROOT / "experiments/astra"


def at(data, selector):
    for key in selector:
        data = data[key]
    return data


def main():
    reg = c.read(c.BASE / "comparison_registry_v2.json")
    units = {u["id"]: u for u in reg["units"]}
    rows = []
    for family, f in reg["new_micro_families"].items():
        for member in f["members"]:
            first, *tail = member.split("|")
            sid = first.split("/")[0]
            uid = first
            if sid == "rounds":
                uid, name = member.rsplit("|", 1)
                tail = [name]
            if sid in {"compact_tuning", "shorter", "matched_type", "native_cpu"}:
                uid = sid + "/broad/" + first.split("/", 1)[1]
            u = units[uid]
            source = c.ROOT / u["source"]
            selector = list(u["source_selector"])
            data = at(c.read(source), selector)
            left = right = None
            weights = None
            old = None
            point = None
            if sid in {"core", "strategy", "components", "allocation", "pools"}:
                view, name = tail
                if sid == "core":
                    selector += ["contrasts", name]
                elif view == "paired":
                    selector += ["paired_comparison", "contrasts", name]
                else:
                    selector += ["paired_comparison", view, "contrasts", name]
                old = at(c.read(source), selector)
                left = old.get("left")
                right = old.get("right")
                weights = old.get("weights")
                point = old.get("mean_difference")
            elif sid == "broad":
                left, right = tail[0].split("-", 1)
                i = next(
                    (
                        i
                        for i, x in enumerate(data["contrasts"])
                        if x["left"] == left and x["right"] == right
                    )
                )
                selector += ["contrasts", i]
                old = data["contrasts"][i]
                point = old["mean_difference"]
            elif sid == "gpu":
                budget = int(tail[0])
                name = tail[1]
                source = source.parent / "primary_comparisons.json"
                d = c.read(source)
                i = next(
                    (
                        i
                        for i, x in enumerate(d["comparisons"])
                        if x["endpoint"] == u["endpoint"]
                        and x["budget"] == budget
                        and (x["contrast"] == name)
                    )
                )
                selector = ["comparisons", i]
                old = d["comparisons"][i]
                point = old["difference"]

                def find_method(score):
                    matches = [
                        x["method"]
                        for x in data["absolute"]
                        if x["budget"] == budget and abs(x["mean"] - score) < 1e-12
                    ]
                    c.require(len(matches) == 1, "GPU method ambiguity")
                    return matches[0]

                left = find_method(old["left"])
                right = find_method(old["right"])
                r.equal(old["left"] - old["right"], point, "gpu family contrast")
            elif sid in {"compact_tuning", "shorter", "native_cpu"}:
                name = first.split("/", 1)[1]
                source = c.ROOT / u["statistics_source"]
                d = c.read(source)
                container = "primary" if sid == "native_cpu" else "contrasts"
                selector = [container, name, "primary"] + (
                    ["unstratified"] if sid == "shorter" else []
                )
                old = at(d, selector)
                point = old["difference"]
                left, right = {
                    "compact_tuning": ("tuned_selected", "bge"),
                    "shorter": ("expanded_tuned_selected", "original_tuned_selected"),
                    "native_cpu": ("full_bge_matched_selected", "full_neural_selected"),
                }[sid]
                r.equal(data["means"][left], old["left"], "historical left identity")
                r.equal(data["means"][right], old["right"], "historical right identity")
                r.equal(
                    data["means"][left] - data["means"][right],
                    point,
                    "historical contrast",
                )
            elif sid == "matched_type":
                group, name = tail
                left, right = name.split("_minus_")
                source = c.ROOT / u["statistics_source"]
                selector = ["groups", group, "primary", name]
                old = at(c.read(source), selector)
                point = old["difference"]
                r.equal(
                    data["group_means"][group][left]["macro_f1"]
                    - data["group_means"][group][right]["macro_f1"],
                    point,
                    "matched type group contrast",
                )
            elif sid == "matched":
                selector += ["contrasts", tail[0]]
                old = data["contrasts"][tail[0]]
                point = old["mean_difference"]
                left = old["left_settings"]
                right = old["right_settings"]
            elif sid == "german_guidelines":
                index = int(tail[0][4:]) - 1
                selector += ["contrasts", index]
                old = data["contrasts"][index]
                left, right = (old["left"], old["right"])
                point = old["metrics"]["macro_f1"]["difference"]
                r.equal(
                    data["common"][left]["mean"]["macro_f1"]
                    - data["common"][right]["mean"]["macro_f1"],
                    point,
                    "German family contrast",
                )
            elif sid in {"feedback", "relevance", "explanations"}:
                name = tail[0]
                selector += ["contrasts", name]
                old = data["contrasts"][name]
                left, right = name.split("_minus_")
                point = old.get("mean_difference", old.get("difference"))
                if sid == "feedback":
                    r.equal(
                        data["groups"][left]["metrics"]["macro_f1"]
                        - data["groups"][right]["metrics"]["macro_f1"],
                        point,
                        "feedback group contrast",
                    )
            elif sid == "transfer":
                selector += ["metrics", "macro_f1"]
                old = data["metrics"]["macro_f1"]
                point = old["difference"]
                left = "equal mean of the three feedback pools, each averaged over decoder repeats"
                right = "shared reference, averaged once over its decoder repeats"
            elif sid == "rounds":
                name = tail[0]
                selector += ["columns", name]
                old = data["columns"][name]
                weights = old
                point = data["estimates"][data["column_order"].index(name)][0]
            else:
                raise ValueError("Unresolved family " + sid)
            c.require(point is not None, "Original macro point missing")
            c.require(
                weights is not None or (left is not None and right is not None),
                "Contrast direction missing",
            )
            rows.append(
                dict(
                    id=member,
                    family=family,
                    unit_id=uid,
                    dimension=u["dimension"],
                    left=left,
                    right=right,
                    weights=weights,
                    source=c.rel(source),
                    source_sha256=c.sha(source),
                    source_selector=selector,
                    original_macro_point=point,
                    new_micro_point=None,
                    planned_family_size=f["m"],
                    planned_confidence=f["confidence"],
                    original_source_descriptor=old,
                    interpretation="post-hoc supporting micro; original macro role, label set, masks and selection retained",
                )
            )
    c.require(
        len(rows) == 77 and len({x["id"] for x in rows}) == 77,
        "Family inventory differs",
    )
    c.save(
        "paired_contrast_manifest.json",
        dict(
            status="all questions resolved before wider micro calculations",
            rows=rows,
            original_point_identity_checks=dict(r.CHECKS),
            max_absolute_error=r.MAX_ERROR,
            sources=c.BINDINGS,
        ),
    )
    print(
        "Resolved",
        len(rows),
        "paired questions; identity checks",
        sum(r.CHECKS.values()),
        flush=True,
    )


if __name__ == "__main__":
    main()
