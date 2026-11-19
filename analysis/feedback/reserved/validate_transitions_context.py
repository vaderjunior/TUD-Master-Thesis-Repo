from collections import Counter, defaultdict
from pathlib import Path
from analysis.feedback.reserved import transitions as a
from analysis.feedback.reserved import transitions_readback as independent


def main():
    done = a.io.read(a.OUT / "analysis_complete.json")
    a.io.verify(a.io.ROOT, done["artifacts"])
    a.need(not (a.OUT / "context_check.json").exists(), "Preserve context check")
    roster, science, predictions, meta = a.load()
    bykey = {r["key"]: r for r in roster}
    totals = defaultdict(Counter)
    keys = defaultdict(set)
    for p in a.rows(a.OUT / "transition_pairs.jsonl.gz"):
        if not p["common_mask_bits"] & 1:
            continue
        t = meta[p["key"], p["treatment"]]
        c = meta[p["key"], p["control"]]
        row = bykey[p["key"]]
        shown = [v["meta"]["gate"] for v in t["delivered_annotations"]]
        visible = (
            "none"
            if not shown
            else (
                "hidden"
                if set(shown) == {None}
                else (
                    "hateful"
                    if set(shown) == {True}
                    else "benign" if set(shown) == {False} else "mixed"
                )
            )
        )
        delivery = (
            "none_selected"
            if t["source_id"] is None
            else (
                "selected"
                if t["source_id"] in t["delivered_correction_ids"]
                else "other_only" if t["delivered_correction_ids"] else "none"
            )
        )
        fifth = t["removed_fifth"]
        cosine = t["selected_correction_cosine"]
        dims = {
            "message": (
                "identical"
                if t["messages_sha256"] == c["messages_sha256"]
                else "changed"
            ),
            "source": row["source"],
            "visible_label": visible,
            "selected_delivery": delivery,
            "selected_matches_gold": (
                "none"
                if t["source_gate"] is None
                else str(t["source_gate"] == row["gold"]["hate"])
            ),
            "removed_fifth_gate": (
                str(fifth["meta"].get("gate")) if fifth else "not_removed"
            ),
            "removed_fifth_known_type": str(
                bool(fifth) and fifth["meta"].get("hate_types") is not None
            ),
            "cosine": (
                "none"
                if cosine is None
                else (
                    "<0.4"
                    if cosine < 0.4
                    else "0.4_to_0.6" if cosine < 0.6 else ">=0.6"
                )
            ),
        }
        for dimension, value in dims.items():
            name = p["contrast"] + "/" + p["endpoint"] + "/" + dimension + "/" + value
            independent.update(totals[name], p)
            keys[name].add(p["key"])
    expected = a.io.read(a.OUT / "context_strata.json")
    a.need(set(expected) == set(totals), "Context stratum coverage differs")
    derived = {"binary_error_difference", "type_exact_difference", "unique_texts"}
    for name, ct in totals.items():
        independent.same_counts(
            ct,
            {k: v for k, v in expected[name].items() if k not in derived},
            "Context stratum count differs",
        )
        a.need(
            len(keys[name]) == expected[name]["unique_texts"],
            "Context stratum text count differs",
        )
    repeats = defaultdict(Counter)
    rkeys = defaultdict(set)
    for row in roster:
        ep = "FNR" if row["gold"]["hate"] else "FPR"
        for condition in science["conditions"]:
            family = condition.rsplit("_s", 1)[0] if "__" in condition else condition
            for first, second in [(0, 1), (0, 2), (1, 2)]:
                value = independent.classify(
                    row["gold"],
                    predictions[row["key"], condition, second],
                    predictions[row["key"], condition, first],
                )
                for name in (family + "/" + ep, "all_conditions/" + ep):
                    independent.update(repeats[name], value)
                    rkeys[name].add(row["key"])
    expected_repeat = a.io.read(a.OUT / "within_condition_repeat_variation.json")
    a.need(set(expected_repeat) == set(repeats), "Repeat coverage differs")
    for name, ct in repeats.items():
        independent.same_counts(
            ct,
            {
                k: v
                for k, v in expected_repeat[name].items()
                if k not in derived | {"interpretation"}
            },
            "Repeat count differs",
        )
        a.need(
            expected_repeat[name]["unique_texts"] == len(rkeys[name]),
            "Repeat text count differs",
        )
    catalog = a.io.read(a.OUT / "example_catalog.json")
    changed = Counter()
    for eid, r in catalog.items():
        variants = list(r["metadata_variants"].values())
        fields = set().union(*(set(v) for v in variants))
        different = {
            k
            for k in fields
            if len({a.catalog_io.signature(v.get(k)) for v in variants}) > 1
        }
        changed.update(different)
        if not eid.startswith("sq3benign-"):
            a.need(
                different <= {"_ranks"},
                "Ordinary annotation changed across context variants",
            )
        else:
            a.need(
                different <= {"gate", "kind", "lang", "source", "_ranks"},
                "Correction fine annotation changed",
            )
            a.need(
                all(v.get("hate_types") is None for v in variants),
                "Correction has fine gold",
            )
            gates = {v["gate"] for v in variants}
            a.need(
                not ({True, False} <= gates),
                "Correction source has contradictory visible labels",
            )
    a.write(
        a.OUT / "context_check.json",
        {
            "status": "context_strata_and_repeat_counts_independently_reconciled",
            "created_utc": a.io.utc(),
            "context_strata_checked": len(totals),
            "repeat_groups_checked": len(repeats),
            "metadata_variant_fields": dict(changed),
            "ordinary_annotation_variation": "Only per-query _ranks differ; all original ordinary labels and texts preserved.",
            "correction_variation": "Visible versus hidden gate and metadata placement fields; no contradictory known gate or invented fine labels.",
            "bindings": {
                a.io.rel(Path(__file__)): a.io.sha(Path(__file__)),
                a.io.rel(Path(independent.__file__)): a.io.sha(
                    Path(independent.__file__)
                ),
                a.io.rel(a.OUT / "analysis_complete.json"): a.io.sha(
                    a.OUT / "analysis_complete.json"
                ),
            },
            "new_model_calls": 0,
        },
    )
    print(
        "All context strata, repeated-output counts and annotation metadata variants reconciled.",
        flush=True,
    )


if __name__ == "__main__":
    with a.io.offline():
        main()
