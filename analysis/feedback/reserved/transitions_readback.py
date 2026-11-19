from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
from pathlib import Path
from analysis.feedback.reserved import transitions as a


def own_category(gold, labels):
    missed = any(x not in labels for x in gold)
    extra = any(x not in gold for x in labels)
    return {
        (False, False): "exact",
        (False, True): "only_extra",
        (True, False): "only_missing",
        (True, True): "both_extra_and_missing",
    }[missed, extra]


def classify(gold, t, c):

    tr, cr = t["result"], c["result"]
    v = {"known_type_gold": gold["hate_type"] is not None}
    if tr is None or cr is None:
        v.update(
            status=(
                "both_null"
                if tr is None and cr is None
                else "treatment_null" if tr is None else "control_null"
            ),
            strict_pair_valid=False,
        )
        return v
    et = int(tr["hate"] != gold["hate"])
    ec = int(cr["hate"] != gold["hate"])
    names = {
        (0, 0): "unchanged_correct",
        (0, 1): "fix",
        (1, 0): "regression",
        (1, 1): "unchanged_error",
    }
    v.update(
        status="valid",
        strict_pair_valid=min(t["n_valid"], c["n_valid"]) == 3,
        binary=names[et, ec],
        gates=[tr["hate"], cr["hate"]],
        both_gates_positive=all([tr["hate"], cr["hate"]]),
        types_changed=sorted(tr["hate_type"]) != sorted(cr["hate_type"]),
    )
    if not v["known_type_gold"]:
        return v
    g = gold["hate_type"]
    tl = tr["hate_type"]
    cl = cr["hate_type"]
    tcat, ccat = own_category(g, tl), own_category(g, cl)
    v.update(
        type_categories=[tcat, ccat],
        type_exact=names[int(tcat != "exact"), int(ccat != "exact")],
        correct_labels_added=sum(x in tl and x not in cl for x in g),
        correct_labels_lost=sum(x in cl and x not in tl for x in g),
        extra_labels_added=sum(x not in g and x not in cl for x in tl),
        extra_labels_removed=sum(x not in g and x not in tl for x in cl),
    )
    return v


def update(ct, v):
    ct["eligible_pairs"] += 1
    ct[v["status"]] += 1
    if v["status"] != "valid":
        return
    ct[v["binary"]] += 1
    for dest, src in [
        ("strict_valid_pairs", "strict_pair_valid"),
        ("types_changed", "types_changed"),
        ("both_gates_positive", "both_gates_positive"),
    ]:
        ct[dest] += int(v[src])
    ct["unknown_type_pairs"] += int(not v["known_type_gold"])
    if not v["known_type_gold"]:
        return
    ct["known_type_pairs"] += 1
    ct["type_" + v["type_exact"]] += 1
    ct["category_" + v["type_categories"][1] + "->" + v["type_categories"][0]] += 1
    for k in (
        "correct_labels_added",
        "correct_labels_lost",
        "extra_labels_added",
        "extra_labels_removed",
    ):
        ct[k] += v[k]
    if v["both_gates_positive"]:
        ct["both_positive_type_" + v["type_exact"]] += 1


def same_counts(expected, actual, label):
    a.need(
        all(
            expected.get(k, 0) == actual.get(k, 0) for k in set(expected) | set(actual)
        ),
        label,
    )


def main():
    done = a.io.read(a.OUT / "analysis_complete.json")
    a.io.verify(a.io.ROOT, done["artifacts"])
    a.io.verify(a.io.ROOT, done["bindings"])
    a.need(
        not (a.OUT / "independent_readback.json").exists(), "Preserve previous readback"
    )
    roster, science, predictions, meta = a.load()
    specs = a.specifications(science)
    summary = a.io.read(a.OUT / "transition_summary.json")
    pairwise = a.io.read(a.OUT / "pairwise_valid_summary.json")
    counts = defaultdict(Counter)
    keys = defaultdict(set)
    pw = defaultdict(Counter)
    text_directions = defaultdict(Counter)
    selected = {}
    for cohort, exclude in [
        ("full", set()),
        ("same_six_excluded", set(science["mandatory_excluded_keys"])),
    ]:
        keep = [r for r in roster if r["key"] not in exclude]
        clusters = defaultdict(list)
        for r in keep:
            clusters[r["gold"]["hate"], r["cluster"]].append(r["key"])
        representatives = {
            min(
                v,
                key=lambda k: hashlib.sha256(
                    ("A15-representative|" + k).encode()
                ).hexdigest(),
            )
            for v in clusters.values()
        }
        for mask in ("ordinary_valid", "all_three_votes", "one_per_cluster"):
            selected[cohort + "/" + mask] = (
                representatives
                if mask == "one_per_cluster"
                else {r["key"] for r in keep}
            )
    pairs = iter(a.rows(a.OUT / "transition_pairs.jsonl.gz"))
    texts = iter(a.rows(a.OUT / "per_text_transitions.jsonl.gz"))
    pair_count = text_count = 0
    for index, s in enumerate(specs, 1):
        for r in roster:
            key = r["key"]
            ep = "FNR" if r["gold"]["hate"] else "FPR"
            arms = [
                [predictions[key, c, rep] for c in conditions for rep in range(3)]
                for conditions in (s["treatment"], s["control"])
            ]
            all_valid = all(p["result"] is not None for arm in arms for p in arm)
            all_strict = all_valid and all(
                p["n_valid"] == 3 for arm in arms for p in arm
            )
            bits = sum(
                1 << i
                for i, mask in enumerate(a.MASK_NAMES)
                if key in selected[mask]
                and (all_strict if mask.endswith("all_three_votes") else all_valid)
            )
            ct = Counter()
            pools = []
            for t, c in s["pairs"]:
                pool = Counter()
                for rep in range(3):
                    saved = next(pairs)
                    pair_count += 1
                    tp, cp = predictions[key, t, rep], predictions[key, c, rep]
                    expected = classify(r["gold"], tp, cp)
                    a.need(
                        [
                            saved[k]
                            for k in (
                                "contrast",
                                "key",
                                "treatment",
                                "control",
                                "repeat",
                            )
                        ]
                        == [s["id"], key, t, c, rep],
                        "Pair identity/order differs",
                    )
                    a.need(
                        saved["endpoint"] == ep and saved["common_mask_bits"] == bits,
                        "Pair eligibility differs",
                    )
                    a.need(
                        saved["same_messages"]
                        == (
                            meta[key, t]["messages_sha256"]
                            == meta[key, c]["messages_sha256"]
                        ),
                        "Pair message identity differs",
                    )
                    a.need(
                        saved["stages"] == [tp["stage"], cp["stage"]]
                        and saved["valid_votes"] == [tp["n_valid"], cp["n_valid"]],
                        "Pair provenance differs",
                    )
                    a.need(
                        all(saved[k] == v for k, v in expected.items()),
                        "Independent classification differs",
                    )
                    a.need(
                        set(saved)
                        == set(expected)
                        | {
                            "contrast",
                            "key",
                            "endpoint",
                            "treatment",
                            "control",
                            "repeat",
                            "common_mask_bits",
                            "same_messages",
                            "stages",
                            "valid_votes",
                        },
                        "Unexpected/missing ledger fields",
                    )
                    update(ct, expected)
                    update(pool, expected)
                    update(pw[s["id"] + "/" + ep], expected)
                    for i, mask in enumerate(a.MASK_NAMES):
                        if bits & (1 << i):
                            update(counts[mask + "/" + s["id"] + "/" + ep], expected)
                            keys[mask + "/" + s["id"] + "/" + ep].add(key)
                pools.append({"treatment": t, "control": c, "counts": dict(pool)})
            st = next(texts)
            text_count += 1
            a.need(
                st["key"] == key
                and st["contrast"] == s["id"]
                and st["common_mask_bits"] == bits,
                "Per-text identity differs",
            )
            same_counts(ct, st["counts"], "Per-text counts differ")
            a.need(pools == st["pools"], "Per-pool counts differ")
            a.need(
                st["actual_outputs_per_arm"] == [len(v) for v in arms],
                "Real reference output count differs",
            )
            if all_valid:
                actual_errors = [
                    sum(p["result"]["hate"] != r["gold"]["hate"] for p in arm)
                    for arm in arms
                ]
                delta = Fraction(actual_errors[0], len(arms[0])) - Fraction(
                    actual_errors[1], len(arms[1])
                )
                direction = (
                    "fewer_errors"
                    if delta < 0
                    else "more_errors" if delta > 0 else "same_average_errors"
                )
                a.need(
                    st["actual_errors_per_arm"] == actual_errors
                    and st["binary_error_difference_fraction"] == str(delta)
                    and st["direction"] == direction,
                    "Exact per-text mean differs",
                )
                for i, mask in enumerate(a.MASK_NAMES):
                    if bits & (1 << i):
                        d = text_directions[mask + "/" + s["id"] + "/" + ep]
                        d[direction] += 1
                        d["any_binary_changes"] += ct["fix"] + ct["regression"] > 0
                        d["offsetting_binary_changes"] += (
                            ct["fix"] > 0 and ct["regression"] > 0
                        )
                        d["equal_average_with_changes"] += (
                            delta == 0 and ct["fix"] + ct["regression"] > 0
                        )
                        d["all_pairs_unchanged_error"] += (
                            ct["unchanged_error"] == ct["valid"]
                        )
        if index % 16 == 0:
            print(f"Independent ledger readback: {index}/64 comparisons", flush=True)
    a.need(
        next(pairs, None) is None and next(texts, None) is None, "Extra ledger records"
    )
    derived = {
        "binary_error_difference",
        "type_exact_difference",
        "unique_texts",
        "common_ids",
        "per_text_direction",
    }
    for name, ct in counts.items():
        s = summary[name]
        same_counts(
            ct,
            {k: v for k, v in s.items() if k not in derived},
            "Aggregate count differs " + name,
        )
        a.need(
            keys[name] == set(s["common_ids"]) and len(keys[name]) == s["unique_texts"],
            "Aggregate keys differ",
        )
        same_counts(
            text_directions[name],
            s["per_text_direction"],
            "Direction aggregate differs",
        )
        delta = Fraction(ct["regression"] - ct["fix"], ct["valid"])
        a.need(
            float(delta) == s["binary_error_difference"],
            "Binary exact aggregate differs",
        )
        if ct.get("known_type_pairs"):
            exact = Fraction(
                ct["type_fix"] - ct["type_regression"], ct["known_type_pairs"]
            )
            a.need(
                float(exact) == s["type_exact_difference"],
                "Exact-type aggregate differs",
            )
    for name, ct in pw.items():
        same_counts(
            ct,
            {k: v for k, v in pairwise[name].items() if k not in derived},
            "Pairwise-valid counts differ",
        )
    catalog = a.io.read(a.OUT / "example_catalog.json")
    context_count = 0
    variant_ids = set()
    for row in a.rows(a.OUT / "contexts.jsonl.gz"):
        m = meta[row["key"], row["condition"]]
        a.need(all(row[k] == v for k, v in m.items()), "Context metadata changed")
        a.need(
            len(row["example_order"])
            == len(row["example_meta_sha256"])
            == m["example_count"],
            "Context order/hash count differs",
        )
        for eid, h in zip(row["example_order"], row["example_meta_sha256"]):
            a.need(
                eid in catalog and h in catalog[eid]["metadata_variants"],
                "Unknown displayed metadata variant",
            )
            variant_ids.add((eid, h))
        context_count += 1
    a.need(
        context_count == 49950
        and pair_count == 384750
        and text_count == 43200
        and len(counts) == 768,
        "Wrong readback coverage",
    )
    a.io.verify(a.io.ROOT, done["artifacts"])
    a.io.verify(a.io.ROOT, done["bindings"])
    a.write(
        a.OUT / "independent_readback.json",
        {
            "status": "all_pair_and_text_transitions_independently_reconciled",
            "created_utc": a.io.utc(),
            "pairs_checked": pair_count,
            "text_records_checked": text_count,
            "mask_endpoint_summaries_checked": len(counts),
            "pairwise_summaries_checked": len(pw),
            "context_rows_checked": context_count,
            "displayed_metadata_variants_linked": len(variant_ids),
            "method": "Independent truth-table and membership arithmetic; exact Fraction per-text means; every written occurrence checked against saved predictions, masks, provenance and context identity. No transition/category/update implementation reused.",
            "bindings": {
                a.io.rel(Path(__file__)): a.io.sha(Path(__file__)),
                a.io.rel(a.OUT / "analysis_complete.json"): a.io.sha(
                    a.OUT / "analysis_complete.json"
                ),
            },
            "new_model_calls": 0,
        },
    )
    print(
        "Independent readback passed for every saved paired occurrence and every per-text record.",
        flush=True,
    )


if __name__ == "__main__":
    with a.io.offline():
        main()
