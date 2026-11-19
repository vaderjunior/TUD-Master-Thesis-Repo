from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import io as textio
import json
from pathlib import Path
from analysis.feedback.reserved import quality as prior
from analysis.feedback.reserved import transitions_metrics as core
from analysis.feedback.reserved import transitions_catalog as catalog_io

io, need, support, rates, reader = (
    prior.io,
    prior.need,
    prior.support,
    prior.rates,
    prior.reader,
)
P4 = prior.previous
OUT = io.ROOT / "experiments/astra/sq3_reserved_part6_results_20261020_v2"
PREP = io.ROOT / "experiments/astra/sq3_reserved_part6_preparation_20261020_v1"
MASK_NAMES = [
    c + "/" + m
    for c in ("full", "same_six_excluded")
    for m in ("ordinary_valid", "all_three_votes", "one_per_cluster")
]
MAIN = "supporting/reserved5__error_balanced-reference5"
SELECTION = (
    io.ROOT / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection"
)


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 6")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(
            value
            if isinstance(value, str)
            else json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        )


def gzopen(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return textio.TextIOWrapper(
        gzip.GzipFile(filename=str(path), mode="xb", mtime=0, compresslevel=5),
        encoding="utf-8",
        newline="\n",
    )


def emit(f, value):
    f.write(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def specifications(science):
    result = []
    for s in (
        support.supporting_specs(science)
        + rates.contrasts(science)[2:]
        + P4.direct_specs(science)
    ):
        if s["endpoint"] != "FPR":
            continue
        result.append(
            {
                "id": s["id"].rsplit("/", 1)[0],
                "treatment": s["treatment_conditions"],
                "control": s["control_conditions"],
                "origin": s["family"],
                "pairs": core.condition_pairs(
                    s["treatment_conditions"], s["control_conditions"]
                ),
            }
        )
    need(
        len(result) == 64 and len({s["id"] for s in result}) == 64,
        "Wrong comparison coverage",
    )
    return result


def load():
    done = io.read(prior.OUT / "complete_v2.json")
    io.verify(io.ROOT, done["artifacts"])
    io.verify(io.ROOT, done["report_binding"])
    roster, science, predictions, assembly = prior.load()
    meta = {
        (r["key"], r["condition"]): r for r in io.rows(P4.OUT / "prompt_metadata.jsonl")
    }
    need(len(meta) == 49950, "Incomplete existing context table")
    for (key, c, rep), p in predictions.items():
        need(
            p["messages_sha256"] == meta[key, c]["messages_sha256"],
            "Prediction/context link differs",
        )
    return roster, science, predictions, meta


def category_visible(meta):
    gates = [d["meta"]["gate"] for d in meta["delivered_annotations"]]
    if not gates:
        return "none"
    if all(g is None for g in gates):
        return "hidden"
    return (
        "hateful"
        if all(g is True for g in gates)
        else "benign" if all(g is False for g in gates) else "mixed"
    )


def contextualize(meta):

    source_records = io.read(SELECTION / "gate_only_records.json")
    statuses = io.read(SELECTION / "candidate_statuses.json")
    source_ids = {r["id"]: sid for sid, r in source_records.items()}
    catalog = {}
    checked = 0
    with gzopen(OUT / "contexts.jsonl.gz") as f:
        for p in io.rows(io.STUDY / "prompts.jsonl"):
            key, c = p["key"], p["condition"]
            m = meta[key, c]
            es = p["context"]["example_groups"][0][1]
            need(
                p["messages_sha256"] == m["messages_sha256"],
                "Frozen message link differs",
            )
            ordinary = [e["id"] for e in es if e["id"] not in source_ids]
            delivered = [source_ids[e["id"]] for e in es if e["id"] in source_ids]
            need(
                ordinary == m["ordinary_ids"]
                and delivered == m["delivered_correction_ids"],
                "Frozen example membership differs",
            )
            metadata_hashes = []
            for e in es:
                canonical = (
                    source_records[source_ids[e["id"]]] if e["id"] in source_ids else e
                )
                metadata_hashes.append(catalog_io.retain(catalog, e))
                need(
                    e["text"] == canonical["text"], "Displayed correction text changed"
                )
                if e["id"] in source_ids:
                    shown = [
                        d for d in m["delivered_annotations"] if d["id"] == e["id"]
                    ]
                    need(
                        len(shown) == 1 and shown[0]["meta"] == e["meta"],
                        "Delivered visible label differs",
                    )
            row = {
                **m,
                "visible_correction_label": category_visible(m),
                "example_order": [e["id"] for e in es],
                "example_meta_sha256": metadata_hashes,
                "selected_previous_status": (
                    statuses[m["source_id"]]["status"] if m["source_id"] else None
                ),
                "selected_screening_decisions": (
                    statuses[m["source_id"]]["decisions"] if m["source_id"] else None
                ),
            }
            emit(f, row)
            checked += 1
    need(checked == 49950, "Incomplete context normalisation")
    write(OUT / "example_catalog.json", catalog)
    return {
        "frozen_prompts_read": checked,
        "unique_examples": len(catalog),
        "metadata_variants": sum(len(v["metadata_variants"]) for v in catalog.values()),
        "examples_with_metadata_variants": sum(
            len(v["metadata_variants"]) > 1 for v in catalog.values()
        ),
        "selected_source_annotations": "Original binary source labels; actual displayed labels remain in each context row.",
    }


def finished(bucket):
    return {**core.finish(bucket["counts"]), "unique_texts": len(bucket["keys"])}


def new_bucket():
    return {"counts": Counter(), "keys": set()}


def add(bucket, key, value):
    bucket["keys"].add(key)
    core.update(bucket["counts"], value)


def mask_sets(roster, science):
    return {
        name: {
            r["key"]
            for r in support.selected_roster(
                roster,
                (
                    science["mandatory_excluded_keys"]
                    if name.startswith("same_six")
                    else []
                ),
                representatives=name.endswith("one_per_cluster"),
            )
        }
        for name in MASK_NAMES
    }


def strata(row, tm, cm):
    cosine = tm["selected_correction_cosine"]
    return {
        "message": (
            "identical" if tm["messages_sha256"] == cm["messages_sha256"] else "changed"
        ),
        "source": row["source"],
        "visible_label": category_visible(tm),
        "selected_matches_gold": (
            "none"
            if tm["source_gate"] is None
            else str(tm["source_gate"] == row["gold"]["hate"])
        ),
        "selected_delivery": (
            "none_selected"
            if tm["source_id"] is None
            else (
                "selected"
                if tm["source_id"] in tm["delivered_correction_ids"]
                else "other_only" if tm["delivered_correction_ids"] else "none"
            )
        ),
        "removed_fifth_gate": (
            "not_removed"
            if not tm["removed_fifth"]
            else str(tm["removed_fifth"]["meta"].get("gate"))
        ),
        "removed_fifth_known_type": str(
            bool(tm["removed_fifth"])
            and tm["removed_fifth"]["meta"].get("hate_types") is not None
        ),
        "cosine": (
            "none"
            if cosine is None
            else "<0.4" if cosine < 0.4 else "0.4_to_0.6" if cosine < 0.6 else ">=0.6"
        ),
    }


def analyze(roster, science, predictions, meta, specs):
    eligible = mask_sets(roster, science)
    groups = defaultdict(new_bucket)
    subgroups = defaultdict(new_bucket)
    pairwise = defaultdict(new_bucket)
    text_summary = defaultdict(Counter)
    common_ids = defaultdict(list)
    ledger_count = 0
    text_count = 0
    with gzopen(OUT / "transition_pairs.jsonl.gz") as fp, gzopen(
        OUT / "per_text_transitions.jsonl.gz"
    ) as ft:
        for si, s in enumerate(specs, 1):
            sid = s["id"]
            conditions = set(s["treatment"] + s["control"])
            for r in roster:
                key = r["key"]
                ep = "FNR" if r["gold"]["hate"] else "FPR"
                actual = [
                    [predictions[key, c, rep] for c in arm for rep in range(3)]
                    for arm in (s["treatment"], s["control"])
                ]
                ordinary = all(p["result"] is not None for arm in actual for p in arm)
                strict = ordinary and all(
                    p["n_valid"] == 3 for arm in actual for p in arm
                )
                maskbits = sum(
                    1 << i
                    for i, n in enumerate(MASK_NAMES)
                    if key in eligible[n]
                    and (strict if n.endswith("all_three_votes") else ordinary)
                )
                ct = Counter()
                by_pool = []
                identical_count = 0
                for t, c in s["pairs"]:
                    poolct = Counter()
                    tm, cm = meta[key, t], meta[key, c]
                    dims = strata(r, tm, cm)
                    same = dims["message"] == "identical"
                    for rep in range(3):
                        tp, cp = predictions[key, t, rep], predictions[key, c, rep]
                        v = core.compare(r["gold"], tp, cp)
                        core.update(ct, v)
                        core.update(poolct, v)
                        base = {
                            "contrast": sid,
                            "key": key,
                            "endpoint": ep,
                            "treatment": t,
                            "control": c,
                            "repeat": rep,
                            "common_mask_bits": maskbits,
                            "same_messages": same,
                            "stages": [tp["stage"], cp["stage"]],
                            "valid_votes": [tp["n_valid"], cp["n_valid"]],
                            **v,
                        }
                        emit(fp, base)
                        ledger_count += 1
                        identical_count += same
                        add(pairwise[sid + "/" + ep], key, v)
                        for i, mask in enumerate(MASK_NAMES):
                            if maskbits & (1 << i):
                                add(groups[mask + "/" + sid + "/" + ep], key, v)
                        if maskbits & 1:
                            for dimension, value in dims.items():
                                add(
                                    subgroups[
                                        sid + "/" + ep + "/" + dimension + "/" + value
                                    ],
                                    key,
                                    v,
                                )
                    by_pool.append(
                        {"treatment": t, "control": c, "counts": dict(poolct)}
                    )
                out = {
                    "contrast": sid,
                    "key": key,
                    "endpoint": ep,
                    "source": r["source"],
                    "cluster": r["cluster"],
                    "common_mask_bits": maskbits,
                    "counts": dict(ct),
                    "pools": by_pool,
                    "identical_message_pair_occurrences": identical_count,
                    "ordinary_common": ordinary,
                    "actual_outputs_per_arm": [len(x) for x in actual],
                    "actual_nulls_per_arm": [
                        sum(p["result"] is None for p in x) for x in actual
                    ],
                }
                if ordinary:
                    errors = [
                        sum(p["result"]["hate"] != r["gold"]["hate"] for p in arm)
                        for arm in actual
                    ]
                    diff = Fraction(errors[0], len(actual[0])) - Fraction(
                        errors[1], len(actual[1])
                    )
                    need(
                        diff == Fraction(ct["regression"] - ct["fix"], ct["valid"]),
                        "Reference pairing changed the mean",
                    )
                    out.update(
                        actual_errors_per_arm=errors,
                        binary_error_difference_fraction=str(diff),
                        direction=core.direction(ct),
                        offsetting_binary_changes=ct["fix"] > 0
                        and ct["regression"] > 0,
                    )
                    for i, mask in enumerate(MASK_NAMES):
                        if maskbits & (1 << i):
                            common_ids[mask + "/" + sid + "/" + ep].append(key)
                            d = text_summary[mask + "/" + sid + "/" + ep]
                            d[out["direction"]] += 1
                            d["any_binary_changes"] += ct["fix"] + ct["regression"] > 0
                            d["offsetting_binary_changes"] += out[
                                "offsetting_binary_changes"
                            ]
                            d["equal_average_with_changes"] += (
                                out["direction"] == "same_average_errors"
                                and ct["fix"] + ct["regression"] > 0
                            )
                            d["all_pairs_unchanged_error"] += (
                                ct["unchanged_error"] == ct["valid"]
                            )
                emit(ft, out)
                text_count += 1
            if si % 8 == 0:
                print(f"Complete transition packages: {si}/64", flush=True)
    need(
        ledger_count == 384750 and text_count == 43200,
        "Incomplete exhaustive transition matrix",
    )
    result = {
        k: {
            **finished(v),
            "common_ids": common_ids[k],
            "per_text_direction": dict(text_summary[k]),
        }
        for k, v in groups.items()
    }
    write(OUT / "transition_summary.json", result)
    write(OUT / "context_strata.json", {k: finished(v) for k, v in subgroups.items()})
    write(
        OUT / "pairwise_valid_summary.json",
        {k: finished(v) for k, v in pairwise.items()},
    )
    return result, ledger_count, text_count


def reconcile(results, specs):
    p2 = io.read(P4.prior.prior.OUT / "primary_and_sensitivities.json")
    p3 = io.read(P4.prior.OUT / "mechanism_and_sensitivities.json")
    p4 = io.read(P4.OUT / "rate_comparisons.json")
    p4s = io.read(P4.OUT / "point_sensitivities.json")
    p4d = io.read(P4.OUT / "comparison_details.json")
    quality = io.read(prior.OUT / "paired_quality.json")
    sensitivity = io.read(prior.OUT / "point_sensitivities.json")
    checks = []
    direction_checks = 0
    quality_checks = 0
    for mask in MASK_NAMES:
        cohort, validity = mask.split("/")
        for s in specs:
            sid = s["id"]
            for ep in ("FPR", "FNR"):
                name = mask + "/" + sid + "/" + ep
                now = results[name]
                if sid.startswith("mechanism/"):
                    old = p3[name]
                else:
                    old = (
                        p4[cohort + "/" + sid + "/" + ep]
                        if validity == "ordinary_valid"
                        else p4s[name]
                    )
                need(
                    set(now["common_ids"]) == set(old["common_ids"])
                    and now["unique_texts"] == old["common_n"],
                    "Prior common mask differs: " + name,
                )
                need(
                    abs(now["binary_error_difference"] - old["difference"]) < 1e-12,
                    "Prior rate differs: " + name,
                )
                checks.append(
                    {
                        "name": name,
                        "common_n": now["unique_texts"],
                        "difference": now["binary_error_difference"],
                    }
                )
                if sid == MAIN:
                    original = p2[mask + "/primary/reference5/" + ep]
                    need(
                        set(now["common_ids"]) == set(original["common_ids"])
                        and abs(now["binary_error_difference"] - original["difference"])
                        < 1e-12,
                        "Primary alias differs",
                    )
                if not sid.startswith("mechanism/") and validity == "ordinary_valid":
                    olddir = p4d[cohort + "/" + sid + "/" + ep]["per_text_direction"]
                    need(
                        all(
                            now["per_text_direction"].get(k, 0) == olddir.get(k, 0)
                            for k in (
                                "fewer_errors",
                                "more_errors",
                                "same_average_errors",
                            )
                        ),
                        "Prior text directions differ",
                    )
                    direction_checks += 1
            if sid.startswith("part4-direct-v1/"):
                continue
            qname = (
                cohort
                + "/"
                + ("supporting_metrics/" if sid.startswith("mechanism/") else "")
                + sid
                + "/types"
            )
            q = (
                quality[qname]
                if validity == "ordinary_valid"
                else sensitivity[qname + "/" + validity]
            )

            now = results[mask + "/" + sid + "/FNR"]
            need(now["unique_texts"] == q["common_n"], "Type common mask differs")
            need(
                abs(
                    now["type_exact_difference"]
                    - q["metrics"]["type_exact_set"]["difference"]
                )
                < 1e-12,
                "Prior exact-set difference differs",
            )
            quality_checks += 1
    write(
        OUT / "prior_reproduction.json",
        {
            "binary_comparisons": checks,
            "binary_count": len(checks),
            "primary_alias_checks": 12,
            "per_text_direction_checks": direction_checks,
            "exact_set_checks": quality_checks,
            "interpretation": "Reproduced prior common masks and point effects; no new confidence intervals or independent-pair inference.",
        },
    )


def repeat_variation(roster, science, predictions, meta):
    counts = defaultdict(Counter)
    textsets = defaultdict(set)
    nulls = Counter()
    for r in roster:
        ep = "FNR" if r["gold"]["hate"] else "FPR"
        for c in science["conditions"]:
            group = c.rsplit("_s", 1)[0] if "__" in c else c
            for rep1, rep2 in ((0, 1), (0, 2), (1, 2)):
                v = core.compare(
                    r["gold"],
                    predictions[r["key"], c, rep2],
                    predictions[r["key"], c, rep1],
                )
                for name in (group + "/" + ep, "all_conditions/" + ep):
                    core.update(counts[name], v)
                    textsets[name].add(r["key"])
            for rep in range(3):
                if predictions[r["key"], c, rep]["result"] is None:
                    nulls[r["key"]] += 1
    write(
        OUT / "within_condition_repeat_variation.json",
        {
            k: {
                **core.finish(v),
                "unique_texts": len(textsets[k]),
                "interpretation": "Three overlapping repeat pairs per query/condition; repeat index is not a randomized treatment or adaptation round.",
            }
            for k, v in counts.items()
        },
    )
    write(OUT / "null_inputs.json", dict(nulls))


def main():
    need(not OUT.exists(), "Preserve existing Part 6 outputs")
    test = io.read(PREP / "implementation_checks.json")
    io.verify(io.ROOT, test["bindings"])
    need(test["test_count"] == 7, "Missing synthetic checks")
    amendment = io.read(PREP / "context_catalog_amendment_v2.json")
    io.verify(io.ROOT, amendment["preserved_v1_artifacts"])
    catalog_tests = io.read(PREP / "catalog_checks_v2.json")
    io.verify(io.ROOT, catalog_tests["bindings"])
    need(catalog_tests["test_count"] == 4, "Missing catalog regression tests")
    roster, science, predictions, meta = load()
    specs = specifications(science)
    sources = [
        Path(__file__),
        Path(core.__file__),
        Path(catalog_io.__file__),
        PREP / "scope.md",
        PREP / "implementation_checks.json",
        PREP / "catalog_checks_v2.json",
        PREP / "context_catalog_amendment_v2.json",
        prior.OUT / "complete_v2.json",
        P4.OUT / "complete.json",
        reader.OUT / "complete.json",
        io.STUDY / "scientific_plan_v1.json",
        io.STUDY / "roster.jsonl",
        io.STUDY / "prompts.jsonl",
        P4.OUT / "prompt_metadata.jsonl",
        SELECTION / "gate_only_records.json",
        SELECTION / "candidate_statuses.json",
    ]
    bindings = {io.rel(p): io.sha(p) for p in sources}
    write(
        OUT / "started.json",
        {
            "created_utc": io.utc(),
            "bindings": bindings,
            "comparisons": specs,
            "mask_bit_order": MASK_NAMES,
            "new_model_calls": 0,
        },
    )
    context = contextualize(meta)
    print("Frozen context/example index checked.", flush=True)
    results, pair_count, text_count = analyze(roster, science, predictions, meta, specs)
    reconcile(results, specs)
    repeat_variation(roster, science, predictions, meta)
    io.verify(io.ROOT, bindings)
    paths = [p for p in OUT.iterdir() if p.is_file()]
    write(
        OUT / "analysis_complete.json",
        {
            "status": "part6_exhaustive_transitions_checked_cases_and_interpretation_pending",
            "created_utc": io.utc(),
            "bindings": bindings,
            "artifacts": {io.rel(p): io.sha(p) for p in paths},
            "context": context,
            "pair_occurrences": pair_count,
            "text_comparison_records": text_count,
            "comparison_packages": 64,
            "mask_endpoint_summaries": len(results),
            "new_model_calls": 0,
            "new_training_fits": 0,
            "remaining_parts": list(range(7, 11)),
        },
    )
    print(
        "Exhaustive Part 6 ledger complete; independent readback and qualitative cases remain.",
        flush=True,
    )


if __name__ == "__main__":
    with io.offline():
        main()
