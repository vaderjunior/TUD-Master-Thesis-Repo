from collections import Counter, defaultdict
import json
import math
import re
import unicodedata
import numpy as np
from analysis.classification.broad import contexts as x

checks = 0
maximum = 0.0


def equal(a, b, path=""):
    global checks, maximum
    if isinstance(a, dict):
        x.require(isinstance(b, dict) and set(a) == set(b), "Keys differ " + path)
        for k in a:
            equal(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, (list, tuple)):
        x.require(
            isinstance(b, (list, tuple)) and len(a) == len(b), "Array differs " + path
        )
        for i, (aa, bb) in enumerate(zip(a, b)):
            equal(aa, bb, path + "/" + str(i))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        x.require(
            isinstance(b, (int, float)) and not isinstance(b, bool),
            "Number type differs " + path,
        )
        err = abs(float(a) - float(b))
        maximum = max(maximum, err)
        x.require(err <= 1e-10 * max(1, abs(float(a))), "Number differs " + path)
        checks += 1
    else:
        x.require(a == b, "Value differs " + path)
        checks += 1


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def avg(v):
    return sum(v) / len(v) if v else None


def hist(v):
    return dict(Counter(map(str, v)))


def ann(hits):
    regular = [h["meta"] for h in hits if not h["meta"].get("illustrative_only")]
    out = {
        k: 0
        for k in (
            "regular",
            "illustrative",
            "hate",
            "nonhate",
            "gate_unknown",
            "auxiliary_flags",
            "all_four_fine_known",
        )
    }
    out["regular"] = len(regular)
    out["illustrative"] = len(hits) - len(regular)
    for m in regular:
        out[
            (
                "gate_unknown"
                if m.get("gate") is None
                else "hate" if m["gate"] else "nonhate"
            )
        ] += 1
        out["auxiliary_flags"] += int(bool(m.get("astra_aux_stgb")))
        out["all_four_fine_known"] += int(
            all(m.get(f) is not None for f in x.FIELDS.values())
        )
    for d, f in x.FIELDS.items():
        known = empty = positive = 0
        labels = Counter()
        for m in regular:
            v = m.get(f)
            if v is not None:
                known += 1
                empty += int(v == [])
                ls = [v] if f == "severity" else v
                positive += int(bool(ls))
                labels.update(ls)
        out[d] = dict(known=known, empty=empty, positive=positive, labels=dict(labels))
    return out


def run():
    receipt = x.c.read(x.BASE / "analysis_complete.json")
    x.r.verify(receipt["sources"])
    x.r.verify(receipt["outputs"])
    ds = {co: {} for co in x.CONDITIONS}
    for d in x.reader.rows(x.BASE / "prompt_descriptors.jsonl"):
        ds[d["condition"]][d["key"]] = d
    membership = x.c.read(x.BASE / "group_membership.json")
    groups, _ = x.reader.scoring.scientific_rosters("broad")
    queries = {q["key"]: q for rr in groups.values() for q in rr}
    kb = {
        co: {v["id"]: v for v in x.reader.rows(x.CORE / name / "records.jsonl")}
        for co, name in [("rag_full", "full"), ("rag_small", "small")]
    }
    small = set(kb["rag_small"])
    raw_ids = {co: {} for co in x.CONDITIONS}
    raw_ann = {co: {} for co in x.CONDITIONS}
    lengths = {co: {} for co in x.CONDITIONS}
    for info in x.c.read(x.reader.BASE / "prompt_setup.json")["conditions"]:
        co = info["condition"]
        bt = defaultdict(list)
        for ident, v in kb.get(co, {}).items():
            if v["kind"] == "example":
                bt[norm(v["text"])].append(ident)
        for row in x.reader.rows(x.ROOT / info["path"]):
            k = row["key"]
            d = ds[co][k]
            ctx = row.get(
                "context", dict(definitions=[], guidelines=[], example_groups=[])
            )
            ex = [h for _, hh in ctx["example_groups"] for h in hh]
            ids = {
                "definitions": [v["id"] for v in ctx["definitions"]],
                "guidelines": [v["id"] for v in ctx["guidelines"]],
                "examples": [v["id"] for v in ex],
            }
            aa = ann(ex)
            equal(ids, d["ids"], k + "/ids")
            equal(aa, d["annotations"], k + "/annotations")
            raw_ids[co][k] = ids
            raw_ann[co][k] = aa
            equal(
                sum(len(m["text"]) for m in row["messages"]),
                d["message_chars"],
                k + "/chars",
            )
            equal(
                sum(len(m["text"].encode("utf8")) for m in row["messages"]),
                d["message_utf8_bytes"],
                k + "/bytes",
            )
            equal(
                sum(h["id"] in small for h in ex), d["examples_in_small"], k + "/small"
            )
            equal(
                sum(
                    h["meta"]["source"].removesuffix("-train") == queries[k]["source"]
                    for h in ex
                ),
                d["same_source_examples"],
                k + "/source",
            )
            equal(
                dict(Counter(h["meta"]["source"] for h in ex)),
                d["sources"],
                k + "/sources",
            )
            equal(
                [h["id"] for h in ex if norm(h["text"]) == norm(queries[k]["text"])],
                d["exact_query_examples"],
                k + "/exact",
            )
            equal(
                bt.get(norm(queries[k]["text"]), []),
                d["exact_candidate_examples"],
                k + "/candidates",
            )
            qa = set(re.findall(r"\w+", norm(queries[k]["text"])))
            sim = []
            for h in ex:
                ha = set(re.findall(r"\w+", norm(h["text"])))
                sim.append(len(qa & ha) / len(qa | ha) if qa | ha else 0.0)
            equal(max(sim, default=0.0), d["max_lexical_jaccard"], k + "/lexical")
            equal(avg(sim), d["mean_lexical_jaccard"], k + "/lexical_mean")
    print("Independent raw-prompt descriptors passed.", flush=True)
    delivery = x.c.read(x.BASE / "delivery_summary.json")
    overlap = x.c.read(x.BASE / "overlap_summary.json")
    for group, keys in membership.items():
        equal(len(keys), overlap[group]["n"], group + "/n")
        for ch in ("definitions", "guidelines", "examples"):
            aa = [raw_ids["rag_full"][k][ch] for k in keys]
            bb = [raw_ids["rag_small"][k][ch] for k in keys]
            old = overlap[group]["channels"][ch]
            equal(
                hist([len(set(a) & set(b)) for a, b in zip(aa, bb)]),
                old["shared_hist"],
                group + "/" + ch + "/overlap",
            )
            equal(
                sum(a == b for a, b in zip(aa, bb)),
                old["same_order"],
                group + "/" + ch + "/order",
            )
            equal(
                sum(set(a) == set(b) for a, b in zip(aa, bb)),
                old["same_set"],
                group + "/" + ch + "/sets",
            )
            equal(
                hist([sum(x == y for x, y in zip(a, b)) for a, b in zip(aa, bb)]),
                old["same_positions_hist"],
                group + "/" + ch + "/positions",
            )
        for co in x.CONDITIONS:
            old = delivery[group][co]
            aa = [raw_ann[co][k] for k in keys]
            dd = [ds[co][k] for k in keys]
            equal(len(keys), old["n"], group + "/" + co + "/n")
            for field in (
                "hate",
                "nonhate",
                "gate_unknown",
                "illustrative",
                "auxiliary_flags",
                "all_four_fine_known",
            ):
                equal(
                    hist([a[field] for a in aa]),
                    old["annotation_histograms"][field],
                    group + "/" + co + "/" + field,
                )
            for dim in x.FIELDS:
                for field in ("known", "empty", "positive"):
                    equal(
                        hist([a[dim][field] for a in aa]),
                        old["field_coverage"][dim][field + "_hist"],
                        group + "/" + co + "/" + dim + "/" + field,
                    )
                ct = Counter()
                for a in aa:
                    ct.update(a[dim]["labels"])
                equal(
                    dict(ct),
                    old["field_coverage"][dim]["label_slots"],
                    group + "/" + co + "/" + dim + "/labels",
                )
            frequency = Counter()
            for k in keys:
                frequency.update(raw_ids[co][k]["examples"])
            equal(
                len(frequency),
                old["unique_hits"]["examples"],
                group + "/" + co + "/unique",
            )
            equal(
                sum(frequency.values()),
                old["example_slots"],
                group + "/" + co + "/slots",
            )
            top = frequency.most_common(20)
            equal(
                [
                    dict(id=i, count=n, source=kb["rag_full"][i]["source"])
                    for i, n in top
                ],
                old["top20"],
                group + "/" + co + "/top",
            )
            total = sum(frequency.values())
            equal(
                sum(n for _, n in top[:10]) / total if total else None,
                old["top10_share"],
                group + "/" + co + "/concentration",
            )
            for metric in (
                "message_chars",
                "message_utf8_bytes",
                "context_chars",
                "max_lexical_jaccard",
            ):
                vals = sorted(d[metric] for d in dd)
                n = len(vals)
                median = (
                    vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
                )
                equal(
                    dict(
                        n=n,
                        minimum=vals[0],
                        maximum=vals[-1],
                        mean=avg(vals),
                        median=float(median),
                    ),
                    old["lengths"][metric],
                    group + "/" + co + "/" + metric,
                )
    print("Independent delivery and overlap summaries passed.", flush=True)
    outcomes = x.c.read(x.BASE / "outcome_associations.json")
    metadata = x.c.read(x.PRIOR / "endpoint_metadata.json")
    for name, md in metadata.items():
        with np.load(x.ROOT / md["path"], allow_pickle=False) as z:
            y = z["gold"]
            p = z["predicted"]
            v = z["valid"]
            gate = z["gate"]
        dim = name.split("/")[1]
        indices = [i for i in range(len(y)) if all(v[:, i])]
        equal(len(indices), outcomes[name]["n_common_nine"], name + "/valid")
        for j, co in enumerate(x.CONDITIONS[:2]):
            pp = p[3 * j : 3 * j + 3]
            gg = gate[3 * j : 3 * j + 3]
            old = outcomes[name]["conditions"][co]
            if dim == "hate":
                expected = []
                for n in range(6):
                    ix = [
                        i for i in indices if raw_ann[co][md["ids"][i]]["nonhate"] == n
                    ]
                    pos = [i for i in ix if y[i, 1]]
                    neg = [i for i in ix if not y[i, 1]]
                    expected.append(
                        dict(
                            nonhate_examples=n,
                            n=len(ix),
                            gold_hate=len(pos),
                            gold_nonhate=len(neg),
                            hate_rate=avg([sum(pp[:, i, 1]) / 3 for i in ix]),
                            fpr=avg([sum(pp[:, i, 1]) / 3 for i in neg]),
                            fnr=avg([sum(1 - pp[:, i, 1]) / 3 for i in pos]),
                        )
                    )
                equal(expected, old, name + "/" + co + "/binary")
            else:
                bins = defaultdict(list)
                for i in indices:
                    a = raw_ann[co][md["ids"][i]][dim]
                    true = {l for n, l in enumerate(md["labels"]) if y[i, n]}
                    found = true & set(a["labels"])
                    level = (
                        "empty_gold"
                        if not true
                        else (
                            "none"
                            if not found
                            else "all" if found == true else "partial"
                        )
                    )
                    bins["annotated_examples", str(a["known"])].append(i)
                    bins["gold_label_coverage", level].append(i)
                expected = []
                for family, labels in [
                    ("annotated_examples", list(map(str, range(6)))),
                    ("gold_label_coverage", ["empty_gold", "none", "partial", "all"]),
                ]:
                    for label in labels:
                        ix = bins[family, label]
                        exact = [
                            sum(
                                bool(np.array_equal(pp[t, i], y[i]))
                                and (dim == "legal" or bool(gg[t, i]))
                                for t in range(3)
                            )
                            / 3
                            for i in ix
                        ]
                        expected.append(
                            dict(
                                family=family,
                                stratum=label,
                                n=len(ix),
                                gate_aware_exact=avg(exact),
                                closed_gate=avg(
                                    [sum(not bool(a) for a in gg[:, i]) / 3 for i in ix]
                                ),
                            )
                        )
                equal(expected, old["strata"], name + "/" + co + "/fine")
                lr = []
                for n, label in enumerate(md["labels"]):
                    for shown in (False, True):
                        ix = [
                            i
                            for i in indices
                            if y[i, n]
                            and (
                                (label in raw_ann[co][md["ids"][i]][dim]["labels"])
                                == shown
                            )
                        ]
                        lr.append(
                            dict(
                                label=label,
                                gold_label_shown=shown,
                                n=len(ix),
                                recall=avg([sum(pp[:, i, n]) / 3 for i in ix]),
                            )
                        )
                equal(lr, old["labels"], name + "/" + co + "/label")
        if dim == "hate":
            for old in outcomes[name]["stable_categories"]:
                chosen = []
                for i in indices:
                    if bool(y[i, 1]) != old["gold_hate"] or (
                        old["source"] != "ALL"
                        and md["rows"][i]["source"] != old["source"]
                    ):
                        continue
                    a = [bool(p[t, i, 1]) == bool(y[i, 1]) for t in range(6)]
                    kind = (
                        "stable_correct"
                        if all(a)
                        else (
                            "stable_wrong"
                            if not any(a)
                            else (
                                "stable_fix"
                                if all(a[:3]) and not any(a[3:])
                                else (
                                    "stable_regression"
                                    if not any(a[:3]) and all(a[3:])
                                    else "variable"
                                )
                            )
                        )
                    )
                    if kind == old["category"]:
                        chosen.append(i)
                equal(len(chosen), old["n"], name + "/stable_count")
                for co in x.CONDITIONS[:2]:
                    dd = [ds[co][md["ids"][i]] for i in chosen]
                    expected = dict(
                        nonhate_examples=avg([d["annotations"]["nonhate"] for d in dd]),
                        same_source_examples=avg(
                            [d["same_source_examples"] for d in dd]
                        ),
                        context_chars=avg([d["context_chars"] for d in dd]),
                        max_lexical_jaccard=avg([d["max_lexical_jaccard"] for d in dd]),
                    )
                    equal(expected, old["conditions"][co], name + "/stable_context")
                shared = [
                    len(
                        set(raw_ids["rag_full"][md["ids"][i]]["examples"])
                        & set(raw_ids["rag_small"][md["ids"][i]]["examples"])
                    )
                    for i in chosen
                ]
                equal(avg(shared), old["shared_examples"], name + "/stable_overlap")
    print("Independent outcome associations passed.", flush=True)
    case_records = x.c.read(x.BASE / "case_contexts.json")
    selection = x.c.read(x.BASE / "case_selection.json")
    equal(sorted(v["key"] for v in case_records), selection["keys"], "case population")
    for case in case_records:
        for pred in case["predictions"]:
            equal(
                pred["messages_sha256"],
                case["prompts"][pred["condition"]]["messages_sha256"],
                "case/prediction binding",
            )
    x.save(
        x.BASE / "independent_verification.json",
        dict(
            status="independent_raw_context_counts_overlaps_lengths_and_all_outcome_associations_passed",
            checks=checks,
            maximum_numeric_difference=maximum,
            inputs=x.bindings(
                [
                    x.BASE / "analysis_complete.json",
                    x.ROOT / "analysis/classification/broad/validate_contexts.py",
                ]
            ),
            case_inputs=len(case_records),
            new_calls=0,
        ),
    )
    print(
        "Independent verification passed:",
        checks,
        "values; max numerical difference",
        maximum,
        flush=True,
    )


if __name__ == "__main__":
    with x.r.original.offline():
        run()
