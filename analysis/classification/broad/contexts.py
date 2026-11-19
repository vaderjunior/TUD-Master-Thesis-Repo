import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from analysis.classification.broad import reader as reader

c = reader.c
r = reader.r
ROOT = reader.ROOT
BASE = ROOT / "experiments/astra/sq2_broad_context_review_20261104_v1"
PRIOR = ROOT / "experiments/astra/sq2_broad_sources_review_20261104_v1"
RETRIEVAL = ROOT / "experiments/astra/later_evaluation_20261008_v1/retrieval_v3"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1/kb"
MATCHED = ROOT / "experiments/astra/sq2_matched_review_20261030_v1"
CONDITIONS = ("rag_full", "rag_small", "zero_shot")
FIELDS = {
    "hate_type": "hate_types",
    "target_group": "target_groups",
    "severity": "severity",
    "legal": "legal",
}
CASES = [
    ROOT / "experiments/astra" / folder / "cases.json"
    for folder in (
        "sq2_broad_errors_review_20261104_v1",
        "sq2_broad_quality_review_20261104_v1",
        "sq2_broad_sources_review_20261104_v1",
    )
]
require = reader.require
bindings = reader.bindings


def safe(v):
    if isinstance(v, np.ndarray):
        return safe(v.tolist())
    if isinstance(v, np.generic):
        return safe(v.item())
    if isinstance(v, dict):
        return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [safe(x) for x in v]
    return v


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()), "Output outside Part 6")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(c.encoded(safe(value)))


def lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(path.resolve().is_relative_to(BASE.resolve()), "Output outside Part 6")
    with path.open("x", encoding="utf8", newline="\n") as f:
        for v in values:
            f.write(
                json.dumps(safe(v), ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def normalized(v):
    return reader.gold.normalized(v)


def tokens(v):
    return set(re.findall("\\w+", normalized(v)))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def mean(v):
    return float(np.mean(v)) if len(v) else None


def hist(v):
    return dict(sorted(Counter((str(x) for x in v)).items()))


def stats(v):
    return dict(
        n=len(v),
        mean=mean(v),
        minimum=min(v) if v else None,
        median=float(np.median(v)) if v else None,
        maximum=max(v) if v else None,
    )


def meta(record):
    return {
        k: record[k] for k in ("kind", "lang", "source", "dimension", "label")
    } | record.get("meta", {})


def prepare_renderer():
    m = reader.prompt.module("corrected")
    template = m._load_template()
    labels = m.render_label_space()
    m._load_template = lambda: template
    m.render_label_space = lambda tax=None: (
        labels
        if tax is None
        else reader.prompt.module("legacy").render_label_space(tax)
    )


def examples(context):
    return [v for _, vv in context["example_groups"] for v in vv]


def channels(context):
    return dict(
        definitions=context["definitions"],
        guidelines=context["guidelines"],
        examples=examples(context),
    )


def values(m, field):
    v = m.get(field)
    return [] if v is None else [v] if field == "severity" else v


def annotations(hits):
    mm = [h["meta"] for h in hits if not h["meta"].get("illustrative_only")]
    out = dict(
        regular=len(mm),
        illustrative=len(hits) - len(mm),
        hate=sum((m.get("gate") is True for m in mm)),
        nonhate=sum((m.get("gate") is False for m in mm)),
        gate_unknown=sum((m.get("gate") is None for m in mm)),
        auxiliary_flags=sum((bool(m.get("astra_aux_stgb")) for m in mm)),
    )
    for dim, field in FIELDS.items():
        out[dim] = dict(
            known=sum((m.get(field) is not None for m in mm)),
            empty=sum((m.get(field) == [] for m in mm)),
            positive=sum((bool(values(m, field)) for m in mm)),
            labels=dict(Counter((x for m in mm for x in values(m, field)))),
        )
    out["all_four_fine_known"] = sum(
        (all((m.get(f) is not None for f in FIELDS.values())) for m in mm)
    )
    return out


def overlap(a, b):
    out = {}
    for ch in ("definitions", "guidelines", "examples"):
        aa = a["ids"][ch]
        bb = b["ids"][ch]
        out[ch] = dict(
            shared=len(set(aa) & set(bb)),
            same_order=aa == bb,
            same_set=set(aa) == set(bb),
            same_positions=sum((x == y for x, y in zip(aa, bb))),
            same_rendered=a["rendered"][ch] == b["rendered"][ch],
        )
    out["same_message"] = a["messages_sha256"] == b["messages_sha256"]
    out["same_delivered_context"] = all(
        (
            v["same_rendered"]
            for k, v in out.items()
            if k in ("definitions", "guidelines", "examples")
        )
    )
    return out


def prior_cases():
    slots = []
    members = {}
    for p in CASES:
        for case in c.read(p):
            mm = (
                case.get("members", [])
                if "members" in case
                else [case] if case.get("key") else []
            )
            slots.append(
                {
                    k: v
                    for k, v in case.items()
                    if k not in ("members", "predictions", "gold")
                }
                | {"source": c.rel(p), "member_keys": [m["key"] for m in mm]}
            )
            for m in mm:
                key = m["key"]
                rec = members.setdefault(
                    key,
                    dict(
                        key=key,
                        gold=m["gold"],
                        case_ids=[],
                        predictions=m["predictions"],
                    ),
                )
                require(rec["gold"] == m["gold"], "Prior case gold disagreement")
                rec["case_ids"].append(case["case_id"])
    return (slots, members)


def freeze():
    metadata = c.read(PRIOR / "endpoint_metadata.json")
    paths = (
        [
            Path(__file__),
            ROOT / "tests/classification/broad/test_contexts.py",
            BASE / "tests.json",
            reader.PLAN,
            reader.BASE / "part1_complete.json",
            reader.BASE / "prompt_setup.json",
            reader.BASE / "predictions.jsonl",
            PRIOR / "part5_complete.json",
            PRIOR / "final_review.json",
            PRIOR / "endpoint_metadata.json",
            RETRIEVAL / "plan.json",
            RETRIEVAL / "complete.json",
            ROOT
            / "experiments/astra/offline_closure_20261006_v1/evaluation_preparation/broad_roster.jsonl",
            ROOT / "analysis/common/prompt.py",
            ROOT / "src/hsrag/prompt.py",
            ROOT / "prompts/classify_v1.txt",
            ROOT / "config/taxonomy.yaml",
            MATCHED / "context_review.json",
            MATCHED / "context_receipt.json",
            ROOT / "Thesis_Log/astra_result/astra_sq2_matched_results_20261030_v1.md",
            ROOT
            / "Thesis_Log/astra_result/astra_remaining_work_before_writing_20261104_v148.md",
        ]
        + CASES
        + [ROOT / m["path"] for m in metadata.values()]
    )
    paths += [CORE / v / "records.jsonl" for v in ("full", "small")] + [
        ROOT / p["path"]
        for p in c.read(reader.BASE / "prompt_setup.json")["conditions"]
    ]
    slots, members = prior_cases()
    save(
        BASE / "analysis_contract.json",
        dict(
            status="part6_scope_before_context_statistics",
            created_utc=c.utc(),
            sources=bindings(paths),
            exposure="Parts 1–5 outcomes and prior case rationales have been read. One first prompt and one KB example were inspected for schema. This is analysis-time supporting scope, not new preregistration.",
            delivery="All 21,627 inputs, exact saved full/small/zero prompts, all original KB records; verify text, metadata, ordering, rendering and every prediction-message binding. No retrieval execution or index opening.",
            inventories="Candidate versus delivered counts by language/channel/source; annotation absence versus observed-empty versus positive, gate polarity, legal-only and illustrative flags; selected unique records and frequency concentration. Every source, phase and HateCheck function retained.",
            comparisons="Per-query channel ID/set/order/rendered overlap, five-slot shared/replacement composition, annotation and label coverage, exact normalized query/example matches against original KBs. Character/UTF8 lengths and lexical token-set Jaccard describe delivery, not semantic relevance, input tokens or causal mechanisms.",
            associations="Descriptive common-nine populations: binary errors by 0..5 negative examples; full/small stable fixes/regressions/shared errors/correct/variable categories by gold polarity. Fine endpoint exactness by 0..5 annotated examples and none/partial/all gold labels shown; per-label exposure and recall. Fixed original gold/masks, all repeats equally averaged within text. No new p-values, best subset, label cutoff or causal adjustment.",
            cases="Union of ALL preselected Part 3/4/5 case slots, retaining empty slots, duplicates and linked members; exact full/small contexts and fixed first-repeat rationales. Review every distinct selected input; no new outcome-based example search.",
            matched="Cross-reference sealed matched-information results with their original population, wording and uncertainty family. Broad contexts are observational; examples, annotations and source composition are not independently randomized here.",
            limits="Part 7 global missingness/replay/whitespace/source-stratification remains separate. Part 8 actual tokens, timing and costs remain separate. Auxiliary flags are illustrative, not legal truth. Gold is never changed.",
            case_slots=len(slots),
            case_inputs=len(members),
            prohibited=[
                "network",
                "Qwen calls",
                "model loading",
                "inference",
                "training",
                "retrieval queries",
                "index opening",
                "source edits",
                "server actions",
            ],
        ),
    )
    save(BASE / "case_selection.json", dict(slots=slots, keys=sorted(members)))


def descriptor(row, query, kb, small_ids, bytext):
    co = row["condition"]
    ctx = row.get("context", dict(definitions=[], guidelines=[], example_groups=[]))
    chs = channels(ctx)
    require(
        reader.prompt.render_messages(query["text"], query["lang"], ctx, "corrected")
        == row["messages"],
        "Saved prompt render mismatch",
    )
    require(
        c.signature(row["messages"]) == row["messages_sha256"],
        "Message digest mismatch",
    )
    if co != "zero_shot":
        require(c.signature(ctx) == row["context_sha256"], "Context digest mismatch")
        require(
            reader.prompt.hit_ids(ctx) == row["retrieved"], "Saved hit IDs mismatch"
        )
    render = {}
    rank = Counter()
    for ch, hits in chs.items():
        require(len({h["id"] for h in hits}) == len(hits), "Duplicate delivered hit")
        for h in hits:
            rec = kb[h["id"]]
            m = {k: v for k, v in h["meta"].items() if k != "_ranks"}
            expected = meta(rec)
            require(
                h["text"] == rec["text"]
                and {k: v for k, v in m.items() if v is not None}
                == {k: v for k, v in expected.items() if v is not None},
                "Saved hit differs from frozen KB " + h["id"],
            )
            require(h["via"] == "rrf", "Unexpected retrieval channel")
            rank["+".join(sorted(h["meta"].get("_ranks", {})))] += 1
            if ch == "examples":
                require(m["lang"] == query["lang"], "Cross-language example")
        isolated = dict(definitions=[], guidelines=[], example_groups=[])
        isolated["example_groups" if ch == "examples" else ch] = (
            ctx["example_groups"] if ch == "examples" else hits
        )
        render[ch] = c.signature(
            reader.prompt.render_messages("", query["lang"], isolated, "corrected")[1][
                "text"
            ]
        )
    ex = chs["examples"]
    qt = tokens(query["text"])
    sim = [jaccard(qt, tokens(h["text"])) for h in ex]
    empty = reader.prompt.render_messages(
        query["text"],
        query["lang"],
        dict(definitions=[], guidelines=[], example_groups=[]),
        "corrected",
    )
    sources = Counter((h["meta"]["source"] for h in ex))
    return dict(
        key=query["key"],
        phase=query["phase"],
        source=query["source"],
        lang=query["lang"],
        condition=co,
        messages_sha256=row["messages_sha256"],
        context_sha256=row.get("context_sha256"),
        ids={ch: [h["id"] for h in hh] for ch, hh in chs.items()},
        rendered=render,
        annotations=annotations(ex),
        sources=dict(sources),
        rank_channels=dict(rank),
        same_source_examples=sum(
            (h["meta"]["source"].removesuffix("-train") == query["source"] for h in ex)
        ),
        examples_in_small=sum((h["id"] in small_ids for h in ex)),
        exact_query_examples=[
            h["id"] for h in ex if normalized(h["text"]) == normalized(query["text"])
        ],
        exact_candidate_examples=(
            bytext.get(normalized(query["text"]), []) if co != "zero_shot" else []
        ),
        message_chars=sum((len(m["text"]) for m in row["messages"])),
        message_utf8_bytes=sum(
            (len(m["text"].encode("utf8")) for m in row["messages"])
        ),
        context_chars=len(row["messages"][1]["text"]) - len(empty[1]["text"]),
        max_lexical_jaccard=max(sim, default=0.0),
        mean_lexical_jaccard=mean(sim),
        example_groups=[(name, len(hh)) for name, hh in ctx["example_groups"]],
    )


def summary(desc, kb):
    ex_counts = Counter((i for d in desc for i in d["ids"]["examples"]))
    nslots = sum(ex_counts.values())
    top = ex_counts.most_common(20)
    aa = [d["annotations"] for d in desc]
    return dict(
        n=len(desc),
        unique_hits={
            ch: len({i for d in desc for i in d["ids"][ch]})
            for ch in ("definitions", "guidelines", "examples")
        },
        example_source_slots=dict(
            Counter(
                (
                    source
                    for d in desc
                    for source, count in d["sources"].items()
                    for _ in range(count)
                )
            )
        ),
        source_slots_mean={
            source: sum((d["sources"].get(source, 0) for d in desc)) / len(desc)
            for source in sorted({s for d in desc for s in d["sources"]})
        },
        annotation_histograms={
            f: hist([a[f] for a in aa])
            for f in (
                "hate",
                "nonhate",
                "gate_unknown",
                "illustrative",
                "auxiliary_flags",
                "all_four_fine_known",
            )
        },
        field_coverage={
            dim: dict(
                known_hist=hist([a[dim]["known"] for a in aa]),
                empty_hist=hist([a[dim]["empty"] for a in aa]),
                positive_hist=hist([a[dim]["positive"] for a in aa]),
                label_slots=dict(
                    Counter(
                        (
                            l
                            for a in aa
                            for l, n in a[dim]["labels"].items()
                            for _ in range(n)
                        )
                    )
                ),
            )
            for dim in FIELDS
        },
        same_source_hist=hist([d["same_source_examples"] for d in desc]),
        small_member_hist=hist([d["examples_in_small"] for d in desc]),
        rank_channels=dict(sum((Counter(d["rank_channels"]) for d in desc), Counter())),
        lengths={
            key: stats([d[key] for d in desc])
            for key in (
                "message_chars",
                "message_utf8_bytes",
                "context_chars",
                "max_lexical_jaccard",
            )
        },
        exact_delivered_match_inputs=sum(
            (bool(d["exact_query_examples"]) for d in desc)
        ),
        exact_candidate_match_inputs=sum(
            (bool(d["exact_candidate_examples"]) for d in desc)
        ),
        example_slots=nslots,
        top20=[dict(id=i, count=n, source=kb[i]["source"]) for i, n in top],
        top10_share=sum((n for _, n in top[:10])) / nslots if nslots else None,
        effective_unique=(
            1 / sum(((v / nslots) ** 2 for v in ex_counts.values())) if nslots else None
        ),
    )


def comparison_summary(vals):
    return dict(
        n=len(vals),
        same_message=sum((v["same_message"] for v in vals)),
        same_delivered_context=sum((v["same_delivered_context"] for v in vals)),
        channels={
            ch: dict(
                shared_hist=hist([v[ch]["shared"] for v in vals]),
                same_order=sum((v[ch]["same_order"] for v in vals)),
                same_set=sum((v[ch]["same_set"] for v in vals)),
                same_rendered=sum((v[ch]["same_rendered"] for v in vals)),
                same_positions_hist=hist([v[ch]["same_positions"] for v in vals]),
            )
            for ch in ("definitions", "guidelines", "examples")
        },
    )


def binary_state(gold, pred):
    f = np.all(pred[:3] == gold)
    s = np.all(pred[3:6] == gold)
    fw = np.all(pred[:3] != gold)
    sw = np.all(pred[3:6] != gold)
    return (
        "stable_fix"
        if f and sw
        else (
            "stable_regression"
            if fw and s
            else (
                "stable_correct"
                if f and s
                else "stable_wrong" if fw and sw else "variable"
            )
        )
    )


def associations(desc, comp):
    metadata = c.read(PRIOR / "endpoint_metadata.json")
    out = {}
    for name, md in metadata.items():
        with np.load(ROOT / md["path"], allow_pickle=False) as z:
            y = z["gold"]
            p = z["predicted"]
            valid = z["valid"]
            gate = z["gate"]
        mask = valid.all(0)
        ii = np.flatnonzero(mask)
        keys = md["ids"]
        dim = name.split("/")[1]
        results = dict(
            n_roster=len(keys),
            n_common_nine=int(mask.sum()),
            population="all nine Qwen settings valid; original endpoint annotation mask",
            conditions={},
        )
        for coindex, co in enumerate(CONDITIONS[:2]):
            dd = [desc[co][keys[i]] for i in ii]
            pp = p[coindex * 3 : coindex * 3 + 3][:, ii]
            gg = gate[coindex * 3 : coindex * 3 + 3][:, ii]
            yy = y[ii]
            if dim == "hate":
                rows = []
                for n in range(6):
                    m = np.array([d["annotations"]["nonhate"] == n for d in dd])
                    positive = yy[m, 1].astype(bool)
                    pred = pp[:, m, 1].astype(bool)
                    rows.append(
                        dict(
                            nonhate_examples=n,
                            n=int(m.sum()),
                            gold_hate=int(positive.sum()),
                            gold_nonhate=int((~positive).sum()),
                            hate_rate=mean(pred.mean(0).tolist()),
                            fpr=mean(pred[:, ~positive].mean(0).tolist()),
                            fnr=mean((~pred[:, positive]).mean(0).tolist()),
                        )
                    )
                results["conditions"][co] = rows
            else:
                exact = (pp == yy).all(2)
                exact = exact if dim == "legal" else exact & gg
                seen = np.array(
                    [
                        [lab in d["annotations"][dim]["labels"] for lab in md["labels"]]
                        for d in dd
                    ],
                    bool,
                )
                positive = yy.astype(bool)
                covered = seen & positive
                support = positive.sum(1)
                cover = covered.sum(1)
                level = np.array(
                    [
                        (
                            "empty_gold"
                            if t == 0
                            else "none" if k == 0 else "all" if k == t else "partial"
                        )
                        for k, t in zip(cover, support)
                    ]
                )
                rows = []
                for family, selectors in [
                    (
                        "annotated_examples",
                        [
                            (
                                str(n),
                                np.array(
                                    [d["annotations"][dim]["known"] == n for d in dd]
                                ),
                            )
                            for n in range(6)
                        ],
                    ),
                    (
                        "gold_label_coverage",
                        [
                            (label, level == label)
                            for label in ("empty_gold", "none", "partial", "all")
                        ],
                    ),
                ]:
                    for label, m in selectors:
                        rows.append(
                            dict(
                                family=family,
                                stratum=label,
                                n=int(m.sum()),
                                gate_aware_exact=mean(exact[:, m].mean(0).tolist()),
                                closed_gate=mean((~gg[:, m]).mean(0).tolist()),
                            )
                        )
                labelrows = []
                for j, label in enumerate(md["labels"]):
                    for shown in (False, True):
                        m = positive[:, j] & (seen[:, j] == shown)
                        labelrows.append(
                            dict(
                                label=label,
                                gold_label_shown=shown,
                                n=int(m.sum()),
                                recall=mean(pp[:, m, j].mean(0).tolist()),
                            )
                        )
                results["conditions"][co] = dict(strata=rows, labels=labelrows)
        if dim == "hate":
            cats = [binary_state(bool(y[i, 1]), p[:, i, 1].astype(bool)) for i in ii]
            rows = []
            for source in ["ALL"] + sorted({md["rows"][i]["source"] for i in ii}):
                for gold in (False, True):
                    for cat in (
                        "stable_fix",
                        "stable_regression",
                        "stable_correct",
                        "stable_wrong",
                        "variable",
                    ):
                        selected = [
                            i
                            for i, catv in zip(ii, cats)
                            if catv == cat
                            and bool(y[i, 1]) == gold
                            and (source == "ALL" or md["rows"][i]["source"] == source)
                        ]
                        keys2 = [keys[i] for i in selected]
                        item = dict(
                            source=source,
                            gold_hate=gold,
                            category=cat,
                            n=len(selected),
                            conditions={},
                        )
                        for co in CONDITIONS[:2]:
                            dd = [desc[co][k] for k in keys2]
                            item["conditions"][co] = dict(
                                nonhate_examples=mean(
                                    [d["annotations"]["nonhate"] for d in dd]
                                ),
                                same_source_examples=mean(
                                    [d["same_source_examples"] for d in dd]
                                ),
                                context_chars=mean([d["context_chars"] for d in dd]),
                                max_lexical_jaccard=mean(
                                    [d["max_lexical_jaccard"] for d in dd]
                                ),
                            )
                        item["shared_examples"] = mean(
                            [comp[k]["examples"]["shared"] for k in keys2]
                        )
                        rows.append(item)
            results["stable_categories"] = rows
        out[name] = results
    return out


def run():
    contract = c.read(BASE / "analysis_contract.json")
    r.verify(contract["sources"])
    prior = c.read(PRIOR / "part5_complete.json")
    r.verify(prior["outputs"])
    r.verify(prior["original_sources"])
    prepare_renderer()
    groups, _ = reader.scoring.scientific_rosters("broad")
    queries = {v["key"]: v for vv in groups.values() for v in vv}
    require(len(queries) == 21627, "Wrong roster")
    slots, case_members = prior_cases()
    require(
        sorted(case_members) == c.read(BASE / "case_selection.json")["keys"],
        "Case set changed",
    )
    kbs = {
        v: {row["id"]: row for row in reader.rows(CORE / v / "records.jsonl")}
        for v in ("full", "small")
    }
    small_ids = set(kbs["small"])
    require(small_ids <= set(kbs["full"]), "Small KB not nested")
    require(
        all((kbs["small"][i] == kbs["full"][i] for i in small_ids)),
        "Shared KB record changed",
    )
    inventory = {}
    descs = {}
    case_prompts = defaultdict(dict)
    setup = c.read(reader.BASE / "prompt_setup.json")
    seen_messages = Counter()
    catalog = {}
    freq = {}
    for kbname, kb in kbs.items():
        inventory[kbname] = dict(
            records=len(kb),
            counts=dict(
                Counter(
                    ("/".join((v["kind"], v["lang"], v["source"])) for v in kb.values())
                )
            ),
            examples=annotations(
                [dict(meta=meta(v)) for v in kb.values() if v["kind"] == "example"]
            ),
        )
    for info in setup["conditions"]:
        co = info["condition"]
        kb = kbs["full" if co == "rag_full" else "small"]
        bytext = defaultdict(list)
        for ident, v in kb.items():
            if v["kind"] == "example":
                bytext[normalized(v["text"])].append(ident)
        descs[co] = {}
        freq[co] = Counter()
        for row in reader.rows(ROOT / info["path"]):
            key = row["key"]
            require(key in queries and key not in descs[co], "Unknown/duplicate prompt")
            query = queries[key]
            require(
                row["condition"] == co
                and row["population"] == query["phase"]
                and (row["lang"] == query["lang"]),
                "Prompt identity mismatch",
            )
            d = descriptor(row, query, kb, small_ids, bytext)
            descs[co][key] = d
            require(
                [len(d["ids"][ch]) for ch in ("definitions", "guidelines", "examples")]
                == ([0, 0, 0] if co == "zero_shot" else [2, 2, 5]),
                "Budget changed",
            )
            for ids in d["ids"].values():
                for ident in ids:
                    catalog[ident] = kb[ident]
                    freq[co][ident] += 1
            if key in case_members:
                case_prompts[key][co] = row
        require(set(descs[co]) == set(queries), "Incomplete prompt matrix")
        print(co, "saved prompt checks passed:", len(descs[co]), flush=True)
    comp = {
        key: overlap(descs["rag_full"][key], descs["rag_small"][key]) for key in queries
    }
    for v in reader.rows(reader.BASE / "predictions.jsonl"):
        ident = (v["key"], v["condition"], v["repeat"])
        require(
            v["repeat"] in (0, 1, 2) and seen_messages[ident] == 0,
            "Prediction identity duplicate",
        )
        require(
            v["messages_sha256"] == descs[v["condition"]][v["key"]]["messages_sha256"],
            "Prediction/context mismatch",
        )
        seen_messages[ident] += 1
    require(len(seen_messages) == 194643, "Prediction matrix incomplete")
    inventories = {}
    overlaps = {}
    group_keys = {"ALL": list(queries)}
    for phase, rr in groups.items():
        group_keys["phase/" + phase] = [v["key"] for v in rr]
        for source in sorted({v["source"] for v in rr}):
            group_keys["source/" + phase + "/" + source] = [
                v["key"] for v in rr if v["source"] == source
            ]
    for function in sorted(
        {json.loads(q["raw"])["functionality"] for q in groups["en_hatecheck_eval"]}
    ):
        group_keys["function/" + function] = [
            q["key"]
            for q in groups["en_hatecheck_eval"]
            if json.loads(q["raw"])["functionality"] == function
        ]
    for label, keys in group_keys.items():
        inventories[label] = {
            co: summary([descs[co][key] for key in keys], kbs["full"])
            for co in CONDITIONS
        }
        overlaps[label] = comparison_summary([comp[k] for k in keys])
    case_records = []
    for key, m in sorted(case_members.items()):
        case_records.append(
            m
            | dict(
                prompts=case_prompts[key],
                descriptors={co: descs[co][key] for co in CONDITIONS},
                comparison=comp[key],
            )
        )
    save(BASE / "candidate_inventory.json", inventory)
    save(BASE / "delivery_summary.json", inventories)
    save(BASE / "overlap_summary.json", overlaps)
    save(BASE / "case_contexts.json", case_records)
    save(BASE / "hit_catalog.json", catalog)
    save(BASE / "hit_frequencies.json", freq)
    lines(
        BASE / "prompt_descriptors.jsonl",
        (descs[co][key] for co in CONDITIONS for key in sorted(queries)),
    )
    lines(
        BASE / "context_comparisons.jsonl",
        (dict(key=k, **comp[k]) for k in sorted(queries)),
    )
    save(BASE / "outcome_associations.json", associations(descs, comp))
    save(BASE / "group_membership.json", group_keys)
    save(
        BASE / "exact_text_matches.json",
        [
            dict(
                key=k,
                condition=co,
                candidate=d["exact_candidate_examples"],
                delivered=d["exact_query_examples"],
            )
            for co in CONDITIONS[:2]
            for k, d in descs[co].items()
            if d["exact_candidate_examples"]
        ],
    )
    save(
        BASE / "analysis_complete.json",
        dict(
            status="part6_saved_context_extraction_and_descriptive_calculations_complete",
            created_utc=c.utc(),
            sources=contract["sources"],
            outputs=bindings([p for p in BASE.iterdir() if p.is_file()]),
            checks=dict(
                prompts=64881,
                prediction_bindings=len(seen_messages),
                saved_hit_occurrences=21627 * 18,
                groups=len(group_keys),
                case_slots=len(slots),
                case_inputs=len(case_records),
                new_retrieval_queries=0,
                new_Qwen_calls=0,
            ),
        ),
    )
    print(
        "Part 6 extraction complete; independent numerical and manual context reviews remain.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "run"))
    args = parser.parse_args()
    with r.original.offline():
        freeze() if args.mode == "freeze" else run()
