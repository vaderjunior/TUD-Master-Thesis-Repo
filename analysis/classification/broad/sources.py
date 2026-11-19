import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.broad import quality as quality

q = quality
reader = q.reader
c = q.c
r = q.r
ROOT = q.ROOT
BASE = ROOT / "experiments/astra/sq2_broad_sources_review_20261104_v1"
CONDITIONS = reader.CONDITIONS
SETTINGS = [(co, j) for co in CONDITIONS for j in range(3)]
require = reader.require
bindings = reader.bindings
safe = q.safe
DRAWS = 10000
SEED = 20261008
FUNCTIONS = {
    "derog_neg_emote_h": "Negative emotions",
    "derog_neg_attrib_h": "Negative attributes",
    "derog_dehum_h": "Dehumanisation",
    "derog_impl_h": "Implicit derogation",
    "threat_dir_h": "Direct threats",
    "threat_norm_h": "Normative threats",
    "slur_h": "Hateful slurs",
    "slur_homonym_nh": "Non-hateful slur homonyms",
    "slur_reclaimed_nh": "Reclaimed slurs",
    "profanity_h": "Hateful profanity",
    "profanity_nh": "Non-hateful profanity",
    "ref_subs_clause_h": "Reference in a later clause",
    "ref_subs_sent_h": "Reference in a later sentence",
    "negate_pos_h": "Negated positive statements",
    "negate_neg_nh": "Negated hateful statements",
    "phrase_question_h": "Hate phrased as a question",
    "phrase_opinion_h": "Hate phrased as an opinion",
    "ident_neutral_nh": "Neutral identity mentions",
    "ident_pos_nh": "Positive identity mentions",
    "counter_quote_nh": "Counter-speech quoting hate",
    "counter_ref_nh": "Counter-speech referring to hate",
    "target_obj_nh": "Abuse of objects",
    "target_indiv_nh": "Abuse of individuals",
    "target_group_nh": "Abuse of unprotected groups",
    "spell_char_swap_h": "Swapped characters",
    "spell_char_del_h": "Deleted characters",
    "spell_space_add_h": "Added spaces",
    "spell_space_del_h": "Removed spaces",
    "spell_leet_h": "Leetspeak substitutions",
}


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()), "Output outside Part 5")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(c.encoded(safe(value)))


def array(path, **values):
    require(path.resolve().is_relative_to(BASE.resolve()), "Array outside Part 5")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def raw(row):
    return json.loads(row["raw"])


def normalized_identity(v):
    return (
        "not_recorded" if v is None or str(v).lower() in ("nan", "none", "") else str(v)
    )


def identifier(v):
    return (
        str(int(v))
        if isinstance(v, (int, float)) and math.isfinite(v) and int(v) == v
        else str(v)
    )


def annotation(row):
    v = raw(row)
    source = row["source"]
    if source == "hatexplain":
        co = Counter(v["votes"])
        lab, n = co.most_common(1)[0]
        require(len(v["votes"]) == 3 and n >= 2, "Ambiguous HateXplain majority")
        return "hatexplain/" + lab
    if source == "implicit_hate":
        return "implicit_hate/" + v["stage1_class"]
    if source == "mhs":
        score = v["score"]
        return "mhs/" + (
            "score<=0"
            if score <= 0
            else (
                "0<score<=0.5"
                if score <= 0.5
                else (
                    "0.5<score<=1.0"
                    if score <= 1
                    else "1.0<score<=1.63" if score <= 1.63 else "score>1.63"
                )
            )
        )
    if source == "detox":
        score = v["hate_speech_score"]
        return "detox/" + (
            "score=0"
            if score == 0
            else (
                "0<score<0.5"
                if score < 0.5
                else (
                    "score=0.5"
                    if score == 0.5
                    else "0.5<score<1" if score < 1 else "score=1"
                )
            )
        )
    if source == "gahd":
        return "gahd/" + v["round"]
    raise ValueError("Unplanned annotation source")


def structure(rows):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    bycase = {identifier(raw(row)["case_id"]): i for i, row in enumerate(rows)}
    require(len(bycase) == len(rows), "Duplicate case identity")
    templates = defaultdict(list)
    texts = defaultdict(list)
    links = []
    for i, row in enumerate(rows):
        meta = raw(row)
        templates[identifier(meta["templ_id"])].append(i)
        texts[row["text_cluster"]].append(i)
        ref = meta["ref_case_id"]
        if ref is not None:
            require(identifier(ref) in bycase, "Missing HateCheck parent")
            j = bycase[identifier(ref)]
            require(i != j, "Self reference")
            union(i, j)
            links.append(
                dict(
                    child=i,
                    parent=j,
                    child_key=row["key"],
                    parent_key=rows[j]["key"],
                    function=meta["functionality"],
                    same_gold=row["gate"] == rows[j]["gate"],
                    cluster=row["functional_cluster"],
                )
            )
    for table in (templates, texts):
        for ii in table.values():
            for i in ii[1:]:
                union(ii[0], i)
    rebuilt = defaultdict(list)
    frozen = defaultdict(list)
    for i, row in enumerate(rows):
        rebuilt[find(i)].append(i)
        frozen[row["functional_cluster"]].append(i)
    require(
        {tuple(v) for v in rebuilt.values()} == {tuple(v) for v in frozen.values()},
        "Frozen connected components differ",
    )
    groups = {
        name: {
            str(k): dict(
                indices=ii,
                functions=sorted({raw(rows[i])["functionality"] for i in ii}),
                gold_counts=dict(Counter(str(rows[i]["gate"]) for i in ii)),
            )
            for k, ii in sorted(table.items())
        }
        for name, table in (("template", templates), ("component", frozen))
    }
    return dict(
        groups=groups,
        links=links,
        counts={
            name: dict(
                units=len(v),
                largest=max(len(x["indices"]) for x in v.values()),
                crosses_function=sum(len(x["functions"]) > 1 for x in v.values()),
                crosses_gold=sum(len(x["gold_counts"]) > 1 for x in v.values()),
            )
            for name, v in groups.items()
        },
        normalized_text_groups=len(texts),
    )


def features(y, p, g, kind):
    require(
        y.ndim == 2
        and p.ndim == 3
        and p.shape[1:] == y.shape
        and g.shape == p.shape[:2],
        "Wrong outcome axes",
    )
    require(
        np.isin(y, [0, 1]).all() and np.isin(p, [0, 1]).all(),
        "Nonbinary label encoding",
    )
    if kind == "binary":
        require(
            y.shape[1] == 2 and (y.sum(1) == 1).all() and (p.sum(2) <= 1).all(),
            "Invalid binary one-hot encoding",
        )
        yy = y[:, 1].astype(bool)
        pp = p[:, :, 1].astype(bool)
        return np.stack((pp & yy, pp & ~yy, ~pp & yy, ~pp & ~yy), axis=-1).astype(float)
    require(kind in ("fine", "legal"), "Unknown endpoint kind")
    if kind == "fine":
        require(not p[~g].any(), "Delivered fine labels under closed gate")
    yy = y.astype(bool)
    pp = p.astype(bool)
    counts = np.stack((pp & yy, pp & ~yy, ~pp & yy), axis=-1).reshape(
        len(p), len(y), -1
    )
    exact = (p == y).all(2)
    tp = (pp & yy).sum(2)
    union = (pp | yy).sum(2)
    jac = np.divide(tp, union, out=np.ones(tp.shape, float), where=union > 0)
    gate_exact = exact if kind == "legal" else exact & g
    linear = np.stack(
        (
            gate_exact,
            exact,
            jac,
            (pp & ~yy).sum(2),
            (~pp & yy).sum(2),
            ~g,
            np.ones(g.shape),
        ),
        axis=-1,
    )
    return np.concatenate((counts, linear), axis=-1).astype(float)


def metrics(t, kind, labels, macro):
    if kind == "binary":
        return q.errors.metrics(t)
    if kind == "links":
        names = [
            "both_correct",
            "both_wrong",
            "child_only_wrong",
            "parent_only_wrong",
            "expected_relation",
        ]
        n = t[..., -1]
        return {"n": n} | {key: q.div(t[..., j], n) for j, key in enumerate(names)}
    n = t[..., -1]
    count = t[..., : 3 * len(labels)].reshape(*t.shape[:-1], len(labels), 3)
    tp, fp, fn = (count[..., j] for j in range(3))
    denom = 2 * tp + fp + fn
    ff = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    idx = [labels.index(x) for x in macro]
    out = dict(
        n=n,
        macro_f1=ff[..., idx].mean(-1) if idx else np.full(n.shape, np.nan),
        label_tp=tp,
        label_fp=fp,
        label_fn=fn,
        label_precision=q.div(tp, tp + fp),
        label_recall=q.div(tp, tp + fn),
        label_f1=ff,
    )
    for j, key in enumerate(
        (
            "gate_aware_exact",
            "exact_set",
            "sample_jaccard",
            "extra_labels",
            "missing_labels",
            "gate_closed",
        )
    ):
        out[key] = q.div(t[..., 3 * len(labels) + j], n)
    return out


def summarize(data, ii, m, weights=None):
    f = data["features"][ii][:, m]
    totals = f.sum(1) if weights is None else np.einsum("snf,n->sf", f, weights)
    values = metrics(totals, data["kind"], data["labels"], data["macro"])
    repeats = [safe({key: v[j] for key, v in values.items()}) for j in range(len(ii))]
    means = {key: safe(np.mean(v, axis=0)) for key, v in values.items()}
    sd = (
        {key: safe(np.std(v, axis=0, ddof=1)) for key, v in values.items()}
        if len(ii) > 1
        else {}
    )
    if data["kind"] == "binary" and (
        not data["gold"][m, 1].any() or not data["gold"][m, 0].any()
    ):
        for d in repeats:
            d["macro_f1"] = None
        means["macro_f1"] = None
        sd["macro_f1"] = None
    return dict(repeats=repeats, mean=means, sd=sd)


def bootstrap(data, ii, m, keys, unit="component"):
    cc = [
        data["clusters"][j] if unit == "component" else data["ids"][j]
        for j in np.flatnonzero(m)
    ]
    unique = sorted(set(cc))
    look = {k: i for i, k in enumerate(unique)}
    f = data["features"][ii][:, m]
    grouped = np.zeros((len(unique), len(ii), f.shape[2]))
    np.add.at(grouped, [look[k] for k in cc], f.transpose(1, 0, 2))
    flat = grouped.reshape(len(unique), -1)
    rng = np.random.default_rng(SEED)
    values = []
    for start in range(0, DRAWS, 64):
        picks = rng.choice(
            list(range(len(unique))),
            (min(64, DRAWS - start), len(unique)),
            replace=True,
        )
        w = np.stack([np.bincount(row, minlength=len(unique)) for row in picks])
        t = (w @ flat).reshape(len(picks), len(ii), f.shape[2])
        mvals = metrics(t, data["kind"], data["labels"], data["macro"])
        values.append(
            np.stack(
                [mvals[k][:, :3].mean(1) - mvals[k][:, 3:].mean(1) for k in keys],
                axis=1,
            )
        )
    draws = np.concatenate(values)
    point = metrics(f.sum(1), data["kind"], data["labels"], data["macro"])
    out = {}
    for j, key in enumerate(keys):
        invalid = int((~np.isfinite(draws[:, j])).sum())
        out[key] = dict(
            difference=safe(point[key][:3].mean() - point[key][3:].mean()),
            ci95=(
                np.quantile(draws[:, j], [0.025, 0.975]).tolist()
                if not invalid
                else None
            ),
            undefined_draws=invalid,
        )
    return (
        dict(
            metrics=out,
            n=int(m.sum()),
            units=len(unique),
            resampling=unit,
            draws=DRAWS,
            seed=SEED,
            role="supporting pointwise; original primary family unchanged",
        ),
        draws,
    )


def group(data, indices, ident, keys=(), item=False):
    select = np.zeros(len(data["ids"]), bool)
    select[indices] = True
    common = select & data["valid"].all(0)
    require(common.any(), "No common Qwen outputs in requested group")
    out = dict(
        id=ident,
        endpoint=data["name"],
        kind=data["kind"],
        indices=indices,
        eligible=len(indices),
        common_indices=np.flatnonzero(common),
        common_nine_n=int(common.sum()),
        labels=data["labels"],
        macro_labels=data["macro"],
        source_counts=dict(Counter(data["rows"][i]["source"] for i in indices)),
        full_gold_support=(
            data["gold"][select].sum(0) if data["kind"] != "links" else []
        ),
        shared_gold_support=(
            data["gold"][common].sum(0) if data["kind"] != "links" else []
        ),
        common_clusters=len({data["clusters"][i] for i in np.flatnonzero(common)}),
        own_valid={},
        common_nine={},
        pairs=[],
    )
    for co in CONDITIONS:
        ii = [SETTINGS.index((co, j)) for j in range(3)]
        own = []
        for s in ii:
            m = select & data["valid"][s]
            v = summarize(data, [s], m)["repeats"][0]
            v.update(
                eligible=len(indices), invalid=int((select & ~data["valid"][s]).sum())
            )
            ratekey = (
                "accuracy"
                if data["kind"] == "binary"
                else "both_correct" if data["kind"] == "links" else "gate_aware_exact"
            )
            validn = int(m.sum())
            x = v[ratekey]
            v["exact_or_accuracy_failure_bounds"] = (
                [
                    x * validn / len(indices),
                    (x * validn + len(indices) - validn) / len(indices),
                ]
                if x is not None
                else [0.0, 1.0]
            )
            own.append(v)
        out["own_valid"][co] = own
        out["common_nine"][co] = summarize(data, ii, common)
    for co in ("rag_small", "zero_shot"):
        ii = list(range(3)) + [SETTINGS.index((co, j)) for j in range(3)]
        m = select & data["valid"][ii].all(0)
        pair = dict(
            reference=co,
            n=int(m.sum()),
            common_indices=np.flatnonzero(m),
            full=summarize(data, ii[:3], m),
            reference_summary=summarize(data, ii[3:], m),
            intervals={},
        )
        for unit in (("component", "item") if item else ("component",)) if keys else ():
            info, dd = bootstrap(data, ii, m, keys, unit)
            path = (
                BASE
                / "draws"
                / (ident.replace("/", "__") + "__" + co + "__" + unit + ".npz")
            )
            array(path, draws=dd, common_indices=np.flatnonzero(m))
            info["path"] = c.rel(path)
            pair["intervals"][unit] = info
        out["pairs"].append(pair)
    return out


def load_data():
    groups, _ = reader.scoring.scientific_rosters("broad")
    endpoints = c.read(reader.BASE / "endpoint_setup.json")
    out = {}
    for name, ep in endpoints.items():
        phase, dim = name.split("/")
        meta = c.read(q.main.BASE / "endpoints" / q.main.slug(name) / "results.json")
        ss = [tuple(v) for v in meta["settings"]]
        ii = [ss.index(s) for s in SETTINGS]
        with np.load(ROOT / meta["encoded_path"], allow_pickle=False) as z:
            y = z["gold"].copy()
            p = z["predicted"][ii].copy()
            v = z["valid"][ii].copy()
        if dim == "hate":
            g = p[:, :, ep["all_labels"].index("hate")].astype(bool)
        else:
            qm = c.read(q.BASE / "endpoints" / q.main.slug(name) / "results.json")
            qs = [tuple(s) for s in qm["settings"]]
            with np.load(ROOT / qm["encoded_path"], allow_pickle=False) as z:
                g = z["gate"][[qs.index(s) for s in SETTINGS]].copy()
        lookup = {row["key"]: row for row in groups[phase]}
        rows = [lookup[key] for key in ep["scorable_ids"]]
        kind = "binary" if dim == "hate" else "legal" if dim == "legal" else "fine"
        if kind == "binary":
            require(
                ep["all_labels"] == ["not_hate", "hate"],
                "Binary label ordering changed",
            )
        out[name] = dict(
            name=name,
            kind=kind,
            labels=ep["all_labels"],
            macro=ep["macro_labels"],
            ids=ep["scorable_ids"],
            rows=rows,
            clusters=ep["row_clusters"],
            gold=y,
            predicted=p,
            gate=g,
            valid=v,
            features=features(y, p, g, kind),
        )
    return out, groups


def freeze():
    paths = [
        reader.PLAN,
        q.BASE / "part4_complete.json",
        q.BASE / "final_review.json",
        q.errors.BASE / "part3_complete.json",
        q.main.BASE / "part2_complete.json",
        reader.BASE / "part1_complete.json",
        Path(__file__),
        ROOT / "data/mappings/mhs_thresholds.md",
    ] + [
        ROOT / "src/hsrag/data" / name
        for name in (
            "hatexplain.py",
            "implicit_hate.py",
            "mhs.py",
            "detox.py",
            "gahd.py",
        )
    ]
    save(
        BASE / "analysis_contract.json",
        dict(
            status="part5_scope_before_source_outcome_scoring",
            created_utc=c.utc(),
            sources=bindings(paths),
            scope="Saved Qwen full/small/no retrieval, three fixed repeats. All five natural binary sources, nine natural fine/legal source populations, all 29 HateCheck functions and 18 hateful target-function populations. Historical encoders remain the original Parts 2/4 references and separate completed GPU analysis; no replacement.",
            prior_exposure="Parts 1–4 reviewed and gold-only metadata inventoried. This supporting scope is recorded at analysis time, not a new preregistration.",
            natural="All EN TEST sources and both DE TEST sources. Source macro-F1/FPR/FNR/error paired 95% intervals. Fine source scores keep endpoint-fixed macro columns and all-label support; four split HX/MHS target populations receive macro/exact/Jaccard intervals, single-source endpoints retain descriptive source attribution and prior Part 2/4 inference.",
            annotation_slices="HateXplain original majority label, unanimity/split and majority-by-agreement; Implicit Hate original stage1 categories (keep eight stage2-only rows separate); fixed MHS score bands <=0, 0..0.5, 0.5..1, 1..1.63, >1.63; DeTox frozen hate-score bands 0, (0,.5), .5, (.5,1), 1; GAHD collection rounds and original source split. Descriptive rates, class denominators explicit; no relabeling or newly inferred demographics.",
            hatecheck="Rebuild all template/reference/text connected components; all 29 function error rates and 18 hateful-function target exactness contrasts. Component and item-resampling sensitivity for these function contrasts, with both displayed. Identity and directed/general slices descriptive, separated by gold polarity. No single-polarity binary macro-F1 claim.",
            links="All saved reference edges; both-correct/both-wrong/child-only-wrong/parent-only-wrong and expected relation, separated by same/changed gold and child function. All/same/changed groups get paired component-bootstrap intervals for both-correct, both-wrong and expected relation. Validity requires both endpoints in every compared setting; dependent edges share inherited components.",
            weights="Predefined descriptive text/equal-function/equal-template/equal-component weights on the explicitly stated common-valid population; retained/empty unit counts and resulting class prevalence/FPR/FNR/error reported. No preferred weighting selected, no new primary estimate. Original global source-stratified/item/replay/whitespace/partial-vote sensitivities remain Part 7.",
            uncertainty=dict(
                draws=DRAWS,
                seed=SEED,
                confidence=0.95,
                role="supporting pointwise, no multiplicity-adjusted family or universal winner",
                fixed_repeats=True,
                nulls="No invalid-as-negative/empty; own/common-six/common-nine masks and full-roster rate bounds explicit. Undefined denominators remain null; retain undefined bootstrap counts, no selective draw dropping.",
            ),
            case_rule="For each of five binary sources, one all-three-repeat-stable full/small fix and regression for each gold class (20 slots), min SHA256 sq2-broad-part5-v1|slot|key. For each of all 29 functions choose a full error with the highest error count across its three valid repeats, then same hash rule; keep empty error slots. Four linked both-correct stable gain/loss cases across same/changed gold, min hash over child+parent. Keep duplicates, empty slots and all candidates. Full and small first-repeat rationales, all nine delivered decisions. Illustrations, not prevalence or causal evidence.",
            functions=FUNCTIONS,
            limits=[
                "different annotation rules and project splits prevent a language ranking",
                "source identity slices are not causal fairness effects",
                "no validated legal advice or faithful internal reasoning",
                "do not count unannotated fields or silently reinterpret empty targets",
            ],
            prohibited=[
                "Qwen calls",
                "training/inference/model loading",
                "retrieval reruns",
                "server access",
                "original/source/writer edits",
                "deletion or relabelling",
            ],
        ),
    )


def select_case(slot, candidates, category, endpoint, extra=None):
    digest = lambda key: hashlib.sha256(
        f"sq2-broad-part5-v1|{slot}|{key}".encode()
    ).hexdigest()
    key = min(candidates, key=lambda k: (digest(k), k)) if candidates else None
    return dict(
        slot=slot,
        category=category,
        endpoint=endpoint,
        candidates=len(candidates),
        key=key,
        selection_sha256=digest(key) if key else None,
        **(extra or {}),
    )


def run():
    contract = c.read(BASE / "analysis_contract.json")
    r.verify(contract["sources"])
    tests = c.read(BASE / "tests_v2.json")
    require(tests["status"] == "passed", "Tests required")
    r.verify(tests["sources"])
    seal = c.read(q.BASE / "part4_complete.json")
    r.verify(seal["outputs"])
    r.verify(seal["original_sources"])
    before = bindings(
        [
            BASE / "analysis_contract.json",
            BASE / "tests_v2.json",
            Path(__file__),
            q.BASE / "part4_complete.json",
        ]
    )
    save(BASE / "preservation_before_v2.json", before)
    data, groups = load_data()
    metadata = {}
    natural = {}
    slices = {}
    fine = {}
    slots = []
    for name, d in data.items():
        path = BASE / "encoded" / q.main.slug(name)
        array(
            path.with_suffix(".npz"),
            gold=d["gold"],
            predicted=d["predicted"],
            valid=d["valid"],
            gate=d["gate"],
            features=d["features"],
        )
        metadata[name] = {
            k: d[k]
            for k in ("name", "kind", "labels", "macro", "ids", "rows", "clusters")
        }
        metadata[name]["path"] = c.rel(path.with_suffix(".npz"))
    save(BASE / "endpoint_metadata.json", metadata)
    for name in ("en_test/hate", "de_test/hate"):
        d = data[name]
        sources = sorted({row["source"] for row in d["rows"]})
        for source in sources:
            ix = [i for i, row in enumerate(d["rows"]) if row["source"] == source]
            key = name + "/" + source
            natural[key] = group(
                d, ix, "source/" + key, ("macro_f1", "fpr", "fnr", "error_rate")
            )
            for gold in (0, 1):
                for category in ("fix", "regression"):
                    candidates = []
                    for i in ix:
                        if not d["valid"][:6, i].all() or int(d["gold"][i, 1]) != gold:
                            continue
                        a = d["predicted"][:3, i, 1] == gold
                        b = d["predicted"][3:6, i, 1] == gold
                        if (
                            (all(a) and not any(b))
                            if category == "fix"
                            else (not any(a) and all(b))
                        ):
                            candidates.append(d["ids"][i])
                    slot = f"{source}/" + ("fn" if gold else "fp") + "/" + category
                    slots.append(
                        select_case(
                            slot,
                            candidates,
                            category,
                            name,
                            dict(source=source, gold=bool(gold)),
                        )
                    )
            print("Natural source:", key, flush=True)
        for title in sorted({annotation(row) for row in d["rows"]}):
            ix = [i for i, row in enumerate(d["rows"]) if annotation(row) == title]
            slices[name + "/" + title] = group(
                d, ix, "annotation/" + name + "/" + title
            )
        if name == "en_test/hate":
            for agreement in ("unanimous", "split"):
                ix = [
                    i
                    for i, row in enumerate(d["rows"])
                    if row["source"] == "hatexplain"
                    and (len(set(raw(row)["votes"])) == 1) == (agreement == "unanimous")
                ]
                slices["hatexplain_agreement/" + agreement] = group(
                    d, ix, "agreement/" + agreement
                )
                for label in ("normal", "offensive", "hatespeech"):
                    jj = [
                        i
                        for i in ix
                        if annotation(d["rows"][i]) == "hatexplain/" + label
                    ]
                    slices["hatexplain_label_agreement/" + label + "/" + agreement] = (
                        group(d, jj, "label_agreement/" + label + "/" + agreement)
                    )
        else:
            for split in sorted(
                {
                    raw(row)["official_split"]
                    for row in d["rows"]
                    if row["source"] == "gahd"
                }
            ):
                ix = [
                    i
                    for i, row in enumerate(d["rows"])
                    if row["source"] == "gahd" and raw(row)["official_split"] == split
                ]
                slices["gahd_original_split/" + split] = group(
                    d, ix, "gahd_original_split/" + split
                )
    save(BASE / "natural_binary.json", natural)
    save(BASE / "annotation_slices.json", slices)
    for name, d in data.items():
        if d["kind"] == "binary" or name.startswith("en_hatecheck"):
            continue
        sources = sorted({row["source"] for row in d["rows"]})
        for source in sources:
            ix = [i for i, row in enumerate(d["rows"]) if row["source"] == source]
            key = name + "/" + source
            fine[key] = group(
                d,
                ix,
                "fine_source/" + key,
                (
                    ("macro_f1", "gate_aware_exact", "sample_jaccard")
                    if len(sources) > 1
                    else ()
                ),
            )
    save(BASE / "fine_sources.json", fine)
    d = data["en_hatecheck_eval/hate"]
    st = structure(d["rows"])
    require(
        len(st["links"]) == 2046
        and st["counts"]["template"]["units"] == 845
        and st["counts"]["component"]["units"] == 570,
        "HateCheck structure changed",
    )
    save(BASE / "hatecheck_structure.json", st)
    function = {}
    ftarget = {}
    identity = {}
    direction = {}
    fine_d = data["en_hatecheck_eval/target_group"]
    functions_present = {raw(row)["functionality"] for row in d["rows"]}
    require(functions_present == set(FUNCTIONS), "Function roster incomplete")
    for fn, title in FUNCTIONS.items():
        ix = [i for i, row in enumerate(d["rows"]) if raw(row)["functionality"] == fn]
        require(
            all(row["gate"] == fn.endswith("_h") for row in [d["rows"][i] for i in ix]),
            "Function polarity changed",
        )
        function[fn] = group(d, ix, "function/" + fn, ("error_rate",), True)
        function[fn].update(
            title=title,
            gold_hate=fn.endswith("_h"),
            templates=len({identifier(raw(d["rows"][i])["templ_id"]) for i in ix}),
        )
        ii = [
            i for i, row in enumerate(fine_d["rows"]) if raw(row)["functionality"] == fn
        ]
        if ii:
            ftarget[fn] = group(
                fine_d, ii, "function_target/" + fn, ("gate_aware_exact",), True
            )
        candidates = [i for i in ix if d["valid"][:3, i].all()]
        errorcount = {
            i: int((d["predicted"][:3, i, 1] != d["gold"][i, 1]).sum())
            for i in candidates
        }
        maximum = max(errorcount.values(), default=0)
        candidates = [
            d["ids"][i] for i in candidates if maximum > 0 and errorcount[i] == maximum
        ]
        slots.append(
            select_case(
                "function/" + fn,
                candidates,
                "full_error",
                d["name"],
                dict(function=fn, full_error_repeats=maximum),
            )
        )
        print("HateCheck function:", fn, flush=True)
    for ident in sorted(
        {normalized_identity(raw(row)["target_ident"]) for row in d["rows"]}
    ):
        for gold in (False, True):
            ix = [
                i
                for i, row in enumerate(d["rows"])
                if normalized_identity(raw(row)["target_ident"]) == ident
                and row["gate"] == gold
            ]
            if ix:
                identity[ident + "/" + str(gold)] = group(
                    d, ix, "identity/" + ident + "/" + str(gold)
                )
        ix = [
            i
            for i, row in enumerate(fine_d["rows"])
            if normalized_identity(raw(row)["target_ident"]) == ident
        ]
        if ix:
            identity[ident + "/target_exact"] = group(
                fine_d, ix, "identity_target/" + ident
            )
    for direct in ("general", "directed", "-"):
        ix = [i for i, row in enumerate(d["rows"]) if raw(row)["direction"] == direct]
        if ix:
            direction[direct] = group(d, ix, "direction/" + direct)
    save(BASE / "hatecheck_functions.json", function)
    save(BASE / "hatecheck_target_functions.json", ftarget)
    save(BASE / "hatecheck_identities.json", identity)
    save(BASE / "hatecheck_direction.json", direction)
    links = st["links"]
    child = [x["child"] for x in links]
    par = [x["parent"] for x in links]
    correct = d["predicted"][:, :, 1] == d["gold"][:, 1]
    both = correct[:, child] & correct[:, par]
    wrong = ~correct[:, child] & ~correct[:, par]
    childwrong = ~correct[:, child] & correct[:, par]
    parentwrong = correct[:, child] & ~correct[:, par]
    relation = (d["predicted"][:, child, 1] == d["predicted"][:, par, 1]) == np.array(
        [x["same_gold"] for x in links]
    )
    require(
        np.array_equal(relation, both | wrong),
        "Relation does not equal both-correct plus both-wrong",
    )
    linked = dict(
        name="hatecheck/links",
        kind="links",
        labels=[],
        macro=[],
        ids=[x["child_key"] + "::" + x["parent_key"] for x in links],
        clusters=[x["cluster"] for x in links],
        rows=[d["rows"][i] for i in child],
        gold=np.zeros((len(links), 0), int),
        valid=d["valid"][:, child] & d["valid"][:, par],
        features=np.stack(
            (both, wrong, childwrong, parentwrong, relation, np.ones(both.shape)),
            axis=-1,
        ).astype(float),
    )
    linked_results = {}
    for title, ix in [
        ("all", list(range(len(links)))),
        ("same_gold", [i for i, x in enumerate(links) if x["same_gold"]]),
        ("changed_gold", [i for i, x in enumerate(links) if not x["same_gold"]]),
    ] + [
        ("function/" + fn, [i for i, x in enumerate(links) if x["function"] == fn])
        for fn in FUNCTIONS
        if any(x["function"] == fn for x in links)
    ]:
        linked_results[title] = group(
            linked,
            ix,
            "linked/" + title,
            (
                ("both_correct", "both_wrong", "expected_relation")
                if "/" not in title
                else ()
            ),
        )
        if title in ("same_gold", "changed_gold"):
            for category in ("fix", "regression"):
                candidates = []
                for i in ix:
                    if not linked["valid"][:6, i].all():
                        continue
                    a = both[:3, i]
                    b = both[3:6, i]
                    if (
                        (all(a) and not any(b))
                        if category == "fix"
                        else (not any(a) and all(b))
                    ):
                        candidates.append(linked["ids"][i])
                slots.append(
                    select_case(
                        "linked/" + title + "/" + category,
                        candidates,
                        "linked_" + category,
                        "hatecheck/links",
                        dict(relation=title),
                    )
                )
    save(BASE / "hatecheck_linked.json", linked_results)
    array(
        BASE / "encoded/linked.npz", features=linked["features"], valid=linked["valid"]
    )
    save(
        BASE / "linked_metadata.json",
        {
            k: linked[k]
            for k in ("name", "kind", "labels", "macro", "ids", "clusters", "rows")
        },
    )
    weights = {}
    schemes = {
        "text": d["ids"],
        "function": [raw(row)["functionality"] for row in d["rows"]],
        "template": [identifier(raw(row)["templ_id"]) for row in d["rows"]],
        "component": d["clusters"],
    }
    for view, ii in [
        ("common_nine", list(range(9))),
        ("full_small", list(range(6))),
        ("full_zero", list(range(3)) + list(range(6, 9))),
    ]:
        m = d["valid"][ii].all(0)
        ww = {}
        for scheme, units in schemes.items():
            selected = [units[i] for i in np.flatnonzero(m)]
            co = Counter(selected)
            w = np.array([1 / (len(co) * co[k]) for k in selected])
            require(abs(w.sum() - 1) < 1e-12, "Unequal-unit weights not normalized")
            ww[scheme] = dict(
                n=int(m.sum()),
                full_units=len(set(units)),
                retained_units=len(co),
                empty_units=len(set(units)) - len(co),
                hateful_weight=float(w @ d["gold"][m, 1]),
                conditions={},
            )
            for cond in dict.fromkeys(SETTINGS[s][0] for s in ii):
                jj = [SETTINGS.index((cond, j)) for j in range(3)]
                ww[scheme]["conditions"][cond] = summarize(d, jj, m, w)
        weights[view] = ww
    save(BASE / "hatecheck_weighting.json", weights)

    for i, slot in enumerate(slots, 1):
        slot["case_id"] = f"S{i:02d}"
    save(BASE / "case_selection.json", dict(rule=contract["case_rule"], slots=slots))
    selected = set()
    for slot in slots:
        if slot["key"]:
            selected.update(slot["key"].split("::"))
    cache = {key: [] for key in selected}
    focused = {}
    oldbindings = c.read(reader.BASE / "item_bindings.json")
    for row in reader.rows(reader.BASE / "predictions.jsonl"):
        if row["key"] not in selected:
            continue
        p = Path(row["result_path"])
        require(
            c.sha(p) == oldbindings[row["result_path"]],
            "Original selected result changed",
        )
        focused[row["result_path"]] = oldbindings[row["result_path"]]
        cache[row["key"]].append(row)
    gold = {row["key"]: row for rr in groups.values() for row in rr}
    cases = []
    for slot in slots:
        members = []
        for key in slot["key"].split("::") if slot["key"] else []:
            require(len(cache[key]) == 9, "Incomplete case prediction matrix")
            members.append(dict(key=key, gold=gold[key], predictions=cache[key]))
        cases.append(
            slot
            | dict(
                status="selected" if slot["key"] else "empty category", members=members
            )
        )
    save(BASE / "cases.json", cases)
    save(BASE / "focused_source_bindings.json", focused)
    provenance = {
        source: dict(
            phases=dict(
                Counter(
                    row["phase"]
                    for rr in groups.values()
                    for row in rr
                    if row["source"] == source
                )
            ),
            original_splits=dict(
                Counter(
                    str(raw(row).get("official_split", "not_recorded"))
                    for rr in groups.values()
                    for row in rr
                    if row["source"] == source
                )
            ),
        )
        for source in sorted({row["source"] for rr in groups.values() for row in rr})
    }
    save(BASE / "population_provenance.json", provenance)
    r.verify(before)
    drawn = 0
    for path in (BASE / "draws").glob("*.npz"):
        with np.load(path, allow_pickle=False) as z:
            drawn += z["draws"].size
    save(
        BASE / "analysis_complete.json",
        dict(
            status="part5_calculation_complete_independent_and_manual_review_pending",
            sources=before,
            outputs=bindings([p for p in BASE.rglob("*") if p.is_file()]),
            natural_binary_sources=len(natural),
            annotation_groups=len(slices),
            fine_source_populations=len(fine),
            hatecheck_functions=len(function),
            hatecheck_target_functions=len(ftarget),
            identity_views=len(identity),
            reference_links=len(links),
            diagnostic_draw_values=drawn,
            case_slots=len(cases),
            selected_slots=sum(bool(v["key"]) for v in cases),
            unique_case_inputs=len(selected),
            direct_result_hashes=len(focused),
            new_Qwen_calls=0,
            new_training=0,
        ),
    )
    print(
        "Part 5 source and functional calculations complete; independent review follows.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["freeze", "run"], required=True)
    args = p.parse_args()
    with r.original.offline(), threadpool_limits(limits=2):
        {"freeze": freeze, "run": run}[args.mode]()
