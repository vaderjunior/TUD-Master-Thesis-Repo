from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.broad import sources as s

checked = 0
maximum = 0.0
draw_values = 0
views = 0
draw_columns = {}


def equal(a, b, label="value"):
    global checked, maximum
    if isinstance(a, dict):
        for k, v in a.items():
            s.require(k in b, label + "/" + k + " absent")
            equal(v, b[k], label + "/" + k)
    elif isinstance(a, (list, tuple)):
        s.require(len(a) == len(b), label + " length")
        for i, (v, w) in enumerate(zip(a, b)):
            equal(v, w, label + "/" + str(i))
    elif isinstance(a, np.ndarray):
        z = np.asarray(b, dtype=float)
        s.require(a.shape == z.shape, label + " shape")
        s.require(np.array_equal(np.isfinite(a), np.isfinite(z)), label + " undefined")
        diff = np.abs(a[np.isfinite(a)] - z[np.isfinite(z)])
        mx = float(diff.max(initial=0))
        s.require(mx < 2e-10, label + " numeric array")
        maximum = max(maximum, mx)
        checked += a.size
    elif isinstance(a, (int, float, np.number)) and not isinstance(a, (bool, np.bool_)):
        if not np.isfinite(a):
            s.require(b is None or not np.isfinite(b), label + " undefined")
            return
        s.require(b is not None and abs(float(a) - float(b)) < 2e-10, label + " number")
        maximum = max(maximum, abs(float(a) - float(b)))
        checked += 1
    else:
        s.require(a == b, label + " differs")


def div(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=b != 0)
    return out


def calc(t, kind, labels, macro):
    if kind == "binary":
        a, b, c, d = np.moveaxis(t, -1, 0)
        n = a + b + c + d
        pos = div(2 * a, 2 * a + b + c)
        neg = div(2 * d, 2 * d + b + c)
        return dict(
            tp=a,
            fp=b,
            fn=c,
            tn=d,
            n=n,
            positive_gold=a + c,
            negative_gold=b + d,
            fpr=div(b, b + d),
            fnr=div(c, c + a),
            precision=div(a, a + b),
            recall=div(a, a + c),
            error_rate=div(b + c, n),
            accuracy=div(a + d, n),
            predicted_hate=div(a + b, n),
            balanced_accuracy=(div(a, a + c) + div(d, d + b)) / 2,
            macro_f1=np.where(
                n > 0,
                (np.nan_to_num(pos, nan=0) + np.nan_to_num(neg, nan=0)) / 2,
                np.nan,
            ),
        )
    if kind == "links":
        return dict(
            n=t[..., -1],
            **{
                k: div(t[..., i], t[..., -1])
                for i, k in enumerate(
                    (
                        "both_correct",
                        "both_wrong",
                        "child_only_wrong",
                        "parent_only_wrong",
                        "expected_relation",
                    )
                )
            }
        )
    z = t[..., : 3 * len(labels)].reshape(*t.shape[:-1], len(labels), 3)
    tp = z[..., 0]
    fp = z[..., 1]
    fn = z[..., 2]
    ff = np.nan_to_num(div(2 * tp, 2 * tp + fp + fn), nan=0)
    n = t[..., -1]
    idx = [labels.index(x) for x in macro]
    out = dict(
        n=n,
        macro_f1=ff[..., idx].mean(-1) if idx else np.full(n.shape, np.nan),
        label_tp=tp,
        label_fp=fp,
        label_fn=fn,
        label_precision=div(tp, tp + fp),
        label_recall=div(tp, tp + fn),
        label_f1=ff,
    )
    out.update(
        {
            k: div(t[..., 3 * len(labels) + i], n)
            for i, k in enumerate(
                (
                    "gate_aware_exact",
                    "exact_set",
                    "sample_jaccard",
                    "extra_labels",
                    "missing_labels",
                    "gate_closed",
                )
            )
        }
    )
    return out


def summary(d, ss, mask, weights=None):
    vv = d["f"][ss][:, mask]
    totals = vv.sum(1) if weights is None else (vv * weights[None, :, None]).sum(1)
    v = calc(totals, d["kind"], d["labels"], d["macro"])
    if d["kind"] == "binary" and len(set(d["gold"][mask, 1])) < 2:
        v["macro_f1"] = np.full(len(ss), np.nan)
    return s.safe(
        dict(
            repeats=[{k: a[j] for k, a in v.items()} for j in range(len(ss))],
            mean={k: a.mean(0) for k, a in v.items()},
            sd={k: a.std(0, ddof=1) for k, a in v.items()} if len(ss) > 1 else {},
        )
    )


def boot(d, ss, mask, info):
    global draw_values
    keys = (
        ["macro_f1", "fpr", "fnr", "error_rate"]
        if d["kind"] == "binary" and len(info["metrics"]) == 4
        else (
            ["macro_f1", "gate_aware_exact", "sample_jaccard"]
            if d["kind"] == "fine" and len(info["metrics"]) == 3
            else (
                ["both_correct", "both_wrong", "expected_relation"]
                if d["kind"] == "links"
                else list(info["metrics"])
            )
        )
    )
    s.require(set(keys) == set(info["metrics"]), "Unexpected saved metric columns")
    draw_columns[info["path"]] = keys
    unit = info["resampling"]
    cc = np.asarray(d["clusters"] if unit == "component" else d["ids"])[mask].tolist()
    unique = sorted(set(cc))
    ix = defaultdict(list)
    for i, key in enumerate(cc):
        ix[key].append(i)
    a = d["f"][ss][:, mask].transpose(1, 0, 2)
    totals = np.array([a[ix[key]].sum(0) for key in unique])
    rng = np.random.default_rng(20261008)
    chunks = []
    for start in range(0, 10000, 8):
        picks = rng.choice(
            len(unique), size=(min(8, 10000 - start), len(unique)), replace=True
        )
        t = totals[picks].sum(1)
        v = calc(t, d["kind"], d["labels"], d["macro"])
        chunks.append(
            np.stack([v[k][:, :3].mean(1) - v[k][:, 3:].mean(1) for k in keys], axis=1)
        )
    draws = np.concatenate(chunks)
    with np.load(s.ROOT / info["path"], allow_pickle=False) as z:
        equal(draws, z["draws"], "draws " + info["path"])
        equal(np.flatnonzero(mask), z["common_indices"], "draw mask")
    point = calc(a.sum(0), d["kind"], d["labels"], d["macro"])
    for j, key in enumerate(keys):
        invalid = int((~np.isfinite(draws[:, j])).sum())
        equal(
            dict(
                difference=s.safe(point[key][:3].mean() - point[key][3:].mean()),
                ci95=(
                    np.quantile(draws[:, j], [0.025, 0.975]).tolist()
                    if not invalid
                    else None
                ),
                undefined_draws=invalid,
            ),
            info["metrics"][key],
            "interval " + key,
        )
    equal(
        dict(n=int(mask.sum()), units=len(unique), draws=10000, seed=20261008),
        info,
        "bootstrap population",
    )
    draw_values += draws.size


def check_group(d, g):
    global views
    indices = g["indices"]
    select = np.zeros(len(d["ids"]), bool)
    select[indices] = True
    common = select & d["valid"].all(0)
    equal(
        dict(
            endpoint=d["name"],
            kind=d["kind"],
            eligible=len(indices),
            common_nine_n=int(common.sum()),
            common_indices=np.flatnonzero(common),
            labels=d["labels"],
            macro_labels=d["macro"],
            source_counts=dict(Counter(d["rows"][i]["source"] for i in indices)),
            common_clusters=len(set(np.asarray(d["clusters"])[common])),
        ),
        g,
        g["id"],
    )
    if d["kind"] != "links":
        equal(
            dict(
                full_gold_support=d["gold"][select].sum(0),
                shared_gold_support=d["gold"][common].sum(0),
            ),
            g,
            "supports",
        )
    for ci, co in enumerate(s.CONDITIONS):
        ss = list(range(3 * ci, 3 * ci + 3))
        equal(summary(d, ss, common), g["common_nine"][co], "common summary")
        for j, setting in enumerate(ss):
            mask = select & d["valid"][setting]
            one = summary(d, [setting], mask)["repeats"][0]
            one.update(
                eligible=len(indices),
                invalid=int((select & ~d["valid"][setting]).sum()),
            )
            rk = (
                "accuracy"
                if d["kind"] == "binary"
                else "both_correct" if d["kind"] == "links" else "gate_aware_exact"
            )
            rate = one[rk]
            n = int(mask.sum())
            one["exact_or_accuracy_failure_bounds"] = (
                [rate * n / len(indices), (rate * n + len(indices) - n) / len(indices)]
                if rate is not None
                else [0.0, 1.0]
            )
            equal(one, g["own_valid"][co][j], "own summary")
    for pair in g["pairs"]:
        ci = s.CONDITIONS.index(pair["reference"])
        ss = list(range(3)) + list(range(3 * ci, 3 * ci + 3))
        mask = select & d["valid"][ss].all(0)
        equal(
            dict(
                n=int(mask.sum()),
                common_indices=np.flatnonzero(mask),
                full=summary(d, ss[:3], mask),
                reference_summary=summary(d, ss[3:], mask),
            ),
            pair,
            "pair summary",
        )
        for info in pair["intervals"].values():
            boot(d, ss, mask, info)
    views += 1


def rebuild_features(d):
    y, p, g = d["gold"], d["predicted"], d["gate"]
    out = []
    for j in range(9):
        rows = []
        for i in range(len(y)):
            if d["kind"] == "binary":
                truth = bool(y[i, 1])
                pred = bool(p[j, i, 1])
                rows.append(
                    [
                        truth and pred,
                        not truth and pred,
                        truth and not pred,
                        not truth and not pred,
                    ]
                )
                continue
            truth = set(np.flatnonzero(y[i]))
            pred = set(np.flatnonzero(p[j, i]))
            flat = []
            for k in range(len(d["labels"])):
                flat.extend(
                    [
                        k in truth and k in pred,
                        k not in truth and k in pred,
                        k in truth and k not in pred,
                    ]
                )
            same = truth == pred
            union = truth | pred
            flat.extend(
                [
                    same and (d["kind"] == "legal" or g[j, i]),
                    same,
                    len(truth & pred) / len(union) if union else 1,
                    len(pred - truth),
                    len(truth - pred),
                    not g[j, i],
                    1,
                ]
            )
            rows.append(flat)
        out.append(rows)
    return np.asarray(out, float)


def annotation(row):
    v = json.loads(row["raw"])
    source = row["source"]
    if source == "hatexplain":
        return source + "/" + max(set(v["votes"]), key=v["votes"].count)
    if source == "implicit_hate":
        return source + "/" + v["stage1_class"]
    if source == "gahd":
        return source + "/" + v["round"]
    x = v["score"] if source == "mhs" else v["hate_speech_score"]
    if source == "mhs":
        labels = [
            "score<=0",
            "0<score<=0.5",
            "0.5<score<=1.0",
            "1.0<score<=1.63",
            "score>1.63",
        ]
        j = sum(x > cut for cut in [0, 0.5, 1, 1.63])
    else:
        labels = ["score=0", "0<score<0.5", "score=0.5", "0.5<score<1", "score=1"]
        j = 0 if x == 0 else 1 if x < 0.5 else 2 if x == 0.5 else 3 if x < 1 else 4
    return source + "/" + labels[j]


def run():
    receipt = s.c.read(s.BASE / "analysis_complete.json")
    s.r.verify(receipt["sources"])
    s.r.verify(receipt["outputs"])
    data = s.c.read(s.BASE / "endpoint_metadata.json")
    setup = s.c.read(s.reader.BASE / "endpoint_setup.json")
    for name, d in data.items():
        with np.load(s.ROOT / d["path"], allow_pickle=False) as z:
            for k in ("gold", "predicted", "gate", "valid"):
                d[k] = z[k].copy()
            stored = z["features"].copy()
        phase, dim = name.split("/")
        truth = []
        for row in d["rows"]:
            value = (
                {"hate" if row["gate"] else "not_hate"}
                if dim == "hate"
                else (
                    {row["severity"]}
                    if dim == "severity"
                    else set(row[s.q.FIELDS[dim]])
                )
            )
            truth.append([int(l in value) for l in d["labels"]])
        equal(np.array(truth), d["gold"], "raw gold " + name)
        equal(d["ids"], [x["key"] for x in d["rows"]], "row ordering")
        equal(d["labels"], setup[name]["all_labels"], "all labels")
        equal(d["macro"], setup[name]["macro_labels"], "fixed macro")
        original = s.c.read(
            s.q.main.BASE / "endpoints" / s.q.main.slug(name) / "results.json"
        )
        ii = [list(map(tuple, original["settings"])).index(x) for x in s.SETTINGS]
        with np.load(s.ROOT / original["encoded_path"], allow_pickle=False) as z:
            equal(z["gold"], d["gold"], "previous gold")
            equal(z["predicted"][ii], d["predicted"], "previous predictions")
            equal(z["valid"][ii], d["valid"], "previous validity")
        d["f"] = rebuild_features(d)
        equal(d["f"], stored, "independent row features " + name)
    hc = data["en_hatecheck_eval/hate"]
    meta = [json.loads(x["raw"]) for x in hc["rows"]]
    st = s.c.read(s.BASE / "hatecheck_structure.json")
    bycase = {str(int(x["case_id"])): i for i, x in enumerate(meta)}
    neighbors = [set() for _ in meta]
    units = {}
    for field in ("templ_id", "text_cluster"):
        gg = defaultdict(list)
        for i, row in enumerate(hc["rows"]):
            gg[str(meta[i][field]) if field == "templ_id" else row[field]].append(i)
        for ids in gg.values():
            for i in ids:
                neighbors[i].update(ids)
        units[field] = gg
    links = []
    for i, v in enumerate(meta):
        if v["ref_case_id"] is None:
            continue
        j = bycase[str(int(v["ref_case_id"]))]
        neighbors[i].add(j)
        neighbors[j].add(i)
        links.append(
            dict(
                child=i,
                parent=j,
                child_key=hc["ids"][i],
                parent_key=hc["ids"][j],
                function=v["functionality"],
                same_gold=bool(hc["gold"][i, 1] == hc["gold"][j, 1]),
                cluster=hc["clusters"][i],
            )
        )
    equal(links, st["links"], "independent links")
    unseen = set(range(len(meta)))
    components = []
    while unseen:
        visit = [min(unseen)]
        seen = set()
        while visit:
            i = visit.pop()
            if i in seen:
                continue
            seen.add(i)
            visit.extend(neighbors[i] - seen)
        unseen -= seen
        components.append(frozenset(seen))
    s.require(
        set(components)
        == {frozenset(v["indices"]) for v in st["groups"]["component"].values()},
        "Independent component graph mismatch",
    )
    equal(845, st["counts"]["template"]["units"], "templates")
    equal(570, len(components), "components")
    equal(2046, len(links), "reference links")
    linked = s.c.read(s.BASE / "linked_metadata.json")
    linked["valid"] = np.array(
        [
            [hc["valid"][j, x["child"]] and hc["valid"][j, x["parent"]] for x in links]
            for j in range(9)
        ]
    )
    f = []
    for j in range(9):
        rows = []
        for x in links:
            a, b = x["child"], x["parent"]
            righta = hc["predicted"][j, a, 1] == hc["gold"][a, 1]
            rightb = hc["predicted"][j, b, 1] == hc["gold"][b, 1]
            relation = (hc["predicted"][j, a, 1] == hc["predicted"][j, b, 1]) == x[
                "same_gold"
            ]
            rows.append(
                [
                    righta and rightb,
                    not righta and not rightb,
                    not righta and rightb,
                    righta and not rightb,
                    relation,
                    1,
                ]
            )
        f.append(rows)
    linked["f"] = np.array(f, float)
    data[linked["name"]] = linked
    with np.load(s.BASE / "encoded/linked.npz", allow_pickle=False) as z:
        equal(linked["f"], z["features"], "linked features")
        equal(linked["valid"], z["valid"], "linked validity")
    collections = {
        name: s.c.read(s.BASE / (name + ".json"))
        for name in (
            "natural_binary",
            "annotation_slices",
            "fine_sources",
            "hatecheck_functions",
            "hatecheck_target_functions",
            "hatecheck_identities",
            "hatecheck_direction",
            "hatecheck_linked",
        )
    }
    for filename, values in collections.items():
        for key, g in values.items():
            d = data[g["endpoint"]]
            if filename in ("natural_binary", "fine_sources"):
                expected = [
                    i
                    for i, row in enumerate(d["rows"])
                    if row["source"] == key.rsplit("/", 1)[1]
                ]
            elif filename in ("hatecheck_functions", "hatecheck_target_functions"):
                expected = [
                    i
                    for i, row in enumerate(d["rows"])
                    if json.loads(row["raw"])["functionality"] == key
                ]
            elif filename == "annotation_slices":
                if key.startswith(("en_test/", "de_test/")):
                    expected = [
                        i
                        for i, row in enumerate(d["rows"])
                        if d["name"] + "/" + annotation(row) == key
                    ]
                elif key.startswith("hatexplain_agreement/"):
                    expected = [
                        i
                        for i, row in enumerate(d["rows"])
                        if row["source"] == "hatexplain"
                        and (len(set(json.loads(row["raw"])["votes"])) == 1)
                        == key.endswith("/unanimous")
                    ]
                elif key.startswith("hatexplain_label_agreement/"):
                    _, lab, agreement = key.split("/")
                    expected = [
                        i
                        for i, row in enumerate(d["rows"])
                        if row["source"] == "hatexplain"
                        and annotation(row) == "hatexplain/" + lab
                        and (len(set(json.loads(row["raw"])["votes"])) == 1)
                        == (agreement == "unanimous")
                    ]
                else:
                    expected = [
                        i
                        for i, row in enumerate(d["rows"])
                        if row["source"] == "gahd"
                        and json.loads(row["raw"])["official_split"]
                        == key.rsplit("/", 1)[1]
                    ]
            elif filename == "hatecheck_direction":
                expected = [
                    i
                    for i, row in enumerate(d["rows"])
                    if json.loads(row["raw"])["direction"] == key
                ]
            elif filename == "hatecheck_linked":
                expected = [
                    i
                    for i, x in enumerate(links)
                    if key == "all"
                    or key == "same_gold"
                    and x["same_gold"]
                    or key == "changed_gold"
                    and not x["same_gold"]
                    or key.startswith("function/")
                    and x["function"] == key.split("/", 1)[1]
                ]
            else:
                ident, kind = key.rsplit("/", 1)
                expected = []
                for i, row in enumerate(d["rows"]):
                    v = json.loads(row["raw"])["target_ident"]
                    tag = (
                        "not_recorded"
                        if v is None or str(v).lower() in ("", "nan", "none")
                        else str(v)
                    )
                    if tag == ident and (
                        kind == "target_exact" or str(row["gate"]) == kind
                    ):
                        expected.append(i)
            equal(expected, g["indices"], "membership " + key)
            check_group(d, g)
        print("Independently checked", filename, len(values), "views", flush=True)
    weights = s.c.read(s.BASE / "hatecheck_weighting.json")
    schemes = {
        "text": hc["ids"],
        "function": [v["functionality"] for v in meta],
        "template": [str(int(v["templ_id"])) for v in meta],
        "component": hc["clusters"],
    }
    for view, ss in [
        ("common_nine", list(range(9))),
        ("full_small", list(range(6))),
        ("full_zero", [0, 1, 2, 6, 7, 8]),
    ]:
        mask = hc["valid"][ss].all(0)
        for scheme, uu in schemes.items():
            selected = np.asarray(uu)[mask].tolist()
            counts = Counter(selected)
            w = np.array([1 / counts[v] for v in selected])
            w = w / w.sum()
            got = weights[view][scheme]
            equal(
                dict(
                    n=int(mask.sum()),
                    full_units=len(set(uu)),
                    retained_units=len(counts),
                    empty_units=len(set(uu)) - len(counts),
                    hateful_weight=float(np.sum(w * hc["gold"][mask, 1])),
                ),
                got,
                "weights",
            )
            for co, summary_saved in got["conditions"].items():
                ci = s.CONDITIONS.index(co)
                equal(
                    summary(hc, list(range(ci * 3, ci * 3 + 3)), mask, w),
                    summary_saved,
                    "weighted summary",
                )

    old = s.c.read(s.q.errors.BASE / "binary_results.json")
    reconciliation = []
    for phase in ("en_test", "de_test", "en_hatecheck_eval"):
        groups = (
            list(collections["hatecheck_functions"].values())
            if phase == "en_hatecheck_eval"
            else [
                g
                for g in collections["natural_binary"].values()
                if g["endpoint"] == phase + "/hate"
            ]
        )
        for co in s.CONDITIONS:
            count = {
                k: sum(g["common_nine"][co]["mean"][k] for g in groups)
                for k in ("tp", "fp", "fn", "tn", "n")
            }
            equal(
                count,
                old[phase]["common_nine"][co]["mean"],
                "prior Part 3 partition " + phase + "/" + co,
            )
            reconciliation.append(phase + "/" + co)

    selection = s.c.read(s.BASE / "case_selection.json")["slots"]
    case_rows = s.c.read(s.BASE / "cases.json")
    for slot, case in zip(selection, case_rows):
        d = data[slot["endpoint"]]
        candidates = []
        category = slot["category"]
        if category == "full_error":
            ids = [
                i
                for i, row in enumerate(d["rows"])
                if json.loads(row["raw"])["functionality"] == slot["function"]
                and d["valid"][:3, i].all()
            ]
            error = {
                i: int(
                    sum(d["predicted"][j, i, 1] != d["gold"][i, 1] for j in range(3))
                )
                for i in ids
            }
            maxi = max(error.values(), default=0)
            candidates = [
                d["ids"][i] for i, n in error.items() if maxi > 0 and n == maxi
            ]
            equal(maxi, slot["full_error_repeats"], "case max errors")
        else:
            for i in range(len(d["ids"])):
                if not d["valid"][:6, i].all():
                    continue
                if category.startswith("linked_"):
                    if links[i]["same_gold"] != (slot["relation"] == "same_gold"):
                        continue
                    correct = d["f"][:, i, 0].astype(bool)
                else:
                    if (
                        d["rows"][i]["source"] != slot["source"]
                        or bool(d["gold"][i, 1]) != slot["gold"]
                    ):
                        continue
                    correct = d["predicted"][:, i, 1] == d["gold"][i, 1]
                isfix = category.endswith("fix")
                a = correct[:3]
                b = correct[3:6]
                if (a.all() and not b.any()) if isfix else (not a.any() and b.all()):
                    candidates.append(d["ids"][i])
        h = lambda key: hashlib.sha256(
            ("sq2-broad-part5-v1|" + slot["slot"] + "|" + key).encode()
        ).hexdigest()
        chosen = (
            sorted(candidates, key=lambda key: (h(key), key))[0] if candidates else None
        )
        equal(
            dict(
                candidates=len(candidates),
                key=chosen,
                selection_sha256=h(chosen) if chosen else None,
            ),
            slot,
            "case selection",
        )
        equal(slot, {k: case[k] for k in slot}, "case metadata")
        for member in case["members"]:
            ep = data[member["gold"]["phase"] + "/hate"]
            i = ep["ids"].index(member["key"])
            for prediction in member["predictions"]:
                j = s.SETTINGS.index((prediction["condition"], prediction["repeat"]))
                got = prediction["result"]
                equal(bool(got is not None), bool(ep["valid"][j, i]), "case valid")
                if got is not None:
                    equal(
                        bool(got["hate"]),
                        bool(ep["predicted"][j, i, 1]),
                        "case decision",
                    )
    s.r.verify(s.c.read(s.BASE / "focused_source_bindings.json"))
    s.r.verify(receipt["outputs"])
    s.r.verify(receipt["sources"])
    s.save(
        s.BASE / "draw_column_layout.json",
        dict(
            note="Explicit saved-array column order from the frozen producer, independently reconstructed; JSON metric dictionaries are alphabetically serialized and do not define array order.",
            paths=draw_columns,
        ),
    )
    s.save(
        s.BASE / "independent_review.json",
        dict(
            status="all_source_function_link_weight_case_and_draw_checks_passed",
            numeric_values_checked=checked,
            maximum_absolute_difference=maximum,
            diagnostic_draw_values=draw_values,
            group_views=views,
            previous_global_partitions=reconciliation,
            reference_graph_rebuilt=True,
            case_slots_checked=len(selection),
            sources=s.bindings(
                [
                    Path(__file__),
                    s.BASE / "analysis_complete.json",
                    s.BASE / "draw_column_layout.json",
                ]
            )
            | receipt["outputs"],
            new_Qwen_calls=0,
        ),
    )
    print(
        "Independent review complete:",
        checked,
        "values;",
        draw_values,
        "draw values;",
        views,
        "views",
        flush=True,
    )


if __name__ == "__main__":
    with s.r.original.offline(), threadpool_limits(limits=2):
        run()
