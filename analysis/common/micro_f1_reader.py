"""Common micro f1 reader."""

from analysis.common import micro_f1_files as c
from analysis.common import micro_f1 as metric
import json
import traceback
from collections import Counter
from datetime import datetime, timezone
import numpy as np

B = c.ROOT / "experiments/astra"
CHECKS = Counter()
MAX_ERROR = 0.0
RESULTS = []


def equal(actual, expected, name):
    global MAX_ERROR
    if actual is None or expected is None:
        c.require(actual is None and expected is None, name + ": unsupported mismatch")
    else:
        a, b = (np.asarray(actual, float), np.asarray(expected, float))
        c.require(a.shape == b.shape, name + ": shape mismatch")
        c.require(np.isfinite(a).all() and np.isfinite(b).all(), name + ": nonfinite")
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        MAX_ERROR = max(MAX_ERROR, error)
        c.require(error < 1e-12, name + ": difference " + str(error))
    CHECKS[name.split(":")[0]] += 1


def counts(gold, predicted, valid=None):
    rawg, rawp = (np.asarray(gold), np.asarray(predicted))
    g, p = (rawg.astype(bool), rawp.astype(bool))
    v = np.ones(len(g), bool) if valid is None else np.asarray(valid, bool)
    c.require(g.shape == p.shape and v.shape == (len(g),), "Count shapes differ")
    c.require(
        np.isin(rawg, [0, 1]).all() and np.isin(rawp[v], [0, 1]).all(), "Nonbinary valid label bits"
    )
    return np.stack(
        ((g & p & v[:, None]).sum(0), (~g & p & v[:, None]).sum(0), (g & ~p & v[:, None]).sum(0)),
        axis=-1,
    )


def scores(gold, predictions, valid, labels, macro_labels):
    ix = [labels.index(x) for x in macro_labels]
    return [metric.macro(counts(gold, p, v), ix) for p, v in zip(predictions, valid)]


def arrays(path):
    with np.load(c.bind(path), allow_pickle=False) as z:
        return {
            k: z[k] for k in z.files if not k.startswith("bootstrap") and (not k.endswith("draws"))
        }


def ids_on(ids, mask):
    return [key for key, yes in zip(ids, mask) if yes]


def bits(values, labels):
    c.require(set(values).issubset(labels), "Unknown label")
    return [x in values for x in labels]


def known(value):
    return value is not None and (not (isinstance(value, float) and np.isnan(value)))


def legacy_data(u):
    import pyarrow.parquet as pq

    data = c.read(c.ROOT / u["source"])
    result = data["endpoints"][u["endpoint"]] if "endpoints" in data else data
    ep = result["endpoint"]
    labels = ep["all_labels"]
    dim = ep["dimension"]
    field = {"hate_type": "hate_types", "target_group": "target_groups", "legal": "legal"}[dim]
    rows = pq.read_table(c.bind(c.ROOT / u["gold_path"])).to_pylist()
    eligible = [r for r in rows if known(r[field]) and (dim == "legal" or r["gate"] is True)]
    ids = [str(r["id"]) for r in eligible]
    c.require(ids == ep["scorable_ids"], "Original gold mask differs")
    g = np.asarray([bits(r[field], labels) for r in eligible], bool)
    equal(g.sum(0), [ep["gold_support"][x] for x in labels], "legacy gold support")
    prepared = {}
    for path, binding in u["assembled"].items():
        p = c.ROOT / path
        c.require(c.sha(p) == binding["sha256"], "Assembled source SHA differs")
        raw = list(c.lines(p))
        c.require(len(raw) == binding["rows"], "Assembled rows differ")
        by_id = {str(r["id"]): r for r in raw}
        c.require(len(by_id) == len(raw), "Duplicate IDs")
        key = (raw[0]["condition"], raw[0]["replicate"])
        c.require(key not in prepared, "Repeated setting")
        pred = np.zeros_like(g)
        valid = np.zeros(len(g), bool)
        for i, identifier in enumerate(ids):
            r = by_id[identifier]
            ans = r.get("result")
            c.require((r["condition"], r["replicate"]) == key, "Mixed setting")
            if ans is not None and ans.get(dim) is not None:
                valid[i] = True
                pred[i] = bits(ans[dim], labels)
                if dim != "legal" and ans.get("hate") is False:
                    c.require(not pred[i].any(), "Non-hate fine labels were not cleared")
        prepared[key] = (pred, valid)
    return (result, ep, ids, g, prepared)


def verify_legacy(u):
    result, ep, ids, g, prepared = legacy_data(u)
    labels = ep["all_labels"]
    ix = [labels.index(x) for x in ep["macro_labels"]]
    for cond, item in result["conditions"].items():
        own = []
        for run in item["runs"]:
            p, v = prepared[cond, run["replicate"]]
            old = run["score"]
            z = counts(g, p, v)
            equal(z, old["per_label_counts"], "legacy raw counts")
            value = metric.macro(z, ix)
            own.append(value)
            equal(value, old["macro_f1_valid_only"], "legacy own macro")
            equal(int(v.sum()), old["n_valid"], "legacy valid rows")
        if item.get("own_valid_summary") and ix:
            equal(own, item["own_valid_summary"]["values"], "legacy repeats")
    if "paired_comparison" in result:
        views = result["paired_comparison"]
        if "contrasts" in views:
            views = {"paired": views}
        for name, view in views.items():
            selected = [k for k in prepared if k[0] in view["conditions"]]
            mask = np.logical_and.reduce([prepared[k][1] for k in selected])
            c.require(
                ids_on(ids, mask) == view["common_valid_ids"], "Legacy shared valid mask differs"
            )
            means = {}
            for cond, old in view["conditions"].items():
                values = [
                    metric.macro(counts(g, prepared[k][0], mask), ix)
                    for k in sorted(selected)
                    if k[0] == cond
                ]
                (
                    equal(values, old["common_valid_summary"]["values"], "legacy common macro")
                    if ix
                    else None
                )
                means[cond] = float(np.mean(values)) if ix else None
            if ix:
                groupmeans = {}
                for name, old in view.get("groups", {}).items():
                    members = [v for k, v in means.items() if k == name or k.startswith(name + "_")]
                    c.require(members, "No group members")
                    groupmeans[name] = float(np.mean(members))
                    equal(groupmeans[name], old["mean"], "legacy pool mean")
                for old in view["contrasts"].values():
                    if old.get("weights"):
                        point = sum((means[k] * w for k, w in old["weights"].items()))
                    elif (
                        old.get("left") in groupmeans | means
                        and old.get("right") in groupmeans | means
                    ):
                        all_means = means | groupmeans
                        point = all_means[old["left"]] - all_means[old["right"]]
                    else:
                        continue
                    equal(point, old["mean_difference"], "legacy contrast point")
    else:
        for name, old in result["contrasts"].items():
            groups = result.get("fewshot_groups", {})

            def members(name):
                return [name] if name in result["conditions"] else groups[name]["conditions"]

            left = members(old["left"])
            right = members(old["right"])
            selected = [k for cond in left + right for k in sorted(prepared) if k[0] == cond]
            mask = np.logical_and.reduce([prepared[k][1] for k in selected])
            c.require(ids_on(ids, mask) == old["common_valid_ids"], "Core paired mask differs")
            vals = [metric.macro(counts(g, prepared[k][0], mask), ix) for k in selected]
            if ix:
                equal(vals, old["per_setting_f1"], "core paired macro")
                n = len(left) * 3
                equal(
                    float(np.mean(vals[:n]) - np.mean(vals[n:])),
                    old["mean_difference"],
                    "core contrast point",
                )
    return dict(
        depth="raw assembled answers plus original parquet annotations",
        rows=len(g),
        settings=len(prepared),
    )


def verify_broad(u):
    d = c.read(c.ROOT / u["source"])
    ep = d["endpoint"]
    labels = ep["all_labels"]
    ix = [labels.index(x) for x in ep["macro_labels"]]
    p = c.ROOT / d["encoded_path"]
    c.require(c.sha(p) == d["encoded_sha256"], "Encoded hash differs")
    a = arrays(p)
    g = a["gold"]
    pred = a["predicted"]
    valid = a["valid"]
    settings = [tuple(x) for x in d["settings"]]
    equal(g.sum(0), [ep["gold_support"][x] for x in labels], "broad gold support")
    for i, (cond, rep) in enumerate(settings):
        z = counts(g, pred[i], valid[i])
        old = d["own_valid"][cond]["repeats"][rep]
        equal(z, a["counts"][i].sum(0), "broad saved counts")
        equal(z, old["per_label_counts"], "broad own counts")
        equal(metric.macro(z, ix), old["macro_f1_valid_only"], "broad own macro")
    for old in d["contrasts"]:
        indices = [
            i
            for cond in (old["left"], old["right"])
            for i, k in enumerate(settings)
            if k[0] == cond
        ]
        mask = valid[indices].all(0)
        c.require(
            ids_on(ep["scorable_ids"], mask) == old["common_valid_ids"], "Broad paired IDs differ"
        )
        vals = [metric.macro(counts(g, pred[i], mask), ix) for i in indices]
        if ix:
            equal(vals, old["common_f1_by_setting"], "broad paired macro")
            equal(
                float(np.mean(vals[:3]) - np.mean(vals[3:])),
                old["mean_difference"],
                "broad contrast point",
            )
    mask = valid[
        [i for i, k in enumerate(settings) if k[0] in {"rag_full", "rag_small", "zero_shot"}]
    ].all(0)
    equal(int(mask.sum()), d["three_condition_common"]["n_common_valid"], "broad nine mask")
    for cond, old in d["three_condition_common"]["per_condition"].items():
        vals = [
            metric.macro(counts(g, pred[i], mask), ix)
            for i, k in enumerate(settings)
            if k[0] == cond
        ]
        if ix:
            equal(vals, old["repeat_macro_f1"], "broad nine macro")
    return dict(
        depth="sealed encoded gold and predictions, counts and original paired masks",
        rows=len(g),
        settings=len(pred),
    )


def verify_gpu(u):
    meta = c.read(c.ROOT / u["metadata_path"])
    a = arrays(c.ROOT / u["prediction_path"])
    labels = meta["labels"]
    ix = meta["supported_indices"]
    g = a["truth"]
    pred = a["predictions"]
    if u["study"] == "gpu":
        point = arrays(c.ROOT / u["count_path"])
        for i, p in enumerate(pred):
            z = counts(g, p)
            equal(z, np.stack([point[k][i] for k in ("tp", "fp", "fn")], axis=-1), "gpu raw counts")
            equal(metric.macro(z, ix), point["scores"][i], "gpu macro")
        d = c.read(c.ROOT / u["source"])
        for old in d["absolute"]:
            selected = [
                i
                for i, m in enumerate(meta["models"])
                if m["budget"] == old["budget"] and m["method"] == old["method"]
            ]
            pools = sorted({meta["models"][i]["pool"] for i in selected})
            means = [
                float(
                    np.mean(
                        [point["scores"][i] for i in selected if meta["models"][i]["pool"] == pool]
                    )
                )
                for pool in pools
            ]
            equal(float(np.mean(means)), old["mean"], "gpu hierarchical mean")
    else:
        old = c.read((c.ROOT / u["source"]).parent / "model_view_scores.json")[meta["endpoint"]]
        c.require(len(old) == len(pred), "Control settings differ")
        for p, m, row in zip(pred, meta["models"], old):
            c.require(m["id"] == row["id"], "Control order differs")
            z = counts(g, p)
            equal(z, np.stack([row[k] for k in ("tp", "fp", "fn")], axis=-1), "gpu control counts")
            equal(metric.macro(z, ix), row["macro_f1"], "gpu control macro")
    equal(
        g.sum(0),
        (
            [meta["gold_support"][x] for x in labels]
            if isinstance(meta["gold_support"], dict)
            else meta["gold_support"]
        ),
        "gpu gold support",
    )
    return dict(
        depth="saved per-text prediction arrays and explicit fit/pool metadata",
        rows=len(g),
        settings=len(pred),
    )


def verify_five(u):
    a = arrays(c.ROOT / u["prediction_path"])
    d = c.read(c.ROOT / u["source"])
    ix = d["supported_indices"]
    for name in ("old", "new"):
        for i, p in enumerate(a[name]):
            z = counts(a["truth"], p)
            equal(metric.macro(z, ix), a[name + "_scores"][i], "five fit macro")
            if name == "new":
                equal(
                    z,
                    np.stack([a["new_" + k][i] for k in ("tp", "fp", "fn")], axis=-1),
                    "five fit counts",
                )
    return dict(
        depth="separate original and retrained prediction arrays", rows=len(a["truth"]), settings=10
    )


def verify_combined(u):
    folder = (c.ROOT / u["source"]).parent
    meta = c.read(folder / "metadata.json")
    d = c.read(folder / "scores.json")
    a = arrays(folder / "aligned.npz")
    c.require(len(a["predictions"]) == len(meta["models"]), "Combined model count differs")
    for mode in ("own_valid", "common_nine"):
        old = d[mode]["runs"]
        for i, p in enumerate(a["predictions"]):
            mask = a["valid"][i] if mode == "own_valid" else a["common"]
            z = counts(a["gold"], p, mask)
            row = old[i]
            equal(z, np.stack([row[k] for k in ("tp", "fp", "fn")], axis=-1), "combined counts")
            equal(metric.macro(z, meta["macro_indices"]), row["macro_f1"], "combined macro")
    return dict(
        depth="saved aligned arrays, original own/common masks",
        rows=len(a["gold"]),
        settings=len(meta["models"]),
    )


def verify_german(u):
    d = c.read(c.ROOT / u["source"])
    meta = c.read(c.ROOT / u["metadata_path"])
    a = arrays(c.ROOT / u["prediction_path"])
    ix = [d["labels"].index(x) for x in d["macro_labels"]]
    mask = a["common_mask"]
    c.require(ids_on(meta["keys"], mask) == d["common_keys"], "German common IDs differ")
    c.require((a["predictions"][:, :, mask] >= 0).all(), "Invalid answer in common mask")
    for j, cond in enumerate(meta["conditions"]):
        for rep in range(3):
            z = counts(a["gold"], a["predictions"][j, rep], mask)
            equal(
                metric.macro(z, ix),
                d["common"][cond]["repeats"][rep]["macro_f1"],
                "german common macro",
            )
            order = meta["confusion_order"]
            equal(
                z,
                a["confusion_counts"][j, rep][:, [order.index(k) for k in ("tp", "fp", "fn")]],
                "german counts",
            )
    return dict(
        depth="saved legal arrays, all-nine valid mask and legal-independent-gate rule",
        rows=len(a["gold"]),
        settings=9,
    )


def verify_matched(u):
    d = c.read(c.ROOT / u["source"])["results"][u["endpoint"]]
    ep = d["endpoint"]
    phase = u["endpoint"].split("/")[0]
    dim = u["dimension"]
    labels = ep["all_labels"]
    ix = [labels.index(x) for x in ep["macro_labels"]]
    gold = [r for r in c.lines(c.ROOT / u["gold_path"]) if r["phase"] == phase]
    field = {"hate_type": "hate_types", "target_group": "target_groups", "legal": "legal"}[dim]
    eligible = [r for r in gold if known(r.get(field)) and (dim == "legal" or r["gate"] is True)]
    ids = [r["key"] for r in eligible]
    c.require(ids == ep["scorable_ids"], "Matched gold mask differs")
    g = np.asarray([bits(r[field], labels) for r in eligible], bool)
    index = {k: i for i, k in enumerate(ids)}
    prepared = {}
    seen = Counter()
    for r in c.lines(c.ROOT / u["prediction_path"]):
        if r["phase"] != phase or r["key"] not in index:
            continue
        key = (r["condition"], r["repeat"])
        if key not in prepared:
            prepared[key] = (np.zeros_like(g), np.zeros(len(g), bool))
        seen[key, r["key"]] += 1
        c.require(seen[key, r["key"]] == 1, "Duplicate matched prediction")
        p, v = prepared[key]
        i = index[r["key"]]
        ans = r.get("result")
        if ans is not None and ans.get(dim) is not None:
            v[i] = True
            p[i] = bits(ans[dim], labels)
    equal(g.sum(0), [ep["gold_support"][x] for x in labels], "matched gold support")
    for name, old in d["per_condition_repeat"].items():
        cond, rep = name.rsplit("/repeat", 1)
        p, v = prepared[cond, int(rep)]
        z = counts(g, p, v)
        equal(z, old["per_label_counts"], "matched raw counts")
        equal(metric.macro(z, ix), old["macro_f1_valid_only"], "matched own macro")
    for old in d["contrasts"].values():
        selected = [tuple(k) for k in old["left_settings"] + old["right_settings"]]
        mask = np.logical_and.reduce([prepared[k][1] for k in selected])
        c.require(ids_on(ids, mask) == old["common_valid_ids"], "Matched paired mask differs")
        vals = [metric.macro(counts(g, prepared[k][0], mask), ix) for k in selected]
        if ix:
            equal(vals, old["common_f1_by_setting"], "matched paired macro")
            n = len(old["left_settings"])
            equal(
                float(np.mean(vals[:n]) - np.mean(vals[n:])),
                old["mean_difference"],
                "matched contrast point",
            )
    return dict(
        depth="saved answer JSONL and gold JSONL with paired group masks",
        rows=len(g),
        settings=len(prepared),
    )


def replay_count_tree(value, labels, macro_labels, trail=""):
    """Check sufficient counts; never recompute F1 from averaged counts."""
    n = 0
    if isinstance(value, dict):
        scope = value.get("macro_labels", macro_labels)
        z = None
        local_labels = labels
        for key in ("per_label_counts", "label_counts", "counts_tp_fp_fn"):
            candidate = value.get(key)
            if isinstance(candidate, list) and np.asarray(candidate).shape == (len(labels), 3):
                z = np.asarray(candidate)
        per = value.get("per_label")
        if (
            isinstance(per, dict)
            and set(labels).issubset(per)
            and all(
                (
                    isinstance(per[x], dict) and all((k in per[x] for k in ("tp", "fp", "fn")))
                    for x in labels
                )
            )
        ):
            z = np.asarray([[per[x][k] for k in ("tp", "fp", "fn")] for x in labels])
        old = value.get("macro_f1", value.get("macro_f1_valid_only", "absent"))
        if z is not None and old != "absent" and np.equal(z, np.floor(z)).all():
            if scope and set(scope).issubset(labels):
                equal(metric.macro(z, [labels.index(x) for x in scope]), old, "saved count macro")
                n += 1
            elif old is None:
                equal(None, old, "unsupported macro")
                n += 1
        for key, child in value.items():
            if key not in {"per_label", "per_label_counts", "label_counts", "counts_tp_fp_fn"}:
                n += replay_count_tree(child, labels, scope, trail + "/" + key)
    elif isinstance(value, list):
        for child in value:
            n += replay_count_tree(child, labels, macro_labels, trail)
    return n


def main():
    import sys

    registry = c.read(c.BASE / "comparison_registry_v2.json")
    mode = sys.argv[1] if len(sys.argv) > 1 else "raw"
    chosen = {
        "legacy_assembled_parquet": verify_legacy,
        "broad_encoded": verify_broad,
        "gpu_arrays": verify_gpu,
        "gpu_control_arrays": verify_gpu,
        "combined_aligned_arrays": verify_combined,
        "matched_jsonl": verify_matched,
        "german_legal_arrays": verify_german,
    }
    for u in registry["units"]:
        fn = chosen.get(u["reader"])
        if u["study"] == "five_fit_update":
            fn = verify_five
        if mode == "raw" and fn is None:
            continue
        if mode == "counts" and (
            fn is not None
            or u["reader"] not in {"saved_setting_counts", "earlier_supporting_counts"}
            or u["study"] in {"relevance", "reserved", "human"}
        ):
            continue
        before = sum(CHECKS.values())
        try:
            if mode == "raw":
                details = fn(u)
            else:
                data = c.read(c.ROOT / u["source"])
                for key in u["source_selector"]:
                    data = data[key]
                n = replay_count_tree(data, u["labels"], u["macro_labels"])
                c.require(n > 0, "No original macro counts found")
                details = dict(
                    depth="saved sufficient counts, not a new raw-answer audit", replayed_scores=n
                )
            RESULTS.append(
                dict(id=u["id"], status="passed", checks=sum(CHECKS.values()) - before, **details)
            )
            print(u["id"], "passed", sum(CHECKS.values()) - before, flush=True)
        except Exception as e:
            RESULTS.append(
                dict(id=u["id"], status="failed", error=str(e), traceback=traceback.format_exc())
            )
            print(u["id"], "FAILED", str(e), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    name = "reader_" + mode + "_" + stamp + ".json"
    c.save(
        name,
        dict(
            created_utc=stamp,
            mode=mode,
            checks=dict(CHECKS),
            max_absolute_error=MAX_ERROR,
            results=RESULTS,
            sources=c.BINDINGS,
            real_micro_calculations=0,
        ),
    )
    print("REPORT", name, "FAILURES", sum((x["status"] == "failed" for x in RESULTS)), flush=True)
    if any((x["status"] == "failed" for x in RESULTS)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
