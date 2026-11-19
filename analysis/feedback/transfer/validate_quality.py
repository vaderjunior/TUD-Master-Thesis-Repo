import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from analysis.feedback.transfer import archive as a

OUT = a.ROOT / "experiments/astra/sq3_transfer_part4_20261023_v1"
VIEWS = {
    "main": (False, False, False),
    "strict": (True, False, False),
    "replay_excluded": (False, False, True),
    "strict_replay_excluded": (True, False, True),
    "representative": (False, True, False),
    "representative_replay_excluded": (False, True, True),
}
FIELDS = {
    "hate_type": "gold_hate_types",
    "target_group": "gold_target_groups",
    "severity": "gold_severity",
    "legal": "gold_legal",
}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def native(v):
    if isinstance(v, np.ndarray):
        return native(v.tolist())
    if isinstance(v, np.generic):
        return native(v.item())
    if isinstance(v, dict):
        return {str(k): native(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [native(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def near(got, want, message):
    got = native(got)
    want = native(want)
    if isinstance(want, dict):
        check(set(got) == set(want), message + " keys")
        for key, value in want.items():
            near(got[key], value, message + "/" + key)
    elif isinstance(want, (list, tuple)):
        check(len(got) == len(want), message + " length")
        for i, value in enumerate(want):
            near(got[i], value, message + "/" + str(i))
    elif want is None or isinstance(want, (str, bool)):
        check(got == want, message)
    else:
        check(got is not None and abs(float(got) - float(want)) < 2e-13, message)


def known(row, head):
    return (
        head in ("binary", "known_answer")
        or row[FIELDS[head]] is not None
        and (head == "legal" or row["gold_hate"])
    )


def bitset(value, labels):
    return sum(1 << labels.index(v) for v in value)


def rebuild(contract, schema):
    labels = contract["labels"]
    features = schema["features"]
    data = {
        h: np.zeros((15758, 4, 3, len(cols)), dtype=np.int16)
        for h, cols in features.items()
    }
    meta = []
    codes = {
        h: np.full((15758, 4, 3), -2 if h == "severity" else -1, dtype=np.int16)
        for h in ("gates", "hate_type", "target_group", "severity")
    }
    votes = np.zeros((15758, 4, 3), dtype=np.int8)
    index = {
        h: {name: i for i, name in enumerate(cols)} for h, cols in features.items()
    }
    for i, (m, roster) in enumerate(
        zip(a.rows(a.OUT / "matrix.jsonl"), a.rows(a.OUT / "roster.jsonl"), strict=True)
    ):
        check(
            m["input_index"] == i and m["key"] == roster["key"],
            "Original identity/order",
        )
        r = {
            key: value
            for key, value in m.items()
            if key not in ("predictions", "masks")
        }
        r["functionality"] = (
            json.loads(roster["raw"])["functionality"]
            if roster["source"] == "hatecheck"
            else None
        )
        profile = [
            h for h in ("hate_type", "target_group", "severity", "legal") if known(r, h)
        ]
        r["gold_profile"] = ("hateful_" if r["gold_hate"] else "benign_") + (
            "+".join(profile) if profile else "gate_only"
        )
        meta.append(r)
        for c in range(4):
            for t in range(3):
                slot = m["predictions"][c][t]
                pred = slot["result"]
                votes[i, c, t] = slot["n_valid"]
                if pred is None:
                    continue
                codes["gates"][i, c, t] = pred["hate"]
                for h in ("hate_type", "target_group"):
                    codes[h][i, c, t] = bitset(pred[h], labels[h])
                codes["severity"][i, c, t] = (
                    labels["severity"].index(pred["severity"])
                    if pred["severity"] is not None
                    else -1
                )
                gate_correct = pred["hate"] == r["gold_hate"]
                all_correct = gate_correct
                for h in ("hate_type", "target_group", "severity", "legal"):
                    if known(r, h):
                        all_correct &= (
                            pred[h] == r[FIELDS[h]]
                            if h == "severity"
                            else bitset(pred[h], labels[h])
                            == bitset(r[FIELDS[h]], labels[h])
                        )
                for h in data:
                    if not known(r, h):
                        continue
                    f = data[h][i, c, t]
                    ix = index[h]
                    f[ix["n"]] = 1
                    if h == "binary":
                        f[
                            ix[
                                ["tn", "fp", "fn", "tp"][
                                    2 * int(r["gold_hate"]) + int(pred["hate"])
                                ]
                            ]
                        ] = 1
                        continue
                    if h == "known_answer":
                        f[ix["exact"]] = all_correct
                        f[ix["gate_correct"]] = gate_correct
                        continue
                    if h == "severity":
                        g = 1 << labels[h].index(r[FIELDS[h]])
                        p = 0 if pred[h] is None else 1 << labels[h].index(pred[h])
                    else:
                        g = bitset(r[FIELDS[h]], labels[h])
                        p = codes[h][i, c, t]
                    for j, label in enumerate(labels[h]):
                        b = 1 << j
                        gp = bool(g & b)
                        pp = bool(p & b)
                        f[ix[label + ":tp"]] = gp and pp
                        f[ix[label + ":fp"]] = pp and not gp
                        f[ix[label + ":fn"]] = gp and not pp
                    f[ix["exact"]] = g == p
                    f[ix["gated_exact"]] = g == p and pred["hate"]
                    f[ix["gate_positive"]] = pred["hate"]
                    if h != "severity":
                        extra = int(p & ~g).bit_count()
                        missing = int(g & ~p).bit_count()
                        values = {
                            "contains_gold": not missing,
                            "only_extra": bool(extra) and not missing,
                            "only_missing": bool(missing) and not extra,
                            "both_extra_missing": bool(extra) and bool(missing),
                            "extra_labels": extra,
                            "missing_labels": missing,
                            "predicted_labels": int(p).bit_count(),
                            "gold_labels": g.bit_count(),
                        }
                    else:
                        values = {
                            f'confusion:{r[FIELDS[h]]}:{pred[h] if pred[h] is not None else "none"}': 1
                        }
                        if pred[h] is not None:
                            distance = labels[h].index(pred[h]) - labels[h].index(
                                r[FIELDS[h]]
                            )
                            values.update(
                                provided=1,
                                rank_absolute=abs(distance),
                                rank_signed=distance,
                                over=distance > 0,
                                under=distance < 0,
                            )
                    for key, value in values.items():
                        f[ix[key]] = value
    codes["votes"] = votes
    return meta, data, codes


def groups(name, meta, codes):
    parts = name.split("/")
    kind = parts[0]
    if kind in ("population", "source_stratified", "item"):
        return [i for i, r in enumerate(meta) if r["population"] == parts[1]]
    if kind == "source":
        return [i for i, r in enumerate(meta) if r["source"] == parts[1]]
    if kind == "function":
        return [i for i, r in enumerate(meta) if r["functionality"] == parts[1]]
    if kind == "profile":
        return [
            i
            for i, r in enumerate(meta)
            if r["population"] == parts[1] and r["gold_profile"] == parts[2]
        ]
    check(kind == "all_gate_positive", "Unknown group kind")
    return [
        i
        for i, r in enumerate(meta)
        if r["population"] == parts[1]
        and all(codes["gates"][i, c, t] == 1 for c in range(4) for t in range(3))
    ]


def select(members, meta, codes, head, view, excluded, arms=(1, 2, 3)):
    strict, reps, ex = VIEWS[view]
    eligible = [
        i
        for i in members
        if known(meta[i], head) and (not ex or meta[i]["key"] not in excluded)
    ]
    if reps:
        candidates = {}
        for i in eligible:
            key = meta[i]["gold_hate"], meta[i]["inference_cluster"]
            rank = hashlib.sha256(
                ("A16-representative|" + meta[i]["key"]).encode()
            ).digest()
            if key not in candidates or rank < candidates[key][0]:
                candidates[key] = (rank, i)
        keep = {r[1] for r in candidates.values()}
        eligible = [i for i in eligible if i in keep]
    common = [
        i
        for i in eligible
        if all(
            codes["gates"][i, c, t] >= 0
            and (not strict or codes["votes"][i, c, t] == 3)
            for c in (0, *arms)
            for t in range(3)
        )
    ]
    return eligible, common


def fraction_scores(counts, head, macro_indices, labels, feature_names, metric_names):
    c = {name: int(value) for name, value in zip(feature_names, counts)}
    n = c["n"]
    div = lambda x, y, zero=None: Fraction(x, y) if y else zero
    out = {}
    if head == "binary":
        tp, tn, fp, fn = [c[x] for x in ("tp", "tn", "fp", "fn")]
        out["macro_f1"] = (
            (
                div(2 * tp, 2 * tp + fp + fn, Fraction(0))
                + div(2 * tn, 2 * tn + fp + fn, Fraction(0))
            )
            / 2
            if n
            else None
        )
        out["balanced_accuracy"] = (
            (Fraction(tp, tp + fn) + Fraction(tn, tn + fp)) / 2
            if tp + fn and tn + fp
            else None
        )
        out["accuracy"] = div(tp + tn, n)
    elif head == "known_answer":
        out = dict(
            known_answer_exact=div(c["exact"], n),
            gate_accuracy=div(c["gate_correct"], n),
        )
    else:
        per = [
            div(
                2 * c[l + ":tp"],
                2 * c[l + ":tp"] + c[l + ":fp"] + c[l + ":fn"],
                Fraction(0),
            )
            for l in labels
        ]
        out["macro_f1"] = (
            sum(per[i] for i in macro_indices) / len(macro_indices)
            if n and macro_indices
            else None
        )
        tp = sum(c[l + ":tp"] for l in labels)
        fp = sum(c[l + ":fp"] for l in labels)
        fn = sum(c[l + ":fn"] for l in labels)
        out.update(
            micro_f1=div(2 * tp, 2 * tp + fp + fn, Fraction(0)) if n else None,
            exact_set=div(c["exact"], n),
            gated_exact_set=div(c["gated_exact"], n),
        )
        if head != "severity":
            for name in (
                "contains_gold",
                "only_extra",
                "only_missing",
                "both_extra_missing",
            ):
                out[name] = div(c[name], n)
            for name in ("extra_labels", "missing_labels", "predicted_labels"):
                out["mean_" + name] = div(c[name], n)
        else:
            out["severity_provided"] = div(c["provided"], n)
            for metric, col in [
                ("ordinal_mae_given_provided", "rank_absolute"),
                ("ordinal_bias_given_provided", "rank_signed"),
                ("over_given_provided", "over"),
                ("under_given_provided", "under"),
            ]:
                out[metric] = div(c[col], c["provided"])
    return [out[name] for name in metric_names]


def fractional_summary(counts, head, indices, labels, cols, metrics):
    rows = [
        [
            fraction_scores(counts[c, t], head, indices, labels, cols, metrics)
            for t in range(3)
        ]
        for c in range(4)
    ]
    avg = lambda seq: sum(seq) / len(seq) if all(x is not None for x in seq) else None
    means = [
        [avg([rows[c][t][j] for t in range(3)]) for j in range(len(metrics))]
        for c in range(4)
    ]
    result = {}
    for j, metric in enumerate(metrics):
        ref = means[0][j]
        treat = avg([means[c][j] for c in (1, 2, 3)])
        result[metric] = dict(
            reference=ref,
            treatment=treat,
            difference=treat - ref if ref is not None and treat is not None else None,
            pool_values=[means[c][j] for c in (1, 2, 3)],
            pool_differences=[
                (
                    means[c][j] - ref
                    if means[c][j] is not None and ref is not None
                    else None
                )
                for c in (1, 2, 3)
            ],
        )
    return rows, means, result


def vector_scores(counts, head, indices, labels, cols, metrics):
    get = lambda name: counts[..., cols.index(name)]

    def div(x, y, zero=np.nan):
        x, y = np.broadcast_arrays(x, y)
        return np.divide(x, y, out=np.full(x.shape, zero, dtype=float), where=y != 0)

    n = get("n")
    out = {}
    if head == "binary":
        tp, tn, fp, fn = [get(s) for s in ("tp", "tn", "fp", "fn")]
        out = dict(
            macro_f1=np.where(
                n > 0,
                (div(2 * tp, 2 * tp + fp + fn, 0) + div(2 * tn, 2 * tn + fp + fn, 0))
                / 2,
                np.nan,
            ),
            balanced_accuracy=(div(tp, tp + fn) + div(tn, tn + fp)) / 2,
            accuracy=div(tp + tn, n),
        )
    elif head == "known_answer":
        out = dict(
            known_answer_exact=div(get("exact"), n),
            gate_accuracy=div(get("gate_correct"), n),
        )
    else:
        fs = [
            div(
                2 * get(l + ":tp"),
                2 * get(l + ":tp") + get(l + ":fp") + get(l + ":fn"),
                0,
            )
            for l in labels
        ]
        out["macro_f1"] = (
            np.where(n > 0, sum(fs[i] for i in indices) / len(indices), np.nan)
            if indices
            else np.full(n.shape, np.nan)
        )
        tp = sum(get(l + ":tp") for l in labels)
        fp = sum(get(l + ":fp") for l in labels)
        fn = sum(get(l + ":fn") for l in labels)
        out.update(
            micro_f1=np.where(n > 0, div(2 * tp, 2 * tp + fp + fn, 0), np.nan),
            exact_set=div(get("exact"), n),
            gated_exact_set=div(get("gated_exact"), n),
        )
        if head != "severity":
            for name in (
                "contains_gold",
                "only_extra",
                "only_missing",
                "both_extra_missing",
            ):
                out[name] = div(get(name), n)
            for name in ("extra_labels", "missing_labels", "predicted_labels"):
                out["mean_" + name] = div(get(name), n)
        else:
            out["severity_provided"] = div(get("provided"), n)
            for metric, col in [
                ("ordinal_mae_given_provided", "rank_absolute"),
                ("ordinal_bias_given_provided", "rank_signed"),
                ("over_given_provided", "over"),
                ("under_given_provided", "under"),
            ]:
                out[metric] = div(get(col), get("provided"))
    return np.stack([out[m] for m in metrics], axis=-1)


def percentile(x, q):
    ordered = np.sort(x, axis=0)
    pos = (len(x) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] * (1 - (pos - lo)) + ordered[hi] * (pos - lo)


def replay(r, meta, data, labels, cols, metrics):
    ids = r["common_indices"]
    eligible = r["eligible_indices"]
    mode = r["mode"]
    seed = int.from_bytes(
        hashlib.sha256(("A16-part4-quality-v1|" + r["id"]).encode()).digest()[:8],
        "little",
    )
    unit = lambda i: meta[i]["key"] if mode == "item" else meta[i]["inference_cluster"]
    units = sorted({unit(i) for i in ids})
    position = {u: j for j, u in enumerate(units)}
    text_unit = np.asarray([position[unit(i)] for i in ids])
    check(
        seed == r["bootstrap"]["seed"] and len(units) == r["bootstrap"]["units"],
        "Bootstrap convention",
    )
    strata = defaultdict(list)
    if mode == "source_stratified":
        membership = defaultdict(set)
        for i in eligible:
            membership[meta[i]["inference_cluster"]].add(meta[i]["source"])
        for j, u in enumerate(units):
            strata[tuple(sorted(membership[u]))].append(j)
        strata = dict(sorted(strata.items()))
    near(
        r["bootstrap"]["strata"],
        [dict(sources=list(key), units=len(value)) for key, value in strata.items()],
        "Whole source strata",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    saved = np.load(a.ROOT / r["draw_file"], allow_pickle=False)
    check(saved.shape == (10000, 6, len(metrics)), "Draw dimensions")
    original = data[ids].reshape(len(ids), -1).astype(float)
    max_error = 0.0
    for start in range(0, 10000, 16):
        size = min(16, 10000 - start)
        chosen = (
            np.concatenate(
                [
                    np.asarray(g)[
                        rng.integers(0, len(g), (size, len(g)), dtype=np.int64)
                    ]
                    for g in strata.values()
                ],
                axis=1,
            )
            if strata
            else rng.integers(0, len(units), (size, len(units)), dtype=np.int64)
        )
        weights = np.asarray([np.bincount(p, minlength=len(units)) for p in chosen])[
            :, text_unit
        ].astype(float)
        count = (weights @ original).reshape(size, 4, 3, len(cols))
        scores = vector_scores(
            count, r["head"], r["macro_indices"], labels, cols, metrics
        )
        condition = scores.sum(axis=2) / 3
        treatment = condition[:, 1:].sum(axis=1) / 3
        expected = np.concatenate(
            [condition, treatment[:, None], (treatment - condition[:, 0])[:, None]],
            axis=1,
        )
        actual = saved[start : start + size]
        check(
            np.array_equal(np.isfinite(expected), np.isfinite(actual))
            and np.allclose(expected, actual, rtol=0, atol=2e-13, equal_nan=True),
            "Original-text bootstrap " + r["id"],
        )
        max_error = max(max_error, float(np.nanmax(np.abs(expected - actual))))
    for j, m in enumerate(metrics):
        finite = np.all(np.isfinite(saved[:, :, j]), axis=1)
        check(
            r["metrics"][m]["finite_bootstrap_draws"] == int(finite.sum()),
            "Finite draws",
        )
        if finite.all():
            near(
                r["metrics"][m]["interval95"],
                [percentile(saved[:, 5, j], 0.025), percentile(saved[:, 5, j], 0.975)],
                "Percentile interval",
            )
            pd = saved[:, 1:4, j] - saved[:, :1, j]
            near(
                r["metrics"][m]["pool_difference_intervals95"],
                np.stack([percentile(pd, 0.025), percentile(pd, 0.975)], axis=1),
                "Pool interval",
            )
        else:
            check(r["metrics"][m]["interval95"] is None, "Undefined interval retained")
    return saved.size, max_error


def check_bounds(r, meta, data, codes, labels, cols, metrics):
    ids = r["eligible_indices"]
    strict = VIEWS[r["view"]][0]
    head = r["head"]
    ix = {name: i for i, name in enumerate(cols)}
    n = len(ids)
    available = (codes["gates"][ids] >= 0) & (
        (codes["votes"][ids] == 3) if strict else True
    )
    observed = (data[ids] * available[..., None]).sum(axis=0, dtype=np.int64)
    worst = observed.copy()
    best = observed.copy()
    missing = (~available).sum(axis=0, dtype=np.int64)
    for c in range(4):
        for t in range(3):
            for local, i in enumerate(ids):
                if available[local, c, t]:
                    continue
                w = worst[c, t]
                b = best[c, t]
                w[0] += 1
                b[0] += 1
                row = meta[i]
                if head == "binary":
                    w[ix["fn" if row["gold_hate"] else "fp"]] += 1
                    b[ix["tp" if row["gold_hate"] else "tn"]] += 1
                elif head == "known_answer":
                    b[ix["exact"]] += 1
                    b[ix["gate_correct"]] += 1
                else:
                    gold = (
                        {row[FIELDS[head]]}
                        if head == "severity"
                        else set(row[FIELDS[head]])
                    )
                    for label in labels:
                        w[ix[label + (":fn" if label in gold else ":fp")]] += 1
                        if label in gold:
                            b[ix[label + ":tp"]] += 1
                    b[ix["exact"]] += 1
                    b[ix["gated_exact"]] += 1
                    if head == "severity":
                        b[ix["provided"]] += 1
    low, _, lowm = fractional_summary(
        worst, head, r["macro_indices"], labels, cols, metrics
    )
    high, _, highm = fractional_summary(
        best, head, r["macro_indices"], labels, cols, metrics
    )
    near(r["missing_bounds"]["missing_outputs"], missing, "Missing output counts")
    for metric, value in r["missing_bounds"]["metrics"].items():
        j = metrics.index(metric)
        l = lowm[metric]
        h = highm[metric]
        d = (
            [l["treatment"] - h["reference"], h["treatment"] - l["reference"]]
            if l["treatment"] is not None and h["reference"] is not None
            else [None, None]
        )
        near(
            value,
            dict(
                reference=[l["reference"], h["reference"]],
                treatment=[l["treatment"], h["treatment"]],
                opposing_difference=d,
                run_low=[[low[c][t][j] for t in range(3)] for c in range(4)],
                run_high=[[high[c][t][j] for t in range(3)] for c in range(4)],
            ),
            "Fractional failure bounds",
        )


def main():
    calc = a.read(OUT / "calculated.json")
    diagnostic = a.read(OUT / "diagnostics_complete.json")
    contract = a.read(OUT / "analysis_contract.json")
    schema = a.read(OUT / "feature_schema.json")
    for name, digest in {
        **calc["bindings"],
        **calc["artifacts"],
        **diagnostic["artifacts"],
        **diagnostic["sources"],
    }.items():
        check(a.sha(a.ROOT / name) == digest, "Changed bound file " + name)
    taxonomy = defaultdict(list)
    head = None
    for line in (
        (a.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8").splitlines()
    ):
        match = re.match(r"^  (\w+):", line)
        if match:
            head = match.group(1)
        match = re.match(r"^      (\w+):", line)
        if match:
            taxonomy[head].append(match.group(1))
    for h in ("hate_type", "target_group", "severity", "legal"):
        check(taxonomy[h] == contract["labels"][h], "Original taxonomy label order")
    meta, data, codes = rebuild(contract, schema)
    near(list(a.rows(OUT / "metadata.jsonl")), meta, "Every metadata row")
    with np.load(OUT / "features.npz", allow_pickle=False) as f:
        for h in data:
            check(np.array_equal(data[h], f[h]), "All original-text features " + h)
    with np.load(OUT / "prediction_codes.npz", allow_pickle=False) as f:
        for h in codes:
            check(np.array_equal(codes[h], f[h]), "Original prediction code " + h)
    science = a.read(a.STUDY / "scientific_plan_v1.json")
    excluded = set(science["mandatory_replay_keys"])
    results = a.read(OUT / "quality_results.json")
    heads = ("binary", "hate_type", "target_group", "severity", "known_answer")
    pops = ("en_test", "en_hatecheck_eval")
    expected = {
        f"population/{p}/{h}/{v}"
        for p in pops
        for h in (*heads, "legal")
        for v in VIEWS
    }
    expected |= {
        f"{mode}/{p}/{h}/main"
        for p, mode in [("en_test", "source_stratified"), ("en_hatecheck_eval", "item")]
        for h in heads
        if any(known(row, h) for row in meta if row["population"] == p)
    }
    expected |= {
        f"source/{s}/{h}/{v}"
        for s in ("hatexplain", "implicit_hate", "mhs")
        for h in heads
        for v in ("main", "strict")
    }
    expected |= {
        f"profile/{p}/{profile}/{v}"
        for p in pops
        for profile in {r["gold_profile"] for r in meta if r["population"] == p}
        for v in ("main", "strict")
    }
    expected |= {
        f"function/{fn}/target_group/{v}"
        for fn in {r["functionality"] for r in meta if r["functionality"] is not None}
        for v in ("main", "strict")
    }
    expected |= {
        f"all_gate_positive/{p}/{h}/main"
        for p in pops
        for h in ("hate_type", "target_group", "severity")
    }
    check(
        set(results) == expected and len(results) == 188,
        "Complete quality-view catalogue",
    )
    for pop, heads in contract["population_macro_labels"].items():
        for h, labels in heads.items():
            if h in ("binary", "known_answer"):
                continue
            truth = [
                row[FIELDS[h]]
                for row in meta
                if row["population"] == pop and known(row, h)
            ]
            positive = Counter(
                v for value in truth for v in ([value] if h == "severity" else value)
            )
            expected = [l for l in contract["labels"][h] if positive[l] >= 10]
            check(labels == expected, "Frozen population label mask")
            field = {
                "hate_type": "hate_types",
                "target_group": "target_groups",
                "severity": "severity",
                "legal": "legal",
            }[h]
            check(
                len(truth) == science["fixed_support"][pop][field]["rows"]
                and dict(positive)
                == science["fixed_support"][pop][field]["positive_support"],
                "Frozen fine supports",
            )
    draw_sets = reused = scalar_count = 0
    maximum_error = 0.0
    view_counts = {}
    for name, r in results.items():
        members = groups(name, meta, codes)
        head = r["head"]
        e, c = select(members, meta, codes, head, r["view"], excluded)
        if head != "legal":
            fixed = contract["population_macro_labels"][r["population"]][head]
            check(
                r["population_macro_labels"] == fixed
                and r["macro_indices"]
                == [contract["labels"][head].index(l) for l in fixed],
                "Population-fixed macro label indices",
            )
        if not e:
            check(
                r["eligible_n"] == r["common_n"] == 0
                and not r["metrics"]
                and r["status"] == "unscorable_no_known_gold",
                "Unscorable head",
            )
            continue
        check(
            r["eligible_indices"] == e and r["common_indices"] == c,
            "Gold/availability masks " + name,
        )
        check(
            r["cluster_count"] == len({meta[i]["inference_cluster"] for i in c}),
            "Quality cluster count",
        )
        check(
            r["gold_profile_counts"]
            == dict(Counter(meta[i]["gold_profile"] for i in c)),
            "Gold profiles",
        )
        cols = schema["features"][head]
        metrics = schema["metrics"][head]
        labels = contract["labels"][head]
        count = data[head][c].sum(axis=0, dtype=np.int64)
        near(r["counts"], count, "Quality counts")
        runs, means, points = fractional_summary(
            count, head, r["macro_indices"], labels, cols, metrics
        )
        near(r["run_scores"], runs, "Run scores")
        near(r["condition_means"], means, "Condition means")
        for metric, p in points.items():
            near(
                {key: r["metrics"][metric][key] for key in p},
                p,
                "Exact fractional points " + name + "/" + metric,
            )
        strict = VIEWS[r["view"]][0]
        mask = (codes["gates"][e] >= 0) & ((codes["votes"][e] == 3) if strict else True)
        own_count = (data[head][e] * mask[..., None]).sum(axis=0, dtype=np.int64)
        rr, mm, pp = fractional_summary(
            own_count, head, r["macro_indices"], labels, cols, metrics
        )
        near(
            r["own_valid_descriptive"],
            dict(counts=own_count, run_scores=rr, condition_means=mm, metrics=pp),
            "Own-valid outputs",
        )
        check_bounds(r, meta, data[head], codes, labels, cols, metrics)
        if head in ("hate_type", "target_group", "severity"):
            local_ids = [i for i in members if known(meta[i], head)]
            support = {
                lab: sum(
                    (
                        meta[i][FIELDS[head]] == lab
                        if head == "severity"
                        else lab in meta[i][FIELDS[head]]
                    )
                    for i in local_ids
                )
                for lab in labels
            }
            near(r["group_full_gold_support"], support, "Group support")
            local = [j for j, lab in enumerate(labels) if support[lab] >= 10]
            check(
                r["group_supported_macro_labels"] == [labels[j] for j in local],
                "Descriptive local label mask",
            )
            _, lm, lp = fractional_summary(count, head, local, labels, cols, metrics)
            near(
                r["group_supported_macro_f1_descriptive"],
                [lm[c][0] for c in range(4)]
                + [lp["macro_f1"]["treatment"], lp["macro_f1"]["difference"]],
                "Local supported F1",
            )
        if "bootstrap" in r:
            if "reuse_of" in r:
                old = results[r["reuse_of"]]
                check(
                    r["common_indices"] == old["common_indices"]
                    and r["draw_file"] == old["draw_file"],
                    "Reused identical endpoint",
                )
                reused += 1
                for m in metrics:
                    for field in (
                        "interval95",
                        "pool_difference_intervals95",
                        "finite_bootstrap_draws",
                    ):
                        near(
                            r["metrics"][m][field],
                            old["metrics"][m][field],
                            "Reused interval",
                        )
            else:
                n, error = replay(r, meta, data[head], labels, cols, metrics)
                draw_sets += 1
                scalar_count += n
                maximum_error = max(maximum_error, error)
                if draw_sets % 8 == 0:
                    print(
                        f"Independently replayed {draw_sets} quality draw sets",
                        flush=True,
                    )
    omissions = a.read(OUT / "omissions.json")
    omission_count = 0
    for name, records in omissions.items():
        r = results[name]
        head = r["head"]
        members = groups(name, meta, codes)
        common = r["common_indices"]
        cols = schema["features"][head]
        metrics = schema["metrics"][head]
        labels = contract["labels"][head]
        expected_ids = {("execution_batch", meta[i]["stage"]) for i in common} | {
            (kind, pool)
            for pool in (1, 2, 3)
            for kind in ("pool_fixed_mask", "pool_recomputed_mask")
        }
        check(
            {(x["kind"], x["omitted"]) for x in records} == expected_ids,
            "Complete omission inventory",
        )
        for record in records:
            arms = [
                c
                for c in (1, 2, 3)
                if record["kind"] == "execution_batch" or c != record["omitted"]
            ]
            keep = (
                [i for i in common if meta[i]["stage"] != record["omitted"]]
                if record["kind"] == "execution_batch"
                else common
            )
            if record["kind"] == "pool_recomputed_mask":
                _, keep = select(
                    members, meta, codes, head, r["view"], excluded, tuple(arms)
                )
            check(record["common_n"] == len(keep), "Omission cohort")
            count = data[head][keep].sum(axis=0, dtype=np.int64)
            rows, means, _ = fractional_summary(
                count, head, r["macro_indices"], labels, cols, metrics
            )
            delta = {
                m: (
                    sum(means[c][j] - means[0][j] for c in arms) / len(arms)
                    if all(means[c][j] is not None for c in [0, *arms])
                    else None
                )
                for j, m in enumerate(metrics)
            }
            near(record["metrics"], delta, "Exact omission points")
            omission_count += 1
    prior = a.read(
        a.ROOT / "experiments/astra/sq3_transfer_part2_20261023_v1/rates.json"
    )
    for pop in ("en_test", "en_hatecheck_eval"):
        for view in VIEWS:
            r = results[f"population/{pop}/binary/{view}"]
            counts = np.asarray(r["counts"])
            fp = prior[f"{view}/{pop}_FPR"]
            fn = prior[f"{view}/{pop}_FNR"]
            check(
                set(r["common_indices"])
                == set(fp["common_indices"]) | set(fn["common_indices"]),
                "Part 2 binary common mask",
            )
            for point, column in [(fp, 3), (fn, 4)]:
                check(
                    int(counts[0, :, column].sum()) == point["reference_error_count"]
                    and int(counts[1:, :, column].sum())
                    == point["treatment_error_count"],
                    "Part 2 errors",
                )
            near(
                r["metrics"]["balanced_accuracy"]["difference"],
                -0.5 * (fp["difference"] + fn["difference"]),
                "Balanced-accuracy identity",
            )
        for view in ("main", "strict"):
            joint = results[f"population/{pop}/known_answer/{view}"]
            profiles = [
                r
                for r in results.values()
                if r["group_kind"] == "gold_coverage_profile"
                and r["population"] == pop
                and r["view"] == view
            ]
            near(
                sum(np.asarray(r["counts"]) for r in profiles),
                joint["counts"],
                "Profile decomposition of full known-answer score",
            )

    check(
        draw_sets == calc["new_bootstrap_sets"] == 71
        and reused == calc["reused_bootstrap_sets"] == 2,
        "Draw inventory",
    )
    receipt = dict(
        status="part4_quality_numerics_pass_diagnostic_verification_pending",
        checked_utc=a.utc(),
        quality_views=len(results),
        original_feature_cells=sum(x.size for x in data.values()),
        new_bootstrap_sets=draw_sets,
        new_draws=draw_sets * 10000,
        new_scalar_values=scalar_count,
        reused_bootstrap_sets=reused,
        max_bootstrap_absolute_error=maximum_error,
        omission_points=omission_count,
        all_fractional_point_scores_and_missing_bounds_checked=True,
        all_frozen_gold_and_taxonomy_masks_checked=True,
        binary_rates_match_part2=True,
        all_gold_profile_counts_reconcile=True,
        calculated_sha256=a.sha(OUT / "calculated.json"),
        diagnostics_complete_sha256=a.sha(OUT / "diagnostics_complete.json"),
        verifier_sha256=a.sha(Path(__file__)),
        new_Qwen_calls=0,
        new_fits=0,
    )
    with (OUT / "independent_numerical_check.json").open("xb") as f:
        f.write(a.encoded(receipt, pretty=True))
    print(
        "All quality counts, fractional points, bounds, omissions and new draws verified.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()
