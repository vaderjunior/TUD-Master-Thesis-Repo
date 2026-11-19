from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import numpy as np
from analysis.feedback.rounds import quality_inputs as a
from analysis.feedback.transfer import validate_quality as independent

FIELDS = {
    "hate_type": "gold_hate_types",
    "target_group": "gold_target_groups",
    "severity": "gold_severity",
    "legal": "gold_legal",
}


def known(r, h):
    return (
        h in ["binary", "known_answer"]
        or r[FIELDS[h]] is not None
        and (h == "legal" or r["gold_hate"])
    )


def bits(value, labels):
    return sum(1 << labels.index(v) for v in set(value))


def near(got, want, message):
    independent.near(got, want, message)


def columns(view):
    rounds = (
        range(1, 5)
        if view == "fixed"
        else [int(view[-1])] if view.startswith("round") else [4]
    )
    c = {}
    for r in rounds:
        for p in a.POLICIES:
            c[f"r{r}_{p}"] = [[r - 1, p, 1]]
        for p in ["reference", "static", "random"]:
            c[f"r{r}_adaptive-{p}"] = [[r - 1, "adaptive", 1], [r - 1, p, -1]]
    if view == "fixed":
        for p in a.POLICIES:
            c["r4-r1_" + p] = [[3, p, 1], [0, p, -1]]
        for p in ["reference", "static", "random"]:
            c["r4-r1_adaptive-" + p] = [
                [3, "adaptive", 1],
                [0, "adaptive", -1],
                [3, p, -1],
                [0, p, 1],
            ]
    return c


def average(values):
    return sum(values) / len(values) if all(v is not None for v in values) else None


def condition_ids(p):
    return (
        [0]
        if p == "reference"
        else [1 + 3 * s + ["adaptive", "static", "random"].index(p) for s in range(3)]
    )


def exact_summary(count, head, macro, labels, names, metrics, cols):
    raw = [
        [
            [
                independent.fraction_scores(
                    count[r, c, t], head, macro, labels, names, metrics
                )
                for t in range(3)
            ]
            for c in range(10)
        ]
        for r in range(4)
    ]
    pm = {
        p: [
            [
                average([raw[r][c][t][j] for c in condition_ids(p) for t in range(3)])
                for j in range(len(metrics))
            ]
            for r in range(4)
        ]
        for p in a.POLICIES
    }
    est = []
    for terms in cols.values():
        row = []
        for j in range(len(metrics)):
            parts = [
                None if pm[p][r][j] is None else coef * pm[p][r][j]
                for r, p, coef in terms
            ]
            row.append(sum(parts) if all(x is not None for x in parts) else None)
        est.append(row)
    return raw, est


def aggregate(values, cols):
    pm = {
        p: sum(values[..., c, t, :] for c in condition_ids(p) for t in range(3))
        / (3 * len(condition_ids(p)))
        for p in a.POLICIES
    }
    return np.stack(
        [
            sum(coef * pm[p][..., rnd, :] for rnd, p, coef in terms)
            for terms in cols.values()
        ],
        axis=-2,
    )


def rebuild(contract, schema):
    labels = contract["labels"]
    names = schema["features"]
    shape = (2077, 4, 10, 3)
    data = {
        h: np.zeros((*shape, len(cols)), dtype=np.int16) for h, cols in names.items()
    }
    codes = {
        h: np.full(shape, -2 if h == "severity" else -1, dtype=np.int16)
        for h in ["gates", "hate_type", "target_group", "severity"]
    }
    codes["votes"] = np.zeros(shape, dtype=np.int8)
    meta = []
    ix = {h: {n: j for j, n in enumerate(cols)} for h, cols in names.items()}
    for i, (r, original) in enumerate(
        zip(
            a.rows(a.READER / "matrix.jsonl"),
            a.rows(a.STUDY / "evaluation.jsonl"),
            strict=True,
        )
    ):
        a.need(
            r["key"] == original["key"] and i == r["input_index"],
            "Original row identity",
        )
        m = {key: value for key, value in r.items() if key != "predictions"}
        profile = [h for h in FIELDS if known(r, h)]
        m.update(
            gold_profile=("hateful_" if r["gold_hate"] else "benign_")
            + ("+".join(profile) if profile else "gate_only"),
            any_fine_gold=bool(profile),
            functionality=(
                json.loads(original["raw"])["functionality"]
                if original["source"] == "hatecheck"
                else None
            ),
        )
        meta.append(m)
        for rnd in range(4):
            for ci in range(10):
                for rep, out in enumerate(r["predictions"][rnd][ci]):
                    idx = (i, rnd, ci, rep)
                    pred = out["result"]
                    codes["votes"][idx] = out["n_valid"]
                    if pred is None:
                        continue
                    codes["gates"][idx] = pred["hate"]
                    for h in ["hate_type", "target_group"]:
                        codes[h][idx] = bits(pred[h], labels[h])
                    codes["severity"][idx] = (
                        -1
                        if pred["severity"] is None
                        else labels["severity"].index(pred["severity"])
                    )
                    gate_correct = pred["hate"] == r["gold_hate"]
                    joint = gate_correct
                    for h in FIELDS:
                        if known(r, h):
                            joint &= (
                                pred[h] == r[FIELDS[h]]
                                if h == "severity"
                                else bits(pred[h], labels[h])
                                == bits(r[FIELDS[h]], labels[h])
                            )
                    for h in data:
                        if not known(r, h):
                            continue
                        f = data[h][idx]
                        j = ix[h]
                        f[j["n"]] = 1
                        if h == "binary":
                            f[
                                j[
                                    ["tn", "fp", "fn", "tp"][
                                        2 * int(r["gold_hate"]) + int(pred["hate"])
                                    ]
                                ]
                            ] = 1
                            continue
                        if h == "known_answer":
                            f[j["exact"]] = joint
                            f[j["gate_correct"]] = gate_correct
                            continue
                        g = (
                            (1 << labels[h].index(r[FIELDS[h]]))
                            if h == "severity"
                            else bits(r[FIELDS[h]], labels[h])
                        )
                        p = (
                            (0 if pred[h] is None else 1 << labels[h].index(pred[h]))
                            if h == "severity"
                            else codes[h][idx]
                        )
                        for bit, lab in enumerate(labels[h]):
                            gp = bool(g & (1 << bit))
                            pp = bool(p & (1 << bit))
                            f[j[lab + ":tp"]] = gp and pp
                            f[j[lab + ":fp"]] = pp and not gp
                            f[j[lab + ":fn"]] = gp and not pp
                        f[j["exact"]] = g == p
                        f[j["gated_exact"]] = g == p and pred["hate"]
                        f[j["gate_positive"]] = pred["hate"]
                        if h == "severity":
                            f[
                                j[
                                    f'confusion:{r[FIELDS[h]]}:{pred[h] if pred[h] is not None else "none"}'
                                ]
                            ] = 1
                            if pred[h] is not None:
                                d = labels[h].index(pred[h]) - labels[h].index(
                                    r[FIELDS[h]]
                                )
                                for key, v in dict(
                                    provided=1,
                                    rank_absolute=abs(d),
                                    rank_signed=d,
                                    over=d > 0,
                                    under=d < 0,
                                ).items():
                                    f[j[key]] = v
                        else:
                            extra = int(p & ~g).bit_count()
                            missing = int(g & ~p).bit_count()
                            for key, v in dict(
                                contains_gold=missing == 0,
                                only_extra=extra > 0 and missing == 0,
                                only_missing=extra == 0 and missing > 0,
                                both_extra_missing=extra > 0 and missing > 0,
                                extra_labels=extra,
                                missing_labels=missing,
                                predicted_labels=int(p).bit_count(),
                                gold_labels=int(g).bit_count(),
                            ).items():
                                f[j[key]] = v
        for rnd in range(4):
            a.need(
                bool(np.all(codes["gates"][i, rnd] >= 0))
                == m["masks"]["round_common"][rnd]
                and bool(np.all(codes["votes"][i, rnd] == 3))
                == m["masks"]["round_strict"][rnd],
                "Independent round masks",
            )
        a.need(
            bool(np.all(codes["gates"][i] >= 0)) == m["masks"]["all_round_common"],
            "Independent all-round mask",
        )
    return meta, data, codes


def membership(name, meta, codes):
    part = name.split("/")
    kind = part[0]
    if kind == "population":
        return [i for i, m in enumerate(meta) if m["population"] == part[1]]
    if kind == "source":
        return [i for i, m in enumerate(meta) if m["source"] == part[1]]
    if kind == "function":
        return [i for i, m in enumerate(meta) if m["functionality"] == part[1]]
    if kind == "all_gates_positive":
        return [
            i
            for i, m in enumerate(meta)
            if m["population"] == part[1] and np.all(codes["gates"][i, 3] == 1)
        ]
    a.need(kind == "profile", "Unknown group")
    return [
        i
        for i, m in enumerate(meta)
        if m["population"] == part[1]
        and (
            m["any_fine_gold"]
            if part[2] == "any_fine"
            else m["gold_profile"] == part[2]
        )
    ]


def cohort(meta, members, head, view, codes):
    eligible = [i for i in members if known(meta[i], head)]
    if view == "replay_excluded":
        eligible = [
            i for i in eligible if not meta[i]["masks"]["replay_exclusion_member"]
        ]
    if view == "representative":
        groups = defaultdict(list)
        for i in eligible:
            groups[(meta[i]["gold_hate"], meta[i]["text_cluster"])].append(i)
        selected = {
            min(
                ids,
                key=lambda i: hashlib.sha256(
                    ("A17-part6-representative|" + meta[i]["key"]).encode()
                ).hexdigest(),
            )
            for ids in groups.values()
        }
        eligible = [i for i in eligible if i in selected]
    if view == "fixed":
        valid = np.all(codes["gates"] >= 0, axis=(1, 2, 3))
    elif view == "strict":
        valid = np.all(codes["votes"][:, 3] == 3, axis=(1, 2))
    else:
        valid = np.all(
            codes["gates"][:, int(view[-1]) - 1 if view.startswith("round") else 3]
            >= 0,
            axis=(1, 2),
        )
    return eligible, [i for i in eligible if valid[i]]


def bounds(meta, data, codes, ids, head, macro, labels, names, metrics):
    obs = data[ids, 3].sum(axis=0, dtype=np.int64)
    lo = obs.copy()
    hi = obs.copy()
    j = {v: i for i, v in enumerate(names)}
    missing = np.zeros((10, 3), dtype=int)
    for i in ids:
        for ci, rep in zip(*np.where(codes["gates"][i, 3] < 0)):
            missing[ci, rep] += 1
            lo[ci, rep, j["n"]] += 1
            hi[ci, rep, j["n"]] += 1
            if head == "binary":
                lo[ci, rep, j["fn" if meta[i]["gold_hate"] else "fp"]] += 1
                hi[ci, rep, j["tp" if meta[i]["gold_hate"] else "tn"]] += 1
            elif head == "known_answer":
                hi[ci, rep, j["exact"]] += 1
                hi[ci, rep, j["gate_correct"]] += 1
            else:
                for lab in labels:
                    positive = (
                        meta[i][FIELDS[head]] == lab
                        if head == "severity"
                        else lab in meta[i][FIELDS[head]]
                    )
                    lo[ci, rep, j[lab + (":fn" if positive else ":fp")]] += 1
                    if positive:
                        hi[ci, rep, j[lab + ":tp"]] += 1
                hi[ci, rep, j["exact"]] += 1
                hi[ci, rep, j["gated_exact"]] += 1
                if head == "severity":
                    hi[ci, rep, j["provided"]] += 1
    ls = independent.vector_scores(lo, head, macro, labels, names, metrics)
    hs = independent.vector_scores(hi, head, macro, labels, names, metrics)
    low = {
        p: sum(ls[c, t] for c in condition_ids(p) for t in range(3))
        / (3 * len(condition_ids(p)))
        for p in a.POLICIES
    }
    high = {
        p: sum(hs[c, t] for c in condition_ids(p) for t in range(3))
        / (3 * len(condition_ids(p)))
        for p in a.POLICIES
    }
    allowed = (
        metrics
        if head in ["binary", "known_answer"]
        else ["macro_f1", "micro_f1", "exact_set", "gated_exact_set"]
        + (["severity_provided"] if head == "severity" else [])
    )
    result = {}
    for m in allowed:
        z = metrics.index(m)
        result[m] = {p: [low[p][z], high[p][z]] for p in a.POLICIES}
        for p in ["reference", "static", "random"]:
            result[m]["adaptive-" + p] = [
                low["adaptive"][z] - high[p][z],
                high["adaptive"][z] - low[p][z],
            ]
    return dict(eligible=len(ids), missing_outputs=missing, metrics=result)


def main():
    assembled = a.read(a.OUT / "assembled.json")
    for path, digest in {
        **assembled["input_bindings"],
        **assembled["source_hashes"],
        **assembled["artifacts"],
    }.items():
        a.need(a.sha(a.ROOT / path) == digest, "Bound evidence changed: " + path)
    contract = a.read(a.OUT / "analysis_contract.json")
    schema = a.read(a.OUT / "feature_schema.json")
    result = a.read(a.OUT / "analysis.json")
    meta, data, codes = rebuild(contract, schema)
    a.need(
        meta == list(a.rows(a.OUT / "metadata.jsonl")), "Independent metadata differs"
    )
    for filename, rebuilt in [("features.npz", data), ("prediction_codes.npz", codes)]:
        with np.load(a.OUT / filename, allow_pickle=False) as saved:
            for head, values in rebuilt.items():
                a.need(
                    np.array_equal(values, saved[head]),
                    "Independent feature/code differs: " + head,
                )
    a.need(int((codes["gates"] < 0).sum()) == 241, "Nulls differ")
    for pop, support in contract["full_roster_support"].items():
        for head, s in support.items():
            ids = [
                i
                for i, m in enumerate(meta)
                if m["population"] == pop and known(m, head)
            ]
            counts = {
                lab: sum(
                    (
                        meta[i][FIELDS[head]] == lab
                        if head == "severity"
                        else lab in meta[i][FIELDS[head]]
                    )
                    for i in ids
                )
                for lab in contract["labels"][head]
            }
            a.need(
                s["known_eligible"] == len(ids)
                and s["known_empty"] == sum(meta[i][FIELDS[head]] == [] for i in ids)
                and s["positive_counts"] == counts,
                "Full-roster support differs",
            )
            a.need(
                contract["macro_labels"][pop][head]
                == [lab for lab in contract["labels"][head] if counts[lab] >= 10],
                "Frozen macro labels differ",
            )
    exact = bootstrap_sets = bootstrap_scalars = bounds_checked = undefined = 0
    maximum = 0.0
    groups = a.read(a.OUT / "groups.json")
    for name, r in result["records"].items():
        head = r["head"]
        members = membership(r["group"], meta, codes)
        a.need(members == groups[r["group"]]["members"], "Group membership differs")
        eligible, ids = cohort(meta, members, head, r["view"], codes)
        a.need(
            eligible == r["eligible_indices"] and ids == r["indices"],
            "Independent cohort differs: " + name,
        )
        cols = columns(r["view"])
        a.need(
            cols == r["columns"] and list(cols) == r["column_order"],
            "Contrast definition differs",
        )
        if not ids:
            a.need(r["status"] != "defined", "Empty view not undefined")
            undefined += 1
            continue
        labels = contract["labels"][head]
        names = schema["features"][head]
        metrics = schema["metrics"][head]
        macro = [
            labels.index(v) for v in contract["macro_labels"][r["population"]][head]
        ]
        count = data[head][ids].sum(axis=0, dtype=np.int64)
        a.need(
            np.array_equal(count, r["counts"]), "Integer sufficient statistics differ"
        )
        raw, est = exact_summary(count, head, macro, labels, names, metrics, cols)
        near(r["run_scores"], raw, "Fraction run scores " + name)
        near(r["estimates"], est, "Fraction contrasts " + name)
        exact += sum(v is not None for row in est for v in row)
        own = data[head][eligible]
        if r["view"] == "strict":
            own = own * (codes["votes"][eligible] == 3)[..., None]
        near(
            r["own_valid"],
            aggregate(
                independent.vector_scores(
                    own.sum(axis=0, dtype=np.int64), head, macro, labels, names, metrics
                ),
                cols,
            ),
            "Own-valid scores",
        )
        if head in ["hate_type", "target_group", "severity"]:
            near(
                r["all_taxonomy_macro_descriptive"],
                aggregate(
                    independent.vector_scores(
                        count, head, list(range(len(labels))), labels, names, metrics
                    ),
                    cols,
                )[:, 0],
                "Full taxonomy macro",
            )
        if "missing_bounds" in r:
            expected = bounds(
                meta, data[head], codes, eligible, head, macro, labels, names, metrics
            )
            near(
                {key: r["missing_bounds"][key] for key in expected},
                expected,
                "Independent null bounds",
            )
            bounds_checked += 1
        if "draw_file" not in r:
            continue
        a.need(
            a.sha(a.ROOT / r["draw_file"]) == r["draw_sha256"], "Draw artifact changed"
        )
        clusters = sorted({meta[i]["inference_cluster"] for i in ids})
        a.need(clusters == r["cluster_labels"], "Resampling units changed")
        lookup = {c: i for i, c in enumerate(clusters)}
        inverse = np.asarray([lookup[meta[i]["inference_cluster"]] for i in ids])
        flat = data[head][ids].reshape(len(ids), -1).astype(float)
        random_seed = int.from_bytes(
            hashlib.sha256(("A17-part6-quality-v1|" + name).encode()).digest()[:8],
            "little",
        )
        a.need(random_seed == r["seed"], "Seed differs")
        rng = np.random.Generator(np.random.PCG64(random_seed))
        with np.load(a.ROOT / r["draw_file"], allow_pickle=False) as f:
            saved = f["draws"]
        for start in range(0, 10000, 16):
            size = min(16, 10000 - start)
            chosen = rng.integers(
                0, len(clusters), (size, len(clusters)), dtype=np.int64
            )
            cw = np.zeros((size, len(clusters)), dtype=float)
            for b, row in enumerate(chosen):
                for cluster in row:
                    cw[b, cluster] += 1
            counts = (cw[:, inverse] @ flat).reshape(size, 4, 10, 3, len(names))
            got = aggregate(
                independent.vector_scores(counts, head, macro, labels, names, metrics),
                cols,
            )
            old = saved[start : start + size]
            a.need(
                np.array_equal(np.isfinite(got), np.isfinite(old)),
                "Undefined bootstrap pattern differs",
            )
            delta = np.abs(got - old)
            finite = np.isfinite(delta)
            err = float(delta[finite].max()) if finite.any() else 0
            maximum = max(maximum, err)
            a.need(err < 2e-13, "Independent bootstrap differs")
        finite = np.isfinite(saved).sum(axis=0)
        a.need(np.array_equal(finite, r["finite_draws"]), "Finite draw counts differ")
        for j, row in enumerate(r["intervals"]):
            for z, ci in enumerate(row):
                if finite[j, z] == 10000:
                    near(
                        ci,
                        [
                            independent.percentile(saved[:, j, z], v)
                            for v in [0.025, 0.975]
                        ],
                        "Sorted percentile check",
                    )
                else:
                    a.need(ci == [None, None], "Undefined interval manufactured")
        bootstrap_sets += 1
        bootstrap_scalars += saved.size
        print(f"Independent quality check {bootstrap_sets}/43: {name}", flush=True)
    a.need(
        bootstrap_sets == 43 and len(result["records"]) == 244,
        "Review coverage differs",
    )
    a.write(
        a.OUT / "independent_verification.json",
        dict(
            status="part6_independent_verification_pass",
            created_utc=a.pure.utc(),
            inputs={a.rel(a.OUT / "assembled.json"): a.sha(a.OUT / "assembled.json")},
            source_hashes={
                a.rel(Path(__file__)): a.sha(Path(__file__)),
                a.rel(Path(independent.__file__)): a.sha(Path(independent.__file__)),
            },
            metadata_rows=2077,
            evaluation_predictions=249240,
            evaluation_nulls=241,
            records=244,
            undefined_views=undefined,
            exact_estimates=exact,
            bootstrap_sets=bootstrap_sets,
            bootstrap_scalars=bootstrap_scalars,
            max_bootstrap_difference=maximum,
            bound_views=bounds_checked,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        f"Independent Part 6 analysis PASS: {exact} exact estimates, {bootstrap_scalars} bootstrap scalars; max difference {maximum:.3g}.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()
