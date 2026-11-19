from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path
import hashlib
import math
import numpy as np
from analysis.feedback.transfer import classifiers as p

a = p.a
OUT = p.OUT
COLS = (
    "linear_baseline",
    "linear_updated",
    "qwen_reference",
    "qwen_updated",
    "linear_change",
    "qwen_change",
    "difference_in_change",
    "updated_absolute_difference",
    *[f"pool_{i}_linear_change" for i in (1, 2, 3)],
    *[f"pool_{i}_qwen_change" for i in (1, 2, 3)],
    *[f"pool_{i}_difference_in_change" for i in (1, 2, 3)],
    *[f"pool_{i}_updated_absolute_difference" for i in (1, 2, 3)],
)
SETTINGS = (
    ("all_permitted", 0.6),
    ("all_permitted", 0.5),
    ("matched_gradient", 0.6),
    ("matched_gradient", 0.5),
)
VIEWS = (
    "main",
    "strict",
    "replay_excluded",
    "strict_replay_excluded",
    "representative",
    "representative_replay_excluded",
)


def near(x, y, label, tol=2e-13):
    if isinstance(y, dict):
        a.need(isinstance(x, dict) and set(x) == set(y), "Key mismatch " + label)
        for key, v in y.items():
            near(x[key], v, label + "/" + str(key), tol)
    elif isinstance(y, (list, tuple, np.ndarray)):
        a.need(len(x) == len(y), "Length mismatch " + label)
        for j, v in enumerate(y):
            near(x[j], v, label + "/" + str(j), tol)
    elif y is None:
        a.need(x is None, "Expected unknown " + label)
    elif isinstance(y, (float, int, np.integer, np.floating, F)) and not isinstance(
        y, bool
    ):
        a.need(
            x is not None
            and math.isfinite(float(x))
            and abs(float(x) - float(y)) <= tol,
            "Numeric mismatch " + label,
        )
    else:
        a.need(x == y, "Value mismatch " + label)


def ratio(n, d):
    return F(int(n), int(d)) if d else F(0)


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A18-transfer-part8-support-v1|" + name).encode()).digest()[:8],
        "little",
    )


def sname(s):
    return SETTINGS[s][0] + "/cutoff_" + str(SETTINGS[s][1]).replace(".", "")


def percentile(values, q):
    ordered = np.sort(values, axis=0)
    at = (len(ordered) - 1) * q
    low = math.floor(at)
    high = math.ceil(at)
    return ordered[low] + (at - low) * (ordered[high] - ordered[low])


def from_counts(counts, n, arms=(1, 2, 3)):
    if n == 0:
        return [{name: None for name in COLS} for _ in range(4)]
    out = []
    q = counts[16:]
    for s in range(4):
        l = counts[s * 4 : s * 4 + 4]
        lb = ratio(l[0], n)
        lu = ratio(sum(l[j] for j in arms), len(arms) * n)
        qr = ratio(q[0], 3 * n)
        qu = ratio(sum(q[j] for j in arms), 3 * len(arms) * n)
        lv = [ratio(l[j] - l[0], n) for j in (1, 2, 3)]
        qv = [ratio(q[j] - q[0], 3 * n) for j in (1, 2, 3)]
        vals = [
            lb,
            lu,
            qr,
            qu,
            lu - lb,
            qu - qr,
            (lu - lb) - (qu - qr),
            lu - qu,
            *lv,
            *qv,
            *[lv[j] - qv[j] for j in range(3)],
            *[ratio(l[j], n) - ratio(q[j], 3 * n) for j in (1, 2, 3)],
        ]
        out.append(dict(zip(COLS, vals)))
    return out


def select(meta, err, votes, endpoint, view, excluded, subset=None, arms=(1, 2, 3)):
    pop, kind = endpoint.rsplit("_", 1)
    ids = [
        i
        for i, m in enumerate(meta)
        if m["population"] == pop
        and m["gold_hate"] == (kind == "FNR")
        and ("excluded" not in view or m["key"] not in excluded)
    ]
    if view.startswith("representative"):
        groups = defaultdict(list)
        for i in ids:
            groups[meta[i]["inference_cluster"]].append(i)
        keep = {
            min(
                group,
                key=lambda i: hashlib.sha256(
                    ("A16-representative|" + meta[i]["key"]).encode()
                ).hexdigest(),
            )
            for group in groups.values()
        }
        ids = [i for i in ids if i in keep]
    if subset is not None:
        permitted = set(subset)
        ids = [i for i in ids if i in permitted]
    common = [
        i
        for i in ids
        if all(
            err[i, c, r] >= 0 and (not view.startswith("strict") or votes[i, c, r] == 3)
            for c in (0, *arms)
            for r in range(3)
        )
    ]
    return ids, common


def independent_bounds(lerr, qerr, votes, ids, strict=False, logical=None):
    if not ids:
        return None
    n = len(ids)
    q = qerr[ids]
    available = q >= 0
    if strict:
        available &= votes[ids] == 3
    if logical is None:
        low = np.where(available, q, 0).sum(axis=(0, 2), dtype=np.int64)
        high = low + (~available).sum(axis=(0, 2), dtype=np.int64)
    else:
        low = logical[0][ids].sum(axis=(0, 2), dtype=np.int64)
        high = logical[1][ids].sum(axis=(0, 2), dtype=np.int64)
    rb = [ratio(low[0], 3 * n), ratio(high[0], 3 * n)]
    ub = [ratio(sum(low[1:]), 9 * n), ratio(sum(high[1:]), 9 * n)]
    db = [ub[0] - rb[1], ub[1] - rb[0]]
    out = []
    for s in range(4):
        l = lerr[ids, s // 2, s % 2].sum(axis=0, dtype=np.int64)
        lb = ratio(l[0], n)
        lu = ratio(sum(l[1:]), 3 * n)
        delta = lu - lb
        out.append(
            dict(
                linear_baseline=lb,
                linear_updated=lu,
                linear_change=delta,
                qwen_reference=rb,
                qwen_updated=ub,
                qwen_change=db,
                difference_in_change=[delta - db[1], delta - db[0]],
                updated_absolute_difference=[lu - ub[1], lu - ub[0]],
            )
        )
    return out


def expected_draw_values(totals, n):

    l = totals[:, :16].reshape(-1, 4, 4)
    q = totals[:, 16:]
    b = l[:, :, 0] * 9
    u = l[:, :, 1:].sum(axis=2) * 3
    rb = np.repeat((q[:, 0] * 3)[:, None], 4, axis=1)
    ru = np.repeat(q[:, 1:].sum(axis=1)[:, None], 4, axis=1)
    values = [b, u, rb, ru, u - b, ru - rb, (u - b) - (ru - rb), u - ru]
    lp = [9 * (l[:, :, j] - l[:, :, 0]) for j in (1, 2, 3)]
    qp = [np.repeat((3 * (q[:, j] - q[:, 0]))[:, None], 4, axis=1) for j in (1, 2, 3)]
    values.extend(lp)
    values.extend(qp)
    values.extend(lp[j] - qp[j] for j in range(3))
    values.extend(
        9 * l[:, :, j] - np.repeat((3 * q[:, j])[:, None], 4, axis=1) for j in (1, 2, 3)
    )
    return np.stack(values, axis=2) / (9 * n[:, None, None])


def verify_bootstrap(family, meta, raw, eligible, common, record):
    boot = record["bootstrap"]
    primary = family.startswith("cohort/main/")
    rng_seed = (
        [2026101224, 2026101225, 2026101226, 2026101227][
            p.ENDPOINTS.index(record["endpoint"])
        ]
        if primary
        else seed(family)
    )
    a.need(
        boot["seed"] == rng_seed
        and boot["draws"] == 10000
        and boot["generator"] == "PCG64",
        "Original seed and draw count",
    )
    groups = defaultdict(list)
    mode = record["resampling_mode"]
    for i in common:
        groups[
            meta[i]["key"] if mode == "item" else meta[i]["inference_cluster"]
        ].append(i)
    labels = sorted(groups)
    sizes = np.asarray([len(groups[label]) for label in labels], dtype=np.int64)
    totals = np.asarray(
        [raw[groups[label]].sum(axis=0) for label in labels], dtype=np.int64
    )
    a.need(len(labels) == boot["units"], "Bootstrap resampling unit count")
    streams = []
    info = []
    if mode == "source_stratified":
        membership = defaultdict(set)
        for i in eligible:
            membership[meta[i]["inference_cluster"]].add(meta[i]["source"])
        strata = defaultdict(list)
        for j, label in enumerate(labels):
            strata[tuple(sorted(membership[label]))].append(j)
        for signature, ids in sorted(strata.items()):
            sub = seed("stratum|" + str(rng_seed) + "|" + "|".join(signature))
            streams.append((np.asarray(ids), np.random.Generator(np.random.PCG64(sub))))
            info.append(
                dict(source_signature=list(signature), units=len(ids), seed=sub)
            )
    a.need(info == boot["strata"], "Whole-cluster source stratification")
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    saved = np.load(a.ROOT / record["draw_file"], mmap_mode="r", allow_pickle=False)
    a.need(saved.shape == (10000, 4, 20), "Draw axes and scalar coverage")
    max_error = 0.0

    for first in range(0, 10000, 17):
        length = min(17, 10000 - first)
        selected = (
            np.concatenate(
                [
                    ids[g.integers(0, len(ids), (length, len(ids)), dtype=np.int64)]
                    for ids, g in streams
                ],
                axis=1,
            )
            if streams
            else rng.integers(0, len(labels), (length, len(labels)), dtype=np.int64)
        )
        weights = np.stack(
            [np.bincount(row, minlength=len(labels)) for row in selected]
        )
        expected = expected_draw_values(weights @ totals, weights @ sizes)
        error = float(np.abs(saved[first : first + length] - expected).max())
        max_error = max(max_error, error)
        a.need(error <= 2e-13, "Independent bootstrap mismatch " + family)
    return saved, max_error


def main():
    a.need(not (OUT / "verified.json").exists(), "Preserve independent verification")
    assembled = a.read(OUT / "assembled.json")
    calc = a.read(OUT / "calculated.json")
    contract = a.read(OUT / "analysis_contract.json")
    for name, digest in {
        **assembled["input_bindings"],
        **assembled["analysis_sources"],
        **calc["artifacts"],
    }.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed evidence " + name)
    meta = list(a.rows(p.P3 / "metadata.jsonl"))
    roster = list(a.rows(a.OUT / "roster.jsonl"))
    linear_roster = list(a.rows(p.LINEAR / "roster.jsonl"))
    index = {r["key"]: j for j, r in enumerate(linear_roster)}
    done = a.read(p.LINEAR / "static_complete.json")
    proof = a.read(p.REVIEW)
    indices = [index[r["key"]] for r in roster]
    N = len(meta)
    prob = np.empty((N, 2, 4))
    pred = np.empty((N, 2, 2, 4), dtype=np.int8)
    consumed = []
    for i, (m, r, j) in enumerate(zip(meta, roster, indices)):
        base = linear_roster[j]
        a.need(
            (
                m["key"],
                r["text"],
                m["gold_hate"],
                m["inference_cluster"],
                m["population"],
            )
            == (
                base["key"],
                base["text"],
                base["gate"],
                base["inference_cluster"],
                base["population"],
            ),
            "Independent exact input join",
        )
    source_catalog = a.read(p.P5 / "source_catalog.json")
    fit_bindings = a.read(OUT / "fit_bindings.json")
    for vi, variant in enumerate(("all_permitted", "matched_gradient")):
        for c, condition in enumerate(p.CONDITIONS):
            job = done["jobs"][variant + "/" + condition]
            path = a.ROOT / job["receipt"]
            record = a.read(path)
            a.need(
                a.sha(path) == job["sha256"] == proof["bindings"][a.rel(path)],
                "Original verified model receipt",
            )
            ids = (
                sorted(
                    ident
                    for ident, q in source_catalog.items()
                    if condition in q["pool_memberships"]
                )
                if c
                else []
            )
            a.need(
                ids
                == record["identity"]["source_ids"]
                == fit_bindings["jobs"][variant + "/" + condition]["source_ids"],
                "Independent source-pool identity",
            )
            if c:
                a.need(
                    len(ids) == 40
                    and sum(
                        source_catalog[ident]["record"]["meta"]["gate"] for ident in ids
                    )
                    == 20,
                    "Same twenty/twenty feedback labels",
                )
            attempt = path.parent / record["attempt"]
            original = np.load(attempt / "probabilities.npy", allow_pickle=False)
            prob[:, vi, c] = original[indices, 0]
            for ti, cutoff in enumerate((0.6, 0.5)):
                bits = np.load(
                    attempt
                    / ("decisions_06.npy" if cutoff == 0.6 else "decisions_05.npy"),
                    allow_pickle=False,
                )
                a.need(
                    np.array_equal(bits[:, 0], original[:, 0] > cutoff),
                    "All saved strict threshold bits",
                )
                pred[:, vi, ti, c] = original[indices, 0] > cutoff
            consumed.append(a.rel(path))
    gold = np.asarray([m["gold_hate"] for m in meta], dtype=np.int8)
    qpred = np.full((N, 4, 3), -1, dtype=np.int8)
    votes = np.zeros_like(qpred)
    for i, row in enumerate(a.rows(a.OUT / "matrix.jsonl")):
        a.need(
            row["key"] == meta[i]["key"] and row["gold_hate"] == bool(gold[i]),
            "Original Qwen join",
        )
        for c in range(4):
            for t in range(3):
                q = row["predictions"][c][t]
                votes[i, c, t] = q["n_valid"]
                qpred[i, c, t] = -1 if q["result"] is None else int(q["result"]["hate"])
    lerr = (pred != gold[:, None, None, None]).astype(np.int8)
    qerr = np.where(qpred < 0, -1, qpred != gold[:, None, None]).astype(np.int8)
    raw = np.zeros((N, 20), dtype=np.int64)
    for s in range(4):
        raw[:, 4 * s : 4 * s + 4] = lerr[:, s // 2, s % 2]
    for c in range(4):
        raw[:, 16 + c] = np.where(qerr[:, c] < 0, 0, qerr[:, c]).sum(axis=1)
    with np.load(OUT / "arrays.npz", allow_pickle=False) as data:
        for name, expected in [
            ("probabilities", prob),
            ("linear_decisions", pred),
            ("linear_errors", lerr),
            ("qwen_decisions", qpred),
            ("qwen_errors", qerr),
            ("valid_votes", votes),
            ("raw_error_counts", raw),
            ("linear_roster_indices", indices),
        ]:
            a.need(
                np.array_equal(data[name], expected),
                "Independent array reconstruction " + name,
            )
    with np.load(p.P7 / "vote_completion_arrays.npz", allow_pickle=False) as data:
        logical = (data["lower_error"], data["upper_error"])
    eras = [
        row["actual_execution_era"]
        for row in p.gzrows(p.P7 / "input_execution.jsonl.gz")
    ]
    families = a.read(OUT / "families.json")
    rates = a.read(OUT / "rates.json")
    excluded = set(contract["replay_exclusions"])
    all_ids = {}
    max_error = 0.0
    scalar_count = 0
    drawsets = 0
    for family, record in families.items():
        parts = family.split("/")
        subset = None
        if parts[0] == "source":
            subset = [i for i, m in enumerate(meta) if m["source"] == parts[1]]
        elif parts[0] == "era":
            subset = [i for i, e in enumerate(eras) if e == parts[1]]
        elif parts[0] == "function":
            subset = [
                i for i, m in enumerate(meta) if m.get("functionality") == parts[1]
            ]
        elif parts[0] == "template":
            subset = [
                i
                for i, m in enumerate(meta)
                if m["source"] == "hatecheck" and str(m["template_id"]) == parts[1]
            ]
        else:
            a.need(parts[0] in ("cohort", "resampling"), "Unexpected family type")
        eligible, common = select(
            meta, qerr, votes, record["endpoint"], record["view"], excluded, subset
        )
        all_ids[family] = (eligible, common)
        a.need(
            record["eligible_indices"] == eligible
            and record["common_indices"] == common
            and record["eligible_n"] == len(eligible)
            and record["common_n"] == len(common),
            "Exact family mask " + family,
        )
        a.need(
            record["source_counts"] == dict(Counter(meta[i]["source"] for i in common))
            and record["clusters"]
            == len({meta[i]["inference_cluster"] for i in common}),
            "Family support " + family,
        )
        near(record["raw_totals"], raw[common].sum(axis=0), "Original raw totals")
        near(
            record["qwen_actual_errors"],
            qerr[common].sum(axis=0),
            "Twelve real Qwen outputs",
        )
        expected = from_counts(raw[common].sum(axis=0), len(common))
        bound = independent_bounds(
            lerr, qerr, votes, eligible, record["view"].startswith("strict")
        )
        logical_bound = independent_bounds(lerr, qerr, votes, eligible, logical=logical)
        draws = None
        if "draw_file" in record:
            draws, error = verify_bootstrap(family, meta, raw, eligible, common, record)
            max_error = max(max_error, error)
            scalar_count += draws.size
            drawsets += 1
        for s in range(4):
            row = rates[sname(s) + "/" + family]
            a.need(
                row["common_n"] == len(common)
                and row["eligible_n"] == len(eligible)
                and row["variant"] == SETTINGS[s][0]
                and row["cutoff"] == SETTINGS[s][1],
                "Rate identity",
            )
            near(row["estimates"], expected[s], family + "/" + str(s))
            near(
                row["full_eligible_missing_bounds"],
                bound[s] if bound else None,
                "Opposing bounds",
            )
            near(
                row["hypothetical_invalid_vote_completion_bounds"],
                logical_bound[s] if logical_bound else None,
                "Logical bounds",
            )
            expected95 = {
                metric: (
                    [
                        percentile(draws[:, s, j], 0.025),
                        percentile(draws[:, s, j], 0.975),
                    ]
                    if draws is not None
                    else None
                )
                for j, metric in enumerate(COLS)
            }
            near(row["intervals95"], expected95, "Independent supporting percentiles")
            primary = family.startswith("cohort/main/") and s == 0
            expected98 = (
                [
                    percentile(draws[:, s, 6], 0.00625),
                    percentile(draws[:, s, 6], 0.99375),
                ]
                if primary
                else None
            )
            near(
                row["primary_interval98_75"],
                expected98,
                "Original primary family percentiles",
            )
        if draws is not None:
            print("Independent scores, masks and all draws pass: " + family, flush=True)
    a.need(
        drawsets == 48 and scalar_count == 38400000, "Full planned resampling coverage"
    )
    a.need(len(rates) == 4 * len(families), "Every setting retained")
    original = a.read(p.P2 / "rates.json")
    for view in VIEWS:
        for endpoint in p.ENDPOINTS:
            q = rates["all_permitted/cutoff_06/cohort/" + view + "/" + endpoint]
            old = original[view + "/" + endpoint]
            a.need(q["common_n"] == old["common_n"], "Same broad transfer mask")
            near(
                q["estimates"]["qwen_reference"],
                old["reference_rate"],
                "Inherited Qwen reference",
            )
            near(
                q["estimates"]["qwen_updated"],
                old["treatment_rate"],
                "Inherited Qwen update",
            )
            near(
                q["estimates"]["qwen_change"], old["difference"], "Inherited Qwen delta"
            )
    omission = a.read(OUT / "omissions.json")
    omission_rows = 0
    for name, rows in omission.items():
        view, endpoint = name.split("/")
        eligible, common = select(meta, qerr, votes, endpoint, view, excluded)
        for row in rows:
            arms = (1, 2, 3)
            if row["kind"].startswith("pool_"):
                removed = p.CONDITIONS.index(row["omitted"])
                arms = tuple(j for j in (1, 2, 3) if j != removed)
                ids = (
                    common
                    if row["kind"] == "pool_fixed_common"
                    else select(meta, qerr, votes, endpoint, view, excluded, arms=arms)[
                        1
                    ]
                )
            else:
                ids = [i for i in common if meta[i]["stage"] != row["omitted"]]
            a.need(
                row["indices"] == ids and row["common_n"] == len(ids),
                "Pool/batch omission selection",
            )
            expected = from_counts(raw[ids].sum(axis=0), len(ids), arms)
            near(
                row["estimates"],
                (
                    [
                        {key: value for key, value in rec.items() if key in COLS[:8]}
                        for rec in expected
                    ]
                    if ids
                    else None
                ),
                "Pool/batch omission arithmetic",
            )
            omission_rows += 4
    influences = defaultdict(list)
    seen = set()
    influence_rows = 0
    influence_context = {}
    for endpoint in p.ENDPOINTS:
        common = all_ids["cohort/main/" + endpoint][1]
        for field in (
            ("inference_cluster", "functionality")
            if endpoint.startswith("en_hatecheck_eval_")
            else ("inference_cluster",)
        ):
            groups = defaultdict(list)
            for i in common:
                groups[meta[i][field]].append(i)
            influence_context[endpoint, field] = (groups, raw[common].sum(axis=0))
    for row in p.gzrows(OUT / "group_omissions.jsonl.gz"):
        endpoint, field = row["endpoint"], row["unit"]
        eligible, common = all_ids["cohort/main/" + endpoint]
        groups, base_counts = influence_context[endpoint, field]
        members = groups[row["omitted"]]
        ident = (endpoint, field, row["omitted"])
        a.need(
            ident not in seen and members == row["indices"],
            "Whole-group omission membership",
        )
        seen.add(ident)
        n = len(common) - len(members)
        counts = base_counts - raw[members].sum(axis=0)
        expected = from_counts(counts, n)[0]["difference_in_change"] if n else None
        a.need(row["remaining_inputs"] == n, "Omitted denominator")
        near(row["difference_in_change"], expected, "Group omission point")
        influences[endpoint + "/" + field].append(row)
        influence_rows += 1
    summaries = a.read(OUT / "group_influence.json")
    for name, rows in influences.items():
        endpoint, field = name.split("/")
        common = all_ids["cohort/main/" + endpoint][1]
        q = summaries[name]
        base = from_counts(raw[common].sum(axis=0), len(common))[0][
            "difference_in_change"
        ]
        finite = [r for r in rows if r["difference_in_change"] is not None]
        a.need(
            len(rows) == len({meta[i][field] for i in common}) == q["groups"],
            "Every group omitted",
        )
        near(q["base_difference_in_change"], base, "Influence base")
        near(
            q["minimum"],
            min(r["difference_in_change"] for r in finite),
            "Influence minimum",
        )
        near(
            q["maximum"],
            max(r["difference_in_change"] for r in finite),
            "Influence maximum",
        )
        a.need(
            q["opposite_sign_omissions"]
            == sum(r["difference_in_change"] * base < 0 for r in finite)
            and q["exact_zero_omissions"]
            == sum(r["difference_in_change"] == 0 for r in finite),
            "Influence exact sign",
        )

    quality = a.read(OUT / "binary_quality.json")
    quality_models = 0
    for name, row in quality.items():
        variant, cut, pop, view = name.split("/")
        vi = ("all_permitted", "matched_gradient").index(variant)
        ti = 0 if cut == "cutoff_06" else 1
        ids = sorted(
            select(meta, qerr, votes, pop + "_FPR", view, excluded)[1]
            + select(meta, qerr, votes, pop + "_FNR", view, excluded)[1]
        )
        a.need(
            ids == row["indices"] and len(ids) == row["common_n"],
            "Quality common population",
        )
        predictions = np.column_stack(
            (pred[ids, vi, ti], qpred[ids].reshape(len(ids), 12))
        )
        counts = np.stack(
            [
                np.bincount(gold[ids] * 2 + predictions[:, j], minlength=4)
                for j in range(16)
            ]
        )
        near(row["actual_model_counts"], counts, "Independent confusion tables")
        quality_models += 16
        metrics = defaultdict(list)
        for tn, fp, fn, tp in counts:
            vals = dict(
                macro_f1=(
                    ratio(2 * tn, 2 * tn + fp + fn) + ratio(2 * tp, 2 * tp + fp + fn)
                )
                / 2,
                balanced_accuracy=(ratio(tn, tn + fp) + ratio(tp, tp + fn)) / 2,
                accuracy=ratio(tn + tp, len(ids)),
                hate_precision=ratio(tp, tp + fp),
                hate_recall=ratio(tp, tp + fn),
                predicted_hate_rate=ratio(fp + tp, len(ids)),
            )
            for metric, value in vals.items():
                metrics[metric].append(value)
        for metric, values in metrics.items():
            b, u, r, t = (
                values[0],
                sum(values[1:4]) / 3,
                sum(values[4:7]) / 3,
                sum(values[7:]) / 9,
            )
            near(
                row["metrics"][metric],
                dict(
                    zip(COLS[:8], [b, u, r, t, u - b, t - r, (u - b) - (t - r), u - t])
                ),
                "Per-real-run metric before averaging",
            )

    trans = a.read(OUT / "transitions.json")
    for name, row in trans.items():
        variant, cut, pop, view = name.split("/")
        vi = ("all_permitted", "matched_gradient").index(variant)
        ti = 0 if cut == "cutoff_06" else 1
        ids = quality[name]["indices"]
        le = lerr[ids, vi, ti]
        qe = qerr[ids]
        ld = le[:, 1:] - le[:, :1]
        qd = qe.sum(axis=2)
        mean_l = np.sign(ld.sum(axis=1)).astype(int)
        mean_q = np.sign(qd[:, 1:].sum(axis=1) - 3 * qd[:, 0]).astype(int)
        mat = np.bincount((mean_l + 1) * 3 + (mean_q + 1), minlength=9).reshape(3, 3)
        near(row["mean_direction_matrix"], mat, "Mean direction cross-tab")
        for j, part in enumerate(row["pools"]):
            delta = ld[:, j]
            rq = qd[:, j + 1] - qd[:, 0]
            mat = np.bincount(
                (np.sign(delta).astype(int) + 1) * 3 + (np.sign(rq).astype(int) + 1),
                minlength=9,
            ).reshape(3, 3)
            near(part["directional_matrix"], mat, "Per-pool direction cross-tab")
            a.need(
                part["linear_fixes"] == int((delta < 0).sum())
                and part["linear_regressions"] == int((delta > 0).sum())
                and part["linear_unchanged_correct"]
                == int(((le[:, 0] == 0) & (le[:, j + 1] == 0)).sum())
                and part["linear_unchanged_error"]
                == int(((le[:, 0] == 1) & (le[:, j + 1] == 1)).sum()),
                "All classifier transitions",
            )
            pairs = np.bincount(
                (2 * le[:, j + 1, None] + qe[:, j + 1]).reshape(-1), minlength=4
            )
            near(part["updated_error_pairs"], pairs, "Dependent actual absolute pairs")
            a.need(part["pair_count"] == 3 * len(ids), "Absolute pair denominator")
        claims = dict(
            linear_consistent_fixes=(
                (le[:, 0] == 1) & np.all(le[:, 1:] == 0, axis=1)
            ).sum(),
            linear_consistent_regressions=(
                (le[:, 0] == 0) & np.all(le[:, 1:] == 1, axis=1)
            ).sum(),
            linear_any_fix=(ld < 0).any(axis=1).sum(),
            linear_any_regression=(ld > 0).any(axis=1).sum(),
            linear_changed_any_pool=(ld != 0).any(axis=1).sum(),
            linear_opposing_pool_directions=(
                (ld < 0).any(axis=1) & (ld > 0).any(axis=1)
            ).sum(),
        )
        for key, value in claims.items():
            a.need(row[key] == int(value), "Exact text transition claim " + key)
    descriptions = a.read(OUT / "probability_changes.json")
    for name, row in descriptions.items():
        variant, condition, label = name.split("/", 2)
        vi = ("all_permitted", "matched_gradient").index(variant)
        c = p.CONDITIONS.index(condition)
        if label == "all":
            ids = list(range(N))
        else:
            field, value = label.split("/", 1)
            field = "functionality" if field == "function" else field
            ids = [i for i, m in enumerate(meta) if m.get(field) == value]
        before = prob[ids, vi, 0]
        after = prob[ids, vi, c]
        delta = after - before
        absolute = np.abs(delta)
        a.need(row["inputs"] == len(ids), "Shift denominator")
        for key, value in dict(
            mean_signed_probability_change=delta.mean(),
            median_absolute_probability_change=percentile(absolute, 0.5),
            p95_absolute_probability_change=percentile(absolute, 0.95),
            maximum_absolute_probability_change=absolute.max(),
        ).items():
            near(row[key], value, "Probability-shift statistic")
        for ti, cutoff in enumerate((0.6, 0.5)):
            bits = before > cutoff
            updated = after > cutoff
            changed = bits != updated
            truth = gold[ids]
            part = row["thresholds"][str(cutoff)]
            expected = dict(
                changed=int(changed.sum()),
                fixes=int((changed & (updated == truth)).sum()),
                regressions=int((changed & (updated != truth)).sum()),
                benign_to_hateful=int(((~bits) & updated).sum()),
                hateful_to_benign=int((bits & (~updated)).sum()),
                changed_input_indices=[ids[j] for j in np.flatnonzero(changed)],
                mean_baseline_distance_to_cutoff_among_changed=(
                    float(np.abs(before[changed] - cutoff).mean())
                    if changed.any()
                    else None
                ),
            )
            near(part, expected, "Exact threshold crossings")
    ledger_count = 0
    for i, row in enumerate(p.gzrows(OUT / "input_comparisons.jsonl.gz")):
        a.need(
            row["input_index"] == i
            and row["key"] == meta[i]["key"]
            and row["gold_hate"] == bool(gold[i]),
            "Full input ledger identity",
        )
        for field, expected in [
            ("linear_probabilities", prob[i]),
            ("linear_decisions", pred[i]),
            ("qwen_decisions", qpred[i]),
            ("valid_votes", votes[i]),
        ]:
            a.need(
                np.array_equal(row[field], expected),
                "Full input ledger original arrays",
            )
        a.need(
            row["main_common"] == bool((qerr[i] >= 0).all())
            and row["strict_common"] == bool((votes[i] == 3).all()),
            "Ledger missingness",
        )
        ledger_count += 1
    a.need(ledger_count == 15758, "Complete input ledger")
    cases = a.read(OUT / "case_selection.json")
    for row in cases:
        pop, cat = row["population"], row["category"]
        candidates = []
        for i, m in enumerate(meta):
            if m["population"] != pop or (qerr[i] < 0).any():
                continue
            le = lerr[i, 0, 0]
            qe = qerr[i]
            ld = int(le[1:].sum() - 3 * le[0])
            qd = int(qe[1:].sum() - 3 * qe[0].sum())
            tests = {
                "linear_consistent_fix": le[0] == 1 and all(le[1:] == 0),
                "linear_consistent_regression": le[0] == 0 and all(le[1:] == 1),
                "linear_improves_qwen_worsens": ld < 0 and qd > 0,
                "linear_worsens_qwen_improves": ld > 0 and qd < 0,
                "linear_unchanged_qwen_improves": ld == 0 and qd < 0,
                "linear_unchanged_qwen_worsens": ld == 0 and qd > 0,
                "updated_linear_correct_qwen_wrong": all(le[1:] == 0)
                and bool((qe[1:] == 1).all()),
                "updated_linear_wrong_qwen_correct": all(le[1:] == 1)
                and bool((qe[1:] == 0).all()),
            }
            if tests[cat]:
                candidates.append(i)
        chosen = (
            min(
                candidates,
                key=lambda i: (
                    hashlib.sha256(
                        (
                            "A18-part8-case-v1|"
                            + pop
                            + "|"
                            + cat
                            + "|"
                            + meta[i]["key"]
                        ).encode()
                    ).hexdigest(),
                    meta[i]["key"],
                ),
            )
            if candidates
            else None
        )
        a.need(
            row["eligible_cases"] == len(candidates)
            and row["input_index"] == chosen
            and row["key"] == (meta[chosen]["key"] if chosen is not None else None),
            "Outcome-category case rule and empty slots",
        )
    a.need(
        calc["bootstrap_scalars"] == scalar_count
        and calc["omission_rows"] == omission_rows
        and calc["influence_rows"] == influence_rows
        and len(cases) == 16,
        "Completion coverage",
    )
    for name, digest in assembled["input_bindings"].items():
        a.need(
            a.sha(a.ROOT / name) == digest, "Original input altered during calculation"
        )
    p.save(
        "verified.json",
        dict(
            status="part8_original_arrays_scores_masks_bounds_bootstraps_diagnostics_independently_verified",
            checked_utc=a.utc(),
            calculated_sha256=a.sha(OUT / "calculated.json"),
            assembled_sha256=a.sha(OUT / "assembled.json"),
            verifier_sha256=a.sha(Path(__file__)),
            consumed_saved_fits=8,
            matched_transfer_inputs=N,
            original_probabilities_selected=int(prob.size),
            original_saved_decision_bits_checked=8 * 16433 * 2,
            qwen_slots=int(qpred.size),
            rate_families=len(families),
            rate_views=len(rates),
            bootstrap_sets=drawsets,
            verified_bootstrap_scalars=scalar_count,
            max_bootstrap_discrepancy=max_error,
            binary_quality_views=len(quality),
            actual_quality_confusion_tables=quality_models,
            transition_views=len(trans),
            probability_shift_views=len(descriptions),
            pool_batch_omission_rows=omission_rows,
            group_omission_rows=influence_rows,
            full_ledger_inputs=ledger_count,
            case_slots=len(cases),
            empty_case_slots=sum(row["input_index"] is None for row in cases),
            original_evidence_unchanged=True,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_primary_families=0,
        ),
    )
    print(
        "All Part 8 independent original-array, point, interval and diagnostic checks pass.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()
