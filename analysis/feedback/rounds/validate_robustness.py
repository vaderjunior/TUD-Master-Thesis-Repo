from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import hashlib
import itertools
import numpy as np
from analysis.feedback.rounds import robustness_inputs as a

POL = ["reference", "adaptive", "static", "random"]
CS = {"reference": [0], "adaptive": [1, 4, 7], "static": [2, 5, 8], "random": [3, 6, 9]}


def close(x, y, message, tol=2e-13):
    a.need(
        np.allclose(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), rtol=0, atol=tol
        ),
        message,
    )


def scalar_quantile(draws, q):
    ordered = np.sort(draws, axis=0)
    at = (len(draws) - 1) * q
    i = int(at)
    f = at - i
    return ordered[i] * (1 - f) + ordered[min(i + 1, len(draws) - 1)] * f


def per_policy(values):
    return np.stack(
        [
            values[:, :, CS[p], :].sum(axis=(2, 3)) * (3 if p == "reference" else 1)
            for p in POL
        ],
        axis=2,
    )


def make_values(counts, ids, cols, round_index=3):
    base = counts[ids, round_index]
    output = []
    for col in cols:
        terms = col.split("-")
        value = base[:, POL.index(terms[0])].copy()
        if len(terms) == 2:
            value -= base[:, POL.index(terms[1])]
        output.append(value)
    return np.stack(output, axis=1)


def main():
    scored = a.read(a.OUT / "scored.json")
    assembly = a.read(a.OUT / "assembled.json")
    a.check(scored["input_bindings"])
    a.check(scored["source_hashes"])
    a.check(scored["artifacts"])
    a.check(assembly["input_bindings"])
    a.check(assembly["artifacts"])
    a.check(assembly["source_hashes"])
    data = list(a.rows(a.READER / "matrix.jsonl"))
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    ex = dict(np.load(a.OUT / "execution_arrays.npz", allow_pickle=False))
    N = len(data)
    for i, m in enumerate(data):
        a.need(
            {k: v for k, v in m.items() if k != "predictions"} == meta[i],
            "Metadata differs",
        )
    gates = np.array(
        [
            [
                [
                    [-1 if v["result"] is None else int(v["result"]["hate"]) for v in c]
                    for c in r
                ]
                for r in m["predictions"]
            ]
            for m in data
        ],
        dtype=np.int8,
    )
    valid = np.array(
        [
            [[[v["n_valid"] for v in c] for c in r] for r in m["predictions"]]
            for m in data
        ],
        dtype=np.int8,
    )
    truth = np.array([m["gold_hate"] for m in data])
    errors = np.where(gates < 0, -1, gates != truth[:, None, None, None]).astype(
        np.int8
    )
    counts = per_policy(np.maximum(errors, 0))
    absent = per_policy(errors < 0)
    bykey = {m["key"]: i for i, m in enumerate(data)}
    seen = set()
    totals = Counter()
    groups = defaultdict(list)
    nullkeys = set()
    partialkeys = set()
    reader = a.read(a.READER / "complete.json")
    null_saved = list(a.rows(a.OUT / "null_outputs.jsonl"))
    partial_saved = list(a.rows(a.OUT / "partial_vote_outputs.jsonl"))
    ident = lambda r: (r["phase"], r["key"], r["condition"], r["repeat"])
    null_map = {ident(r): r for r in null_saved}
    partial_map = {ident(r): r for r in partial_saved}
    stages_checked = 0
    for stage in reader["stage_receipts"]:
        for r in a.rows(a.ROOT / stage["predictions"]):
            key = ident(r)
            a.need(key not in seen, "Duplicate compact identity")
            seen.add(key)
            totals["outputs"] += 1
            totals["workers_" + str(r["effective_worker_limit"])] += 1
            for f in ["repairs", "parse_failures", "gate_normalised", "normalisations"]:
                totals[f] += r[f]
            totals["nulls"] += r["n_valid"] == 0
            totals["ties"] += r["uncertain"]
            totals["votes_" + str(r["n_valid"])] += 1
            names = [
                "all",
                "role/" + r["role"],
                "phase/" + r["phase"],
                "stage/" + r["phase"] + "/" + r["stage"],
            ]
            t = datetime.fromisoformat(r["item_completed_utc"]).timestamp()
            if r["role"] == "evaluation":
                i = bykey[r["key"]]
                rr = r["round"] - 1
                c = a.CONDITIONS.index(r["condition"])
                d = r["repeat"]
                pos = (i, rr, c, d)
                for name, value in [
                    ("repairs", r["repairs"]),
                    ("retries", r["accounting"]["retry_attempts"]),
                    ("parse_failures", r["parse_failures"]),
                    ("ties", r["uncertain"]),
                    ("workers", r["effective_worker_limit"]),
                    ("completed_unix", t),
                    ("stage", int(r["stage"].split("_")[-1])),
                ]:
                    a.need(ex[name][pos] == value, "Execution array differs: " + name)
                a.need(
                    ex["seen"][pos] == 1
                    and ex["vote_gates"][pos].tolist()
                    == [-1 if v is None else int(v["hate"]) for v in r["votes"]],
                    "Raw vote mapping differs",
                )
                a.need(
                    r["result"] == data[i]["predictions"][rr][c][d]["result"]
                    and valid[pos] == r["n_valid"],
                    "Matrix-output mapping differs",
                )
                names += [
                    "population/" + r["population"],
                    "round/" + str(r["round"]) + "/" + r["population"],
                    "source/" + meta[i]["source"] + "/gold_" + str(int(truth[i])),
                    "condition/" + r["condition"],
                    "repeat/" + str(d),
                ]
            for name in names:
                groups[name].append(
                    (
                        r["n_valid"],
                        r["repairs"],
                        r["parse_failures"],
                        int(r["uncertain"]),
                        r["gate_normalised"],
                        r["accounting"]["retry_attempts"],
                        t,
                        r["reported_models"],
                        r["finish_reasons"],
                        r["response_classes"],
                    )
                )
            if r["n_valid"] < 3:
                target = null_map if r["n_valid"] == 0 else partial_map
                a.need(
                    key in target and all(r[f] == v for f, v in target[key].items()),
                    "Preserved invalid-output row differs",
                )
                (nullkeys if r["n_valid"] == 0 else partialkeys).add(key)
        stages_checked += 1
    a.need(
        len(seen) == 259224
        and totals["workers_128"] == 259224
        and len(nullkeys) == len(null_map) == 242
        and len(partialkeys) == len(partial_map) == 6342,
        "Execution/invalid inventory count",
    )
    saved_execution = a.read(a.OUT / "execution.json")
    for name, rs in groups.items():
        g = saved_execution["groups"][name]
        a.need(
            g["outputs"] == len(rs)
            and dict(Counter(str(r[0]) for r in rs)) == g["votes"],
            "Failure-group counts differ",
        )
        for j, f in [
            (1, "repairs"),
            (2, "parse_failures"),
            (3, "ties"),
            (4, "gate_normalised"),
            (5, "retries"),
        ]:
            a.need(sum(r[j] for r in rs) == g[f], "Failure-group sum differs")
        a.need(
            min(r[6] for r in rs) == g["first_unix"]
            and max(r[6] for r in rs) == g["last_unix"],
            "Completion time span differs",
        )
        for j, f in [
            (7, "reported_models"),
            (8, "finish_reasons"),
            (9, "response_classes"),
        ]:
            count = Counter()
            for r in rs:
                count.update(r[j])
            a.need(dict(count) == g[f], "Failure-group categories differ")
    del groups
    print(
        "All 259224 compact execution records, vote slots and invalid-output ledgers independently match.",
        flush=True,
    )

    strict = (valid[:, 3] == 3).all(axis=(1, 2))
    final = (gates[:, 3] >= 0).all(axis=(1, 2))
    fixed = (gates >= 0).all(axis=(1, 2, 3))
    repgroups = defaultdict(list)
    for i, m in enumerate(meta):
        repgroups[m["population"], m["gold_hate"], m["text_cluster"]].append(i)
    reps = {
        sorted(
            ids,
            key=lambda i: hashlib.sha256(
                ("A17-part7-representative|" + meta[i]["key"]).encode()
            ).digest(),
        )[0]
        for ids in repgroups.values()
    }
    masks = {
        "final": final,
        "strict": strict,
        "fixed": fixed,
        "strict_fixed": (valid == 3).all(axis=(1, 2, 3)),
        "representative": np.array([i in reps and final[i] for i in range(N)]),
        "replay_excluded": np.array(
            [
                final[i] and not m["masks"]["replay_exclusion_member"]
                for i, m in enumerate(meta)
            ]
        ),
        "no_ties": final & (ex["ties"][:, 3].sum(axis=(1, 2)) == 0),
        "no_repairs": final & (ex["repairs"][:, 3].sum(axis=(1, 2)) == 0),
        "no_transport_retry": final & (ex["retries"][:, 3].sum(axis=(1, 2)) == 0),
        "policy_only": (gates[:, 3, 1:] >= 0).all(axis=(1, 2)),
    }
    for p in ["static", "random", "reference"]:
        masks["paired_" + p] = (gates[:, 3, CS["adaptive"] + CS[p]] >= 0).all(
            axis=(1, 2)
        )
    result = a.read(a.OUT / "analysis.json")
    maxdiff = 0.0
    scalar_count = 0
    points_checked = 0
    drawsets = 0
    for name, r in result["records"].items():
        eligible = np.array(
            [
                m["population"] == r["population"]
                and m["gold_hate"] == (r["endpoint"] == "FNR")
                for m in meta
            ]
        )
        ids = np.flatnonzero(eligible & masks[r["view"]])
        a.need(ids.tolist() == r["indices"], "Sensitivity cohort differs")
        if not len(ids):
            a.need(r["status"] == "undefined_empty", "Empty cohort not recorded")
            continue
        values = make_values(counts, ids, r["columns"])
        labels = [
            str(i) if r["bootstrap_unit"] == "row" else meta[i]["inference_cluster"]
            for i in ids
        ]
        unique = sorted(set(labels))
        lookup = {v: i for i, v in enumerate(unique)}
        member = np.array([lookup[v] for v in labels])
        sizes = np.bincount(member, minlength=len(unique))
        a.need(
            unique == r["cluster_labels"] and sizes.tolist() == r["cluster_sizes"],
            "Bootstrap grouping differs",
        )
        for j, col in enumerate(r["columns"]):
            if r["weighting"] == "text":
                point = Fraction(int(values[:, j].sum()), 9 * len(ids))
            else:
                point = sum(
                    (
                        Fraction(int(values[member == g, j].sum()), 9 * int(sizes[g]))
                        for g in range(len(unique))
                    ),
                    Fraction(),
                ) / len(unique)
            close(float(point), r["points"][col], "Exact point differs")
            points_checked += 1
        saved = np.load(a.ROOT / r["draw_file"], allow_pickle=False)
        saved = (
            saved["draws"] if r["draw_format"] == "npz" else saved[:, r["draw_columns"]]
        )
        a.need(a.sha(a.ROOT / r["draw_file"]) == r["draw_sha256"], "Draw file changed")
        drawsets += 1
        rng = np.random.Generator(np.random.PCG64(r["seed"]))
        rebuilt = np.empty_like(saved)
        for start in range(0, 10000, 32):
            choices = rng.integers(
                0, len(unique), size=(min(32, 10000 - start), len(unique))
            )
            group_weights = np.stack(
                [np.bincount(row, minlength=len(unique)) for row in choices]
            )
            row_weights = group_weights[:, member]
            if r["weighting"] == "text":
                calc = row_weights @ values / (9 * row_weights.sum(axis=1)[:, None])
            else:
                calc = (
                    (row_weights / sizes[member][None, :]) @ values / (9 * len(unique))
                )
            rebuilt[start : start + len(choices)] = calc
        delta = float(np.max(np.abs(rebuilt - saved)))
        maxdiff = max(maxdiff, delta)
        a.need(delta < 2e-13, "Independent bootstrap differs")
        scalar_count += saved.size
        for level, field in [(0.95, "intervals"), (0.9875, "family_level_sensitivity")]:
            if r[field] is None:
                continue
            lo = scalar_quantile(rebuilt, (1 - level) / 2)
            hi = scalar_quantile(rebuilt, 1 - (1 - level) / 2)
            for j, col in enumerate(r["columns"]):
                if col in r[field]:
                    close([lo[j], hi[j]], r[field][col], "Sorted percentile differs")
    print(
        f"Independent points/draws PASS: {points_checked}, {scalar_count} scalars; max difference {maxdiff:.3g}",
        flush=True,
    )

    low_table = np.zeros((2, 27), dtype=np.int8)
    high_table = np.zeros_like(low_table)
    for pattern in itertools.product([-1, 0, 1], repeat=3):
        code = sum((v + 1) * 3**j for j, v in enumerate(pattern))
        free = [j for j, v in enumerate(pattern) if v < 0]
        answers = []
        for bits in itertools.product([0, 1], repeat=len(free)):
            completed = list(pattern)
            for j, b in zip(free, bits):
                completed[j] = b
            answers.append(sum(completed) >= 2)
        for gold in [0, 1]:
            errs = [int(x != gold) for x in answers]
            low_table[gold, code] = min(errs)
            high_table[gold, code] = max(errs)
    v = ex["vote_gates"]
    codes = ((v + 1) * np.array([1, 3, 9])).sum(axis=-1)
    lo = low_table[truth.astype(int)[:, None, None, None], codes]
    hi = high_table[truth.astype(int)[:, None, None, None], codes]
    low_counts = per_policy(lo)
    high_counts = per_policy(hi)
    for item in result["bounds"]:
        ids = np.array(
            [
                i
                for i, m in enumerate(meta)
                if m["population"] == item["population"]
                and m["gold_hate"] == (item["endpoint"] == "FNR")
            ]
        )
        rr = item["round"] - 1
        for kind, l, h in [
            ("aggregate_null", counts, counts + absent),
            ("vote_completion", low_counts, high_counts),
        ]:
            b = item[kind]
            lc = l[ids, rr].sum(axis=0)
            hc = h[ids, rr].sum(axis=0)
            den = 9 * len(ids)
            a.need(
                b["lower_counts"] == lc.tolist()
                and b["upper_counts"] == hc.tolist()
                and b["denominator"] == den,
                "Bound integer counts differ",
            )
            for col, x in b["intervals"].items():
                terms = col.split("-")
                left = POL.index(terms[0])
                right = POL.index(terms[-1])
                expected = (
                    [lc[left] / den, hc[left] / den]
                    if len(terms) == 1
                    else [(lc[left] - hc[right]) / den, (hc[left] - lc[right]) / den]
                )
                close(x, expected, "Bound endpoints differ")
        a.need(
            int((lo[ids, rr] != hi[ids, rr]).sum())
            == item["variable_under_vote_completion"],
            "Variable completion outputs differ",
        )
    omissions = 0
    from analysis.feedback.rounds.subgroups_inputs import annotation

    functions = {
        r["key"]: annotation(r).get("functionality")
        for r in a.rows(a.STUDY / "evaluation.jsonl")
    }
    for item in a.rows(a.OUT / "omissions.jsonl"):
        eligible = np.array(
            [
                m["population"] == item["population"]
                and m["gold_hate"] == (item["endpoint"] == "FNR")
                for m in meta
            ]
        )
        ids = np.flatnonzero(eligible & masks[item["view"]])

        def group(i):
            return (
                f'evaluation_{int(ex["stage"][i,3,0,0]):02}'
                if item["grouping"] == "batch"
                else (
                    functions[meta[i]["key"]]
                    if item["grouping"] == "function"
                    else meta[i][item["grouping"]]
                )
            )

        keep = np.array([i for i in ids if group(i) != item["group"]], dtype=int)
        a.need(
            len(keep) == item["remaining"] and len(ids) - len(keep) == item["removed"],
            "Omission membership differs",
        )
        if len(keep):
            close(
                make_values(counts, keep, item["columns"]).sum(axis=0)
                / (9 * len(keep)),
                item["points"],
                "Omission points differ",
            )
        else:
            a.need(item["points"] is None, "Undefined omission filled")
        omissions += 1
    for item in result["stream_repeat_variation"]:
        ids = np.array(
            [
                i
                for i, m in enumerate(meta)
                if m["population"] == item["population"]
                and m["gold_hate"] == (item["endpoint"] == "FNR")
                and fixed[i]
            ]
        )
        for pol in POL:
            cs = CS[pol]
            ds = [0, 1, 2]
            if item["omitted_kind"] == "stream" and pol != "reference":
                cs = [c for j, c in enumerate(cs) if j != item["omitted_index"]]
            if item["omitted_kind"] == "repeat":
                ds = [d for d in ds if d != item["omitted_index"]]
            if item["omitted_kind"] == "single_cell":
                cs = [0 if pol == "reference" else CS[pol][item["stream"]]]
                ds = [item["repeat"]]
            numerator = sum(
                int(errors[i, 3, c, d]) for i in ids for c in cs for d in ds
            )
            close(
                float(Fraction(numerator, len(ids) * len(cs) * len(ds))),
                item["rates"][pol],
                "Stream/repeat exclusion differs",
            )
    for item in result["filter_composition"]:
        ids = [
            i
            for i, m in enumerate(meta)
            if m["population"] == item["population"]
            and m["gold_hate"] == (item["endpoint"] == "FNR")
            and final[i]
            and bool(masks[item["filter"]][i]) == item["kept"]
        ]
        a.need(
            len(ids) == item["n"]
            and dict(Counter(meta[i]["source"] for i in ids)) == item["source_counts"],
            "Filter composition differs",
        )
        if ids:
            for col, value in item["points"].items():
                close(
                    float(
                        Fraction(
                            int(make_values(counts, ids, [col]).sum()), 9 * len(ids)
                        )
                    ),
                    value,
                    "Filter composition rate differs",
                )
    for item in result["reference_comparisons"]:
        ids = [
            i
            for i, m in enumerate(meta)
            if m["population"] == item["population"]
            and m["gold_hate"] == (item["endpoint"] == "FNR")
            and fixed[i]
        ]
        rr = item["round"] - 1
        ref = float(Fraction(int(counts[ids, rr, 0].sum()), 9 * len(ids)))
        adaptive = float(Fraction(int(counts[ids, rr, 1].sum()), 9 * len(ids)))
        first = float(Fraction(int(counts[ids, 0, 0].sum()), 9 * len(ids)))
        for field, value in [
            ("reference_rate", ref),
            ("adaptive_rate", adaptive),
            ("contemporary_difference", adaptive - ref),
            ("fixed_first_reference_difference", adaptive - first),
            ("reference_drift", ref - first),
        ]:
            close(value, item[field], "Contemporaneous-reference decomposition differs")
    original = a.read(a.PART2 / "errors_v1/results.json")["records"]
    for item in result["primary_unchanged"]:
        for f, v in item.items():
            a.need(
                original[item["name"]][f if f != "clusters" else "inference_clusters"]
                == v,
                "Primary family changed",
            )
    quality = a.read(a.PART6 / "analysis.json")["records"]
    for item in result["quality_views"]:
        r = quality[item["name"]]
        for f, v in item.items():
            if f != "name":
                a.need(r.get(f) == v, "Inherited complete-quality view changed")
    a.write(
        a.OUT / "independent_verification.json",
        dict(
            status="part7_independent_verification_pass",
            created_utc=a.pure.utc(),
            inputs=a.bindings([a.OUT / "assembled.json", a.OUT / "scored.json"]),
            source_sha256=a.sha(Path(__file__)),
            compact_outputs=len(seen),
            evaluation_outputs=int((ex["seen"] == 1).sum()),
            stages=stages_checked,
            nulls=len(null_map),
            partial_vote_outputs=len(partial_map),
            execution_groups=len(saved_execution["groups"]),
            exact_points=points_checked,
            bootstrap_sets=drawsets,
            new_bootstrap_sets=scored["new_draw_sets"],
            bootstrap_scalars=scalar_count,
            max_bootstrap_difference=maxdiff,
            bound_records=len(result["bounds"]),
            omissions=omissions,
            stream_repeat_views=len(result["stream_repeat_variation"]),
            quality_views=len(result["quality_views"]),
            primary_contrasts=4,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print("Part 7 independent reconstruction PASS.", flush=True)


if __name__ == "__main__":
    with a.pure.offline():
        main()
