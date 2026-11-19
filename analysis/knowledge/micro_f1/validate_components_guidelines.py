from analysis.knowledge.micro_f1 import components_guidelines as p
from collections import Counter, defaultdict
from pathlib import Path
from itertools import product
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, multilabel_confusion_matrix

ROOT = p.BASE
RUN = ROOT / "analysis_v2"
C = Counter()


def eq(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=False), (a, b)


def scores(y, x, scope):
    return np.array(
        [
            np.mean([f1_score(y[:, j], x[:, j], zero_division=0) for j in scope]),
            f1_score(y, x, average="micro", zero_division=0),
            f1_score(y[:, scope].ravel(), x[:, scope].ravel(), zero_division=0),
        ]
    )


def weighted(value, w, settings, repeat=None, omit=None):
    out = np.zeros(value.shape[:-2] + (3,))
    for name, weight in w.items():
        ix = [
            i
            for i, (cond, r) in enumerate(settings)
            if cond == name
            and (repeat is None or r == repeat)
            and (omit is None or r != omit)
        ]
        out += weight * value[..., ix, :].mean(-2)
    return out


def fixtures():
    v = np.ones((27, 5), bool)
    v[0, 0] = False
    v[24, 1] = False
    assert (
        v[:24].all(0).sum() == 4
        and v[[21, 22, 23, 24, 25, 26]].all(0).sum() == 4
        and (v.all(0).sum() == 3)
    )
    settings = [(s, r) for s in ["none", "e", "d", "ed"] for r in range(3)]
    z = np.repeat(
        np.array([[0.1, 0.2, 0.2], [0.3, 0.4, 0.4], [0.2, 0.3, 0.3], [0.7, 0.8, 0.8]]),
        3,
        0,
    )
    eq(weighted(z, {"ed": 1, "e": -1, "d": -1, "none": 1}, settings), [0.3, 0.3, 0.3])
    low = {"a": np.array([0.2, 0.3]), "b": np.array([0.4, 0.7])}
    eq(2 * low["a"] - 2 * low["b"][::-1], [-1.0, -0.2])
    y = np.array([[1, 0, 0], [0, 0, 0]], bool)
    x = y.copy()
    eq(scores(y, x, [0, 1, 2]), [1 / 3, 1.0, 1.0])
    assert scores(y, np.zeros_like(x), [0, 1, 2])[1] == 0
    y = np.eye(3, dtype=bool)
    x = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], bool)
    assert scores(y, x, [0, 1, 2])[0] != scores(y, x, [0, 1, 2])[1]
    C["synthetic_contract_fixtures"] = 5


def run():
    fixtures()
    idx = p.c.read(RUN / "analysis_index.json")
    plan = p.c.read(ROOT / "analysis_plan.json")
    questions = []
    allchecks = {}
    cache = p.verified_run(p.GERMAN_READER)
    robust = p.verified_run(p.GERMAN_ROBUST)
    german = {}
    for row in p.c.lines(cache / "predictions.jsonl"):
        if row["phase"] in ["de_legal_test", "de_legal_dev_eval"]:
            german[row["phase"], row["condition"], row["repeat"], row["key"]] = row
    eps = p.c.read(cache / "endpoint_setup.json")
    quality = p.c.read(robust / "quality_subset_scores.json")
    for path in idx["outputs"]:
        r = p.c.read(p.c.ROOT / path)
        u = r["unit"]
        isg = u["study"] == "german_guidelines"
        a = p.h.read_arrays(p.c.ROOT / r["arrays"])
        labels = a["labels"].tolist()
        scope = [labels.index(l) for l in a["macro_labels"]]
        settings = [tuple(x) for x in r["settings"]]
        if isg:
            phase = u["endpoint"].split("/")[0]
            ep = eps[phase]
            assert a["ids"].tolist() == ep["keys"]
            eq(a["gold"], [[l in g for l in labels] for g in ep["gold"]])
            raw = {
                (cond, rep, key): v
                for (pop, cond, rep, key), v in german.items()
                if pop == phase
            }
        else:
            field = {
                "hate_type": "hate_types",
                "target_group": "target_groups",
                "legal": "legal",
            }[u["dimension"]]
            rows = pq.read_table(p.c.bind(p.c.ROOT / u["gold_path"])).to_pylist()
            eligible = [
                x
                for x in rows
                if x[field] is not None
                and (u["dimension"] == "legal" or x["gate"] is True)
            ]
            assert a["ids"].tolist() == [str(x["id"]) for x in eligible]
            eq(a["gold"], [[l in x[field] for l in labels] for x in eligible])
            raw = {}
            for name in u["assembled"]:
                for x in p.c.lines(p.c.ROOT / name):
                    raw[x["condition"], x["replicate"], str(x["id"])] = x
        for j, (cond, rep) in enumerate(settings):
            for i, key in enumerate(a["ids"]):
                x = raw[cond, rep, str(key)]
                ans = x["result"]
                assert bool(a["valid"][j, i]) == (ans is not None)
                assert a["nvalid"][j, i] == x["n_valid"]
                if isg:
                    assert (
                        a["repairs"][j, i] == x["repairs"]
                        and a["ties"][j, i] == x["uncertain"]
                    )
                if ans is not None:
                    eq(a["pred"][j, i], [l in ans[u["dimension"]] for l in labels])
                    assert a["gate"][j, i] == ans["hate"]
                C["raw_answer_view_checks"] += 1
        common = a["valid"].all(0)
        eq(common, a["common"])
        assert a["ids"][common].tolist() == r["common_ids"]
        for mode in ["own", "common"]:
            for j in range(len(settings)):
                mask = a["valid"][j] if mode == "own" else common
                y, x = (a["gold"][mask], a["pred"][j, mask])
                stored = r[mode][j]
                eq(scores(y, x, scope), [stored["scores"][m] for m in p.h.METHODS])
                cm = multilabel_confusion_matrix(y, x)
                eq(
                    np.stack([cm[:, 1, 1], cm[:, 0, 1], cm[:, 1, 0]], -1),
                    stored["tp_fp_fn"],
                )
                eq((y == x).all(1).mean(), stored["set_only_exact"])
                if u["dimension"] != "legal":
                    eq(
                        ((y == x).all(1) & a["gate"][j, mask]).mean(),
                        stored["complete_field_exact"],
                    )
                C["sklearn_setting_population_checks"] += 1
        for j in range(len(settings)):
            missing = np.flatnonzero(~a["valid"][j])
            bits = len(missing) * len(labels)
            assert bits <= 14, "Unexpected exhaustive missing bound size"
            if len(missing):
                vals = []
                for candidate in product([False, True], repeat=bits):
                    x = a["pred"][j].copy()
                    x[missing] = np.array(candidate).reshape(len(missing), len(labels))
                    vals.append(scores(a["gold"], x, scope))
                vals = np.array(vals)
                for k, m in enumerate(p.h.METHODS):
                    eq(
                        [vals[:, k].min(), vals[:, k].max()],
                        r["own"][j]["full_roster_missing_bounds"][m],
                    )
                C["exhaustive_missing_assignment_checks"] += len(vals)
        point = np.array([[v["scores"][m] for m in p.h.METHODS] for v in r["common"]])
        for key, con in r["contrasts"].items():
            w = con["weights"]
            assert abs(sum(w.values())) < 1e-12
            eq(
                weighted(point, w, settings),
                [con["metrics"][m]["difference"] for m in p.h.METHODS],
            )
            if con["question_id"]:
                questions.append(con["question_id"])
            for rep in range(3):
                eq(
                    weighted(point, w, settings, repeat=rep),
                    [con["by_repeat"][str(rep)][m] for m in p.h.METHODS],
                )
                eq(
                    weighted(point, w, settings, omit=rep),
                    [con["leave_one_repeat_out"][str(rep)][m] for m in p.h.METHODS],
                )
                C["repeat_diagnostics"] += 2
            for m in p.h.METHODS:
                acc = np.zeros(2)
                for cond, weight in w.items():
                    ix = [i for i, k in enumerate(settings) if k[0] == cond]
                    v = np.mean(
                        [r["own"][i]["full_roster_missing_bounds"][m] for i in ix], 0
                    )
                    acc += np.array([min(weight * v), max(weight * v)])
                eq(acc, con["full_roster_missing_bounds"][m])
                C["signed_missing_bound_checks"] += 1
        for mode, info in r["bootstrap_files"].items():
            b = p.h.read_arrays(p.c.ROOT / info["path"])
            mask = b["mask"]
            stored = b["scores"]
            eq(
                mask,
                (
                    common & (a["nvalid"] == 3).all(0)
                    if mode == "strict_three_votes"
                    else common
                ),
            )
            members = [np.array([i]) for i in np.flatnonzero(mask)]
            strata = {}
            if isg and mode == "item":
                design = p.c.read(p.c.ROOT / u["resampling_source"])
                indices = np.flatnonzero(mask)
                members = [indices[ix] for ix in design["cluster_rows"]]
                strata = {
                    str(j): s["cluster_indices"] for j, s in enumerate(design["strata"])
                }
            elif mode == "cluster":
                members = [
                    np.flatnonzero(mask & (a["clusters"] == k))
                    for k in sorted(set(a["clusters"][mask]))
                ]
            if not strata:
                strata = defaultdict(list)
                for j, ix in enumerate(members):
                    strata[
                        (
                            str(a["sources"][ix[0]])
                            if mode == "source_stratified"
                            else "all"
                        )
                    ].append(j)
            matrix = []
            for ix in members:
                y = a["gold"][ix].astype(int)
                x = a["pred"][:, ix].astype(int)
                matrix.append(
                    np.stack(
                        [
                            (x * y).sum(1),
                            x.sum(1),
                            np.broadcast_to(y.sum(0), (len(settings), len(labels))),
                        ],
                        -1,
                    )
                )
            matrix = np.array(matrix).reshape(len(members), -1)
            rng = np.random.default_rng(info["seed"])
            maximum = 0.0
            check = []
            for start in range(0, 10000, 64):
                size = min(64, 10000 - start)
                if mode in ["item", "strict_three_votes"] and (
                    not (isg and mode == "item")
                ):
                    picks = rng.integers(0, len(members), (size, len(members)))
                else:
                    picks = np.concatenate(
                        [
                            rng.choice(v, (size, len(v)), replace=True)
                            for _, v in sorted(strata.items())
                        ],
                        1,
                    )
                freq = np.array([np.bincount(x, minlength=len(members)) for x in picks])
                total = (freq @ matrix).reshape(size, len(settings), len(labels), 3)
                tp, pp, aa = np.moveaxis(total, -1, 0)
                den = pp + aa
                lf = np.divide(
                    2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0
                )

                def micro(ix):
                    dd = den[..., ix].sum(-1)
                    return np.divide(
                        2 * tp[..., ix].sum(-1),
                        dd,
                        out=np.zeros_like(dd, dtype=float),
                        where=dd != 0,
                    )

                computed = np.stack(
                    [
                        lf[..., scope].mean(-1),
                        micro(list(range(len(labels)))),
                        micro(scope),
                    ],
                    -1,
                )
                eq(computed, stored[start : start + size])
                maximum = max(
                    maximum, float(np.max(abs(computed - stored[start : start + size])))
                )
                check.append(computed)
                for wanted in [0, 4999, 9999]:
                    if start <= wanted < start + size:
                        ix = np.concatenate([members[i] for i in picks[wanted - start]])
                        for j in range(len(settings)):
                            eq(
                                scores(a["gold"][ix], a["pred"][j, ix], scope),
                                stored[wanted, j],
                            )
                            C["sklearn_bootstrap_setting_checks"] += 1
            check = np.concatenate(check)
            for key, con in r["contrasts"].items():
                dd = weighted(check, con["weights"], settings)
                dest = (
                    con["metrics"]
                    if mode == "item"
                    else con["sensitivities"][mode]["metrics"]
                )
                for j, m in enumerate(p.h.METHODS):
                    eq(np.quantile(dd[:, j], [0.025, 0.975]), dest[m]["ci95"])
                    C["independent_intervals"] += 1
                    if mode == "item" and dest[m]["adjusted_confidence"]:
                        q = (1 - dest[m]["adjusted_confidence"]) / 2
                        eq(
                            np.quantile(dd[:, j], [q, 1 - q]),
                            dest[m]["adjusted_interval"],
                        )
                        C["independent_adjusted_intervals"] += 1
            C["independent_bootstrap_values"] += stored.size
            allchecks[u["id"] + "/" + r["view"] + "/" + mode] = dict(
                values=stored.size, maximum_error=maximum
            )
        if isg:
            for name, v in r["quality"].items():
                mask = common & (
                    (a["nvalid"] == 3).all(0)
                    if name == "three_valid_inner_votes"
                    else (
                        (a["repairs"] == 0).all(0)
                        if name == "no_repairs"
                        else (~a["ties"]).all(0)
                    )
                )
                assert int(mask.sum()) == v["n"] == quality[phase][name]["n"]
                eq(a["gold"][mask].sum(0), v["gold_support"])
                for cond in r["conditions"]:
                    ix = [j for j, k in enumerate(settings) if k[0] == cond]
                    ss = np.array(
                        [scores(a["gold"][mask], a["pred"][j, mask], scope) for j in ix]
                    )
                    C["sklearn_quality_setting_checks"] += len(ix)
                    eq(
                        ss.mean(0),
                        [v["groups"][cond]["scores"][m] for m in p.h.METHODS],
                    )
                    for k, metric in enumerate(["macro_f1", "micro_f1"]):
                        eq(
                            ss[:, k],
                            [
                                q[metric]
                                for q in quality[phase][name]["scores"][cond]["repeats"]
                            ],
                        )
        if not isg:
            earlier = ROOT / u["study"] / u["endpoint"] / r["view"] / "results.json"
            if earlier.exists():
                eqjson = p.c.read(earlier)
                assert (
                    eqjson["groups"] == r["groups"]
                    and eqjson["contrasts"] == r["contrasts"]
                )
                C["first_attempt_results_unchanged"] += 1
        print("Independent Part 6 PASS", u["id"], r["view"], flush=True)
    wanted = [q for f in plan["new_families"].values() for q in f["members"]]
    assert sorted(wanted) == sorted(questions) and len(questions) == 9
    p.c.bind(Path(__file__))
    p.save(
        ROOT / "independent_verification.json",
        dict(
            status="passed",
            checks=dict(C),
            bootstrap_checks=allchecks,
            questions=questions,
            source_sha256=p.c.BINDINGS,
        ),
    )


if __name__ == "__main__":
    run()
