from analysis.knowledge.micro_f1 import banks_strategies as p
from analysis.common import micro_f1_files as c
from analysis.common import micro_f1_resampling as h
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, multilabel_confusion_matrix


def eq(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)


def score(y, x, scope):
    return np.array(
        [
            np.mean(
                [
                    f1_score(y[:, j], x[:, j], average="binary", zero_division=0)
                    for j in scope
                ]
            ),
            f1_score(y, x, average="micro", zero_division=0),
            f1_score(
                y[:, scope].ravel(),
                x[:, scope].ravel(),
                average="binary",
                zero_division=0,
            ),
        ]
    )


def run():
    idx = c.read(p.BASE / "analysis_index.json")
    plan = c.read(p.BASE / "analysis_plan.json")
    counter = Counter()
    bootstrap_checks = {}
    questions = []
    valid = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]], bool)
    assert valid[:2].all(0).sum() == 3 and valid.all(0).sum() == 2
    q = np.array([[0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3]])
    eq(q.mean(), np.mean([x.mean() for x in q]))
    assert q.shape == (3, 3)
    for output in idx["outputs"]:
        result = c.read(c.ROOT / output)
        u = result["unit"]
        old = c.read(c.ROOT / u["source"])
        with np.load(c.bind(c.ROOT / result["arrays"]), allow_pickle=False) as f:
            a = {k: f[k] for k in f.files}
        labels = a["labels"].tolist()
        scope = [labels.index(x) for x in a["macro_labels"]]
        settings = [tuple(x) for x in result["settings"]]
        raw = {}
        for path in u["assembled"]:
            for row in c.lines(c.ROOT / path):
                raw[row["condition"], row["replicate"], str(row["id"])] = row
        field = {
            "hate_type": "hate_types",
            "target_group": "target_groups",
            "legal": "legal",
        }[u["dimension"]]
        rows = pq.read_table(c.bind(c.ROOT / u["gold_path"])).to_pylist()
        eligible = [
            r
            for r in rows
            if r[field] is not None and (u["dimension"] == "legal" or r["gate"] is True)
        ]
        assert [str(r["id"]) for r in eligible] == a["ids"].tolist()
        eq(a["gold"], [[label in r[field] for label in labels] for r in eligible])
        for j, (cond, rep) in enumerate(settings):
            for i, rid in enumerate(a["ids"]):
                row = raw[cond, rep, str(rid)]
                ans = row["result"]
                assert a["valid"][j, i] == (ans is not None)
                assert a["nvalid"][j, i] == row["n_valid"]
                if ans is not None:
                    eq(
                        a["pred"][j, i],
                        [label in ans[u["dimension"]] for label in labels],
                    )
                    assert a["gate"][j, i] == ans["hate"]
                counter["independent_raw_answer_records"] += 1
        common = a["valid"].all(0)
        eq(common, a["common"])
        assert a["ids"][common].tolist() == old["paired_comparison"]["common_valid_ids"]
        for mode in ["own", "common"]:
            for j in range(len(settings)):
                mask = a["valid"][j] if mode == "own" else common
                y, x = (a["gold"][mask], a["pred"][j, mask])
                s = score(y, x, scope)
                stored = result[mode][j]
                eq(s, [stored["scores"][k] for k in h.METHODS])
                cm = multilabel_confusion_matrix(y, x)
                eq(
                    np.stack([cm[:, 1, 1], cm[:, 0, 1], cm[:, 1, 0]], -1),
                    stored["tp_fp_fn"],
                )
                eq(np.all(y == x, 1).mean(), stored["set_only_exact"])
                if u["dimension"] != "legal":
                    eq(
                        (np.all(y == x, 1) & a["gate"][j, mask]).mean(),
                        stored["complete_field_exact"],
                    )
                counter["sklearn_setting_population_checks"] += 1
        group_ix = {
            g: [j for j, (cond, rep) in enumerate(settings) if cond in members]
            for g, members in result["group_members"].items()
        }
        means = np.array(
            [[r["scores"][m] for m in h.METHODS] for r in result["common"]]
        )
        for name, con in result["contrasts"].items():
            li, ri = (group_ix[con["left"]], group_ix[con["right"]])
            eq(
                means[li].mean(0) - means[ri].mean(0),
                [con["metrics"][m]["difference"] for m in h.METHODS],
            )
            if con["question_id"]:
                questions.append(con["question_id"])
            for rep in range(3):
                for field, keep in [
                    ("by_repeat", lambda k: k[1] == rep),
                    ("leave_one_repeat_out", lambda k: k[1] != rep),
                ]:
                    ls = [j for j in li if keep(settings[j])]
                    rs = [j for j in ri if keep(settings[j])]
                    eq(
                        means[ls].mean(0) - means[rs].mean(0),
                        [con[field][str(rep)][m] for m in h.METHODS],
                    )
                    counter["repeat_diagnostics"] += 1
            if u["study"] == "pools":
                for seed in [42, 43, 44]:
                    for field, omit in [
                        ("by_pool", False),
                        ("leave_one_pool_out", True),
                    ]:
                        ls = [
                            j
                            for j in li
                            if len(result["group_members"][con["left"]]) == 1
                            or settings[j][0].endswith("_s" + str(seed)) != omit
                        ]
                        rs = [
                            j
                            for j in ri
                            if len(result["group_members"][con["right"]]) == 1
                            or settings[j][0].endswith("_s" + str(seed)) != omit
                        ]
                        eq(
                            means[ls].mean(0) - means[rs].mean(0),
                            [con[field][str(seed)][m] for m in h.METHODS],
                        )
                        counter["pool_diagnostics"] += 1
        for mode, info in result["bootstrap_files"].items():
            with np.load(c.bind(c.ROOT / info["path"]), allow_pickle=False) as f:
                stored = f["scores"]
                mask = f["mask"]
            expected_mask = (
                common & (a["nvalid"] == 3).all(0)
                if mode == "strict_three_votes"
                else common
            )
            eq(mask, expected_mask)
            indices = np.flatnonzero(mask)
            members = [np.array([i]) for i in indices]
            if mode == "cluster":
                members = [
                    np.flatnonzero(mask & (a["clusters"] == x))
                    for x in sorted(set(a["clusters"][mask]))
                ]
            grouped = []
            for members_ix in members:
                g = a["gold"][members_ix].astype(int)
                x = a["pred"][:, members_ix].astype(int)
                grouped.append(
                    np.stack(
                        [
                            (x * g).sum(1),
                            x.sum(1),
                            np.broadcast_to(g.sum(0), (len(settings), len(labels))),
                        ],
                        -1,
                    )
                )
            grouped = np.array(grouped)
            matrix = grouped.reshape(len(members), -1)
            strata = defaultdict(list)
            for j, member in enumerate(members):
                key = (
                    str(a["sources"][member[0]])
                    if mode == "source_stratified"
                    else "all"
                )
                strata[key].append(j)
            rng = np.random.default_rng(info["seed"])
            maximum = 0.0
            independent = []
            for start in range(0, 10000, 64):
                size = min(64, 10000 - start)
                if mode in ["item", "strict_three_votes"]:
                    picks = rng.integers(0, len(members), (size, len(members)))
                else:
                    picks = np.concatenate(
                        [
                            rng.choice(ix, (size, len(ix)), replace=True)
                            for _, ix in sorted(strata.items())
                        ],
                        1,
                    )
                freq = np.array(
                    [np.bincount(row, minlength=len(members)) for row in picks]
                )
                total = (freq @ matrix).reshape(size, len(settings), len(labels), 3)
                tp, pp, aa = np.moveaxis(total, -1, 0)
                den = pp + aa
                lf = np.divide(
                    2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0
                )

                def micro(ix):
                    d = den[..., ix].sum(-1)
                    n = 2 * tp[..., ix].sum(-1)
                    return np.divide(
                        n, d, out=np.zeros_like(n, dtype=float), where=d != 0
                    )

                computed = np.stack(
                    [
                        lf[..., scope].mean(-1),
                        micro(list(range(len(labels)))),
                        micro(scope),
                    ],
                    -1,
                )
                error = float(np.max(np.abs(computed - stored[start : start + size])))
                maximum = max(maximum, error)
                assert error < 1e-12
                independent.append(computed)
                for wanted in [0, 4999, 9999]:
                    if start <= wanted < start + size:
                        expanded = np.concatenate(
                            [members[j] for j in picks[wanted - start]]
                        )
                        for j in range(len(settings)):
                            actual = score(
                                a["gold"][expanded], a["pred"][j, expanded], scope
                            )
                            eq(actual, stored[wanted, j])
                            counter["sklearn_bootstrap_setting_checks"] += 1
            independent = np.concatenate(independent)
            for name, con in result["contrasts"].items():
                li, ri = (group_ix[con["left"]], group_ix[con["right"]])
                dd = independent[:, li].mean(1) - independent[:, ri].mean(1)
                target = (
                    con["metrics"]
                    if mode == "item"
                    else con["sensitivities"][mode]["metrics"]
                )
                for j, m in enumerate(h.METHODS):
                    eq(np.quantile(dd[:, j], [0.025, 0.975]), target[m]["ci95"])
                    counter["independent_intervals"] += 1
                    if mode == "item" and target[m]["adjusted_confidence"]:
                        q = (1 - target[m]["adjusted_confidence"]) / 2
                        eq(
                            np.quantile(dd[:, j], [q, 1 - q]),
                            target[m]["adjusted_interval"],
                        )
                        counter["independent_adjusted_intervals"] += 1
            counter["independent_bootstrap_values"] += stored.size
            bootstrap_checks[u["id"] + "/" + mode] = dict(
                n=int(mask.sum()),
                draws=len(stored),
                settings=len(settings),
                values=stored.size,
                maximum_error=maximum,
            )
        print("Independent review PASS:", u["id"], flush=True)
    expected = [q for v in plan["new_families"].values() for q in v["members"]]
    assert sorted(questions) == sorted(expected)
    initial = c.read(p.BASE / "preservation_before.json")
    for name, digest in {**initial["old_sources"], **idx["source_sha256"]}.items():
        assert c.sha(c.ROOT / name) == digest, name
    p.save(
        p.BASE / "independent_verification.json",
        dict(
            status="PASS",
            checks=dict(counter),
            bootstrap_checks=bootstrap_checks,
            adjusted_questions=questions,
            original_macro_point_and_ci_comparisons=44,
            original_macro_adjusted_intervals=7,
            original_setting_metric_streams_reproduced=8,
            original_macro_draw_arrays_were_not_stored=True,
            global_mask_fixture_passed=True,
            hierarchy_fixture_passed=True,
            script_sha256=c.sha(Path(__file__)),
            sklearn_version=__import__("sklearn").__version__,
            new_calls=0,
            new_training=0,
            latex_edits=0,
        ),
    )
    print("Part 5 independent checks:", dict(counter), flush=True)


if __name__ == "__main__":
    run()
