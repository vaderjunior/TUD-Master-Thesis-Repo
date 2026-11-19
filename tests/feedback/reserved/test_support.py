from copy import deepcopy
from collections import Counter
from fractions import Fraction
import unittest
import numpy as np
from analysis.feedback.reserved import support as s
from analysis.feedback.reserved import rates as rates


def fixture():
    roster = [
        {
            "key": k,
            "gold": {"hate": hate, "hate_type": fine},
            "cluster": cluster,
            "source": source,
        }
        for k, hate, fine, cluster, source in [
            ("a", False, None, "c1", "x"),
            ("b", False, None, "c1", "x"),
            ("c", True, ["explicit"], "c2", "y"),
            ("d", True, ["irony", "grievance"], "c3", "y"),
        ]
    ]
    pred = {
        (r["key"], c, rep): {
            "result": {
                "hate": r["gold"]["hate"],
                "hate_type": r["gold"]["hate_type"] or [],
            },
            "n_valid": 3,
        }
        for r in roster
        for c in ("t1", "t2", "t3", "reference5")
        for rep in range(3)
    }
    spec = {
        "id": "synthetic",
        "endpoint": "FPR",
        "treatment_conditions": ["t1", "t2", "t3"],
        "control_conditions": ["reference5"],
        "seed": 487,
        "draws": 271,
        "confidence": 0.975,
    }
    return roster, pred, spec


class SupportTests(unittest.TestCase):
    def test_hand_confusion_and_unknown_type_truth(self):
        r, p, _ = fixture()
        p["a", "t1", 0]["result"] = {"hate": True, "hate_type": ["threatening"]}
        p["c", "t1", 0]["result"] = {"hate": False, "hate_type": []}
        v = s.metric_rows(r, p, "t1", 0)
        self.assertEqual(
            [v["counts"][k] for k in ("tp", "tn", "fp", "fn")], [1, 1, 1, 1]
        )
        self.assertEqual(v["metrics"]["binary_macro_f1"], 0.5)
        self.assertEqual(v["metrics"]["balanced_accuracy"], 0.5)
        self.assertEqual(v["counts"]["type_valid"], 2)
        self.assertEqual(v["metrics"]["type_macro_f1_fixed7"], 2 / 7)
        self.assertEqual(v["metrics"]["type_exact_set"], 0.5)
        self.assertEqual(v["per_label"]["threatening"]["fp"], 0)

    def test_no_valid_is_not_zero_and_reduced_votes_strict(self):
        r, p, _ = fixture()
        for row in r:
            p[row["key"], "t1", 0]["n_valid"] = 2
        self.assertEqual(s.metric_rows(r, p, "t1", 0)["counts"]["valid"], 4)
        self.assertIsNone(
            s.metric_rows(r, p, "t1", 0, strict=True)["metrics"]["binary_macro_f1"]
        )
        for row in r:
            p[row["key"], "t1", 0] = {"result": None, "n_valid": 0}
        self.assertIsNone(
            s.metric_rows(r, p, "t1", 0)["metrics"]["type_macro_f1_fixed7"]
        )

    def test_fixed_seven_support_not_observed_only(self):
        r, p, _ = fixture()
        v = s.metric_rows(r, p, "t1", 0)
        self.assertEqual(v["metrics"]["type_macro_f1_fixed7"], 3 / 7)
        self.assertEqual(v["metrics"]["type_exact_set"], 1)
        self.assertEqual(v["metrics"]["type_mean_cardinality"], 1.5)

    def test_independent_expanded_metric_bootstrap(self):
        r, p, spec = fixture()
        p["a", "t1", 0]["result"]["hate"] = True
        p["c", "t2", 1]["result"] = {"hate": False, "hate_type": []}
        p["d", "reference5", 2]["result"]["hate_type"] = ["irony", "explicit"]
        for fine in (False, True):
            summary, draws = s.paired_metrics(r, p, spec, fine=fine)
            rows = [x for x in r if not fine or x["gold"]["hate_type"] is not None]
            labels = sorted({x["cluster"] for x in rows})
            rng = np.random.Generator(np.random.PCG64(spec["seed"]))
            manual = []
            for _ in range(spec["draws"]):
                sample = [
                    x
                    for i in rng.integers(0, len(labels), len(labels))
                    for x in rows
                    if x["cluster"] == labels[i]
                ]
                arm_values = []
                for arm in (spec["treatment_conditions"], spec["control_conditions"]):
                    values = []
                    for c in arm:
                        for rep in range(3):
                            if fine:
                                scores = []
                                exact = 0
                                for name in s.TYPES:
                                    tp = fp = fn = 0
                                    for x in sample:
                                        gold = name in x["gold"]["hate_type"]
                                        pred = (
                                            name
                                            in p[x["key"], c, rep]["result"][
                                                "hate_type"
                                            ]
                                        )
                                        tp += gold and pred
                                        fp += not gold and pred
                                        fn += gold and not pred
                                    scores.append(
                                        Fraction(2 * tp, 2 * tp + fp + fn)
                                        if 2 * tp + fp + fn
                                        else Fraction(0)
                                    )
                                exact = sum(
                                    set(x["gold"]["hate_type"])
                                    == set(p[x["key"], c, rep]["result"]["hate_type"])
                                    for x in sample
                                )
                                values.append(
                                    [float(sum(scores) / 7), exact / len(sample)]
                                )
                            else:
                                ct = Counter(
                                    (
                                        "t"
                                        if x["gold"]["hate"]
                                        == p[x["key"], c, rep]["result"]["hate"]
                                        else "f"
                                    )
                                    + (
                                        "p"
                                        if p[x["key"], c, rep]["result"]["hate"]
                                        else "n"
                                    )
                                    for x in sample
                                )
                                tp, tn, fp, fn = (
                                    ct[k] for k in ("tp", "tn", "fp", "fn")
                                )
                                pos = (
                                    Fraction(2 * tp, 2 * tp + fp + fn)
                                    if 2 * tp + fp + fn
                                    else Fraction(0)
                                )
                                neg = (
                                    Fraction(2 * tn, 2 * tn + fp + fn)
                                    if 2 * tn + fp + fn
                                    else Fraction(0)
                                )
                                bal = (
                                    float(Fraction(tp, tp + fn) + Fraction(tn, tn + fp))
                                    / 2
                                    if tp + fn and tn + fp
                                    else float("nan")
                                )
                                values.append([float((pos + neg) / 2), bal])
                    arm_values.append(np.mean(values, axis=0))
                manual.append(arm_values[0] - arm_values[1])
            np.testing.assert_allclose(
                draws, manual, rtol=0, atol=1e-14, equal_nan=True
            )
            if not fine:
                self.assertGreater(
                    summary["metrics"]["balanced_accuracy"]["undefined_draws"], 0
                )
                self.assertIsNone(summary["metrics"]["balanced_accuracy"]["interval95"])

    def test_paired_type_mask_retains_only_known_truth_and_every_output(self):
        r, p, spec = fixture()
        p["c", "t3", 2] = {"result": None, "n_valid": 0}
        summary, _ = s.paired_metrics(r, p, spec, fine=True)
        self.assertEqual(summary["common_ids"], ["d"])
        self.assertEqual(summary["original_n"], 2)
        self.assertEqual(summary["invalid_ids"], ["c"])

    def test_independent_rate_draws_reject_corruption(self):
        r, p, spec = fixture()
        p["a", "t2", 0]["result"]["hate"] = True
        c = rates.cohort(r, p, spec)
        _, draws = rates.summarize(c, spec)
        self.assertEqual(
            s.independent_rate_draws(r, p, spec, c, draws)["draws_checked"], 271
        )
        draws[270] += 0.001
        with self.assertRaises(ValueError):
            s.independent_rate_draws(r, p, spec, c, draws)

    def test_cluster_strata_never_split_mixed_source_cluster(self):
        r, p, spec = fixture()
        r[1]["source"] = "z"
        c = rates.cohort(r, p, spec)
        result, draws = s.source_bootstrap(c, r, spec)
        self.assertEqual(
            result["strata"], [{"source_signature": ["x", "z"], "clusters": 1}]
        )
        np.testing.assert_array_equal(draws, np.zeros(271))

    def test_omission_removes_whole_batch_and_matched_pool(self):
        r, p, spec = fixture()
        spec["treatment_conditions"] = [
            "t_s2026100441",
            "t_s2026100442",
            "t_s2026100443",
        ]
        renamed = {
            (
                k,
                (
                    spec["treatment_conditions"][int(c[1:]) - 1]
                    if c.startswith("t")
                    else c
                ),
                rep,
            ): v
            for (k, c, rep), v in p.items()
        }
        out = s.omission_points(
            r, renamed, spec, {"a": "b1", "b": "b1", "c": "b2", "d": "b2"}
        )
        self.assertEqual(len(out["pool"]), 3)
        self.assertIsNone(out["execution_batch"][0]["difference"])
        self.assertEqual(out["execution_batch"][1]["common_n"], 2)

    def test_transitions_identical_prompts_and_missing_pairs(self):
        r, p, _ = fixture()
        p["a", "reference5", 0]["result"]["hate"] = True
        p["b", "t1", 2] = {"result": None, "n_valid": 0}
        meta = {
            (x["key"], c): {
                "messages_sha256": x["key"],
                "source_gate": False,
                "delivered_correction_ids": [],
            }
            for x in r
            for c in ("t1", "reference5")
        }
        result = s.transitions(r, p, "t1", "reference5", meta)
        self.assertEqual(result["groups"]["identical"]["fix"], 1)
        self.assertEqual(result["groups"]["all"]["missing_pairs"], 1)
        self.assertEqual(result["groups"]["all"]["eligible_pairs"], 12)
        self.assertEqual(len(result["cases"]), 1)


if __name__ == "__main__":
    unittest.main()
