from copy import deepcopy
import unittest
from analysis.feedback.reserved import controls as a
from analysis.feedback.reserved import controls_context as c


class Part3(unittest.TestCase):
    def fixture(self, t_count=3, c_count=3):
        roster = [
            {
                "key": "k" + str(i),
                "cluster": "g" + str(i // 2),
                "source": "s" + str(i // 2),
                "gold": {"hate": i > 2},
            }
            for i in range(6)
        ]
        ts = ["t" + str(i) for i in range(t_count)]
        cs = ["c" + str(i) for i in range(c_count)]
        pred = {}
        for i, row in enumerate(roster):
            for j, condition in enumerate(ts + cs):
                for rep in range(3):
                    pred[row["key"], condition, rep] = {
                        "result": {"hate": (i * 11 + j * 5 + rep * 2) % 7 < 2},
                        "n_valid": 3,
                    }
        spec = {
            "id": "synthetic",
            "endpoint": "FPR",
            "family": "mechanism",
            "treatment_conditions": ts,
            "control_conditions": cs,
            "seed": 1234,
            "draws": 1000,
            "confidence": 1 - 0.05 / 6,
        }
        return roster, pred, spec

    def test_integer_nine_vs_nine_with_duplicate_clusters(self):
        roster, pred, spec = self.fixture()
        cohort = a.rates.cohort(roster, pred, spec)
        result, draws = a.rates.summarize(cohort, spec)
        check = a.integer_draw_check(roster, pred, spec, cohort, draws)
        self.assertEqual(check["outputs_per_input"], [9, 9])
        self.assertEqual(check["tails"][0]["q"], "1/240")
        self.assertAlmostEqual(check["exact_interval"][0], result["family_interval"][0])
        a.prior.exact_check(roster, pred, spec, result)

    def test_integer_nine_vs_three_and_three_vs_three(self):
        for nt, nc in ((3, 1), (1, 1)):
            roster, pred, spec = self.fixture(nt, nc)
            spec["confidence"] = 0.95
            cohort = a.rates.cohort(roster, pred, spec)
            result, draws = a.rates.summarize(cohort, spec)
            check = a.integer_draw_check(roster, pred, spec, cohort, draws)
            self.assertEqual(check["outputs_per_input"], [3 * nt, 3 * nc])
            self.assertEqual(check["tails"][0]["q"], "1/40")
            a.prior.exact_check(roster, pred, spec, result)

    def test_source_strata_integer(self):
        roster, pred, spec = self.fixture()
        cohort = a.rates.cohort(roster, pred, spec)
        result, draws = a.support.source_bootstrap(cohort, roster, spec)
        check = a.integer_draw_check(roster, pred, spec, cohort, draws, stratified=True)
        self.assertTrue(check["source_stratified"])
        self.assertAlmostEqual(check["exact_interval"][1], result["interval95"][1])

    def test_null_mask_and_strict_unknown_are_not_errors(self):
        roster, pred, spec = self.fixture()
        pred["k0", "t0", 0] = {"result": None, "n_valid": 0}
        pred["k1", "c0", 1]["n_valid"] = 2
        for strict, n in ((False, 2), (True, 1)):
            cohort = a.rates.cohort(roster, pred, spec, strict=strict)
            result, draws = a.rates.summarize(cohort, spec)
            self.assertEqual(result["common_n"], n)
            a.prior.exact_check(roster, pred, spec, result, strict)
            a.integer_draw_check(roster, pred, spec, cohort, draws)

    def test_supporting_exclusion_passed_to_independent_checker(self):
        roster, pred, spec = self.fixture(1, 1)
        spec["confidence"] = 0.95
        excluded = ["k0"]
        cohort = a.rates.cohort(roster, pred, spec, excluded=excluded)
        result, draws = a.rates.summarize(cohort, spec)
        with self.assertRaises(ValueError):
            a.prior.exact_check(roster, pred, spec, result)
        checked = a.prior.exact_check(roster, pred, spec, result, excluded=excluded)
        self.assertEqual(checked["eligible_n"], 2)
        a.integer_draw_check(roster, pred, spec, cohort, draws)

    def prompt_fixture(self):
        ordinary = [
            {
                "id": str(i),
                "text": "example " + str(i),
                "meta": {"gate": False},
                "via": "rrf",
            }
            for i in range(5)
        ]
        base = {
            "text": "query",
            "context": {
                "definitions": [],
                "guidelines": [],
                "example_groups": [["Labelled examples", ordinary]],
            },
            "messages": [
                {"role": "system", "text": "task"},
                {"role": "user", "text": "body"},
            ],
        }
        correction = {
            "id": "source",
            "text": "correction",
            "meta": dict(
                gate=True,
                hate_types=None,
                target_groups=None,
                severity=None,
                legal=None,
            ),
            "via": "rrf",
        }
        return base, correction

    def test_context_preservation_and_reject_displacement(self):
        base, source = self.prompt_fixture()
        for mode, count in (
            ("reference4", 4),
            ("reserved5", 4),
            ("masked5", 4),
            ("extended6", 5),
        ):
            p = deepcopy(base)
            p["context"]["example_groups"][0][1] = deepcopy(c.examples(base)[:count])
            if mode != "reference4":
                hit = deepcopy(source)
                if mode == "masked5":
                    hit["meta"]["gate"] = None
                c.examples(p).append(hit)
            c.check_intervention(base, p, mode, source)
            p["context"]["example_groups"][0][1][0]["text"] = "changed"
            with self.assertRaises(ValueError):
                c.check_intervention(base, p, mode, source)

    def test_masking_removes_label_and_never_invents_non_hate(self):
        base, source = self.prompt_fixture()
        reserved = deepcopy(base)
        c.examples(reserved)[4] = source
        reserved["messages"][1][
            "text"
        ] = 'before\n- "correction"\n  labels: hate=true\n\nafter'
        masked = deepcopy(reserved)
        c.examples(masked)[4]["meta"]["gate"] = None
        masked["messages"][1]["text"] = reserved["messages"][1]["text"].replace(
            "hate=true", ""
        )
        c.check_masked_message(reserved, masked)
        masked["messages"][1]["text"] = reserved["messages"][1]["text"].replace(
            "hate=true", "hate=false"
        )
        with self.assertRaises(ValueError):
            c.check_masked_message(reserved, masked)

    def test_transition_sign_and_unique_texts(self):
        from collections import Counter

        g = Counter(keys=set())
        c.pair_update(g, "one", True, False, False)
        c.pair_update(g, "one", False, False, False)
        result = c.finish_groups({"x": g})["x"]
        self.assertEqual(result["unique_texts"], 1)
        self.assertEqual(result["difference"], 0.5)
        self.assertEqual(result["regression"], 1)


if __name__ == "__main__":
    unittest.main()
