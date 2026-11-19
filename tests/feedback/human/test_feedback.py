from copy import deepcopy
import unittest
from analysis.feedback.human import feedback_common as c
from analysis.feedback.human import feedback_score as s


def item(
    alias="T1", *, a=True, b=True, at=("explicit",), bt=("explicit",), old=True, ot=()
):
    def j(g, t):
        return {
            "binary_hate": g,
            "type_state": (
                "assigned" if g and t else "unassigned" if g else "not_applicable"
            ),
            "types": list(t),
        }

    return {
        "packet_id": alias,
        "id": alias,
        "key": alias,
        "text": alias,
        "cluster": alias,
        "gold": {"hate": True, "hate_type": ["inferiority"]},
        "judgments": {"A_project": j(a, at), "B_project": j(b, bt)},
        "model_repeat_0": {"hate": old, "hate_type": list(ot)},
        "transfer_eligible": True,
        "is_correction_source": False,
        "selected_correction": "R014",
        "retrieval_replay_flag": False,
        "touches_training": False,
    }


def pred(hate=True, types=("explicit",), n=3):
    return {
        "result": {"hate": hate, "hate_type": list(types)} if n else None,
        "n_valid": n,
    }


def matrix(rows, p=None):
    return {
        (r["key"], arm, rep): deepcopy(p or pred())
        for r in rows
        for arm in c.CONDITIONS
        for rep in range(3)
    }


class HumanFeedbackTests(unittest.TestCase):
    def test_unknown_gate_is_not_nonhate(self):
        self.assertIsNone(c.agreed_correction(item(a=None, b=False)))

    def test_disagreed_gate_is_not_adjudicated(self):
        self.assertIsNone(c.agreed_correction(item(a=True, b=False)))

    def test_partial_types_never_become_union_or_intersection(self):
        r = item(
            at=("explicit", "inferiority"), bt=("explicit", "stereotyping"), old=False
        )
        self.assertIsNone(c.agreed_correction(r)["hate_types"])

    def test_partial_types_without_gate_error_are_not_selected(self):
        self.assertIsNone(
            c.agreed_correction(item(at=("explicit",), bt=("inferiority",)))
        )

    def test_already_matching_model_excluded_even_if_dataset_differs(self):
        self.assertIsNone(c.agreed_correction(item(ot=("explicit",))))

    def test_dataset_control_does_not_invent_unavailable_types(self):
        r = item(bt=("inferiority",), old=False)
        self.assertIsNone(
            c.correction_record(r, "dataset_feedback5")["meta"]["hate_types"]
        )

    def test_known_nonhate_has_empty_not_unknown_types(self):
        self.assertEqual(c.agreed_correction(item(a=False, b=False))["hate_types"], [])

    def test_predicted_gate_is_respected(self):
        p = {"hate": False, "hate_type": ["explicit"]}
        self.assertEqual(s.metric(p, True, ["explicit"], "complete"), 0)
        self.assertEqual(s.metric(p, True, ["explicit"], "jaccard"), 0)

    def test_gate_and_complete_set_are_distinct(self):
        p = {"hate": True, "hate_type": ["explicit", "inferiority"]}
        self.assertEqual(s.metric(p, True, ["explicit"], "gate"), 1)
        self.assertEqual(s.metric(p, True, ["explicit"], "complete"), 0)
        self.assertEqual(s.metric(p, True, ["explicit"], "jaccard"), 0.5)

    def test_common_mask_and_opposing_missing_bounds(self):
        rows = [item("x"), item("y")]
        p = matrix(rows)
        for rep in range(3):
            p["y", "human_feedback5", rep] = pred(n=0)
        d = s.endpoint(rows, p, "A", "complete")
        self.assertEqual(d["annotation_eligible_texts"], 2)
        self.assertEqual(d["common_valid_texts"], 1)
        self.assertEqual(
            d["contrasts"]["human_feedback5_minus_dataset_feedback5"][
                "missing_output_bounds_full_eligible"
            ],
            [-0.5, 0.0],
        )

    def test_unresolved_annotation_excluded_not_penalized(self):
        rows = [item("x", a=None), item("y")]
        self.assertEqual(
            s.endpoint(rows, matrix(rows), "A", "gate")["annotation_eligible_texts"], 1
        )

    def test_strict_vote_mask(self):
        rows = [item()]
        p = matrix(rows, pred(n=2))
        self.assertEqual(s.endpoint(rows, p, "A", "complete")["common_valid_texts"], 1)
        self.assertEqual(
            s.endpoint(rows, p, "A", "complete", True)["common_valid_texts"], 0
        )

    def test_repeats_are_averaged_not_independent_texts(self):
        rows = [item()]
        p = matrix(rows)
        p["T1", "human_feedback5", 1] = pred(types=())
        d = s.endpoint(rows, p, "A", "complete")
        self.assertEqual(d["common_valid_texts"], 1)
        self.assertAlmostEqual(
            d["arms"]["human_feedback5"]["mean_text_agreement"], 2 / 3
        )
        self.assertAlmostEqual(
            d["contrasts"]["human_feedback5_minus_dataset_feedback5"][
                "mean_text_transition_counts"
            ]["regressed"],
            1 / 3,
        )

    def test_fixed_seven_labels_and_absent_support(self):
        rows = [item()]
        d = s.label_statistics(rows, matrix(rows), "A")
        self.assertAlmostEqual(d["arms"]["reference5"]["mean_macro_f1"], 1 / 7)
        self.assertEqual(
            d["arms"]["reference5"]["per_repeat"][0]["labels"]["irony"]["support"], 0
        )

    def test_source_uptake_not_pooled_into_transfer(self):
        rows = [item("source"), item("other")]
        rows[0].update(is_correction_source=True, transfer_eligible=False)
        d = s.analyze(rows, matrix(rows))
        self.assertEqual(len(d["source_uptake"]), 1)
        self.assertEqual(d["cohort_sizes"]["other_all"], 1)

    def test_incomplete_identity_matrix_rejected(self):
        rows = [item()]
        p = matrix(rows)
        p.pop(next(iter(p)))
        with self.assertRaises(RuntimeError):
            s.analyze(rows, p)

    def test_correction_context_does_not_mutate_original(self):
        base = {
            "definitions": [{}, {}],
            "guidelines": [{}, {}],
            "example_groups": [
                [
                    "Labelled examples",
                    [{"id": str(i), "text": str(i), "meta": {}} for i in range(5)],
                ]
            ],
        }
        before = deepcopy(base)
        new = c.context(base, c.correction_record(item(), "human_feedback5"))
        self.assertEqual(base, before)
        self.assertEqual(len(new["example_groups"][0][1]), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
