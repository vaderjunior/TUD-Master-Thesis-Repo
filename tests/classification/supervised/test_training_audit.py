import copy
import math
from types import SimpleNamespace
import unittest

import numpy as np
from sklearn.metrics import f1_score
from analysis.classification.supervised import training_audit as t

from training.recovery import update_rules as rules


def trace_fixture(total=2, amp=False):
    count = (total - 1) * 16 + 1
    fit = [dict(id=str(i), target=i % 2) for i in range(count)]
    held = [dict(id="held", target=1)]
    job = dict(
        head="hate", mode="search", config="fixture", seed=0, fold=0, id="fixture"
    )
    cfg = dict(id="fixture", epochs=1, learning_rate=2e-5, loss="unweighted")
    y = np.eye(2, dtype=int)[[x["target"] for x in fit]]
    split = dict(
        fit=fit, held=held, weights={"unweighted": t.class_weights("hate", y, False)}
    )
    steps = []
    scale = 65536.0
    success = 0
    for i in range(total):
        skip = amp and i == 1
        lr = (
            0.0
            if success == 0
            else cfg["learning_rate"] * (total - success) / max(1, total - 1)
        )
        after = scale / 2 if skip else scale
        steps.append(
            dict(
                step=i + 1,
                epoch=1,
                lr=lr,
                skipped=skip,
                scale_before=scale,
                scale_after=after,
                grad_norm="nan" if skip else 1.0,
            )
        )
        if not skip:
            success += 1
        scale = after
    unchanged = total == 1 or amp
    fingerprint = dict(
        raw_sha256="0" * 64, zero_canonical_sha256="1" * 64, negative_zero_count=0
    )
    changes = {k: 0.0 if unchanged else 0.01 for k in t.WATCHED}
    eq = (
        dict(before=fingerprint, after=fingerprint, changes=changes)
        if unchanged
        else None
    )
    m = dict(
        job=copy.deepcopy(job),
        configuration=copy.deepcopy(cfg),
        pilot_only=False,
        fit_ids=[x["id"] for x in fit],
        held_ids=["held"],
        planned_steps=total,
        attempted_steps=total,
        successful_steps=success,
        skipped_steps=total - success,
        steps=steps,
        epochs=[
            dict(
                epoch=1,
                attempted_steps=total,
                successful_steps=success,
                training_objective=1.0,
                held_unweighted_loss=1.0,
            )
        ],
        initialization_only=total == 1,
        changes=changes,
        initial_parameter_fingerprint=fingerprint if unchanged else None,
        final_parameter_fingerprint=fingerprint if unchanged else None,
        class_weights=split["weights"]["unweighted"],
        loading=dict(
            error_msgs=[],
            mismatched_keys=[],
            missing_keys=[
                "classifier.dense.weight",
                "classifier.dense.bias",
                "classifier.out_proj.weight",
                "classifier.out_proj.bias",
            ],
            unexpected_keys=["lm_head.bias"],
        ),
        optimizer="AdamW; correct bias correction; no bias/LayerNorm weight decay; linear warmup/decay",
        training_precision="FP16 autocast, FP32 parameters",
        inference_precision="FP32; no autocast",
        model_directory=None,
        export_check=None,
        train_seconds=0.1,
        total_seconds=0.2,
        token_lengths=dict(fit_at128=0, fit_max=12),
    )
    if amp:
        m.update(update_state="amp_initialization_only", accepted_positive_updates=0)
    return m, job, split, cfg, eq


def data_fixture():
    texts = ["A", "B", "C", "D"]
    gate_ids = ["a", "b", "c", "d"]
    heads = {}
    for head in t.r.HEADS:
        rows = []
        for i in range(4):
            target = (
                i % 2
                if head in ("hate", "severity")
                else [int(i % 2 == 0), int(i % 2 == 1)]
            )
            rows.append(
                dict(
                    id=gate_ids[i],
                    feature_row=i,
                    target=target,
                    cluster=t.r.signature(t.r.normalized(texts[i])),
                    text_hash=t.r.signature(texts[i]),
                    source="synthetic",
                )
            )
        heads[head] = dict(rows=rows, labels=["a", "b"], kind="fixture")
    return dict(
        texts=texts,
        gate_ids=gate_ids,
        heads=heads,
        folds={str(s): [0, 1, 2, 3] for s in (42, 43, 44)},
    )


class TraceTests(unittest.TestCase):
    def check(self, total=2, amp=False):
        m, j, s, c, e = trace_fixture(total, amp)
        return t.trace_review(m, j, s, c, rules, e)

    def bad(self, edit, total=2, amp=False):
        m, j, s, c, e = trace_fixture(total, amp)
        edit(m, j, s, c, e)
        with self.assertRaises((t.r.IntegrityError, RuntimeError)):
            t.trace_review(m, j, s, c, rules, e)

    def test_trained(self):
        self.assertEqual(self.check()["positive_updates"], 1)

    def test_one_step_not_called_trained(self):
        self.assertEqual(self.check(1)["state"], "initialization_only")

    def test_two_step_amendment(self):
        self.assertEqual(self.check(2, True)["state"], "amp_initialization_only")

    def test_wrong_learning_rate(self):
        self.bad(lambda m, *_: m["steps"][1].update(lr=1e-5))

    def test_wrong_seed(self):
        self.bad(lambda m, j, *_: j.update(seed=2))

    def test_evaluation_id_in_fit(self):
        self.bad(lambda m, *_: m["fit_ids"].append("test-row"))

    def test_wrong_horizon(self):
        self.bad(lambda m, *_: m.update(planned_steps=3))

    def test_missing_epoch(self):
        self.bad(lambda m, *_: m.update(epochs=[]))

    def test_wrong_class_weights(self):
        self.bad(lambda m, *_: m.update(class_weights={"balanced": True}))

    def test_skipped_step_accounting(self):
        self.bad(lambda m, *_: m.update(skipped_steps=1))

    def test_scaler_continuity(self):
        self.bad(lambda m, *_: m["steps"][1].update(scale_before=32768.0))

    def test_nonfinite_accepted_gradient(self):
        self.bad(lambda m, *_: m["steps"][1].update(grad_norm="nan"))

    def test_unexpected_no_update(self):
        self.bad(lambda m, *_: m.update(changes={k: 0.0 for k in t.WATCHED}))

    def test_missing_parameter_proof(self):
        m, j, s, c, e = trace_fixture(2, True)
        with self.assertRaises(RuntimeError):
            t.trace_review(m, j, s, c, rules, None)

    def test_changed_whole_model_fingerprint(self):
        def edit(m, j, s, c, e):
            m["final_parameter_fingerprint"] = dict(
                m["final_parameter_fingerprint"], zero_canonical_sha256="2" * 64
            )

        self.bad(edit, amp=True)

    def test_wrong_amended_state(self):
        self.bad(lambda m, *_: m.update(update_state="trained"), amp=True)

    def test_wrong_pretrained_head(self):
        self.bad(lambda m, *_: m["loading"]["missing_keys"].append("encoder.weight"))

    def test_pilot_excluded(self):
        self.bad(lambda m, *_: m.update(pilot_only=True))

    def test_search_must_not_export(self):
        self.bad(lambda m, *_: m.update(model_directory="/model"))

    def test_nonfinite_epoch_loss(self):
        self.bad(lambda m, *_: m["epochs"][0].update(held_unweighted_loss=float("nan")))


class ScoringTests(unittest.TestCase):
    def test_single_supported_label_uses_positive_class_F1(self):
        y = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        z = np.array([[0, 0], [0, 0], [1, 0], [0, 0]])
        self.assertEqual(t.macro_score(y, z, [1]), 0.0)

    def test_scores_against_sklearn(self):
        rng = np.random.default_rng(7)
        for columns in ([0], [1, 3], [0, 1, 2, 3]):
            y = rng.integers(0, 2, (45, 4))
            z = rng.integers(0, 2, (45, 4))
            self.assertAlmostEqual(
                t.macro_score(y, z, columns),
                np.mean([f1_score(y[:, i], z[:, i]) for i in columns]),
                places=14,
            )

    def test_empty_denominator_zero(self):
        self.assertEqual(
            t.macro_score(np.zeros((4, 1), int), np.zeros((4, 1), int), [0]), 0.0
        )

    def test_train_support_two_each(self):
        self.assertEqual(
            t.supported_columns(np.array([[1, 1], [1, 0], [0, 0], [0, 0]])), [0]
        )

    def test_gate_and_cutoff_ties(self):
        probs = np.full((2, 2), 0.5)
        self.assertTrue((t.decisions("hate", probs, 0.5) == [1, 0]).all())
        self.assertTrue(
            (
                t.decisions("hate_type", probs, 0.5, np.array([True, False]))
                == [[1, 1], [0, 0]]
            ).all()
        )

    def test_severity_tie_first(self):
        self.assertEqual(
            t.decisions("severity", np.full((1, 3), 0.5), None).tolist(), [[1, 0, 0]]
        )

    def test_wrong_gate_shape(self):
        with self.assertRaises(t.r.IntegrityError):
            t.decisions("hate_type", np.full((2, 2), 0.5), 0.5, np.array([True]))

    def test_oof_duplicate(self):
        out = np.full((3, 4, 2), np.nan)
        t.fill_oof(out, 0, [1, 3], np.ones((2, 2)))
        with self.assertRaises(t.r.IntegrityError):
            t.fill_oof(out, 0, [1, 3], np.ones((2, 2)))

    def test_oof_wrong_width(self):
        with self.assertRaises(t.r.IntegrityError):
            t.fill_oof(np.full((3, 4, 2), np.nan), 0, [1, 3], np.ones((2, 1)))

    def test_neural_tie_order(self):
        cfg = {
            "short": dict(epochs=1, learning_rate=5e-6, loss="unweighted"),
            "long": dict(epochs=2, learning_rate=2e-5, loss="balanced"),
        }
        rows = [
            dict(config=c, mean_score=0.5, threshold=v)
            for c in cfg
            for v in (0.4, 0.5, 0.6)
        ]
        self.assertEqual(
            min(rows, key=lambda x: t.neural_rank(x, cfg)),
            dict(config="short", mean_score=0.5, threshold=0.5),
        )

    def test_linear_tie_order(self):
        rows = [
            dict(C=c, score=0.5, threshold=v)
            for c in (0.1, 1.0)
            for v in (0.3, 0.5, 0.7)
        ]
        self.assertEqual(
            min(rows, key=t.linear_rank), dict(C=0.1, score=0.5, threshold=0.5)
        )


class PartitionTests(unittest.TestCase):
    def test_valid_data(self):
        result, splits = t.data_review(data_fixture(), [dict(text="unseen")])
        self.assertEqual(result["texts"], 4)
        self.assertEqual(splits["hate", 4]["held"], [])

    def test_normalized_overlap(self):
        with self.assertRaises(t.r.IntegrityError):
            t.data_review(data_fixture(), [dict(text=" A ")])

    def test_wrong_annotation_identity(self):
        data = data_fixture()
        data["heads"]["hate_type"]["rows"][0]["id"] = "other"
        with self.assertRaises(t.r.IntegrityError):
            t.data_review(data, [])

    def test_changed_text(self):
        data = data_fixture()
        data["texts"][0] = "changed"
        with self.assertRaises(t.r.IntegrityError):
            t.data_review(data, [])

    def test_group_leak(self):
        data = data_fixture()
        data["texts"][1] = " a "
        with self.assertRaises(t.r.IntegrityError):
            t.data_review(data, [])

    def test_wrong_fold(self):
        data = data_fixture()
        data["folds"]["42"][0] = 5
        with self.assertRaises(t.r.IntegrityError):
            t.data_review(data, [])

    def test_duplicate_train_id(self):
        data = data_fixture()
        data["gate_ids"][0] = data["gate_ids"][1]
        with self.assertRaises(t.r.IntegrityError):
            t.data_review(data, [])


def selection_fixture():
    data = data_fixture()
    configs = {
        k: dict(id=k, epochs=e, learning_rate=rate, loss=loss)
        for k, e, rate, loss in (
            ("l1_e20_u", 20, 2e-5, "unweighted"),
            ("l1_e20_b", 20, 2e-5, "balanced"),
            ("fast", 1, 5e-6, "unweighted"),
        )
    }
    rng = np.random.default_rng(4)
    oof = {h: {cfg: rng.random((3, 4, 2)) for cfg in configs} for h in t.r.HEADS}
    families = {}
    states = {
        (h, c, s, f): "trained"
        for h in t.r.HEADS
        for c in configs
        for s in t.CV_SEEDS
        for f in range(5)
    }
    for family in t.r.FAMILIES:
        heads = {}
        gate = None
        allowed = (
            list(configs)
            if family == "tuned"
            else ["l1_e20_u" if family == "fixed_unweighted" else "l1_e20_b"]
        )
        for h in t.r.HEADS:
            yy = t.truth(h, data["heads"][h])
            candidates = []
            for cfg in allowed:
                for cutoff in ((None,) if h == "severity" else t.CUTOFFS):
                    scores = []
                    for si in range(3):
                        p = oof[h][cfg][si]
                        if h == "hate":
                            yes = p[:, 1] > cutoff
                            z = np.column_stack((~yes, yes)).astype(int)
                        elif h == "severity":
                            z = np.eye(2, dtype=int)[p.argmax(1)]
                        else:
                            z = (p >= cutoff).astype(int)
                        if gate is not None:
                            z[~gate[si]] = 0
                        scores.append(
                            float(
                                np.mean(
                                    [
                                        f1_score(yy[:, j], z[:, j], zero_division=0)
                                        for j in range(2)
                                    ]
                                )
                            )
                        )
                    candidates.append(
                        dict(
                            config=cfg,
                            threshold=cutoff,
                            mean_score=float(np.mean(scores)),
                            seed_scores=scores,
                        )
                    )
            chosen = copy.deepcopy(
                min(candidates, key=lambda x: t.neural_rank(x, configs))
            )
            heads[h] = dict(
                selected=chosen, candidates=candidates, selection_indices=[0, 1]
            )
            if h == "hate":
                gate = oof[h][chosen["config"]][:, :, 1] > chosen["threshold"]
        families[family] = heads
    return data, families, oof, configs, states


class SelectionTests(unittest.TestCase):
    def test_reconstruct_every_candidate(self):
        result, _ = t.review_selection(*selection_fixture())
        self.assertEqual(result["candidate_checks"], 140)
        self.assertEqual(result["selected_sklearn_score_checks"], 36)

    def test_bad_candidate_score(self):
        data, families, oof, cfg, states = selection_fixture()
        families["tuned"]["hate_type"]["candidates"][0]["mean_score"] += 0.01
        with self.assertRaises(t.r.IntegrityError):
            t.review_selection(data, families, oof, cfg, states)

    def test_missing_candidate(self):
        data, families, oof, cfg, states = selection_fixture()
        families["tuned"]["severity"]["candidates"].pop()
        with self.assertRaises(t.r.IntegrityError):
            t.review_selection(data, families, oof, cfg, states)

    def test_wrong_winner(self):
        data, families, oof, cfg, states = selection_fixture()
        values = families["tuned"]["hate"]["candidates"]
        families["tuned"]["hate"]["selected"] = max(
            values, key=lambda x: t.neural_rank(x, cfg)
        )
        with self.assertRaises(t.r.IntegrityError):
            t.review_selection(data, families, oof, cfg, states)

    def test_wrong_support(self):
        data, families, oof, cfg, states = selection_fixture()
        families["tuned"]["hate_type"]["selection_indices"] = [0]
        with self.assertRaises(t.r.IntegrityError):
            t.review_selection(data, families, oof, cfg, states)


if __name__ == "__main__":
    unittest.main(verbosity=2)
