import contextlib
import copy
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import numpy as np
from training.matched_annotations import run as m

c = m.c


def fixture():
    rows = [
        dict(
            id=f"toy-{i}",
            feature_row=i,
            vector_index=i,
            cluster=f"g{i}",
            target=[int((i + j) % 4 == 0) for j in range(7)],
            source="synthetic",
        )
        for i in range(67)
    ]
    return dict(
        pool="toy",
        heads={
            m.HEAD: dict(
                labels=[f"label{j}" for j in range(7)], kind="multilabel", rows=rows
            )
        },
        texts=[f"Synthetic training row {i}" for i in range(67)],
        folds={"42": [i % 5 for i in range(67)]},
        gate_oof=[i % 6 != 0 for i in range(67)],
    )


class Contracts(unittest.TestCase):
    def test_data(self):
        m.audit_data(fixture())

    def test_missing_row(self):
        d = fixture()
        d["heads"][m.HEAD]["rows"].pop()
        with self.assertRaises(RuntimeError):
            m.audit_data(d)

    def test_label_shape(self):
        d = fixture()
        d["heads"][m.HEAD]["rows"][0]["target"] = [2] * 7
        with self.assertRaises(RuntimeError):
            m.audit_data(d)

    def test_crossfold_group(self):
        d = fixture()
        d["heads"][m.HEAD]["rows"][0]["cluster"] = "g1"
        with self.assertRaises(RuntimeError):
            m.audit_data(d)

    def test_empty_fold(self):
        d = fixture()
        d["folds"]["42"] = [i % 4 for i in range(67)]
        with self.assertRaises(RuntimeError):
            m.audit_data(d)

    def test_gate_length(self):
        d = fixture()
        d["gate_oof"].pop()
        with self.assertRaises(RuntimeError):
            m.audit_data(d)

    def test_gate_really_suppresses(self):
        pp = np.full((2, 7), 0.8)
        out = c.decisions(m.HEAD, pp, 0.5, [True, False])
        np.testing.assert_array_equal(out, [[1] * 7, [0] * 7])

    def test_exact_cutoff(self):
        out = c.decisions(m.HEAD, np.full((1, 7), 0.5), 0.5, [True])
        self.assertEqual(out.sum(), 7)

    def test_job_inventory(self):
        j = m.search_jobs("x")
        self.assertEqual(len(j), 180)
        self.assertEqual(len({r["id"] for r in j}), 180)
        self.assertEqual({r["seed"] for r in j}, {0, 1, 2})

    def test_final_reuse(self):
        s = {f: dict(selected=dict(config="l1_e20_b")) for f in c.FAMILIES}
        j = m.final_jobs("x", s)
        self.assertEqual(len(j), 5)
        self.assertEqual({v["seed"] for v in j}, {10, 11, 12, 13, 14})

    def test_selection_all_seeds_and_controls(self):
        d = fixture()
        y = np.asarray([r["target"] for r in d["heads"][m.HEAD]["rows"]])
        good = y * 0.8 + 0.1
        oo = {cfg: np.stack([good] * 3) for cfg in c.config_map()}
        s = m.select_neural(d, oo)
        self.assertEqual(set(s), set(c.FAMILIES))
        self.assertEqual(s["tuned"]["selected"]["config"], "l0_e20_u")
        self.assertEqual(len(s["tuned"]["candidates"]), 108)
        self.assertEqual(len(s["fixed_balanced"]["candidates"]), 9)
        self.assertEqual(len(s["tuned"]["selected"]["seed_scores"]), 3)

    def test_incomplete_oof(self):
        d = fixture()
        oo = {cfg: np.full((3, 67, 7), 0.5) for cfg in c.config_map()}
        oo["l0_e20_u"][0, 0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            m.select_neural(d, oo)

    def test_write_isolation(self):
        old = c.BASE
        with m.isolated():
            self.assertEqual(c.BASE, m.BASE)
            with self.assertRaises(RuntimeError):
                c.own(old / "should_not_exist.json")
        self.assertEqual(c.BASE, old)

    def test_wrong_feature_text(self):
        with self.assertRaises(RuntimeError):
            m.check_feature_origin(
                {"key": "a", "exact_text_sha256": "one"},
                {"key": "a", "exact_text_sha256": "two"},
            )

    def test_wrong_feature_key(self):
        with self.assertRaises(RuntimeError):
            m.check_feature_origin(
                {"key": "a", "exact_text_sha256": "one"},
                {"key": "b", "exact_text_sha256": "one"},
            )

    def test_matching_feature_origin(self):
        m.check_feature_origin(
            {"key": "a", "exact_text_sha256": "one"},
            {"key": "a", "exact_text_sha256": "one"},
        )

    def test_synthetic_pipeline_resume_corruption(self):
        target = m.BASE / "development" / ("synthetic_" + uuid.uuid4().hex[:12])
        d = fixture()
        count = {"fits": 0, "failed": False}
        fakeplan = {
            "inputs": {},
            "prepared": {},
            "base_cache": {"snapshot": "synthetic only"},
            "synthetic": True,
        }

        def fake_train(attempt, j, dd, cache, **kw):
            count["fits"] += 1
            if count["fits"] == 5 and not count["failed"]:
                count["failed"] = True
                raise RuntimeError("synthetic interrupted fit")
            spec, fit, held = m.trainer.split(dd, j)
            cfg = c.config_map()[j["config"]]
            total = math.ceil(len(fit) / 16) * cfg["epochs"]
            z = np.full((len(held), 7), 0.125, dtype=np.float32)
            yy = np.asarray([v["target"] for v in fit])
            epochs = []
            for epoch in range(1, cfg["epochs"] + 1):
                v = dict(epoch=epoch, training_objective=0.5)
                if held:
                    v["held_unweighted_loss"] = float(
                        np.mean(
                            np.logaddexp(0.0, z.astype(float))
                            - np.asarray([r["target"] for r in held]) * z
                        )
                    )
                epochs.append(v)
            meta = dict(
                job=j,
                pilot_only=False,
                configuration=cfg,
                fit_ids=[v["id"] for v in fit],
                held_ids=[v["id"] for v in held],
                attempted_steps=total,
                planned_steps=total,
                successful_steps=total,
                steps=[dict(skipped=False) for _ in range(total)],
                epochs=epochs,
                class_weights=c.class_weights(m.HEAD, yy, 7, cfg["loss"] == "balanced"),
                export_check={"clean_reload": True, "max_abs_logit_difference": 0},
            )
            c.save(attempt / "training.json", meta)
            if held:
                c.array(attempt / "held_logits.npy", z)
            else:
                c.save(attempt / "model/synthetic.json", {"synthetic": True})
            return meta

        def fake_linear(pool, ps):
            path = target / f"linear/{pool}/synthetic.json"
            m.save(path, {"synthetic": True})
            m.save(
                target / f"linear/{pool}/complete.json",
                {"plan_sha256": ps, "outputs": m.hashes([path])},
            )

        fakecuda = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
        with (
            patch.object(m, "BASE", target),
            patch.object(m, "POOL_NAMES", ["toy"]),
            m.isolated(),
            patch.object(m, "validate", return_value=fakeplan),
            patch.object(m, "make_features"),
            patch.object(m, "linear_stage", side_effect=fake_linear),
            patch.object(m.trainer, "train_attempt", side_effect=fake_train),
            patch.object(m.trainer, "setup", return_value=fakecuda),
            patch.object(m.trainer, "load", return_value=(object(), object(), {})),
            patch.object(m.trainer, "tokens", side_effect=lambda tok, texts: texts),
            patch.object(
                m.trainer,
                "infer",
                side_effect=lambda model, enc: np.full(
                    (len(enc), 7), 0.125, dtype=np.float32
                ),
            ),
            patch.object(
                m.shutil, "disk_usage", return_value=SimpleNamespace(free=200 * 2**30)
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            m.save(target / "plan.json", fakeplan)
            m.save(target / "data/toy.json", d)
            m.save(
                target / "data/evaluation_inputs.json",
                [
                    dict(key=f"eval-{i}", text=f"Synthetic evaluation {i}")
                    for i in range(600)
                ],
            )
            m.arrays(
                target / "data/evaluation_features.npz",
                dict(gate=np.array([i % 2 for i in range(600)])),
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic interrupted"):
                m.run()
            self.assertEqual(len(list((target / "search").rglob("complete.json"))), 4)
            m.run()
            before = count["fits"]
            m.run()
            self.assertEqual(count["fits"], before)
            self.assertEqual(len(list((target / "search").rglob("complete.json"))), 180)
            paths = list((target / "predictions").rglob("*.npz"))
            self.assertEqual(len(paths), 30)
            with np.load(paths[0]) as a:
                self.assertEqual(a["delivered"].shape, (600, 7))
                self.assertEqual(a["delivered"][::2].sum(), 0)
            chosen = paths[0]
            previous = chosen.read_bytes()
            chosen.write_bytes(b"synthetic corruption")
            with self.assertRaises(RuntimeError):
                m.run()
            chosen.write_bytes(previous)
            complete = target / "pool_complete/toy.json"
            previous = complete.read_bytes()
            bad = c.read(complete)
            bad["pool"] = "wrong"
            complete.write_bytes(c.encoded(bad))
            with self.assertRaises(RuntimeError):
                m.verify_pool("toy", c.sha(target / "plan.json"))
            complete.write_bytes(previous)
        self.assertNotEqual(c.BASE, target)


def main():
    sources = m.hashes(m.source_files())
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    c.require(result.wasSuccessful(), "Matched type tests failed")
    c.require(sources == m.hashes(m.source_files()), "Source edited during tests")
    m.save(
        m.test_receipt_path(),
        dict(
            passed=True,
            tests=result.testsRun,
            synthetic_experiment_only=True,
            sources=sources,
            Qwen_calls=0,
            CUDA=False,
        ),
    )


if __name__ == "__main__":
    main()
