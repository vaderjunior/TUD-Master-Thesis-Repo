import unittest
import uuid
import numpy as np
from analysis.common import runtime as c
from training.feedback import static as m


class LinearUpdate(unittest.TestCase):

    def cpu(self):
        return m.engine(c.LINEAR / "offline_check" / ("linear_" + uuid.uuid4().hex))

    def sample(self):
        rng = np.random.default_rng(912)
        x = rng.normal(size=(64, 8))
        y = (x[:, 0] + 0.2 * x[:, 1] > 0).astype(int)[:, None]
        return (x, y)

    def test_output_isolation(self):
        cpu = self.cpu()
        with self.assertRaises(RuntimeError):
            cpu.save(c.A15 / "forbidden.json", {})
        with self.assertRaises(RuntimeError):
            m.engine(c.A15)

    def test_original_recipe_and_independent_inference(self):
        cpu = self.cpu()
        x, y = self.sample()
        model = cpu.fit_linear(x, y, 10.0, 42)
        p = cpu.predict_linear(model, x)
        other = m.independent_prob(model, x)
        self.assertTrue(np.allclose(p, other, atol=1e-12, rtol=0))
        for cutoff in (0.5, 0.6):
            self.assertTrue(np.array_equal(p > cutoff, other > cutoff))

    def test_deterministic_replay(self):
        cpu = self.cpu()
        x, y = self.sample()
        a = cpu.fit_linear(x, y, 10.0, 42)
        b = cpu.fit_linear(x, y, 10.0, 42)
        self.assertTrue(all((np.array_equal(a[k], b[k]) for k in a)))

    def test_append_labels_changes_only_declared_fit(self):
        cpu = self.cpu()
        x, y = self.sample()
        original = x.copy()
        a = cpu.fit_linear(x, y, 10.0, 42)
        xx = np.concatenate([x, x[:4]])
        yy = np.concatenate([y, 1 - y[:4]])
        b = cpu.fit_linear(xx, yy, 10.0, 42)
        self.assertTrue(np.array_equal(x, original))
        self.assertFalse(np.array_equal(a["coef"], b["coef"]))
        self.assertEqual(b["coef"].shape, (1, 8))

    def test_constant_head(self):
        cpu = self.cpu()
        x, y = self.sample()
        model = cpu.fit_linear(x, np.ones_like(y), 10.0, 42)
        self.assertTrue(
            np.array_equal(cpu.predict_linear(model, x), m.independent_prob(model, x))
        )

    def test_strict_cutoff(self):
        cpu = self.cpu()
        p = np.array([[0.5], [0.6], [0.600001]])
        self.assertEqual(cpu.decisions("hate", p, 0.6)[:, 1].tolist(), [0, 0, 1])

    def test_atomic_fit_resume_and_reload(self):
        cpu = self.cpu()
        folder = cpu.BASE / "jobs/fixture"
        x, y = self.sample()
        calls = []

        def run(attempt):
            calls.append(1)
            model = cpu.fit_linear(x, y, 10.0, 42)
            cpu.arrays(attempt / "model.npz", model)
            cpu.array(attempt / "probabilities.npy", cpu.predict_linear(model, x))
            return {"synthetic_only": True}

        a = cpu.task(folder, {"fixture": 1}, run)
        b = cpu.task(folder, {"fixture": 1}, lambda _: self.fail("Completed fit reran"))
        self.assertEqual(a, b)
        self.assertEqual(len(calls), 1)
        restored = cpu.load_arrays(folder / a["attempt"] / "model.npz")
        self.assertTrue(
            np.allclose(
                cpu.predict_linear(restored, x),
                m.independent_prob(restored, x),
                rtol=0,
                atol=1e-12,
            )
        )

    def test_corruption_blocks_reuse(self):
        cpu = self.cpu()
        folder = cpu.BASE / "jobs/fixture"
        a = cpu.task(
            folder,
            {"fixture": 2},
            lambda p: cpu.save(p / "value.json", {"synthetic": True})
            or {"synthetic_only": True},
        )
        (folder / a["attempt"] / "value.json").write_text("{}", encoding="utf8")
        with self.assertRaises(RuntimeError):
            cpu.inspect_task(folder, {"fixture": 2})


if __name__ == "__main__":
    with c.offline():
        unittest.main()
