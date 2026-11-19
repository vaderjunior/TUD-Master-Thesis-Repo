import unittest
import numpy as np
from training.frozen_features import roberta as r


class PoolingContracts(unittest.TestCase):
    def test_padding_is_excluded_and_normalization_is_per_text(self):
        hidden = np.array(
            [
                [[3.0, 0.0], [0.0, 4.0], [999.0, 999.0]],
                [[0.0, 2.0], [999.0, 999.0], [999.0, 999.0]],
            ]
        )
        mask = np.array([[1, 1, 0], [1, 0, 0]])
        out = r.pool_numpy(hidden, mask)
        self.assertTrue(np.allclose(out["mean"], [[0.6, 0.8], [0.0, 1.0]]))
        self.assertTrue(np.allclose(out["cls"], [[1.0, 0.0], [0.0, 1.0]]))

    def test_empty_or_zero_pooling_stops(self):
        with self.assertRaises(RuntimeError):
            r.pool_numpy(np.ones((1, 2, 3)), np.zeros((1, 2)))
        with self.assertRaises(RuntimeError):
            r.pool_numpy(np.zeros((1, 2, 3)), np.ones((1, 2)))


if __name__ == "__main__":
    unittest.main()
