"""Check the corrected ordered-cluster reader with synthetic counts."""

import hashlib
import numpy as np
from analysis.knowledge.micro_f1 import components_guidelines as reader


def test_ordered_cluster_rows_replay():
    gold = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], bool)
    pred = np.stack([np.roll(gold, i % 4, axis=0) for i in range(9)])
    rows = [[0, 1], [2], [3]]
    rng = np.random.default_rng(55)
    stream = hashlib.sha256()
    raw, expected = [], []
    for start in range(0, 8, 3):
        picks = rng.choice([0, 1, 2], (min(3, 8 - start), 3), replace=True)
        stream.update(picks.astype("<i4").tobytes())
        for pick in picks:
            selected = [r for cluster in pick for r in rows[cluster]]
            settings, scores = [], []
            for setting in pred:
                counts = []
                for label in range(3):
                    y, x = gold[selected, label], setting[selected, label]
                    counts.append(
                        [
                            int((~y & ~x).sum()),
                            int((~y & x).sum()),
                            int((y & ~x).sum()),
                            int((y & x).sum()),
                        ]
                    )
                a = np.array(counts)
                denominator = 2 * a[:, 3] + a[:, 1] + a[:, 2]
                f1 = np.divide(2 * a[:, 3], denominator, out=np.zeros(3), where=denominator != 0)
                total = denominator.sum()
                micro = 2 * a[:, 3].sum() / total if total else 0
                settings.append(counts)
                scores.append([f1.mean(), micro, micro])
            raw.append(settings)
            expected.append(scores)
    design = {
        "clusters": ["first", "second", "third"],
        "cluster_rows": rows,
        "seed": 55,
        "draws": 8,
        "block_size": 3,
        "strata": [{"cluster_indices": [0, 1, 2]}],
        "pick_stream_sha256": stream.hexdigest(),
    }
    result = reader.replay_german(
        {
            "original_arrays": {"bootstrap_confusions": np.array(raw)},
            "indices": [0, 1, 2],
            "gold": gold,
            "pred": pred,
            "valid": np.ones((9, 4), bool),
            "mask": np.ones(4, bool),
            "design": design,
        }
    )
    np.testing.assert_allclose(result["scores"], expected, atol=1e-12, rtol=0)
    assert result["checks"]["direct_expansions"] == 3
    assert result["checks"]["pick_stream_sha256"] == stream.hexdigest()
