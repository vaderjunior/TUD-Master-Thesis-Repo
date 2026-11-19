import numpy as np
from pathlib import Path
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import reader as r
from analysis.knowledge.policy import metrics as m


def run():
    rng = np.random.default_rng(902)
    gold = rng.integers(0, 2, (8, 3)).astype(bool)
    pred = rng.integers(0, 2, (3, 3, 8, 3)).astype(bool)
    state = pred.astype(int) @ np.array([1, 2, 4])
    exact = (pred == gold).all(axis=-1)
    count = m.contributions(gold, pred).transpose(2, 0, 1, 3, 4).reshape(8, -1)
    agree = np.array(
        [
            [
                sum((state[ci, a, i] == state[ci, b, i] for a, b in m.PAIRS)) / 3
                for ci in range(3)
            ]
            for i in range(8)
        ]
    )
    cluster_rows = [[0, 1], [2], [3], [4, 5], [6], [7]]
    strata = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    features = np.array(
        [
            np.r_[count[idx].sum(axis=0), agree[idx].sum(axis=0), len(idx)]
            for idx in cluster_rows
        ]
    )
    draws, seed = (41, 7)
    actual = m.bootstrap_from_features(features, strata, 3, draws, seed, chunk=draws)
    chosen = np.zeros((draws, 6), int)
    rng = np.random.default_rng(seed)
    for ids in strata:
        chosen[:, ids] = rng.multinomial(
            len(ids), np.full(len(ids), 1 / len(ids)), size=draws
        )
    checks = 0
    for draw, row in enumerate(chosen):
        indices = [
            i
            for cluster, copies in enumerate(row)
            for _ in range(int(copies))
            for i in cluster_rows[cluster]
        ]
        ma = []
        mi = []
        pa = []
        for ci in range(3):
            macro = []
            micro = []
            for repeat in range(3):
                label_counts = []
                for label in range(3):
                    tp = fp = fn = 0
                    for i in indices:
                        yy = bool(gold[i, label])
                        pp = bool(pred[ci, repeat, i, label])
                        tp += yy and pp
                        fp += not yy and pp
                        fn += yy and (not pp)
                    label_counts.append((tp, fp, fn))
                scores = [
                    2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
                    for tp, fp, fn in label_counts
                ]
                tp, fp, fn = (sum((q[j] for q in label_counts)) for j in range(3))
                macro.append(sum(scores) / 3)
                micro.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
            ma.append(sum(macro) / 3)
            mi.append(sum(micro) / 3)
            pa.append(
                sum(
                    (
                        sum((state[ci, a, i] == state[ci, b, i] for a, b in m.PAIRS))
                        for i in indices
                    )
                )
                / (3 * len(indices))
            )
        for name, expected in [
            ("macro_f1", ma),
            ("micro_f1_all_labels", mi),
            ("mean_pairwise_repeat_agreement", pa),
        ]:
            c.require(
                np.allclose(actual[name][draw], expected, atol=1e-12, rtol=0),
                "Explicit cluster draw differs",
            )
            checks += 3
    c.require(
        all((np.ptp(a, axis=0).max() > 0 for a in actual.values())),
        "Synthetic draws are degenerate",
    )
    for left, right in m.CONTRASTS:
        fixes = (exact[left] & ~exact[right]).sum(axis=1)
        breaks = (~exact[left] & exact[right]).sum(axis=1)
        c.require(
            np.array_equal(
                fixes - breaks, exact[left].sum(axis=1) - exact[right].sum(axis=1)
            ),
            "Paired transitions differ",
        )
    r.save(
        r.BASE / "extra_checks.json",
        dict(
            status="passed",
            explicit_bootstrap_values=checks,
            description="41 variable-size paired cluster draws over two source strata; nine settings share each sampled text; scalar TP/FP/FN/F1 and pairwise agreement independently reproduced.",
            metric_source_sha256=c.sha(Path(m.__file__)),
            test_source_sha256=c.sha(Path(__file__)),
        ),
    )
    print("Additional explicit bootstrap values checked:", checks, flush=True)


if __name__ == "__main__":
    with c.offline():
        run()
