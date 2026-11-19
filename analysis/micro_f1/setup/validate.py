from analysis.common import micro_f1_files as c
from analysis.common import micro_f1 as m
from analysis.common import micro_f1_reader as r
import numpy as np
import sklearn
from sklearn.metrics import f1_score, multilabel_confusion_matrix


def main():
    rng = np.random.default_rng(2026111301)
    checks = 0
    maxerror = 0.0
    for labels in (3, 7):
        for iteration in range(100):
            n = int(rng.integers(2, 60))
            gold = rng.integers(0, 2, (n, labels))
            pred = rng.integers(0, 2, (n, labels))
            mask = rng.random(n) > 0.2
            if not mask.any():
                mask[0] = True
            own = r.counts(gold, pred, mask)
            independent = multilabel_confusion_matrix(gold[mask], pred[mask])
            expected = np.stack(
                [independent[:, 1, 1], independent[:, 0, 1], independent[:, 1, 0]],
                axis=-1,
            )
            c.require(np.array_equal(own, expected), "Independent confusion mismatch")
            checks += 1
            for indices in (list(range(labels)), list(range(labels - 1))):
                for mode in ("macro", "micro"):
                    actual = getattr(m, mode)(own, indices)
                    wanted = f1_score(
                        gold[mask],
                        pred[mask],
                        labels=indices,
                        average=mode,
                        zero_division=0,
                    )
                    error = abs(float(actual) - float(wanted))
                    maxerror = max(maxerror, error)
                    c.require(error < 1e-12, "Independent F1 mismatch")
                    checks += 1
    sample = [
        dict(gate=True, types=["a"]),
        dict(gate=False, types=["a"]),
        dict(gate=True, types=None),
        dict(gate=True, types=[]),
    ]
    eligible = [
        i for i, x in enumerate(sample) if x["gate"] is True and x["types"] is not None
    ]
    c.require(eligible == [0, 3], "Unknown annotation handling")
    checks += 1
    predicted = np.array([[0, 0], [0, 1]])
    z = r.counts([[1, 0], [0, 0]], predicted)
    c.require(z.tolist() == [[0, 0, 1], [0, 1, 0]], "Gate/empty gold distinction")
    checks += 1
    try:
        c.blocked()
    except RuntimeError:
        checks += 1
    else:
        raise ValueError("Offline guard did not block")
    c.save(
        "independent_synthetic_check.json",
        dict(
            status="passed",
            synthetic_only=True,
            checks=checks,
            sklearn_version=sklearn.__version__,
            maximum_absolute_error=maxerror,
            no_real_micro_scored=True,
            network_guard_checked=True,
        ),
    )
    print(
        "Independent synthetic checks passed", checks, "max error", maxerror, flush=True
    )


if __name__ == "__main__":
    main()
