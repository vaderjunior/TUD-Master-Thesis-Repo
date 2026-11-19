from analysis.common import micro_f1_files as c
from analysis.common import micro_f1 as metric
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = c.ROOT
BASE = ROOT / "experiments/astra/micro_f1_sq2_supervised_20261113_v1"
SETUP = c.BASE
REG = None
MANIFEST = None
METRICS = ("macro", "micro", "micro_macro_labels")
CHECKS = defaultdict(int)
ERRORS = defaultdict(float)


def load_registry():
    global REG, MANIFEST
    if REG is None or MANIFEST is None:
        REG = c.read(SETUP / "comparison_registry_v2.json")
        MANIFEST = c.read(SETUP / "paired_contrast_manifest.json")


def equal(a, b, name, tol=1e-12):
    if a is None or b is None:
        c.require(a is None and b is None, name + " availability")
    else:
        a, b = (np.asarray(a, float), np.asarray(b, float))
        c.require(a.shape == b.shape, name + " shape")
        e = float(np.max(np.abs(a - b))) if a.size else 0.0
        c.require(np.isfinite(e) and e < tol, name + ": " + str(e))
        ERRORS[name] = max(ERRORS[name], e)
    CHECKS[name] += 1


def default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def save(path, value):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Outside new analysis folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            value, f, indent=2, ensure_ascii=False, allow_nan=False, default=default
        )
        f.write("\n")


def arrays(path):
    with np.load(c.bind(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def array(path, **v):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Outside output folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **v)


def score(counts, idx):
    return np.stack(
        [metric.macro(counts, idx), metric.micro(counts), metric.micro(counts, idx)], -1
    )


def ci(x, confidence=0.95):
    return np.quantile(x, [(1 - confidence) / 2, 1 - (1 - confidence) / 2], axis=0)


def interval(point, draws, family=None):
    return {
        name: dict(
            value=float(point[j]),
            ci95=ci(draws[:, j]).tolist(),
            adjusted_interval=(
                ci(draws[:, j], 1 - 0.05 / family).tolist()
                if family and name != "macro"
                else None
            ),
            adjusted_confidence=(
                1 - 0.05 / family if family and name != "macro" else None
            ),
        )
        for j, name in enumerate(METRICS)
    }


def counts(g, p):
    return metric.contributions(g, p, np.ones(p.shape[:2], bool))


def bootstrap(cube, clusters, idx, seed, *, sources=None, draws=10000):
    cube = np.asarray(cube)
    c.require(cube.ndim == 4 and cube.shape[1] == len(clusters), "Wrong count axes")
    truth = cube[..., 0] + cube[..., 2]
    pred = cube[..., 0] + cube[..., 1]
    c.require(
        np.isin(cube, [0, 1]).all()
        and np.isin(pred, [0, 1]).all()
        and np.isin(truth, [0, 1]).all(),
        "Nonbinary per-item counts",
    )
    c.require(
        np.all(truth == truth[0]), "Different gold or invalid answers in count cube"
    )
    unique, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    models, n, labels = cube.shape[:3]
    hashes = {}
    positions = []
    back = []
    for i in range(models):
        key = hashlib.sha256(pred[i].tobytes()).hexdigest()
        if key not in hashes:
            hashes[key] = len(positions)
            positions.append(i)
        else:
            c.require(
                np.array_equal(pred[i], pred[positions[hashes[key]]]),
                "Prediction hash collision",
            )
        back.append(hashes[key])
    nm = len(positions)
    terms = np.zeros((len(unique), 2 * nm * labels + labels), float)
    for i, j in enumerate(positions):
        for k in range(labels):
            terms[:, i * labels + k] = np.bincount(
                inverse, weights=cube[j, :, k, 0], minlength=len(unique)
            )
            terms[:, nm * labels + i * labels + k] = np.bincount(
                inverse, weights=pred[j, :, k], minlength=len(unique)
            )
    for k in range(labels):
        terms[:, 2 * nm * labels + k] = np.bincount(
            inverse, weights=truth[0, :, k], minlength=len(unique)
        )
    members = defaultdict(set)
    if sources is not None:
        for i, s in zip(inverse, sources):
            members[int(i)].add(s)
    strata = defaultdict(list)
    for i in range(len(unique)):
        strata[tuple(sorted(members[i])) if sources is not None else ("all",)].append(i)
    rng = np.random.default_rng(seed)
    out = np.empty((draws, models, 3), float)
    digest = hashlib.sha256()
    checks = 0
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        picks = np.concatenate(
            [
                rng.choice(group, (size, len(group)), replace=True)
                for _, group in sorted(strata.items())
            ],
            axis=1,
        )
        digest.update(picks.astype("<i8", copy=False).tobytes())
        weights = np.asarray(
            [np.bincount(x, minlength=len(unique)) for x in picks], float
        )
        total = weights @ terms
        c.require(
            np.array_equal(total[0], terms[picks[0]].sum(0)),
            "Direct expansion mismatch",
        )
        checks += 1
        tp = total[:, : nm * labels].reshape(size, nm, labels)
        pp = total[:, nm * labels : 2 * nm * labels].reshape(size, nm, labels)
        gg = total[:, 2 * nm * labels :][:, None, :]
        den = pp + gg
        lf = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
        dd = den.sum(-1)
        micro = np.divide(2 * tp.sum(-1), dd, out=np.zeros_like(dd), where=dd > 0)
        rr = den[..., idx].sum(-1)
        rest = np.divide(
            2 * tp[..., idx].sum(-1), rr, out=np.zeros_like(rr), where=rr > 0
        )
        z = np.stack([lf[..., idx].mean(-1), micro, rest], -1)
        zz = score(np.stack([tp, pp - tp, np.broadcast_to(gg, tp.shape) - tp], -1), idx)
        equal(z, zz, "independent bootstrap metric values")
        out[start : start + size] = z[:, back]
        if (start + size) % 2048 == 0 or start + size == draws:
            print(
                "Bootstrap",
                models,
                "views,",
                nm,
                "unique:",
                start + size,
                "/",
                draws,
                flush=True,
            )
    return (
        out,
        dict(
            draws=draws,
            seed=seed,
            clusters=len(unique),
            model_views=models,
            unique_prediction_matrices=nm,
            picks_sha256=digest.hexdigest(),
            direct_checks=checks,
            metric_values=draws * models * 3,
            source_stratified=sources is not None,
        ),
    )


def aggregate(models, values, variant=False):
    pools = {}
    budgets = {}
    members = {}
    for m in models:
        method = m["method"] + ("/" + m["variant"] if variant else "")
        key = (m["pool"], method)
        members.setdefault(key, [])
    for i, m in enumerate(models):
        members[
            m["pool"], m["method"] + ("/" + m["variant"] if variant else "")
        ].append(i)
    for key, ix in members.items():
        pools[key] = values[..., ix, :].mean(-2)
    for size in sorted({m["budget"] for m in models}):
        for method in sorted({k[1] for k in members}):
            pp = sorted(
                {
                    m["pool"]
                    for m in models
                    if m["budget"] == size
                    and m["method"] + ("/" + m["variant"] if variant else "") == method
                }
            )
            if pp:
                budgets[size, method] = np.mean([pools[p, method] for p in pp], 0)
    return (pools, budgets)


def freeze():
    load_registry()
    c.require(not BASE.exists(), "Preserve existing Part 3")
    BASE.mkdir()
    seal = c.read(SETUP / "complete.json")
    for p in (
        SETUP / "comparison_registry_v2.json",
        SETUP / "paired_contrast_manifest.json",
        SETUP / "scoring_contract.json",
    ):
        c.require(c.sha(p) == seal["artifact_sha256"][c.rel(p)], "Setup changed")
    studies = [s for s in REG["studies"] if 3 in s["parts"]]
    ids = {s["id"] for s in studies}
    save(
        BASE / "analysis_plan.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            studies=studies,
            units=[u["id"] for u in REG["units"] if u["study"] in ids],
            paired_questions=[r for r in MANIFEST["rows"] if r["family"] in ids],
            scope="All registered supervised points; original equal-budget and cutoff-control comparisons; separate five-fit sensitivity; curated tuning, shorter schedules, matched 67-label pools and native references. Finish the shared common-nine classifier display without altering Part 2.",
            estimation="Original TRAIN-selected configurations/cutoffs and gold masks. Per-fit scores, then equal pools. No refitting, inference, best-seed selection, or micro-based retuning.",
            uncertainty="Reproduce original paired cluster bootstrap streams and macro values before accepting micro intervals. 22 supplementary adjusted questions in five frozen families; other existing controls supporting95. Original macro families unchanged.",
            diagnostics="Label counts and weights, precision/recall, exact/gated answers, per-pool/seed variation and symmetric pool omission; existing default/selected and head/gate cutoff controls. Five-fit old/new paired effects, one-third contribution and separate full-budget sensitivity. Source-stratified checks retained for earlier studies with those original checks. Later cross-study robustness remains Part 8.",
            exclusions="No binary/severity micro, no changes to experiments, no model calls or new training. No exhaustive new subgroup search or micro-optimised model selection.",
        ),
    )
    save(BASE / "preservation_before.json", dict(sources=c.BINDINGS))
    print("Part 3 setup frozen", flush=True)


if __name__ == "__main__":
    freeze()
