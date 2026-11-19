from pathlib import Path
import uuid
import numpy as np
from scipy.special import expit
from training.text_budgets.extended import common as b
from training.frozen_features import fit as linear
from training.text_budgets.extended import audit as audit

c = b.c


def check_heads(data, heads, dimension):
    from sklearn.metrics import f1_score

    gate = None
    checked = 0
    for head in c.HEADS:
        spec = data["heads"][head]
        saved = heads[head]
        truth = c.truth(head, spec["rows"], spec["labels"])
        supported = np.flatnonzero(
            (truth.sum(0) >= 2) & ((1 - truth).sum(0) >= 2)
        ).tolist()
        c.require(
            saved["labels"] == spec["labels"]
            and saved["training_ids"] == [r["id"] for r in spec["rows"]]
            and saved["selection_indices"] == supported
            and not saved["untrained"]
            and not saved["selection_fallback"],
            "Linear TRAIN support/order differs",
        )
        candidates = saved["candidates"]
        expected = {
            (strength, cutoff)
            for strength in linear.CS
            for cutoff in ((None,) if head == "severity" else linear.THRESHOLDS)
        }
        c.require(
            len(candidates) == len(expected)
            and {(v["C"], v["threshold"]) for v in candidates} == expected,
            "Incomplete linear candidate set",
        )
        ranked = sorted(
            candidates,
            key=lambda v: (
                -v["score"],
                v["C"],
                abs((0.5 if v["threshold"] is None else v["threshold"]) - 0.5),
                -(v["threshold"] or 0),
            ),
        )
        c.require(saved["selected"] == ranked[0], "Linear tie ordering differs")
        probs = np.asarray(saved["oof_probabilities"], dtype=float)
        width = 1 if head == "hate" else len(spec["labels"])
        c.require(
            probs.shape == (len(spec["rows"]), width)
            and np.isfinite(probs).all()
            and ((probs >= 0) & (probs <= 1)).all(),
            "Linear OOF probabilities invalid",
        )
        if head == "hate":
            p = probs[:, 0] > saved["selected"]["threshold"]
            pred = np.column_stack((~p, p))
            gate = p
        elif head == "severity":
            pred = np.eye(width, dtype=bool)[probs.argmax(1)]
        else:
            pred = probs >= saved["selected"]["threshold"]
        if head != "hate":
            pred[~gate[[r["feature_row"] for r in spec["rows"]]]] = 0
        score = np.mean(
            [f1_score(truth[:, j], pred[:, j], zero_division=0) for j in supported]
        )
        c.require(
            abs(score - saved["selected"]["score"]) < 1e-12,
            "Linear selected OOF score differs",
        )
        c.require(len(saved["models"]) == width, "Linear head width differs")
        for model in saved["models"]:
            if "constant" in model:
                c.require(
                    model == {"constant": 0} or model == {"constant": 1},
                    "Invalid constant classifier",
                )
            else:
                c.require(
                    len(model["coef"]) == dimension
                    and np.isfinite(model["coef"]).all()
                    and np.isfinite(model["intercept"])
                    and model["iterations"] <= 2000,
                    "Invalid fitted coefficient",
                )
        checked += 1
    return checked


def checked_probabilities(heads, x):
    probabilities = linear.probabilities(heads, x)
    for head in c.HEADS:
        independent = np.column_stack(
            [
                (
                    np.full(len(x), m["constant"], dtype=float)
                    if "constant" in m
                    else expit(
                        np.asarray(x, dtype=float) @ np.asarray(m["coef"], dtype=float)
                        + m["intercept"]
                    )
                )
                for m in heads[head]["models"]
            ]
        )
        c.require(
            np.allclose(independent, probabilities[head], rtol=0, atol=1e-12),
            "Independent serialized coefficient prediction differs",
        )
    return probabilities


def fit(pool, plan_sha):
    data = b.data(pool)
    root = c.BASE / "linear" / pool
    path = root / "fit_complete.json"
    feature_proof = c.read(c.BASE / "features/complete.json")
    c.verify(feature_proof["outputs"])
    identity = dict(
        pool=pool,
        plan_sha256=plan_sha,
        fold_assignment=42,
        TRAIN_sha256=c.sha(c.BASE / "data/train" / f"{pool}.json"),
        features_receipt_sha256=c.sha(c.BASE / "features/complete.json"),
    )
    if path.exists():
        proof = c.read(path)
        c.require(proof["identity"] == identity, "Linear fit identity differs")
        c.verify(proof["outputs"])
        saved = c.read(b.ROOT / proof["fit_path"])
        check_heads(data, saved["heads"], 768)
        with np.load(b.ROOT / proof["oof_path"], allow_pickle=False) as z:
            oof = {
                h: {
                    str(strength): z[h + "__" + str(strength)] for strength in linear.CS
                }
                for h in c.HEADS
            }
            linear.check_selection(data, 42, saved["heads"], oof)
        return saved["heads"], proof
    rows = c.read(c.BASE / "data/TRAIN_features.json")
    position = {v["key"][6:]: i for i, v in enumerate(rows)}
    x = np.load(c.BASE / "features/TRAIN_mean.npy", mmap_mode="r", allow_pickle=False)
    train = np.asarray(x[[position[k] for k in data["gate_ids"]]], dtype=float)
    directory = root / "attempts" / uuid.uuid4().hex[:12]
    b.save(directory / "started.json", identity)
    cached, counts = linear.make_oof(data, train, 42)
    heads = linear.fit_recipe(data, train, 42, cached)
    checks = linear.check_selection(data, 42, heads, cached)
    check_heads(data, heads, 768)
    fit_path = directory / "fit.json"
    oof_path = directory / "oof.npz"
    b.save(fit_path, dict(identity=identity, heads=heads, checks=checks, counts=counts))
    b.arrays(
        oof_path,
        {
            h + "__" + str(k): v
            for h, values in cached.items()
            for k, v in values.items()
        },
    )
    proof = dict(
        identity=identity,
        fit_path=c.rel(fit_path),
        oof_path=c.rel(oof_path),
        outputs=b.hashes([fit_path, oof_path]),
    )
    b.save(path, proof)
    return heads, proof


def reuse_bge(pool, seed, plan_sha):
    directory = b.POOLS / "jobs" / f"{pool}_fold{seed}"
    proof = c.read(directory / "complete.json")
    c.require(
        proof["plan_sha256"] == c.sha(b.POOLS / "plan.json")
        and proof["job"] == f"{pool}_fold{seed}",
        "Historical BGE origin differs",
    )
    c.verify(proof["outputs"])
    fit = c.read(directory / "fit.json")
    c.require(
        fit["plan_sha256"] == proof["plan_sha256"]
        and fit["job"]["pool"] == pool
        and fit["job"]["seed"] == seed,
        "Historical BGE fit identity differs",
    )
    c.require(
        fit["checks"]["candidate_score_checks"] == 112,
        "Historical BGE all-candidate audit missing",
    )
    data = b.data(pool)
    check_heads(data, fit["heads"], 1024)
    record = dict(
        plan_sha256=plan_sha,
        pool=pool,
        fold_assignment=seed,
        historical_all_candidate_checks=112,
        selected_OOF_and_coefficients_rechecked=True,
        all_candidate_OOF_recomputed=False,
        note="Historical all-candidate scores are provenance-verified, not falsely claimed as newly recomputed.",
        outputs={
            **b.hashes([directory / "complete.json", directory / "fit.json"]),
            **proof["outputs"],
        },
    )
    b.save(c.BASE / "linear" / pool / f"bge_fold{seed}_reuse.json", record)
    return fit["heads"], record


def predict(pool, method, seed, heads, source, plan_sha):
    data = b.data(pool)
    kind = "bge" if method == "bge" else "roberta"
    with np.load(c.BASE / "data/evaluation_features.npz", allow_pickle=False) as z:
        x = z[kind]
    check_heads(data, heads, x.shape[1])
    prob = checked_probabilities(heads, x)

    gate = prob["hate"][:, 0]
    canonical = {**prob, "hate": np.column_stack((1 - gate, gate))}
    directory = c.BASE / "predictions" / pool
    name = f"{method}_fold{seed}"
    outputs = {}
    pp = directory / f"{name}_probabilities.npz"
    b.arrays(pp, prob)
    outputs.update(b.hashes([pp]))
    for cutoff in ("selected", "default"):
        value = audit.decoded(canonical, heads, cutoff)

        g = None
        for head in c.HEADS:
            t = (
                None
                if head == "severity"
                else (
                    0.5 if cutoff == "default" else heads[head]["selected"]["threshold"]
                )
            )
            expected = linear.old.decisions(head, prob[head], t, g)
            c.require(
                np.array_equal(expected, value[head]),
                "Linear/neural decoding conventions differ",
            )
            if head == "hate":
                g = expected[:, 1].astype(bool)
        path = directory / f"{name}_{cutoff}.npz"
        b.arrays(path, value)
        outputs.update(b.hashes([path]))
    result = dict(
        plan_sha256=plan_sha,
        pool=pool,
        method=method,
        fold_assignment=seed,
        rows=len(x),
        source_receipt=source,
        outputs=outputs,
        evaluation_gold_used=False,
    )
    b.save(c.BASE / "linear" / pool / f"{name}_predictions.json", result)
    return result


def run(plan_sha):
    outputs = {}
    for pool in b.NAMES:
        b.deadline_guard()
        b.resource.resource_guard(3.5, 12)
        heads, proof = fit(pool, plan_sha)
        r = predict(
            pool,
            "frozen_roberta",
            42,
            heads,
            c.rel(c.BASE / "linear" / pool / "fit_complete.json"),
            plan_sha,
        )
        outputs.update(r["outputs"])
        outputs.update(proof["outputs"])
        for seed in (42, 43, 44):
            heads, proof = reuse_bge(pool, seed, plan_sha)
            r = predict(
                pool,
                "bge",
                seed,
                heads,
                c.rel(c.BASE / "linear" / pool / f"bge_fold{seed}_reuse.json"),
                plan_sha,
            )
            outputs.update(r["outputs"])
            outputs.update(proof["outputs"])
        print(
            "Matched frozen-representation fits/predictions verified:", pool, flush=True
        )
    outputs.update(
        b.hashes(
            p for p in (c.BASE / "linear").rglob("*.json") if p.name != "complete.json"
        )
    )
    result = dict(
        status="linear_fits_and_text_only_predictions_complete",
        plan_sha256=plan_sha,
        outputs=outputs,
        new_frozen_roberta_fits=9,
        reused_BGE_fits=27,
        scientific_scores_computed=False,
    )
    b.save(c.BASE / "linear/complete.json", result)
    return result
