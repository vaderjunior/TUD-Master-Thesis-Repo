import argparse
import gc
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from training.cpu import common as c
from training.cpu import prepare as prep


def identity(plan_sha, **values):
    return dict(plan_sha256=plan_sha, **values)


def load_texts():
    return {r["id"]: r["text"] for r in c.rows(c.BASE / "data/train_texts.jsonl")}


def dense_rows(rows):
    matrix = np.load(
        c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    return np.asarray(matrix[[r["vector_index"] for r in rows]], dtype=np.float64)


def matrix_load(path, representation, part):
    if representation == "bge":
        return np.load(path / (part + ".npy"), allow_pickle=False)
    return sparse.load_npz(path / (part + ".npz"))


def features(directory, ident, representation, fit, held, texts):
    c.require(
        len(fit) and not {r["id"] for r in fit} & {r["id"] for r in held},
        "Invalid feature split",
    )
    c.require(
        not {c.cluster(texts[r["id"]]) for r in fit}
        & {c.cluster(texts[r["id"]]) for r in held},
        "Feature TRAIN/held text overlap",
    )
    ident = dict(
        ident,
        representation=representation,
        fit_hash=c.signature(fit),
        held_hash=c.signature(held),
    )

    def make(attempt):
        start = time.perf_counter()
        if representation == "bge":
            x = dense_rows(fit)
            v = dense_rows(held) if held else np.empty((0, 1024), dtype=np.float64)
            c.array(attempt / "fit.npy", x)
            c.array(attempt / "held.npy", v)
            widths = dict(bge=1024)
        else:
            x, state, idf = c.fit_text_features([texts[r["id"]] for r in fit])
            v = (
                c.transform_texts([texts[r["id"]] for r in held], state, idf)
                if held
                else sparse.csr_matrix((0, x.shape[1]))
            )
            c.sparse_save(attempt / "fit.npz", x)
            c.sparse_save(attempt / "held.npz", v)
            c.save(attempt / "vocabulary.json", state)
            c.arrays(attempt / "idf.npz", idf)
            widths = {k: len(v["vocabulary"]) for k, v in state.items()}
        detail = dict(
            fit_rows=len(fit),
            held_rows=len(held),
            dimensions=x.shape[1],
            feature_seconds=time.perf_counter() - start,
            widths=widths,
            resource_after=c.resources(),
        )
        c.save(
            attempt / "memberships.json",
            dict(fit_ids=[r["id"] for r in fit], held_ids=[r["id"] for r in held]),
        )
        return detail

    receipt = c.task(directory, ident, make)
    return c.task_path(directory, receipt)


def candidate(directory, ident, feature_path, representation, yy, strength, seed):
    ident = dict(
        ident,
        feature_receipt_sha256=c.sha(feature_path.parent.parent / "complete.json"),
        target_sha256=c.signature(yy.tolist()),
        C=strength,
        seed=seed,
    )

    def make(attempt):
        x = matrix_load(feature_path, representation, "fit")
        held = matrix_load(feature_path, representation, "held")
        c.resource_guard(1.0, 2.0)
        start = time.perf_counter()
        model = c.fit_linear(x, yy, strength, seed)
        seconds = time.perf_counter() - start
        c.arrays(attempt / "model.npz", model)
        restored = c.load_arrays(attempt / "model.npz")
        prob = c.predict_linear(restored, held)
        c.require(
            np.array_equal(prob, c.predict_linear(model, held)),
            "Saved coefficient round trip differs",
        )
        c.array(attempt / "held.npy", prob)
        return dict(
            C=strength,
            seed=seed,
            fit_rows=len(yy),
            feature_dimensions=x.shape[1],
            constants=model["constants"].tolist(),
            iterations=model["iterations"].tolist(),
            positive_support=yy.sum(axis=0).tolist(),
            fit_seconds=seconds,
            resource_after=c.resources(),
        )

    receipt = c.task(directory, ident, make)
    return c.task_path(directory, receipt)


def evaluate(directory, ident, feature_path, model_path, representation, evaluation):
    ident = dict(
        ident,
        feature_receipt_sha256=c.sha(feature_path.parent.parent / "complete.json"),
        model_receipt_sha256=c.sha(model_path.parent.parent / "complete.json"),
    )

    def make(attempt):
        model = c.load_arrays(model_path / "model.npz")
        parts = []
        zero_rows = 0
        if representation == "bge":

            cache = np.load(c.BASE / "data/evaluation_vectors.npy", allow_pickle=False)
        else:
            state = c.read(feature_path / "vocabulary.json")
            idf = c.load_arrays(feature_path / "idf.npz")
        start = time.perf_counter()
        for offset in range(0, len(evaluation), 256):
            if representation == "bge":
                x = np.asarray(cache[offset : offset + 256], dtype=np.float64)
            else:
                x = c.transform_texts(
                    [r["text"] for r in evaluation[offset : offset + 256]], state, idf
                )
                zero_rows += int(np.sum(x.getnnz(axis=1) == 0))
            prob = c.predict_linear(model, x)

            independent = np.column_stack(
                [
                    (
                        np.full(x.shape[0], value, dtype=float)
                        if value >= 0
                        else 1
                        / (
                            1
                            + np.exp(
                                -np.clip(
                                    np.asarray(x @ model["coef"][j]).reshape(-1)
                                    + model["intercept"][j],
                                    -700,
                                    700,
                                )
                            )
                        )
                    )
                    for j, value in enumerate(model["constants"])
                ]
            )
            c.require(
                np.allclose(prob, independent, rtol=0, atol=1e-12),
                "Independent inference differs",
            )
            for cutoff in c.CUTOFFS:
                c.require(
                    np.array_equal(prob > cutoff, independent > cutoff)
                    and np.array_equal(prob >= cutoff, independent >= cutoff),
                    "Floating-point difference changes a cutoff decision",
                )
            c.require(
                np.array_equal(prob.argmax(axis=1), independent.argmax(axis=1)),
                "Independent inference changes argmax",
            )
            parts.append(prob)
        result = np.concatenate(parts)
        c.require(
            result.shape[0] == len(evaluation), "Incomplete evaluation predictions"
        )
        c.array(attempt / "probabilities.npy", result)
        return dict(
            rows=len(evaluation),
            labels=result.shape[1],
            inference_seconds=time.perf_counter() - start,
            zero_feature_rows=zero_rows,
            numerical_reconstruction_checked=True,
            resource_after=c.resources(),
        )

    receipt = c.task(directory, ident, make)
    return c.task_path(directory, receipt)


def dependencies(directory):
    return {c.rel(p): c.sha(p) for p in sorted(Path(directory).rglob("complete.json"))}


def verify_dependencies(bindings):
    c.verify(bindings)
    for name in bindings:
        path = c.ROOT / name
        saved = c.read(path)
        c.inspect_task(path.parent, saved["identity"])


def group_path(stage, head, seed):
    return c.BASE / f"groups/{stage}_s{seed}/{head}"


def group_identity(plan_sha, stage, head, seed):
    return identity(plan_sha, kind="head_complete", stage=stage, head=head, seed=seed)


def verified_group(stage, head, seed, plan_sha):
    directory = group_path(stage, head, seed)
    receipt = c.inspect_task(
        directory / "result", group_identity(plan_sha, stage, head, seed)
    )
    if receipt is None:
        return None
    result = c.read(c.task_path(directory / "result", receipt) / "head.json")
    verify_dependencies(result["dependencies"])
    c.require(
        c.sha(directory / "selection.json") == result["selection_file_sha256"],
        "Head selection record changed",
    )
    if result["selected_oof"] is not None:
        c.require(
            c.sha(c.ROOT / result["selected_oof"]) == result["selected_oof_sha256"],
            "Selected OOF probabilities changed",
        )
    return result


def worker(stage, head, seed, plan_sha):
    done = verified_group(stage, head, seed, plan_sha)
    if done is not None:
        print("VERIFIED HEAD", stage, head, seed, flush=True)
        return
    c.current_cuda_guard()
    c.resource_guard()
    directory = group_path(stage, head, seed)
    texts = load_texts()
    evaluation = c.read(c.BASE / "data/evaluation.json")
    representation = "bge" if stage == "full_bge" else "tfidf"
    base_ident = identity(plan_sha, stage=stage, head=head, seed=seed)
    candidates = {}
    model_paths = {}
    gate = None
    if stage == "compact_tfidf":
        data = c.read(c.BASE / "data/compact.json")
        spec = data["heads"][head]
        rows = spec["rows"]
        yy = c.target_bits(head, spec)
        folds = np.asarray(data["folds"][str(seed)])
        assigned = folds[[r["feature_row"] for r in rows]]
        candidates = {strength: np.full(yy.shape, np.nan) for strength in c.CS}
        for fold in range(5):
            fit_index = np.flatnonzero(assigned != fold)
            held_index = np.flatnonzero(assigned == fold)
            c.require(
                len(fit_index), "Empty compact fit fold; stop without inventing labels"
            )
            if not len(held_index):
                continue
            fs = features(
                directory / f"features/fold{fold}",
                dict(base_ident, kind="features", fold=fold),
                representation,
                [rows[i] for i in fit_index],
                [rows[i] for i in held_index],
                texts,
            )
            for strength in c.CS:
                model = candidate(
                    directory / f"search/fold{fold}/C{strength:g}",
                    dict(base_ident, kind="candidate", fold=fold),
                    fs,
                    representation,
                    yy[fit_index],
                    strength,
                    seed,
                )
                candidates[strength][held_index] = np.load(
                    model / "held.npy", allow_pickle=False
                )
            print(
                f"{stage}/{head}/fold-setting{seed}: fold {fold + 1}/5 verified",
                flush=True,
            )
        if head != "hate":
            g = verified_group(stage, "hate", seed, plan_sha)
            c.require(g is not None, "Gate selection must precede fine-head selection")
            gp = np.load(c.ROOT / g["selected_oof"], allow_pickle=False)
            gate = c.decisions("hate", gp, g["selection"]["selected"]["threshold"])[
                :, 1
            ][[r["feature_row"] for r in rows]]
        selected = c.selection(head, yy, candidates, gate, compact=True)
        selected_oof = candidates[selected["selected"]["C"]]
        c.array(directory / "selected_oof.npy", selected_oof)
        fs = features(
            directory / "features/final",
            dict(base_ident, kind="features", fold="final"),
            representation,
            rows,
            [],
            texts,
        )
        strength = selected["selected"]["C"]
        final_model = candidate(
            directory / "final/compact",
            dict(base_ident, kind="final_compact"),
            fs,
            representation,
            yy,
            strength,
            seed,
        )
        variants = dict(compact=(fs, final_model))
    else:
        spec = c.read(c.BASE / "data/full_heads.json")[head]
        fit = spec["train"]
        held = spec["val"]
        yy = c.target_bits(head, spec, fit)
        yv = c.target_bits(head, spec, held)
        fs = features(
            directory / "features/matched",
            dict(base_ident, kind="features", partition="gradient_train"),
            representation,
            fit,
            held,
            texts,
        )
        for strength in c.CS:
            model = candidate(
                directory / f"search/C{strength:g}",
                dict(base_ident, kind="candidate"),
                fs,
                representation,
                yy,
                strength,
                seed,
            )
            candidates[strength] = np.load(model / "held.npy", allow_pickle=False)
            model_paths[strength] = model
            print(f"{stage}/{head}: C={strength:g} verified", flush=True)
        selected = c.selection(head, yv, candidates)
        strength = selected["selected"]["C"]
        variants = dict(matched_gradient=(fs, model_paths[strength]))
        all_rows = fit + held
        ya = c.target_bits(head, spec, all_rows)
        af = features(
            directory / "features/all_permitted",
            dict(base_ident, kind="features", partition="all_permitted"),
            representation,
            all_rows,
            [],
            texts,
        )
        am = candidate(
            directory / "final/all_permitted",
            dict(base_ident, kind="all_permitted_refit"),
            af,
            representation,
            ya,
            strength,
            seed,
        )
        variants["all_permitted"] = (af, am)

    c.save(directory / "selection.json", dict(identity=base_ident, selection=selected))
    output = {}
    for name, (feature_path, model_path) in variants.items():
        dest = evaluate(
            directory / f"inference/{name}",
            dict(base_ident, kind="inference", variant=name),
            feature_path,
            model_path,
            representation,
            evaluation,
        )
        output[name] = dict(
            probabilities=c.rel(dest / "probabilities.npy"),
            model=c.rel(model_path / "model.npz"),
            features=c.rel(feature_path),
            rows=len(evaluation),
        )
    deps = dependencies(directory)
    if stage == "compact_tfidf" and head != "hate":
        gate_dir = group_path(stage, "hate", seed) / "result"
        deps[c.rel(gate_dir / "complete.json")] = c.sha(gate_dir / "complete.json")

    def finish(attempt):
        result = dict(
            identity=base_ident,
            selection=selected,
            labels=spec["labels"],
            variants=output,
            selected_oof=(
                c.rel(directory / "selected_oof.npy")
                if stage == "compact_tfidf"
                else None
            ),
            selected_oof_sha256=(
                c.sha(directory / "selected_oof.npy")
                if stage == "compact_tfidf"
                else None
            ),
            dependencies=deps,
            selection_file_sha256=c.sha(directory / "selection.json"),
        )
        c.save(attempt / "head.json", result)
        return dict(
            head=head,
            stage=stage,
            seed=seed,
            variants=list(output),
            evaluation_scores_computed=False,
        )

    c.task(directory / "result", group_identity(plan_sha, stage, head, seed), finish)
    verified_group(stage, head, seed, plan_sha)
    print(
        "HEAD COMPLETE:",
        stage,
        head,
        seed,
        "(saved predictions; no evaluation scoring)",
        flush=True,
    )


def neural_cutoffs(plan_sha):
    full = c.read(c.BASE / "data/full_heads.json")
    for job in c.read(c.BASE / "data/neural.json"):
        head, seed = job["head"], job["seed"]
        directory = c.BASE / f"neural_cutoffs/{head}_s{seed}"
        ident = identity(plan_sha, kind="retained_neural_cutoff", **job)

        def make(attempt):
            spec = full[head]
            saved = list(c.rows(c.ROOT / job["logits"]))
            c.require(
                [str(r["id"]) for r in saved] == [r["source_id"] for r in spec["val"]],
                "Neural validation order differs",
            )
            c.require(
                [r["target"] for r in saved] == [r["target"] for r in spec["val"]],
                "Neural validation labels differ",
            )
            prob = c.neural_probabilities(head, [r["logits"] for r in saved])
            yy = c.target_bits(head, spec, spec["val"])
            selected = c.selection(head, yy, {1.0: prob})
            selected["C_is_placeholder_not_neural_hyperparameter"] = True
            c.save(
                attempt / "selection.json",
                dict(
                    job=job,
                    selection=selected,
                    policy="Head-native global cutoff only. This does not yet apply cutoffs to evaluation logits.",
                ),
            )
            c.array(attempt / "validation_probabilities.npy", prob)
            return dict(rows=len(saved), head=head, seed=seed, new_neural_inference=0)

        c.task(directory, ident, make)
    print(
        "Saved 12 TRAIN-internal neural cutoff selections; no new neural inference.",
        flush=True,
    )


def technical_pilot(plan_sha):
    directory = c.BASE / "runtime_pilot"
    ident = identity(plan_sha, kind="TRAIN_only_technical_pilot")

    def make(attempt):
        full = c.read(c.BASE / "data/full_heads.json")
        texts = load_texts()
        ordered = sorted(full["hate"]["train"], key=lambda r: c.text_sha(r["id"]))[:512]
        fit, held = ordered[:416], ordered[416:]
        yy = c.target_bits("hate", full["hate"], fit)
        detail = {}
        c.require(
            len(np.unique(yy)) == 2,
            "Technical pilot lacks both classes; review without redrawing",
        )
        for representation in ("bge", "tfidf"):
            start = time.perf_counter()
            fp = features(
                directory / representation / "features",
                dict(ident, kind="pilot_features"),
                representation,
                fit,
                held,
                texts,
            )
            cp = candidate(
                directory / representation / "candidate",
                dict(ident, kind="pilot_candidate"),
                fp,
                representation,
                yy,
                1.0,
                42,
            )
            prob = np.load(cp / "held.npy", allow_pickle=False)
            c.require(prob.shape == (96, 1), "Technical pilot predictions incomplete")
            detail[representation] = dict(
                seconds=time.perf_counter() - start,
                resource=c.resources(),
                fitted_texts=len(fit),
                held_technical_texts=len(held),
            )
        c.save(
            attempt / "pilot.json",
            dict(
                status="technical_checks_passed",
                results=detail,
                no_F1_computed=True,
                excluded_from_scientific_results=True,
                full_data_runtime_and_peak_not_estimated=True,
            ),
        )
        return dict(pilot_only=True, dependencies=dependencies(directory))

    receipt = c.task(directory / "result", ident, make)
    verify_dependencies(receipt["detail"]["dependencies"])
    print(
        "TRAIN-only technical pilot passed. Full-data runtime remains to be measured.",
        flush=True,
    )


def group_roster():
    return [
        (stage, head, seed)
        for stage, seeds in [
            ("full_bge", [42]),
            ("full_tfidf", [42]),
            ("compact_tfidf", c.SEEDS),
        ]
        for seed in seeds
        for head in c.HEADS
    ]


def assemble(plan_sha):
    for stage, seeds, variants in [
        ("full_bge", [42], ["matched_gradient", "all_permitted"]),
        ("full_tfidf", [42], ["matched_gradient", "all_permitted"]),
        ("compact_tfidf", c.SEEDS, ["compact"]),
    ]:
        for seed in seeds:
            heads = {h: verified_group(stage, h, seed, plan_sha) for h in c.HEADS}
            c.require(all(heads.values()), "Cannot assemble an incomplete pipeline")
            for variant in variants:
                directory = c.BASE / f"pipelines/{stage}_s{seed}_{variant}"
                ident = identity(
                    plan_sha,
                    kind="pipeline",
                    stage=stage,
                    seed=seed,
                    variant=variant,
                    head_receipts={
                        h: c.sha(group_path(stage, h, seed) / "result/complete.json")
                        for h in c.HEADS
                    },
                )

                def make(attempt):
                    probs = {
                        h: np.load(
                            c.ROOT / heads[h]["variants"][variant]["probabilities"],
                            allow_pickle=False,
                        )
                        for h in c.HEADS
                    }
                    out = {}
                    for policy in ("default", "selected"):
                        cutoffs = {
                            h: (
                                (None if h == "severity" else 0.5)
                                if policy == "default"
                                else heads[h]["selection"]["selected"]["threshold"]
                            )
                            for h in c.HEADS
                        }
                        gate = c.decisions("hate", probs["hate"], cutoffs["hate"])
                        out[policy + "_hate"] = gate
                        for h in c.HEADS[1:]:
                            raw = c.decisions(h, probs[h], cutoffs[h])
                            delivered = raw.copy()
                            delivered[gate[:, 1] == 0] = 0
                            out[policy + "_" + h + "_ungated"] = raw
                            out[policy + "_" + h] = delivered
                    c.arrays(attempt / "decisions.npz", out)
                    c.save(
                        attempt / "contract.json",
                        dict(
                            identity=ident,
                            labels={h: heads[h]["labels"] for h in c.HEADS},
                            evaluation_sha256=c.sha(c.BASE / "data/evaluation.json"),
                            selected_cutoffs={
                                h: heads[h]["selection"]["selected"]["threshold"]
                                for h in c.HEADS
                            },
                            default_cutoffs="0.5 gate/fine; severity argmax; same selected weights as selected policy",
                            probabilities={
                                h: heads[h]["variants"][variant]["probabilities"]
                                for h in c.HEADS
                            },
                        ),
                    )
                    return dict(
                        rows=len(probs["hate"]),
                        policies=["default", "selected"],
                        evaluation_gold_read=False,
                    )

                c.task(directory, ident, make)


def readiness():
    plan = prep.validate()
    ready = c.read(c.BASE / "readiness.json")
    c.require(
        ready["status"] == "code_data_tests_ready_runtime_pilot_required"
        and ready["plan_sha256"] == c.sha(c.BASE / "plan.json"),
        "Readiness seal differs",
    )
    c.require(
        ready["tests_sha256"] == c.sha(c.BASE / "qa/tests.json"),
        "Test evidence changed",
    )
    return plan, c.sha(c.BASE / "plan.json")


def verify_complete(plan_sha):
    done = c.read(c.BASE / "complete.json")
    c.require(
        done["plan_sha256"] == plan_sha
        and done["status"] == "CPU_fitting_predictions_complete_review_pending",
        "Completion identity differs",
    )
    verify_dependencies(done["outputs"])
    for stage, head, seed in group_roster():
        c.require(
            verified_group(stage, head, seed, plan_sha) is not None,
            "Completed head missing",
        )
    return done


def run():
    _, plan_sha = readiness()
    with c.file_lock(c.BASE / "launcher.lock"):
        if (c.BASE / "complete.json").exists():
            verify_complete(plan_sha)
            print("VERIFIED COMPLETE. No new fits.", flush=True)
            return
        c.current_cuda_guard()
        snapshot = c.resource_guard()
        started = c.BASE / "execution_started.json"
        if not started.exists():
            c.save(
                started,
                dict(
                    plan_sha256=plan_sha, utc=c.utc(), resources=snapshot, Qwen_calls=0
                ),
            )

        def child(*args):
            c.current_cuda_guard()
            c.resource_guard()
            subprocess.run(
                [sys.executable, "-X", "utf8", "-B", "-u", "-m", __spec__.name, *args],
                cwd=c.ROOT,
                check=True,
            )

        child("pilot")
        child("neural")
        for stage, head, seed in group_roster():
            if verified_group(stage, head, seed, plan_sha) is not None:
                print("Verified saved head", stage, head, seed, flush=True)
                continue
            print("Starting", stage, head, seed, flush=True)
            child("worker", "--stage", stage, "--head", head, "--seed", str(seed))
        child("assemble")
        outputs = {
            c.rel(group_path(stage, head, seed) / "result/complete.json"): c.sha(
                group_path(stage, head, seed) / "result/complete.json"
            )
            for stage, head, seed in group_roster()
        }
        for folder in ("pipelines", "neural_cutoffs"):
            outputs.update(
                {c.rel(p): c.sha(p) for p in (c.BASE / folder).glob("*/complete.json")}
            )
        path = c.BASE / "runtime_pilot/result/complete.json"
        outputs[c.rel(path)] = c.sha(path)
        prep.validate()
        verify_dependencies(outputs)
        c.save(
            c.BASE / "complete.json",
            dict(
                status="CPU_fitting_predictions_complete_review_pending",
                utc=c.utc(),
                plan_sha256=plan_sha,
                outputs=outputs,
                Qwen_calls=0,
                CUDA_used=False,
                scientific_review_required=True,
                remaining="Apply neural cutoff selections to saved evaluation logits, independent scoring/paired analysis and scientific review.",
            ),
        )
        verify_complete(plan_sha)
        print(
            "CPU BASELINES COMPLETE. Tell Astra to inspect. Scientific review and neural-cutoff evaluation remain.",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "mode",
        choices=["run", "preflight", "pilot", "neural", "worker", "assemble", "verify"],
    )
    ap.add_argument("--stage", choices=["full_bge", "full_tfidf", "compact_tfidf"])
    ap.add_argument("--head", choices=c.HEADS)
    ap.add_argument("--seed", type=int, choices=c.SEEDS)
    args = ap.parse_args()
    if args.mode == "run":
        run()
        return
    _, plan_sha = readiness()
    if args.mode == "preflight":
        c.current_cuda_guard()
        print(c.resource_guard(), flush=True)
        return
    if args.mode == "verify":
        verify_complete(plan_sha)
        print("All saved CPU outputs verify.", flush=True)
        return
    c.current_cuda_guard()
    c.resource_guard()
    if args.mode == "pilot":
        technical_pilot(plan_sha)
    elif args.mode == "neural":
        neural_cutoffs(plan_sha)
    elif args.mode == "assemble":
        assemble(plan_sha)
    else:
        c.require(
            args.stage and args.head and args.seed is not None,
            "Worker stage/head/seed required",
        )
        c.require(
            args.stage == "compact_tfidf" or args.seed == 42,
            "Full-data seed must remain 42",
        )
        worker(args.stage, args.head, args.seed, plan_sha)


if __name__ == "__main__":
    with c.offline(), threadpool_limits(limits=1):
        main()
