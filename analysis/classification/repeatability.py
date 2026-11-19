"Measure agreement between saved outer repeats."

from __future__ import annotations
from src.hsrag.paths import PROJECT_ROOT

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def blocked(*args, **kwargs):
    raise RuntimeError("This analysis is offline; network access is forbidden")


import numpy as np
from analysis.common.study import offline

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq2_consistency_review_20261113_v1"
READER = ROOT / "experiments/astra/sq2_broad_reader_review_20261104_v1"
MAIN = ROOT / "experiments/astra/sq2_broad_main_review_20261104_v1"
ROBUST = ROOT / "experiments/astra/sq2_broad_robustness_review_20261104_v1"

CONDITIONS = ("rag_full", "rag_small", "zero_shot")
SETTINGS = [(c, r) for c in CONDITIONS for r in range(3)]
PAIRS = ((0, 1), (0, 2), (1, 2))
CONTRASTS = ((0, 2), (1, 2), (0, 1))
DRAWS = 50000
SEED = 20261113


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    path = Path(path)
    require(
        path.resolve().is_relative_to(BASE.resolve()),
        "New output outside analysis folder",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=serial,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as f:
        f.write(raw)


def verify(bindings):
    for name, expected in bindings.items():
        require(sha(ROOT / name) == expected, "Changed evidence: " + name)


def slug(name):
    return name.replace("/", "__")


@offline()
def freeze():
    require(not (BASE / "analysis_contract.json").exists(), "Contract already frozen")
    epmap = read(READER / "endpoint_setup.json")
    required = {}

    def bind(path, expected=None):
        actual = sha(path)
        require(
            expected is None or actual == expected,
            "Original seal mismatch: " + str(path),
        )
        required[rel(path)] = actual

    assembly = read(READER / "assembly.json")
    setup = read(READER / "setup_complete.json")
    part1 = read(READER / "part1_complete.json")
    require(
        part1["status"] == "part1_reader_and_comparison_setup_independently_verified",
        "Reader not sealed",
    )
    for path in (READER / "assembly.json", READER / "setup_complete.json"):
        bind(path, part1["outputs"][rel(path)])
    bind(READER / "part1_complete.json")
    bind(
        READER / "predictions.jsonl",
        assembly["outputs"][rel(READER / "predictions.jsonl")],
    )
    bind(
        READER / "endpoint_setup.json",
        setup["outputs"][rel(READER / "endpoint_setup.json")],
    )
    bind(
        READER / "population_setup.json",
        setup["outputs"][rel(READER / "population_setup.json")],
    )
    meta = read(ROBUST / "metadata_complete.json")
    diag = read(ROBUST / "diagnostics_complete.json")
    for path in (
        ROBUST / "metadata_complete.json",
        ROBUST / "diagnostics_complete.json",
        ROBUST / "independent_metadata_verification.json",
    ):
        bind(path)
    for filename in (
        "metadata_index.json",
        "prediction_metadata.npz",
        "validity_inventory.json",
    ):
        path = ROBUST / filename
        bind(path, meta["outputs"][rel(path)])
    path = ROBUST / "repeat_and_inner_diagnostics.json"
    bind(path, diag["outputs"][rel(path)])
    scores = read(MAIN / "scores_complete.json")
    completion = read(MAIN / "part2_complete.json")
    require(
        completion["status"] == "part2_main_comparisons_independently_verified",
        "Main analysis not sealed",
    )
    bind(MAIN / "part2_complete.json")
    bind(
        MAIN / "scores_complete.json",
        completion["outputs"][rel(MAIN / "scores_complete.json")],
    )
    for name in sorted(epmap):
        path = MAIN / "endpoints" / slug(name) / "results.json"
        bind(path, scores["outputs"][rel(path)])
        result = read(path)
        require(result["endpoint"] == epmap[name], "Endpoint contract mismatch")
        bind(ROOT / result["encoded_path"], result["encoded_sha256"])
    roster = (
        ROOT
        / "experiments/astra/offline_closure_20261006_v1/evaluation_preparation/broad_roster.jsonl"
    )
    bind(roster, setup["sources"][rel(roster)])
    bind(ROOT / "Thesis_Log/astra_result/astra_sq2_broad_robustness_20261104_v2.md")
    bind(
        ROOT
        / "Thesis_Log/astra_result/astra_sq2_broad_robustness_tables_20261104_v1.md"
    )
    save(BASE / "preservation_before.json", dict(evidence=required))
    contract = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="post_hoc_offline_plan_frozen_before_new_pairwise_computation",
        context="Earlier three-repeat unanimity outcomes were already known. This is not a new preregistered experiment.",
        endpoints=sorted(epmap),
        conditions=CONDITIONS,
        outer_repeat_pairs=PAIRS,
        primary_estimand="For each text, mean of three unordered pairs of the three saved voted outer-repeat answers; then mean across texts.",
        equality="Hate: boolean; legal: exact label set independent of hate; fine fields: joint (hate gate, exact set or severity class). Ignore rationale prose and label order.",
        correctness="Exact match to original endpoint annotation; fine fields additionally require predicted hate=true. Known empty remains distinct from missing annotation.",
        main_population="Original endpoint-eligible texts with all nine final answers valid, same texts for all three conditions.",
        sensitivities=[
            "Full/no retrieval all-six-valid cohort",
            "All 27 inner slots valid on the main common-nine cohort",
            "Full/no retrieval matched available outer pairs with equal weight per available pair",
            "Conservative missing-pair bounds on the full originally eligible cohort",
            "Three pair-specific effects; no resampling repeats as independent texts",
            "Gold hate/nonhate strata for hate; equal-negative-gate contribution for fine fields",
        ],
        uncertainty=dict(
            draws=DRAWS,
            seed=SEED,
            generator="numpy PCG64 default_rng; endpoint index added to seed",
            sampling="Resample original whole clusters with replacement; same draws for all conditions; texts retain their weights through numerator/denominator ratio.",
            clusters="Original normalized-text clusters except original connected template/reference/text components for HateCheck.",
            intervals="Two-sided percentile 95% for every contrast. Additionally 1-0.05/11 intervals for the eleven full-minus-no-retrieval endpoint differences.",
            family_size=11,
            family="Post-hoc full/no retrieval agreement contrasts including DEV and the sparse German-target diagnostic; not the original F1 primary family.",
            limits="Conditional on saved prompts, model, texts and three outer repeats. Not decoder-population uncertainty, independent replication, causal retrieval-relevance identification, or simultaneous inference over all secondary contrasts.",
        ),
        fixed_outputs=[
            "Reproduce all 33 previous three-repeat stability/correctness rows",
            "Pairwise agreement and paired differences",
            "Correct versus wrong agreeing pairs and repeat unanimity",
            "Coverage and declared sensitivities",
            "Machine-readable tables",
        ],
        forbidden=[
            "New model/API/model-list calls",
            "Model loading or training",
            "New retrieval",
            "Changes to old results or thesis",
            "Server operations",
        ],
        evidence_sha256=sha(BASE / "preservation_before.json"),
    )
    save(BASE / "analysis_contract.json", contract)
    print(
        "Frozen offline contract; checked", len(required), "evidence files.", flush=True
    )


def load_endpoint(name, meta, keys):
    result = read(MAIN / "endpoints" / slug(name) / "results.json")
    ep = result["endpoint"]
    with np.load(ROOT / result["encoded_path"], allow_pickle=False) as z:
        order = [result["settings"].index(list(s)) for s in SETTINGS]
        p = (
            z["predicted"][order]
            .copy()
            .reshape(3, 3, len(ep["scorable_ids"]), -1)
            .astype(bool)
        )
        valid = z["valid"][order].copy().reshape(3, 3, -1)
        gold = z["gold"].copy().astype(bool)
    ii = [keys[k] for k in ep["scorable_ids"]]
    gates = meta["gate"][:, ii].reshape(3, 3, -1).astype(bool)
    nv = meta["n_valid"][:, ii].reshape(3, 3, -1)
    require(np.array_equal(valid, nv > 0), "Validity differs from original metadata")
    fine = ep["dimension"] not in ("hate", "legal")
    if fine:
        require(
            not np.any(p & ~gates[..., None]), "Uncleared negative-gate fine answer"
        )
    weights = 2 ** np.arange(p.shape[-1], dtype=np.int64)
    states = p.astype(np.int64) @ weights
    if fine:
        states += gates * 128
    exact = np.all(p == gold, axis=-1)
    if fine:
        exact &= gates
    return ep, p, gold, valid, gates, nv, states, exact


def pair_arrays(states, correct):
    agree = np.stack([states[:, a] == states[:, b] for a, b in PAIRS], axis=1)
    correct_pairs = np.stack([correct[:, a] & correct[:, b] for a, b in PAIRS], axis=1)
    require(not np.any(correct_pairs & ~agree), "Two correct endpoint states disagree")
    for j, (a, b) in enumerate(PAIRS):
        require(
            not np.any(agree[:, j] & (correct[:, a] != correct[:, b])),
            "Same answer with different correctness",
        )
    return agree, correct_pairs, agree & ~correct_pairs


def bootstrap(pair_counts, clusters, draws, seed):
    "Each row is one text; three columns hold condition counts out of 3 pairs."
    clusters = np.asarray(clusters)
    unique, inverse = np.unique(clusters, return_inverse=True)
    sizes = np.bincount(inverse)
    grouped = np.stack(
        [
            np.bincount(inverse, weights=pair_counts[:, j])
            for j in range(pair_counts.shape[1])
        ],
        axis=1,
    )
    rng = np.random.default_rng(seed)
    values = np.empty((draws, pair_counts.shape[1]), dtype=np.float64)
    for start in range(0, draws, 128):
        stop = min(start + 128, draws)
        chosen = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        denom = sizes[chosen].sum(axis=1) * 3
        values[start:stop] = grouped[chosen].sum(axis=1) / denom[:, None]
    return values


def interval(values, confidence):
    alpha = (1 - confidence) / 2
    return np.quantile(values, [alpha, 1 - alpha], method="linear").tolist()


def contrasts(means):
    return {
        CONDITIONS[a] + "_minus_" + CONDITIONS[b]: float(means[a] - means[b])
        for a, b in CONTRASTS
    }


def subset_result(mask, agree):
    n = int(mask.sum())
    means = agree[:, :, mask].mean(axis=(1, 2)) if n else np.full(3, np.nan)
    return dict(
        n=n,
        agreement=means.tolist() if n else None,
        differences=contrasts(means) if n else None,
    )


@offline()
def analyze():
    begin = time.monotonic()
    contract = read(BASE / "analysis_contract.json")
    require(contract["uncertainty"]["draws"] == DRAWS, "Frozen settings differ")
    require(
        read(BASE / "tests.json")["status"] == "passed", "Synthetic tests must pass"
    )
    before = read(BASE / "preservation_before.json")
    verify(before["evidence"])
    epmap = read(READER / "endpoint_setup.json")
    index = read(ROBUST / "metadata_index.json")
    require(
        [tuple(s) for s in index["settings"]] == SETTINGS, "Metadata settings differ"
    )
    keys = {k: i for i, k in enumerate(index["keys"])}
    with np.load(ROBUST / "prediction_metadata.npz", allow_pickle=False) as z:
        meta = {k: z[k].copy() for k in z.files}
    old = {
        (r["endpoint"], r["condition"]): r["outer"]
        for r in read(ROBUST / "repeat_and_inner_diagnostics.json")
    }
    result = []
    old_checks = []
    for no, name in enumerate(sorted(epmap)):
        ep, p, gold, valid, gates, nv, states, exact = load_endpoint(name, meta, keys)
        agree, cp, wp = pair_arrays(states, exact)
        common = valid.all(axis=(0, 1))
        agree_common = agree[:, :, common]
        n = int(common.sum())
        means = agree_common.mean(axis=(1, 2))
        per_text_counts = agree_common.sum(axis=1).T
        clusters = np.asarray(ep["row_clusters"])[common]
        boot = bootstrap(per_text_counts, clusters, DRAWS, SEED + no)
        folder = BASE / "endpoints" / slug(name)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "agreement_draws.npy").open("xb") as f:
            np.save(f, boot, allow_pickle=False)
        with (folder / "text_contributions.npz").open("xb") as f:
            np.savez_compressed(
                f,
                ids=np.array(ep["scorable_ids"]),
                common=common,
                valid=valid,
                gate=gates,
                n_valid=nv,
                states=states,
                correct=exact,
                agree=agree,
                correct_pairs=cp,
                wrong_pairs=wp,
                clusters=np.array(ep["row_clusters"]),
            )
        condition_rows = []
        for j, co in enumerate(CONDITIONS):
            same = agree_common[j].all(axis=0)
            ex = exact[j][:, common]
            counts = dict(
                n_common_nine=n,
                same_answer_all_repeats=int(same.sum()),
                same_wrong_answer=int((same & ~ex.any(axis=0)).sum()),
                all_repeats_correct=int(ex.all(axis=0).sum()),
                all_repeats_wrong=int((~ex.any(axis=0)).sum()),
                variable_correctness=int((ex.any(axis=0) & ~ex.all(axis=0)).sum()),
                correct_repeats_hist={
                    str(k): int(v) for k, v in Counter(ex.sum(axis=0).tolist()).items()
                },
            )
            original = old[name, co]
            for key, value in counts.items():
                require(
                    value == original[key],
                    "Original stability result differs: " + name + "/" + co + "/" + key,
                )
            require(
                np.allclose(
                    ex.mean(axis=1), original["per_repeat_exact"], atol=1e-14, rtol=0
                ),
                "Original exact rate differs",
            )
            old_checks.append(dict(endpoint=name, condition=co, reproduced=counts))
            row = dict(
                condition=co,
                **counts,
                agreeing_pairs=int(agree_common[j].sum()),
                pair_denominator=3 * n,
                pairwise_agreement=float(means[j]),
                ci95=interval(boot[:, j], 0.95),
                per_pair_agreement=agree_common[j].mean(axis=1),
                correct_agreeing_pairs=int(cp[j][:, common].sum()),
                wrong_agreeing_pairs=int(wp[j][:, common].sum()),
                correct_pair_share=float(cp[j][:, common].mean()),
                wrong_pair_share=float(wp[j][:, common].mean()),
                per_repeat_exact=ex.mean(axis=1),
                eligible_null_final_answers=int((~valid[j]).sum()),
                eligible_texts_with_missing_outer=int((~valid[j]).any(axis=0).sum()),
                stable_correct=int(ex.all(axis=0).sum()),
            )
            if ep["dimension"] not in ("hate", "legal"):
                negative_pairs = np.stack(
                    [~gates[j, a, common] & ~gates[j, b, common] for a, b in PAIRS]
                )
                require(
                    not np.any(negative_pairs & ~agree_common[j]),
                    "Cleared negative answers disagree",
                )
                row["negative_gate_agreeing_pairs"] = int(negative_pairs.sum())
                row["negative_gate_pair_share"] = float(negative_pairs.mean())
                row["positive_gate_agreeing_pair_share"] = float(
                    (agree_common[j] & ~negative_pairs).mean()
                )
            condition_rows.append(row)
        contrast_rows = []
        for a, b in CONTRASTS:
            diff = boot[:, a] - boot[:, b]
            row = dict(
                left=CONDITIONS[a],
                right=CONDITIONS[b],
                difference=float(means[a] - means[b]),
                ci95=interval(diff, 0.95),
                per_pair_difference=(
                    agree_common[a].mean(axis=1) - agree_common[b].mean(axis=1)
                ),
                correct_pair_difference=float(
                    cp[a][:, common].mean() - cp[b][:, common].mean()
                ),
                wrong_pair_difference=float(
                    wp[a][:, common].mean() - wp[b][:, common].mean()
                ),
            )
            if (a, b) == (0, 2):
                row["family11_confidence"] = 1 - 0.05 / 11
                row["family11_interval"] = interval(diff, 1 - 0.05 / 11)
                row["family11_interval_first_half"] = interval(
                    diff[: DRAWS // 2], 1 - 0.05 / 11
                )
                row["family11_interval_second_half"] = interval(
                    diff[DRAWS // 2 :], 1 - 0.05 / 11
                )
            contrast_rows.append(row)
        six = valid[[0, 2]].all(axis=(0, 1))
        strict = (nv == 3).all(axis=(0, 1))
        require(
            not np.any(strict & ~common), "Inner complete not subset of final complete"
        )
        six_means = agree[[0, 2]][:, :, six].mean(axis=(1, 2))
        available = np.stack(
            [valid[0, a] & valid[0, b] & valid[2, a] & valid[2, b] for a, b in PAIRS]
        )
        avail_num = [int((agree[j] & available).sum()) for j in (0, 2)]
        bounds = []

        full_denom = len(ep["scorable_ids"]) * 3
        for j in (0, 2):
            observed = np.stack([valid[j, a] & valid[j, b] for a, b in PAIRS])
            known = int((agree[j] & observed).sum())
            missing = int((~observed).sum())
            bounds.append(
                dict(
                    condition=CONDITIONS[j],
                    lower=known / full_denom,
                    upper=(known + missing) / full_denom,
                    known_agreeing_pairs=known,
                    missing_pairs=missing,
                    denominator=full_denom,
                )
            )
        sensitivity = dict(
            full_no_all_six_valid=dict(
                n=int(six.sum()),
                agreement=six_means,
                difference=float(six_means[0] - six_means[1]),
            ),
            all_27_inner_valid=subset_result(strict, agree),
            full_no_matched_available_pairs=dict(
                n_pairs=int(available.sum()),
                n_texts=int(available.any(axis=0).sum()),
                agreeing_pairs=avail_num,
                agreement=[v / available.sum() for v in avail_num],
                difference=(avail_num[0] - avail_num[1]) / available.sum(),
            ),
            full_roster_missing_pair_bounds=dict(
                conditions=bounds,
                difference_lower=bounds[0]["lower"] - bounds[1]["upper"],
                difference_upper=bounds[0]["upper"] - bounds[1]["lower"],
                note="Conservative logical bounds, not confidence intervals; unknown pair assignments need not be jointly attainable.",
            ),
        )
        strata = {}
        if ep["dimension"] == "hate":
            for value in (0, 1):
                strata["gold_hate_" + str(value)] = subset_result(
                    common & (gold[:, ep["all_labels"].index("hate")] == value), agree
                )
        row = dict(
            endpoint=name,
            phase=ep["phase"],
            dimension=ep["dimension"],
            originally_eligible=len(ep["scorable_ids"]),
            n_common_nine=n,
            excluded_missing=len(ep["scorable_ids"]) - n,
            n_clusters=len(set(clusters)),
            original_clusters=len(set(ep["row_clusters"])),
            unique_sources=dict(
                Counter(np.asarray(ep["row_sources"])[common].tolist())
            ),
            conditions=condition_rows,
            contrasts=contrast_rows,
            sensitivities=sensitivity,
            strata=strata,
            sparse_annotation_diagnostic=name == "de_test/target_group",
            development_endpoint=ep["phase"] == "larger_dev",
            bootstrap_draws_path=rel(folder / "agreement_draws.npy"),
            bootstrap_draws_sha256=sha(folder / "agreement_draws.npy"),
        )
        save(folder / "results.json", row)
        result.append(row)
        print(
            name,
            "n=",
            n,
            "agreement=",
            [round(100 * v, 4) for v in means],
            "full-minus-none pp=",
            round(100 * (means[0] - means[2]), 4),
            flush=True,
        )
    save(
        BASE / "original_stability_reproduction.json",
        dict(status="all_33_previous_rows_reproduced", rows=old_checks),
    )
    save(
        BASE / "results.json",
        dict(
            status="pairwise_analysis_complete_pending_independent_verification",
            conditions=CONDITIONS,
            endpoints=result,
            elapsed_seconds=time.monotonic() - begin,
            sources=dict(
                contract=sha(BASE / "analysis_contract.json"),
                tests=sha(BASE / "tests.json"),
                preservation=sha(BASE / "preservation_before.json"),
                code=sha(__file__),
            ),
            new_Qwen_calls=0,
            training=0,
            new_inference=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "analyze"))
    args = parser.parse_args()
    {"freeze": freeze, "analyze": analyze}[args.mode]()
