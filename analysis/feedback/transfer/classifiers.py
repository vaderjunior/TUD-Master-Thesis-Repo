import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import Counter
from pathlib import Path
import gzip
import io
import json
import sys
import unittest
import numpy as np
from analysis.feedback.transfer import archive as a
from analysis.feedback.transfer import effects_metrics as masks
from analysis.feedback.transfer import classifiers_metrics as k
from tests.feedback.transfer import test_classifiers as tests

OUT = a.ROOT / "experiments/astra/sq3_transfer_part8_20261023_v1"
P2 = a.ROOT / "experiments/astra/sq3_transfer_part2_20261023_v1"
P3 = a.ROOT / "experiments/astra/sq3_transfer_part3_20261023_v1"
P5 = a.ROOT / "experiments/astra/sq3_transfer_part5_20261023_v1"
P7 = a.ROOT / "experiments/astra/sq3_transfer_part7_20261023_v1"
LINEAR = a.ROOT / "experiments/astra/sq3_linear_update_20261012_v1"
REVIEW = a.ROOT / "experiments/astra/sq3_linear_static_review_20261014_v1/review.json"
CONDITIONS = (
    "reference",
    "error_balanced_s2026100441",
    "error_balanced_s2026100442",
    "error_balanced_s2026100443",
)
ENDPOINTS = (
    "en_test_FPR",
    "en_test_FNR",
    "en_hatecheck_eval_FPR",
    "en_hatecheck_eval_FNR",
)
POPS = ("en_test", "en_hatecheck_eval")


def native(value):
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): native(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def save(name, value):
    path = OUT / name
    a.need(path.resolve().is_relative_to(OUT), "Write outside new Part 8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value if isinstance(value, bytes) else a.encoded(native(value), pretty=True)
        )


def array(name, **values):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def gzopen(name):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return io.TextIOWrapper(
        gzip.GzipFile(filename=str(path), mode="xb", mtime=0, compresslevel=5),
        encoding="utf8",
        newline="\n",
    )


def emit(f, value):
    f.write(
        json.dumps(
            native(value),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def gzrows(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def prepare():
    a.need(not OUT.exists(), "Preserve prior Part 8 output")
    previous = a.read(P7 / "complete.json")
    check = a.read(P7 / "publication/final_check.json")
    a.need(
        check["complete_sha256"] == a.sha(P7 / "complete.json")
        and previous["completed_transfer_parts"] == list(range(1, 8)),
        "Latest verified Part 7 checkpoint",
    )
    plan = a.read(LINEAR / "plan.json")
    a.need(
        plan["principal_variant"] == "all_permitted"
        and plan["cutoff"] == 0.6
        and plan["cutoff_operator"] == ">"
        and plan["bootstrap_seeds"] == [2026101224, 2026101225, 2026101226, 2026101227]
        and plan["primary_confidence"] == 0.9875,
        "Original classifier primary contract",
    )
    bindings = {
        **previous["input_bindings"],
        **previous["analysis_sources"],
        **previous["artifacts"],
    }
    for path in (
        P7 / "complete.json",
        P7 / "publication/final_check.json",
        REVIEW,
        LINEAR / "plan.json",
        LINEAR / "STUDY_PROTOCOL.md",
        LINEAR / "static_complete.json",
        LINEAR / "static_schedule.json",
        LINEAR / "roster.jsonl",
        LINEAR / "data.json",
    ):
        bindings[a.rel(path)] = a.sha(path)
    for name, digest in bindings.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed bound evidence " + name)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    a.need(result.wasSuccessful(), "Comparison fixtures failed " + log.getvalue())
    save(
        "analysis_contract.json",
        dict(
            created_utc=a.utc(),
            status="part8_implementation_recorded_before_new_joint_broad_scoring",
            input_bindings=bindings,
            timing="The original four primary difference-in-change contrasts, settings, masks and seeds were frozen on 12 October. Detailed supporting implementation is recorded now, after Qwen Parts 1–7 and the earlier DEV comparison are known, before this new joint broad-transfer scoring. Supporting additions are not a new prospective discovery family.",
            primary=dict(
                variant="all_permitted",
                cutoff_strictly_greater_than=0.6,
                endpoints=ENDPOINTS,
                formula="(updated linear - baseline linear) - (updated Qwen - reference Qwen)",
                confidence=0.9875,
                draws=10000,
                seeds=plan["bootstrap_seeds"],
            ),
            settings=k.SETTINGS,
            conditions=CONDITIONS,
            views=list(masks.VIEWS),
            replay_exclusions=a.read(a.STUDY / "scientific_plan_v1.json")[
                "mandatory_replay_keys"
            ],
            comparison="Identical A16 endpoint-eligible and all-arm valid masks; same original inference clusters and fixed representative rule. Deterministic classifier baseline once, three actual updated pool fits. Qwen reference three actual outputs, updates nine. Same source IDs and labels, different update mechanisms and retained training/context information.",
            bootstrap=dict(
                generator="PCG64",
                draws=10000,
                unit="Whole inference cluster, item-weighted",
                columns=k.COLUMNS,
                settings_axis=k.SETTINGS,
                family_seed_rule="Main endpoint streams use the four original A18 seeds. Other family streams: first eight SHA256 bytes, little endian, of A18-transfer-part8-support-v1| plus family ID. All four sensitivity settings use the same paired resample indices.",
                quantiles="Linear interpolation, 95% supporting intervals throughout. Only principal all_permitted >0.6 main difference-in-change uses the prespecified 98.75% family.",
                resampling="Six global masks; source-stratified English and item-level HateCheck on main/strict/replay-excluded; three natural sources on main/strict. Source strata from endpoint-eligible cluster source signatures before validity; do not split mixed-source clusters.",
            ),
            additional_points=[
                "All individual fits/pools and Qwen repeats; fixed-mask and recomputed-mask leave-one-pool, all execution-batch omissions.",
                "All 29 HateCheck functions on main/strict and every original template by eligible endpoint on main; all four classifier settings.",
                "Actual eight-worker and 128-worker homogeneous input eras on main/strict; descriptive, not causal concurrency effects.",
                "Binary macro F1, balanced accuracy, accuracy, precision, recall and positive prediction rates for six full-population masks; score each actual fit/repeat separately before averaging. These extra quality views are points only; no new fine-head system is constructed.",
                "Full-roster classifier probability shifts and threshold crossings, per-pool fixes/regressions, paired input/pool directional cross-tabs and classifier/Qwen absolute disagreements.",
                "Principal leave-each-inference-cluster and HateCheck-function difference-in-change point influence; no omission intervals.",
                "Complete input ledger; deterministic hash-selected cases by predeclared outcome categories and population, including empty selection slots. Cases are illustrations, not prevalence estimates.",
            ],
            cases=dict(
                setting="all_permitted >0.6",
                mask="main common",
                slots_per_population=[
                    "linear_consistent_fix",
                    "linear_consistent_regression",
                    "linear_improves_qwen_worsens",
                    "linear_worsens_qwen_improves",
                    "linear_unchanged_qwen_improves",
                    "linear_unchanged_qwen_worsens",
                    "updated_linear_correct_qwen_wrong",
                    "updated_linear_wrong_qwen_correct",
                ],
                selection="Minimum SHA256 of A18-part8-case-v1|population|category|key; fixed original key ordering breaks exact hash ties. No replacement if empty.",
            ),
            missingness="Opposing unknown Qwen output assignments on full eligible cohorts, all classifier decisions fixed. Null-only, strict under-three unknown, and hypothetical invalid-vote completion from verified Part 7. Bound difference-in-change reverses the Qwen delta range; these are identification bounds, not CIs.",
            fit_proof="Reuse independently verified 38-fit static study. Read/hash the eight consumed baseline/balanced-error fits and saved arrays; reproduce strict cutoff bits and exact input/pool joins. No refit, model optimization, embedding or repeated full raw audit.",
            limitations=[
                "Previously exposed TEST and functional HateCheck, not new deployment holdouts.",
                "Feedback is RAG-error-selected simulated binary labels; no real expert/adaptive classifier acquisition claim.",
                "Fixed overlapping pools and repeated outputs are dependent.",
                "Same retained label budget does not mean identical context exposure, token count, prior supervision or optimization.",
                "Fine heads unchanged; this comparison is binary only.",
                "Cumulative repeated-round classifier work remains pending.",
                "No new threshold, correction weighting, policy or winning-pool selection.",
            ],
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_annotations=0,
            analysis_sources={
                a.rel(Path(module.__file__)): a.sha(Path(module.__file__))
                for module in (sys.modules[__name__], k, tests, masks, a)
            },
        ),
    )
    save(
        "focused_tests.json",
        dict(passed=True, tests_run=result.testsRun, log=log.getvalue()),
    )
    print("Part 8 contract and twenty focused checks saved.", flush=True)


def assemble():
    a.need(
        not (OUT / "assembled.json").exists() and not (OUT / "arrays.npz").exists(),
        "Preserve existing Part 8 assembly",
    )
    contract = a.read(OUT / "analysis_contract.json")
    review = a.read(REVIEW)
    done = a.read(LINEAR / "static_complete.json")
    data = a.read(LINEAR / "data.json")
    sources = a.read(P5 / "source_catalog.json")
    for group in ("input_bindings", "analysis_sources"):
        for name, digest in contract[group].items():
            a.need(
                a.sha(a.ROOT / name) == digest, "Changed assembly dependency " + name
            )
    a.need(
        review["unique_fits"] == 38
        and review["updated_fits"] == 36
        and review["probabilities_independently_reconstructed"] == 624454
        and review["saved_decisions_independently_reconstructed"] == 1248908,
        "Original independent fit proof",
    )
    for name in (
        "plan.json",
        "static_complete.json",
        "static_schedule.json",
        "roster.jsonl",
        "data.json",
    ):
        a.need(
            review["bindings"][a.rel(LINEAR / name)] == a.sha(LINEAR / name),
            "Consumed classifier source changed",
        )
    linear_roster = list(a.rows(LINEAR / "roster.jsonl"))
    index = {r["key"]: i for i, r in enumerate(linear_roster)}
    roster = list(a.rows(a.OUT / "roster.jsonl"))
    meta = list(a.rows(P3 / "metadata.jsonl"))
    matrix = list(a.rows(a.OUT / "matrix.jsonl"))
    a.need(
        len(index) == len(linear_roster) == 16433
        and len(roster) == len(meta) == len(matrix) == 15758,
        "Exact broad and union coverage",
    )
    indices = []
    qwen = np.full((15758, 4, 3), -1, dtype=np.int8)
    votes = np.zeros_like(qwen)
    for i, (r, m, q) in enumerate(zip(roster, meta, matrix)):
        l = linear_roster[index[r["key"]]]
        indices.append(index[r["key"]])
        a.need(
            (r["key"], r["text"], r["gate"], r["inference_cluster"], r["population"])
            == (
                l["key"],
                l["text"],
                l["gate"],
                l["inference_cluster"],
                l["population"],
            ),
            "Classifier identity/text/gold/cluster join",
        )
        a.need(
            m["key"] == q["key"] == r["key"]
            and m["gold_hate"] == q["gold_hate"] == l["gate"]
            and m["input_index"] == q["input_index"] == i,
            "Original matrix/metadata join",
        )
        for c in range(4):
            for t in range(3):
                v = q["predictions"][c][t]
                votes[i, c, t] = v["n_valid"]
                qwen[i, c, t] = -1 if v["result"] is None else int(v["result"]["hate"])
    gold = np.asarray([m["gold_hate"] for m in meta], dtype=np.int8)
    a.need(
        np.array_equal(qwen < 0, votes == 0), "Null classifier-independent Qwen mask"
    )
    probabilities = np.empty((15758, 2, 4), dtype=np.float64)
    decisions = np.empty((15758, 2, 2, 4), dtype=np.int8)
    bindings = {}
    jobs = {}
    pool_ids = {
        c: sorted(ident for ident, v in sources.items() if c in v["pool_memberships"])
        for c in CONDITIONS[1:]
    }
    a.need(
        len(set().union(*map(set, pool_ids.values()))) == 98
        and all(len(ids) == 40 for ids in pool_ids.values()),
        "Three original forty-label pools",
    )
    candidate_labels = dict(zip(data["candidate_ids"], data["candidate_labels"]))
    candidates = {
        r["id"]: r
        for r in a.rows(
            a.ROOT
            / "experiments/astra/sq3_additions_20261004_v1/benign_selection/roster.jsonl"
        )
    }
    for c, ids in pool_ids.items():
        a.need(
            sum(candidate_labels[ident] for ident in ids) == 20,
            "Balanced source labels",
        )
        for ident in ids:
            a.need(
                candidate_labels[ident]
                == sources[ident]["record"]["meta"]["gate"]
                == candidates[ident]["gold"]["hate"],
                "Same source gate label",
            )
            a.need(
                candidates[ident]["text"] == sources[ident]["record"]["text"],
                "Same source text",
            )
    for vi, variant in enumerate(k.VARIANTS):
        for ci, condition in enumerate(CONDITIONS):
            name = variant + "/" + condition
            job = done["jobs"][name]
            receipt = a.ROOT / job["receipt"]
            record = a.read(receipt)
            ids = [] if ci == 0 else pool_ids[condition]
            a.need(
                a.sha(receipt) == job["sha256"] == review["bindings"][a.rel(receipt)],
                "Consumed fit receipt identity",
            )
            a.need(
                ids
                == job["source_ids"]
                == record["identity"]["source_ids"]
                == record["detail"]["source_ids"],
                "Exactly matching source set",
            )
            a.need(
                record["identity"]["variant"] == variant
                and record["identity"]["C"] == 10
                and record["identity"]["seed"] == 42,
                "Frozen fit recipe",
            )
            bindings[a.rel(receipt)] = a.sha(receipt)
            for rel, digest in record["outputs"].items():
                path = receipt.parent / rel
                a.need(
                    a.sha(path) == digest == review["bindings"][a.rel(path)],
                    "Saved fit artifact hash",
                )
                bindings[a.rel(path)] = digest
            attempt = receipt.parent / record["attempt"]
            prob = np.load(attempt / "probabilities.npy", allow_pickle=False)
            a.need(
                prob.shape == (16433, 1)
                and np.isfinite(prob).all()
                and ((prob >= 0) & (prob <= 1)).all(),
                "Complete saved probabilities",
            )
            probabilities[:, vi, ci] = prob[indices, 0]
            for ti, cutoff in enumerate(k.CUTOFFS):
                suffix = "06" if cutoff == 0.6 else "05"
                bits = np.load(attempt / f"decisions_{suffix}.npy", allow_pickle=False)
                a.need(
                    bits.shape == prob.shape and np.array_equal(bits, prob > cutoff),
                    "Original strict cutoff bits",
                )
                decisions[:, vi, ti, ci] = bits[indices, 0]
            attribution = a.read(attempt / "attribution.json")
            proof = review["jobs"][a.rel(receipt)]
            a.need(
                attribution["fit_rows"] == len(data["partitions"][variant]) + len(ids)
                and attribution["negative"] + attribution["positive"]
                == attribution["fit_rows"],
                "Fit class counts",
            )
            a.need(
                proof["source_ids"] == ids
                and proof["baseline_reproduced"] == (ci == 0)
                and not attribution["fine_heads_changed"],
                "Saved model proof and binary-only scope",
            )
            jobs[name] = dict(
                receipt=a.rel(receipt),
                receipt_sha256=a.sha(receipt),
                attempt=a.rel(attempt),
                source_ids=ids,
                fit_rows=attribution["fit_rows"],
                negative=attribution["negative"],
                positive=attribution["positive"],
                class_weights=proof["class_weights"],
                baseline_reproduced=proof["baseline_reproduced"],
                fit_seconds=attribution["fit_seconds"],
                iterations=attribution["iterations"],
                old_independent_max_abs_difference=proof[
                    "independent_max_abs_difference"
                ],
            )
    qerrors = np.where(qwen < 0, -1, qwen != gold[:, None, None]).astype(np.int8)
    lerrors = (decisions != gold[:, None, None, None]).astype(np.int8)
    array(
        "arrays.npz",
        probabilities=probabilities,
        linear_decisions=decisions,
        linear_errors=lerrors,
        qwen_decisions=qwen,
        qwen_errors=qerrors,
        valid_votes=votes,
        linear_roster_indices=np.asarray(indices, dtype=np.int64),
        raw_error_counts=k.raw_rows(lerrors, qerrors),
    )
    save(
        "array_schema.json",
        dict(
            inputs=15758,
            probability_axes=["input", "variant", "condition"],
            decision_axes=["input", "variant", "cutoff", "condition"],
            qwen_axes=["input", "condition", "repeat"],
            variants=k.VARIANTS,
            cutoffs=k.CUTOFFS,
            conditions=CONDITIONS,
            draw_columns=k.COLUMNS,
            settings=k.SETTINGS,
            missing_qwen_code=-1,
        ),
    )
    save(
        "fit_bindings.json",
        dict(
            jobs=jobs,
            pool_ids=pool_ids,
            unique_source_texts=98,
            memberships=120,
            previous_static_fits=38,
            consumed_fits=8,
            reused_baselines=2,
            reused_updates=6,
            new_fits=0,
            original_fit_review=a.rel(REVIEW),
        ),
    )
    files = [path for path in OUT.iterdir() if path.is_file()]
    save(
        "assembled.json",
        dict(
            status="part8_saved_fits_and_transfer_joined",
            created_utc=a.utc(),
            input_bindings={**contract["input_bindings"], **bindings},
            analysis_sources=contract["analysis_sources"],
            artifacts={a.rel(path): a.sha(path) for path in files},
            matched_inputs=15758,
            consumed_fits=8,
            original_fit_probability_rows_checked=8 * 16433,
            saved_decision_bits_checked=8 * 16433 * 2,
            qwen_output_slots=189096,
            probabilities_selected=15758 * 8,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "Eight saved fits exactly joined to all 15,758 transfer inputs; no fitting or calls.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        if sys.argv[1:] == ["prepare"]:
            prepare()
        elif sys.argv[1:] == ["assemble"]:
            assemble()
        else:
            raise ValueError("Use prepare or assemble; neither starts any experiment")
