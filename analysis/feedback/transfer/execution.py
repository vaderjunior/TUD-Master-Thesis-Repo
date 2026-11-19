import os

for n in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[n] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import gzip
import io
import json
import unittest
import numpy as np
from analysis.feedback.transfer import archive as a
from analysis.feedback.transfer import execution_metrics as k

OUT = a.ROOT / "experiments/astra/sq3_transfer_part7_20261023_v1"
P2 = a.ROOT / "experiments/astra/sq3_transfer_part2_20261023_v1"
P3 = a.ROOT / "experiments/astra/sq3_transfer_part3_20261023_v1"
P4 = a.ROOT / "experiments/astra/sq3_transfer_part4_20261023_v1"
P5 = a.ROOT / "experiments/astra/sq3_transfer_part5_20261023_v1"
P6 = a.ROOT / "experiments/astra/sq3_transfer_part6_20261023_v1"
B = a.EXECUTION / "concurrency_20261021_v1"
M = a.EXECUTION / "concurrency_20261021_v3/manifest.json"
CONDITIONS = (
    "reference",
    "error_balanced_s2026100441",
    "error_balanced_s2026100442",
    "error_balanced_s2026100443",
)
POPS = ("en_test", "en_hatecheck_eval")


def native(v):
    if isinstance(v, np.ndarray):
        return native(v.tolist())
    if isinstance(v, np.generic):
        return native(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(key): native(value) for key, value in v.items()}
    if isinstance(v, (list, tuple)):
        return [native(x) for x in v]
    return v


def save(name, value):
    path = OUT / name
    a.need(path.resolve().is_relative_to(OUT), "Write left new analysis")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value if isinstance(value, bytes) else a.encoded(native(value), pretty=True)
        )


def gzopen(name):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return io.TextIOWrapper(
        gzip.GzipFile(filename=str(path), mode="xb", mtime=0, compresslevel=5),
        encoding="utf8",
        newline="\n",
    )


def emit(f, row):
    f.write(
        json.dumps(
            native(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def gzrows(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def main():
    a.need(
        not (OUT / "analysis_contract.json").exists()
        and not (OUT / "assembled.json").exists(),
        "Preserve previous Part 7 analysis",
    )
    OUT.mkdir(exist_ok=True)
    prior = a.read(P6 / "complete.json")
    check = a.read(P6 / "publication/final_check.json")
    a.need(
        prior["completed_transfer_parts"] == [1, 2, 3, 4, 5, 6]
        and check["complete_sha256"] == a.sha(P6 / "complete.json"),
        "Latest verified checkpoint",
    )
    bindings = {
        **prior["input_bindings"],
        **prior["analysis_sources"],
        **prior["artifacts"],
    }
    for path in (
        P6 / "complete.json",
        P6 / "publication/final_check.json",
        P6 / "publication/check_amendment_v2.json",
        a.ROOT / "analysis/feedback/transfer/validate_publication.py",
        M,
        B / "boundary.json",
    ):
        bindings[a.rel(path)] = a.sha(path)
    amendment = a.read(M)
    boundary = a.read(B / "boundary.json")
    checkpoint = a.ROOT / boundary["checkpoint"]
    a.need(
        amendment["workers"] == 128
        and amendment["original_plan_workers"] == 8
        and amendment["bindings"][a.rel(B / "boundary.json")]
        == bindings[a.rel(B / "boundary.json")],
        "Runtime amendment binding",
    )
    v1 = a.read(B / "manifest.json")
    cp_done = checkpoint.parent / "complete.json"
    cp = a.read_bound(cp_done, v1["bindings"][a.rel(cp_done)])
    a.need(
        cp["evidence_inventory"] == a.rel(checkpoint)
        and cp["evidence_inventory_sha256"] == a.sha(checkpoint),
        "Checkpoint inventory chain",
    )
    bindings[a.rel(cp_done)] = v1["bindings"][a.rel(cp_done)]
    bindings[a.rel(checkpoint)] = cp["evidence_inventory_sha256"]
    provenance = []
    for path in sorted((B / "provenance").glob("transfer_*/*.json")):
        record = a.read(path)
        a.need(
            record["effective_workers"] == 128
            and record["amendment"] == dict(path=a.rel(M), sha256=a.sha(M)),
            "Unexpected runtime provenance",
        )
        at = a.ROOT / record["original_attempt"]
        start = a.read(at / "start.json")
        inv = a.read(at / "inventory.json")
        for f in (path, at / "start.json", at / "inventory.json"):
            bindings[a.rel(f)] = a.sha(f)
        provenance.append(
            dict(
                stage=record["stage"],
                path=a.rel(path),
                started_utc=start["started_utc"],
                pending=inv["pending_identity_hashes"],
                previously_complete=inv["completed"],
            )
        )
    a.need(
        len(provenance) == 19
        and [x["stage"] for x in provenance]
        == [f"transfer_{i:02}" for i in range(4, 23)],
        "Nineteen actual 128-worker attempts",
    )
    for name, digest in bindings.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed inherited evidence " + name)
    sources = [
        Path(__file__),
        Path(k.__file__),
        a.ROOT / "tests/feedback/transfer/test_execution.py",
    ]
    save(
        "analysis_contract.json",
        dict(
            created_utc=a.utc(),
            input_bindings=bindings,
            analysis_sources={a.rel(f): a.sha(f) for f in sources},
            preserved_results="Reuse Parts 2–4 cluster/item/source-stratified draws, strict/replay/representative masks, missing bounds, source views and all pool/batch omissions with their independent proofs. Do not generate replacement predictions or refit anything.",
            runtime="Use the bound pre-change completed identity set and actual 128-worker pending inventories, not receipt timestamps, to classify every output. Whole-input eras are all_8, all_128 or cross_boundary. Keep earlier coarse batch eras separately. Validate original receipt/result hashes for all completed-before-change slots.",
            inference="Era comparisons are observational and contain different texts. The same alias does not prove an unchanged backend checkpoint. No causal concurrency effect, equivalence, or source-adjusted causal claim. Original four-endpoint 98.75% family unchanged.",
            new_rate_views="Both homogeneous eras on all six inherited masks; main/strict source and coarse-batch-era points; cross-boundary and omit-cross-boundary views. All four endpoints remain separate, including empty cells. Main/strict homogeneous-era differences get 16 supporting cluster-bootstrap sets, 10000 PCG64 draws, 95% intervals and deterministic A16-part7-support-v1|name seeds; no new confirmatory family.",
            representative_rule="Select global endpoint/head eligible representatives using the inherited rule before validity, then intersect with the era; never replace a representative merely because it lies in another era.",
            quality="All existing scorable head metrics on main/strict homogeneous-era and omit-boundary cohorts, per actual run then equal means. No new quality intervals. Known gold remains fixed and incomplete; no all-schema accuracy claim.",
            missingness="Separate null aggregate outputs, under-three-valid-vote outputs, and affected inputs. Publish source/gold/condition/repeat/batch/actual-era composition and original parse/finish summaries. Bounds are fixed-roster identification ranges, not confidence intervals.",
            additional_vote_sensitivity="Hypothetically fill only invalid raw vote slots with arbitrary hate booleans, retaining every valid vote, and recompute a three-vote strict majority. Two agreeing valid votes already fix the binary majority. Bound full-roster reference/treatment error differences; not a rerun or the actual majority-of-valid-votes estimator. No fine-field completion bound or missing-at-random assumption.",
            influence="Whole-cluster and HateCheck whole-function omissions on main/strict endpoint common masks; points only, keep every group. Existing pool/batch omissions and all earlier supporting qualifications remain. Descriptive source standardization uses global main-common endpoint source weights, not fitted weights.",
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_primary_families=0,
        ),
    )
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.feedback.transfer.test_execution"
    )
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    save(
        "focused_tests.json",
        dict(
            tests_run=tests.testsRun, passed=tests.wasSuccessful(), log=log.getvalue()
        ),
    )
    a.need(tests.wasSuccessful(), "Boundary fixtures failed")
    before = a.read(checkpoint)
    old = set(boundary["completed_prediction_identity_hashes"])
    a.need(
        len(old) == 31565 and not boundary["partial_item_identity_hashes"],
        "Unexpected partial/runtime boundary",
    )
    pending = {item: q for q in provenance for item in q["pending"]}
    a.need(
        len(pending) == sum(len(q["pending"]) for q in provenance) == 157531
        and not old.intersection(pending),
        "Overlapping runtime assignments",
    )
    meta = list(a.rows(a.OUT / "matrix.jsonl"))
    N = len(meta)
    workers = np.zeros((N, 4, 3), dtype=np.int16)
    missing = np.full((N, 4, 3), -1, dtype=np.int8)
    counts = Counter()
    models = Counter()
    classes = Counter()
    finish = Counter()
    nulls = []
    inventory = []
    summary = defaultdict(
        lambda: dict(
            outputs=0,
            valid_vote_counts=Counter(),
            repairs=0,
            parse_failures=0,
            finish_reasons=Counter(),
            response_classes=Counter(),
            reported_models=Counter(),
            gate_ties=0,
        )
    )
    with gzopen("output_execution.jsonl.gz") as out:
        for path in sorted((a.OUT / "assembly").glob("transfer_*.jsonl")):
            for row in a.rows(path):
                i, c, r = (
                    row["input_index"],
                    CONDITIONS.index(row["condition"]),
                    row["repeat"],
                )
                item = Path(row["item_receipt"]).parent.name
                a.need(
                    workers[i, c, r] == 0 and meta[i]["key"] == row["key"],
                    "Duplicate or wrong slot",
                )
                w = 8 if item in old else 128
                a.need(item in old or item in pending, "Output has no runtime identity")
                if w == 8:
                    a.need(
                        before[row["item_receipt"]] == row["item_receipt_sha256"]
                        and before[
                            row["item_receipt"].removesuffix("complete.json")
                            + "result.json"
                        ]
                        == row["result_sha256"],
                        "Old completed evidence changed",
                    )
                else:
                    q = pending[item]
                    a.need(
                        row["stage"] == q["stage"]
                        and row["item_completed_utc"] >= q["started_utc"],
                        "New output precedes its actual runtime attempt",
                    )
                workers[i, c, r] = w
                missing[i, c, r] = row["n_valid"]
                counts[str(w)] += 1
                models.update(row["reported_models"])
                classes.update(row["response_classes"])
                finish.update(row["finish_reasons"])
                compact = {
                    name: row[name]
                    for name in (
                        "input_index",
                        "key",
                        "population",
                        "stage",
                        "condition",
                        "repeat",
                        "item_receipt",
                        "item_receipt_sha256",
                        "result_sha256",
                        "item_completed_utc",
                        "n_valid",
                        "uncertain",
                        "repairs",
                        "parse_failures",
                        "reported_models",
                        "finish_reasons",
                        "response_classes",
                    )
                }
                compact.update(
                    effective_workers=w,
                    identity_hash=item,
                    prior_checkpoint_member=w == 8,
                    runtime_provenance=pending[item]["path"] if w == 128 else None,
                )
                emit(out, compact)
                if row["n_valid"] == 0:
                    nulls.append(compact)
                groupnames = [
                    f"all/workers_{w}",
                    f"{row['population']}/workers_{w}",
                    f"{meta[i]['source']}/gold_{int(meta[i]['gold_hate'])}/workers_{w}",
                    f"batch/{row['stage']}",
                ]
                for name in groupnames:
                    v = summary[name]
                    v["outputs"] += 1
                    v["valid_vote_counts"][str(row["n_valid"])] += 1
                    v["repairs"] += row["repairs"]
                    v["parse_failures"] += row["parse_failures"]
                    v["gate_ties"] += int(row["uncertain"])
                    for f in ("finish_reasons", "response_classes", "reported_models"):
                        v[f].update(row[f])
            print("Bound actual execution policy: " + path.stem, flush=True)
    a.need(
        np.all(np.isin(workers, [8, 128]))
        and dict(counts) == {"8": 31565, "128": 157531},
        "Runtime output totals",
    )
    eras = [k.era(w) for w in workers]
    a.need(
        Counter(eras) == Counter(all_8=2630, all_128=13127, cross_boundary=1),
        "Runtime input totals",
    )
    for i, m in enumerate(meta):
        inventory.append(
            {
                name: m[name]
                for name in (
                    "input_index",
                    "key",
                    "population",
                    "source",
                    "stage",
                    "execution_period",
                    "inference_cluster",
                    "gold_hate",
                    "masks",
                )
            }
            | dict(
                actual_execution_era=eras[i],
                effective_workers=workers[i].tolist(),
                valid_votes=missing[i].tolist(),
            )
        )
    crossing = [inventory[i] for i, e in enumerate(eras) if e == "cross_boundary"]
    a.need(
        crossing[0]["key"] in boundary["mixed_inputs"],
        "Cross-boundary identity differs",
    )
    with (OUT / "workers.npz").open("xb") as f:
        np.savez_compressed(f, workers=workers)
    with gzopen("input_execution.jsonl.gz") as f:
        for row in inventory:
            emit(f, row)
    save(
        "cross_boundary.json",
        dict(
            inputs=crossing,
            original_boundary_mixed_inputs=boundary["mixed_inputs"],
            note="Completed predictions are entirely old or entirely new; no partially saved prediction at the parallelism switch. One input has aggregate predictions from both runtime policies.",
        ),
    )
    save("null_outputs.json", nulls)
    save("execution_failure_groups.json", summary)
    save(
        "execution_boundary.json",
        dict(
            status="identity_bound_execution_reconstruction_complete",
            output_counts=counts,
            input_counts=Counter(eras),
            partial_predictions_at_switch=0,
            original_receipt_and_result_hash_pairs_matched=31565,
            actual_amendment=a.rel(M),
            actual_amendment_sha256=a.sha(M),
            original_workers=8,
            new_workers=128,
            actual_provenance_records=19,
            earlier_32_worker_manifest="Prepared historical configuration, not an observed 32-worker scientific era.",
            provenance=[
                {k: v for k, v in q.items() if k != "pending"}
                | dict(pending_outputs=len(q["pending"]))
                for q in provenance
            ],
            reported_models=models,
            response_classes=classes,
            finish_reasons=finish,
            original_scientific_plan_unchanged=True,
            diagnostic_requests_excluded=True,
            timestamps_used_to_assign_workers=False,
        ),
    )
    artifacts = {a.rel(path): a.sha(path) for path in OUT.iterdir() if path.is_file()}
    save(
        "assembled.json",
        dict(
            status="part7_runtime_and_missingness_assembled",
            created_utc=a.utc(),
            input_bindings=bindings,
            analysis_sources={a.rel(f): a.sha(f) for f in sources},
            artifacts=artifacts,
            actual_predictions=189096,
            inputs=15758,
            old_predictions=31565,
            new_predictions=157531,
            null_predictions=len(nulls),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "Part 7 execution boundary assembled without raw replay or calls.", flush=True
    )


if __name__ == "__main__":
    with a.offline():
        main()
