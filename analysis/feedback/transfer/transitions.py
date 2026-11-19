import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
from collections import Counter
from pathlib import Path
import gzip
import io
import json
import unittest
import numpy as np
from analysis.feedback.transfer import archive as a
from analysis.feedback.transfer import transitions_metrics as k

P2 = a.ROOT / "experiments/astra/sq3_transfer_part2_20261023_v1"
P4 = a.ROOT / "experiments/astra/sq3_transfer_part4_20261023_v1"
P5 = a.ROOT / "experiments/astra/sq3_transfer_part5_20261023_v1"
OUT = a.ROOT / "experiments/astra/sq3_transfer_part6_20261023_v1"
CONDITIONS = (
    "reference",
    "error_balanced_s2026100441",
    "error_balanced_s2026100442",
    "error_balanced_s2026100443",
)
VIEWS = (
    "main",
    "strict",
    "replay_excluded",
    "strict_replay_excluded",
    "representative",
    "representative_replay_excluded",
)


def native(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(key): native(value) for key, value in v.items()}
    if isinstance(v, (list, tuple)):
        return [native(x) for x in v]
    return v


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value if isinstance(value, bytes) else a.encoded(native(value), pretty=True)
        )


def gzopen(path):
    return io.TextIOWrapper(
        gzip.GzipFile(filename=str(path), mode="xb", mtime=0, compresslevel=5),
        encoding="utf8",
        newline="\n",
    )


def emit(stream, value):
    stream.write(
        json.dumps(
            native(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def rows(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def main():
    a.need(not OUT.exists(), "Preserve prior Part 6 work")
    OUT.mkdir()
    prior = a.read(P5 / "complete.json")
    a.need(
        a.read(P5 / "publication/final_check.json")["complete_sha256"]
        == a.sha(P5 / "complete.json"),
        "Current Part 5 checkpoint",
    )
    bindings = {
        **prior["input_bindings"],
        **prior["analysis_sources"],
        **prior["artifacts"],
    }
    for path in [P5 / "complete.json", P5 / "publication/final_check.json"]:
        bindings[a.rel(path)] = a.sha(path)
    for name, digest in bindings.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed prior evidence " + name)
    sources = [
        Path(__file__),
        Path(k.__file__),
        a.ROOT / "tests/feedback/transfer/test_transitions.py",
    ]
    save(
        "analysis_contract.json",
        dict(
            created_utc=a.utc(),
            input_bindings=bindings,
            analysis_sources={a.rel(p): a.sha(p) for p in sources},
            outcomes="Frozen dataset gold and Part 4 eligibility; unknown fields stay unscored. Binary and known-field exact are distinct. Complete output changes include unannotated legal fields and are not all-schema accuracy.",
            pairs="Three matched repeat-index comparisons per input and pool; reference repeats are reused across the three pools. Also compare all 3x3 repeat combinations and six global alignments. Pair labels are not common random seeds or causal individual effects.",
            consistency="A consistent fix requires all three reference aggregates wrong and all nine feedback aggregates correct; reverse for consistent regression. Persistent correct/wrong require agreement across all twelve. Missing outputs never prove consistency.",
            repeat_levels="Within-prediction votes and across-three-aggregate decoder repeats are separate. Canonical set equality ignores label ordering. Full five-field signatures describe agreement, not correctness.",
            masks="Use existing Part 2 endpoint masks and Part 4 head masks for all six views, selected before further transition summaries. Full/pairwise-available diagnostics retain missing counts separately. No new bootstrap family.",
            inference="All new counts, subgroups and cases are descriptive. Existing error and exact-score net effects reconcile with independently verified Parts 2 and 4. No causality claim for same-message versus different-message subsets or backend randomness.",
            cases="Mechanical slots chosen using minimum SHA256(A16-part6-cases-v1|slot|key), independently within each slot. Strict all-36-vote inputs only; empty slots stay empty. Include all eight outcome-independent Part 5 context examples separately. Overlapping chosen keys are deduplicated without substituting another case.",
            case_slots="Per population x gold polarity: consistent fix, consistent regression, persistently wrong. Per natural source: consistent fix/regression. Per population: opposing pool directions, equal average with offsetting changes, within-condition gate repeat variation, same feedback-message variation. Per scorable fine head: one-pool all-repeat exact fix/regression and binary fix with persistent joint known-field error. No output case is chosen by appealing prose.",
            preservation="Original model calls, prompts, predictions, gold, plans and earlier results immutable. Reuse completed raw audits; no redundant full raw replay.",
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.feedback.transfer.test_transitions"
    )
    test = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    save(
        "focused_tests.json",
        dict(tests_run=test.testsRun, passed=test.wasSuccessful(), log=log.getvalue()),
    )
    a.need(test.wasSuccessful(), "Boundary tests failed")
    matrix = list(a.rows(a.OUT / "matrix.jsonl"))
    meta = list(a.rows(P4 / "metadata.jsonl"))
    N = len(matrix)
    H = len(k.HEADS)
    a.need(
        N == 15758
        and all(
            r["input_index"] == i and meta[i]["key"] == r["key"]
            for i, r in enumerate(matrix)
        ),
        "Matrix identity",
    )
    codes = np.full((N, 4, 3, 5), k.MISSING, dtype=np.int16)
    scores = np.full((N, 4, 3, H), -2, dtype=np.int8)
    nv = np.zeros((N, 4, 3), dtype=np.int8)
    for i, r in enumerate(matrix):
        for c in range(4):
            for rep in range(3):
                value = r["predictions"][c][rep]
                codes[i, c, rep] = k.code(value["result"])
                scores[i, c, rep] = k.correctness(meta[i], value["result"])
                nv[i, c, rep] = value["n_valid"]
    with np.load(P4 / "prediction_codes.npz", allow_pickle=False) as f:
        a.need(np.array_equal(nv, f["votes"]), "Vote counts differ from Part 4")
        for fkey, j in [
            ("gates", 0),
            ("hate_type", 1),
            ("target_group", 2),
            ("severity", 3),
        ]:
            old = f[fkey]
            a.need(
                np.array_equal(codes[..., j][nv > 0], old[nv > 0]),
                "Valid code differs " + fkey,
            )
    with np.load(P4 / "features.npz", allow_pickle=False) as f:
        schema = a.read(P4 / "feature_schema.json")["features"]
        for j, head in enumerate(k.HEADS):
            base = head.removeprefix("gated_")
            column = "gated_exact" if head.startswith("gated_") else "exact"
            if head == "binary":
                continue
            feature = f[base][..., schema[base].index(column)]
            a.need(
                np.array_equal(
                    feature[scores[..., j] >= 0], scores[..., j][scores[..., j] >= 0]
                ),
                "Exactness differs " + head,
            )
    transition = np.where(
        (scores[:, 0, None] < 0) | (scores[:, 1:] < 0),
        np.where((scores[:, 0, None] == -2) | (scores[:, 1:] == -2), -2, -1),
        2 * scores[:, 0, None] + scores[:, 1:],
    ).astype(np.int8)
    pairvalid = (nv[:, 0, None] > 0) & (nv[:, 1:] > 0)
    changed = (codes[:, 0, None] != codes[:, 1:]).astype(np.int8)
    changed = np.concatenate(
        [changed, changed.any(axis=-1, keepdims=True).astype(np.int8)], axis=-1
    )
    changed[~pairvalid] = -1

    ctx = [
        {
            name: value
            for name, value in r.items()
            if name not in ("ordinary_hits", "ordinary_cosines", "text")
        }
        for r in a.rows(P5 / "query_contexts.jsonl")
    ]
    delivery = list(a.rows(P5 / "delivery.jsonl"))
    messages = np.empty((N, 4), dtype="U64")
    for i in range(N):
        messages[i, 0] = ctx[i]["reference_messages_sha256"]
        for c in range(1, 4):
            d = delivery[i * 3 + c - 1]
            a.need((d["input_index"], d["pool_index"]) == (i, c), "Delivery order")
            messages[i, c] = d["messages_sha256"]
    votes = np.full((N, 4, 3, 3, 5), k.MISSING, dtype=np.int16)
    distinct = np.zeros((N, 4, 3, 6), dtype=np.int8)
    vote_pairs = np.zeros_like(distinct)
    unseen = np.zeros((N, 4, 3), dtype=bool)
    ties = np.zeros_like(unseen)
    seen = np.zeros_like(unseen)
    with gzopen(OUT / "prediction_provenance.jsonl.gz") as out:
        for path in sorted((a.OUT / "assembly").glob("transfer_*.jsonl")):
            for r in a.rows(path):
                i, c, rep = (
                    r["input_index"],
                    CONDITIONS.index(r["condition"]),
                    r["repeat"],
                )
                a.need(not seen[i, c, rep], "Duplicate actual prediction")
                seen[i, c, rep] = True
                a.need(
                    k.code(r["result"]) == tuple(codes[i, c, rep])
                    and r["messages_sha256"] == messages[i, c]
                    and r["n_valid"] == nv[i, c, rep],
                    "Original prediction binding",
                )
                votes[i, c, rep] = [k.code(v) for v in r["votes"]]
                v = k.vote_stats(r["votes"], r["result"])
                distinct[i, c, rep] = v["distinct"]
                vote_pairs[i, c, rep] = v["disagree_pairs"]
                unseen[i, c, rep] = v["aggregate_not_observed"]
                ties[i, c, rep] = r["uncertain"]
                a.need(v["n_valid"] == r["n_valid"], "Actual vote count")
                emit(
                    out,
                    {
                        name: r[name]
                        for name in (
                            "input_index",
                            "key",
                            "condition",
                            "repeat",
                            "stage",
                            "execution_period",
                            "messages_sha256",
                            "item_receipt",
                            "item_receipt_sha256",
                            "result_sha256",
                            "n_valid",
                            "uncertain",
                        )
                    },
                )
            print("Loaded saved vote structure: " + path.stem, flush=True)
    a.need(seen.all(), "Actual prediction missing")
    with (OUT / "arrays.npz").open("xb") as f:
        np.savez_compressed(
            f,
            codes=codes,
            scores=scores,
            n_valid=nv,
            transitions=transition,
            changed=changed,
            messages=messages,
            votes=votes,
            vote_distinct=distinct,
            vote_disagree_pairs=vote_pairs,
            aggregate_not_observed=unseen,
            gate_ties=ties,
        )
    paircounts = Counter()
    profiles = []
    with gzopen(OUT / "transition_pairs.jsonl.gz") as out, gzopen(
        OUT / "per_text.jsonl.gz"
    ) as pertext:
        for i, r in enumerate(matrix):
            for c in range(1, 4):
                d = delivery[i * 3 + c - 1]
                for rep in range(3):
                    status = (
                        "valid"
                        if pairvalid[i, c - 1, rep]
                        else (
                            "both_missing"
                            if nv[i, 0, rep] == nv[i, c, rep] == 0
                            else (
                                "reference_missing"
                                if nv[i, 0, rep] == 0
                                else "feedback_missing"
                            )
                        )
                    )
                    paircounts[status] += 1
                    emit(
                        out,
                        dict(
                            input_index=i,
                            key=r["key"],
                            pool_index=c,
                            repeat=rep,
                            status=status,
                            transitions={
                                head: int(transition[i, c - 1, rep, j])
                                for j, head in enumerate(k.HEADS)
                            },
                            changed={
                                field: int(changed[i, c - 1, rep, j])
                                for j, field in enumerate((*k.FIELDS, "full_output"))
                            },
                            valid_votes=[int(nv[i, 0, rep]), int(nv[i, c, rep])],
                            same_messages=messages[i, 0] == messages[i, c],
                            source_id=d["source_id"],
                            reference_messages_sha256=messages[i, 0],
                            feedback_messages_sha256=messages[i, c],
                        ),
                    )
            prof = {
                head: k.profile(scores[i, ..., j].tolist())
                for j, head in enumerate(k.HEADS)
            }
            profiles.append(prof)
            emit(
                pertext,
                dict(
                    input_index=i,
                    key=r["key"],
                    population=r["population"],
                    source=r["source"],
                    gold_hate=r["gold_hate"],
                    inference_cluster=r["inference_cluster"],
                    masks=r["masks"],
                    profiles=prof,
                    binary_repeat_pattern=[
                        [
                            None if nv[i, c, rp] == 0 else int(codes[i, c, rp, 0])
                            for rp in range(3)
                        ]
                        for c in range(4)
                    ],
                    correction_sources=[
                        delivery[i * 3 + c]["source_id"] for c in range(3)
                    ],
                    correction_gates=[
                        delivery[i * 3 + c]["source_gate"] for c in range(3)
                    ],
                    context_reference="sq3_transfer_part5_20261023_v1/query_contexts.jsonl by input_index",
                    predictions_reference="sq3_transfer_part1_20261023_v1/matrix.jsonl by input_index",
                ),
            )
    save(
        "array_schema.json",
        dict(
            conditions=CONDITIONS,
            fields=k.FIELDS,
            heads=k.HEADS,
            transitions=dict(enumerate(k.TRANSITIONS)),
            unscorable_transition=-2,
            missing_transition=-1,
            score_codes={
                "unknown_gold": -2,
                "missing_output": -1,
                "incorrect": 0,
                "correct": 1,
            },
            prediction_codes=dict(
                missing=k.MISSING,
                labels=k.LABELS,
                severity={"none": -1, "low": 0, "medium": 1, "high": 2},
            ),
            shape=[N, 4, 3],
            vote_shape=[N, 4, 3, 3],
            pair_shape=[N, 3, 3],
            field_order=[
                "hate",
                "hate_type",
                "target_group",
                "severity",
                "legal",
                "full_output",
            ],
            note="All repeat comparisons are dependent. Missing slots are retained; canonical set order does not create a change. Exact fine and gated exact have separate heads.",
        ),
    )
    files = [path for path in OUT.iterdir() if path.is_file()]
    save(
        "assembled.json",
        dict(
            status="part6_transitions_assembled_independent_verification_pending",
            created_utc=a.utc(),
            input_bindings=bindings,
            analysis_sources={a.rel(p): a.sha(p) for p in sources},
            artifacts={a.rel(p): a.sha(p) for p in files},
            inputs=N,
            actual_predictions=int(seen.sum()),
            valid_votes=int(nv.sum()),
            pair_counts=dict(paircounts),
            all_pairs=sum(paircounts.values()),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "Assembled all text/pool/repeat transitions and saved votes; original evidence unchanged.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()
