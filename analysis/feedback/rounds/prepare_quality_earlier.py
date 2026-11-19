from collections import Counter
from pathlib import Path
import io
import json
import unittest
import numpy as np
from analysis.feedback.rounds import quality_inputs as a
from analysis.feedback.rounds import quality_metrics as q

k = a.quality


def freeze(matrix):
    a.need(not a.OUT.exists(), "Preserve existing Part 6 files")
    done = a.read(a.PART5 / "final_verification.json")
    a.need(
        done["status"] == "iterative_part5_complete_with_final_readback"
        and done["complete_sha256"] == a.sha(a.PART5 / "complete.json")
        and done["readback_sha256"] == a.sha(a.PART5 / "publication_v1/readback.json"),
        "Part 5 checkpoint changed",
    )
    reader = a.read(a.READER / "complete.json")
    a.need(
        reader["artifacts"][a.rel(a.READER / "matrix.jsonl")]
        == a.sha(a.READER / "matrix.jsonl"),
        "Verified matrix changed",
    )
    from tests.feedback.rounds import test_quality as tests

    log = io.StringIO()
    tested = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    a.need(tested.wasSuccessful(), "Quality fixtures failed: " + log.getvalue())
    support = {}
    labels = {}
    for pop in a.POPS:
        support[pop] = {}
        labels[pop] = {"binary": k.LABELS["binary"], "known_answer": []}
        for head, field in k.GOLD.items():
            ids = [
                i
                for i, r in enumerate(matrix)
                if r["population"] == pop and k.eligible(r, head)
            ]
            count = {
                lab: sum(
                    (
                        matrix[i][field] == lab
                        if head == "severity"
                        else lab in matrix[i][field]
                    )
                    for i in ids
                )
                for lab in k.LABELS[head]
            }
            support[pop][head] = dict(
                known_eligible=len(ids),
                known_empty=sum(matrix[i][field] == [] for i in ids),
                positive_counts=count,
            )
            labels[pop][head] = [lab for lab in k.LABELS[head] if count[lab] >= 10]
    paths = [
        a.STUDY / "evaluation.jsonl",
        a.STUDY / "STUDY_PROTOCOL.md",
        a.STUDY / "plan.json",
        a.REVIEW / "review_plan.json",
        a.READER / "matrix.jsonl",
        a.READER / "complete.json",
        a.READER / "independent_verification.json",
        a.PART5 / "complete.json",
        a.PART5 / "final_verification.json",
        a.PART5 / "publication_v1/readback.json",
        a.PART5 / "metadata.jsonl",
        a.PART5 / "analysis.json",
        a.PART2 / "errors_v1/results.json",
        a.PART3 / "assembled.json",
        a.PART3 / "prompt_bindings.jsonl",
        a.ROOT
        / "experiments/astra/later_evaluation_20261008_v1/scoring_contract_v1.json",
    ]
    sources = [
        Path(__file__),
        Path(a.__file__),
        Path(q.__file__),
        Path(k.__file__),
        Path(tests.__file__),
    ]
    contract = dict(
        status="part6_rules_frozen_before_new_quality_scoring",
        created_utc=a.pure.utc(),
        timing="After generation and Parts 2–5 binary/context/durability findings; known-gold coverage inspected, before new fine-quality scoring. Exploratory, not a new prospective study.",
        input_bindings={a.rel(p): a.sha(p) for p in paths},
        source_hashes={a.rel(p): a.sha(p) for p in sources},
        labels=k.LABELS,
        heads=a.HEADS,
        full_roster_support=support,
        macro_labels=labels,
        minimum_full_roster_support=10,
        label_rule="Use full iterative population roster support >=10, frozen across views, sources, profiles and draws. Retain every taxonomy label in per-label scores, micro/exact metrics and full-taxonomy-macro descriptive sensitivity. The selected macro set may differ from the earlier full-transfer population.",
        gold_rule="Fine heads require known gold and gold-hate, legal independent. Unknown is not empty. Known-empty labels stay scored; exact versus gate-and-exact are distinct. No eligible legal gold in either population.",
        joint="Known-field exact means correct gate and every available eligible fine field. It does not certify unknown fields or explanations. Separate all known-gold profiles and the subset with any fine gold; no text has all three fine heads.",
        averaging="Score each actual round/condition/decoder-repeat run separately. Mean three repeats and three streams per policy; reference has three actual decisions. F1 is not computed from pooled repeated predictions.",
        views="Population: fixed all-round/all-arm common, final all-arm common, final strict, final replay-excluded and one-per-text-cluster representative, plus four round-specific masks. Main fixed/final/strict have intervals. Source fixed/final/strict and annotation-profile fixed/final/strict; final intervals. All hateful HateCheck target functions fixed/final point views. Final all-gates-positive fine subsets are descriptive outcome-selected points.",
        representatives="Before availability, minimum SHA256(A17-part6-representative|key) per eligible population/head, gold polarity and normalized-text cluster. This is one-per-text-cluster, not one-per-inference-cluster.",
        uncertainty=dict(
            draws=10000,
            generator="PCG64",
            level=0.95,
            seed_prefix="A17-part6-quality-v1|",
            unit="Whole sorted inherited inference clusters, same multiplicity for all four rounds and actual runs; score weighted run counts before policy means.",
            interpretation="Supporting nominal intervals only; no new discovery family and no change to Part 2 primary inference. Conditional on saved outputs and overlapping streams. Undefined draws are counted; an interval requires all draws finite.",
        ),
        missing="Null predictions remain absent. Own-valid run scores and final full-roster opposing correctness bounds accompany population/source/profile final contrasts. Strict filtering changes support. No ordinal distance assigned to missing severity.",
        diagnostics="Every eligible label TP/FP/FN, precision/recall/F1; exact/contains/extra-only/missing-only/both and cardinalities; severity none/low/medium/high matrix and conditional ordinal error. Final policy/reference and adaptive/control gate partitions; adjacent current-policy transitions and fixed-cohort exact paths. These are dependent paired counts, not causal mediation.",
        cases="For each of English type/target/severity and HateCheck target: hash-first fixed-cohort text in retained current/reference mean exact benefit, fine-only regression with both gates positive, and binary-gate fix without exact fine fix. Also hash-first known-empty gold text per scored set head. For each population known-answer: hash-first average final benefit and regression among texts with any fine gold. Keep empty slots; do not select on natural-language appeal.",
        preservation="All source labels, saved predictions, votes, original prompts, protocols and prior analyses stay unchanged. No generation, embeddings, training, SQ2 or writer-owned edits.",
        new_Qwen_calls=0,
        new_fits=0,
        new_embeddings=0,
    )
    a.write(a.OUT / "analysis_contract.json", contract)
    a.write(
        a.OUT / "tests.json",
        dict(passed=True, tests_run=tested.testsRun, log=log.getvalue()),
    )
    return contract


def main():
    matrix = list(a.rows(a.READER / "matrix.jsonl"))
    a.need(len(matrix) == 2077, "Roster size changed")
    contract = freeze(matrix)
    original = list(a.rows(a.STUDY / "evaluation.jsonl"))
    meta = [a.metadata(r) for r in matrix]
    shape = (2077, 4, 10, 3)
    features = {h: np.zeros((*shape, len(k.names(h))), dtype=np.int16) for h in a.HEADS}
    codes = {
        h: np.full(shape, -2 if h == "severity" else -1, dtype=np.int16)
        for h in ["gates", "hate_type", "target_group", "severity"]
    }
    codes["votes"] = np.zeros(shape, dtype=np.int8)
    for i, (r, raw) in enumerate(zip(matrix, original, strict=True)):
        a.need(
            r["input_index"] == i and r["key"] == raw["key"], "Input identity changed"
        )
        for head, field in k.GOLD.items():
            a.need(
                r[field]
                == raw[
                    {
                        "gold_hate_types": "hate_types",
                        "gold_target_groups": "target_groups",
                        "gold_severity": "severity",
                        "gold_legal": "legal",
                    }[field]
                ],
                "Original gold differs",
            )
        meta[i]["functionality"] = (
            json.loads(raw["raw"])["functionality"]
            if raw["source"] == "hatecheck"
            else None
        )
        for rnd in range(4):
            for ci in range(10):
                for rep, out in enumerate(r["predictions"][rnd][ci]):
                    pred = out["result"]
                    codes["votes"][i, rnd, ci, rep] = out["n_valid"]
                    if pred is not None:
                        codes["gates"][i, rnd, ci, rep] = pred["hate"]
                        for head in ["hate_type", "target_group"]:
                            a.need(
                                len(pred[head]) == len(set(pred[head]))
                                and set(pred[head]) <= set(k.LABELS[head]),
                                "Final voted set differs",
                            )
                            codes[head][i, rnd, ci, rep] = sum(
                                1 << k.LABELS[head].index(lab) for lab in pred[head]
                            )
                        codes["severity"][i, rnd, ci, rep] = (
                            -1
                            if pred["severity"] is None
                            else k.LABELS["severity"].index(pred["severity"])
                        )
                        if not pred["hate"]:
                            a.need(
                                pred["hate_type"] == []
                                and pred["target_group"] == []
                                and pred["severity"] is None,
                                "False gate not normalised",
                            )
                    for head in a.HEADS:
                        features[head][i, rnd, ci, rep] = k.features(r, pred, head)
        a.need(
            bool(np.all(codes["gates"][i] >= 0)) == r["masks"]["all_round_common"],
            "Fixed mask differs",
        )
    valid = codes["gates"] >= 0
    a.need(
        int((~valid).sum()) == 241 and np.array_equal(~valid, codes["votes"] == 0),
        "Null preservation differs",
    )
    a.lines(a.OUT / "metadata.jsonl", meta)
    for name, values in [("features.npz", features), ("prediction_codes.npz", codes)]:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **values)
        a.write(a.OUT / name, buffer.getvalue())
    a.write(
        a.OUT / "feature_schema.json",
        dict(
            features={h: k.names(h) for h in a.HEADS},
            metrics={h: k.metric_names(h) for h in a.HEADS},
            axes=["input", "round", "actual_condition", "decoder_repeat", "feature"],
            conditions=a.CONDITIONS,
            policies=a.POLICIES,
        ),
    )
    results = {}
    groups = {}
    draw_sets = 0

    def evaluate(group, pop, head, members, view, boot=False, kind="population"):
        nonlocal draw_sets
        name = f"{group}|{head}|{view}"
        a.need(name not in results, "Duplicate view")
        groups[group] = dict(population=pop, kind=kind, members=members)
        eligible, ids = q.selected(meta, members, head, view)
        rnds = (
            [1, 2, 3, 4]
            if view == "fixed"
            else [int(view[-1])] if view.startswith("round") else [4]
        )
        cols = q.columns(rnds)
        labels = contract["macro_labels"][pop][head]
        indices = [k.LABELS[head].index(v) for v in labels]
        r = dict(
            group=group,
            population=pop,
            head=head,
            view=view,
            kind=kind,
            eligible_indices=eligible,
            indices=ids,
            eligible_n=len(eligible),
            n=len(ids),
            macro_labels=labels,
            macro_indices=indices,
            columns=cols,
            column_order=list(cols),
            metrics=k.metric_names(head) if head in a.HEADS else [],
            status=(
                "defined"
                if ids
                else (
                    "unscorable_no_known_gold"
                    if not eligible
                    else "undefined_no_valid_outputs"
                )
            ),
        )
        results[name] = r
        if not ids:
            return
        data = features[head]
        count = data[ids].sum(axis=0, dtype=np.int64)
        scores = k.score(count, head, indices)
        est = q.summarize(scores, cols)
        r.update(
            counts=count,
            run_scores=scores,
            estimates=est,
            clusters=len({meta[i]["inference_cluster"] for i in ids}),
            profiles=dict(Counter(meta[i]["gold_profile"] for i in ids)),
            own_valid=q.summarize(
                k.score(data[eligible].sum(axis=0, dtype=np.int64), head, indices), cols
            ),
            all_taxonomy_macro_descriptive=(
                q.summarize(
                    k.score(count, head, list(range(len(k.LABELS[head])))), cols
                )[:, 0]
                if head in ["hate_type", "target_group", "severity"]
                else None
            ),
        )
        if view == "strict":
            mask = (codes["votes"][eligible] == 3)[..., None]
            r["own_valid"] = q.summarize(
                k.score(
                    (data[eligible] * mask).sum(axis=0, dtype=np.int64), head, indices
                ),
                cols,
            )
        if view == "final" and kind in ["population", "source", "profile"]:
            r["missing_bounds"] = q.missing_bounds(
                meta, data, eligible, valid, head, indices
            )
        if boot:
            draw, labels_ = q.boot(meta, data, ids, head, indices, cols, a.seed(name))
            path = a.OUT / "draws" / f"{a.seed(name):016x}.npz"
            buffer = io.BytesIO()
            np.savez_compressed(buffer, draws=draw)
            a.write(path, buffer.getvalue())
            draw_sets += 1
            finite = np.isfinite(draw).sum(axis=0)
            intervals = np.full((*est.shape, 2), np.nan)
            for j in range(est.shape[0]):
                for z in range(est.shape[1]):
                    if finite[j, z] == 10000:
                        intervals[j, z] = np.quantile(draw[:, j, z], [0.025, 0.975])
            r.update(
                draw_file=a.rel(path),
                draw_sha256=a.sha(path),
                seed=a.seed(name),
                cluster_labels=labels_,
                finite_draws=finite,
                intervals=intervals,
            )
        print(
            f"Quality {name}: {len(ids)}/{len(eligible)} texts"
            + ("; cluster intervals saved" if boot else ""),
            flush=True,
        )

    for pop in a.POPS:
        members = [i for i, m in enumerate(meta) if m["population"] == pop]
        for head in [*a.HEADS, "legal"]:
            for view in [
                "fixed",
                "final",
                "strict",
                "replay_excluded",
                "representative",
                *[f"round{r}" for r in range(1, 5)],
            ]:
                evaluate(
                    "population/" + pop,
                    pop,
                    head,
                    members,
                    view,
                    boot=view in ["fixed", "final", "strict"],
                )
        for profile in sorted({meta[i]["gold_profile"] for i in members}):
            selected = [i for i in members if meta[i]["gold_profile"] == profile]
            for view in ["fixed", "final", "strict"]:
                evaluate(
                    "profile/" + pop + "/" + profile,
                    pop,
                    "known_answer",
                    selected,
                    view,
                    boot=view == "final",
                    kind="profile",
                )
        anyfine = [i for i in members if meta[i]["any_fine_gold"]]
        for view in ["fixed", "final", "strict"]:
            evaluate(
                "profile/" + pop + "/any_fine",
                pop,
                "known_answer",
                anyfine,
                view,
                boot=view == "final",
                kind="profile",
            )
        for head in ["hate_type", "target_group", "severity"]:
            selected = [i for i in members if np.all(codes["gates"][i, 3] == 1)]
            evaluate(
                "all_gates_positive/" + pop,
                pop,
                head,
                selected,
                "final",
                kind="conditional",
            )
    for source in ["hatexplain", "implicit_hate", "mhs"]:
        ids = [i for i, m in enumerate(meta) if m["source"] == source]
        for head in a.HEADS:
            for view in ["fixed", "final", "strict"]:
                evaluate(
                    "source/" + source,
                    "en_test",
                    head,
                    ids,
                    view,
                    boot=view == "final",
                    kind="source",
                )
    for fn in sorted({m["functionality"] for m in meta if m["source"] == "hatecheck"}):
        ids = [i for i, m in enumerate(meta) if m["functionality"] == fn]
        for view in ["fixed", "final"]:
            evaluate(
                "function/" + fn,
                "en_hatecheck_eval",
                "target_group",
                ids,
                view,
                kind="function",
            )
    a.write(a.OUT / "groups.json", groups)
    a.write(
        a.OUT / "analysis.json",
        dict(
            status="part6_scored_pending_independent_verification",
            created_utc=a.pure.utc(),
            records=results,
            records_count=len(results),
            bootstrap_sets=draw_sets,
            evaluation_inputs=2077,
            evaluation_predictions=249240,
            evaluation_nulls=241,
            full_gold_profiles=dict(
                Counter(m["population"] + "|" + m["gold_profile"] for m in meta)
            ),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    files = [
        "metadata.jsonl",
        "features.npz",
        "prediction_codes.npz",
        "feature_schema.json",
        "analysis.json",
        "groups.json",
        "tests.json",
        "analysis_contract.json",
    ]
    a.write(
        a.OUT / "assembled.json",
        dict(
            status="part6_calculated_pending_independent_verification",
            created_utc=a.pure.utc(),
            input_bindings=contract["input_bindings"],
            source_hashes=contract["source_hashes"],
            artifacts={a.rel(a.OUT / f): a.sha(a.OUT / f) for f in files},
            bootstrap_sets=draw_sets,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        f"Part 6 quality matrix scored: {len(results)} views, {draw_sets} bootstrap sets. Independent verification next.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()
