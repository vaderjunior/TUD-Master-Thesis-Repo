from collections import Counter, defaultdict
import io
from itertools import product
from pathlib import Path
import time
import unittest
import numpy as np
from analysis.feedback.rounds import trajectories_inputs as a
from analysis.feedback.rounds import trajectories_metrics as k


def build():
    start = time.monotonic()
    a.need(not a.OUT.exists(), "Preserve an existing Part 4 directory")
    release = a.read(a.PART3 / "final_verification.json")
    a.need(
        release["status"] == "iterative_part3_complete_with_final_readback"
        and release["complete_sha256"] == a.sha(a.PART3 / "complete.json")
        and release["readback_sha256"]
        == a.sha(a.PART3 / "publication_v1/readback.json"),
        "Part 3 final readback is required",
    )
    reader = a.read(a.READER / "complete.json")
    reader_proof = a.read(a.READER / "independent_verification.json")
    a.need(
        reader_proof["status"] == "independent_reader_verification_pass"
        and reader_proof["reader_complete_sha256"] == a.sha(a.READER / "complete.json"),
        "Current reader proof missing",
    )
    authoritative = {
        **reader["artifacts"],
        **a.read(a.PART3 / "assembled.json")["artifacts"],
        **a.read(a.PART3 / "assembled.json")["input_bindings"],
    }
    inputs = [
        a.PART3 / "final_verification.json",
        a.PART3 / "complete.json",
        a.PART3 / "publication_v1/readback.json",
        a.PART3 / "analysis.json",
        a.PART3 / "context_independent_verification.json",
        a.PART3 / "numerical_independent_verification.json",
        a.PART3 / "deliveries.jsonl",
        a.PART3 / "prompt_bindings.jsonl",
        a.READER / "matrix.jsonl",
        a.READER / "complete.json",
        a.READER / "independent_verification.json",
        a.PART2 / "errors_v1/results.json",
        a.PART2 / "errors_v1/independent_verification.json",
        a.REVIEW / "review_plan.json",
        a.STUDY / "STUDY_PROTOCOL.md",
        a.STUDY / "plan.json",
        a.STUDY / "evaluation.jsonl",
    ]
    bindings = {a.rel(p): a.sha(p) for p in inputs}
    for p, h in bindings.items():
        a.need(
            p not in authoritative or h == authoritative[p],
            "Changed verified input: " + p,
        )
    from tests.feedback.rounds import test_trajectories as tests

    log = io.StringIO()
    tested = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    a.need(tested.wasSuccessful(), log.getvalue())
    contract = dict(
        status="durability_conventions_before_part4_scoring",
        created_utc=a.pure.utc(),
        timing="After Parts 2 and 3 outcomes are known, before computing individual longitudinal results. This is a supporting descriptive review, not new prospective experiment design.",
        scope="Binary correctness only; keep fixed gold, all policies, all rounds, all streams/repeats, reference and nulls. Fine-answer quality remains Part 6.",
        main_cohort="Part 2 all-round all-arm common-valid mask. Adjacent-pair masks intersect the two all-arm round masks. Same-round reference comparisons also retain per-round common masks.",
        units="A text has nine treatment final decisions per policy/round (three source-order streams x three decoder repeats), or three actual reference decisions. Each final decision is an inherited vote aggregate. Dependent slots are never independent texts.",
        paired_paths="Five error bits: actual round-one reference repeat followed by treatment rounds 1,2,3,4 with matching repeat indices. The one reference repeat is reused for each stream. Reference control paths duplicate their first point only to share the schema; their round-one event cohort is not compared.",
        reference_control="Verify unchanged reference prompt hashes across rounds. Report adjacent transition rates for its three actual repeats; do not multiply its observation count to match treatments.",
        events="A fix is a wrong-to-correct paired saved-decision transition; a new error is correct-to-wrong. Count unique affected texts separately. Each transition cohort partitions into uninterrupted through the horizon, returned to the post-event state by round four after reversal, or the opposite state at round four. Round-four events have no follow-up.",
        strong_text_fix="Initial reference wrong in all three saved decisions, feedback correct in all nine in round one. A durable observed fix remains correct in all nine decisions in all later rounds. Any later wrong decision is a reversal; final recovery versus final loss remain separate. This is observed consistency, not a guarantee of future correctness.",
        new_error="A sustained new-error text starts with all three initial reference decisions correct and has all nine treatment decisions wrong for an uninterrupted suffix beginning no later than round three. Late-only round-four errors cannot establish persistence.",
        average_advantage="Keep integer treatment-error count minus three times reference-error count per input and round. Better/equal/worse paths are separate from strong fixes. Show both contemporaneous reference and fixed round-one reference to expose moving-control effects.",
        pairing_stress="All 216 permutations of repeat indices in rounds 2–4 within each stream, holding the round-one pairing fixed. Each permutation is applied to every eligible text in that stream. Report exact initial-fix retention envelopes; each envelope endpoint can use a different coherent alignment. Also enumerate six repeat pairings for each adjacent pair/stream. These are sensitivity ranges, not confidence intervals; initial reference/treatment pairing remains a limitation.",
        context="Join the verified Part 3 delivery identity/hash. Partition adjacent treatment changes into same prompt, newly added correction, or another old correction. Same-prompt flips show response variability and do not identify its cause.",
        case_rule="For each of two populations and eight declared adaptive-path categories, take the smallest SHA256(A17-part4-case-v1|population|slot|key) among fixed-cohort eligible texts. Preserve empty slots. Cases are outcome-informed illustrations selected by a fixed rule, not prevalence estimates or causal proofs.",
        case_slots=list(k.CASE_SLOTS),
        inference="No new bootstrap or confirmatory family. Exact descriptive path counts and denominators; Part 2 family-adjusted error-rate results remain authoritative. All selected case prompts/results/votes retain provenance.",
        source_hashes={
            a.rel(Path(f)): a.sha(Path(f))
            for f in [__file__, a.__file__, k.__file__, tests.__file__]
        },
        input_bindings=bindings,
        new_Qwen_calls=0,
        new_fits=0,
        new_embeddings=0,
    )
    a.write(a.OUT / "analysis_contract.json", contract)
    a.write(
        a.OUT / "tests.json",
        dict(passed=True, tests_run=tested.testsRun, output=log.getvalue()),
    )
    matrix = list(a.rows(a.READER / "matrix.jsonl"))
    N = len(matrix)
    error = np.full((N, 4, 10, 3), -1, dtype=np.int8)
    for i, row in enumerate(matrix):
        a.need(i == row["input_index"], "Reader input ordering changed")
        for rnd, conditions in enumerate(row["predictions"]):
            for col, repeats in enumerate(conditions):
                for rep, p in enumerate(repeats):
                    if p["result"] is not None:
                        error[i, rnd, col, rep] = (
                            p["result"]["hate"] != row["gold_hate"]
                        )
    fixed = (error >= 0).all(axis=(1, 2, 3))
    round_common = (error >= 0).all(axis=(2, 3))
    a.need(
        int((error < 0).sum()) == 241 and N == 2077,
        "Evaluation null/roster counts differ",
    )
    for i, row in enumerate(matrix):
        a.need(
            row["masks"]["all_round_common"] == bool(fixed[i])
            and row["masks"]["round_common"] == round_common[i].tolist(),
            "Inherited masks differ",
        )
    arrays = {p: a.policy_array(error, p) for p in ["reference", *a.POLICIES]}
    reference = arrays["reference"]
    deliveries = [
        d for d in a.rows(a.PART3 / "deliveries.jsonl") if d["role"] == "evaluation"
    ]
    contexts = {
        (d["input_index"], d["round"], d["policy"], d["stream"]): d for d in deliveries
    }
    refs = {
        (p["key"], int(p["phase"][1])): p["messages_sha256"]
        for p in a.rows(a.PART3 / "prompt_bindings.jsonl")
        if p["condition"] == "reference" and p["phase"].endswith("_evaluation")
    }
    for row in matrix:
        a.need(
            len({refs[row["key"], r] for r in range(1, 5)}) == 1,
            "Reference messages changed across rounds",
        )
    input_rows = []
    profiles = {}
    case_candidates = defaultdict(list)
    for i, row in enumerate(matrix):
        entry = {
            f: row[f]
            for f in [
                "input_index",
                "key",
                "population",
                "source",
                "gold_hate",
                "inference_cluster",
                "functional_cluster",
                "text_cluster",
                "overlap_cluster",
                "masks",
            ]
        }
        entry.update(reference_errors_by_repeat=reference[i].tolist(), policies={})
        for p in a.POLICIES:
            witnesses = []
            chosen = {str(s): [] for s in a.STREAMS}
            for sidx, stream in enumerate(a.STREAMS):
                for rnd in range(1, 5):
                    d = contexts[i, rnd, p, stream]
                    chosen[str(stream)].append(
                        {
                            f: d[f]
                            for f in [
                                "source_id",
                                "source_gate",
                                "added_round",
                                "messages_sha256",
                                "removed_id",
                                "removed_gate",
                            ]
                        }
                    )
                    if rnd > 1:
                        old = contexts[i, rnd - 1, p, stream]
                        if old["source_id"] == d["source_id"]:
                            a.need(
                                old["messages_sha256"] == d["messages_sha256"],
                                "Same source changed prompt",
                            )
                        if old["messages_sha256"] == d["messages_sha256"]:
                            for rep in range(3):
                                before, after = map(
                                    int,
                                    [
                                        arrays[p][i, rnd - 2, 3 * sidx + rep],
                                        arrays[p][i, rnd - 1, 3 * sidx + rep],
                                    ],
                                )
                                if min(before, after) >= 0 and before != after:
                                    witnesses.append(
                                        dict(
                                            round=rnd,
                                            stream=stream,
                                            repeat=rep,
                                            old_error=before,
                                            new_error=after,
                                            source_id=d["source_id"],
                                            messages_sha256=d["messages_sha256"],
                                        )
                                    )
            profile = k.input_profile(arrays[p][i], reference[i])
            profiles[i, p] = profile
            entry["policies"][p] = dict(
                errors_by_repeat=arrays[p][i].tolist(),
                profile=profile,
                contexts=chosen,
                same_prompt_flip_witnesses=witnesses,
            )
            if p == "adaptive" and fixed[i]:
                for slot in k.candidate_cases(profile, bool(witnesses)):
                    case_candidates[row["population"], slot].append(
                        (k.case_rank(row["population"], slot, row["key"]), i)
                    )
        input_rows.append(entry)
    a.lines(a.OUT / "input_trajectories.jsonl", input_rows)
    cases = []
    for pop in a.POPS:
        for slot in k.CASE_SLOTS:
            candidates = sorted(case_candidates[pop, slot])
            chosen = candidates[0] if candidates else None
            cases.append(
                dict(
                    population=pop,
                    slot=slot,
                    candidates=len(candidates),
                    status="selected" if chosen else "empty",
                    input_index=chosen[1] if chosen else None,
                    key=matrix[chosen[1]]["key"] if chosen else None,
                    selection_hash=chosen[0] if chosen else None,
                )
            )
    a.write(a.OUT / "case_selection.json", cases)

    def slot_rows():
        for i, row in enumerate(matrix):
            for p in ["reference", *a.POLICIES]:
                e = arrays[p][i]
                width = e.shape[1]
                for col in range(width):
                    path = [int(reference[i, 0, col % 3]), *map(int, e[:, col])]
                    yield dict(
                        input_index=i,
                        key=row["key"],
                        population=row["population"],
                        gold_hate=row["gold_hate"],
                        policy=p,
                        stream=None if p == "reference" else a.STREAMS[col // 3],
                        repeat=col % 3,
                        eligible_all_round=bool(fixed[i]),
                        errors=path,
                        **k.path_profile(path),
                    )

    a.lines(a.OUT / "slot_paths.jsonl", slot_rows())
    primary = a.read(a.PART2 / "errors_v1/results.json")["records"]
    reference_pairs = []
    adjacent = []
    context_transitions = []
    unique = []
    relative = []
    path_counts = []
    lifetimes = []
    origins = []
    variation = []
    alignment = []
    pairing = []
    masks = []
    for pop in a.POPS:
        for endpoint, gold in [("FPR", False), ("FNR", True)]:
            population = np.array(
                [r["population"] == pop and r["gold_hate"] == gold for r in matrix]
            )
            fixed_indices = np.flatnonzero(population & fixed)
            masks.append(
                dict(
                    population=pop,
                    endpoint=endpoint,
                    roster=int(population.sum()),
                    fixed_inputs=len(fixed_indices),
                    fixed_clusters=len(
                        {matrix[i]["inference_cluster"] for i in fixed_indices}
                    ),
                    round_inputs=[
                        int((population & round_common[:, r]).sum()) for r in range(4)
                    ],
                    adjacent_pair_inputs=[
                        int(
                            (
                                population & round_common[:, r - 1] & round_common[:, r]
                            ).sum()
                        )
                        for r in range(1, 4)
                    ],
                )
            )
            for p, e in arrays.items():
                width = e.shape[2]
                ref0 = reference[fixed_indices, 0, :]
                if p != "reference":
                    summaries = [profiles[i, p] for i in fixed_indices]
                    flags = [
                        "all_saved_correct",
                        "all_saved_wrong",
                        "initial_reference_all_wrong",
                        "initial_reference_all_correct",
                        "initial_consistent_fix",
                        "durable_initial_fix",
                        "initial_fix_reversal",
                        "initial_fix_recovered",
                        "initial_fix_lost_at_r4",
                        "delayed_clean_suffix",
                        "sustained_new_error",
                        "recovered_new_error",
                        "persistent_error",
                    ]
                    unique.append(
                        dict(
                            population=pop,
                            endpoint=endpoint,
                            policy=p,
                            inputs=len(fixed_indices),
                            **{f: sum(r[f] for r in summaries) for f in flags},
                            r4_vs_r1_improved=sum(
                                r["r4_vs_r1_error_change"] < 0 for r in summaries
                            ),
                            r4_vs_r1_worsened=sum(
                                r["r4_vs_r1_error_change"] > 0 for r in summaries
                            ),
                            r4_vs_r1_equal=sum(
                                r["r4_vs_r1_error_change"] == 0 for r in summaries
                            ),
                            clean_suffix_start_counts=dict(
                                Counter(
                                    str(r["clean_suffix_from_round"]) for r in summaries
                                )
                            ),
                            wrong_suffix_start_counts=dict(
                                Counter(
                                    str(r["wrong_suffix_from_round"]) for r in summaries
                                )
                            ),
                        )
                    )
                    for basis in ["contemporaneous", "initial_reference"]:
                        signs = Counter(r[basis]["pattern"] for r in summaries)
                        relative.append(
                            dict(
                                population=pop,
                                endpoint=endpoint,
                                policy=p,
                                basis=basis,
                                inputs=len(fixed_indices),
                                patterns={
                                    "".join(v): signs["".join(v)]
                                    for v in product("BW=", repeat=4)
                                },
                                initial_advantage_states=dict(
                                    Counter(
                                        r[basis]["initial_advantage_state"]
                                        for r in summaries
                                    )
                                ),
                                mixed_direction=sum(
                                    r[basis]["mixed_direction"] for r in summaries
                                ),
                            )
                        )
                for rnd in range(1, 5):
                    for mask in ["all_round", "round_common"]:
                        ix = (
                            fixed_indices
                            if mask == "all_round"
                            else np.flatnonzero(population & round_common[:, rnd - 1])
                        )
                        old = (
                            reference[ix, rnd - 1, :]
                            if p == "reference"
                            else np.tile(reference[ix, rnd - 1, :], (1, 3))
                        )
                        if p != "reference":
                            reference_pairs.append(
                                dict(
                                    population=pop,
                                    endpoint=endpoint,
                                    policy=p,
                                    round=rnd,
                                    mask=mask,
                                    **k.transition_summary(old, e[ix, rnd - 1, :]),
                                )
                            )
                        record = (
                            primary[f"trajectory_{pop}_{endpoint}"]
                            if mask == "all_round"
                            else primary[f"round_{pop}_{endpoint}_r{rnd}"]
                        )
                        a.need(
                            abs(
                                float(e[ix, rnd - 1, :].mean())
                                - record["points"][f"r{rnd}_{p}"]
                            )
                            < 1e-13,
                            "Part 2 marginal changed",
                        )
                    if rnd == 1:
                        continue
                    for mask in ["all_round", "pair_common"]:
                        ix = (
                            fixed_indices
                            if mask == "all_round"
                            else np.flatnonzero(
                                population
                                & round_common[:, rnd - 2]
                                & round_common[:, rnd - 1]
                            )
                        )
                        summary = k.transition_summary(
                            e[ix, rnd - 2, :], e[ix, rnd - 1, :]
                        )
                        adjacent.append(
                            dict(
                                population=pop,
                                endpoint=endpoint,
                                policy=p,
                                round=rnd,
                                mask=mask,
                                **summary,
                            )
                        )
                        a.need(
                            summary["new_errors"] - summary["old_errors"]
                            == summary["counts"]["new_error"]
                            - summary["counts"]["fix"],
                            "Transition telescoping failed",
                        )

                    grouped = defaultdict(list)
                    for i in fixed_indices:
                        for stream_index, stream in enumerate(
                            [None] if p == "reference" else a.STREAMS
                        ):
                            if p == "reference":
                                state = "same_prompt"
                            else:
                                old = contexts[i, rnd - 1, p, stream]
                                new = contexts[i, rnd, p, stream]
                                state = (
                                    "same_prompt"
                                    if old["messages_sha256"] == new["messages_sha256"]
                                    else (
                                        "new_correction"
                                        if new["added_round"] == rnd
                                        else "other_old_correction"
                                    )
                                )
                            sl = slice(3 * stream_index, 3 * stream_index + 3)
                            grouped[state].append(
                                (i, e[i, rnd - 2, sl], e[i, rnd - 1, sl])
                            )
                    for state, rows in grouped.items():
                        summary = k.transition_summary(
                            np.array([r[1] for r in rows]),
                            np.array([r[2] for r in rows]),
                        )
                        context_transitions.append(
                            dict(
                                population=pop,
                                endpoint=endpoint,
                                policy=p,
                                round=rnd,
                                state=state,
                                unit=(
                                    "text-stream assignment"
                                    if p != "reference"
                                    else "text"
                                ),
                                unique_inputs=len({r[0] for r in rows}),
                                **summary,
                            )
                        )
                body = e[fixed_indices]
                base = ref0 if p == "reference" else np.tile(ref0, (1, 3))
                paths = np.concatenate(
                    [base[:, :, None], body.transpose(0, 2, 1)], axis=2
                )
                counts = Counter(
                    "".join(map(str, path)) for path in paths.reshape((-1, 5)).tolist()
                )
                path_counts.append(
                    dict(
                        population=pop,
                        endpoint=endpoint,
                        policy=p,
                        inputs=len(fixed_indices),
                        outcomes_per_input=width,
                        patterns={
                            "".join(v): counts["".join(v)]
                            for v in product("01", repeat=5)
                        },
                    )
                )
                origin_counts = Counter()
                cohorts = defaultdict(Counter)
                for path in paths.reshape((-1, 5)).tolist():
                    profile = k.path_profile(path)
                    origin_counts[profile["final_origin"]] += 1
                    for rnd in range(2 if p == "reference" else 1, 5):
                        for event in ["fix", "new_error"]:
                            lifetime = k.event_lifetime(path, rnd, event)
                            if lifetime is not None:
                                cohorts[rnd, event][lifetime["state"]] += 1
                origins.append(
                    dict(
                        population=pop,
                        endpoint=endpoint,
                        policy=p,
                        inputs=len(fixed_indices),
                        paired_slots=paths.shape[0] * width,
                        counts=dict(origin_counts),
                    )
                )
                for rnd in range(2 if p == "reference" else 1, 5):
                    for event in ["fix", "new_error"]:
                        cc = cohorts[rnd, event]
                        lifetimes.append(
                            dict(
                                population=pop,
                                endpoint=endpoint,
                                policy=p,
                                round=rnd,
                                event=event,
                                events=sum(cc.values()),
                                future_rounds=4 - rnd,
                                censored_at_horizon=rnd == 4,
                                counts={
                                    s: cc[s]
                                    for s in [
                                        "uninterrupted",
                                        "returned_by_r4",
                                        "opposite_at_r4",
                                    ]
                                },
                            )
                        )
                for col in range(width):
                    local = paths[:, col, :].tolist()
                    hist = Counter(
                        k.path_profile(z)["initial_fix_state"] for z in local
                    )
                    variation.append(
                        dict(
                            population=pop,
                            endpoint=endpoint,
                            policy=p,
                            stream=None if p == "reference" else a.STREAMS[col // 3],
                            repeat=col % 3,
                            inputs=len(fixed_indices),
                            round_errors=[int(body[:, r, col].sum()) for r in range(4)],
                            initial_fixes=sum(z[0] == 1 and z[1] == 0 for z in local),
                            initial_regressions=sum(
                                z[0] == 0 and z[1] == 1 for z in local
                            ),
                            initial_fix_states=dict(hist),
                        )
                    )
                for sidx, stream in enumerate(
                    [None] if p == "reference" else a.STREAMS
                ):
                    local = body[:, :, 3 * sidx : 3 * sidx + 3]
                    for rnd in range(2, 5):
                        for pi, perm in enumerate(k.PERMS):
                            summary = k.transition_summary(
                                local[:, rnd - 2, :], local[:, rnd - 1, list(perm)]
                            )
                            pairing.append(
                                dict(
                                    population=pop,
                                    endpoint=endpoint,
                                    policy=p,
                                    stream=stream,
                                    round=rnd,
                                    permutation_index=pi,
                                    fixes=summary["counts"]["fix"],
                                    new_errors=summary["counts"]["new_error"],
                                    net_error_change=summary["new_errors"]
                                    - summary["old_errors"],
                                )
                            )
                    if p != "reference":
                        alignment.extend(
                            dict(
                                population=pop,
                                endpoint=endpoint,
                                policy=p,
                                stream=stream,
                                **row,
                            )
                            for row in k.alignment_counts(ref0, local)
                        )
            print(
                f"All longitudinal paths scored: {pop}, {endpoint}, {len(fixed_indices)} fixed texts",
                flush=True,
            )
    a.lines(a.OUT / "repeat_alignment_enumeration.jsonl", alignment)
    envelopes = []
    for pop in a.POPS:
        for ep in ["FPR", "FNR"]:
            for p in a.POLICIES:
                streams = []
                for stream in a.STREAMS:
                    rows = [
                        r
                        for r in alignment
                        if (r["population"], r["endpoint"], r["policy"], r["stream"])
                        == (pop, ep, p, stream)
                    ]
                    streams.append(
                        dict(
                            stream=stream,
                            initial_fixes=rows[0]["initial_fixes"],
                            ranges={
                                f: [min(r[f] for r in rows), max(r[f] for r in rows)]
                                for f in [
                                    "retained_all_later",
                                    "lost_by_r4",
                                    "recovered_by_r4",
                                ]
                            },
                        )
                    )
                envelopes.append(
                    dict(
                        population=pop,
                        endpoint=ep,
                        policy=p,
                        initial_fixes=sum(r["initial_fixes"] for r in streams),
                        streams=streams,
                        aggregate_ranges={
                            f: [
                                sum(r["ranges"][f][bound] for r in streams)
                                for bound in [0, 1]
                            ]
                            for f in [
                                "retained_all_later",
                                "lost_by_r4",
                                "recovered_by_r4",
                            ]
                        },
                    )
                )
    summary = dict(
        status="part4_scored_independent_review_pending",
        created_utc=a.pure.utc(),
        masks=masks,
        reference_comparisons=reference_pairs,
        adjacent_transitions=adjacent,
        context_transitions=context_transitions,
        unique_input_profiles=unique,
        relative_paths=relative,
        slot_pattern_counts=path_counts,
        event_lifetimes=lifetimes,
        final_error_origins=origins,
        repeat_variation=variation,
        adjacent_repeat_pairings=pairing,
        initial_fix_alignment_envelopes=envelopes,
        roster_inputs=N,
        fixed_inputs=int(fixed.sum()),
        preserved_evaluation_nulls=int((error < 0).sum()),
        slot_paths=N * 30,
        fixed_slot_paths=int(fixed.sum()) * 30,
        case_slots=len(cases),
        selected_case_slots=sum(c["status"] == "selected" for c in cases),
        no_new_inference_family=True,
        new_Qwen_calls=0,
        new_fits=0,
        new_embeddings=0,
    )
    a.write(a.OUT / "analysis.json", summary)
    for name, digest in bindings.items():
        a.need(a.sha(a.ROOT / name) == digest, "Input changed during Part 4")
    a.write(
        a.OUT / "assembled.json",
        dict(
            status="part4_assembled_scored_pending_independent_verification",
            created_utc=a.pure.utc(),
            input_bindings=bindings,
            artifacts={a.rel(p): a.sha(p) for p in a.OUT.iterdir() if p.is_file()},
            elapsed_seconds=time.monotonic() - start,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "Part 4 trajectories, all declared cases and complete repeat-alignment sensitivities saved.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        build()
