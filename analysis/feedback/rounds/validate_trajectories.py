from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations, product
import hashlib
from pathlib import Path
import numpy as np
from analysis.feedback.rounds import trajectories_inputs as io


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def equal(x, y, where):
    if isinstance(y, dict):
        check(set(x) == set(y), where + " keys")
        for k in y:
            equal(x[k], y[k], where + "/" + str(k))
    elif isinstance(y, (list, tuple)):
        check(len(x) == len(y), where + " length")
        for i, v in enumerate(y):
            equal(x[i], v, where + "/" + str(i))
    elif isinstance(y, float):
        check(x is not None and abs(x - y) < 1e-13, where)
    else:
        check(x == y, where)


def relative(delta):
    pattern = "".join({-1: "B", 0: "=", 1: "W"}[(v > 0) - (v < 0)] for v in delta)
    if pattern[0] != "B":
        state = "not_initially_better"
    elif pattern == "BBBB":
        state = "better_every_round"
    elif pattern[-1] == "W":
        state = "worse_at_r4"
    elif pattern[-1] == "=":
        state = "equal_at_r4"
    elif "W" in pattern[1:3]:
        state = "better_at_r4_after_harm"
    else:
        state = "better_at_r4_after_tie"
    return dict(
        pattern=pattern,
        initial_advantage_state=state,
        ever_better="B" in pattern,
        ever_worse="W" in pattern,
        mixed_direction="B" in pattern and "W" in pattern,
    )


def profile(feedback, reference):
    if any(v < 0 for row in feedback + reference for v in row):
        return {"status": "missing"}
    n = [sum(r) for r in feedback]
    r = [sum(x) for x in reference]
    zero = {i + 1 for i, v in enumerate(n) if v == 0}
    nine = {i + 1 for i, v in enumerate(n) if v == 9}
    clean = next((i for i in range(1, 5) if set(range(i, 5)) <= zero), None)
    wrong = next((i for i in range(1, 5) if set(range(i, 5)) <= nine), None)
    fix = r[0] == 3 and 1 in zero
    delta = [n[i] - 3 * r[i] for i in range(4)]
    fixed_delta = [v - 3 * r[0] for v in n]
    return dict(
        status="complete",
        errors=n,
        reference_errors=r,
        all_saved_correct=len(zero) == 4,
        all_saved_wrong=len(nine) == 4,
        initial_reference_all_wrong=r[0] == 3,
        initial_reference_all_correct=r[0] == 0,
        initial_consistent_fix=fix,
        durable_initial_fix=fix and len(zero) == 4,
        initial_fix_reversal=fix and len(zero) < 4,
        initial_fix_recovered=fix and 4 in zero and not {2, 3} <= zero,
        initial_fix_lost_at_r4=fix and 4 not in zero,
        delayed_clean_suffix=r[0] == 3 and clean in [2, 3, 4],
        clean_suffix_from_round=clean,
        wrong_suffix_from_round=wrong,
        sustained_new_error=r[0] == 0 and wrong in [1, 2, 3],
        recovered_new_error=r[0] == 0 and 4 in zero and not {1, 2, 3} <= zero,
        persistent_error=r[0] == 3 and len(nine) == 4,
        r4_vs_r1_error_change=n[3] - n[0],
        contemporaneous_delta_over_9=delta,
        initial_reference_delta_over_9=fixed_delta,
        contemporaneous=relative(delta),
        initial_reference=relative(fixed_delta),
    )


ORIGINS = {
    "0000": "always_correct_since_r1",
    "0001": "new_error_since_r1",
    "0010": "recovered_new_error",
    "0011": "new_error_since_r1",
    "0100": "recovered_new_error",
    "0101": "new_error_since_r1",
    "0110": "recovered_new_error",
    "0111": "new_error_since_r1",
    "1000": "retained_later_fix",
    "1001": "recurrent_error",
    "1010": "recovered_after_reversal",
    "1011": "recurrent_error",
    "1100": "retained_later_fix",
    "1101": "recurrent_error",
    "1110": "retained_later_fix",
    "1111": "always_wrong_since_r1",
}


def path_stats(path):
    pattern = "".join("?" if v == -1 else str(v) for v in path)
    if "?" in pattern:
        return dict(status="missing", pattern=pattern)
    fixes = [j for j in range(1, 5) if path[j - 1 : j + 1] == [1, 0]]
    new = [j for j in range(1, 5) if path[j - 1 : j + 1] == [0, 1]]
    fix = pattern.startswith("10")
    state = (
        "not_initial_fix"
        if not fix
        else (
            "retained_all_later"
            if pattern == "10000"
            else "lost_by_r4" if path[-1] else "recovered_by_r4"
        )
    )
    return dict(
        status="complete",
        pattern=pattern,
        fixes=fixes,
        new_errors=new,
        change_events=len(fixes) + len(new),
        initial_fix=fix,
        initial_regression=pattern.startswith("01"),
        initial_fix_state=state,
        final_origin=ORIGINS[pattern[1:]],
        first_feedback_correct_round=next(
            (i for i in range(1, 5) if not path[i]), None
        ),
    )


def paired(old, new):
    n = len(old)
    width = len(old[0]) if n else 0
    counts = Counter()
    diff = []
    fix_text = []
    bad_text = []
    for x, y in zip(old, new):
        check(all(v in [0, 1] for v in x + y), "Missing paired value")
        pairs = list(zip(x, y))
        counts.update(pairs)
        diff.append(sum(y) - sum(x))
        fix_text.append((1, 0) in pairs)
        bad_text.append((0, 1) in pairs)
    oldn = sum(sum(x) for x in old)
    newn = sum(sum(x) for x in new)
    return dict(
        inputs=n,
        outcomes_per_input=width,
        paired_slots=n * width,
        counts=dict(
            correct_correct=counts[0, 0],
            new_error=counts[0, 1],
            fix=counts[1, 0],
            error_error=counts[1, 1],
        ),
        old_errors=oldn,
        new_errors=newn,
        change=float(Fraction(newn - oldn, n * width)) if n else None,
        improved_inputs=sum(d < 0 for d in diff),
        worsened_inputs=sum(d > 0 for d in diff),
        equal_mean_inputs=diff.count(0),
        inputs_with_any_fix=sum(fix_text),
        inputs_with_any_new_error=sum(bad_text),
        inputs_with_both=sum(x and y for x, y in zip(fix_text, bad_text)),
    )


def main():
    assembled = io.read(io.OUT / "assembled.json")
    contract = io.read(io.OUT / "analysis_contract.json")
    bindings = {
        **assembled["input_bindings"],
        **assembled["artifacts"],
        **contract["source_hashes"],
    }
    for name, digest in bindings.items():
        check(io.sha(io.ROOT / name) == digest, "Changed bound input " + name)
    matrix = list(io.rows(io.READER / "matrix.jsonl"))
    result = io.read(io.OUT / "analysis.json")
    data = {}
    reference = {}
    fixed = []
    masks = {}
    for i, row in enumerate(matrix):
        raw = [
            [
                [
                    (
                        -1
                        if pred["result"] is None
                        else int(pred["result"]["hate"] != row["gold_hate"])
                    )
                    for pred in reps
                ]
                for reps in rnd
            ]
            for rnd in row["predictions"]
        ]
        reference[i] = [r[0] for r in raw]
        data[i, "reference"] = reference[i]
        for p in io.POLICIES:
            columns = [io.CONDITIONS.index(f"{p}_s{s}") for s in io.STREAMS]
            data[i, p] = [[v for c in columns for v in rnd[c]] for rnd in raw]
        rounds = [all(v >= 0 for col in rnd for v in col) for rnd in raw]
        masks[i] = rounds
        check(
            rounds == row["masks"]["round_common"]
            and all(rounds) == row["masks"]["all_round_common"],
            "Original masks independently reproduced",
        )
        if all(rounds):
            fixed.append(i)
    fixed = set(fixed)
    context = {
        (r["input_index"], r["round"], r["policy"], r["stream"]): r
        for r in io.rows(io.PART3 / "deliveries.jsonl")
        if r["role"] == "evaluation"
    }
    stored_inputs = list(io.rows(io.OUT / "input_trajectories.jsonl"))
    profiles = {}
    witnesses = {}
    for i, row in enumerate(stored_inputs):
        original = matrix[i]
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
        ]:
            equal(row[f], original[f], "Input metadata")
        equal(row["reference_errors_by_repeat"], reference[i], "All reference gates")
        for p, stored in row["policies"].items():
            equal(stored["errors_by_repeat"], data[i, p], "All treatment gates")
            profiles[i, p] = profile(data[i, p], reference[i])
            equal(stored["profile"], profiles[i, p], "Every unique-input profile")
            expected_contexts = {str(s): [] for s in io.STREAMS}
            witness = []
            for sidx, s in enumerate(io.STREAMS):
                for rnd in range(1, 5):
                    d = context[i, rnd, p, s]
                    expected_contexts[str(s)].append(
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
                    if rnd == 1:
                        continue
                    prev = context[i, rnd - 1, p, s]
                    if prev["messages_sha256"] == d["messages_sha256"]:
                        for rep in range(3):
                            old, new = (
                                data[i, p][rnd - 2][3 * sidx + rep],
                                data[i, p][rnd - 1][3 * sidx + rep],
                            )
                            if old >= 0 and new >= 0 and old != new:
                                witness.append(
                                    dict(
                                        round=rnd,
                                        stream=s,
                                        repeat=rep,
                                        old_error=old,
                                        new_error=new,
                                        source_id=d["source_id"],
                                        messages_sha256=d["messages_sha256"],
                                    )
                                )
            equal(stored["contexts"], expected_contexts, "All delivered identities")
            equal(
                stored["same_prompt_flip_witnesses"],
                witness,
                "Identical-prompt flip witnesses",
            )
            witnesses[i, p] = witness
    patterns = defaultdict(Counter)
    seen = set()
    slot_count = 0
    for row in io.rows(io.OUT / "slot_paths.jsonl"):
        i, p, rep, s = row["input_index"], row["policy"], row["repeat"], row["stream"]
        col = rep if p == "reference" else 3 * io.STREAMS.index(s) + rep
        identity = i, p, col
        check(identity not in seen, "Duplicate slot path")
        seen.add(identity)
        path = [reference[i][0][rep]] + [data[i, p][rnd][col] for rnd in range(4)]
        expected = dict(
            input_index=i,
            key=matrix[i]["key"],
            population=matrix[i]["population"],
            gold_hate=matrix[i]["gold_hate"],
            policy=p,
            stream=s,
            repeat=rep,
            eligible_all_round=i in fixed,
            errors=path,
            **path_stats(path),
        )
        equal(row, expected, "Every saved-decision path")
        if i in fixed:
            patterns[
                matrix[i]["population"], "FNR" if matrix[i]["gold_hate"] else "FPR", p
            ][expected["pattern"]] += 1
        slot_count += 1
    check(slot_count == 2077 * 30 and len(seen) == slot_count, "All path identities")

    def cohort(pop, ep, mask="all_round", rnd=None):
        return [
            i
            for i, r in enumerate(matrix)
            if r["population"] == pop
            and r["gold_hate"] == (ep == "FNR")
            and (
                i in fixed
                if mask == "all_round"
                else (
                    masks[i][rnd - 1]
                    if mask == "round_common"
                    else masks[i][rnd - 2] and masks[i][rnd - 1]
                )
            )
        ]

    for row in result["masks"]:
        pop, ep = row["population"], row["endpoint"]
        ix = cohort(pop, ep)
        equal(
            row,
            dict(
                population=pop,
                endpoint=ep,
                roster=sum(
                    r["population"] == pop and r["gold_hate"] == (ep == "FNR")
                    for r in matrix
                ),
                fixed_inputs=len(ix),
                fixed_clusters=len({matrix[i]["inference_cluster"] for i in ix}),
                round_inputs=[
                    len(cohort(pop, ep, "round_common", r)) for r in range(1, 5)
                ],
                adjacent_pair_inputs=[
                    len(cohort(pop, ep, "pair_common", r)) for r in range(2, 5)
                ],
            ),
            "Mask denominators",
        )
    for row in result["reference_comparisons"]:
        pop, ep, p, rnd, mask = [
            row[k] for k in ["population", "endpoint", "policy", "round", "mask"]
        ]
        ix = cohort(pop, ep, mask, rnd)
        equal(
            row,
            dict(
                population=pop,
                endpoint=ep,
                policy=p,
                round=rnd,
                mask=mask,
                **paired(
                    [reference[i][rnd - 1] * 3 for i in ix],
                    [data[i, p][rnd - 1] for i in ix],
                ),
            ),
            "Same-round reference transitions",
        )
    for row in result["adjacent_transitions"]:
        pop, ep, p, rnd, mask = [
            row[k] for k in ["population", "endpoint", "policy", "round", "mask"]
        ]
        ix = cohort(pop, ep, mask, rnd)
        equal(
            row,
            dict(
                population=pop,
                endpoint=ep,
                policy=p,
                round=rnd,
                mask=mask,
                **paired(
                    [data[i, p][rnd - 2] for i in ix], [data[i, p][rnd - 1] for i in ix]
                ),
            ),
            "Adjacent transitions",
        )
    for row in result["context_transitions"]:
        pop, ep, p, rnd, state = [
            row[k] for k in ["population", "endpoint", "policy", "round", "state"]
        ]
        old = []
        new = []
        chosen = set()
        for i in cohort(pop, ep):
            for sidx, s in enumerate([None] if p == "reference" else io.STREAMS):
                if p == "reference":
                    cat = "same_prompt"
                else:
                    before, after = context[i, rnd - 1, p, s], context[i, rnd, p, s]
                    cat = (
                        "same_prompt"
                        if before["messages_sha256"] == after["messages_sha256"]
                        else (
                            "new_correction"
                            if after["added_round"] == rnd
                            else "other_old_correction"
                        )
                    )
                if cat == state:
                    old.append(data[i, p][rnd - 2][3 * sidx : 3 * sidx + 3])
                    new.append(data[i, p][rnd - 1][3 * sidx : 3 * sidx + 3])
                    chosen.add(i)
        equal(
            row,
            dict(
                population=pop,
                endpoint=ep,
                policy=p,
                round=rnd,
                state=state,
                unit="text" if p == "reference" else "text-stream assignment",
                unique_inputs=len(chosen),
                **paired(old, new),
            ),
            "Context-state transition partition",
        )
    print(
        "All 62,310 paths, per-input gates, reference/adjacent/context transitions and masks agree.",
        flush=True,
    )
    for row in result["unique_input_profiles"]:
        pop, ep, p = [row[k] for k in ["population", "endpoint", "policy"]]
        pr = [profiles[i, p] for i in cohort(pop, ep)]
        expected = {k: row[k] for k in ["population", "endpoint", "policy"]}
        expected["inputs"] = len(pr)
        for key in [
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
        ]:
            expected[key] = sum(r[key] for r in pr)
        expected.update(
            r4_vs_r1_improved=sum(r["r4_vs_r1_error_change"] < 0 for r in pr),
            r4_vs_r1_worsened=sum(r["r4_vs_r1_error_change"] > 0 for r in pr),
            r4_vs_r1_equal=sum(r["r4_vs_r1_error_change"] == 0 for r in pr),
            clean_suffix_start_counts=dict(
                Counter(str(r["clean_suffix_from_round"]) for r in pr)
            ),
            wrong_suffix_start_counts=dict(
                Counter(str(r["wrong_suffix_from_round"]) for r in pr)
            ),
        )
        equal(row, expected, "Unique-input consistency counts")
    for row in result["relative_paths"]:
        pop, ep, p, basis = [
            row[k] for k in ["population", "endpoint", "policy", "basis"]
        ]
        pr = [profiles[i, p][basis] for i in cohort(pop, ep)]
        counts = Counter(r["pattern"] for r in pr)
        equal(
            row,
            dict(
                population=pop,
                endpoint=ep,
                policy=p,
                basis=basis,
                inputs=len(pr),
                patterns={
                    "".join(x): counts["".join(x)] for x in product("BW=", repeat=4)
                },
                initial_advantage_states=dict(
                    Counter(r["initial_advantage_state"] for r in pr)
                ),
                mixed_direction=sum(r["mixed_direction"] for r in pr),
            ),
            "Mean-advantage trajectory",
        )
    for row in result["slot_pattern_counts"]:
        key = tuple(row[k] for k in ["population", "endpoint", "policy"])
        n = len(cohort(key[0], key[1]))
        equal(
            row,
            dict(
                population=key[0],
                endpoint=key[1],
                policy=key[2],
                inputs=n,
                outcomes_per_input=3 if key[2] == "reference" else 9,
                patterns={
                    "".join(x): patterns[key]["".join(x)]
                    for x in product("01", repeat=5)
                },
            ),
            "All 32 binary path patterns",
        )
    for row in result["event_lifetimes"]:
        pop, ep, p, rnd, event = [
            row[k] for k in ["population", "endpoint", "policy", "round", "event"]
        ]
        count = Counter()
        pair = "10" if event == "fix" else "01"
        post = pair[1]
        for path, n in patterns[pop, ep, p].items():
            if path[rnd - 1 : rnd + 1] != pair:
                continue
            state = (
                "uninterrupted"
                if all(bit == post for bit in path[rnd + 1 :])
                else "returned_by_r4" if path[-1] == post else "opposite_at_r4"
            )
            count[state] += n
        equal(
            row,
            dict(
                population=pop,
                endpoint=ep,
                policy=p,
                round=rnd,
                event=event,
                events=sum(count.values()),
                future_rounds=4 - rnd,
                censored_at_horizon=rnd == 4,
                counts={
                    s: count[s]
                    for s in ["uninterrupted", "returned_by_r4", "opposite_at_r4"]
                },
            ),
            "Event lifetime partition",
        )
    for row in result["final_error_origins"]:
        pop, ep, p = [row[k] for k in ["population", "endpoint", "policy"]]
        counts = Counter()
        for pattern, n in patterns[pop, ep, p].items():
            counts[ORIGINS[pattern[1:]]] += n
        equal(
            row,
            dict(
                population=pop,
                endpoint=ep,
                policy=p,
                inputs=len(cohort(pop, ep)),
                paired_slots=sum(counts.values()),
                counts=dict(counts),
            ),
            "Final error origins",
        )
    for row in result["repeat_variation"]:
        pop, ep, p, s, rep = [
            row[k] for k in ["population", "endpoint", "policy", "stream", "repeat"]
        ]
        ix = cohort(pop, ep)
        col = rep if p == "reference" else io.STREAMS.index(s) * 3 + rep
        paths = [
            [reference[i][0][rep]] + [data[i, p][r][col] for r in range(4)] for i in ix
        ]
        equal(
            row,
            dict(
                population=pop,
                endpoint=ep,
                policy=p,
                stream=s,
                repeat=rep,
                inputs=len(ix),
                round_errors=[sum(path[r + 1] for path in paths) for r in range(4)],
                initial_fixes=sum(path[:2] == [1, 0] for path in paths),
                initial_regressions=sum(path[:2] == [0, 1] for path in paths),
                initial_fix_states=dict(
                    Counter(path_stats(path)["initial_fix_state"] for path in paths)
                ),
            ),
            "All stream/decoder cells",
        )
    perms = list(permutations([0, 1, 2]))
    for row in result["adjacent_repeat_pairings"]:
        pop, ep, p, s, rnd, pi = [
            row[k]
            for k in [
                "population",
                "endpoint",
                "policy",
                "stream",
                "round",
                "permutation_index",
            ]
        ]
        base = 0 if p == "reference" else 3 * io.STREAMS.index(s)
        ix = cohort(pop, ep)
        pairs = Counter(
            (data[i, p][rnd - 2][base + r], data[i, p][rnd - 1][base + perms[pi][r]])
            for i in ix
            for r in range(3)
        )
        equal(
            row,
            dict(
                population=pop,
                endpoint=ep,
                policy=p,
                stream=s,
                round=rnd,
                permutation_index=pi,
                fixes=pairs[1, 0],
                new_errors=pairs[0, 1],
                net_error_change=pairs[0, 1] - pairs[1, 0],
            ),
            "Six-way adjacent repeat pairings",
        )

    enumerated = list(io.rows(io.OUT / "repeat_alignment_enumeration.jsonl"))
    by_group = defaultdict(list)
    for row in enumerated:
        by_group[
            tuple(row[k] for k in ["population", "endpoint", "policy", "stream"])
        ].append(row)
    popcount = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int64)
    alignment_values = 0
    for (pop, ep, p, s), group in by_group.items():
        ix = cohort(pop, ep)
        base = 3 * io.STREAMS.index(s)
        e = np.asarray(
            [[data[i, p][r][base : base + 3] for r in range(4)] for i in ix],
            dtype=np.uint8,
        )
        ref = np.asarray([reference[i][0] for i in ix], dtype=np.uint8)
        powers = np.array([1, 2, 4], dtype=np.uint8)
        initial = ((ref == 1) & (e[:, 0, :] == 0)).astype(np.uint8) @ powers
        allbits = {
            (r, pi): (e[:, r, list(perm)] @ powers)
            for r in [1, 2, 3]
            for pi, perm in enumerate(perms)
        }
        check(
            len(group) == 216
            and len({tuple(r["permutation_indices"]) for r in group}) == 216,
            "Complete permutation enumeration",
        )
        for row in group:
            a, b, c = [
                allbits[r + 1, pi] for r, pi in enumerate(row["permutation_indices"])
            ]
            retain = int(popcount[initial & (7 ^ (a | b | c))].sum())
            lost = int(popcount[initial & c].sum())
            recover = int(popcount[initial & (7 ^ c) & (a | b)].sum())
            equal(
                row,
                dict(
                    population=pop,
                    endpoint=ep,
                    policy=p,
                    stream=s,
                    permutation_indices=row["permutation_indices"],
                    initial_fixes=int(popcount[initial].sum()),
                    retained_all_later=retain,
                    lost_by_r4=lost,
                    recovered_by_r4=recover,
                ),
                "Bit-mask future-alignment replay",
            )
            alignment_values += 3
    for row in result["initial_fix_alignment_envelopes"]:
        pop, ep, p = [row[k] for k in ["population", "endpoint", "policy"]]
        streams = []
        for s in io.STREAMS:
            group = by_group[pop, ep, p, s]
            streams.append(
                dict(
                    stream=s,
                    initial_fixes=group[0]["initial_fixes"],
                    ranges={
                        f: [min(r[f] for r in group), max(r[f] for r in group)]
                        for f in ["retained_all_later", "lost_by_r4", "recovered_by_r4"]
                    },
                )
            )
        equal(
            row,
            dict(
                population=pop,
                endpoint=ep,
                policy=p,
                initial_fixes=sum(r["initial_fixes"] for r in streams),
                streams=streams,
                aggregate_ranges={
                    f: [sum(r["ranges"][f][b] for r in streams) for b in [0, 1]]
                    for f in ["retained_all_later", "lost_by_r4", "recovered_by_r4"]
                },
            ),
            "Alignment envelopes",
        )
    cases = io.read(io.OUT / "case_selection.json")
    for case in cases:
        pop, slot = case["population"], case["slot"]
        eligible = []
        for i in cohort(pop, "FPR") + cohort(pop, "FNR"):
            pr = profiles[i, "adaptive"]
            ok = (
                bool(witnesses[i, "adaptive"])
                if slot == "identical_prompt_flip"
                else (
                    pr["contemporaneous"]["mixed_direction"]
                    if slot == "mixed_relative_direction"
                    else pr[slot]
                )
            )
            if ok:
                eligible.append(
                    (
                        hashlib.sha256(
                            f'A17-part4-case-v1|{pop}|{slot}|{matrix[i]["key"]}'.encode()
                        ).hexdigest(),
                        i,
                    )
                )
        chosen = min(eligible) if eligible else None
        equal(
            case,
            dict(
                population=pop,
                slot=slot,
                candidates=len(eligible),
                status="selected" if chosen else "empty",
                input_index=chosen[1] if chosen else None,
                key=matrix[chosen[1]]["key"] if chosen else None,
                selection_hash=chosen[0] if chosen else None,
            ),
            "Outcome-informed but fixed-rule case selection",
        )

    points = io.read(io.PART2 / "errors_v1/results.json")["records"]
    checks = 0
    for pop in io.POPS:
        for ep in ["FPR", "FNR"]:
            ix = cohort(pop, ep)
            for p in ["reference", *io.POLICIES]:
                width = 3 if p == "reference" else 9
                for rnd in range(4):
                    actual = float(
                        Fraction(sum(sum(data[i, p][rnd]) for i in ix), width * len(ix))
                    )
                    equal(
                        actual,
                        points[f"trajectory_{pop}_{ep}"]["points"][f"r{rnd+1}_{p}"],
                        "Part 2 exact marginal",
                    )
                    checks += 1
    for name, digest in bindings.items():
        check(
            io.sha(io.ROOT / name) == digest, "Input changed during independent review"
        )
    io.write(
        io.OUT / "independent_verification.json",
        dict(
            status="part4_independent_verification_pass",
            created_utc=io.pure.utc(),
            source_sha256=io.sha(Path(__file__)),
            verified_input_profiles=2077 * 3,
            verified_gate_slots=2077 * 4 * 10 * 3,
            verified_slot_paths=slot_count,
            verified_fixed_slot_paths=len(fixed) * 30,
            reference_comparison_rows=len(result["reference_comparisons"]),
            adjacent_transition_rows=len(result["adjacent_transitions"]),
            context_transition_rows=len(result["context_transitions"]),
            event_lifetime_rows=len(result["event_lifetimes"]),
            repeat_cells=len(result["repeat_variation"]),
            adjacent_pairings=len(result["adjacent_repeat_pairings"]),
            future_alignments=len(enumerated),
            independent_alignment_counts=alignment_values,
            part2_marginal_checks=checks,
            case_slots=len(cases),
            input_bindings={
                io.rel(p): io.sha(p)
                for p in [
                    io.OUT / "assembled.json",
                    io.OUT / "analysis.json",
                    io.OUT / "analysis_contract.json",
                    io.READER / "matrix.jsonl",
                ]
            },
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_bootstrap_draws=0,
        ),
    )
    print(
        f"Independent verification passed: {slot_count:,} paths; {len(enumerated):,} future alignments; {checks} unchanged Part 2 marginals.",
        flush=True,
    )


if __name__ == "__main__":
    with io.pure.offline():
        main()
