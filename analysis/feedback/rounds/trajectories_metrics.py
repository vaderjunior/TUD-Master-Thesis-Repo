from collections import Counter
from itertools import permutations, product
import hashlib
import numpy as np

TRANSITIONS = ("correct_correct", "new_error", "fix", "error_error")
PERMS = list(permutations(range(3)))
CASE_SLOTS = (
    "durable_initial_fix",
    "initial_fix_reversal",
    "delayed_clean_suffix",
    "sustained_new_error",
    "recovered_new_error",
    "persistent_error",
    "mixed_relative_direction",
    "identical_prompt_flip",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def transition(old, new):
    require(old in [-1, 0, 1] and new in [-1, 0, 1], "Unknown error code")
    return "missing" if min(old, new) < 0 else TRANSITIONS[2 * old + new]


def path_profile(path):

    require(
        len(path) == 5 and all(v in [-1, 0, 1] for v in path), "Wrong longitudinal path"
    )
    if -1 in path:
        return {
            "status": "missing",
            "pattern": "".join("?" if x < 0 else str(x) for x in path),
        }
    changes = [transition(x, y) for x, y in zip(path[:-1], path[1:])]
    fixes = [i + 1 for i, v in enumerate(changes) if v == "fix"]
    regressions = [i + 1 for i, v in enumerate(changes) if v == "new_error"]
    initial_fix = path[0] == 1 and path[1] == 0
    initial_regression = path[0] == 0 and path[1] == 1
    initial_fix_state = "not_initial_fix"
    if initial_fix:
        initial_fix_state = (
            "retained_all_later"
            if not any(path[2:])
            else "recovered_by_r4" if path[-1] == 0 else "lost_by_r4"
        )
    first, last = path[1], path[-1]
    body = path[1:]
    if last:
        final_origin = (
            "always_wrong_since_r1"
            if all(body)
            else "recurrent_error" if first else "new_error_since_r1"
        )
    elif not any(body):
        final_origin = "always_correct_since_r1"
    elif not first:
        final_origin = "recovered_new_error"
    else:
        first_correct = body.index(0)
        final_origin = (
            "recovered_after_reversal"
            if any(body[first_correct:])
            else "retained_later_fix"
        )
    return dict(
        status="complete",
        pattern="".join(map(str, path)),
        fixes=fixes,
        new_errors=regressions,
        change_events=len(fixes) + len(regressions),
        initial_fix=initial_fix,
        initial_regression=initial_regression,
        initial_fix_state=initial_fix_state,
        final_origin=final_origin,
        first_feedback_correct_round=next(
            (r for r in range(1, 5) if path[r] == 0), None
        ),
    )


def event_lifetime(path, rnd, event):
    require(event in ["fix", "new_error"] and 1 <= rnd <= 4, "Event definition")
    require(-1 not in path and len(path) == 5, "Missing lifetime path")
    if transition(path[rnd - 1], path[rnd]) != event:
        return None
    target = 0 if event == "fix" else 1
    suffix = path[rnd + 1 :]
    state = (
        "uninterrupted"
        if all(v == target for v in suffix)
        else "returned_by_r4" if path[-1] == target else "opposite_at_r4"
    )
    return dict(state=state, future_rounds=4 - rnd, censored_at_horizon=rnd == 4)


def transition_summary(old, new):
    old, new = np.asarray(old, dtype=np.int8), np.asarray(new, dtype=np.int8)
    require(
        old.shape == new.shape
        and old.ndim == 2
        and np.all(old >= 0)
        and np.all(new >= 0),
        "Complete paired two-dimensional arrays required",
    )
    counts = np.bincount((2 * old + new).ravel(), minlength=4)
    differences = new.sum(axis=1) - old.sum(axis=1)
    paired = {name: int(counts[i]) for i, name in enumerate(TRANSITIONS)}
    n = old.size
    return dict(
        inputs=old.shape[0],
        outcomes_per_input=old.shape[1],
        paired_slots=n,
        counts=paired,
        old_errors=int(old.sum()),
        new_errors=int(new.sum()),
        change=(int(new.sum()) - int(old.sum())) / n if n else None,
        improved_inputs=int((differences < 0).sum()),
        worsened_inputs=int((differences > 0).sum()),
        equal_mean_inputs=int((differences == 0).sum()),
        inputs_with_any_fix=int(((old == 1) & (new == 0)).any(axis=1).sum()),
        inputs_with_any_new_error=int(((old == 0) & (new == 1)).any(axis=1).sum()),
        inputs_with_both=int(
            (
                ((old == 1) & (new == 0)).any(axis=1)
                & ((old == 0) & (new == 1)).any(axis=1)
            ).sum()
        ),
    )


def relative_path(delta):
    signs = "".join("B" if v < 0 else "W" if v > 0 else "=" for v in delta)
    if delta[0] >= 0:
        initial = "not_initially_better"
    elif all(v < 0 for v in delta):
        initial = "better_every_round"
    elif delta[-1] > 0:
        initial = "worse_at_r4"
    elif delta[-1] == 0:
        initial = "equal_at_r4"
    elif any(v > 0 for v in delta[1:3]):
        initial = "better_at_r4_after_harm"
    else:
        initial = "better_at_r4_after_tie"
    return dict(
        pattern=signs,
        initial_advantage_state=initial,
        ever_better=any(v < 0 for v in delta),
        ever_worse=any(v > 0 for v in delta),
        mixed_direction=any(v < 0 for v in delta) and any(v > 0 for v in delta),
    )


def input_profile(feedback, reference):
    e, r = np.asarray(feedback), np.asarray(reference)
    require(e.shape == (4, 9) and r.shape == (4, 3), "Wrong input policy/repeat axes")
    if np.any(e < 0) or np.any(r < 0):
        return {"status": "missing"}
    counts, ref = e.sum(axis=1).astype(int).tolist(), r.sum(axis=1).astype(int).tolist()
    initial_fix = ref[0] == 3 and counts[0] == 0
    clean_start = next(
        (i + 1 for i in range(4) if all(x == 0 for x in counts[i:])), None
    )
    wrong_start = next(
        (i + 1 for i in range(4) if all(x == 9 for x in counts[i:])), None
    )
    delta = [x - 3 * y for x, y in zip(counts, ref)]
    initial_delta = [x - 3 * ref[0] for x in counts]
    return dict(
        status="complete",
        errors=counts,
        reference_errors=ref,
        all_saved_correct=all(x == 0 for x in counts),
        all_saved_wrong=all(x == 9 for x in counts),
        initial_reference_all_wrong=ref[0] == 3,
        initial_reference_all_correct=ref[0] == 0,
        initial_consistent_fix=initial_fix,
        durable_initial_fix=initial_fix and all(x == 0 for x in counts),
        initial_fix_reversal=initial_fix and any(x > 0 for x in counts[1:]),
        initial_fix_recovered=initial_fix
        and any(x > 0 for x in counts[1:3])
        and counts[-1] == 0,
        initial_fix_lost_at_r4=initial_fix and counts[-1] > 0,
        delayed_clean_suffix=ref[0] == 3
        and clean_start is not None
        and clean_start > 1,
        clean_suffix_from_round=clean_start,
        wrong_suffix_from_round=wrong_start,
        sustained_new_error=ref[0] == 0
        and wrong_start is not None
        and wrong_start <= 3,
        recovered_new_error=ref[0] == 0
        and any(x > 0 for x in counts[:3])
        and counts[-1] == 0,
        persistent_error=ref[0] == 3 and all(x == 9 for x in counts),
        r4_vs_r1_error_change=counts[-1] - counts[0],
        contemporaneous_delta_over_9=delta,
        initial_reference_delta_over_9=initial_delta,
        contemporaneous=relative_path(delta),
        initial_reference=relative_path(initial_delta),
    )


def alignment_counts(reference_r1, stream_errors):

    r, e = np.asarray(reference_r1), np.asarray(stream_errors)
    require(r.shape == (len(e), 3) and e.shape[1:] == (4, 3), "Alignment dimensions")
    require(np.all(r >= 0) and np.all(e >= 0), "Missing alignment cohort")
    fixed = (r == 1) & (e[:, 0, :] == 0)
    n = int(fixed.sum())
    result = []
    for indices in product(range(6), repeat=3):
        later = [e[:, rnd + 1, list(PERMS[p])] for rnd, p in enumerate(indices)]
        uninterrupted = fixed & (later[0] == 0) & (later[1] == 0) & (later[2] == 0)
        lost = fixed & (later[-1] == 1)
        recovered = fixed & (later[-1] == 0) & ((later[0] == 1) | (later[1] == 1))
        counts = [int(x.sum()) for x in [uninterrupted, lost, recovered]]
        require(sum(counts) == n, "Alignment partition")
        result.append(
            dict(
                permutation_indices=list(indices),
                initial_fixes=n,
                retained_all_later=counts[0],
                lost_by_r4=counts[1],
                recovered_by_r4=counts[2],
            )
        )
    return result


def candidate_cases(profile, same_prompt_flip):
    if profile["status"] != "complete":
        return []
    return [
        name
        for name in CASE_SLOTS
        if (
            same_prompt_flip
            if name == "identical_prompt_flip"
            else (
                profile["contemporaneous"]["mixed_direction"]
                if name == "mixed_relative_direction"
                else profile[name]
            )
        )
    ]


def case_rank(population, slot, key):
    return hashlib.sha256(
        f"A17-part4-case-v1|{population}|{slot}|{key}".encode()
    ).hexdigest()
