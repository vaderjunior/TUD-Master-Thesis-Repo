"Check the final training update states."

from src.hsrag.paths import PROJECT_ROOT
from training.full_data import common as c, plan as p, runtime as r
from training.recovery import update_rules as rules

POLICY = PROJECT_ROOT / "artifacts/sq2_recovery_20261030_v1/recovery_policy.json"
POLICY_SHA = "f85be2ff42751dbfa31646703a8c51aa406a011abaf140665f0c14cd5eb7c24b"
STOP_PATH = "jobs/uniform_compact_s43/severity/l0_e1_b/s0_f0/attempts/690d043c5fa245e4af1b4cc9c31732cb/stopped.json"
STOP_SHA = "3536e3cba241c60dd449c833e6b759a603031a558afc307e2b6f39fbb3292320"


def policy_sha256():
    rules.require(c.sha(POLICY) == POLICY_SHA, "Recovery policy changed.")
    return POLICY_SHA


def reviewed_stop(path):
    if c.read(path)["resumable"] is True:
        return True
    return path == r.OUTPUT / STOP_PATH and c.sha(path) == STOP_SHA


def completed_state(job, data, attempt, planned):
    if planned == "empty_held_fold":
        return planned
    metadata = c.read(attempt / "training.json")
    proof = attempt / "parameter_equivalence.json"
    state = rules.verify(metadata, c.read(proof) if proof.exists() else None)
    rules.require(
        state == planned or (planned == "trained" and state == rules.AMP_STATE),
        "Unexpected completed state.",
    )
    return state


def expected_saved(data, job, saved, attempt):
    planned, steps = p.expected_state(data, job)
    state = completed_state(job, data, attempt, planned)
    rules.require(
        saved["state"] == state and saved["steps"] == steps,
        "Saved fit classification differs.",
    )
    if state == rules.AMP_STATE:
        rules.require(
            saved.get("verification_amendment_sha256") == policy_sha256(),
            "Unbound initialization-only result.",
        )
    if "verification_amendment_sha256" in saved:
        rules.require(
            saved["verification_amendment_sha256"] == policy_sha256(),
            "Different verification amendment.",
        )
    return state, steps
