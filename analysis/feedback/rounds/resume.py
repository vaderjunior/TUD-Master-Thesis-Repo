import argparse
from collections import Counter
import math
from pathlib import Path

from analysis.feedback.rounds import resume_original as previous

a, c, q = previous.a, previous.c, previous.q
original = previous.original
BASE = c.ITERATIVE / "execution/ranking_20261025_v3"
MANIFEST, READY = BASE / "manifest.json", BASE / "ready.json"
LAUNCHER = c.ROOT / "scripts/resume_feedback_rounds.ps1"
NEW_PHASES = {
    "r3_evaluation",
    "r4_evaluation",
    *(f"r4_source_s{s}" for s in c.STREAM_SEEDS),
}


class VerificationBucket(original.LocalBucket):
    """Keep the independent scorer; sum its IDFs in the library's order."""

    def __init__(self, records, vectors):
        super().__init__(records, vectors)
        frequencies = Counter(word for counts in self.counts for word in counts)
        raw = {
            word: math.log(len(self.ids) - freq + 0.5) - math.log(freq + 0.5)
            for word, freq in frequencies.items()
        }
        total = 0.0
        for value in raw.values():
            total += value
        floor = 0.25 * total / len(raw)
        self.idf = {
            word: (value if value >= 0 else floor) for word, value in raw.items()
        }


def phase_builder():
    return previous.bind(original.make_phase, LocalBucket=VerificationBucket)


def phase_marker(root):
    root = Path(root)
    return dict(
        status="independent_bm25_sequential_idf_sum_verified",
        manifest_sha256=a.digest(MANIFEST),
        phase=previous.project_relative(root),
        parent_scientific_plan_sha256=a.digest(c.ITERATIVE / "plan.json"),
        scientific_plan_sha256=a.digest(root / "scientific_plan_v1.json"),
        prompt_sha256=a.digest(root / "prompts.jsonl"),
        verification_sha256=a.digest(root / "independent_verification.json"),
        complete_sha256=a.digest(root / "complete.json"),
        primary_retriever_changed=False,
        ranking_assertions_relaxed=False,
        actual_preparation_Qwen_calls=0,
    )


def record_phase(root):
    root = Path(root)
    a.require(
        root.parent == c.ITERATIVE / "phases" and root.name in NEW_PHASES,
        "Unexpected amended phase",
    )
    c.save(root / "execution/ranking_20261025_v3/verification.json", phase_marker(root))


def make_phase(root, *args, **kwargs):
    root = Path(root)
    amended = root.name in NEW_PHASES
    if amended:
        a.require(root.parent == c.ITERATIVE / "phases", "Unexpected amended root")
        if (root / "execution/live").exists():
            a.require(
                a.read(root / "execution/ranking_20261025_v3/verification.json")
                == phase_marker(root),
                "Missing or changed checker provenance for live phase",
            )
    b = phase_builder()(root, *args, **kwargs)
    if amended:
        record_phase(root)
    return b


def bound_controller():
    return previous.bind(previous.bound_controller(), make_phase=make_phase)


def validate(*, ready=True):
    previous.validate()
    manifest = a.read(MANIFEST)
    a.require(
        manifest["status"] == "offline_verified_independent_bm25_checker_continuation"
        and manifest["parent_path_manifest_sha256"] == a.digest(previous.MANIFEST)
        and manifest["parent_scientific_plan_sha256"]
        == a.digest(c.ITERATIVE / "plan.json"),
        "Ranking continuation identity differs",
    )
    a.require(
        manifest["effective_workers"] == 128
        and manifest["new_transport_allowance"] == 0
        and manifest["new_scientific_predictions"] == 0
        and manifest["api_contract"] == a.CONTRACT,
        "Ranking continuation scope differs",
    )
    for key in ("source_hashes", "checkpoint_hashes", "evidence_hashes"):
        previous.previous.transport.verify_files(manifest[key])
    a.require(
        a.read(BASE / "tests.json")["passed"]
        and a.read(BASE / "preflight.json")["passed"],
        "Independent checker verification incomplete",
    )
    root = c.ITERATIVE / "phases/r3_evaluation"
    a.require(
        a.read(root / "execution/ranking_20261025_v3/verification.json")
        == phase_marker(root),
        "Prepared round-three checker provenance differs",
    )
    if ready:
        receipt = a.read(READY)
        a.require(
            receipt["status"] == "offline_verified_ready_for_user"
            and receipt["manifest_sha256"] == a.digest(MANIFEST),
            "Ranking continuation is not sealed",
        )
        previous.previous.transport.verify_files(receipt["verification_hashes"])
    return manifest


def final_check():
    previous.final_check()
    markers = {}
    for name in sorted(NEW_PHASES):
        root = c.ITERATIVE / "phases" / name
        marker = root / "execution/ranking_20261025_v3/verification.json"
        a.require(
            a.read(marker) == phase_marker(root), "Final checker provenance differs"
        )
        markers[previous.project_relative(marker)] = a.digest(marker)
    c.save(
        BASE / "live_ranking_runtime_complete.json",
        dict(
            status="all_rounds_and_checker_continuation_verified_scientific_review_pending",
            manifest_sha256=a.digest(MANIFEST),
            phase_markers=markers,
            original_complete_sha256=a.digest(c.ITERATIVE / "complete.json"),
            path_complete_sha256=a.digest(
                previous.BASE / "live_path_runtime_complete.json"
            ),
            actual_finalization_Qwen_calls=0,
        ),
    )
    print(
        "All feedback rounds and continuation checks complete. Ask Astra to review the saved results.",
        flush=True,
    )


def run():
    with a.item_lock(
        a.storage_path(previous.previous.runtime.BASE / "runtime_launcher.lock")
    ):
        validate()
        c.resources(2.5, 20.0)
        bound_controller()()
        final_check()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["validate", "run"], default="validate", nargs="?"
    )
    args = parser.parse_args()
    with original.threadpool_limits(limits=1):
        if args.mode == "run":
            run()
        else:
            with previous.previous.inspection.offline():
                validate()
                c.resources(2.5, 20.0)
                print(
                    "Offline check passed. Saved rounds are preserved; corrected checker ready. Zero Qwen calls.",
                    flush=True,
                )
