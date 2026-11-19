import argparse
import ast
import inspect
import json
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch
from analysis.feedback.supplements import archive as original

ROOT = original.ROOT
EXEC = ROOT / "experiments/astra/sq3_five_slot_20261010_v2/execution"
DEST = ROOT / "experiments/astra/sq3_five_slot_results_20261012_v1"
REVIEW = DEST / "review"
PLAN_SHA = "cd3c415b166159624323bba92e3a8dfd738ccb15ac9d4a65859cccb0ea8b816d"
STUDIES = {"five_slot": EXEC.parent}
SEEDS = [2026100441, 2026100442, 2026100443]
LABELS = original.LABELS
FIELDS = original.FIELDS
read, rows, sha, sig, rel, require, utc, forbid, vote = (
    getattr(original, n)
    for n in ["read", "rows", "sha", "sig", "rel", "require", "utc", "forbid", "vote"]
)


def save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(DEST), "Write outside owned result study")
    data = original.canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, "Preserve previous analysis artifact")
        return
    with path.open("xb") as f:
        f.write(data)


def verify(mapping):
    for name, digest in mapping.items():
        require(sha(ROOT / name) == digest, "Changed input: " + name)


def plan():
    target = DEST / "analysis_plan.json"
    if target.exists():
        verify(read(target)["inputs"])
        print("Existing A14 analysis plan verified.")
        return
    require(sha(EXEC / "plan.json") == PLAN_SHA, "Execution plan changed")
    paths = [
        EXEC / n
        for n in ["plan.json", "schedule.json", "live/complete.json", "ready.json"]
    ]
    paths += [
        EXEC.parent / n
        for n in [
            "scientific_plan_v1.json",
            "STUDY_PROTOCOL.md",
            "roster.jsonl",
            "prompts.jsonl",
            "prepared.json",
            "preparation_plan.json",
            "independent_verification.json",
            "cold_replay_review_v1/qualification.json",
            "cold_replay_review_v1/sensitivity_plan.json",
            "cold_replay_review_v1/PROTOCOL_AMENDMENT.md",
        ]
    ]
    paths += [
        Path(__file__),
        Path(original.__file__),
        ROOT / "analysis/feedback/insertion/scores_earlier.py",
        ROOT / "analysis/feedback/insertion/scores.py",
        ROOT / "analysis/feedback/supplements/scores.py",
        ROOT / "analysis/feedback/balanced/scores.py",
        ROOT / "analysis/feedback/balanced/validate_scores.py",
    ]
    selection = (
        ROOT / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection"
    )
    paths += [
        selection / n
        for n in ["actual_pools.json", "candidate_statuses.json", "summary.json"]
    ]
    inputs = {rel(p): sha(p) for p in paths}
    save(
        target,
        dict(
            created_utc=utc(),
            status="supporting_analysis_details_recorded_before_new_effectiveness_scores",
            inputs=inputs,
            primary="Unchanged frozen A14 endpoints: error_balanced minus reference FPR on 208 benign and FNR on 467 hateful inputs. Three correction pools x three repeats, versus three actual shared-reference repeats. All 12 outputs nonnull per population. PCG64 seeds 2026101014/2026101015; 10000 whole original text-cluster draws; 97.5% two-sided co-primary intervals.",
            primary_sensitivities="All-three-valid-vote mask, stable one-per-cluster representatives using minimum SHA256(A14-representative|key) within each gold-polarity cluster, and opposing missing-as-error/non-error bounds. Use frozen primary implementation plus separate arithmetic verification.",
            common_exclusion="Repeat both primary and all supporting gate/type/binary contrasts after excluding exactly the same six frozen keys in every arm/repeat; retain original labels, seeds, null rules and clusters. Full roster stays primary.",
            supporting_seeds=dict(
                benign=2026101214,
                hateful=2026101215,
                source_stratified=2026101216,
                binary=2026101217,
                types=2026101218,
                source_start=2026101220,
            ),
            supporting="Seven policy groups. All 57 outputs nonnull per population for common comparisons, plus strict-three-vote sensitivity. Mean each run metric, average repeats within pool then equally average pools. Paired 95% intervals and conservative full-roster missing bounds; supporting comparisons remain exploratory. Source-stratified benign sampling retains whole clusters; report each benign source.",
            types="Fixed seven original hate-type labels on original 467 hateful rows only. Report macro-F1, label support/TP/FP/FN, exact-set accuracy, cardinality and gate transitions. No invented target/severity/legal truth. Gate/type/binary all-run own-valid denominators also retained.",
            deletion="Use primary common masks for leave-one-pool and leave-one-execution-batch-out point effects; reference always its three true repeats. Repeat on full and six-key-excluded rosters.",
            mechanism="All 12825 prompt snapshots: verify five ordinary slots, constant definitions/guidelines, delivered correction IDs/polarity/rank/cosine, replaced baseline IDs, same-message and no-delivery reranking. Conditional strata are descriptive, not randomized causal effects. Fixed cosine bins <.4, .4-.5, .5-.6, >=.6; distinguish empty correction delivery. Report pool overlap and unused added records. Compare to reference with matched decoder index solely for descriptive transition accounting.",
            historical="Prior guaranteed-sixth-slot study is contextual only; pool membership and roster matched, but retrieval/delivery, context budget and generation phase differ. Verify message equality where it holds instead of assuming it. No causal delivery-policy estimate or hidden-backend-equivalence claim.",
            operational="Verify final automatic raw-hash-audit chain; rehash every result/receipt, independently reparse every saved raw response and recompute all final votes. Check every failed request record/recovery and original retry limit. The finalizer already rehashed every raw payload/intent/outcome. Reuse that completed full-file traversal transparently, not claim it was newly repeated.",
            limitations="Previously exposed English DEV, simulated existing binary labels, three overlapping source pools. Fewer false positives with more false negatives is a trade-off; uncertainty is not demonstrated non-inferiority. No expert feedback or untouched TEST confirmation. No new model calls, CUDA, regeneration, source edits or outcome-based selection.",
        ),
    )
    print("A14 analysis details frozen before effectiveness scoring.", flush=True)


def review():
    if (REVIEW / "complete.json").exists():
        done = read(REVIEW / "complete.json")
        verify(done["inputs"])
        verify(done["outputs"])
        print("Existing full saved-response review verified.")
        return
    tree = ast.parse(inspect.getsource(original.review))
    imports = guards = 0

    class Bind(ast.NodeTransformer):

        def visit_ImportFrom(self, node):
            nonlocal imports
            if any(
                (
                    n.name == "astra_sq3_additions_recovery_support_v1"
                    for n in node.names
                )
            ):
                imports += 1
                return None
            return node

        def visit_Constant(self, node):
            nonlocal guards
            if node.value == 22536:
                guards += 1
                return ast.copy_location(ast.Constant(38475), node)
            return node

    tree = Bind().visit(tree)
    require(imports == guards == 1, "Unexpected reviewed source structure")
    namespace = (
        vars(original)
        | globals()
        | {"recovery": SimpleNamespace(validate=lambda: {"calls": []})}
    )
    exec(
        compile(ast.fix_missing_locations(tree), str(Path(__file__)), "exec"), namespace
    )
    namespace["review"]()


def probe():
    from analysis.common import schema as isolated

    jobs = read(EXEC / "schedule.json")["stages"][0]["jobs"][:200]
    backends = {}
    count = 0
    began = time.perf_counter()
    for j in jobs:
        ident = dict(
            experiment_sha256=PLAN_SHA,
            condition=j["study"] + "::" + j["condition"],
            repeat=j["repeat"],
            key=j["key"],
        )
        root = EXEC / "live" / j["study"]
        folder = root / "items" / sig(ident)
        value = read(folder / "result.json")
        receipt = read(folder / "complete.json")
        require(
            sha(folder / "result.json")
            == receipt["hashes"][(folder / "result.json").relative_to(root).as_posix()],
            "Changed probe result",
        )
        if j["schema_path"] not in backends:
            backends[j["schema_path"]] = isolated.adapter(ROOT / j["schema_path"])
        for run in value["raw_runs"]:
            for raw in run:
                backends[j["schema_path"]].bundle.parser.parse_response(raw)
                count += 1
    elapsed = time.perf_counter() - began
    print(
        json.dumps(
            dict(
                probe_items=len(jobs),
                raw_replies=count,
                seconds=elapsed,
                rough_full_minutes=elapsed / len(jobs) * 38475 / 60,
                new_model_calls=0,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["plan", "probe", "review"])
    args = p.parse_args()
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ):
        globals()[args.mode]()
