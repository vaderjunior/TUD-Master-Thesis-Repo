import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch
from analysis.feedback.supplements import archive as original

ROOT = original.ROOT
EXEC = ROOT / "experiments/astra/sq3_balanced_corrections_20261006_v1/execution"
DEST = ROOT / "experiments/astra/sq3_balanced_results_20261010_v1"
REVIEW = DEST / "review"
PLAN_SHA = "bd4d60cd2d8ccd7b5292ba6e6dea2d75f89053f6820a528de8fb3fcea00585e5"
STUDIES = {"balanced_corrections": EXEC.parent}
LABELS = original.LABELS
FIELDS = original.FIELDS
SEEDS = [2026100441, 2026100442, 2026100443]
read, rows, sha, sig, rel, require, utc, forbid, vote = (
    getattr(original, n)
    for n in ["read", "rows", "sha", "sig", "rel", "require", "utc", "forbid", "vote"]
)


def save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(DEST), "Write outside isolated result study")
    data = original.canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, "Preserve existing analysis artifact")
        return
    with path.open("xb") as f:
        f.write(data)


def verify(mapping):
    for name, digest in mapping.items():
        require(sha(ROOT / name) == digest, "Changed source: " + name)


def plan():
    path = DEST / "analysis_plan.json"
    if path.exists():
        verify(read(path)["inputs"])
        print("Existing analysis protocol verified.")
        return
    require(sha(EXEC / "plan.json") == PLAN_SHA, "Changed execution plan")
    names = [
        "plan.json",
        "schedule.json",
        "live/complete.json",
        "recovery_20261006_v1/manifest.json",
        "recovery_20261006_v1/ready.json",
    ]
    inputs = {rel(EXEC / n): sha(EXEC / n) for n in names}
    for name in [
        "scientific_plan_v1.json",
        "STUDY_PROTOCOL.md",
        "roster.jsonl",
        "prompts.jsonl",
        "choices.jsonl",
        "summary.json",
        "complete.json",
        "prepared.json",
        "preparation_plan.json",
        "independent_verification.json",
    ]:
        p = EXEC.parent / name
        inputs[rel(p)] = sha(p)
    selection = (
        ROOT / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection"
    )
    for p in sorted(selection.glob("*.json")):
        inputs[rel(p)] = sha(p)
    for p in [
        Path(__file__),
        Path(original.__file__),
        ROOT / "analysis/feedback/supplements/scores.py",
    ]:
        inputs[rel(p)] = sha(p)
    save(
        path,
        dict(
            created_utc=utc(),
            status="analysis_details_recorded_before_new_effectiveness_scores",
            inputs=inputs,
            primary="Unchanged scientific_plan_v1: error_balanced minus error_positive FPR on 208 benign and FNR on 467 hateful texts; equal three-pool/three-repeat averages, common all-18-nonnull mask separately per polarity. PCG64, 10000 whole-text-cluster draws; seeds 202610044/202610045, 97.5% two-outcome family intervals. No non-inferiority claim from nonsignificant recall loss.",
            averaging="Compute each run metric first, average three repeats inside each pool and then three pools. Reference has three actual repeats, not three independently replicated pools.",
            sensitivity="Both primaries: all-three-valid-vote mask; one stable representative per original overlap cluster selected before validity masks; full original-roster null-as-error per arm and opposing missing-output bounds. Keep all 12 original nulls. Bootstrap same primary seeds for primary sensitivities.",
            representative_rule="Minimum SHA256(20261010-balanced-representative|key) across original 675-text cluster, selected before validity masks; then intersect population.",
            supporting="Seven policy groups including reference, common all-57-nonnull mask per polarity for error rates. Supporting 95% intervals, PCG64 seeds 202610101 benign and 202610102 hateful. All pairwise differences supporting, never promoted into the primary family. Source-stratified benign pooled resampling seed 202610103; per-source rate intervals seed 202610110+sorted-source index. Equal policy weighting does not imply independent pools.",
            combined_binary="Common all-57-nonnull mask on full roster; balanced accuracy and binary macro-F1 computed per setting then averaged. 10000 original text-cluster bootstrap draws, seed 202610104. Artificial 467:208 prevalence, not deployment accuracy. Full-roster opposite-bound missing-label assignments separately per run.",
            fine_labels="Only seven hate-type labels have original annotated gold beyond the gate; evaluate on original 467 hateful texts with delivered (gate-normalised) labels. Common all-57-nonnull mask, fixed seven-label set, F1 per run then equal averages; supporting seed 202610105. No target/severity/legal truth is inferred from missing annotations. Report per-label support/TP/FP/FN, exact match, predicted label count, gate-only versus already-gate-positive transitions.",
            supporting_robustness="All-pool/repeat values, all-three-vote common support, leave-one-pool-out and leave-one-execution-batch-out primary point differences; source-specific benign comparisons. Such deletion/stratum diagnostics are supporting, with selection now fixed before effectiveness scores.",
            mechanism="Compare matched positive/balanced pairs by seed and repeat; report retained versus replaced correction identities, delivered benign/hateful polarity, cosine, shared-pool halves, and reference-relative beneficial/harmful gate transitions. Delivery and similarity strata are observational, not randomized causal effects. No outcome-selected threshold or best-pool selection.",
            continuity="Fresh no-correction messages must match the historical 675-text baseline; old/new decoder phase comparisons are descriptive with previous exposure and unverified hidden backend checkpoint, not a treatment effect.",
            review="Bind final automatic full-file audit chain; rehash every result/receipt, reparse every saved successful raw response with original schema, independently vote, verify every failed request and explicit eight-call amendment. The user-completed finalizer already reread all raw intent/outcome/payload hashes. Do not unnecessarily repeat that whole disk traversal.",
            operations="Retain all retries, repairs, nulls, model aliases, response IDs and token/request counts. Separate recorded active request time from restart/outage wall duration; hidden remote work behind failed transport remains unknown.",
            exclusions="No new model calls, CUDA, model selection, source resampling, outcome-driven prompt replacement or human annotation inference. Earlier studies remain preserved.",
        ),
    )
    print("Balanced analysis details recorded before scoring.", flush=True)


def review():
    from analysis.feedback.balanced import recovery_support as recovery

    tree = ast.parse(inspect.getsource(original.review))
    imports = 0
    guards = 0

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
    require(imports == guards == 1, "Unexpected review template structure")
    ns = vars(original) | globals() | {"recovery": recovery}
    exec(compile(ast.fix_missing_locations(tree), str(Path(__file__)), "exec"), ns)
    ns["review"]()


def probe(limit=200):

    from analysis.common import schema as isolated

    schedule = read(EXEC / "schedule.json")
    jobs = schedule["stages"][0]["jobs"][:limit]
    backends = {}
    raw_count = 0
    start = time.perf_counter()
    for j in jobs:
        ident = dict(
            experiment_sha256=PLAN_SHA,
            condition=j["study"] + "::" + j["condition"],
            repeat=j["repeat"],
            key=j["key"],
        )
        root = EXEC / "live" / j["study"]
        folder = root / "items" / sig(ident)
        receipt = read(folder / "complete.json")
        data = read(folder / "result.json")
        require(
            sha(folder / "result.json")
            == receipt["hashes"][(folder / "result.json").relative_to(root).as_posix()],
            "Changed probe item",
        )
        if j["schema_path"] not in backends:
            backends[j["schema_path"]] = isolated.adapter(ROOT / j["schema_path"])
        for raw_run in data["raw_runs"]:
            for text in raw_run:
                backends[j["schema_path"]].bundle.parser.parse_response(text)
                raw_count += 1
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            dict(
                probe_items=len(jobs),
                raw_replies=raw_count,
                seconds=elapsed,
                estimated_full_minutes=elapsed / len(jobs) * 38475 / 60,
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
        {"plan": plan, "probe": probe, "review": review}[args.mode]()
