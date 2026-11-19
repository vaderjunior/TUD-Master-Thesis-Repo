from src.hsrag.paths import PROJECT_ROOT
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

ROOT = PROJECT_ROOT
OUT = ROOT / "experiments/astra/sq3_linear_dev_results_20261020_v1"
LINEAR = ROOT / "experiments/astra/sq3_linear_update_20261012_v1"
REVIEW = ROOT / "experiments/astra/sq3_linear_static_review_20261014_v1/review.json"
RESERVED = ROOT / "experiments/astra/sq3_reserved_slot_20261012_v1"
ASSEMBLY = ROOT / "experiments/astra/sq3_reserved_audited_assembly_20261019_v1"
TRANSFER = ROOT / "experiments/astra/sq3_transfer_20261012_v1"
GROUPS = [
    f"{policy}_{composition}"
    for policy in ("error", "correct", "random")
    for composition in ("positive", "balanced")
]
MODES = ["reserved5", "natural5", "extended6", "masked5"]
VARIANTS = ["all_permitted", "matched_gradient"]
CUTOFFS = [0.6, 0.5]
SEEDS = [2026100441, 2026100442, 2026100443]
VIEWS = {
    "common": (False, False, False),
    "excluded": (False, False, True),
    "strict": (True, False, False),
    "strict_excluded": (True, False, True),
    "representative": (False, True, False),
    "representative_excluded": (False, True, True),
}
ARMS = ["linear_baseline", "linear_updated", "rag_reference", "rag_updated"]
ESTIMATES = ARMS + [
    "linear_change",
    "rag_change",
    "difference_in_change",
    "updated_absolute_difference",
]


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def logical(p):
    return Path(str(p).removeprefix("\\\\?\\")).resolve()


def physical(p):
    p = str(logical(p))
    return Path("\\\\?\\" + p) if os.name == "nt" else Path(p)


def rel(p):
    return logical(p).relative_to(ROOT).as_posix()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(physical(p).read_text(encoding="utf-8-sig"))


def rows(p):
    with physical(p).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(p):
    h = hashlib.sha256()
    with physical(p).open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def new(p, value):
    p = logical(p)
    need(p.is_relative_to(OUT), "Write outside this new analysis")
    p.parent.mkdir(parents=True, exist_ok=True)
    with physical(p).open("x", encoding="utf8", newline="\n") as stream:
        stream.write(
            value
            if isinstance(value, str)
            else json.dumps(
                value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        )


def new_array(p, values):
    import numpy as np

    p = logical(p)
    need(p.is_relative_to(OUT), "Array outside this new analysis")
    p.parent.mkdir(parents=True, exist_ok=True)
    with physical(p).open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def forbid(*args, **kwargs):
    raise AssertionError("Offline analysis: network and experiment execution forbidden")


@contextmanager
def offline():
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket.socket, "connect_ex", forbid
    ), patch.object(socket, "create_connection", forbid):
        yield


def verify(bindings, base=ROOT):
    for name, expected in bindings.items():
        p = logical(base / name)
        need(p.is_relative_to(ROOT), "Input outside workspace")
        need(sha(p) == expected, "Changed evidence: " + name)


def running_plan():
    p = TRANSFER / "execution/plan.json"
    need(
        sha(p) == "ca08393271cfe2461966376daebe45d5d156dcf0acc38f4ce0e9f480f8a2ed5d",
        "Running plan changed",
    )
    d = read(p)
    verify(d["source_hashes"])
    return {rel(p): sha(p), **d["source_hashes"]}


def freeze():

    need(not OUT.exists(), "Preserve this analysis version")
    sources = [
        Path(__file__),
        REVIEW,
        LINEAR / "STUDY_PROTOCOL.md",
        LINEAR / "plan.json",
        LINEAR / "static_complete.json",
        LINEAR / "static_schedule.json",
        LINEAR / "roster.jsonl",
        RESERVED / "roster.jsonl",
        RESERVED / "scientific_plan_v1.json",
        ASSEMBLY / "complete.json",
        ROOT
        / "experiments/astra/sq3_reserved_part10_results_20261020_v1/complete.json",
    ]
    spec = {
        "created_utc": utc(),
        "scope": "Supporting exposed-English-DEV paired binary analysis only",
        "timing": "Implementation details fixed after completed RAG findings were known, before "
        "opening/classifying the joint classifier/RAG outcomes in this analysis. "
        "This is not a new prospective confirmatory family.",
        "parent_protocol": rel(LINEAR / "STUDY_PROTOCOL.md"),
        "inputs": 675,
        "expected_benign": 208,
        "expected_hateful": 467,
        "expected_clusters": 674,
        "groups": GROUPS,
        "pool_seeds": SEEDS,
        "variants": VARIANTS,
        "cutoffs_strictly_greater_than": CUTOFFS,
        "principal_specification": {
            "variant": "all_permitted",
            "cutoff": 0.6,
            "mode": "reserved5",
            "group": "error_balanced",
        },
        "endpoints": ["FPR", "FNR"],
        "arms": ARMS,
        "estimates": ESTIMATES,
        "views": VIEWS,
        "modes": MODES,
        "estimation": "Common input mask across reference and all three RAG pool arms and repeats. "
        "Classifier is deterministic: baseline once, three updated fits averaged. "
        "Reference retains its three real outputs, not nine independent outputs. "
        "Use integer errors scaled to ninths for exact point arithmetic.",
        "uncertainty": {
            "confidence": 0.95,
            "draws": 10000,
            "generator": "PCG64",
            "unit": "whole original text cluster, item weighted",
            "seed_rule": "first 8 SHA256 bytes as big-endian integer of "
            "'A18-DEV-v1|' + mode/group/view/endpoint",
            "coverage": "All reserved5 groups, variants, thresholds and six views. "
            "All 95% intervals are supporting, unadjusted, conditional "
            "on these fixed pools and decoder repeats; no winner selection.",
        },
        "other_modes": "All natural5/extended6/masked5 groups and six masks are descriptive points "
        "only. Masked5 intentionally withholds RAG labels and is not an "
        "equal-information method contest. Reference4 is already a context "
        "control in the completed RAG study, not a classifier-update arm.",
        "diagnostics": [
            "opposing RAG missing-output bounds on every original eligible denominator",
            "source breakdowns, all individual pool deltas and leave-one-pool means",
            "leave-one-original-execution-batch points using an all-arm input exclusion",
            "same fixed mask for each diagnostic's methods",
            "every DEV text probability, cutoff decision, pool update and RAG repeat retained",
            "fit class counts, rebalanced class weights and recorded fit resources",
        ],
        "boundaries": [
            "No 15,758-row broad-transfer outcome scoring",
            "No cumulative-round fit or trajectory until actual audited source pools exist",
            "No new call, model-list GET, fit, embedding, annotation or original-file mutation",
            "Binary head only; not a complete multi-head result",
            "Source pools were RAG-selected, not independently optimized classifier feedback",
            "Small marginal update: 40 appended labels versus 55,753/50,173 prior TRAIN labels",
            "No new bootstrap family-adjusted primary claim; broader four endpoints stay pending",
        ],
        "input_bindings": {rel(p): sha(p) for p in sources},
        "running_dependencies_before": running_plan(),
    }
    new(OUT / "analysis_plan.json", spec)
    print(
        json.dumps(
            {
                "status": "supporting_analysis_scope_frozen",
                "path": rel(OUT / "analysis_plan.json"),
            }
        )
    )


def load():
    import numpy as np

    spec = read(OUT / "analysis_plan.json")
    verify(spec["input_bindings"])
    a = read(ASSEMBLY / "complete.json")
    need(
        sha(ASSEMBLY / "complete.json")
        == "f6bbca18ca59dcfcf7a639399acd8367bcdbcc27cdb424143afe0254abaf7d03",
        "Audited assembly receipt changed",
    )
    need(
        a["status"] == "complete_from_automatically_audited_receipts_unscored",
        "Assembly provenance",
    )
    verify(a["artifacts"])

    old = read(REVIEW)
    need(
        old["unique_fits"] == 38
        and old["updated_fits"] == 36
        and old["probabilities_independently_reconstructed"] == 624454,
        "Prior fit proof differs",
    )
    for p in (
        LINEAR / "plan.json",
        LINEAR / "static_complete.json",
        LINEAR / "roster.jsonl",
        LINEAR / "static_schedule.json",
        RESERVED / "roster.jsonl",
    ):
        need(
            old["bindings"].get(rel(p)) == sha(p),
            "Static proof does not bind " + rel(p),
        )
    roster = list(rows(RESERVED / "roster.jsonl"))
    lr = list(rows(LINEAR / "roster.jsonl"))
    index = {r["key"]: i for i, r in enumerate(lr) if r["population"] == "a15_dev"}
    need(
        len(roster) == len(index) == 675 and len({r["key"] for r in roster}) == 675,
        "DEV union differs",
    )
    indices = [index[r["key"]] for r in roster]
    for r, i in zip(roster, indices):
        l = lr[i]
        need(
            (r["id"], r["text"], r["gold"]["hate"], r["cluster"])
            == (l["id"], l["text"], l["gate"], l["inference_cluster"]),
            "Text/gold/cluster alignment",
        )
    need(
        sum(r["gold"]["hate"] for r in roster) == 467
        and len({r["cluster"] for r in roster}) == 674,
        "Frozen label/cluster support",
    )
    scientific = read(RESERVED / "scientific_plan_v1.json")
    conditions = scientific["conditions"]
    qi = {c: i for i, c in enumerate(conditions)}
    ki = {r["key"]: i for i, r in enumerate(roster)}
    predictions = np.full((675, 74, 3), -2, dtype=np.int8)
    votes = np.zeros_like(predictions)
    stages = np.zeros_like(predictions)
    total = 0
    for name in a["artifacts"]:
        if not name.endswith(".jsonl"):
            continue
        for p in rows(ROOT / name):
            loc = ki[p["key"]], qi[p["condition"]], p["repeat"]
            need(predictions[loc] == -2, "Duplicate RAG identity")
            v = p["n_valid"]
            need(
                0 <= v <= 3 and (p["result"] is None) == (v == 0),
                "RAG validity contract",
            )
            predictions[loc] = -1 if v == 0 else int(p["result"]["hate"])
            votes[loc] = v
            stages[loc] = int(p["stage"].rsplit("_", 1)[-1])
            total += 1
    need(
        total == 149850
        and (predictions >= -1).all()
        and int((predictions < 0).sum()) == 84,
        "Incomplete original matrix",
    )
    done = read(LINEAR / "static_complete.json")
    entries = read(LINEAR / "static_schedule.json")["entries"]
    names = [e["condition"] for e in entries]
    need(
        set(names)
        == {"reference"} | {g + "_s" + str(s) for g in GROUPS for s in SEEDS},
        "Static source pool schedule",
    )
    probabilities = {}
    job_summary = {}
    bindings = {}
    for variant in VARIANTS:
        probabilities[variant] = {}
        for entry in entries:
            name = variant + "/" + entry["condition"]
            j = done["jobs"][name]
            p = ROOT / j["receipt"]
            need(
                sha(p) == j["sha256"] == old["bindings"][rel(p)], "Fit receipt binding"
            )
            d = read(p)
            verify(d["outputs"], p.parent)
            need(
                sorted(entry["ids"]) == j["source_ids"] == d["identity"]["source_ids"],
                "Correction IDs differ",
            )
            attempt = p.parent / d["attempt"]
            prob = np.load(
                physical(attempt / "probabilities.npy"),
                mmap_mode="r",
                allow_pickle=False,
            )
            need(prob.shape == (16433, 1), "Probability shape")
            probabilities[variant][entry["condition"]] = np.asarray(
                prob[indices, 0]
            ).copy()
            for cutoff, suffix in ((0.5, "05"), (0.6, "06")):
                dec = np.load(
                    physical(attempt / f"decisions_{suffix}.npy"),
                    mmap_mode="r",
                    allow_pickle=False,
                )
                need(
                    np.array_equal(dec[indices, 0], prob[indices, 0] > cutoff),
                    "Cutoff mismatch",
                )
            job_summary[name] = old["jobs"][rel(p)]
            bindings[rel(p)] = sha(p)
            bindings.update({rel(p.parent / k): v for k, v in d["outputs"].items()})
    return {
        "roster": roster,
        "science": scientific,
        "conditions": conditions,
        "qi": qi,
        "rag": predictions,
        "votes": votes,
        "stages": stages,
        "linear": probabilities,
        "jobs": job_summary,
        "bindings": bindings,
        "review_resources": old["resources"],
    }


def seed_for(name):
    return int.from_bytes(
        hashlib.sha256(("A18-DEV-v1|" + name).encode()).digest()[:8], "big"
    )


def representative_indices(roster, eligible):
    grouped = defaultdict(list)
    for i in eligible:
        grouped[roster[i]["cluster"]].append(i)
    chosen = {
        min(
            g,
            key=lambda i: hashlib.sha256(
                ("A15-representative|" + roster[i]["key"]).encode()
            ).hexdigest(),
        )
        for g in grouped.values()
    }
    return [i for i in eligible if i in chosen]


def select(data, mode, group, view, endpoint):
    import numpy as np

    strict, representatives, excluded = VIEWS[view]
    truth = endpoint == "FNR"
    roster = data["roster"]
    exclusion = set(data["science"]["mandatory_excluded_keys"]) if excluded else set()
    eligible = [
        i
        for i, r in enumerate(roster)
        if r["gold"]["hate"] == truth and r["key"] not in exclusion
    ]
    if representatives:
        eligible = representative_indices(roster, eligible)
    conditions = ["reference5"] + [mode + "__" + group + "_s" + str(s) for s in SEEDS]
    ci = [data["qi"][c] for c in conditions]
    p = data["rag"][:, ci, :]
    v = data["votes"][:, ci, :]
    valid = (p >= 0) & ((v == 3) if strict else True)
    common = [i for i in eligible if valid[i].all()]
    return eligible, common, p, valid, ci


def arm_units(data, group, variant, cutoff, p, truth):

    import numpy as np

    base = data["linear"][variant]["reference"] > cutoff
    updates = np.column_stack(
        [data["linear"][variant][group + "_s" + str(s)] > cutoff for s in SEEDS]
    )
    return np.column_stack(
        [
            (base != truth).astype(np.int64) * 9,
            (updates != truth).sum(axis=1) * 3,
            (p[:, 0, :] != truth).sum(axis=1) * 3,
            (p[:, 1:, :] != truth).sum(axis=(1, 2)),
        ]
    )


def expand(values):
    import numpy as np

    a = np.asarray(values, dtype=float)
    return np.concatenate(
        [
            a,
            (a[..., 1] - a[..., 0])[..., None],
            (a[..., 3] - a[..., 2])[..., None],
            (a[..., 1] - a[..., 0] - a[..., 3] + a[..., 2])[..., None],
            (a[..., 1] - a[..., 3])[..., None],
        ],
        axis=-1,
    )


def bootstrap(roster, common, matrix, seed):

    import numpy as np

    labels = sorted({roster[i]["cluster"] for i in common})
    pos = {x: i for i, x in enumerate(labels)}
    totals = np.zeros((len(labels), matrix.shape[1]), dtype=np.int64)
    sizes = np.zeros(len(labels), dtype=np.int64)
    for i in common:
        j = pos[roster[i]["cluster"]]
        totals[j] += matrix[i]
        sizes[j] += 1
    rng = np.random.Generator(np.random.PCG64(seed))
    distribution = np.empty((10000, matrix.shape[1]), dtype=float)
    for start in range(0, 10000, 100):
        ix = rng.integers(0, len(labels), size=(100, len(labels)))
        distribution[start : start + 100] = totals[ix].sum(axis=1) / (
            9 * sizes[ix].sum(axis=1)[:, None]
        )
    return distribution, len(labels)


def missing_bounds(units, p, valid, eligible, truth):

    import numpy as np

    known_errors = ((p != truth) & valid).astype(int)
    low = np.column_stack(
        [known_errors[:, 0].sum(axis=1) * 3, known_errors[:, 1:].sum(axis=(1, 2))]
    )
    miss = np.column_stack(
        [(~valid[:, 0]).sum(axis=1) * 3, (~valid[:, 1:]).sum(axis=(1, 2))]
    )
    lu = units[eligible, 1].sum() / (9 * len(eligible))
    ld = (units[eligible, 1] - units[eligible, 0]).sum() / (9 * len(eligible))
    lo = low[eligible].sum(axis=0) / (9 * len(eligible))
    hi = (low + miss)[eligible].sum(axis=0) / (9 * len(eligible))
    return {
        "rag_change": [float(lo[1] - hi[0]), float(hi[1] - lo[0])],
        "difference_in_change": [float(ld - hi[1] + lo[0]), float(ld - lo[1] + hi[0])],
        "updated_absolute_difference": [float(lu - hi[1]), float(lu - lo[1])],
        "missing_real_outputs": [
            int((~valid[eligible, 0]).sum()),
            int((~valid[eligible, 1:]).sum()),
        ],
    }


def diagnostics(
    data, mode, group, view, endpoint, common, p, ci, variant, cutoff, units
):
    import numpy as np

    roster = data["roster"]
    truth = endpoint == "FNR"
    parts = {}
    for source in sorted({r["source"] for r in roster}):
        ids = [i for i in common if roster[i]["source"] == source]
        parts[source] = {
            "n": len(ids),
            "estimates": (
                dict(zip(ESTIMATES, expand(units[ids].mean(axis=0) / 9).tolist()))
                if ids
                else None
            ),
        }
    pools = []
    base = data["linear"][variant]["reference"] > cutoff
    qb = (p[common, 0] != truth).mean()
    for j, s in enumerate(SEEDS):
        u = data["linear"][variant][group + "_s" + str(s)] > cutoff
        lb = float((base[common] != truth).mean())
        lu = float((u[common] != truth).mean())
        qu = float((p[common, j + 1] != truth).mean())
        pools.append(
            {
                "seed": s,
                "n": len(common),
                "estimates": dict(zip(ESTIMATES, expand([lb, lu, qb, qu]).tolist())),
                "classifier_fixes": int(
                    ((base[common] != truth) & (u[common] == truth)).sum()
                ),
                "classifier_regressions": int(
                    ((base[common] == truth) & (u[common] != truth)).sum()
                ),
            }
        )
    omissions = []
    for omit in range(3):
        keep = [q for j, q in enumerate(pools) if j != omit]
        omissions.append(
            {
                "omitted_seed": SEEDS[omit],
                "n": len(common),
                "estimates": {
                    k: float(np.mean([q["estimates"][k] for q in keep]))
                    for k in ESTIMATES
                },
            }
        )
    batch_omissions = []
    for b in range(1, 17):
        keep = [i for i in common if not (data["stages"][i, ci, :] == b).any()]
        batch_omissions.append(
            {
                "omitted_batch": b,
                "n": len(keep),
                "estimates": (
                    dict(zip(ESTIMATES, expand(units[keep].mean(axis=0) / 9).tolist()))
                    if keep
                    else None
                ),
            }
        )
    return {
        "sources": parts,
        "pools": pools,
        "leave_one_pool": omissions,
        "leave_one_batch": batch_omissions,
        "note": "Descriptive fixed-mask diagnostics; omissions are not confidence intervals.",
    }


def analyze():
    import numpy as np
    from threadpoolctl import threadpool_limits

    need(not (OUT / "analysis.json").exists(), "Preserve existing scored output")
    data = load()
    new(
        OUT / "started.json",
        {"utc": utc(), "plan_sha256": sha(OUT / "analysis_plan.json")},
    )
    roster = data["roster"]
    settings = [(v, c) for v in VARIANTS for c in CUTOFFS]
    results = []
    diagnostic = []
    with threadpool_limits(limits=1):
        for mode in MODES:
            for group in GROUPS:
                for view in VIEWS:
                    for endpoint in ("FPR", "FNR"):
                        name = "/".join((mode, group, view, endpoint))
                        eligible, common, p, valid, ci = select(
                            data, mode, group, view, endpoint
                        )
                        need(common, "Unexpected empty main/source-unfiltered cohort")
                        units = [
                            arm_units(data, group, v, c, p, endpoint == "FNR")
                            for v, c in settings
                        ]
                        combined = np.concatenate(units, axis=1)
                        draws, nclusters = (
                            bootstrap(roster, common, combined, seed_for(name))
                            if mode == "reserved5"
                            else (None, len({roster[i]["cluster"] for i in common}))
                        )
                        drawfile = None
                        if draws is not None:
                            drawfile = (
                                OUT / "draws" / (name.replace("/", "__") + ".npy")
                            )
                            new_array(drawfile, draws)
                        for setting, ((variant, cutoff), u) in enumerate(
                            zip(settings, units)
                        ):
                            point = expand(u[common].sum(axis=0) / (9 * len(common)))
                            result = {
                                "id": f"{name}/{variant}/gt{cutoff}",
                                "mode": mode,
                                "group": group,
                                "view": view,
                                "endpoint": endpoint,
                                "variant": variant,
                                "cutoff": cutoff,
                                "eligible_n": len(eligible),
                                "common_n": len(common),
                                "clusters": nclusters,
                                "common_ids": [roster[i]["key"] for i in common],
                                "invalid_ids": [
                                    roster[i]["key"]
                                    for i in eligible
                                    if i not in set(common)
                                ],
                                "integer_arm_totals_ninths": u[common]
                                .sum(axis=0)
                                .tolist(),
                                "estimates": dict(zip(ESTIMATES, point.tolist())),
                                "missing_bounds": missing_bounds(
                                    u, p, valid, eligible, endpoint == "FNR"
                                ),
                            }
                            if draws is not None:
                                dd = expand(draws[:, setting * 4 : (setting + 1) * 4])
                                intervals = np.quantile(
                                    dd, [0.025, 0.975], axis=0, method="linear"
                                ).T
                                result.update(
                                    interval95=dict(zip(ESTIMATES, intervals.tolist())),
                                    draws_path=rel(drawfile),
                                    draws_column_start=setting * 4,
                                    seed=seed_for(name),
                                    draws=10000,
                                )
                            results.append(result)
                            if mode in ("reserved5", "natural5") and view in (
                                "common",
                                "strict_excluded",
                            ):
                                diagnostic.append(
                                    {
                                        "id": result["id"],
                                        **diagnostics(
                                            data,
                                            mode,
                                            group,
                                            view,
                                            endpoint,
                                            common,
                                            p,
                                            ci,
                                            variant,
                                            cutoff,
                                            u,
                                        ),
                                    }
                                )
                print("Scored supporting DEV", mode, group, flush=True)
    new(
        OUT / "analysis.json",
        {
            "status": "supporting_DEV_scores_require_independent_review",
            "results": results,
            "settings_order": settings,
            "new_calls": 0,
            "new_fits": 0,
            "ongoing_transfer_predictions_read": 0,
        },
    )
    new(OUT / "diagnostics.json", diagnostic)
    new(
        OUT / "fit_resources.json",
        {
            "jobs": data["jobs"],
            "recorded_resources": data["review_resources"],
            "new_fits": 0,
            "new_embeddings": 0,
            "timing_limitation": "Recorded concurrent laptop measurements; cached "
            "embeddings and inherited RAG source screening are not free.",
        },
    )
    casepath = OUT / "all_DEV_cases.jsonl"
    with physical(casepath).open("x", encoding="utf8", newline="\n") as stream:
        for i, r in enumerate(roster):
            case = dict(r)
            case["linear"] = {
                v: {c: float(p[i]) for c, p in dd.items()}
                for v, dd in data["linear"].items()
            }
            case["rag"] = {
                c: {
                    "hate": [None if x < 0 else bool(x) for x in data["rag"][i, j]],
                    "n_valid": data["votes"][i, j].tolist(),
                    "stage": data["stages"][i, j].tolist(),
                }
                for j, c in enumerate(data["conditions"])
            }
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    new(OUT / "evidence_bindings.json", data["bindings"])
    need(
        running_plan()
        == read(OUT / "analysis_plan.json")["running_dependencies_before"],
        "Running dependencies changed",
    )
    print(
        json.dumps(
            {
                "results": len(results),
                "diagnostic_groups": len(diagnostic),
                "bootstrap_files": len(list((OUT / "draws").glob("*.npy"))),
            }
        )
    )


def synthetic_check():

    import numpy as np

    r = [{"cluster": "A"}, {"cluster": "A"}, {"cluster": "B"}]
    m = np.array([[0, 3, 3, 9], [0, 0, 0, 3], [9, 6, 0, 0]])
    d, k = bootstrap(r, [0, 1, 2], m, 7)
    expected = [np.mean(m[:2], axis=0) / 9, np.mean(m, axis=0) / 9, m[2] / 9]
    need(
        k == 2 and all(any(np.allclose(row, x) for x in expected) for row in d),
        "Whole-cluster weighted sampling failed",
    )
    need(np.allclose(expand([0.2, 0.1, 0.3, 0.25])[-2:], [-0.05, -0.15]), "Delta signs")
    p = np.array([[[0, -1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0]]])
    v = p >= 0
    b = missing_bounds(np.array([[0, 3, 0, 0]]), p, v, [0], False)
    need(
        np.allclose(b["difference_in_change"], [-1 / 9, 2 / 9]),
        "Opposing missing bounds",
    )
    need(b["missing_real_outputs"] == [1, 0], "Reference output multiplicity")

    need(
        (np.array([0.6, 0.60001, 0.5]) > 0.6).tolist() == [False, True, False],
        "Strict cutoff",
    )
    print("Passed hand-counted cluster, sign, missingness, repeat and cutoff checks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "test", "analyze"])
    args = parser.parse_args()
    with offline():
        {"freeze": freeze, "test": synthetic_check, "analyze": analyze}[args.mode]()


if __name__ == "__main__":
    main()
