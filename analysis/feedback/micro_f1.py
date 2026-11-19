from analysis.common import micro_f1_files as c
from analysis.common import micro_f1_resampling as h
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np

BASE = c.ROOT / "experiments/astra/micro_f1_sq3_20261113_v1"
PRIOR = c.ROOT / "experiments/astra/micro_f1_sq1_components_guidelines_20261113_v1"
B = c.ROOT / "experiments/astra"
STUDIES = [
    "feedback",
    "relevance",
    "explanations",
    "balanced",
    "ordinary_insertion",
    "reserved",
    "transfer",
    "rounds",
    "human",
]
TYPE = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
TARGET = [
    "race",
    "religion",
    "gender",
    "sexual_orientation",
    "disability",
    "national_origin",
    "other",
]


def save(path, value):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Write outside new Part 7 evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            value, f, indent=2, ensure_ascii=False, allow_nan=False, default=h.default
        )
        f.write("\n")


def arrays(path, **value):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Array outside Part 7")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **value)


def freeze():
    c.require(not BASE.exists(), "Preserve existing Part 7 work")
    reg = c.read(c.BASE / "comparison_registry_v2.json")
    seal = c.read(c.BASE / "complete.json")
    for name in [
        "comparison_registry_v2.json",
        "paired_contrast_manifest.json",
        "scoring_contract.json",
    ]:
        path = c.BASE / name
        c.require(
            c.sha(path) == seal["artifact_sha256"][c.rel(path)], "Frozen setup changed"
        )
    old = c.read(PRIOR / "source_preservation.json")
    for name, digest in old["source_sha256"].items():
        c.require(c.sha(c.ROOT / name) == digest, "Prior source changed: " + name)
    changes = dict()
    BASE.mkdir(parents=True)
    save(
        BASE / "preservation_before.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            old_sources=old["source_sha256"],
        ),
    )
    save(
        BASE / "analysis_plan.json",
        dict(
            status="Part 7 scope fixed before new micro calculations",
            part=7,
            units=[u for u in reg["units"] if u["study"] in STUDIES],
            families={
                k: reg["new_micro_families"][k]
                for k in ["feedback", "sq3_followups", "transfer", "rounds"]
            },
            scope="All 67 registered units. Expand the reserved-slot unit into all 56 saved type contrasts (full and six-key exclusion); the human unit into all eight cohorts and three separate annotation targets. Retain the saved strict/representative balanced and ordinary-insertion views as supporting checks.",
            inference="Original paired masks, seeds and 10000 item/whole-cluster draws. Fixed pools, streams and decoder repeats; average run scores, never pool their predictions or duplicate the reference. New supplementary families: 1/3/3/9 questions. Original macro families remain unchanged; other comparisons nominal95. Human scores are descriptive without intervals.",
            metrics="Original macro, full seven-label micro and micro on original macro labels. Later low-support exclusions remain fixed. Unknown fine gold stays ineligible; empty known sets remain valid. All outputs retain their predicted hate gates.",
            boundaries="Binary hate, FPR/FNR, ordinal severity and same-feedback binary classifiers remain unchanged. Focal-category restoration averages seven distinct interventions and is not one ordinary multilabel predictor; retain its original estimand instead of constructing misleading micro across interventions.",
            diagnostics="Own/common counts and metrics, exact sets, gate misses, cardinality, per-label weights/contributions, original full-roster missing-output bounds, repeat and pool/stream variation, all saved strict/exclusion/representative views and fixed-cohort round trajectories. Source uptake remains separate from transfer. Reviewer A/B/dataset gold never merged.",
            additional_sensitivity="For early feedback/relevance/explanation and reserved contrasts, add a symmetric strict-three-vote point sensitivity only; no new inferential family. Existing saved balanced/ordinary/transfer/round strict views retain their original intervals. Cross-study robustness remains Part 8.",
            verification="Replay every original registered macro and every applicable stored micro score and interval. Replay saved bootstrap arrays where present. Independently calculate sklearn points, expanded sampled counts and alternate-denominator bootstrap scores; check label-scope, weighting, masks and failure-bound fixtures. Preserve original evidence and all current thesis hashes.",
            writing="Append plain-English findings and exact insertion instructions M62 onward to the existing handover. Keep primary macro plots/results, disclose every metric-sensitive inference including adverse results, do not edit LaTeX.",
            model_calls=0,
            training=0,
            retrieval=0,
            latex_edits=0,
        ),
    )


def slug(d):
    return hashlib.sha256(d["id"].encode()).hexdigest()[:20]


def group_score(d, scores, repeat=None, omit=None, pool=None):
    out = []
    for name, js in d["groups"].items():
        keep = [
            j
            for j in js
            if (repeat is None or d["settings"][j][-1] == repeat)
            and (omit is None or d["settings"][j][-1] != omit)
        ]
        if pool is not None:
            pools = sorted({d["settings"][j][-2] for j in keep})
            if len(pools) > 1:
                keep = [j for j in keep if d["settings"][j][-2] == pools[pool]]
        c.require(keep, "Empty group")
        out.append(scores[..., keep, :].mean(-2))
    return np.stack(out, -2)


def column_score(d, groups):
    names = list(d["groups"])
    return np.stack(
        [
            sum(
                (
                    coef * groups[..., names.index(name), :]
                    for name, coef in weights.items()
                )
            )
            for weights in d["columns"].values()
        ],
        -2,
    )


def boot(d, draws=10000):
    pos = np.flatnonzero(d["mask"])
    raw = h.encode(d["gold"], d["pred"], d["valid"])[:, pos]
    keys = (
        list(range(len(pos)))
        if d["sampling"] == "item"
        else [d["clusters"][i] for i in pos]
    )
    names = keys if d["sampling"] == "item" else sorted(set(keys))
    loc = {k: i for i, k in enumerate(names)}
    grouped = np.zeros((len(names), len(d["settings"]), len(d["labels"]), 3), np.int64)
    for j, key in enumerate(keys):
        grouped[loc[key]] += raw[:, j]
    flat = grouped.reshape(len(names), -1).astype(float)
    rng = np.random.default_rng(d["seed"])
    output = np.empty((draws, len(d["columns"]), 3))
    groupout = np.empty((draws, len(d["groups"]), 3))
    dig = hashlib.sha256()
    nchecks = 0
    sampled = []
    score_values = 0
    for start in range(0, draws, 64):
        n = min(64, draws - start)
        picks = rng.integers(0, len(names), (n, len(names)), dtype=np.int64)
        dig.update(picks.astype("<i4").tobytes())
        freq = np.stack([np.bincount(p, minlength=len(names)) for p in picks]).astype(
            float
        )
        counts = (freq @ flat).reshape(n, len(d["settings"]), len(d["labels"]), 3)
        h.equal(counts[0], grouped[picks[0]].sum(0), "Direct cluster expansion")
        nchecks += 1
        vals = h.values(counts, d["macro_indices"])
        tp = counts[..., 0]
        pp = tp + counts[..., 1]
        aa = tp + counts[..., 2]
        den = pp + aa
        f = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den != 0)
        full = den.sum(-1)
        rest = den[..., d["macro_indices"]].sum(-1)
        other = np.stack(
            [
                f[..., d["macro_indices"]].mean(-1),
                np.divide(
                    2 * tp.sum(-1), full, out=np.zeros_like(full), where=full != 0
                ),
                np.divide(
                    2 * tp[..., d["macro_indices"]].sum(-1),
                    rest,
                    out=np.zeros_like(rest),
                    where=rest != 0,
                ),
            ],
            -1,
        )
        h.equal(vals, other, "Independent bootstrap denominator arithmetic")
        score_values += vals.size
        gr = group_score(d, vals)
        groupout[start : start + n] = gr
        output[start : start + n] = column_score(d, gr)
        if start in [0, 64, 4992, 9984]:
            sampled.append(
                dict(
                    draw=start,
                    picks=picks[0].tolist(),
                    counts=counts[0],
                    setting_scores=vals[0],
                )
            )
    return dict(
        columns=output,
        groups=groupout,
        n_clusters=len(names),
        cluster_order=[str(k) for k in names],
        pick_stream_sha256=dig.hexdigest(),
        expanded_checks=nchecks,
        independent_setting_metric_values=score_values,
        sampled=sampled,
    )


def bounds(d, own, weights):
    value = np.zeros((3, 2))
    for name, w in weights.items():
        pair = np.mean(
            [
                [s["full_roster_missing_bounds"][m] for m in h.METHODS]
                for s in [own[j] for j in d["groups"][name]]
            ],
            0,
        )
        value += w * (pair if w > 0 else pair[:, ::-1])
    return {m: value[j].tolist() for j, m in enumerate(h.METHODS)}


def classification(v):
    return "positive" if v[0] > 0 else "negative" if v[1] < 0 else "includes_zero"


def run_one(d, out):
    from sklearn.metrics import f1_score, multilabel_confusion_matrix

    c.require(len(d["ids"]) == len(set(d["ids"])), "Duplicate eligible ID")
    c.require(len(d["settings"]) == len(set(d["settings"])), "Duplicate actual setting")
    c.require(
        d["mask"].any() and (not np.any(d["mask"] & ~d["valid"].all(0))),
        "Invalid common mask",
    )
    c.require(
        not np.any(d["pred"] & ~d["gate"][..., None]), "Labels despite closed gate"
    )
    own = []
    common = []
    strict = []
    smask = (d["valid"] & (d["nvalid"] == 3)).all(0)
    for j in range(len(d["settings"])):
        own.append(
            h.setting_summary(
                d["gold"],
                d["pred"][j],
                d["valid"][j],
                d["labels"],
                d["macro_labels"],
                d["gate"][j],
            )
        )
        common.append(
            h.setting_summary(
                d["gold"][d["mask"]],
                d["pred"][j, d["mask"]],
                np.ones(d["mask"].sum(), bool),
                d["labels"],
                d["macro_labels"],
                d["gate"][j, d["mask"]],
            )
        )
        strict.append(
            h.setting_summary(
                d["gold"][smask],
                d["pred"][j, smask],
                np.ones(smask.sum(), bool),
                d["labels"],
                d["macro_labels"],
                d["gate"][j, smask],
            )
        )
        for mask, summ in [(d["valid"][j], own[-1]), (d["mask"], common[-1])]:
            y = d["gold"][mask]
            p = d["pred"][j, mask]
            sk = [
                f1_score(
                    y[:, d["macro_indices"]],
                    p[:, d["macro_indices"]],
                    average="macro",
                    zero_division=0,
                ),
                f1_score(y, p, average="micro", zero_division=0),
                f1_score(
                    y[:, d["macro_indices"]],
                    p[:, d["macro_indices"]],
                    average="micro",
                    zero_division=0,
                ),
            ]
            h.equal(
                sk,
                [summ["scores"][m] for m in h.METHODS],
                "Independent sklearn own/common scores",
            )
            conf = multilabel_confusion_matrix(y, p)
            h.equal(
                conf[:, [1, 0, 1], [1, 1, 0]],
                summ["tp_fp_fn"],
                "Independent sklearn counts",
            )
    vals = np.array([[s["scores"][m] for m in h.METHODS] for s in common])
    groupmeans = group_score(d, vals)
    points = column_score(d, groupmeans)
    old_settings = d.get("old_settings", {})
    for j, k in enumerate(d["settings"]):
        if k in old_settings:
            h.equal(vals[j, 0], old_settings[k]["macro"], "Original setting macro")
            if "counts" in old_settings[k]:
                h.equal(
                    common[j]["tp_fp_fn"],
                    old_settings[k]["counts"],
                    "Original setting counts",
                )
    for name, old in d.get("old_groups", {}).items():
        h.equal(
            groupmeans[list(d["groups"]).index(name), 0], old, "Original group macro"
        )
    bs = None if d.get("descriptive") else boot(d)
    original_bound_own = own
    if "original_bound_valid" in d:
        original_bound_own = [
            h.setting_summary(
                d["gold"],
                d["pred"][j],
                d["original_bound_valid"][j],
                d["labels"],
                d["macro_labels"],
                d["gate"][j],
            )
            for j in range(len(d["settings"]))
        ]
    records = {}
    families = c.read(BASE / "analysis_plan.json")["families"]
    allquestions = {q: f for f in families.values() for q in f["members"]}
    for j, (name, weights) in enumerate(d["columns"].items()):
        old = d.get("old_columns", {}).get(name, {})
        point = points[j]
        qid = d["unit_id"] + "|" + name
        family = allquestions.get(qid) if d.get("registered_main", True) else None
        if "point" in old:
            h.equal(point[0], old["point"], "Original macro point")
        if "micro" in old:
            h.equal(point[1], old["micro"], "Original stored micro point")
        if bs is not None and old.get("ci95") is not None:
            h.equal(h.interval(bs["columns"][:, j, 0]), old["ci95"], "Original macro95")
        if bs is not None and old.get("micro_ci95") is not None:
            h.equal(
                h.interval(bs["columns"][:, j, 1]),
                old["micro_ci95"],
                "Original stored micro95",
            )
        if bs is not None and old.get("adjusted_interval") is not None:
            h.equal(
                h.interval(bs["columns"][:, j, 0], old["adjusted_confidence"]),
                old["adjusted_interval"],
                "Original adjusted macro",
            )
        mb = bounds(d, own, weights)
        original_mb = bounds(d, original_bound_own, weights)
        if old.get("missing_bounds") is not None:
            h.equal(
                original_mb["macro"],
                old["missing_bounds"],
                "Original macro missing bounds",
            )
        if old.get("micro_missing_bounds") is not None:
            h.equal(
                mb["micro"],
                old["micro_missing_bounds"],
                "Original stored micro missing bounds",
            )
        metrics = {}
        for mi, m in enumerate(h.METHODS):
            ci = None if bs is None else h.interval(bs["columns"][:, j, mi])
            conf = (
                old.get("adjusted_confidence")
                if m == "macro"
                else family["confidence"] if family else None
            )
            adj = (
                None
                if bs is None or conf is None
                else h.interval(bs["columns"][:, j, mi], conf)
            )
            metrics[m] = dict(
                value=float(point[mi]),
                ci95=ci,
                adjusted_interval=adj,
                adjusted_confidence=conf,
                classification95=None if ci is None else classification(ci),
            )
        variation = {}
        for kind, options in [
            ("repeat", range(3)),
            ("omit_repeat", range(3)),
            ("pool_or_stream", range(3)),
        ]:
            keyword = {
                "repeat": "repeat",
                "omit_repeat": "omit",
                "pool_or_stream": "pool",
            }[kind]
            variation[kind] = [
                column_score(d, group_score(d, vals, **{keyword: i}))[j].tolist()
                for i in options
            ]
        strictpoint = None
        if smask.any():
            strictpoint = column_score(
                d,
                group_score(
                    d, np.array([[s["scores"][m] for m in h.METHODS] for s in strict])
                ),
            )[j].tolist()
        records[name] = dict(
            weights=weights,
            contrast=sum(weights.values()) == 0,
            metrics=metrics,
            full_roster_missing_bounds=mb,
            original_ordinary_valid_bounds=(
                original_mb if "original_bound_valid" in d else None
            ),
            new_micro_question=qid if family else None,
            variation=variation,
            strict_point=strictpoint,
        )
    if bs is not None and d.get("old_draw_path"):
        path = c.bind(c.ROOT / d["old_draw_path"])
        if path.suffix == ".npz":
            arr = h.read_arrays(path)["draws"]
        else:
            arr = np.load(path, allow_pickle=False)
        for col, oldcol, metric, oldmetric in d["old_draw_columns"]:
            h.equal(
                bs["columns"][:, list(d["columns"]).index(col), metric],
                arr[:, oldcol, oldmetric] if arr.ndim == 3 else arr[:, oldmetric],
                "Original saved draw replay",
            )
    groups = {
        name: dict(
            common=h.average([common[j] for j in js]),
            own_valid=h.average([own[j] for j in js]),
            strict=h.average([strict[j] for j in js]),
            setting_indices=js,
            mean_per_label={
                field: np.mean(
                    [
                        [
                            common[j]["per_label"][l][field]
                            for l in range(len(d["labels"]))
                        ]
                        for j in js
                    ],
                    0,
                )
                for field in [
                    "tp",
                    "fp",
                    "fn",
                    "gold_positive",
                    "predicted_positive",
                    "f1",
                    "micro_weight",
                    "macro_weight",
                    "restricted_micro_weight",
                    "full_micro_contribution",
                    "restricted_minus_macro_contribution",
                ]
            },
        )
        for name, js in d["groups"].items()
    }
    source_info = {
        k: d[k]
        for k in [
            "id",
            "unit_id",
            "study",
            "endpoint",
            "view",
            "labels",
            "macro_labels",
            "macro_indices",
            "settings",
            "ids",
            "clusters",
            "groups",
            "columns",
            "seed",
            "sampling",
        ]
    }
    result = dict(
        **source_info,
        n_eligible=len(d["ids"]),
        n_common=int(d["mask"].sum()),
        n_strict=int(smask.sum()),
        common_ids=[x for i, x in enumerate(d["ids"]) if d["mask"][i]],
        group_results=groups,
        columns_results=records,
        own_settings=own,
        common_settings=common,
        bootstrap=(
            None
            if bs is None
            else {
                k: v for k, v in bs.items() if k not in ["columns", "groups", "sampled"]
            }
        ),
        descriptive=bool(d.get("descriptive")),
        raw_records_checked=len(d["ids"]) * len(d["settings"]),
        original_draw_path=d.get("old_draw_path"),
        original_draw_columns=d.get("old_draw_columns"),
        provenance=d.get("provenance"),
        selection_notes=d.get("selection_notes")
    )
    stem = slug(d)
    arrays(
        out / (stem + "_data.npz"),
        gold=d["gold"],
        pred=d["pred"],
        valid=d["valid"],
        gate=d["gate"],
        nvalid=d["nvalid"],
        mask=d["mask"],
    )
    if bs is not None:
        arrays(out / (stem + "_draws.npz"), columns=bs["columns"], groups=bs["groups"])
        save(out / (stem + "_samples.json"), bs["sampled"])
    save(out / (stem + ".json"), result)
    print(
        d["id"],
        int(d["mask"].sum()),
        "texts;",
        len(records),
        "columns; passed",
        flush=True,
    )
    return c.rel(out / (stem + ".json"))


def run(section, attempt="analysis_v2"):
    from analysis.feedback import micro_f1_inputs as readers

    c.require(
        attempt in ["analysis_v2", "analysis_v3", "analysis_v4"],
        "Unexpected attempt name",
    )
    out = BASE / attempt / section
    c.require(not out.exists(), "Preserve completed/partial section")
    out.mkdir(parents=True)
    paths = []
    try:
        for d in getattr(readers, section)():
            paths.append(run_one(d, out))
        save(
            out / "index.json",
            dict(
                status="passed",
                section=section,
                outputs=paths,
                checks=dict(h.CHECKS),
                max_error=dict(h.ERRORS),
                source_sha256=c.BINDINGS,
                source_code={
                    c.rel(Path(__file__)): c.sha(__file__),
                    c.rel(Path(readers.__file__)): c.sha(readers.__file__),
                },
            ),
        )
    except Exception:
        import traceback

        save(
            out / "failure.json",
            dict(
                traceback=traceback.format_exc(),
                completed=paths,
                checks=dict(h.CHECKS),
                source_sha256=c.BINDINGS,
            ),
        )
        raise


if __name__ == "__main__":
    import sys

    (
        freeze()
        if sys.argv[1] == "freeze"
        else run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "analysis_v2")
    )
