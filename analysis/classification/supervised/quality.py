from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.supervised import archive as r
from analysis.classification.supervised import metrics as m
from analysis.classification.supervised import controls as c
from analysis.classification.supervised import errors as e

BASE = r.ROOT / "experiments/astra/sq2_gpu_quality_review_20261103_v1"
SOURCE = e.BASE / "runs/54014ea4bfd9"
TRAIN = m.TRAIN
BOOT_KEYS = (
    "exact_set",
    "gate_aware_exact",
    "sample_jaccard",
    "sample_f1",
    "only_missing",
    "only_extra",
    "mixed",
    "predicted_labels",
    "empty_rate",
    "hamming_error",
    "raw_exact",
    "gate_pass",
)
PAIRS = (
    ("tuned/selected", "bge/selected"),
    ("frozen_roberta/selected", "bge/selected"),
    ("tuned/selected", "fixed_balanced/selected"),
) + tuple((f + "/selected", f + "/default") for f in c.METHODS)
save = c.save


def div(a, b, empty=0.0):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, empty, dtype=float), where=b != 0)


def quality(y, p, g, raw):
    m.validate_binary(y, p, list(range(y.shape[-1])))
    r.require(
        raw.shape == p.shape
        and g.shape == p.shape[:2]
        and np.isin(raw, [0, 1]).all()
        and np.isin(g, [0, 1]).all(),
        "Invalid raw/gate shapes.",
    )
    y = y.astype(bool)
    p = p.astype(bool)
    g = g.astype(bool)
    raw = raw.astype(bool)
    r.require(np.array_equal(p, raw & g[:, :, None]), "Gated outputs disagree.")
    tp = (p & y).sum(2)
    fp = (p & ~y).sum(2)
    fn = (~p & y).sum(2)
    exact = (fp + fn) == 0
    row = dict(
        exact_set=exact,
        gate_aware_exact=exact & g,
        sample_jaccard=div(tp, tp + fp + fn, 1),
        sample_f1=div(2 * tp, 2 * tp + fp + fn, 1),
        only_missing=(fn > 0) & (fp == 0),
        only_extra=(fp > 0) & (fn == 0),
        mixed=(fp > 0) & (fn > 0),
        predicted_labels=p.sum(2),
        empty_rate=~p.any(2),
        hamming_error=(fp + fn) / y.shape[1],
        raw_exact=(raw == y).all(2),
        gate_pass=g,
    )
    r.require(
        np.all(
            row["exact_set"].astype(int)
            + row["only_missing"]
            + row["only_extra"]
            + row["mixed"]
            == 1
        ),
        "Set-error partition fails.",
    )
    score, counts = m.point_scores(y, p, list(range(y.shape[1])))
    point = {key: value.mean(1) for key, value in row.items()}
    point.update(
        macro_f1=score,
        micro_precision=div(tp.sum(1), (tp + fp).sum(1)),
        micro_recall=div(tp.sum(1), (tp + fn).sum(1)),
        micro_f1=div(2 * tp.sum(1), (2 * tp + fp + fn).sum(1)),
        missing_labels=fn.sum(1),
        extra_labels=fp.sum(1),
        label_precision=div(counts["tp"], counts["tp"] + counts["fp"]),
        label_recall=div(counts["tp"], counts["tp"] + counts["fn"]),
        label_f1=counts["per_label_f1"],
        label_tp=counts["tp"],
        label_fp=counts["fp"],
        label_fn=counts["fn"],
        predicted_label_support=p.sum(1),
        gold_label_support=y.sum(0),
        gold_cardinality=np.bincount(y.sum(1), minlength=y.shape[1] + 1),
        predicted_cardinality=np.asarray(
            [np.bincount(x.sum(1), minlength=y.shape[1] + 1) for x in p]
        ),
    )
    return point, row


def profiles(models, rows):
    keys = None
    result = []
    for metric, value in rows.items():
        _, budget = c.aggregate(models, np.asarray(value, dtype=float).T)
        if keys is None:
            keys = list(budget)
        r.require(keys == list(budget), "Profile key order changed.")
        result.append(np.asarray([budget[key] for key in keys]))
    return keys, np.stack(result, axis=2)


def row_bootstrap(values, clusters, draws=10000, batch=64):
    r.require(
        values.ndim == 3 and values.shape[1] == len(clusters) and len(clusters) > 0,
        "Invalid row bootstrap axes.",
    )
    unique, inv = np.unique(clusters, return_inverse=True)
    K, N, F = values.shape
    terms = np.empty((len(unique), K * F), float)
    for k in range(K):
        for f in range(F):
            terms[:, k * F + f] = np.bincount(
                inv, weights=values[k, :, f], minlength=len(unique)
            )
    sizes = np.bincount(inv, minlength=len(unique))
    rng = np.random.default_rng(20261020)
    digest = hashlib.sha256()
    output = np.empty((draws, K, F))
    for first in range(0, draws, batch):
        n = min(batch, draws - first)
        picks = rng.integers(len(unique), size=(n, len(unique)), dtype=np.int64)
        digest.update(picks.astype("<i8", copy=False).tobytes())
        weights = np.asarray(
            [np.bincount(x, minlength=len(unique)) for x in picks], float
        )
        output[first : first + n] = (
            (weights @ terms) / (weights @ sizes)[:, None]
        ).reshape(n, K, F)
    return output, dict(
        draws=draws,
        seed=20261020,
        clusters=len(unique),
        picks_sha256=digest.hexdigest(),
    )


def summaries(models, point, row, clusters):
    output = dict(model=point, pool={}, budget={}, contrasts=[], intervals={})
    for metric, values in point.items():
        a = np.asarray(values)
        if a.shape == (len(models),):
            output["pool"][metric], output["budget"][metric] = c.aggregate(models, a)
        elif a.ndim == 2 and a.shape[0] == len(models):
            pp, bb = c.aggregate(models, a.T)
            output["pool"][metric] = {k: v.tolist() for k, v in pp.items()}
            output["budget"][metric] = {k: v.tolist() for k, v in bb.items()}
    keys, profile = profiles(models, row)
    draws, info = row_bootstrap(profile, clusters)
    for j, key in enumerate(keys):
        output["intervals"][key] = {
            metric: c.interval(profile[j, :, k].mean(), draws[:, j, k])
            for k, metric in enumerate(row)
        }
    for budget in m.BUDGETS:
        for after, before in PAIRS:
            a, b = keys.index(f"{budget}/{after}"), keys.index(f"{budget}/{before}")
            output["contrasts"].append(
                dict(
                    budget=budget,
                    after=after,
                    before=before,
                    metrics={
                        metric: c.interval(
                            profile[a, :, k].mean() - profile[b, :, k].mean(),
                            draws[:, a, k] - draws[:, b, k],
                        )
                        for k, metric in enumerate(row)
                    },
                )
            )
    output["bootstrap"] = info
    output["bootstrap_keys"] = keys
    output["bootstrap_metrics"] = list(row)
    return output, profile, draws


def severity_detail(y, p, g):
    r.require(
        y.shape[1] == 3 and np.all(y.sum(1) == 1) and np.array_equal(p.sum(2), g),
        "Severity must be one class when gate is open.",
    )
    gold = y.argmax(1)
    pred = p.argmax(2)
    g = g.astype(bool)
    difference = pred - gold
    counts = np.zeros((len(p), 3, 4), int)
    for label in range(3):
        mask = gold == label
        counts[:, label, 0] = (~g[:, mask]).sum(1)
        for guess in range(3):
            counts[:, label, guess + 1] = (g[:, mask] & (pred[:, mask] == guess)).sum(1)
    masks = dict(
        blocked=~g,
        correct=g & (difference == 0),
        under_one=g & (difference == -1),
        under_two=g & (difference == -2),
        over_one=g & (difference == 1),
        over_two=g & (difference == 2),
    )
    r.require(
        np.all(sum(x.astype(int) for x in masks.values()) == 1),
        "Severity error partition fails.",
    )
    return dict(
        confusion=counts,
        gold_support=y.sum(0),
        model_counts={k: x.sum(1) for k, x in masks.items()},
        model_rates={k: x.mean(1) for k, x in masks.items()},
        conditional_absolute_ordinal_error=div(
            (np.abs(difference) * g).sum(1), g.sum(1)
        ),
        conditional_error_defined=g.any(1),
    )


def joint_data(tm, tz, sm, sz):
    r.require(tm["base_models"] == sm["base_models"], "Joint model identities differ.")
    tix = {key: i for i, key in enumerate(tm["ids"])}
    r.require(
        len(tix) == len(tm["ids"])
        and set(sm["ids"]) <= set(tix)
        and len(set(sm["ids"])) == len(sm["ids"]),
        "Joint eligibility mismatch.",
    )
    positions = np.asarray([tix[key] for key in sm["ids"]])
    g = sz["gate"].astype(bool)
    r.require(np.array_equal(tz["gate"][:, positions], g), "Joint hate gate differs.")
    te = (tz["gated"][:, positions] == tz["truth"][positions]).all(2)
    se = (sz["gated"] == sz["truth"]).all(2)
    tr = (tz["raw"][:, positions] == tz["truth"][positions]).all(2)
    sr = (sz["raw"] == sz["truth"]).all(2)
    masks = dict(
        joint_exact=g & te & se,
        gate_closed=~g,
        target_wrong_only=g & ~te & se,
        severity_wrong_only=g & te & ~se,
        both_wrong=g & ~te & ~se,
    )
    r.require(
        np.all(sum(x.astype(int) for x in masks.values()) == 1),
        "Joint outcome partition fails.",
    )
    masks.update(
        target_exact_with_gate=g & te,
        severity_exact_with_gate=g & se,
        raw_joint_exact=tr & sr,
        correct_raw_pair_blocked=~g & tr & sr,
        gate_pass=g,
    )
    return positions, masks


def freeze():
    BASE.mkdir(parents=True, exist_ok=True)
    paths = [
        e.BASE / "part4_complete.json",
        c.BASE / "part3_complete.json",
        m.BASE / "part2_complete.json",
        r.ROOT
        / "experiments/astra/sq2_gpu_training_review_20261103_v1/part1_complete.json",
        TRAIN / "complete.json",
        e.ROSTER,
        Path(__file__),
        Path(r.__file__),
        Path(c.__file__),
        Path(m.__file__),
    ]
    save(
        BASE / "analysis_contract.json",
        dict(
            part=5,
            utc=datetime.now(timezone.utc).isoformat(),
            question="How correct are the individual fine labels and whole delivered annotated answers, rather than just aggregate F1?",
            prior_exposure="Parts 1–4 and the five-fit type supplement have already been reviewed; all new measures are supporting.",
            endpoints=list(r.PRIMARY[1:]),
            methods=list(c.METHODS),
            policies=list(r.POLICIES),
            seeds=list(r.SEEDS),
            pools=list(r.POOLS),
            uncertainty=dict(
                draws=10000,
                seed=20261020,
                interval=0.95,
                paired_normalized_text_clusters=True,
                conditional_on_fixed_fits_and_pools=True,
                unadjusted_supporting=True,
                original_24_primary_family_unchanged=True,
            ),
            row_metrics=list(BOOT_KEYS),
            comparisons=[list(x) for x in PAIRS],
            scope=[
                "Per-label TP/FP/FN, precision/recall/F1, gold support and contribution to macro-F1 differences; all labels retained",
                "Exact sets, gate-aware exact sets, sample Jaccard/F1, extra-only/missing-only/mixed partitions, cardinality and emission counts",
                "Default versus TRAIN-selected outputs for all five procedures; no new cutoff selection",
                "TRAIN annotation support from sealed Part 1 records, not an equal-annotations assumption",
                "Severity confusion with an explicit blocked column; adjacent/two-level under/over errors, no fake severity class for blocked outputs",
                "Joint gate+target+severity on larger-DEV MHS intersection, with row-aligned masks, model identities and disjoint failure paths",
                "Inventory annotation combinations; do not score absent type/target/severity fields or fabricate four-head exactness",
                "Paired exact-answer fixes/harms for selected tuned/BGE, frozen/BGE, tuned/fixed-balanced and each default/selected policy",
                "Separate updated five-type-model quality sensitivity; original primary evidence unchanged",
            ],
            limitations=[
                "Structured descriptions are not validated faithful rationales",
                "No new human relabeling or result-selected model",
                "Label-specific outputs are descriptive; no new primary multiplicity family",
                "Different populations prevent pooling fine tasks into one full-answer rate",
                "No training, inference, Qwen, weights, server access or original/writer-owned edits",
            ],
            inputs={str(p.relative_to(r.ROOT)): r.sha(p) for p in paths},
        ),
    )
    print("Part 5 analysis contract saved before new quality scoring.", flush=True)


def transitions(models, row):
    result = []
    for budget in m.BUDGETS:
        for after, before in PAIRS:
            pairs = []
            for pool in r.POOLS:
                if m.budget(pool) == budget:
                    pairs += e.pair_indices(models, after, before, pool)
            for metric in (
                ("gate_aware_exact", "exact_set")
                if "exact_set" in row
                else ("joint_exact",)
            ):
                a = row[metric][[x for x, y in pairs]].astype(bool)
                b = row[metric][[y for x, y in pairs]].astype(bool)
                fixed = ~b & a
                lost = b & ~a
                result.append(
                    dict(
                        budget=budget,
                        after=after,
                        before=before,
                        metric=metric,
                        fixed=float(fixed.sum(1).mean()),
                        lost=float(lost.sum(1).mean()),
                        net=float((a.sum(1) - b.sum(1)).mean()),
                        distinct_fixed_any=int(fixed.any(0).sum()),
                        distinct_lost_any=int(lost.any(0).sum()),
                        persistent_fixed=int(fixed.all(0).sum()),
                        persistent_lost=int(lost.all(0).sum()),
                    )
                )
    return result


def run():
    plan = r.read(BASE / "analysis_contract.json")
    for name, digest in plan["inputs"].items():
        r.checked_file(r.ROOT / name, digest)
    folder = BASE / "runs" / uuid.uuid4().hex[:12]
    start = time.monotonic()
    save(
        folder / "started.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            contract_sha256=r.sha(BASE / "analysis_contract.json"),
        ),
    )
    print("RUN " + str(folder), flush=True)
    bindings = {}
    part4 = r.read(e.BASE / "part4_complete.json")["artifacts"]

    def checked(path):
        r.checked_file(path, part4[str(path.relative_to(r.ROOT))], bindings)
        return (
            r.read(path)
            if path.suffix == ".json"
            else r.arrays_from_bytes(path.read_bytes())
        )

    models = checked(SOURCE / "binary_metadata.json")["models"]
    save(folder / "models.json", models)
    roster = [
        json.loads(line) for line in e.ROSTER.read_text(encoding="utf8").splitlines()
    ]
    inventory = Counter()
    for row in roster:
        if row["phase"] not in ("en_test", "larger_dev"):
            continue
        fields = tuple(
            h for h, f in r.FIELDS.items() if h != "hate" and row[f] is not None
        )
        inventory[
            row["phase"], row["source"], str(row["gate"]), ",".join(fields) or "none"
        ] += 1
    save(
        folder / "annotation_inventory.json",
        [
            dict(phase=a, source=b, gold_hate=d, observed_fine_fields=f, rows=n)
            for (a, b, d, f), n in sorted(inventory.items())
        ],
    )
    trainreceipt = r.read(TRAIN / "complete.json")
    training = {}
    for pool in r.POOLS:
        path = TRAIN / pool / "data_review.json"
        r.checked_file(
            path, trainreceipt["outputs"][f"{pool}/data_review.json"], bindings
        )
        training[pool] = r.read(path)["heads"]
    save(folder / "training_support.json", training)
    fine = {}
    for endpoint in r.PRIMARY[1:]:
        slug = m.slug(endpoint)
        meta = checked(SOURCE / (slug + "_metadata.json"))
        z = checked(SOURCE / (slug + "_flow.npz"))
        r.require(meta["base_models"] == models, "Fine model roster differs.")
        fine[endpoint] = (meta, z)
        point, row = quality(z["truth"], z["gated"], z["gate"], z["raw"])
        summary, profile, draws = summaries(models, point, row, meta["clusters"])
        summary.update(
            endpoint=endpoint,
            rows=len(z["truth"]),
            labels=meta["labels"],
            ids=meta["ids"],
            clusters=meta["clusters"],
        )
        summary["transitions"] = transitions(models, row)
        summary["macro_contributions"] = []
        for budget in m.BUDGETS:
            for after, before in PAIRS[:3]:
                av = np.asarray(summary["budget"]["label_f1"][f"{budget}/{after}"])
                bv = np.asarray(summary["budget"]["label_f1"][f"{budget}/{before}"])
                delta = av - bv
                r.require(
                    abs(
                        delta.mean()
                        - (
                            summary["budget"]["macro_f1"][f"{budget}/{after}"]
                            - summary["budget"]["macro_f1"][f"{budget}/{before}"]
                        )
                    )
                    < 1e-14,
                    "Label contributions fail to reconcile.",
                )
                summary["macro_contributions"].append(
                    dict(
                        budget=budget,
                        after=after,
                        before=before,
                        label_f1_difference=delta,
                        macro_contribution=delta / len(delta),
                        total=delta.mean(),
                    )
                )
        save(folder / (slug + "_quality.json"), summary)
        m.array(folder / (slug + "_row_bootstrap.npz"), profiles=profile, draws=draws)
        save(folder / (slug + "_metadata.json"), meta)
        if meta["head"] == "severity":
            sd = severity_detail(z["truth"], z["gated"], z["gate"])
            sd["budget_rates"] = {
                key: c.aggregate(models, value)[1]
                for key, value in sd["model_rates"].items()
            }
            _, bc = c.aggregate(models, sd["confusion"].reshape(len(models), -1).T)
            sd["budget_confusion"] = {
                key: values.reshape(3, 4) for key, values in bc.items()
            }
            save(folder / "severity_details.json", sd)
        print(
            endpoint + ": label, set, cardinality and paired quality checks complete.",
            flush=True,
        )
    tm, tz = fine["larger_dev/target_group"]
    sm, sz = fine["larger_dev/severity"]
    positions, row = joint_data(tm, tz, sm, sz)
    point = {key: value.mean(1) for key, value in row.items()}
    summary, profile, draws = summaries(models, point, row, sm["clusters"])
    summary.update(
        rows=len(sm["ids"]),
        ids=sm["ids"],
        clusters=sm["clusters"],
        target_positions=positions,
        observed_fields=["hate", "target_group", "severity"],
        unobserved_fields=["hate_type"],
        transitions=transitions(models, row),
    )
    save(folder / "joint_quality.json", summary)
    m.array(folder / "joint_row_bootstrap.npz", profiles=profile, draws=draws)
    print("Aligned MHS joint answers and disjoint failure paths complete.", flush=True)

    meta, z = fine["en_test/hate_type"]
    supp = checked(SOURCE / "supplement_type_flow.npz")
    raw = supp["raw"]
    g = supp["gate"]
    p = raw * g[:, :, None]
    point, row = quality(supp["truth"], p, g, raw)
    summary, profile, draws = summaries(models, point, row, meta["clusters"])
    summary.update(
        rows=len(meta["ids"]), labels=meta["labels"], original_primary_unchanged=True
    )
    original = r.read(folder / "en_test__hate_type_quality.json")
    origdraw = r.arrays_from_bytes(
        (folder / "en_test__hate_type_row_bootstrap.npz").read_bytes()
    )["draws"]
    changes = {}
    for j, key in enumerate(summary["bootstrap_keys"]):
        changes[key] = {
            metric: c.interval(
                summary["budget"][metric][key] - original["budget"][metric][key],
                draws[:, j, k] - origdraw[:, j, k],
            )
            for k, metric in enumerate(row)
        }
    summary["paired_changes_from_original"] = changes
    save(folder / "supplement_type_quality.json", summary)
    m.array(folder / "supplement_type_row_bootstrap.npz", profiles=profile, draws=draws)
    save(folder / "source_bindings.json", bindings)
    r.require("torch" not in sys.modules, "Unexpected model library loaded.")
    save(
        folder / "complete.json",
        dict(
            status="complete",
            utc=datetime.now(timezone.utc).isoformat(),
            seconds=time.monotonic() - start,
            outputs={
                str(p.relative_to(folder)): r.sha(p)
                for p in folder.rglob("*")
                if p.is_file()
            },
            models=306,
            training_calls=0,
            inference_calls=0,
            qwen_calls=0,
            server_actions=0,
            weights_loaded=0,
        ),
    )
    print("COMPLETE " + str(folder), flush=True)


if __name__ == "__main__":
    with r.offline(), threadpool_limits(limits=2):
        if sys.argv[1:] == ["freeze"]:
            freeze()
        elif sys.argv[1:] == ["run"]:
            run()
        else:
            raise SystemExit("Use freeze or run.")
