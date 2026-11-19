import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from analysis.classification.combined import prepare as a

ROOT = a.ROOT
BASE = ROOT / "experiments/astra/sq2_combined_performance_20261104_v1"
SETUP = a.BASE
CPU = ROOT / "experiments/astra/sq2_cpu_baselines_20261012_v2"
CPUREVIEW = ROOT / "experiments/astra/sq2_cpu_review_20261012_v1"
TUNE = ROOT / "experiments/astra/compact_roberta_tuning_20261010_v1"
SHORT = ROOT / "experiments/astra/compact_roberta_short_20261012_v1"
SHORTREVIEW = ROOT / "experiments/astra/roberta_short_review_20261020_v1"
COMPACT = ROOT / "experiments/astra/compact_broad_20261008_v1"
SUPP = ROOT / "experiments/astra/sq2_type_sensitivity_20261103_v1/runs/2b1cee266dcb"
GPUMAIN = ROOT / "experiments/astra/sq2_gpu_main_review_20261103_v1/runs/9deace7a4235"
VARIANTS = [
    f"{m}_{p}"
    for m in ("fixed_unweighted", "fixed_balanced", "tuned")
    for p in ("default", "selected")
]
PRIMARY = [
    "en_test/hate",
    "en_test/hate_type",
    "larger_dev/target_group",
    "larger_dev/severity",
]


def save(name, value):
    p = BASE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(a.g.encoded(value))


def arr(name, **values):
    p = BASE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        np.savez_compressed(f, **values)


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def pipes():
    out = []
    for p in sorted((CPU / "pipelines").glob("*/complete.json")):
        receipt = a.read(p)
        attempt = p.parent / receipt["attempt"]
        c = a.read(attempt / "contract.json")
        ident = c["identity"]
        root = (
            ident["stage"]
            if ident["stage"] == "compact_tfidf"
            else ident["stage"]
            + ("_matched" if ident["variant"] == "matched_gradient" else "_refit")
        )
        out.append(
            dict(
                root=root,
                seed=ident["seed"],
                receipt=p,
                contract=attempt / "contract.json",
                predictions=attempt / "decisions.npz",
            )
        )
    a.require(len(out) == 7, "Expected seven retained CPU pipelines")
    return out


def sources():
    pp = [
        Path(__file__),
        Path(__file__).with_name("astra_sq2_combined_performance_tests_v1.py"),
        a.BROAD / "predictions.jsonl",
        SETUP / "part1_complete.json",
        SETUP / "analysis_contract.json",
        SETUP / "evaluation_alignment.json",
        SETUP / "endpoint_registry.json",
        SETUP / "training_information.json",
        SETUP / "comparison_map.json",
        ROOT
        / "Thesis_Log/astra_result/astra_remaining_work_before_writing_20261104_v153.md",
        ROOT
        / "Thesis_Log/astra_result/astra_sq2_combined_synthesis_plan_20261104_v1.md",
        a.GPU / "models.json",
        a.GPU / "complete.json",
        a.ROSTER,
        a.g.LINEAR / "data/inference.json",
        a.g.LINEAR / "data/scopes.json",
        CPUREVIEW / "scores.json",
        CPUREVIEW / "statistics.json",
        CPUREVIEW / "complete.json",
        SHORTREVIEW / "scores.json",
        SHORTREVIEW / "statistics.json",
        SHORTREVIEW / "complete.json",
        TUNE / "data/train.json",
        SHORT / "data/inference.json",
        SHORT / "data/scopes.json",
        SUPP / "paired_full_predictions.npz",
        SUPP / "complete.json",
        SUPP / "main_comparison.json",
        SUPP / "independent_review.json",
        GPUMAIN / "primary_comparisons.json",
        a.MATCHED / "metrics.json",
        a.MATCHED / "complete.json",
        ROOT
        / "experiments/astra/sq2_broad_errors_review_20261104_v1/binary_results.json",
        ROOT
        / "experiments/astra/sq2_gpu_sources_review_20261104_v1/analysis/endpoints.json",
        ROOT / "experiments/astra/sq2_broad_synthesis_20261104_v1/part9_complete.json",
        ROOT / "experiments/astra/sq2_gpu_synthesis_20261104_v1/part9_complete.json",
    ]
    for p in pipes():
        pp.extend(p[k] for k in ("receipt", "contract", "predictions"))
    pp += [CPUREVIEW / f"neural/broad_s{s}_decisions.npz" for s in range(3)]
    pp += [COMPACT / f"predictions_seed{s}.jsonl" for s in (42, 43, 44)]
    for folder in (TUNE, SHORT):
        pp += [
            folder / "predictions" / f"{v}_s{s}.npz"
            for v in VARIANTS
            for s in range(10, 15)
        ]
    for name in a.read(a.BROAD / "endpoint_setup.json"):
        slug = name.replace("/", "__")
        pp += [
            a.MAIN / "endpoints" / slug / "results.json",
            a.MAIN / "endpoints" / slug / "encoded.npz",
        ]
        if (a.GPU / (slug + "_arrays.npz")).exists():
            pp += [
                a.GPU / (slug + "_arrays.npz"),
                a.GPU / (slug + "_metadata.json"),
                SETUP / "alignment" / (slug + ".npz"),
            ]
    names = [
        "astra_sq2_broad_final_synthesis_20261104_v2.md",
        "astra_sq2_gpu_final_synthesis_20261104_v1.md",
        "astra_sq2_matched_results_20261030_v1.md",
        "astra_sq2_short_training_results_20261020_v1.md",
        "astra_sq2_cpu_results_20261012_v1.md",
        "astra_sq2_compact_bge_results_20261008_v1.md",
        "astra_sq2_compact_controls_results_20261008_v1.md",
        "astra_sq2_compact_pool_results_20261008_v1.md",
        "astra_sq2_roberta_tuning_results_20261011_v1.md",
        "astra_sq2_matched_type_results_20261012_v1.md",
        "astra_sq2_type_update_results_20261103_v1.md",
        "astra_core_results_summary_20260920_v1.md",
        "astra_sq2_label_vote_results_20261001_v1.md",
        "astra_sq2_broad_main_comparisons_20261104_v1.md",
        "astra_sq2_broad_errors_and_gate_20261104_v1.md",
        "astra_sq2_broad_sources_functions_20261104_v2.md",
        "astra_sq2_gpu_sources_functions_20261104_v1.md",
    ]
    pp += [a.REPORTS / n for n in names if (a.REPORTS / n).exists()]
    return pp


def freeze():
    from tests.classification.combined.test_metrics import run_tests

    tests = run_tests()
    seal = a.read(SETUP / "part1_complete.json")
    a.verify(seal["outputs"])
    a.verify(seal["original_sources"])
    for p in pipes():
        rec = a.read(p["receipt"])
        for f in (p["contract"], p["predictions"]):
            a.require(
                a.g.sha(f)
                == rec["outputs"][f.relative_to(p["receipt"].parent).as_posix()],
                "CPU prediction receipt differs",
            )
    for name in ("astra_session.md", "astra_review.md"):
        p = BASE / "log_preimages" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as f:
            f.write((ROOT / "Thesis_Log" / name).read_bytes())
    save("tests.json", tests)
    save(
        "analysis_contract.json",
        dict(
            status="combined_SQ2_part2_scope_recorded_before_new_aligned_scoring",
            created_utc=a.utc(),
            question="How does RAG-based prompting compare to standard prompting without retrieval, measured with multi-label metrics?",
            priority="Lead with the original full/no-retrieval package contrast; retain original full/small, matched-information and supervised families.",
            exposure="All standalone results and comparison setup already reviewed. New cross-study views are retrospective, descriptive and conditional, not new preregistration or independent confirmation.",
            endpoints=list(a.read(a.BROAD / "endpoint_setup.json")),
            supervised_endpoints=[
                v["endpoint"] for v in a.read(SETUP / "evaluation_alignment.json")
            ],
            methods="All 306 original uniform GPU views; separate five-fit update sensitivity; historical Qwen-embedded encoder references; all retained CPU native/default/refit/compact-TFIDF views; all original/expanded curated neural variants/seeds; three curated BGE folds; all Qwen repeats.",
            masks="Primary new contextual display: original common-nine Qwen masks. Also retain each model own-valid/full-eligible result, original Qwen six-setting contrasts, fixed macro labels and original failure bounds. No imputation or new exclusions.",
            scoring="Delivered macro F1, all per-label TP/FP/FN/precision/recall/F1; mean predicted/extra/missing labels; binary FP/FN/FPR/FNR/precision/recall/error and fine-head gate suppression. No new exact/joint/rationale analysis in this part.",
            zero_denominators="Label F1/precision/recall use zero_division=0. Binary precision is zero when no hate is predicted, with a separate no_predicted_hate flag. FPR/FNR and binary recall are undefined when their gold denominator is absent; absent supported labels stay in the original macro columns.",
            aggregation="Individual fit/repeat metrics, then within each retained TRAIN pool, then equal pool means. Context variants/decoder repeats/initializations/folds remain different units. No pooling predictions as extra texts or best-seed/policy selection.",
            heterogeneity="All natural sources on the common-nine endpoint masks; every HateCheck function on its same available rows, error rates for single-polarity binary subsets. Original endpoint macro columns remain frozen. Retain every stratum and unsupported/undefined cell.",
            error_overlap="On each shared binary endpoint, average the actual Cartesian Qwen-full repeat x supervised fit pairs, with equal pool weights. Partition both correct, only Qwen wrong, only comparator wrong, both wrong. Descriptive complementarity only; no tested ensemble or independence claim.",
            display_focus=[
                "Qwen full/small/zero on all endpoints",
                "Original primary/supporting intervals without alteration",
                "Uniform selected tuned/BGE/frozen RoBERTa at every budget, with fixed/default controls retained in appendices",
                "All native full-data and curated references; selected procedures lead, default and all-permitted refits explicit",
                "Binary overlap focus fixed in advance: tuned 5000, BGE 5000, full selected neural, full selected BGE, expanded curated tuned",
            ],
            inference="No new confidence intervals, hypothesis tests, bootstrap draws, primary families, equivalence claims, overall winner vote or score averaging across endpoints. Original source intervals retain their original masks/families; new aligned method differences are descriptive points.",
            verification="Independently reconstruct metrics using separate confusion-count arithmetic; reproduce original own/full scores and primary points; verify label, pool, per-fit and subgroup aggregation plus binary pair partitions. Bind publication claims/tables/figures to checked numerical records.",
            deferred="Part 3 complete-answer quality and mechanism synthesis; Part 4 robustness/resources/final scientific answer and global records; separate Part 5 writing handover.",
            sources=a.bind(sources()),
            new_Qwen_calls=0,
            new_training=0,
            new_inference=0,
            new_retrieval=0,
            server_actions=0,
            writer_edits=0,
        ),
    )
    print(
        "Combined Part 2 scope frozen; no new inferential family or experiment.",
        flush=True,
    )


def decode_result(result, head, labels):
    if head == "hate":
        positive = {"hate" if result["hate"] else "not_hate"}
    elif head == "severity":
        positive = {result[head]} if result[head] is not None else set()
    else:
        positive = set(result[head])
    a.require(positive <= set(labels), "Unknown delivered label")
    return [int(l in positive) for l in labels]


def extract():
    a.verify(a.read(BASE / "analysis_contract.json")["sources"])
    labels = {
        h: v["labels"] for h, v in a.read(TUNE / "data/train.json")["heads"].items()
    }
    labels["hate"] = ["not_hate", "hate"]
    inference = a.read(a.g.LINEAR / "data/inference.json")
    pos = {v["key"]: i for i, v in enumerate(inference)}
    scopes = a.read(a.g.LINEAR / "data/scopes.json")
    local = {v["key"]: i for i, v in enumerate(scopes["broad"])}
    a.require(
        [(r["key"], r["text"]) for r in a.read(SHORT / "data/inference.json")]
        == [(r["key"], r["text"]) for r in inference],
        "Shorter input identity differs",
    )
    cpu = [(p, load(p["predictions"])) for p in pipes()]
    native = [load(CPUREVIEW / f"neural/broad_s{s}_decisions.npz") for s in range(3)]
    curated = {
        (root, v, s): load(folder / "predictions" / f"{v}_s{s}.npz")
        for root, folder in [("curated_original", TUNE), ("curated_expanded", SHORT)]
        for v in VARIANTS
        for s in range(10, 15)
    }
    bge = {
        s: {
            r["key"]: r["result"]
            for r in a.rows(COMPACT / f"predictions_seed{s}.jsonl")
        }
        for s in (42, 43, 44)
    }
    supplement = load(SUPP / "paired_full_predictions.npz")
    gm = a.read(a.GPU / "models.json")
    outputs = []
    summary = []
    qgates = {}
    for r in a.rows(a.BROAD / "predictions.jsonl"):
        k = (r["phase"], r["key"], r["condition"], r["repeat"])
        a.require(k not in qgates, "Duplicate saved Qwen gate")
        qgates[k] = None if r["result"] is None else int(r["result"]["hate"])
    a.require(len(qgates) == 194643, "Saved Qwen gate roster incomplete")
    for endpoint, ep in a.read(a.BROAD / "endpoint_setup.json").items():
        slug = endpoint.replace("/", "__")
        head = ep["dimension"]
        ids = ep["scorable_ids"]
        n = len(ids)
        labs = ep["all_labels"]
        result = a.read(a.MAIN / "endpoints" / slug / "results.json")
        q = load(ROOT / result["encoded_path"])
        gold = q["gold"].astype(np.uint8)
        common = q["valid"][:9].all(0)
        models = []
        pred = []
        valid = []
        gates = []

        def add(group, pool, seed, values, gate, ok, source, original_method=None):
            values = np.asarray(values)
            ok = np.asarray(ok, dtype=bool)
            a.require(
                values.shape == (n, len(labs))
                and gate.shape == (n,)
                and ok.shape == (n,),
                "Aligned shape differs",
            )
            a.require(
                set(np.unique(values[ok])) <= {0, 1}
                and set(np.unique(gate[ok])) <= {0, 1},
                "Nonbinary valid decision",
            )
            models.append(
                dict(
                    group=group,
                    pool=pool,
                    seed=seed,
                    source=a.rel(source),
                    original_method=original_method,
                )
            )
            pred.append(values.astype(np.uint8))
            gates.append(gate.astype(np.uint8))
            valid.append(ok)

        hate_lookup = {}
        hate_endpoint = endpoint.split("/")[0] + "/hate"
        if (
            a.MAIN / "endpoints" / hate_endpoint.replace("/", "__") / "results.json"
        ).exists():
            hv = a.read(
                a.MAIN / "endpoints" / hate_endpoint.replace("/", "__") / "results.json"
            )
            hz = load(ROOT / hv["encoded_path"])
            hi = {k: i for i, k in enumerate(hv["endpoint"]["scorable_ids"])}
            hate_lookup = {
                tuple(setting): hz["predicted"][j, :, 1][[hi[k] for k in ids]]
                for j, setting in enumerate(hv["settings"])
            }
        for j, (co, s) in enumerate(result["settings"]):
            group = {
                "encoder_full": "historical_full/default",
                "encoder_kb": "historical_compact/default",
                "encoder_gbert": "historical_gbert/default",
            }.get(co, "qwen/" + co)
            gate = (
                q["predicted"][j, :, labs.index("hate")]
                if head == "hate"
                else hate_lookup.get((co, s), np.zeros(n, dtype=np.uint8))
            )
            if co.startswith("rag_") or co == "zero_shot":
                saved = [qgates[(endpoint.split("/")[0], k, co, s)] for k in ids]
                a.require(
                    np.array_equal(
                        np.asarray([v is not None for v in saved]), q["valid"][j]
                    ),
                    "Saved gate/output validity mismatch",
                )
                if head == "hate" or (co, s) in hate_lookup:
                    a.require(
                        np.array_equal(
                            gate[q["valid"][j]],
                            np.asarray(saved, dtype=object)[q["valid"][j]],
                        ),
                        "Qwen gate/encoded head disagree",
                    )
                gate = np.asarray(
                    [0 if v is None else v for v in saved], dtype=np.uint8
                )
            add(
                group,
                "fixed",
                s,
                q["predicted"][j],
                gate,
                q["valid"][j],
                ROOT / result["encoded_path"],
                co,
            )
        if (a.GPU / (slug + "_metadata.json")).exists():
            meta = a.read(a.GPU / (slug + "_metadata.json"))
            z = load(a.GPU / (slug + "_arrays.npz"))
            align = load(SETUP / "alignment" / (slug + ".npz"))
            a.require(
                meta["labels"] == labs
                and [a.key(r) for r in meta["records"]] == ids
                and np.array_equal(z["truth"], gold),
                "GPU alignment drift",
            )
            a.require(np.array_equal(align["common_nine"], common), "Part 1 mask drift")
            for j, m in enumerate(gm):
                add(
                    f"gpu/{m['budget']}/{m['method']}/{m['variant']}",
                    m["pool"],
                    m["seed"],
                    z["gated"][j],
                    z["gate"][j],
                    np.ones(n, bool),
                    a.GPU / (slug + "_arrays.npz"),
                    m["id"],
                )

            ix = [pos[k] for k in ids]
            cols = [labels[head].index(l) for l in labs]
            for j, m in enumerate(gm):
                if m["budget"] != 147 or m["method"] != "tuned":
                    continue
                if m["pool"] == "uniform_compact_s42":
                    v = supplement[f"{m['variant']}__new__{head}"][m["seed"] - 10][ix][
                        :, cols
                    ]
                    gate = supplement[f"{m['variant']}__new__hate"][m["seed"] - 10][
                        ix, 1
                    ]
                    old = supplement[f"{m['variant']}__old__{head}"][m["seed"] - 10][
                        ix
                    ][:, cols]
                    a.require(
                        np.array_equal(old, z["gated"][j]),
                        "Supplement original array mismatch",
                    )
                else:
                    v = z["gated"][j]
                    gate = z["gate"][j]
                source = (
                    SUPP / "paired_full_predictions.npz"
                    if m["pool"] == "uniform_compact_s42"
                    else a.GPU / (slug + "_arrays.npz")
                )
                add(
                    "gpu/147/tuned_update_sensitivity/" + m["variant"],
                    m["pool"],
                    m["seed"],
                    v,
                    gate,
                    np.ones(n, bool),
                    source,
                    m["id"],
                )
            for p, zp in cpu:
                for policy in ("default", "selected"):
                    add(
                        "cpu/" + p["root"] + "/" + policy,
                        "fixed",
                        p["seed"],
                        zp[policy + "_" + head][ix][:, cols],
                        zp[policy + "_hate"][ix, 1],
                        np.ones(n, bool),
                        p["predictions"],
                        p["root"] + "_" + policy,
                    )
            lx = [local[k] for k in ids]
            for s, zp in enumerate(native):
                for policy in ("default", "selected"):
                    add(
                        "cpu/full_neural/" + policy,
                        "fixed",
                        s,
                        zp[policy + "_" + head][lx][:, cols],
                        zp[policy + "_hate"][lx, 1],
                        np.ones(n, bool),
                        CPUREVIEW / f"neural/broad_s{s}_decisions.npz",
                        "full_neural_" + policy,
                    )
            for (root, variant, s), zp in curated.items():
                add(
                    root + "/" + variant,
                    "fixed",
                    s,
                    zp[head][ix][:, cols],
                    zp["hate"][ix, 1],
                    np.ones(n, bool),
                    (TUNE if root == "curated_original" else SHORT)
                    / "predictions"
                    / f"{variant}_s{s}.npz",
                    root.replace("curated_", "") + "_" + variant,
                )
            for s, rr in bge.items():
                values = np.asarray(
                    [decode_result(rr[k], head, labs) for k in ids], dtype=np.uint8
                )
                add(
                    "curated_bge/selected",
                    "fixed",
                    s,
                    values,
                    np.asarray([rr[k]["hate"] for k in ids], dtype=np.uint8),
                    np.ones(n, bool),
                    COMPACT / f"predictions_seed{s}.jsonl",
                    "bge",
                )
        predictions = np.stack(pred)
        validity = np.stack(valid)
        gate = np.stack(gates)
        a.require(
            np.all(validity[:, common]),
            "Common-nine contains a missing comparator result",
        )

        for m in models:
            m["gate_available"] = head != "legal" and (
                head == "hate"
                or m["group"].startswith(("qwen/", "gpu/", "cpu/", "curated_"))
                or (m["original_method"], m["seed"]) in hate_lookup
            )
        md = dict(
            endpoint=endpoint,
            ids=ids,
            labels=labs,
            macro_labels=ep["macro_labels"],
            macro_indices=[labs.index(l) for l in ep["macro_labels"]],
            gold_support=ep["gold_support"],
            models=models,
            eligible=n,
            common_nine=int(common.sum()),
            original_contrasts=result["contrasts"],
        )
        save("endpoints/" + slug + "/metadata.json", md)
        arr(
            "endpoints/" + slug + "/aligned.npz",
            gold=gold,
            predictions=predictions,
            valid=validity,
            gate=gate,
            common=common,
        )
        outputs += [
            BASE / "endpoints" / slug / "metadata.json",
            BASE / "endpoints" / slug / "aligned.npz",
        ]
        summary.append(
            dict(
                endpoint=endpoint,
                models=len(models),
                groups=len(set(m["group"] for m in models)),
                eligible=n,
                common_nine=int(common.sum()),
            )
        )
        print(
            endpoint,
            ":",
            len(models),
            "retained views aligned;",
            int(common.sum()),
            "common texts.",
            flush=True,
        )
    save(
        "extraction_complete.json",
        dict(
            status="saved_predictions_aligned_without_inference",
            created_utc=a.utc(),
            endpoints=summary,
            outputs=a.bind(outputs),
            new_model_calls=0,
        ),
    )


def divide(x, y):
    return np.divide(x, y, out=np.zeros_like(x, dtype=float), where=y != 0)


def metrics(y, x, macro, head, gate=None):

    y = np.asarray(y, dtype=bool)
    x = np.asarray(x, dtype=bool)
    n = len(y)
    a.require(n > 0 and y.shape == x.shape, "Empty/misaligned metric input")
    tp = (y & x).sum(0)
    fp = (~y & x).sum(0)
    fn = (y & ~x).sum(0)
    f = divide(2 * tp, 2 * tp + fp + fn)
    result = dict(
        n=n,
        macro_f1=float(f[macro].mean()) if macro else None,
        tp=tp.tolist(),
        fp=fp.tolist(),
        fn=fn.tolist(),
        gold_support=y.sum(0).tolist(),
        label_f1=f.tolist(),
        label_precision=divide(tp, tp + fp).tolist(),
        label_recall=divide(tp, tp + fn).tolist(),
        mean_predicted_labels=float(x.sum(1).mean()),
        mean_extra_labels=float((x & ~y).sum(1).mean()),
        mean_missing_labels=float((y & ~x).sum(1).mean()),
    )
    if gate is not None:
        result["gate_rejection_rate"] = float((~np.asarray(gate, dtype=bool)).mean())
    if head == "hate":
        neg = int((~y[:, 1]).sum())
        pos = int(y[:, 1].sum())
        FP = int(fp[1])
        FN = int(fn[1])
        TP = int(tp[1])
        TN = neg - FP
        result["binary"] = dict(
            tp=TP,
            tn=TN,
            fp=FP,
            fn=FN,
            gold_negative=neg,
            gold_positive=pos,
            fpr=FP / neg if neg else None,
            fnr=FN / pos if pos else None,
            precision=TP / (TP + FP) if TP + FP else 0.0,
            no_predicted_hate=int(TP + FP == 0),
            recall=TP / pos if pos else None,
            error_rate=(FP + FN) / n,
            predicted_hate_rate=float(x[:, 1].mean()),
        )
    return result


def mean_objects(values):
    first = values[0]
    if isinstance(first, dict):
        return {k: mean_objects([v[k] for v in values]) for k in first}
    if isinstance(first, list):
        return [mean_objects([v[i] for v in values]) for i in range(len(first))]
    if first is None:
        a.require(all(v is None for v in values), "Mixed undefined/defined aggregation")
        return None
    return float(np.mean(values))


def aggregate(models, values):
    groups = defaultdict(lambda: defaultdict(list))
    for m, v in zip(models, values):
        groups[m["group"]][m["pool"]].append(v)
    out = {}
    for group, pools in groups.items():
        means = {p: mean_objects(v) for p, v in pools.items()}
        point = mean_objects(list(means.values()))
        scores = [v["macro_f1"] for pp in pools.values() for v in pp]
        out[group] = dict(
            mean=point,
            pools=means,
            models=sum(map(len, pools.values())),
            pool_count=len(pools),
            fit_F1_range=[min(scores), max(scores)] if scores[0] is not None else None,
            pool_F1_range=(
                [
                    min(v["macro_f1"] for v in means.values()),
                    max(v["macro_f1"] for v in means.values()),
                ]
                if scores[0] is not None
                else None
            ),
        )
    return out


def score():
    a.verify(a.read(BASE / "analysis_contract.json")["sources"])
    a.verify(a.read(BASE / "extraction_complete.json")["outputs"])
    gold = {r["key"]: r for r in a.rows(a.ROSTER)}
    allout = []
    contrasts = []
    subgroups = []
    overlaps = []
    for e in a.read(BASE / "extraction_complete.json")["endpoints"]:
        ep = e["endpoint"]
        slug = ep.replace("/", "__")
        md = a.read(BASE / "endpoints" / slug / "metadata.json")
        z = load(BASE / "endpoints" / slug / "aligned.npz")
        head = ep.split("/")[1]
        y, x, valid, common = z["gold"], z["predictions"], z["valid"], z["common"]
        models = md["models"]
        macro = md["macro_indices"]
        per = {}
        for mode in ("own_valid", "common_nine"):
            values = []
            for j, m in enumerate(models):
                mask = valid[j] if mode == "own_valid" else common
                values.append(
                    metrics(
                        y[mask],
                        x[j, mask],
                        macro,
                        head,
                        z["gate"][j, mask] if m["gate_available"] else None,
                    )
                )

            per[mode] = dict(runs=values, groups=aggregate(models, values))
        for group, v in per["common_nine"]["groups"].items():
            own = per["own_valid"]["groups"][group]
            v["common_minus_own_F1"] = (
                None
                if v["mean"]["macro_f1"] is None
                else v["mean"]["macro_f1"] - own["mean"]["macro_f1"]
            )
        save("endpoints/" + slug + "/scores.json", dict(endpoint=ep, **per))
        allout.append(BASE / "endpoints" / slug / "scores.json")

        for c in md["original_contrasts"]:
            if c.get("mean_difference") is None:
                contrasts.append(dict(endpoint=ep, original=c, reproduced=None))
                continue
            li = [
                i
                for i, m in enumerate(models)
                if m["original_method"] == c["left"]
                and i
                < len(a.read(a.MAIN / "endpoints" / slug / "results.json")["settings"])
            ]
            ri = [
                i
                for i, m in enumerate(models)
                if m["original_method"] == c["right"]
                and i
                < len(a.read(a.MAIN / "endpoints" / slug / "results.json")["settings"])
            ]
            mask = valid[li + ri].all(0)
            left = np.mean(
                [metrics(y[mask], x[j, mask], macro, head)["macro_f1"] for j in li]
            )
            right = np.mean(
                [metrics(y[mask], x[j, mask], macro, head)["macro_f1"] for j in ri]
            )
            a.require(
                abs(left - c["left_common_mean"]) < 1e-12
                and abs(right - c["right_common_mean"]) < 1e-12
                and int(mask.sum()) == c["n_common_valid"],
                "Original Qwen paired result drift",
            )
            contrasts.append(
                dict(
                    endpoint=ep,
                    original=c,
                    reproduced=dict(
                        left=float(left),
                        right=float(right),
                        difference=float(left - right),
                    ),
                )
            )

        records = [gold[k] for k in md["ids"]]
        groups = {
            "source:" + s: np.array([r["source"] == s for r in records])
            for s in sorted({r["source"] for r in records})
        }
        if ep.startswith("en_hatecheck_eval/"):
            functions = [json.loads(r["raw"])["functionality"] for r in records]
            groups.update(
                {
                    "function:" + f: np.array([v == f for v in functions])
                    for f in sorted(set(functions))
                }
            )
        for name, subset in groups.items():
            mask = common & subset
            if not mask.any():
                subgroups.append(dict(endpoint=ep, stratum=name, n=0, groups={}))
                continue
            values = [
                metrics(
                    y[mask],
                    x[j, mask],
                    macro,
                    head,
                    z["gate"][j, mask] if m["gate_available"] else None,
                )
                for j, m in enumerate(models)
            ]
            subgroups.append(
                dict(
                    endpoint=ep,
                    stratum=name,
                    n=int(mask.sum()),
                    eligible=int(subset.sum()),
                    groups=aggregate(models, values),
                )
            )
        if head == "hate":
            qidx = [j for j, m in enumerate(models) if m["group"] == "qwen/rag_full"]
            yy = y[common, 1]
            qw = (x[qidx][:, common, 1] != yy).mean(0)
            for group in per["common_nine"]["groups"]:
                if group.startswith("qwen/"):
                    continue
                poolmap = defaultdict(list)
                for j, m in enumerate(models):
                    if m["group"] == group:
                        poolmap[m["pool"]].append(j)
                cw = np.mean(
                    [
                        np.mean(x[jj][:, common, 1] != yy, axis=0)
                        for jj in poolmap.values()
                    ],
                    axis=0,
                )
                overlap = dict(
                    both_correct=float(((1 - qw) * (1 - cw)).sum()),
                    qwen_only_wrong=float((qw * (1 - cw)).sum()),
                    comparator_only_wrong=float(((1 - qw) * cw).sum()),
                    both_wrong=float((qw * cw).sum()),
                )
                a.require(
                    abs(sum(overlap.values()) - len(yy)) < 1e-9,
                    "Binary pair partition does not sum to population",
                )
                overlaps.append(
                    dict(
                        endpoint=ep,
                        comparator=group,
                        n=len(yy),
                        mean_pair_counts=overlap,
                        qwen_errors=float(qw.sum()),
                        comparator_errors=float(cw.sum()),
                        pool_count=len(poolmap),
                        interpretation="Mean of Cartesian saved repeat/fit pairs, equal pool weight; not a trained ensemble or independent probability model.",
                    )
                )
        print(
            ep,
            ": per-fit, equal-pool, label, source/function and binary-error views scored.",
            flush=True,
        )
    save("original_qwen_contrasts.json", contrasts)
    save("subgroups.json", subgroups)
    save("binary_error_overlap.json", overlaps)
    allout += [
        BASE / "original_qwen_contrasts.json",
        BASE / "subgroups.json",
        BASE / "binary_error_overlap.json",
    ]
    save(
        "scoring_complete.json",
        dict(
            status="all_declared_descriptive_views_scored",
            created_utc=a.utc(),
            outputs=a.bind(allout),
            new_bootstrap_draws=0,
            new_inferential_tests=0,
            new_model_calls=0,
            original_contrasts=len(contrasts),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["freeze", "extract", "score"], required=True)
    args = p.parse_args()
    with a.g.offline():
        globals()[args.mode]()
    a.require(
        not any(
            k in sys.modules
            for k in ("torch", "transformers", "sentence_transformers", "openai")
        ),
        "Unexpected model/client import",
    )
