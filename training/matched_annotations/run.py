import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import time
from unittest.mock import patch
import numpy as np
from training.roberta import common as c
from training.roberta import train as trainer
from training.frozen_features import compact as bge
from training.frozen_features import pools as pools
from training.frozen_features import roberta as feature

ROOT = c.ROOT
BASE = ROOT / "experiments/astra/matched_type_pools_20261011_v1"
TUNING = c.BASE
BALANCE = ROOT / "experiments/astra/compact_balance_20261008_v1"
BROAD = ROOT / "experiments/astra/compact_broad_20261008_v1"
VECTORS = ROOT / "experiments/astra/later_evaluation_20261008_v1/retrieval_v3/vectors"
POOL_NAMES = [f"equal_type67_s{s}" for s in range(101, 106)] + [
    f"balanced_type67_s{s}" for s in range(111, 116)
]
HEAD = "hate_type"


@contextmanager
def isolated():

    def offline(*args, **kwargs):
        raise RuntimeError("Matched type study is offline; network disabled")

    with patch.object(c, "BASE", BASE), patch.object(
        socket.socket, "connect", offline
    ), patch.object(socket, "create_connection", offline):
        yield


def save(path, value):
    with isolated():
        c.save(path, value)


def array(path, value):
    with isolated():
        c.array(path, value)


def arrays(path, value):
    with isolated():
        c.arrays(path, value)


def source_files():
    return [
        Path(__file__),
        ROOT / "tests/training/matched_annotations/test_run.py",
        ROOT / "scripts/run_matched_annotations.ps1",
        Path(c.__file__),
        Path(trainer.__file__),
        Path(bge.__file__),
        Path(pools.__file__),
        Path(feature.__file__),
        ROOT / "training/roberta/tokenizer.py",
        ROOT / "training/frozen_features/fit.py",
        ROOT / "analysis/common/study.py",
        ROOT / "analysis/common/api.py",
    ]


def hashes(paths):
    return {c.rel(p): c.sha(p) for p in paths}


def test_receipt_path():
    return BASE / (
        "development/tests_" + c.signature(hashes(source_files()))[:16] + ".json"
    )


def check_feature_origin(row, origin):
    c.require(
        row["key"] == origin["key"]
        and row["exact_text_sha256"] == origin["exact_text_sha256"],
        "Feature key/text provenance differs",
    )


def verify_draft_sources():
    draft = c.read(BASE / "draft.json")
    c.verify(draft["inputs"])
    c.verify(draft["prepared"])
    inputs = c.read(BASE / "data/evaluation_inputs.json")
    compact = c.read(bge.BASE / "plan.json")
    for p in [bge.BASE / "evaluation_inputs.json", bge.BASE / "evaluation_vectors.npy"]:
        expected = {**compact["inputs"], **compact["prepared"]}.get(c.rel(p))
        c.require(expected == c.sha(p), "Original compact evaluation vectors changed")
    broader = c.read(BROAD / "plan.json")
    p = BROAD / "prediction_inputs.json"
    c.require(broader["inputs"][c.rel(p)] == c.sha(p), "Broader vector mapping changed")
    cp_path = c.CORE / "dev_inputs.jsonl"
    c.require(
        compact["inputs"][c.rel(cp_path)] == c.sha(cp_path),
        "Original core texts changed",
    )
    core_rows = [
        json.loads(s) for s in cp_path.read_text(encoding="utf8").splitlines() if s
    ]
    origins = [
        json.loads(s)
        for s in (VECTORS / "query_origins.jsonl")
        .read_text(encoding="utf8")
        .splitlines()
        if s
    ]
    bp = {v["key"]: v for v in c.read(p)}
    cp = {v["item_id"]: v for v in c.read(bge.BASE / "evaluation_inputs.json")}
    cm = {v["key"]: v["source_id"] for v in c.read(TUNING / "data/scopes.json")["core"]}
    for k in ["experiments/astra/compact_bge_20261008_v1/fit_seed42.json"]:
        c.require(broader["inputs"][k] == c.sha(ROOT / k), "Fixed gate source changed")
    groups = {
        v["cluster"] for v in c.read(bge.BASE / "train.json")["heads"]["hate"]["rows"]
    }
    for row in inputs:
        c.require(
            pools.c.signature(bge.norm(row["text"])) not in groups,
            "Gate TRAIN/evaluation text overlap",
        )
        if row["key"] in bp:
            check_feature_origin(row, origins[bp[row["key"]]["vector_index"]])
        if row["key"] in cm:
            original = cp[cm[row["key"]]]
            source = core_rows[original["source_row"]]
            c.require(
                original["key"] == source["key"] == row["key"]
                and source["text"] == row["text"],
                "Core feature text/index differs",
            )
    return draft


def replay_bge_references():
    matrix = np.load(c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r")
    checked = []
    for pool in POOL_NAMES:
        d = data(pool)
        dd = dict(
            labels=d["heads"][HEAD]["labels"],
            rows=d["heads"][HEAD]["rows"],
            folds=d["folds"],
        )
        xx = np.asarray(
            matrix[[v["vector_index"] for v in dd["rows"]]], dtype=np.float64
        )
        fitted, n = pools.fit_type(dd, xx, 42, np.asarray(d["gate_oof"]))
        original = c.read(ROOT / d["source_bge_fit"])["heads"][HEAD]
        c.require(
            fitted == original,
            "Original BGE reference does not reproduce before long training",
        )
        checked.append(
            dict(
                pool=pool,
                candidate_checks=n,
                source_sha256=c.sha(ROOT / d["source_bge_fit"]),
            )
        )
    save(
        BASE / "development/bge_reference_replay.json",
        dict(
            status="pass",
            reference_reconstructions=10,
            pools=checked,
            new_scientific_results=False,
        ),
    )
    return checked


def definition():
    return dict(
        study="Matching methods across ten existing 67-type-annotation pools",
        stage="A: type budgets only",
        pools=POOL_NAMES,
        methods=[
            "unchanged BGE linear fold42",
            "frozen pretrained RoBERTa mean + matched balanced logistic",
            "fine-tuned RoBERTa: fixed unweighted, fixed balanced, TRAIN-selected",
        ],
        gate="One identical frozen original BGE gate, fold setting42, for all methods. Head-selection inputs overlapping its training groups use stored OOF gate probabilities; other inputs use its frozen final gate. No new gate labels or weights.",
        annotation_budget="67 type labels per pool, plus147 fixed gate annotations; union212-214 IDs. This is a type-head mechanism study, not a whole matched147-text deployment system.",
        folds="Existing pool fold42; shared across frozen and neural methods. Preserve existing sparse/constant-label handling and no redraw.",
        neural=dict(
            configurations=c.configs(),
            search_seeds=c.CV_SEEDS,
            final_seeds=c.FINAL_SEEDS,
            cutoffs=c.THRESHOLDS,
            selection="Mean per-neural-seed OOF delivered macro-F1 using the common gate; deterministic inherited ties. Same full grid for every pool, supported labels >=2 positives/negatives.",
            controls="Fixed unweighted and fixed balanced l1_e20 recipes. Every family yields default and TRAIN-selected cutoffs on identical weights. Preserve all five final seeds.",
        ),
        linear="Existing balanced L2 logistic C=.1,1,10,100; global cutoff .1-.9; same folds and common gate. Reproduce existing BGE fold42 exactly before reusing it. Frozen RoBERTa mean pooling is fixed, no CLS winner search.",
        representation="Pinned original roberta-base, guarded historical max128 tokenizer; mean of nonpadding final hidden states including special tokens, L2 normalized, FP32 SDPA. Reuse checked evaluation features; extract only missing TRAIN identities.",
        endpoints=[
            "core/en_dev_eval_sq3_types",
            "core/en_dev_eval_types",
            "broad/en_test/hate_type",
        ],
        primary="Four TEST contrasts: neural tuned-selected minus BGE and frozen-RoBERTa minus BGE, each averaged within equal67 and balanced67. Average neural seeds within pool then pools equally. 10000 paired whole-text-cluster draws shared across methods/pools;98.75% intervals conditional on fitted models and fixed pools.",
        supporting="Every pool/seed, default/balanced controls, head-only F1, exact sets, label cardinality/precision/recall, ranking, gate suppression, learning curves and costs. Pool spread separate from initialization. Joint paired pool/text resampling is exploratory sensitivity with only5 draws/group, not extra independent test datasets.",
        exposures="Motivated by prior exposed results. TRAIN-only selection is fairer procedure development, not untouched TEST confirmation, a new algorithm, universal representation dominance or unbiased OOF performance.",
        resume="Single local lock. Immutable receipts; verified fits skipped, incomplete fits restarted from pretrained weights with old attempt preserved. Prediction chunks verified/reused; orphan arrays compared to deterministic rerun. No Qwen.",
        resources="1800 CV fits; up to150 final checkpoints (about70GiB). Require85GiB free before first scientific job and12GiB before later fits. No old files deleted. Technical pilots excluded.",
        remaining="StageB nine same-total-text pools/all-head comparisons still to prepare, then full-English-data BGE and word/character TF-IDF. Neither is included or cancelled by this command.",
    )


def audit_data(data):
    rows = data["heads"][HEAD]["rows"]
    folds = data["folds"]["42"]
    y = np.asarray([r["target"] for r in rows])
    c.require(
        len(rows) == 67 and len(data["texts"]) == 67 and (len(folds) == 67),
        "Annotation budget differs",
    )
    c.require(
        [r["feature_row"] for r in rows] == list(range(67))
        and len({r["id"] for r in rows}) == 67,
        "Row identity",
    )
    c.require(
        y.shape == (67, 7) and set(np.unique(y)) <= {0, 1}, "Label shape or values"
    )
    c.require(
        set(folds) == set(range(5)) and len(data["gate_oof"]) == 67, "Folds/gate shape"
    )
    c.require(set(data["gate_oof"]) <= {True, False}, "Gate decisions invalid")
    for k in range(5):
        fit = [r for r, f in zip(rows, folds) if f != k]
        held = [r for r, f in zip(rows, folds) if f == k]
        c.require(
            fit
            and held
            and (not {r["cluster"] for r in fit} & {r["cluster"] for r in held}),
            "Fold leakage/empty partition",
        )
    c.require(
        ((y.sum(0) >= 2) & ((1 - y).sum(0) >= 2)).any(), "No supported selection labels"
    )


def prepare():
    if (BASE / "draft.json").exists():
        d = c.read(BASE / "draft.json")
        c.verify(d["inputs"])
        c.verify(d["prepared"])
        print("Existing matched draft verified")
        return
    original = c.read(TUNING / "plan.json")
    c.verify(
        {k: v for k, v in original["inputs"].items() if k.endswith((".py", ".ps1"))}
    )
    compact = c.read(bge.BASE / "plan.json")
    record_path = c.CORE / "kb/full/records.jsonl"
    vector_path = c.CORE / "retrieval_v1/full_vectors.npy"
    for p in [record_path, vector_path]:
        c.require(
            c.sha(p) == compact["inputs"][c.rel(p)], "Full training source changed"
        )
    records = [
        json.loads(s) for s in record_path.read_text(encoding="utf8").splitlines() if s
    ]
    index = {r["id"]: i for i, r in enumerate(records)}
    byid = {r["id"]: r for r in records}
    matrix = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    small = c.read(bge.BASE / "train.json")
    gate = c.read(bge.BASE / "fit_seed42.json")["heads"]["hate"]
    gate_groups = {
        r["cluster"]: i for i, r in enumerate(small["heads"]["hate"]["rows"])
    }
    files = [
        record_path,
        vector_path,
        bge.BASE / "plan.json",
        bge.BASE / "train.json",
        bge.BASE / "fit_seed42.json",
        TUNING / "plan.json",
        TUNING / "complete.json",
        TUNING / "data/inference.json",
        TUNING / "data/scopes.json",
        feature.BASE / "feature_inputs.json",
        feature.BASE / "features_mean.npy",
        feature.BASE / "plan.json",
        feature.BASE / "features_complete.json",
        BROAD / "plan.json",
        BROAD / "prediction_inputs.json",
        VECTORS / "query_origins.jsonl",
        VECTORS / "query_vectors.npy",
        VECTORS / "complete.json",
        bge.BASE / "evaluation_inputs.json",
        bge.BASE / "evaluation_vectors.npy",
    ]
    c.require(
        c.sha(feature.BASE / "features_mean.npy")
        == c.read(feature.BASE / "features_complete.json")["outputs"][
            c.rel(feature.BASE / "features_mean.npy")
        ],
        "Frozen evaluation features changed",
    )
    vi = c.read(VECTORS / "complete.json")
    c.verify(vi["outputs"])
    full_inputs = c.read(TUNING / "data/inference.json")
    scopes = c.read(TUNING / "data/scopes.json")
    inputmap = {v["key"]: v for v in full_inputs}
    scope_maps = {s: {v["source_id"]: v["key"] for v in vv} for s, vv in scopes.items()}
    endpoints = {}
    keys = set()
    for scope, name in [
        ("core", "en_dev_eval_sq3_types"),
        ("core", "en_dev_eval_types"),
        ("broad", "en_test/hate_type"),
    ]:
        ep = c.read(bge.BASE / "plan.json" if scope == "core" else BROAD / "plan.json")[
            "endpoints"
        ][name]
        kk = [scope_maps[scope][x] for x in ep["scorable_ids"]]
        keys.update(kk)
        endpoints[scope + "/" + name] = dict(endpoint=ep, keys=kk)
    inputs = [inputmap[k] for k in sorted(keys)]
    feature_inputs = c.read(feature.BASE / "feature_inputs.json")
    existing = {v["key"]: (i, v) for i, v in enumerate(feature_inputs)}
    qcore = {
        v["item_id"]: i
        for i, v in enumerate(c.read(bge.BASE / "evaluation_inputs.json"))
    }
    qbro = {
        v["key"]: v["vector_index"] for v in c.read(BROAD / "prediction_inputs.json")
    }
    core_inverse = {v["key"]: v["source_id"] for v in scopes["core"]}
    corevectors = np.load(bge.BASE / "evaluation_vectors.npy", mmap_mode="r")
    broadvectors = np.load(VECTORS / "query_vectors.npy", mmap_mode="r")
    ex = np.stack(
        [
            (
                corevectors[qcore[core_inverse[v["key"]]]]
                if v["key"] in core_inverse
                else broadvectors[qbro[v["key"]]]
            )
            for v in inputs
        ]
    )
    for v in inputs:
        c.require(
            v["key"] in existing and v["text"] == existing[v["key"]][1]["text"],
            "Evaluation feature identity differs",
        )
    rf = np.load(feature.BASE / "features_mean.npy", mmap_mode="r")
    rx = np.asarray(rf[[existing[v["key"]][0] for v in inputs]])
    eg = bge.binary_predict(gate["models"][0], ex) > gate["selected"]["threshold"]
    prepared = []
    inventory = {}
    trainfeatures = {}
    evaluation_norm = {bge.norm(v["text"]) for v in inputs}
    for name in POOL_NAMES:
        parent = BALANCE if name.startswith("balanced") else pools.BASE
        source = parent / f"train/{name}.json"
        fitpath = parent / f"jobs/{name}_fold42/fit.json"
        receipt = parent / f"jobs/{name}_fold42/complete.json"
        old = c.read(source)
        sourceplan = c.read(parent / "plan.json")
        c.require(
            sourceplan["prepared"][c.rel(source)] == c.sha(source),
            "Pool membership changed",
        )
        c.verify(c.read(receipt)["outputs"])
        rr = []
        texts = []
        overlap = 0
        for i, v in enumerate(old["rows"]):
            rec = byid[v["id"]]
            c.require(
                index[v["id"]] == v["vector_index"]
                and v["cluster"] == pools.c.signature(bge.norm(rec["text"])),
                "Training origin differs",
            )
            c.require(
                rec["kind"] == "example"
                and rec["lang"] == "en"
                and (rec["meta"]["gate"] is True),
                "Wrong training source",
            )
            target = [int(l in rec["meta"]["hate_types"]) for l in old["labels"]]
            c.require(target == v["target"], "Pool labels changed")
            c.require(
                bge.norm(rec["text"]) not in evaluation_norm,
                "TRAIN/evaluation text overlap",
            )
            rr.append(dict(v, feature_row=i))
            texts.append(rec["text"])
            trainfeatures[v["id"]] = dict(
                key="train/" + v["id"],
                text=rec["text"],
                exact_text_sha256=hashlib.sha256(rec["text"].encode()).hexdigest(),
            )
        x = np.asarray(matrix[[v["vector_index"] for v in rr]], dtype=np.float64)
        gp = bge.binary_predict(gate["models"][0], x)
        for i, v in enumerate(rr):
            if v["cluster"] in gate_groups:
                gi = gate_groups[v["cluster"]]
                gp[i] = gate["oof_probabilities"][gi][0]
                overlap += 1
                c.require(
                    old["folds"]["42"][i] == small["folds"]["42"][gi],
                    "Gate/type fold group differs",
                )
        data = dict(
            pool=name,
            heads={HEAD: dict(labels=old["labels"], kind="multilabel", rows=rr)},
            texts=texts,
            folds={"42": old["folds"]["42"]},
            gate_oof=(gp > gate["selected"]["threshold"]).tolist(),
            source_train=c.rel(source),
            source_bge_fit=c.rel(fitpath),
        )
        audit_data(data)
        path = BASE / f"data/{name}.json"
        save(path, data)
        prepared.append(path)
        inventory[name] = dict(
            type_annotations=67,
            gate_annotations=147,
            union_training_ids=len(set(small["gate_ids"]) | {v["id"] for v in rr}),
            gate_overlap_groups=overlap,
            labels=dict(
                zip(
                    old["labels"], np.asarray([v["target"] for v in rr]).sum(0).tolist()
                )
            ),
            fold_counts=dict(Counter(old["folds"]["42"])),
            source_train_sha256=c.sha(source),
            source_fit_sha256=c.sha(fitpath),
        )
        files += [source, fitpath, receipt, parent / "plan.json"]
    trainfeatures = [trainfeatures[k] for k in sorted(trainfeatures)]
    save(BASE / "data/feature_inputs.json", trainfeatures)
    save(BASE / "data/evaluation_inputs.json", inputs)
    save(BASE / "data/endpoints.json", endpoints)
    save(BASE / "data/inventory.json", inventory)
    arrays(
        BASE / "data/evaluation_features.npz",
        dict(bge=ex, roberta=rx, gate=eg.astype(np.int64)),
    )
    prepared += [
        BASE / f"data/{n}"
        for n in [
            "feature_inputs.json",
            "evaluation_inputs.json",
            "endpoints.json",
            "inventory.json",
            "evaluation_features.npz",
        ]
    ]
    save(BASE / "protocol.json", definition())
    prepared.append(BASE / "protocol.json")
    save(
        BASE / "draft.json",
        dict(
            status="data_protocol_frozen_before_new_scientific_fits",
            base_cache=original["base_cache"],
            environment=c.environment(),
            inputs=hashes(sorted(set(files))),
            prepared=hashes(prepared),
            training_feature_identities=len(trainfeatures),
            evaluation_identities=len(inputs),
            max_final_models=150,
            search_fits=1800,
        ),
    )
    print(
        json.dumps(
            dict(
                status="prepared",
                pools=10,
                training_feature_identities=len(trainfeatures),
                evaluation_identities=len(inputs),
                search_fits=1800,
                max_final_models=150,
            )
        ),
        flush=True,
    )


def data(pool):
    c.require(pool in POOL_NAMES, "Unknown pool")
    d = c.read(BASE / f"data/{pool}.json")
    audit_data(d)
    return d


def job(pool, cfg, seed, fold=None):
    return dict(
        id=f"{pool}/{HEAD}/{cfg}/s{seed}" + ("" if fold is None else f"_f{fold}"),
        pool=pool,
        mode="final" if fold is None else "search",
        head=HEAD,
        config=cfg,
        seed=seed,
        fold=fold,
    )


def search_jobs(pool):
    return [
        job(pool, cfg["id"], s, k)
        for cfg in c.configs()
        for s in c.CV_SEEDS
        for k in range(5)
    ]


def check_training(j, d, plan_sha):
    with isolated():
        receipt, attempt = c.verify_receipt(c.job_dir(j) / "complete.json", j, plan_sha)
    m = c.read(attempt / "training.json")
    spec, fit, held = trainer.split(d, j)
    c.require(
        m["job"] == j
        and (not m["pilot_only"])
        and (m["configuration"] == c.config_map()[j["config"]]),
        "Wrong scientific training identity",
    )
    c.require(
        m["fit_ids"] == [v["id"] for v in fit]
        and m["held_ids"] == [v["id"] for v in held],
        "Training/held membership differs",
    )
    c.require(
        m["attempted_steps"]
        == m["planned_steps"]
        == math.ceil(len(fit) / 16) * m["configuration"]["epochs"],
        "Incomplete step schedule",
    )
    c.require(
        m["successful_steps"] == sum((not v["skipped"] for v in m["steps"]))
        and m["successful_steps"] > 0,
        "Step accounting",
    )
    c.require(
        len(m["epochs"]) == m["configuration"]["epochs"]
        and all((np.isfinite(v["training_objective"]) for v in m["epochs"])),
        "Incomplete/nonfinite curves",
    )
    weights = c.class_weights(
        HEAD,
        np.asarray([v["target"] for v in fit]),
        7,
        m["configuration"]["loss"] == "balanced",
    )
    c.require(m["class_weights"] == weights, "Weights use wrong labels")
    if held:
        z = np.load(attempt / "held_logits.npy", allow_pickle=False)
        c.require(
            z.shape == (len(held), 7) and np.isfinite(z).all(), "Held logits invalid"
        )
        y = np.asarray([v["target"] for v in held])
        loss = float(np.mean(np.logaddexp(0.0, z.astype(float)) - y * z))
        c.require(
            abs(loss - m["epochs"][-1]["held_unweighted_loss"]) < 2e-06,
            "Held loss reconstruction differs",
        )
    else:
        c.require(
            m["export_check"]["clean_reload"]
            and m["export_check"]["max_abs_logit_difference"] <= 0.0001,
            "CPU export probe failed",
        )
    return attempt


def select_neural(d, oof):
    y = np.asarray([v["target"] for v in d["heads"][HEAD]["rows"]])
    support = np.flatnonzero((y.sum(0) >= 2) & ((1 - y).sum(0) >= 2))
    gate = np.asarray(d["gate_oof"])
    answer = {}
    from sklearn.metrics import f1_score

    for family in c.FAMILIES:
        configs = (
            list(c.config_map())
            if family == "tuned"
            else [c.protocol()["fixed_configs"][family]]
        )
        candidates = []
        for cfg in configs:
            pp = oof[cfg]
            c.require(
                pp.shape == (3, len(y), 7) and np.isfinite(pp).all(),
                "Incomplete OOF probabilities",
            )
            for threshold in c.THRESHOLDS:
                ss = []
                for row in pp:
                    pred = c.decisions(HEAD, row, threshold, gate)
                    score = c.macro(y, pred, support)
                    independent = f1_score(
                        y[:, support],
                        pred[:, support],
                        average="macro",
                        zero_division=0,
                    )
                    c.require(
                        abs(score - independent) < 1e-12, "Selection metric differs"
                    )
                    ss.append(score)
                candidates.append(
                    dict(
                        config=cfg,
                        threshold=threshold,
                        seed_scores=ss,
                        mean_score=float(np.mean(ss)),
                    )
                )
        answer[family] = dict(
            selected=min(candidates, key=c.candidate_key),
            candidates=candidates,
            selection_indices=support.tolist(),
        )
    return answer


def select(pool, plan_sha):
    d = data(pool)
    oof = {cfg: np.full((3, 67, 7), np.nan) for cfg in c.config_map()}
    folds = np.asarray(d["folds"]["42"])
    receipts = {}
    for j in search_jobs(pool):
        a = check_training(j, d, plan_sha)
        oof[j["config"]][j["seed"], folds == j["fold"]] = c.probabilities(
            HEAD, np.load(a / "held_logits.npy", allow_pickle=False)
        )
        receipts[c.rel(BASE / "search" / j["id"] / "complete.json")] = c.sha(
            BASE / "search" / j["id"] / "complete.json"
        )
    selection = select_neural(d, oof)
    save(
        BASE / f"selections/{pool}.json",
        dict(
            plan_sha256=plan_sha,
            pool=pool,
            families=selection,
            search_receipts=receipts,
        ),
    )
    return selection


def final_jobs(pool, selection):
    return [
        job(pool, cfg, s)
        for cfg in sorted({v["selected"]["config"] for v in selection.values()})
        for s in c.FINAL_SEEDS
    ]


def preflight():
    c.require(not (BASE / "plan.json").exists(), "Scientific plan already sealed")
    draft = verify_draft_sources()
    replay_bge_references()
    d = data(POOL_NAMES[0])
    out = BASE / "development/pilot_v1"
    c.require(not (out / "complete.json").exists(), "Preserve completed pilot")
    tok, model, _ = feature.load("cpu", draft)
    probe = [
        "A neutral test sentence.",
        "A slightly longer text about a quiet walk beside the river.",
        "Short.",
        "Words in a different order still need stable padding.",
    ]
    cpu = feature.infer(tok, model, probe, "cpu", check_numpy=True)["mean"]
    del tok, model
    gc.collect()
    array(out / "CPU_features.npy", cpu)
    torch = trainer.setup()
    tok, model, _ = feature.load("cuda", draft)
    gpu = feature.infer(tok, model, probe, "cuda", check_numpy=True)["mean"]
    drift = float(np.max(np.abs(cpu - gpu)))
    c.require(drift <= 0.0001, "Feature CPU/GPU drift")
    array(out / "GPU_features.npy", gpu)
    del tok, model
    gc.collect()
    torch.cuda.empty_cache()
    fits = []
    for cfg in ["l1_e20_u", "l1_e20_b"]:
        j = job(POOL_NAMES[0], cfg, 909, 0)
        attempt = out / cfg
        attempt.mkdir(parents=True, exist_ok=False)
        with isolated():
            m = trainer.train_attempt(
                attempt,
                j,
                d,
                draft["base_cache"]["snapshot"],
                pilot=True,
                save_model=True,
            )
        c.require(
            m["attempted_steps"] == 8
            and m["pilot_only"]
            and (m["export_check"]["max_abs_logit_difference"] <= 0.0001),
            "Pilot incomplete",
        )
        fits.append(
            dict(
                config=cfg,
                steps=8,
                seconds=m["total_seconds"],
                peak_GiB=m["peak_allocated_gib"],
                cpu_export=m["export_check"],
            )
        )
    save(
        out / "complete.json",
        dict(
            status="technical_pilot_pass",
            excluded_from_scientific_results=True,
            feature_max_abs_difference=drift,
            neural_fits=fits,
            outputs=hashes([p for p in out.rglob("*") if p.is_file()]),
            sources=hashes(source_files()),
            Qwen_calls=0,
        ),
    )
    print(
        "Two excluded8-step GPU fits, CPU model exports and frozen feature CPU/GPU comparison passed.",
        flush=True,
    )


def seal():
    d = verify_draft_sources()
    pilot = c.read(BASE / "development/pilot_v1/complete.json")
    c.require(pilot["status"] == "technical_pilot_pass", "Pilot required")
    c.verify(pilot["outputs"])
    c.verify(pilot["sources"])
    tests = c.read(test_receipt_path())
    c.require(tests["passed"], "Tests required")
    c.verify(tests["sources"])
    replay = c.read(BASE / "development/bge_reference_replay.json")
    c.require(
        replay["status"] == "pass" and replay["reference_reconstructions"] == 10,
        "BGE reference replay required",
    )
    plan = dict(
        d,
        status="sealed_ready_no_scientific_new_fits",
        inputs={
            **d["inputs"],
            **hashes(source_files()),
            **hashes(
                [
                    BASE / "draft.json",
                    BASE / "development/pilot_v1/complete.json",
                    test_receipt_path(),
                    BASE / "development/bge_reference_replay.json",
                ]
            ),
        },
        pilot_sha256=c.sha(BASE / "development/pilot_v1/complete.json"),
        tests_sha256=c.sha(test_receipt_path()),
    )
    save(BASE / "plan.json", plan)
    print("Matched type study sealed. No scientific training started.", flush=True)


def validate():
    p = c.read(BASE / "plan.json")
    c.verify(p["inputs"])
    c.verify(p["prepared"])
    c.require(p["environment"] == c.environment(), "Changed runtime environment")
    for name, h in p["base_cache"]["files"].items():
        c.require(
            c.sha(Path(p["base_cache"]["snapshot"]) / name) == h,
            "Pretrained cache changed",
        )
    return p


def make_features(plan):
    done = BASE / "features/complete.json"
    if done.exists():
        v = c.read(done)
        c.require(v["plan_sha256"] == c.sha(BASE / "plan.json"), "Feature plan differs")
        c.verify(v["outputs"])
        return
    rows = c.read(BASE / "data/feature_inputs.json")
    old = c.read(feature.BASE / "feature_inputs.json")
    lookup = {v["key"]: (i, v) for i, v in enumerate(old)}
    source = np.load(feature.BASE / "features_mean.npy", mmap_mode="r")
    out = np.empty((len(rows), 768), dtype=np.float32)
    missing = []
    reused = 0
    for i, row in enumerate(rows):
        if row["key"] in lookup:
            k, prior = lookup[row["key"]]
            c.require(prior["text"] == row["text"], "Cached TRAIN feature text differs")
            out[i] = source[k]
            reused += 1
        else:
            missing.append(i)
    tok = model = None
    chunks = []
    try:
        for start in range(0, len(missing), 64):
            ix = missing[start : start + 64]
            path = BASE / f"features/chunks/{start:06d}.npy"
            receipt = path.with_suffix(".json")
            identity = dict(
                plan_sha256=c.sha(BASE / "plan.json"),
                positions=ix,
                input_sha256=c.signature([rows[i] for i in ix]),
            )
            if receipt.exists():
                rr = c.read(receipt)
                c.require(
                    rr["identity"] == identity and c.sha(path) == rr["sha256"],
                    "Feature chunk differs",
                )
                values = np.load(path, allow_pickle=False)
            else:
                if model is None:
                    trainer.setup()
                    tok, model, _ = feature.load("cuda", plan)
                values = np.concatenate(
                    [
                        feature.infer(
                            tok,
                            model,
                            [rows[i]["text"] for i in ix[j : j + 16]],
                            "cuda",
                        )["mean"]
                        for j in range(0, len(ix), 16)
                    ]
                )
                if path.exists():
                    c.require(
                        np.array_equal(np.load(path, allow_pickle=False), values),
                        "Interrupted feature chunk differs; preserve",
                    )
                else:
                    array(path, values)
                save(receipt, dict(identity=identity, sha256=c.sha(path)))
            c.require(
                values.shape == (len(ix), 768)
                and np.isfinite(values).all()
                and np.allclose(np.linalg.norm(values, axis=1), 1, atol=0.0002),
                "Invalid feature chunk",
            )
            out[ix] = values
            chunks += [path, receipt]
            print(
                "Matched TRAIN features:",
                min(start + 64, len(missing)),
                "/",
                len(missing),
                flush=True,
            )
    finally:
        if model is not None:
            del model, tok
            gc.collect()
            import torch

            torch.cuda.empty_cache()
    array(BASE / "features/mean.npy", out)
    save(
        done,
        dict(
            plan_sha256=c.sha(BASE / "plan.json"),
            outputs=hashes([BASE / "features/mean.npy", *chunks]),
            reused_old_rows=reused,
            new_rows=len(missing),
        ),
    )


def linear_stage(pool, plan_sha):
    directory = BASE / f"linear/{pool}"
    done = directory / "complete.json"
    if done.exists():
        v = c.read(done)
        c.require(v["plan_sha256"] == plan_sha, "Linear plan differs")
        c.verify(v["outputs"])
        return
    d = data(pool)
    dd = dict(
        labels=d["heads"][HEAD]["labels"],
        rows=d["heads"][HEAD]["rows"],
        folds=d["folds"],
    )
    gate = np.asarray(d["gate_oof"])
    seed = 42
    matrix = np.load(c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r")
    bx = np.asarray(matrix[[v["vector_index"] for v in dd["rows"]]], dtype=np.float64)
    new_bge, checks = pools.fit_type(dd, bx, seed, gate)
    old = c.read(ROOT / d["source_bge_fit"])["heads"][HEAD]
    c.require(
        new_bge == old, "Reproduced BGE pool fit differs; do not change the reference"
    )
    fi = c.read(BASE / "data/feature_inputs.json")
    positions = {v["key"]: i for i, v in enumerate(fi)}
    matrix = np.load(BASE / "features/mean.npy", mmap_mode="r")
    rx = np.asarray(
        matrix[[positions["train/" + v["id"]] for v in dd["rows"]]], dtype=np.float64
    )
    rob, checks2 = pools.fit_type(dd, rx, seed, gate)
    save(
        directory / "fits.json",
        dict(
            plan_sha256=plan_sha,
            pool=pool,
            bge_imported=old,
            roberta=rob,
            candidate_checks=checks + checks2,
        ),
    )
    with np.load(BASE / "data/evaluation_features.npz", allow_pickle=False) as e:
        values = {
            m: np.column_stack(
                [
                    bge.binary_predict(v, np.asarray(e[m], dtype=np.float64))
                    for v in head["models"]
                ]
            )
            for m, head in [("bge", old), ("roberta", rob)]
        }
    arrays(directory / "probabilities.npz", values)
    save(
        done,
        dict(
            plan_sha256=plan_sha,
            outputs=hashes([directory / "fits.json", directory / "probabilities.npz"]),
        ),
    )


def complete_pool(pool, selection, plan_sha):
    d = data(pool)
    inputs = c.read(BASE / "data/evaluation_inputs.json")
    paths = []
    with np.load(BASE / "data/evaluation_features.npz", allow_pickle=False) as e:
        gate = e["gate"].astype(bool)
    for j in final_jobs(pool, selection):
        check_training(j, d, plan_sha)
        with isolated():
            trainer.inference_job(j, d, plan_sha, inputs)
        z = np.load(BASE / "inference" / j["id"] / "logits.npy", allow_pickle=False)
        c.require(z.shape == (len(inputs), 7), "Final inference size differs")
        pp = c.probabilities(HEAD, z)
        for family, value in selection.items():
            if value["selected"]["config"] != j["config"]:
                continue
            for policy in ["default", "selected"]:
                threshold = (
                    0.5 if policy == "default" else value["selected"]["threshold"]
                )
                raw = c.decisions(HEAD, pp, threshold)
                delivered = raw.copy()
                delivered[~gate] = 0
                path = BASE / f"predictions/{pool}/{family}_{policy}_s{j['seed']}.npz"
                arrays(path, dict(ungated=raw, delivered=delivered))
                paths.append(path)
        paths.append(BASE / "inference" / j["id"] / "complete.json")
        paths.append(BASE / "refits" / j["id"] / "complete.json")
    paths += [BASE / f"selections/{pool}.json", BASE / f"linear/{pool}/complete.json"]
    save(
        BASE / f"pool_complete/{pool}.json",
        dict(
            status="training_and_predictions_complete",
            plan_sha256=plan_sha,
            pool=pool,
            final_models=len(final_jobs(pool, selection)),
            predictions=30 * len(inputs),
            outputs=hashes(paths),
        ),
    )


def verify_pool(pool, plan_sha):
    receipt = c.read(BASE / f"pool_complete/{pool}.json")
    c.require(
        receipt["pool"] == pool
        and receipt["plan_sha256"] == plan_sha
        and (receipt["status"] == "training_and_predictions_complete"),
        "Pool completion identity differs",
    )
    c.verify(receipt["outputs"])
    d = data(pool)
    selected = c.read(BASE / f"selections/{pool}.json")
    c.require(
        selected["plan_sha256"] == plan_sha and selected["pool"] == pool,
        "Selection identity differs",
    )
    c.verify(selected["search_receipts"])
    for j in search_jobs(pool):
        check_training(j, d, plan_sha)
    for j in final_jobs(pool, selected["families"]):
        check_training(j, d, plan_sha)
        done = c.read(BASE / "inference" / j["id"] / "complete.json")
        c.require(
            done["job"] == j and done["plan_sha256"] == plan_sha,
            "Inference identity differs",
        )
        c.verify(done["artifacts"])
    linear = c.read(BASE / f"linear/{pool}/complete.json")
    c.require(linear["plan_sha256"] == plan_sha, "Linear completion differs")
    c.verify(linear["outputs"])


def run():
    plan = validate()
    plan_sha = c.sha(BASE / "plan.json")
    with isolated(), c.launcher_lock():
        if (BASE / "complete.json").exists():
            done = c.read(BASE / "complete.json")
            c.require(done["plan_sha256"] == plan_sha, "Study completion differs")
            c.verify(done["outputs"])
            for pool in POOL_NAMES:
                verify_pool(pool, plan_sha)
            print("VERIFIED COMPLETE. No new fitting.", flush=True)
            return
        started = BASE / "execution_started.json"
        if not started.exists():
            c.require(
                shutil.disk_usage(ROOT).free >= 85 * 2**30,
                "Need85GiB free for retained checkpoints; preserve files and ask Astra before proceeding",
            )
            save(started, dict(plan_sha256=plan_sha, utc=c.utc()))
        make_features(plan)
        for pool in POOL_NAMES:
            done = BASE / f"pool_complete/{pool}.json"
            if done.exists():
                verify_pool(pool, plan_sha)
                print("Verified completed pool", pool, flush=True)
                continue
            d = data(pool)
            print(
                "POOL",
                pool,
                "180 TRAIN CV fits followed by final fits/predictions",
                flush=True,
            )
            for j in search_jobs(pool):
                if not (BASE / "search" / j["id"] / "complete.json").exists():
                    c.require(
                        shutil.disk_usage(ROOT).free >= 12 * 2**30,
                        "Less than12GiB free; preserve outputs",
                    )
                trainer.execute(j, d, plan, plan_sha)
            selection = select(pool, plan_sha)
            linear_stage(pool, plan_sha)
            for j in final_jobs(pool, selection):
                if not (BASE / "refits" / j["id"] / "complete.json").exists():
                    c.require(
                        shutil.disk_usage(ROOT).free >= 12 * 2**30,
                        "Less than12GiB free; preserve outputs",
                    )
                trainer.execute(j, d, plan, plan_sha)
            complete_pool(pool, selection, plan_sha)
            print("POOL AUDIT PASSED:", pool, flush=True)
        receipts = [BASE / f"pool_complete/{p}.json" for p in POOL_NAMES]
        c.verify(plan["inputs"])
        c.verify(plan["prepared"])
        save(
            BASE / "complete.json",
            dict(
                status="all_ten_matched_type_pools_complete",
                plan_sha256=plan_sha,
                outputs=hashes(receipts),
                Qwen_calls=0,
                note="All saved outputs require scientific review; no evaluation scores or preferred model selected by this launcher.",
            ),
        )
        print(
            "ALL TEN MATCHED TYPE POOLS COMPLETE. Tell Astra to inspect. No next study started.",
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "preflight", "seal", "run"])
    args = ap.parse_args()
    with isolated():
        globals()[args.mode]()
