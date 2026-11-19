from collections import Counter, defaultdict
from pathlib import Path
import argparse
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from training.cpu import validate as r
from training.roberta import results as earlier

c = r.c
previous = earlier.previous
ENC = previous.ENC


def neural_probability(head, logits):
    z = np.asarray(logits, dtype=np.float64)
    c.require(np.isfinite(z).all(), "Nonfinite retained logits")
    if head == "hate":
        exp = np.exp(z - z.max(1, keepdims=True))
        return (exp / exp.sum(1, keepdims=True))[:, 1:2]
    if head == "severity":
        exp = np.exp(z - z.max(1, keepdims=True))
        return exp / exp.sum(1, keepdims=True)
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def neural():
    c.require(
        c.read(r.OUT / "inventory.json")["status"] == "pass", "CPU inventory required"
    )
    full = c.read(r.BASE / "data/full_heads.json")
    scopes = c.read(r.BASE / "data/scopes.json")
    evaluation = c.read(r.BASE / "data/evaluation.json")
    inputs = {v["key"]: v for v in evaluation}
    plan = c.read(ENC / "plan.json")
    plan_sha = c.sha(ENC / "plan.json")
    bindings = {c.rel(ENC / "plan.json"): plan_sha}
    jobs = c.read(r.BASE / "data/neural.json")
    choices = {}
    probabilities = {}
    records = []
    overlaps = []
    chunk_count = 0
    raw_count = 0
    for job in jobs:
        head, seed = job["head"], job["seed"]
        spec = full[head]
        original = next(v for v in plan["jobs"] if v["job"] == job["job"])
        c.require(
            original["checkpoint_sha256"] == job["checkpoint_sha256"]
            and original["meta"]["labels"] == spec["labels"],
            "Checkpoint/labels changed",
        )
        weight = c.ROOT / job["checkpoint"] / "model.safetensors"
        c.require(
            c.sha(weight) == job["checkpoint_sha256"], "Retained neural weights changed"
        )
        bindings[c.rel(weight)] = job["checkpoint_sha256"]
        rec, attempt = r.receipt(r.BASE / f"neural_cutoffs/{head}_s{seed}")
        saved = c.read(attempt / "selection.json")
        rows = list(c.rows(c.ROOT / job["logits"]))
        c.require(
            [str(v["id"]) for v in rows] == [v["source_id"] for v in spec["val"]],
            "Native-validation identity differs",
        )
        c.require(
            [v["target"] for v in rows] == [v["target"] for v in spec["val"]],
            "Native-validation targets differ",
        )
        p = neural_probability(head, [v["logits"] for v in rows])
        y = r.target(head, spec, spec["val"])
        selected = r.select(head, y, {1.0: p})
        r.check_selection(saved["selection"], selected)
        c.require(
            np.allclose(
                p,
                np.load(attempt / "validation_probabilities.npy", allow_pickle=False),
                rtol=0,
                atol=1e-15,
            ),
            "Validation probability conversion differs",
        )
        choices[head, seed] = selected
        for scope in ("core", "broad"):
            if scope == "core":
                source = (
                    c.ROOT / original["reference_attempt"] / "dev_head_logits.jsonl"
                )
                provenance = c.read(source.parent / "complete.json")
                digest = provenance["files"][c.rel(source)]
                bindings[c.rel(source.parent / "complete.json")] = c.sha(
                    source.parent / "complete.json"
                )
            else:
                folder = ENC / "predictions" / job["job"]
                source = folder / "heads.jsonl"
                provenance = c.read(folder / "complete.json")
                c.require(
                    provenance["job"] == job["job"]
                    and provenance["plan_sha256"] == plan_sha
                    and provenance["status"] == "fixed_checkpoint_inference_complete",
                    "Wrong broad inference receipt",
                )
                c.verify(provenance["outputs"])
                bindings.update(provenance["outputs"])
                digest = provenance["outputs"][c.rel(source)]
                bindings[c.rel(folder / "complete.json")] = c.sha(
                    folder / "complete.json"
                )
            c.require(c.sha(source) == digest, "Saved neural logits changed")
            bindings[c.rel(source)] = digest
            raw = list(c.rows(source))
            raw_count += len(raw)
            expected = scopes[scope]
            c.require(
                [v["key"] for v in raw] == [v["key"] for v in expected],
                "Retained inference order differs",
            )
            for item in raw:
                c.require(
                    item["text_sha256"] == inputs[item["key"]]["exact_text_sha256"]
                    and item["job"] == job["job"]
                    and item["labels"] == spec["labels"]
                    and item["checkpoint_sha256"] == job["checkpoint_sha256"],
                    "Raw neural provenance mismatch",
                )
            logits = np.asarray([v["logits"] for v in raw])
            p = neural_probability(head, logits)
            decoded = (
                logits.argmax(1).tolist()
                if head in ("hate", "severity")
                else (logits >= 0).astype(int).tolist()
            )
            c.require(
                decoded == [v["decision"] for v in raw],
                "Historical decisions differ from raw logits",
            )
            default = r.bits(head, p, None if head == "severity" else 0.5)
            compare = (
                default.argmax(1).tolist()
                if head in ("hate", "severity")
                else default.astype(int).tolist()
            )
            c.require(
                compare == decoded, "Probability conversion changes historical decision"
            )
            if scope == "broad":
                chunk_rows = []
                for path in sorted((folder / "chunks").glob("chunk_*.json")):
                    chunk = c.read(path)
                    c.require(
                        c.signature(chunk["rows"]) == chunk["rows_sha256"]
                        and chunk["stamp"]
                        == dict(plan_sha256=plan_sha, job=job["job"]),
                        "Raw inference chunk changed",
                    )
                    chunk_rows.extend(chunk["rows"])
                    chunk_count += 1
                c.require(chunk_rows == raw, "Chunk assembly differs from saved logits")
                del chunk_rows
            probabilities[scope, seed, head] = p
            records.append(
                dict(
                    scope=scope,
                    head=head,
                    seed=seed,
                    rows=len(raw),
                    checkpoint_sha256=job["checkpoint_sha256"],
                    selected_threshold=selected["selected"]["threshold"],
                    native_validation_rows=len(spec["val"]),
                )
            )
        ix = {v["key"]: i for i, v in enumerate(scopes["core"])}
        jx = {v["key"]: i for i, v in enumerate(scopes["broad"])}
        shared = sorted(ix.keys() & jx.keys())
        a = probabilities["core", seed, head][[ix[k] for k in shared]]
        b = probabilities["broad", seed, head][[jx[k] for k in shared]]
        cutoff = choices[head, seed]["selected"]["threshold"]
        overlaps.append(
            dict(
                head=head,
                seed=seed,
                shared=len(shared),
                max_probability_difference=float(np.max(np.abs(a - b), initial=0)),
                selected_decision_differences=int(
                    np.any(
                        r.bits(head, a, cutoff) != r.bits(head, b, cutoff), axis=1
                    ).sum()
                ),
                note="Keep corresponding scope-specific saved logits, never choose a preferable version for shared texts.",
            )
        )
    for scope in ("core", "broad"):
        old = previous.predictions(scope)
        for seed in range(3):
            out = {}
            probs = {h: probabilities[scope, seed, h] for h in c.HEADS}
            for policy in ("default", "selected"):
                thresholds = {
                    h: (
                        choices[h, seed]["selected"]["threshold"]
                        if policy == "selected"
                        else (None if h == "severity" else 0.5)
                    )
                    for h in c.HEADS
                }
                gate = r.bits("hate", probs["hate"], thresholds["hate"])
                out[policy + "_hate"] = gate
                for h in c.HEADS[1:]:
                    raw = r.bits(h, probs[h], thresholds[h])
                    given = raw.copy()
                    given[~gate[:, 1]] = False
                    out[policy + "_" + h] = given
                    out[policy + "_" + h + "_ungated"] = raw
            for i, row in enumerate(scopes[scope]):
                original = old["encoder_full", seed][row["source_id"]]
                c.require(
                    bool(out["default_hate"][i, 1]) == original["hate"],
                    "Default neural gate no longer reproduces",
                )
                for h in c.HEADS[1:]:
                    values = [
                        label
                        for label, bit in zip(full[h]["labels"], out["default_" + h][i])
                        if bit
                    ]
                    expected = (
                        ([original[h]] if original[h] is not None else [])
                        if h == "severity"
                        else original[h]
                    )
                    c.require(
                        values == expected,
                        "Default neural assembly no longer reproduces",
                    )
            r.save_arrays(f"neural/{scope}_s{seed}_probabilities.npz", probs)
            r.save_arrays(f"neural/{scope}_s{seed}_decisions.npz", out)
    r.save(
        "neural_audit.json",
        dict(
            status="pass",
            raw_head_rows_checked=raw_count,
            chunk_files_checked=chunk_count,
            jobs=records,
            choices={f"{h}_s{s}": v for (h, s), v in choices.items()},
            scope_overlap=overlaps,
            bindings=bindings,
            source_sha256=c.sha(Path(__file__)),
            new_neural_inference=0,
            Qwen_calls=0,
            CUDA_used=False,
            policy="Only the saved native-validation cutoffs changed. Default decoded outputs reproduce the historical assemblies exactly.",
        ),
    )
    print(
        "All saved full-neural logits checked; all 12 cutoff selections reconstructed; both policies saved.",
        flush=True,
    )


def endpoint_context(scope, name):
    plan = c.read(
        previous.BGE / "plan.json" if scope == "core" else previous.BROAD / "plan.json"
    )
    ep = plan["endpoints"][name]
    rows, kept, ids, y = earlier.gold(scope, name, ep)
    scopes = c.read(r.BASE / "data/scopes.json")[scope]
    positions = {v["source_id"]: v["position"] for v in scopes}
    local = {v["source_id"]: i for i, v in enumerate(scopes)}
    return ep, kept, ids, y, positions, local


def model_arrays(scope):
    models = {}
    full = c.read(r.BASE / "data/full_heads.json")
    labels = {
        h: full[h]["labels"] if h != "hate" else ["not_hate", "hate"] for h in c.HEADS
    }
    for path in sorted((r.BASE / "pipelines").glob("*/complete.json")):
        _, attempt = r.receipt(path.parent)
        contract = c.read(attempt / "contract.json")
        data = c.load_arrays(attempt / "decisions.npz")
        probs = {
            h: np.load(c.ROOT / name, allow_pickle=False)
            for h, name in contract["probabilities"].items()
        }
        ident = contract["identity"]
        stage = ident["stage"]
        seed = ident["seed"]
        variant = ident["variant"]
        root = (
            ("full_bge" if stage == "full_bge" else "full_tfidf")
            + ("_matched" if variant == "matched_gradient" else "_refit")
            if stage != "compact_tfidf"
            else "compact_tfidf"
        )
        for policy in ("selected", "default"):
            models[root + "_" + policy, seed] = dict(
                kind="linear",
                decisions={h: data[policy + "_" + h].astype(bool) for h in c.HEADS},
                ungated={
                    h: data[
                        policy + "_" + h + ("_ungated" if h != "hate" else "")
                    ].astype(bool)
                    for h in c.HEADS
                },
                probabilities=probs,
                labels=labels,
                cutoffs=(
                    contract["selected_cutoffs"]
                    if policy == "selected"
                    else {h: None if h == "severity" else 0.5 for h in c.HEADS}
                ),
            )
    choices = c.read(r.OUT / "neural_audit.json")["choices"]
    for seed in range(3):
        data = c.load_arrays(r.OUT / f"neural/{scope}_s{seed}_decisions.npz")
        prob = c.load_arrays(r.OUT / f"neural/{scope}_s{seed}_probabilities.npz")
        for policy in ("selected", "default"):
            models["full_neural_" + policy, seed] = dict(
                kind="neural",
                decisions={h: data[policy + "_" + h].astype(bool) for h in c.HEADS},
                ungated={
                    h: data[
                        policy + "_" + h + ("_ungated" if h != "hate" else "")
                    ].astype(bool)
                    for h in c.HEADS
                },
                probabilities=prob,
                labels=labels,
                cutoffs={
                    h: (
                        choices[f"{h}_s{seed}"]["selected"]["threshold"]
                        if policy == "selected"
                        else (None if h == "severity" else 0.5)
                    )
                    for h in c.HEADS
                },
            )
    return models


def summary(y, x, ep):
    stats, counts = previous.summarize_bits(y, x, ep)
    stats.update(
        empty_prediction_rate=float((~x.any(1)).mean()) if len(y) else None,
        mean_extra_labels=float((x & ~y).sum(1).mean()) if len(y) else None,
        mean_missing_labels=float((~x & y).sum(1).mean()) if len(y) else None,
    )
    return stats, counts


def scores():
    c.require(
        c.read(r.OUT / "linear_audit.json")["status"] == "pass"
        and c.read(r.OUT / "neural_audit.json")["status"] == "pass",
        "Independent model audits required",
    )
    cells = {}
    slices = {}
    diagnostics = {}
    cache = {}
    historical = 0
    old = c.read(earlier.OUT / "scores.json")["cells"]
    coretexts = {
        previous.norm(row["text"])
        for row in previous.rows(previous.CORE / "dev_inputs.jsonl")
        if row["lang"] == "en"
    }
    inputs = c.read(r.BASE / "data/evaluation.json")
    scope_rows = c.read(r.BASE / "data/scopes.json")
    prepared = c.read(r.BASE / "plan.json")["inputs"]
    gold_bindings = {}
    for scope in ("core", "broad"):
        models = model_arrays(scope)
        eps = c.read(
            previous.BGE / "plan.json"
            if scope == "core"
            else previous.BROAD / "plan.json"
        )["endpoints"]
        for name in eps:
            ep, kept, ids, y, positions, local = endpoint_context(scope, name)
            head = ep["dimension"]
            macro = [ep["all_labels"].index(v) for v in ep["macro_labels"]]
            goldpath = (
                c.ROOT / f"data/processed/{name}.parquet"
                if scope == "core"
                else previous.PREP / "broad_roster.jsonl"
            )

            epplan = c.read(
                previous.BGE / "plan.json"
                if scope == "core"
                else previous.BROAD / "plan.json"
            )
            expected = epplan["inputs"][c.rel(goldpath)]
            c.require(c.sha(goldpath) == expected, "Frozen gold changed")
            gold_bindings[c.rel(goldpath)] = expected
            c.require(
                all(
                    inputs[positions[k]]["text"] == row["text"]
                    for k, row in zip(ids, kept)
                ),
                "Gold/prediction text differs",
            )
            runs = defaultdict(list)
            groups = {
                f"source:{s}": np.asarray([v["source"] == s for v in kept])
                for s in sorted({v["source"] for v in kept})
            }
            if scope == "broad":
                groups["outside_original_core"] = np.asarray(
                    [previous.norm(v["text"]) not in coretexts for v in kept]
                )
            for (method, seed), model in models.items():
                ix = [
                    (positions if model["kind"] == "linear" else local)[k] for k in ids
                ]
                order = [model["labels"][head].index(v) for v in ep["all_labels"]]
                x = model["decisions"][head][ix][:, order]
                u = model["ungated"][head][ix][:, order]
                gate = model["decisions"]["hate"][ix, 1]
                prob = model["probabilities"][head][ix]
                if head == "hate":
                    prob = np.column_stack((1 - prob[:, 0], prob[:, 0]))
                prob = prob[:, order]
                stat, count = summary(y, x, ep)
                stat["seed"] = seed
                runs[method].append(stat)
                if method == "full_neural_default":
                    expected = next(
                        v
                        for v in old[scope + "/" + name]["runs"]["encoder_full"]
                        if v["seed"] == seed
                    )
                    c.require(
                        all(stat[k] == v for k, v in expected.items()),
                        "Historical full-neural score differs",
                    )
                    historical += 1
                slices[f"{scope}/{name}/{method}/{seed}"] = {
                    g: summary(y[mask], x[mask], ep)[0]
                    for g, mask in groups.items()
                    if mask.any()
                }
                rank = {}
                for j, label in enumerate(ep["all_labels"]):
                    yy = y[:, j]
                    pp = prob[:, j]
                    rank[label] = dict(
                        average_precision=(
                            float(average_precision_score(yy, pp)) if yy.any() else None
                        ),
                        auroc=(
                            float(roc_auc_score(yy, pp))
                            if yy.any() and (~yy).any()
                            else None
                        ),
                        positives_blocked_by_gate=(
                            int((yy & ~gate).sum()) if head != "hate" else None
                        ),
                    )
                diagnostics[f"{scope}/{name}/{method}/{seed}"] = dict(
                    ungated=summary(y, u, ep)[0],
                    gate_negative_rate=float((~gate).mean()),
                    labels=rank,
                    macro_average_precision_ungated=(
                        float(
                            np.mean(
                                [
                                    rank[ep["all_labels"][j]]["average_precision"]
                                    for j in macro
                                ]
                            )
                        )
                        if all(
                            rank[ep["all_labels"][j]]["average_precision"] is not None
                            for j in macro
                        )
                        else None
                    ),
                )
                if scope == "broad":
                    cache[f"{name}|{method}|{seed}"] = count

            references = {
                method: old[scope + "/" + name]["runs"][method]
                for method in ["bge", "encoder_kb", "tuned_selected"]
            }
            cells[scope + "/" + name] = dict(
                n=len(ids),
                endpoint={k: v for k, v in ep.items() if k != "scorable_ids"},
                runs=dict(runs),
                means={
                    k: float(np.mean([v["macro_f1"] for v in vs]))
                    for k, vs in runs.items()
                },
                sample_sds={
                    k: (
                        float(np.std([v["macro_f1"] for v in vs], ddof=1))
                        if len(vs) > 1
                        else None
                    )
                    for k, vs in runs.items()
                },
                earlier_context=references,
            )
            print(
                scope,
                name,
                {
                    m: round(cells[scope + "/" + name]["means"][m], 6)
                    for m in [
                        "full_bge_matched_selected",
                        "full_tfidf_matched_selected",
                        "full_neural_selected",
                        "full_neural_default",
                        "compact_tfidf_selected",
                    ]
                },
                flush=True,
            )
    r.save_arrays("paired_counts.npz", cache)
    r.save(
        "scores.json",
        dict(
            status="pass",
            cells=cells,
            score_cells=sum(
                len(vs) for cell in cells.values() for vs in cell["runs"].values()
            ),
            historical_full_neural_cells_reproduced=historical,
            gold_bindings=gold_bindings,
            source_sha256=c.sha(Path(__file__)),
            Qwen_calls=0,
        ),
    )
    r.save(
        "slices.json",
        dict(
            status="pass",
            exploratory=True,
            cells=slices,
            fixed_endpoint_macro_labels=True,
        ),
    )
    r.save(
        "diagnostics.json",
        dict(
            status="pass",
            exploratory=True,
            cells=diagnostics,
            caveat="Ungated head outputs and ranks use the same gold-hate eligibility masks. They are diagnostics, not an approved replacement gate or new model selection.",
        ),
    )


def bootstrap(
    counts,
    clusters,
    sources,
    indices,
    left=1,
    draws=10000,
    stratified=False,
    seed=20261012,
):
    unique = sorted(set(clusters))
    mapping = {k: i for i, k in enumerate(unique)}
    aggregate = np.zeros(
        (len(unique), counts.shape[0], counts.shape[2], 3), dtype=np.int64
    )
    members = defaultdict(set)
    for i, key in enumerate(clusters):
        aggregate[mapping[key]] += counts[:, i]
        members[key].add(sources[i])
    strata = defaultdict(list)
    for key, j in mapping.items():
        strata[tuple(sorted(members[key])) if stratified else ("all",)].append(j)
    rng = np.random.default_rng(seed)
    deltas = []
    flat = aggregate.reshape(len(unique), -1).astype(float)
    for start in range(0, draws, 64):
        batch = min(64, draws - start)
        picks = np.concatenate(
            [
                rng.choice(v, (batch, len(v)), replace=True)
                for _, v in sorted(strata.items())
            ],
            axis=1,
        )
        weights = np.array(
            [np.bincount(p, minlength=len(unique)) for p in picks], dtype=float
        )
        total = (weights @ flat).reshape(batch, counts.shape[0], counts.shape[2], 3)
        c.require(
            np.array_equal(total[0], aggregate[picks[0]].sum(0)),
            "Cluster aggregation differs from direct draw",
        )
        tp, fp, fn = total[..., 0], total[..., 1], total[..., 2]
        den = 2 * tp + fp + fn
        f = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den != 0)[
            ..., indices
        ].mean(-1)
        deltas.extend((f[:, :left].mean(1) - f[:, left:].mean(1)).tolist())
    total = counts.sum(1)
    tp, fp, fn = total[..., 0], total[..., 1], total[..., 2]
    den = 2 * tp + fp + fn
    f = np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0)[
        :, indices
    ].mean(-1)
    return dict(
        left=float(f[:left].mean()),
        right=float(f[left:].mean()),
        difference=float(f[:left].mean() - f[left:].mean()),
        ci9875=np.quantile(deltas, [0.00625, 0.99375]).tolist(),
        ci95=np.quantile(deltas, [0.025, 0.975]).tolist(),
        rows=len(clusters),
        clusters=len(unique),
        draws=draws,
        seed=seed,
        source_stratified=stratified,
        left_fixed_fits=left,
        right_fixed_fits=len(counts) - left,
    )


def statistics():
    c.require(c.read(r.OUT / "scores.json")["status"] == "pass", "Scores required")
    primary = {}
    supporting = {}
    with np.load(r.OUT / "paired_counts.npz", allow_pickle=False) as saved:
        for name in r.PRIMARY:
            ep, kept, ids, y, _, _ = endpoint_context("broad", name)
            ix = [ep["all_labels"].index(v) for v in ep["macro_labels"]]
            counts = np.stack(
                [saved[f"{name}|full_bge_matched_selected|42"]]
                + [saved[f"{name}|full_neural_selected|{s}"] for s in range(3)]
            )
            clusters = [v["text_cluster"] for v in kept]
            sources = [v["source"] for v in kept]
            primary[name] = dict(
                primary=bootstrap(counts, clusters, sources, ix),
                source_stratified=bootstrap(
                    counts, clusters, sources, ix, stratified=True
                ),
            )
            print("PRIMARY", name, primary[name]["primary"], flush=True)

        for name in ["en_hatecheck_eval/hate", "en_hatecheck_eval/target_group"]:
            ep, kept, ids, y, _, _ = endpoint_context("broad", name)
            ix = [ep["all_labels"].index(v) for v in ep["macro_labels"]]
            counts = np.stack(
                [saved[f"{name}|full_bge_matched_selected|42"]]
                + [saved[f"{name}|full_neural_selected|{s}"] for s in range(3)]
            )
            supporting[name] = dict(
                functional=bootstrap(
                    counts,
                    [v["functional_cluster"] for v in kept],
                    [v["source"] for v in kept],
                    ix,
                ),
                item_sensitivity=bootstrap(
                    counts, ids, [v["source"] for v in kept], ix
                ),
            )
    r.save(
        "statistics.json",
        dict(
            status="pass",
            family=r.PRIMARY,
            confidence=0.9875,
            primary=primary,
            exploratory_functional=supporting,
            warning="Conditional on these fitted models and sampled text/template clusters. Three neural seeds are kept separate; the deterministic BGE fit is not duplicated. Prior exposure remains. Supporting comparisons use 95% intervals.",
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["neural", "scores", "statistics"])
    args = parser.parse_args()
    with c.offline(), threadpool_limits(limits=1):
        globals()[args.mode]()


if __name__ == "__main__":
    main()
