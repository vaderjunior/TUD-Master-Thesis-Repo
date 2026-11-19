import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import numpy as np
from analysis.feedback.transfer import contexts as p
from analysis.feedback.transfer import contexts_metrics as k

a = p.a
OUT = p.OUT
VIEWS = (
    "main",
    "strict",
    "replay_excluded",
    "strict_replay_excluded",
    "representative",
    "representative_replay_excluded",
)


def quantiles(values):
    x = np.asarray(values, dtype=float)
    return dict(
        n=len(x),
        mean=float(x.mean()) if len(x) else None,
        quantiles=np.quantile(x, [0, 0.25, 0.5, 0.75, 1]).tolist() if len(x) else None,
    )


def arrays(rows, n):
    out = {
        name: np.zeros((n, 3), dtype=dtype)
        for name, dtype in [
            ("source_gate", bool),
            ("removed_gate", bool),
            ("cosine_bin", int),
            ("removed_any_fine", bool),
            ("reference_hateful_examples", int),
        ]
    }
    out["similarity_relation"] = np.empty((n, 3), dtype="U6")
    for r in rows:
        i, c = r["input_index"], r["pool_index"] - 1
        for key in out:
            out[key][i, c] = (
                any(r["removed_known_fine"].values())
                if key == "removed_any_fine"
                else r[key]
            )
    return out


def strata(x):
    shape = x["source_gate"].shape
    out = {"all": np.ones(shape, dtype=bool)}
    out.update({f"correction/{int(g)}": x["source_gate"] == g for g in (False, True)})
    out.update(
        {
            f"swap/{int(old)}-{int(new)}": (x["removed_gate"] == old)
            & (x["source_gate"] == new)
            for old in (False, True)
            for new in (False, True)
        }
    )
    out.update({f"cosine/{j}": x["cosine_bin"] == j for j in range(5)})
    out.update(
        {
            f"cosine_vs_fifth/{value}": x["similarity_relation"] == value
            for value in ("lower", "tie", "higher")
        }
    )
    out.update(
        {f"removed_fine/{int(g)}": x["removed_any_fine"] == g for g in (False, True)}
    )
    out.update(
        {
            f"baseline_hateful/{j}": x["reference_hateful_examples"] == j
            for j in range(6)
        }
    )
    return out


def group_rate(ids, mask, errors, meta):
    ids = np.asarray(ids, dtype=int)
    take = mask[ids]
    ref = errors[ids, 0]
    trt = errors[ids, 1:]
    n = take.sum(axis=1)
    features = np.stack(
        [n, ref.sum(axis=1) * n, (trt.sum(axis=2) * take).sum(axis=1)], axis=1
    )
    used = n > 0
    features = features[used]
    texts = ids[used]
    r = k.point(features.sum(axis=0))
    r.update(
        unique_texts=len(texts),
        clusters=len({meta[i]["inference_cluster"] for i in texts}),
        reference_errors_by_repeat=(ref * n[:, None]).sum(axis=0),
        treatment_errors_by_repeat=(trt * take[:, :, None]).sum(axis=(0, 1)),
        per_pool=[
            dict(
                k.point(
                    [
                        int(take[:, c].sum()),
                        int((ref * take[:, c, None]).sum()),
                        int((trt[:, c] * take[:, c, None]).sum()),
                    ]
                ),
                reference_errors_by_repeat=(ref * take[:, c, None]).sum(axis=0),
                treatment_errors_by_repeat=(trt[:, c] * take[:, c, None]).sum(axis=0),
            )
            for c in range(3)
        ],
    )
    return r, features, texts


def main():
    done = a.read(OUT / "assembled.json")
    for name, digest in {
        **done["input_bindings"],
        **done["analysis_sources"],
        **done["artifacts"],
    }.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed context evidence " + name)
    a.need(not (OUT / "calculated.json").exists(), "Preserve previous context analysis")
    rows = list(a.rows(OUT / "delivery.jsonl"))
    contexts = [
        {key: value for key, value in r.items() if key not in ("ordinary_hits", "text")}
        for r in a.rows(OUT / "query_contexts.jsonl")
    ]
    meta = list(a.rows(p.P4 / "metadata.jsonl"))
    matrix = list(a.rows(a.OUT / "matrix.jsonl"))
    n = len(meta)
    catalog = a.read(OUT / "source_catalog.json")
    pools = a.read(OUT / "pool_inventory.json")["pools"]
    with np.load(p.P4 / "prediction_codes.npz", allow_pickle=False) as f:
        gates = f["gates"]
    a.need(len(rows) == n * 3 and len(contexts) == n, "Context matrix sizes")
    x = arrays(rows, n)
    groups = strata(x)
    exposure = {}
    grouped = defaultdict(list)
    for r in rows:
        i = r["input_index"]
        m = meta[i]
        dimensions = [
            ("population", r["population"]),
            ("source", r["source"]),
            ("gold", r["population"] + "/" + str(int(r["gold_hate"]))),
        ]
        if r["functionality"] is not None:
            dimensions.append(("function", r["functionality"]))
        for view in ("full", "common", "strict"):
            if view == "common" and not matrix[i]["masks"]["common_valid"]:
                continue
            if view == "strict" and not matrix[i]["masks"]["strict_all_votes"]:
                continue
            for kind, value in dimensions:
                grouped[
                    view + "/" + kind + "/" + value + "/" + str(r["pool_index"])
                ].append(r)
    for name, rr in sorted(grouped.items()):
        pool = pools[rr[0]["pool_index"] - 1]
        counts = Counter(r["source_id"] for r in rr)
        N = len(rr)
        exposure[name] = dict(
            assignments=N,
            unique_texts=len({r["key"] for r in rr}),
            inference_clusters=len({r["inference_cluster"] for r in rr}),
            pool=pool["condition"],
            source_counts={sid: counts[sid] for sid in pool["ids"]},
            source_concentration=k.concentration([counts[sid] for sid in pool["ids"]]),
            inserted_hateful=sum(r["source_gate"] for r in rr),
            removed_hateful=sum(r["removed_gate"] for r in rr),
            swap_counts=dict(
                Counter(f"{int(r['removed_gate'])}-{int(r['source_gate'])}" for r in rr)
            ),
            hateful_example_change=dict(
                Counter(str(r["hateful_count_change"]) for r in rr)
            ),
            cosine=quantiles([r["cosine"] for r in rr]),
            fifth_cosine=quantiles([r["removed_cosine"] for r in rr]),
            cosine_difference=quantiles([r["cosine_difference"] for r in rr]),
            similarity_relation=dict(Counter(r["similarity_relation"] for r in rr)),
            cosine_bins={
                str(i): sum(r["cosine_bin"] == i for r in rr) for i in range(5)
            },
            dense_winner_selected=sum(r["dense_winner_same"] for r in rr),
            retrieval_channels=dict(Counter(r["retrieval_channels"] for r in rr)),
            dense_cosine_regret=quantiles([r["dense_cosine_regret"] for r in rr]),
            token_difference=quantiles([r["token_difference"] for r in rr]),
            removed_source=dict(Counter(r["removed_source"] for r in rr)),
            removed_known_fields={
                f: sum(r["removed_known_fine"][f] for r in rr) for f in k.FIELDS
            },
            removed_last_known_field={
                f: sum(
                    r["removed_known_fine"][f]
                    and r["remaining_known_field_counts"][f] == 0
                    for r in rr
                )
                for f in k.FIELDS
            },
        )
    shared = {}
    for pop in ("en_test", "en_hatecheck_eval"):
        for view in ("full", "common", "strict"):
            ids = [
                i
                for i, m in enumerate(meta)
                if m["population"] == pop
                and (
                    view == "full"
                    or matrix[i]["masks"][
                        "common_valid" if view == "common" else "strict_all_votes"
                    ]
                )
            ]
            triples = [[rows[i * 3 + c] for c in range(3)] for i in ids]
            shared[pop + "/" + view] = dict(
                texts=len(ids),
                unique_source_choices_histogram=dict(
                    Counter(str(len({r["source_id"] for r in rr})) for rr in triples)
                ),
                unique_feedback_message_histogram=dict(
                    Counter(
                        str(len({r["messages_sha256"] for r in rr})) for rr in triples
                    )
                ),
                same_source_pairs={
                    f"{i+1}-{j+1}": sum(
                        rr[i]["source_id"] == rr[j]["source_id"] for rr in triples
                    )
                    for i in range(3)
                    for j in range(i + 1, 3)
                },
                same_message_pairs={
                    f"{i+1}-{j+1}": sum(
                        rr[i]["messages_sha256"] == rr[j]["messages_sha256"]
                        for rr in triples
                    )
                    for i in range(3)
                    for j in range(i + 1, 3)
                },
                correction_labels_agree=sum(
                    len({r["source_gate"] for r in rr}) == 1 for rr in triples
                ),
                unique_feedback_messages=len(
                    {r["messages_sha256"] for rr in triples for r in rr}
                ),
            )
            a.need(
                all(
                    rr[i]["messages_sha256"] == rr[j]["messages_sha256"]
                    for rr in triples
                    for i in range(3)
                    for j in range(i + 1, 3)
                    if rr[i]["source_id"] == rr[j]["source_id"]
                ),
                "Identical inserted sources produce different messages",
            )
    p.save("exposure.json", exposure)
    p.save("shared_exposure.json", shared)
    errors = (
        gates != np.asarray([m["gold_hate"] for m in meta])[:, None, None]
    ).astype(np.int8)
    prior = a.read(
        a.ROOT / "experiments/astra/sq3_transfer_part2_20261023_v1/rates.json"
    )
    outcomes = {}
    draw_count = 0

    def evaluate(name, ids, mask, baseN, boot=False, reuse=None):
        nonlocal draw_count
        result, features, texts = group_rate(ids, mask, errors, meta)
        result.update(
            id=name,
            base_common_texts=baseN,
            contribution_to_group_difference=(
                (result["treatment_errors"] - result["reference_errors"]) / (9 * baseN)
                if baseN
                else None
            ),
            weighting="Each selected query-pool assignment has equal weight; reference repeats reused across pool comparisons. Pool-specific rates also retained.",
            status="defined" if result["assignments"] else "undefined_no_assignments",
        )
        if boot and result["assignments"]:
            seed = int.from_bytes(
                hashlib.sha256(("A16-part5-context-v1|" + name).encode()).digest()[:8],
                "little",
            )
            clusters = [meta[i]["inference_cluster"] for i in texts]
            draws = k.boot(features, clusters, seed)
            path = (
                OUT / "draws" / f"{hashlib.sha256(name.encode()).hexdigest()[:24]}.npy"
            )
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as f:
                np.save(f, draws, allow_pickle=False)
            result.update(
                interval95=np.quantile(draws[:, 2], [0.025, 0.975]).tolist(),
                bootstrap=dict(
                    seed=seed,
                    draws=10000,
                    unit="whole frozen inference cluster",
                    units=len(set(clusters)),
                    generator="PCG64",
                    block_size=32,
                ),
                draw_file=a.rel(path),
                text_indices=texts.tolist(),
                per_text_counts=features.tolist(),
            )
            draw_count += 1
        if reuse is not None:
            q = prior[reuse]
            a.need(
                abs(result["reference"] - q["reference_rate"]) < 1e-14
                and abs(result["treatment"] - q["treatment_rate"]) < 1e-14,
                "Part 2 main rate reconciliation",
            )
            result.update(interval95=q["interval95"], reused_part2=repr(reuse))
        outcomes[name] = p.native(result)

    for pop in ("en_test", "en_hatecheck_eval"):
        for endpoint in ("FPR", "FNR"):
            ep = pop + "_" + endpoint
            for view in VIEWS:
                q = prior[view + "/" + ep]
                ids = q["common_indices"]
                for group, mask in groups.items():
                    name = f"population/{ep}/{view}/{group}"
                    evaluate(
                        name,
                        ids,
                        mask,
                        len(ids),
                        boot=view == "main"
                        and group.startswith(("correction/", "swap/")),
                        reuse=view + "/" + ep if group == "all" else None,
                    )
            print("Context outcome strata: " + ep, flush=True)
    for source in ("hatexplain", "implicit_hate", "mhs"):
        for endpoint in ("FPR", "FNR"):
            for view in ("main", "strict"):
                ids = [
                    i
                    for i in prior[view + "/en_test_" + endpoint]["common_indices"]
                    if meta[i]["source"] == source
                ]
                for group, mask in groups.items():
                    evaluate(
                        f"source/{source}/{endpoint}/{view}/{group}",
                        ids,
                        mask,
                        len(ids),
                    )
    for function in sorted(
        {m["functionality"] for m in meta if m["functionality"] is not None}
    ):
        for endpoint in ("FPR", "FNR"):
            ids = [
                i
                for i in prior["main/en_hatecheck_eval_" + endpoint]["common_indices"]
                if meta[i]["functionality"] == function
            ]
            for group, mask in groups.items():
                if group == "all" or group.startswith(("correction/", "swap/")):
                    evaluate(
                        f"function/{function}/{endpoint}/main/{group}",
                        ids,
                        mask,
                        len(ids),
                    )

    quality = a.read(p.P4 / "quality_results.json")
    fine = {}
    with np.load(p.P4 / "features.npz", allow_pickle=False) as f:
        features = {h: f[h] for h in ("hate_type", "target_group", "severity")}
    schema = a.read(p.P4 / "feature_schema.json")
    field = {
        "hate_type": "hate_types",
        "target_group": "target_groups",
        "severity": "severity",
    }
    for pop in ("en_test", "en_hatecheck_eval"):
        for head in field:
            for view in VIEWS:
                q = quality[f"population/{pop}/{head}/{view}"]
                if not q["common_n"]:
                    continue
                ids = q["common_indices"]
                cols = schema["features"][head]
                for measure in ("exact", "gated_exact"):
                    failure = 1 - features[head][..., cols.index(measure)]
                    for kind in ("removed_known", "removed_last_known"):
                        yes = np.asarray(
                            [
                                r["removed_known_fine"][field[head]]
                                and (
                                    kind != "removed_last_known"
                                    or r["remaining_known_field_counts"][field[head]]
                                    == 0
                                )
                                for r in rows
                            ]
                        ).reshape(n, 3)
                        for flag in (False, True):
                            z, _, _ = group_rate(ids, yes == flag, failure, meta)
                            name = f"{pop}/{head}/{view}/{measure}/{kind}/{int(flag)}"

                            z.update(
                                id=name,
                                reference_exact=(
                                    1 - z["reference"]
                                    if z["reference"] is not None
                                    else None
                                ),
                                treatment_exact=(
                                    1 - z["treatment"]
                                    if z["treatment"] is not None
                                    else None
                                ),
                                exact_difference=(
                                    -z["difference"]
                                    if z["difference"] is not None
                                    else None
                                ),
                            )
                            fine[name] = p.native(z)
    p.save("conditional_rates.json", outcomes)
    p.save("fine_information_associations.json", fine)
    files = [
        OUT / name
        for name in (
            "exposure.json",
            "shared_exposure.json",
            "conditional_rates.json",
            "fine_information_associations.json",
        )
    ] + sorted((OUT / "draws").glob("*.npy"))
    p.save(
        "calculated.json",
        dict(
            status="part5_exposure_and_associations_calculated_independent_verification_pending",
            created_utc=a.utc(),
            assembled_sha256=a.sha(OUT / "assembled.json"),
            source_sha256=a.sha(Path(__file__)),
            artifacts={a.rel(path): a.sha(path) for path in files},
            exposure_views=len(exposure),
            conditional_rate_views=len(outcomes),
            fine_information_views=len(fine),
            new_bootstrap_sets=draw_count,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        f"Calculated {len(exposure)} exposure views, {len(outcomes)} rate views, {len(fine)} fine-information views and {draw_count} draw sets.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()
