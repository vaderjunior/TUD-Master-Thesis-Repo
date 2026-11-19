import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from analysis.classification.combined import metrics as p

a = p.a
ROOT = a.ROOT
BASE = ROOT / "experiments/astra/sq2_combined_quality_mechanisms_20261104_v1"
BQ = ROOT / "experiments/astra/sq2_broad_quality_review_20261104_v1"
CTX = ROOT / "experiments/astra/sq2_broad_context_review_20261104_v1"
ROB = ROOT / "experiments/astra/sq2_broad_robustness_review_20261104_v1"
GQ = ROOT / "experiments/astra/sq2_gpu_quality_review_20261103_v1/runs/5047fbc65599"
GC = ROOT / "experiments/astra/sq2_gpu_controls_review_20261103_v1/runs/b32f91704667"
GE = ROOT / "experiments/astra/sq2_gpu_errors_review_20261103_v1/runs/54014ea4bfd9"


def endpoints():
    return [
        x for x in a.read(a.BROAD / "endpoint_setup.json") if not x.endswith("/hate")
    ]


def save(name, v):
    f = BASE / name
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("xb") as h:
        h.write(a.g.encoded(v))


def mean_available(values):
    known = [v for v in values if v is not None]
    return float(np.mean(known)) if known else None


def aggregate(models, runs):
    out = {}
    for group in sorted({m["group"] for m in models}):
        idx = [i for i, m in enumerate(models) if m["group"] == group]
        pools = defaultdict(list)
        for i in idx:
            pools[models[i]["pool"]].append(i)
        pm = {
            name: {
                k: mean_available([runs[i]["metrics"][k] for i in ii])
                for k in runs[ii[0]]["metrics"]
            }
            for name, ii in pools.items()
        }
        out[group] = dict(
            mean={
                k: mean_available([v[k] for v in pm.values()])
                for k in next(iter(pm.values()))
            },
            pools=pm,
            model_indices=idx,
            pool_count=len(pools),
            fit_count=len(idx),
            n_per_fit=[runs[i]["n"] for i in idx],
            defined_fits={
                k: sum(runs[i]["metrics"][k] is not None for i in idx)
                for k in runs[idx[0]]["metrics"]
            },
        )
    return out


def row_quality(y, pr, gate, legal=False):
    y = np.asarray(y, dtype=bool)
    pr = np.asarray(pr, dtype=bool)
    gate = np.asarray(gate, dtype=bool)
    a.require(
        y.shape == pr.shape and gate.shape == (len(y),), "Quality dimensions differ"
    )
    if not legal:
        a.require(
            not np.any(pr[~gate]),
            "Negative hate gate has a nonempty delivered fine field",
        )
    tp = (y & pr).sum(1)
    ex = (~y & pr).sum(1)
    miss = (y & ~pr).sum(1)
    gy = y.sum(1)
    py = pr.sum(1)
    exact = (ex + miss) == 0
    union = tp + ex + miss
    return dict(
        exact_set=exact,
        gate_aware_exact=exact if legal else exact & gate,
        sample_jaccard=np.divide(tp, union, out=np.ones(len(y)), where=union != 0),
        sample_f1=np.divide(2 * tp, gy + py, out=np.ones(len(y)), where=(gy + py) != 0),
        extra_only=(ex > 0) & (miss == 0),
        missing_only=(ex == 0) & (miss > 0),
        mixed=(ex > 0) & (miss > 0),
        has_extra=ex > 0,
        has_missing=miss > 0,
        extra_labels=ex,
        missing_labels=miss,
        true_positive_labels=tp,
        gold_labels=gy,
        predicted_labels=py,
        predicted_empty=py == 0,
        empty_gold=gy == 0,
        gate_pass=gate,
        empty_exact_wrong_gate=(
            (gy == 0) & exact & ~gate if not legal else np.zeros(len(y), bool)
        ),
        positive_gold_blocked=(gy > 0) & ~gate if not legal else np.zeros(len(y), bool),
        correct_cardinality=gy == py,
        cardinality_matches_but_wrong=(gy == py) & ~exact,
        hamming_error=(ex + miss) / y.shape[1],
    )


def summarize(y, pr, gate, legal=False):
    r = row_quality(y, pr, gate, legal)
    n = len(y)
    metrics = {k: float(v.mean()) if n else None for k, v in r.items()}

    metrics["exact_given_gate"] = (
        float(r["exact_set"][gate.astype(bool)].mean()) if np.any(gate) else None
    )
    counts = {k: float(v.sum()) for k, v in r.items()}
    card = np.zeros((y.shape[1] + 1, y.shape[1] + 1), dtype=int)
    np.add.at(card, (y.sum(1).astype(int), pr.sum(1).astype(int)), 1)
    return dict(n=n, metrics=metrics, counts=counts, cardinality=card.tolist())


def joint_rows(t, s, gate):
    return dict(
        joint_exact=gate & t & s,
        gate_closed=~gate,
        target_only_wrong=gate & ~t & s,
        severity_only_wrong=gate & t & ~s,
        both_wrong=gate & ~t & ~s,
    )


def flow_rows(y, raw, gate):
    y = y.astype(bool)
    raw = raw.astype(bool)
    gate = gate.astype(bool)
    g = gate[:, None]
    exact = np.all(y == raw, axis=1)
    return dict(
        raw_missing_gate_open=(y & ~raw & g).sum(1),
        raw_missing_gate_closed=(y & ~raw & ~g).sum(1),
        correct_labels_blocked=(y & raw & ~g).sum(1),
        extra_labels_blocked=(~y & raw & ~g).sum(1),
        raw_exact=exact,
        correct_raw_set_blocked=exact & ~gate,
        wrong_raw_set_passed=~exact & gate,
        raw_and_gate_correct=exact & gate,
    )


def get(endpoint):
    slug = endpoint.replace("/", "__")
    folder = p.BASE / "endpoints" / slug
    md = a.read(folder / "metadata.json")
    z = p.load(folder / "aligned.npz")
    q = a.read(BQ / "endpoints" / slug / "results.json")
    old = p.load(ROOT / q["encoded_path"])
    k = len(q["settings"])
    a.require(
        md["ids"] == q["ids"] and md["labels"] == q["labels"],
        "Prior quality identity mismatch",
    )
    for newkey, oldkey in [
        ("gold", "gold"),
        ("predictions", "predicted"),
        ("valid", "valid"),
    ]:
        v = z[newkey] if newkey == "gold" else z[newkey][:k]
        a.require(np.array_equal(v, old[oldkey]), "Earlier quality decisions differ")
    a.require(
        np.array_equal(z["common"], old["common_nine"]),
        "Prior quality common mask differs",
    )
    restored = []
    for j, (co, seed) in enumerate(q["settings"]):
        m = md["models"][j]
        a.require(
            m["original_method"] == co and m["seed"] == seed,
            "Historical view ordering differs",
        )
        if m["gate_available"]:
            a.require(
                np.array_equal(
                    z["gate"][j, z["valid"][j]], old["gate"][j, z["valid"][j]]
                ),
                "Historical saved gate disagreement",
            )
        else:
            restored.append(j)
    z["gate"][:k] = old["gate"]
    return md, z, restored


def sources():
    files = [
        Path(__file__),
        Path(__file__).with_name("astra_sq2_combined_quality_tests_v1.py"),
        p.BASE / "part2_complete.json",
        p.BASE / "analysis_contract.json",
        a.BASE / "part1_complete.json",
        a.BASE / "training_information.json",
        a.BASE / "evaluation_alignment.json",
        a.PLAN,
        a.REPORTS / "astra_remaining_work_before_writing_20261104_v154.md",
        a.GPU / "models.json",
        a.ROSTER,
    ]
    for e in endpoints():
        slug = e.replace("/", "__")
        files += [
            p.BASE / "endpoints" / slug / n
            for n in ("metadata.json", "aligned.npz", "scores.json")
        ]
        files += [BQ / "endpoints" / slug / n for n in ("results.json", "encoded.npz")]
        if (a.GPU / (slug + "_arrays.npz")).exists():
            files += [a.GPU / (slug + "_arrays.npz"), a.GPU / (slug + "_metadata.json")]

    for folder in (GQ, GC, GE):
        files.extend(sorted(folder.glob("*.json")))
    for folder, names in [
        (
            BQ,
            [
                "part4_complete.json",
                "joint_results.json",
                "normalization_diagnostic.json",
                "additional_summaries.json",
                "annotation_inventory.json",
            ],
        ),
        (
            CTX,
            [
                "part6_complete.json",
                "channel_inventory.json",
                "delivery_summary.json",
                "overlap_summary.json",
                "derived_findings.json",
                "outcome_associations.json",
                "case_contexts.json",
                "manual_review.json",
                "context_reading_inventory.json",
            ],
        ),
        (
            ROB,
            [
                "part7_complete.json",
                "derived_findings.json",
                "repeat_and_inner_diagnostics.json",
            ],
        ),
        (
            a.MATCHED,
            [
                "complete.json",
                "metrics.json",
                "diagnostics.json",
                "quality_intervals.json",
                "context_review.json",
                "prompt_checks.json",
            ],
        ),
        (p.SUPP, ["complete.json", "main_comparison.json"]),
        (p.CPUREVIEW, ["scores.json"]),
        (p.SHORTREVIEW, ["scores.json"]),
    ]:
        files += [folder / n for n in names]
    names = [
        "astra_sq2_broad_answer_quality_20261104_v1.md",
        "astra_sq2_broad_retrieval_contexts_20261104_v2.md",
        "astra_sq2_broad_robustness_20261104_v2.md",
        "astra_sq2_gpu_training_cutoffs_20261103_v1.md",
        "astra_sq2_gpu_errors_and_gate_20261103_v1.md",
        "astra_sq2_gpu_fine_answer_quality_20261103_v1.md",
        "astra_sq2_gpu_robustness_20261104_v1.md",
        "astra_sq2_matched_results_20261030_v1.md",
        "astra_sq2_gpu_final_synthesis_20261104_v1.md",
        "astra_sq2_combined_performance_errors_20261104_v2.md",
        "astra_sq2_combined_comparison_setup_20261104_v1.md",
    ]
    files += [a.REPORTS / n for n in names if (a.REPORTS / n).exists()]
    return sorted(set(files))


def freeze():
    from tests.classification.combined.test_quality import run_tests

    tests = run_tests()
    seal = a.read(p.BASE / "part2_complete.json")
    a.verify(seal["outputs"])
    a.verify(seal["original_sources"])
    checked = []
    for f in (
        BQ / "part4_complete.json",
        CTX / "part6_complete.json",
        GQ.parents[1] / "part5_complete.json",
        GE.parents[1] / "part4_complete.json",
        GC.parents[1] / "part3_complete.json",
    ):
        seal = a.read(f)
        bindings = seal.get("outputs", seal.get("artifacts"))
        a.require(bindings is not None, "Missing seal map")
        a.verify(bindings)
        checked.append(dict(path=a.rel(f), bindings=len(bindings)))
    for n in ("astra_session.md", "astra_review.md"):
        dest = BASE / "log_preimages" / n
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as h:
            h.write((ROOT / "Thesis_Log" / n).read_bytes())
    save("tests.json", tests)
    save(
        "analysis_contract.json",
        dict(
            created_utc=a.utc(),
            status="retrospective_part3_scope_before_new_combined_quality_scoring",
            question="How does RAG-based prompting compare to standard prompting without retrieval, measured with multi-label metrics?",
            aim="Connect complete structured answers to editable external knowledge and separate accuracy, consistency, format and explanation faithfulness.",
            exposure="Standalone quality/mechanism findings are known. This is retrospective descriptive integration, not preregistration or independent confirmation.",
            endpoints=endpoints(),
            methods="Every retained Part2 view and separate five-fit sensitivity. Never select a seed, cutoff or model using these scores.",
            masks="Own-valid/full-eligible plus the unchanged Qwen common-nine mask. Missing outputs never become benign or empty labels; observed empty gold differs from unknown annotations.",
            exact="All output label columns count, even unsupported macro columns. Fine fields require their own predicted hate gate; legal is independent. Jaccard/F1 of two observed empty sets is one, with gate-aware exactness reported separately.",
            joints="Only hate+target+severity on genuine MHS co-annotated TEST and larger DEV identities. Same fit/policy gate and validity must agree. No all-field score, no benign false-alarm claim.",
            aggregation="Per view, within TRAIN pool, equal pool means. Undefined conditional exactness excludes undefined fits with explicit defined-fit counts; it is not an oracle intervention.",
            new_descriptives="All fine exact/partial/cardinality measures and joint partitions on both masks; all gold-cardinality strata and existing source labels; raw-head/gate loss partition for every original uniform GPU view on both masks. All defaults and selected cutoffs retained.",
            gate_reader="Recover historical larger-DEV gates from previously checked broader-quality encoded decisions after exact identity/decision/validity checks. Do not alter Part2 caches. Legal gates are available but never required for legal correctness.",
            raw_head_limit="Only supervised raw heads are observed. Qwen already-parsed/voted outputs cannot reveal a latent ungated classifier; no oracle gate accuracy is invented.",
            mechanisms="Reuse original same-weights cutoff factorization, cross-head/gate swaps, annotation coverage, TRAIN selection, matched-information controlled prompts, actual contexts and fixed casebook. Distinguish deterministic output effects, intervention evidence, confounded associations and illustrative rationale issues.",
            uncertainty="No new confidence intervals, bootstrap draws, tests or primary family. Carry original quality intervals with original pair masks and supporting roles. New cross-study points do not prove superiority/equivalence.",
            remaining="Part4 confidence/resources/global reconciliation/final scientific closure; separate Part5 writing handover. No writer edits now.",
            verified_prior_seals=checked,
            sources=a.bind(sources()),
            new_Qwen_calls=0,
            new_training=0,
            new_inference=0,
            new_retrieval=0,
            new_bootstrap_draws=0,
            server_actions=0,
        ),
    )
    print("Part3 scope and boundary tests saved; prior evidence intact.", flush=True)


def analyze():
    a.verify(a.read(BASE / "analysis_contract.json")["sources"])
    summary = []
    cache = {}
    raw_results = {}
    strata = {}
    roster = {a.key(r): r for r in a.rows(a.ROSTER)}
    gm = a.read(a.GPU / "models.json")
    for endpoint in endpoints():
        md, z, restored = get(endpoint)
        cache[endpoint] = (md, z)
        models = md["models"]
        legal = endpoint.endswith("/legal")
        slug = endpoint.replace("/", "__")
        result = dict(
            endpoint=endpoint,
            models=models,
            ids=md["ids"],
            labels=md["labels"],
            eligible=len(md["ids"]),
            common_nine_n=int(z["common"].sum()),
            restored_saved_gate_views=restored,
        )
        for mode in ("own_valid", "common_nine"):
            runs = []
            for j in range(len(models)):
                mask = z["valid"][j] if mode == "own_valid" else z["common"]
                a.require(np.all(z["valid"][j, mask]), "Missing output reached scoring")
                runs.append(
                    summarize(
                        z["gold"][mask],
                        z["predictions"][j, mask],
                        z["gate"][j, mask],
                        legal,
                    )
                )
            result[mode] = dict(runs=runs, groups=aggregate(models, runs))
        save("endpoints/" + slug + ".json", result)

        f = BASE / "endpoints" / (slug + "_gates.npz")
        with f.open("xb") as h:
            np.savez_compressed(h, gate=z["gate"])
        summary.append(
            dict(
                endpoint=endpoint,
                eligible=len(md["ids"]),
                common_nine=int(z["common"].sum()),
                views=len(models),
                groups=len(result["common_nine"]["groups"]),
                restored_gate_views=restored,
            )
        )

        st = {}
        src = np.asarray([roster[k]["source"] for k in md["ids"]])
        card = z["gold"].sum(1)
        masks = {
            **{"source/" + str(s): src == s for s in sorted(set(src))},
            **{"gold_labels/" + str(int(c)): card == c for c in sorted(set(card))},
        }
        for name, mask in masks.items():
            mask = mask & z["common"]
            rr = [
                summarize(
                    z["gold"][mask],
                    z["predictions"][j, mask],
                    z["gate"][j, mask],
                    legal,
                )
                for j in range(len(models))
            ]
            st[name] = dict(n=int(mask.sum()), groups=aggregate(models, rr))
        strata[endpoint] = st
        source = a.GPU / (slug + "_arrays.npz")
        if source.exists():
            gz = p.load(source)
            meta = a.read(a.GPU / (slug + "_metadata.json"))
            a.require(
                np.array_equal(gz["truth"], z["gold"])
                and [a.key(r) for r in meta["records"]] == md["ids"],
                "Raw-head identity mismatch",
            )
            ix = [
                j
                for j, m in enumerate(models)
                if m["group"].startswith("gpu/") and "sensitivity" not in m["group"]
            ]
            a.require(len(ix) == len(gm), "Raw-head model count mismatch")
            for j, m in enumerate(gm):
                a.require(
                    models[ix[j]]["original_method"] == m["id"],
                    "Raw-head view order mismatch",
                )
                a.require(
                    np.array_equal(gz["gated"][j], z["predictions"][ix[j]])
                    and np.array_equal(gz["gate"][j], z["gate"][ix[j]]),
                    "Raw-head delivered output mismatch",
                )
                a.require(
                    np.array_equal(
                        gz["gated"][j], gz["raw"][j] * gz["gate"][j, :, None]
                    ),
                    "Raw/gate composition mismatch",
                )
            bymask = {}
            for mode, mask in [
                ("own_valid", np.ones(len(z["gold"]), bool)),
                ("common_nine", z["common"]),
            ]:
                rr = []
                for j in range(len(gm)):
                    rows = flow_rows(
                        gz["truth"][mask], gz["raw"][j, mask], gz["gate"][j, mask]
                    )
                    rr.append(
                        dict(
                            n=int(mask.sum()),
                            metrics={k: float(v.mean()) for k, v in rows.items()},
                            counts={k: float(v.sum()) for k, v in rows.items()},
                        )
                    )
                bymask[mode] = dict(
                    runs=rr, groups=aggregate([models[i] for i in ix], rr)
                )
            raw_results[endpoint] = dict(model_indices=ix, **bymask)
        print(endpoint, "quality complete", len(models), "saved views", flush=True)
    save("quality_strata.json", strata)
    save("raw_head_gate_flows.json", raw_results)
    joints = {}
    for phase in ("en_test", "larger_dev"):
        tm, tz = cache[phase + "/target_group"]
        sm, sz = cache[phase + "/severity"]
        lookup = {k: i for i, k in enumerate(tm["ids"])}
        ii = np.asarray([lookup[k] for k in sm["ids"]])
        a.require(
            [(m["group"], m["pool"], m["seed"]) for m in tm["models"]]
            == [(m["group"], m["pool"], m["seed"]) for m in sm["models"]],
            "Joint fit identity mismatch",
        )
        a.require(
            np.array_equal(tz["valid"][:, ii], sz["valid"]),
            "Joint output validity differs",
        )
        for j in range(len(sm["models"])):
            a.require(
                np.array_equal(
                    tz["gate"][j, ii][sz["valid"][j]], sz["gate"][j][sz["valid"][j]]
                ),
                "Joint own gate differs",
            )
        a.require(
            np.array_equal(tz["common"][ii], sz["common"]), "Joint common masks differ"
        )
        a.require(
            all(
                roster[k]["source"] == "mhs"
                and roster[k]["gate"] is True
                and roster[k]["target_groups"] is not None
                and roster[k]["severity"] is not None
                for k in sm["ids"]
            ),
            "Joint gold availability mismatch",
        )
        v = dict(
            ids=sm["ids"],
            models=sm["models"],
            fields=["hate", "target_group", "severity"],
            eligible=len(sm["ids"]),
            common_nine_n=int(sz["common"].sum()),
        )
        for mode in ("own_valid", "common_nine"):
            rr = []
            for j in range(len(sm["models"])):
                mask = sz["valid"][j] if mode == "own_valid" else sz["common"]
                gate = sz["gate"][j, mask].astype(bool)
                te = np.all(
                    tz["predictions"][j, ii][mask] == tz["gold"][ii][mask], axis=1
                )
                se = np.all(sz["predictions"][j, mask] == sz["gold"][mask], axis=1)
                rows = joint_rows(te, se, gate)
                a.require(
                    np.all(sum(rows.values()) == 1),
                    "Joint failure partition incomplete",
                )
                rr.append(
                    dict(
                        n=int(mask.sum()),
                        metrics={k: float(x.mean()) for k, x in rows.items()},
                        counts={k: int(x.sum()) for k, x in rows.items()},
                    )
                )
            v[mode] = dict(runs=rr, groups=aggregate(sm["models"], rr))
        joints[phase] = v
    save("joint_quality.json", joints)

    orig = {
        e: a.read(BQ / "endpoints" / e.replace("/", "__") / "results.json")["pairs"]
        for e in endpoints()
    }
    orig["joints"] = {
        k: v["pairs"] for k, v in a.read(BQ / "joint_results.json").items()
    }
    save("original_quality_contrasts.json", orig)
    save(
        "analysis_complete.json",
        dict(
            created_utc=a.utc(),
            endpoints=summary,
            joint_phases=list(joints),
            sources=a.read(BASE / "analysis_contract.json")["sources"],
            new_experiments=0,
        ),
    )


if __name__ == "__main__":
    a.g.offline()
    args = argparse.ArgumentParser()
    args.add_argument("mode", choices=["freeze", "analyze"])
    mode = args.parse_args().mode
    globals()[mode]()
