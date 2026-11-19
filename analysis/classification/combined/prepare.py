import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
from analysis.classification.supervised import archive as g

ROOT = g.ROOT
BASE = ROOT / "experiments/astra/sq2_combined_setup_20261104_v1"
REPORTS = ROOT / "Thesis_Log/astra_result"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BROAD = ROOT / "experiments/astra/sq2_broad_reader_review_20261104_v1"
MAIN = ROOT / "experiments/astra/sq2_broad_main_review_20261104_v1"
GPU = ROOT / "experiments/astra/sq2_gpu_sources_review_20261104_v1/extraction"
MATCHED = ROOT / "experiments/astra/sq2_matched_review_20261030_v1"
ROSTER = (
    ROOT
    / "experiments/astra/offline_closure_20261006_v1/evaluation_preparation/broad_roster.jsonl"
)
FIELDS = dict(
    hate="gate",
    hate_type="hate_types",
    target_group="target_groups",
    severity="severity",
    legal="legal",
)
PLAN = ROOT / "docs/classification.md"

require = g.require


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def rows(p):
    with Path(p).open(encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def rel(p):
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()


def bind(paths):
    return {rel(p): g.sha(p) for p in paths}


def verify(values):
    for p, expected in values.items():
        require(
            g.sha(Path(p) if Path(p).is_absolute() else ROOT / p) == expected,
            "Changed source: " + p,
        )


def save(p, value):
    p = Path(p)
    require(p.resolve().is_relative_to(BASE), "Output outside current analysis")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(g.encoded(value))


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf8", newline="\n") as f:
        f.write(value)


def clean(v):
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def key(row):
    if row.get("key"):
        return row["key"]
    ident = str(row["id"])
    if ident.startswith(("en/", "de/")):
        return ident
    require(row.get("lang") in ("en", "de"), "No explicit language for unqualified ID")
    return row["lang"] + "/" + ident


def index(rr):
    out = {}
    for row in rr:
        k = key(row)
        require(k not in out, "Duplicate canonical input " + k)
        out[k] = row
    return out


def labels(row, dim):
    value = clean(row.get(FIELDS[dim]))
    gate = clean(row.get("gate"))
    require(gate is None or type(gate) is bool, "Nonboolean gold gate")
    if value is None or dim not in ("hate", "legal") and gate is not True:
        return None
    if dim == "hate":
        return {"hate" if value else "not_hate"}
    if dim == "severity":
        return {value}
    require(isinstance(value, list), "Multilabel annotation is not a list")
    return set(value)


def field_signature(row):
    return {
        f: (
            sorted(clean(row.get(f)))
            if isinstance(clean(row.get(f)), list)
            else clean(row.get(f))
        )
        for f in FIELDS.values()
    }


def normalized_sha(text):
    return g.digest(g.normalized(text).encode("utf8"))


def parquet(p):
    import pandas as pd

    return [
        {k: clean(v) for k, v in row.items()}
        for row in pd.read_parquet(p).to_dict("records")
    ]


def registry(name, rr, ep):
    selected = [r for r in rr if labels(r, ep["dimension"]) is not None]
    kk = [key(r) for r in selected]
    saved = [
        x if x.startswith(("en/", "de/")) else (selected[0]["lang"] + "/" + x)
        for x in ep["scorable_ids"]
    ]
    require(kk == saved, "Saved eligibility/order differs: " + name)
    support = Counter(x for r in selected for x in labels(r, ep["dimension"]))
    require(
        all(support[x] == ep["gold_support"][x] for x in ep["all_labels"]),
        "Gold support differs: " + name,
    )
    require(
        ep["macro_labels"] == [x for x in ep["all_labels"] if support[x] >= 10],
        "Macro labels differ: " + name,
    )
    require(
        all(labels(r, ep["dimension"]) <= set(ep["all_labels"]) for r in selected),
        "Unknown gold label",
    )
    return dict(
        name=name,
        dimension=ep["dimension"],
        input_rows=len(rr),
        eligible=len(kk),
        eligible_ids=kk,
        labels=ep["all_labels"],
        macro_labels=ep["macro_labels"],
        gold_support=ep["gold_support"],
        observed_empty=sum(labels(r, ep["dimension"]) == set() for r in selected),
        gold_not_annotated=sum(
            clean(r.get(FIELDS[ep["dimension"]])) is None for r in rr
        ),
        observed_but_gate_ineligible=sum(
            clean(r.get(FIELDS[ep["dimension"]])) is not None
            and labels(r, ep["dimension"]) is None
            for r in rr
        ),
    )


def paths():
    reports = [
        "astra_remaining_work_before_writing_20261104_v152.md",
        "astra_sq2_broad_final_synthesis_20261104_v2.md",
        "astra_sq2_gpu_final_synthesis_20261104_v1.md",
        "astra_sq2_matched_results_20261030_v1.md",
        "astra_sq2_short_training_results_20261020_v1.md",
        "astra_sq2_cpu_results_20261012_v1.md",
        "astra_sq2_cpu_protocol_20261011_v1.md",
        "astra_sq2_compact_bge_results_20261008_v1.md",
        "astra_sq2_compact_controls_results_20261008_v1.md",
        "astra_sq2_compact_pool_results_20261008_v1.md",
        "astra_sq2_compact_mechanism_results_20261008_v1.md",
        "astra_sq2_compact_broad_results_20261008_v1.md",
        "astra_sq2_roberta_tuning_results_20261011_v1.md",
        "astra_sq2_matched_type_results_20261012_v1.md",
        "astra_core_results_summary_20260920_v1.md",
        "astra_core_encoder_results_20260915_v5.md",
        "astra_sq2_label_vote_results_20261001_v1.md",
        "astra_pooled_fineheads_results_20260930_v1.md",
        "astra_sq2_type_update_results_20261103_v1.md",
        "astra_sq2_broad_main_comparisons_20261104_v1.md",
        "astra_sq2_broad_answer_quality_20261104_v1.md",
        "astra_sq2_broad_retrieval_contexts_20261104_v2.md",
        "astra_sq2_gpu_robustness_20261104_v1.md",
    ]

    reports = [REPORTS / n for n in reports if (REPORTS / n).exists()]
    pp = reports + [
        Path(__file__),
        ROOT / "tests/classification/combined/test_prepare.py",
        BASE / "tests.json",
        PLAN,
        ROSTER,
        BROAD / "endpoint_setup.json",
        MATCHED / "metrics.json",
        MATCHED / "gold_rosters.jsonl",
        MATCHED / "predictions.jsonl",
        GPU / "complete.json",
        GPU / "context.json",
        GPU / "models.json",
        g.LINEAR / "data/scopes.json",
        g.LINEAR / "data/endpoints.json",
        g.LINEAR / "data/inference.json",
        ROOT / "experiments/astra/compact_bge_20261008_v1/plan.json",
        ROOT / "experiments/astra/compact_broad_20261008_v1/plan.json",
        ROOT / "experiments/astra/sq2_cpu_baselines_20261011_v1/plan.json",
        ROOT / "experiments/astra/sq2_cpu_baselines_20261011_v1/data/scopes.json",
        ROOT / "experiments/astra/roberta_short_review_20261020_v1/scoring_plan.json",
        ROOT
        / "experiments/astra/sq2_gpu_robustness_review_20261104_v1/nesting_correction/review.json",
        CORE / "kb/full/records.jsonl",
        CORE / "kb/small/records.jsonl",
        CORE / "api_execution_v1/scoring_plan.json",
    ]
    for arm in ("kb", "full"):
        for head in g.HEADS:
            pp += [
                CORE / "encoder_data" / arm / (head + "_" + part + ".parquet")
                for part in ("train", "val")
            ]
            pp += [CORE / "encoder_data" / arm / (head + "_meta.json")]
    for pool in g.POOLS:
        pp += [
            g.PACKAGE / "data/train" / f"{pool}.json",
            g.LINEAR / "data/train" / f"{pool}.json",
        ]
    for name in read(g.LINEAR / "data/endpoints.json")["core"]:
        pp.append(ROOT / f"data/processed/{name}.parquet")
    pp += [
        ROOT / "data/processed/de_legal_dev_eval.parquet",
        ROOT / "data/processed/de_dev_eval.parquet",
    ]
    for ep in read(BROAD / "endpoint_setup.json"):
        slug = ep.replace("/", "__")
        pp += [
            MAIN / "endpoints" / slug / "results.json",
            MAIN / "endpoints" / slug / "encoded.npz",
        ]
        if (GPU / (slug + "_metadata.json")).exists():
            pp += [GPU / (slug + "_metadata.json"), GPU / (slug + "_arrays.npz")]
    return pp


def freeze():
    require(not (BASE / "analysis_contract.json").exists(), "Scope already frozen")
    require(PLAN.exists(), "Record the combined plan first")
    seals = [
        ("sq2_broad_synthesis_20261104_v1/part9_complete.json", "outputs"),
        ("sq2_gpu_synthesis_20261104_v1/part9_complete.json", "artifacts"),
        ("sq2_matched_review_20261030_v1/complete.json", "outputs"),
        ("roberta_short_review_20261020_v1/complete.json", "bindings"),
    ]
    verified = []
    old = {}
    sealpaths = []
    for name, field in seals:
        p = ROOT / "experiments/astra" / name
        v = read(p)
        verify(v[field])
        old.update(v[field])
        sealpaths.append(p)
        verified.append(
            dict(
                path=rel(p),
                sha256=g.sha(p),
                status=v["status"],
                bound_outputs=len(v[field]),
            )
        )
    p = GPU / "complete.json"
    v = read(p)
    for n, h in v["outputs"].items():
        require(g.sha(GPU / n) == h, "Changed GPU extraction " + n)
    require(
        v["qwen_calls"] == 0 and v["training_calls"] == 0 and v["weights_loaded"] == 0,
        "Extraction is not offline",
    )
    save(
        BASE / "analysis_contract.json",
        dict(
            status="combined_SQ2_part1_scope_recorded_before_alignment",
            created_utc=utc(),
            question="How does RAG-based prompting compare to standard prompting without retrieval, measured with multi-label metrics?",
            purpose="Validate common evaluation identities, labels, masks, available training information, model identities and inferential roles. Prepare combined synthesis without computing new performance results.",
            exposure="Prior standalone outcomes already read. This is retrospective integration, not preregistration or an untouched holdout.",
            sources=bind(paths() + sealpaths),
            prior_seals=verified,
            verified_prior_outputs=old,
            computations=[
                "Exact canonical IDs, text, annotations, label columns and clusters across saved readers.",
                "Gold-eligible and all-repeat validity masks; complete-case and full-roster denominators kept separately.",
                "Actual training pool/head memberships, full/compact partitions, overlap and nesting.",
                "Separate generation repeats, model fits, TRAIN pool draws, cutoff views, and reused outputs.",
                "Classify within-study comparisons, aligned cross-study context and incompatible causal contrasts.",
            ],
            no_actions=[
                "No F1 recalculation, new performance contrast, bootstrap draw, winner selection or outcome-dependent exclusion.",
                "No training, inference, model loading, embedding, retrieval, Qwen call, server action or writer edit.",
                "No change to original gold, macro-label supports, primary families, frozen recipes or archived results.",
                "No combined writing handover or whole-project closure in Part 1.",
            ],
            future_cross_study="Part 2 may descriptively score retained supervised predictions on the frozen Qwen common-nine mask and report all fitted settings, plus the original full-roster scores. These are new aligned contextual views, not equal-information causal effects or new original primary results. Any new interval protocol must be recorded before calculating it.",
        ),
    )
    print(
        "Scope frozen; verified",
        len(old),
        "prior immutable output bindings.",
        flush=True,
    )


def evaluation():
    contract = read(BASE / "analysis_contract.json")
    verify(contract["sources"])
    broad = list(rows(ROSTER))
    bi = index(broad)
    matched = list(rows(MATCHED / "gold_rosters.jsonl"))
    mi = index(matched)
    bep = read(BROAD / "endpoint_setup.json")
    lep = read(g.LINEAR / "data/endpoints.json")
    reg = []
    align = []
    output = []
    for name, ep in bep.items():
        phase, dim = name.split("/")
        rr = [r for r in broad if r["phase"] == phase]
        rv = registry("broad/" + name, rr, ep)
        reg.append(rv)
        v = read(MAIN / "endpoints" / name.replace("/", "__") / "results.json")
        require(v["endpoint"] == ep, "Reader endpoint drift")
        with np.load(ROOT / v["encoded_path"], allow_pickle=False) as z:
            gold = z["gold"]
            valid = z["valid"][:9]
        ids = rv["eligible_ids"]
        truth = np.array(
            [[label in labels(bi[k], dim) for label in rv["labels"]] for k in ids],
            dtype=np.uint8,
        )
        require(np.array_equal(gold, truth), "Broader saved gold differs")
        require(
            v["settings"][:9]
            == [
                [co, r]
                for co in ("rag_full", "rag_small", "zero_shot")
                for r in range(3)
            ],
            "Unexpected repeated settings",
        )
        require(
            valid.shape == (9, len(ids)) and valid.dtype == np.bool_,
            "Invalid validity matrix",
        )
        for s in range(9):
            co, rep = v["settings"][s]
            declared = set(ep["own_invalid_ids"][co + "/" + str(rep)])
            require(
                declared == {ids[j] for j in range(len(ids)) if not valid[s, j]},
                "Own-invalid IDs differ",
            )
        common = valid.all(0)
        require(
            {ids[j] for j in range(len(ids)) if not common[j]}
            == set(ep["three_Qwen_conditions_common_invalid_ids"]),
            "Common-nine mask differs",
        )
        rv.update(
            common_nine=int(common.sum()),
            own_valid=[int(x.sum()) for x in valid],
            pair_six={
                pair: int(valid[ii].all(0).sum())
                for pair, ii in [
                    ("full_small", list(range(6))),
                    ("full_zero", [0, 1, 2, 6, 7, 8]),
                    ("small_zero", list(range(3, 9))),
                ]
            },
        )
        gp = GPU / (name.replace("/", "__") + "_metadata.json")
        if gp.exists():
            m = read(gp)
            grecs = m["records"]
            gi = index(grecs)
            require(set(gi) == set(ids), "GPU/Qwen eligible IDs differ")
            require(
                m["labels"] == ep["all_labels"]
                and lep["broad"][name]["macro_labels"] == ep["macro_labels"],
                "GPU/Qwen label contracts differ",
            )
            positions = {k: i for i, k in enumerate(ids)}
            order = np.array([positions[key(r)] for r in grecs], dtype=np.int32)
            with np.load(
                GPU / (name.replace("/", "__") + "_arrays.npz"), allow_pickle=False
            ) as z:
                gy = z["truth"]
                shape = z["gated"].shape
            require(
                shape == (306, len(ids), len(m["labels"])),
                "GPU cached prediction shape differs",
            )
            require(np.array_equal(gy, truth[order]), "GPU/Qwen saved truth differs")
            for i, r in enumerate(grecs):
                br = bi[key(r)]
                require(r["text"] == br["text"], "GPU/Qwen input text differs")
                require(
                    field_signature(r) == field_signature(br),
                    "GPU/Qwen raw gold differs",
                )
                expected = (
                    br.get("functional_cluster")
                    if phase == "en_hatecheck_eval"
                    else normalized_sha(br["text"])
                )
                require(
                    m["clusters"][i] == expected
                    and ep["row_clusters"][order[i]] == expected,
                    "GPU/Qwen cluster differs",
                )
            dest = BASE / "alignment" / (name.replace("/", "__") + ".npz")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as f:
                np.savez_compressed(
                    f,
                    gpu_to_qwen_rows=order,
                    gold=truth[order],
                    valid=valid[:, order],
                    common_nine=common[order],
                    full_small=valid[:6, order].all(0),
                    full_zero=valid[[0, 1, 2, 6, 7, 8]][:, order].all(0),
                    small_zero=valid[3:, order].all(0),
                )
            output.append(dest)
            align.append(
                dict(
                    endpoint=name,
                    eligible=len(ids),
                    common_nine=int(common.sum()),
                    excluded=len(ids) - int(common.sum()),
                    labels=rv["labels"],
                    macro_labels=rv["macro_labels"],
                    gpu_ids=[key(r) for r in grecs],
                    qwen_ids=ids,
                    same_order=bool(np.array_equal(order, np.arange(len(order)))),
                    raw_gold_identical=True,
                    text_identical=True,
                    clusters_identical=True,
                    gpu_model_settings=306,
                    source_metadata=rel(gp),
                    source_predictions=rel(
                        GPU / (name.replace("/", "__") + "_arrays.npz")
                    ),
                    qwen_source=rel(ROOT / v["encoded_path"]),
                    alignment=rel(dest),
                    sha256=g.sha(dest),
                )
            )
        print(
            "Verified evaluation identity:",
            name,
            len(ids),
            "eligible,",
            int(common.sum()),
            "common-nine.",
            flush=True,
        )
    core = {}
    core_groups = {}
    for name, ep in lep["core"].items():
        rr = parquet(ROOT / f"data/processed/{name}.parquet")
        core_groups[name] = index(rr)
        for k, r in core_groups[name].items():
            require(k not in core, "Core duplicates across separate endpoint rosters")
            core[k] = r
        reg.append(registry("core/" + name, rr, ep))
    de_core = {}
    for name in ("de_legal_dev_eval", "de_dev_eval"):
        rr = parquet(ROOT / f"data/processed/{name}.parquet")
        de_core.update(index(rr))
        core_groups[name] = index(rr)
    metrics = read(MATCHED / "metrics.json")
    for name, v in metrics["results"].items():
        reg.append(
            registry(
                "matched/" + name,
                [r for r in matched if r["phase"] == name.split("/")[0]],
                v["endpoint"],
            )
        )

    scopes = read(g.LINEAR / "data/scopes.json")
    inputs = read(g.LINEAR / "data/inference.json")
    require(len(inputs) == 19082, "Wrong supervised inference roster")
    for scope, rr in scopes.items():
        source = core if scope == "core" else bi
        for r in rr:
            require(
                inputs[r["position"]]["text"] == source[r["key"]]["text"],
                "Supervised inference text changed",
            )
    require(
        read(ROOT / "experiments/astra/compact_bge_20261008_v1/plan.json")["endpoints"]
        == lep["core"],
        "Compact core endpoint differs",
    )
    require(
        read(ROOT / "experiments/astra/compact_broad_20261008_v1/plan.json")[
            "endpoints"
        ]
        == lep["broad"],
        "Compact broad endpoint differs",
    )
    populations = {
        "core_English": core,
        "core_German": de_core,
        "matched_DEV": mi,
        "broader_all": bi,
        "broader_English": {k: v for k, v in bi.items() if k.startswith("en/")},
        "broader_German": {k: v for k, v in bi.items() if k.startswith("de/")},
    }
    for phase in sorted({r["phase"] for r in broad}):
        populations[phase] = {k: v for k, v in bi.items() if v["phase"] == phase}
    overlaps = []
    wanted = [
        ("core_English", "matched_DEV"),
        ("core_German", "matched_DEV"),
        ("core_English", "broader_all"),
        ("core_German", "broader_all"),
        ("matched_DEV", "broader_all"),
        ("matched_DEV", "en_test"),
        ("matched_DEV", "larger_dev"),
        ("matched_DEV", "en_hatecheck_eval"),
    ]
    for left, right in wanted:
        a, b = populations[left], populations[right]
        ids = sorted(set(a) & set(b))
        text = {normalized_sha(r["text"]) for r in a.values()} & {
            normalized_sha(r["text"]) for r in b.values()
        }
        require(
            all(
                a[k]["text"] == b[k]["text"]
                and field_signature(a[k]) == field_signature(b[k])
                for k in ids
            ),
            "Shared evaluation gold/text changed",
        )
        overlaps.append(
            dict(
                left=left,
                right=right,
                left_rows=len(a),
                right_rows=len(b),
                shared_ids=len(ids),
                shared_normalized_texts=len(text),
                ids=ids,
                shared_phase_counts=dict(
                    Counter(
                        (
                            b[k]["phase"]
                            if "phase" in b[k]
                            else next(n for n, v in core_groups.items() if k in v)
                        )
                        for k in ids
                    )
                ),
            )
        )

    joint = []
    for phase in ("en_test", "larger_dev"):
        rr = [r for r in broad if r["phase"] == phase]
        ids = [
            key(r)
            for r in rr
            if all(
                labels(r, d) is not None for d in ("hate", "target_group", "severity")
            )
        ]
        validity = {}
        for d in ("target_group", "severity"):
            v = read(MAIN / "endpoints" / (phase + "__" + d) / "results.json")
            with np.load(ROOT / v["encoded_path"], allow_pickle=False) as z:
                vv = z["valid"][:9].all(0)
            validity[d] = {
                k: bool(vv[i]) for i, k in enumerate(v["endpoint"]["scorable_ids"])
            }
        common = [k for k in ids if all(validity[d][k] for d in validity)]
        joint.append(
            dict(
                phase=phase,
                heads=["hate", "target_group", "severity"],
                eligible=len(ids),
                common_nine=len(common),
                ids=ids,
                common_nine_ids=common,
            )
        )
    save(BASE / "endpoint_registry.json", reg)
    save(BASE / "evaluation_alignment.json", align)
    save(
        BASE / "population_overlap.json",
        dict(
            populations={k: len(v) for k, v in populations.items()},
            pairs=overlaps,
            supervised_unique_inputs=len(inputs),
            supervised_core_broad_overlap=len(set(core) & set(bi)),
            joint_cohorts=joint,
        ),
    )
    save(
        BASE / "evaluation_complete.json",
        dict(
            status="identity_gold_cluster_and_mask_alignment_passed",
            created_utc=utc(),
            endpoints=len(reg),
            direct_broad_gpu_endpoints=len(align),
            new_scores=0,
            outputs=bind(
                output
                + [
                    BASE / "endpoint_registry.json",
                    BASE / "evaluation_alignment.json",
                    BASE / "population_overlap.json",
                ]
            ),
        ),
    )


def training():
    contract = read(BASE / "analysis_contract.json")
    verify(contract["sources"])
    banks = {
        name: {
            r["id"]: r
            for r in rows(CORE / "kb" / name / "records.jsonl")
            if r["kind"] == "example"
        }
        for name in ("small", "full")
    }
    require(set(banks["small"]) <= set(banks["full"]), "Compact bank not subset")
    require(
        all(r == banks["full"][k] for k, r in banks["small"].items()),
        "Shared KB record differs",
    )
    english = {
        name: {k: r for k, r in bank.items() if r["lang"] == "en"}
        for name, bank in banks.items()
    }

    def headrows(bank, head):
        return {k: r for k, r in bank.items() if labels(r["meta"], head) is not None}

    inventory = []
    native = []
    for name, bank in banks.items():
        inventory.append(
            dict(
                name="Qwen_" + name,
                rows=len(bank),
                languages=dict(Counter(r["lang"] for r in bank.values())),
                english_head_counts={
                    h: len(headrows(english[name], h)) for h in g.HEADS
                },
                distinct_normalized_texts=len(
                    {normalized_sha(r["text"]) for r in bank.values()}
                ),
            )
        )
    for arm, bname in [("kb", "small"), ("full", "full")]:
        for h in g.HEADS:
            parts = {
                s: parquet(CORE / "encoder_data" / arm / (h + "_" + s + ".parquet"))
                for s in ("train", "val")
            }
            joined = {str(r["id"]): r for rr in parts.values() for r in rr}
            eligible = headrows(english[bname], h)
            require(
                len(joined) == sum(len(rr) for rr in parts.values()),
                "Native train/validation ID overlap",
            )
            require(
                set(joined) == set(eligible),
                "Native permitted head IDs not identical to corresponding KB",
            )
            require(
                all(
                    r["text"] == eligible[k]["text"]
                    and labels(r, h) == labels(eligible[k]["meta"], h)
                    for k, r in joined.items()
                ),
                "Native source annotations differ",
            )
            native.append(
                dict(
                    arm=arm,
                    head=h,
                    gradient_train=len(parts["train"]),
                    internal_validation=len(parts["val"]),
                    all_permitted=len(joined),
                    kb_membership_and_gold_equal=True,
                    train_ids=sorted(str(r["id"]) for r in parts["train"]),
                    validation_ids=sorted(str(r["id"]) for r in parts["val"]),
                )
            )
    pools = {}
    pstats = []
    overlaps = []
    for name in g.POOLS:
        d = read(g.PACKAGE / "data/train" / f"{name}.json")
        l = read(g.LINEAR / "data/train" / f"{name}.json")
        require(d == l, "GPU/linear TRAIN package differs")
        ids = set(d["gate_ids"])
        require(
            len(ids) == len(d["gate_ids"]) == len(d["texts"]), "TRAIN IDs duplicated"
        )
        require(ids <= set(english["full"]), "Pool outside permitted English source")
        require(
            all(
                t == english["full"][k]["text"]
                for k, t in zip(d["gate_ids"], d["texts"])
            ),
            "Pool texts differ from bank",
        )
        counts = {}
        targets = {}
        for h, spec in d["heads"].items():
            rr = spec["rows"]
            hh = {r["id"] for r in rr}
            expect = ids & set(headrows(english["full"], h))
            require(
                hh == expect and len(hh) == len(rr),
                "Head mask does not match observed gold",
            )
            for r in rr:
                target = r["target"]
                labs = spec["labels"]
                positive = (
                    {labs[int(target)]}
                    if spec["kind"] in ("binary", "multiclass")
                    else {x for x, v in zip(labs, target) if v}
                )
                if h == "hate":
                    positive = {"hate" if x == "true" else "not_hate" for x in positive}
                require(
                    positive == labels(english["full"][r["id"]]["meta"], h),
                    "Training label mismatch",
                )
            counts[h] = len(rr)
            targets[h] = dict(
                Counter(x for k in hh for x in labels(english["full"][k]["meta"], h))
            )
        pools[name] = ids
        cur = set(english["small"])
        shared = ids & cur
        pstats.append(
            dict(
                pool=name,
                total_texts=len(ids),
                annotated_heads=counts,
                positive_support=targets,
                curated_overlap=len(shared),
                curated_shared_ids=sorted(shared),
                source_counts=dict(Counter(english["full"][k]["source"] for k in ids)),
                training_ids=sorted(ids),
                same_GPU_and_linear_data=True,
            )
        )
    for seed in (42, 43, 44):
        for a, b in [("compact", "1000"), ("1000", "5000")]:
            left, right = f"uniform_{a}_s{seed}", f"uniform_{b}_s{seed}"
            require(pools[left] <= pools[right], "Expected nested pool chain failed")
            overlaps.append(
                dict(
                    left=left,
                    right=right,
                    shared=len(pools[left] & pools[right]),
                    left_subset_of_right=True,
                )
            )
    same_budget = []
    for budget in ("compact", "1000", "5000"):
        for a, b in ((42, 43), (42, 44), (43, 44)):
            x, y = f"uniform_{budget}_s{a}", f"uniform_{budget}_s{b}"
            same_budget.append(dict(left=x, right=y, shared=len(pools[x] & pools[y])))
    evaluation = list(rows(ROSTER)) + list(rows(MATCHED / "gold_rosters.jsonl"))
    for name in read(g.LINEAR / "data/endpoints.json")["core"]:
        evaluation += parquet(ROOT / f"data/processed/{name}.parquet")
    eval_text = {normalized_sha(r["text"]) for r in evaluation}
    leak = {
        name: len(eval_text & {normalized_sha(r["text"]) for r in bank.values()})
        for name, bank in banks.items()
    }
    require(
        not any(leak.values()), "Exact normalized training/evaluation overlap detected"
    )
    gate_train = {
        a: set(
            next(
                v["train_ids"] for v in native if v["arm"] == a and v["head"] == "hate"
            )
        )
        for a in ("kb", "full")
    }
    gate_validation_overlap = [
        dict(
            arm=v["arm"],
            head=v["head"],
            fine_validation=len(v["validation_ids"]),
            overlaps_gate_training=len(set(v["validation_ids"]) & gate_train[v["arm"]]),
        )
        for v in native
        if v["head"] != "hate"
    ]
    save(
        BASE / "training_information.json",
        dict(
            candidate_banks=inventory,
            native_partitions=native,
            uniform_pools=pstats,
            nested_chains=overlaps,
            same_budget_pair_overlap=same_budget,
            native_fine_validation_in_gate_training=gate_validation_overlap,
            exact_normalized_evaluation_overlap=leak,
            meaning="Total texts, observed annotations, gradient-fitting rows, selection information and retrieved slots are separate budgets. No training or retrieval is performed.",
        ),
    )
    save(
        BASE / "training_complete.json",
        dict(
            status="training_membership_annotation_and_budget_alignment_passed",
            created_utc=utc(),
            pools=len(pstats),
            native_heads=len(native),
            new_fits=0,
            outputs=bind([BASE / "training_information.json"]),
        ),
    )
    print(
        "Verified nine equal-budget pools and eight native head partitions.", flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode", choices=["freeze", "evaluation", "training"], required=True
    )
    a = p.parse_args()
    with g.offline():
        globals()[a.mode]()
    require(
        not any(
            n in sys.modules
            for n in ("torch", "transformers", "sentence_transformers", "openai")
        ),
        "Unexpected model/client import",
    )
