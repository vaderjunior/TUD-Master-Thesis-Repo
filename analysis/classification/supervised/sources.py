from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.supervised import archive as r
from analysis.classification.supervised import metrics as m
from analysis.classification.supervised import controls as c
from analysis.classification.supervised import errors as e
from analysis.classification.supervised import quality as q

BASE = r.ROOT / "experiments/astra/sq2_gpu_sources_review_20261104_v1"
QUALITY = q.BASE / "runs/5047fbc65599"
ERRORS = e.BASE / "runs/54014ea4bfd9"
SOURCES = ("hatexplain", "implicit_hate", "mhs")
FUNCTIONS = {
    "derog_neg_emote_h": "Negative emotions",
    "derog_neg_attrib_h": "Negative attributes",
    "derog_dehum_h": "Dehumanisation",
    "derog_impl_h": "Implicit derogation",
    "threat_dir_h": "Direct threats",
    "threat_norm_h": "Normative threats",
    "slur_h": "Hateful slurs",
    "slur_homonym_nh": "Non-hateful slur homonyms",
    "slur_reclaimed_nh": "Reclaimed slurs",
    "profanity_h": "Hateful profanity",
    "profanity_nh": "Non-hateful profanity",
    "ref_subs_clause_h": "Reference in a later clause",
    "ref_subs_sent_h": "Reference in a later sentence",
    "negate_pos_h": "Negated positive statements",
    "negate_neg_nh": "Negated hateful statements",
    "phrase_question_h": "Hate phrased as a question",
    "phrase_opinion_h": "Hate phrased as an opinion",
    "ident_neutral_nh": "Neutral identity mentions",
    "ident_pos_nh": "Positive identity mentions",
    "counter_quote_nh": "Counter-speech quoting hate",
    "counter_ref_nh": "Counter-speech referring to hate",
    "target_obj_nh": "Abuse of objects",
    "target_indiv_nh": "Abuse of individuals",
    "target_group_nh": "Abuse of unprotected groups",
    "spell_char_swap_h": "Swapped characters",
    "spell_char_del_h": "Deleted characters",
    "spell_space_add_h": "Added spaces",
    "spell_space_del_h": "Removed spaces",
    "spell_leet_h": "Leetspeak substitutions",
}
save = q.save


def freeze():
    paths = [
        Path(__file__),
        Path(r.__file__),
        Path(m.__file__),
        Path(c.__file__),
        Path(e.__file__),
        Path(q.__file__),
        e.ROSTER,
        q.BASE / "part5_complete.json",
        e.BASE / "part4_complete.json",
        QUALITY / "models.json",
        r.ROOT / "Thesis_Log/astra_result/astra_sq2_stage_b_protocol_20261020_v1.md",
        r.ROOT / "experiments/astra/sq2_gpu_review_20261103_v1/reader_ready.json",
    ]
    save(
        BASE / "analysis_contract.json",
        dict(
            part=6,
            utc=datetime.now(timezone.utc).isoformat(),
            scope="All eight frozen broad English GPU endpoints; natural-source strata and all 29 HateCheck functions. No German, legal, Qwen scoring, core-exclusion or new fitting.",
            prior_exposure="Parts 1–5 reviewed. Annotation structure inspected; this is an analysis-time supporting scope, not new preregistration.",
            methods=list(c.METHODS),
            policies=list(r.POLICIES),
            pools=list(r.POOLS),
            seeds=list(r.SEEDS),
            groups=[
                "English TEST hate by all three sources; original annotation strata; HateXplain vote agreement",
                "Targets on TEST and larger DEV separately by HateXplain/MHS; type/severity remain their single annotated source",
                "All 29 HateCheck functions: errors at original text weighting, gold polarity, template/connected-group counts",
                "HateCheck original target identities split by gold polarity; diagnostic rates, no demographic causal fairness claim",
                "All linked reference pairs: both correct and expected same/changed decision; source links remain descriptive",
                "All original templates and inherited connected groups: points and equal-group diagnostic means",
                "Fine-label source support, raw/gated F1 and exactness; endpoint macro labels fixed even on subsets",
            ],
            uncertainty=dict(
                draws=10000,
                seed=20261020,
                level=0.95,
                role="Unadjusted supporting, conditional on fixed fits/pools; original 24 adjusted primary contrasts unchanged",
                natural_clusters="Frozen normalized-text clusters",
                hatecheck_clusters="Frozen whole connected template/reference/normalized-text components",
                single_polarity="Report error rate; absent-class FPR/FNR undefined, never a zero success",
                label_support="Original endpoint macro columns fixed across subsets and resamples; do not infer missing fine labels",
            ),
            cases=dict(
                budget=5000,
                procedure="tuned/selected",
                comparator="bge/selected",
                order=[
                    "One highest-frequency tuned error in each of the 29 functions in stored order; empty slots retained",
                    "One persistent FP and FN per natural source, if present",
                    "One gate-blocked raw-correct and one gate-passed raw-wrong target answer per larger-DEV source",
                    "One type exact gain and one type exact loss versus BGE; one severity under-by-two and one over-by-two",
                    "One linked both-correct gain/loss per same-gold/changed-gold stratum, if present",
                ],
                ranking="Mean across 15 matched neural views, descending; then normalized-text hash and identity. Do not repeat an identity; keep all candidate rankings and empty slots.",
                interpretation="Illustrations conditional on the recorded outcomes, not representative coding, relabeling or internal reasoning evidence.",
            ),
            preservation=dict(
                training_calls=0,
                qwen_calls=0,
                weights_loaded=0,
                server_actions=0,
                original_predictions_unchanged=True,
            ),
            functions=FUNCTIONS,
            inputs={str(p.relative_to(r.ROOT)): r.sha(p) for p in paths},
        ),
    )
    print("Part 6 supporting scope frozen before new outcome scoring.", flush=True)


def extract():
    for name, digest in r.read(BASE / "analysis_contract.json")["inputs"].items():
        r.checked_file(r.ROOT / name, digest)
    out = BASE / "extraction"
    out.mkdir(exist_ok=False)
    models = r.read(QUALITY / "models.json")
    lookup = {x["id"]: j for j, x in enumerate(models)}
    study = r.GpuStudy()
    linear = r.LinearStudy(study)
    context = r.SharedContext(study, linear)
    views = {
        name: view for (scope, name), view in context.views.items() if scope == "broad"
    }
    matrices = {
        name: dict(
            truth=view["truth"].astype(np.uint8),
            gated=np.empty(
                (len(models), len(view["records"]), len(view["label_indices"])),
                np.uint8,
            ),
            raw=np.empty(
                (len(models), len(view["records"]), len(view["label_indices"])),
                np.uint8,
            ),
            gate=np.empty((len(models), len(view["records"])), np.uint8),
        )
        for name, view in views.items()
    }
    filled = set()
    saved_bytes = 0
    for pool in r.POOLS:
        with study.pool(pool) as saved:
            for method in c.METHODS:
                for seed in (r.SEEDS if method in c.FAMILIES else (None,)):
                    for policy in r.POLICIES:
                        pred = (
                            saved.predictions(method, policy, seed)
                            if method in c.FAMILIES
                            else linear.predictions(pool, method, 42, policy)
                        )
                        ident = (
                            f"{pool}/{method}/"
                            + ("fold42" if seed is None else f"s{seed}")
                            + "/"
                            + policy
                        )
                        j = lookup[ident]
                        r.require(j not in filled, "Duplicate extracted model.")
                        filled.add(j)
                        for name, view in views.items():
                            head = view["endpoint"]["dimension"]
                            ix = view["positions"]
                            cols = view["label_indices"]
                            z = matrices[name]
                            z["gated"][j] = pred[head][ix][:, cols]
                            z["raw"][j] = pred[head + "_ungated"][ix][:, cols]
                            z["gate"][j] = pred["hate"][ix, 1]
            save(out / "archive_bindings" / f"{pool}.json", saved.store.read_bindings)
            saved_bytes += saved.store.bytes_read
        linear.release_pool()
        print(pool + ": saved outputs checked and aligned; no model run.", flush=True)
    r.require(len(filled) == len(models) == 306, "Incomplete extracted model views.")
    save(out / "models.json", models)
    for name, view in views.items():
        slug = m.slug(name)
        z = matrices[name]
        head = view["endpoint"]["dimension"]
        labels = view["endpoint"]["all_labels"]
        records = [
            {
                k: x.get(k)
                for k in (
                    "key",
                    "id",
                    "text",
                    "source",
                    "phase",
                    "raw",
                    "gate",
                    "hate_types",
                    "target_groups",
                    "severity",
                    "text_cluster",
                    "functional_cluster",
                )
            }
            for x in view["records"]
        ]
        save(
            out / (slug + "_metadata.json"),
            dict(
                endpoint=name,
                head=head,
                labels=labels,
                support=[labels.index(x) for x in view["endpoint"]["macro_labels"]],
                records=records,
                clusters=view["clusters"],
                positions=view["positions"],
                gold_support=view["endpoint"]["gold_support"],
            ),
        )
        m.array(out / (slug + "_arrays.npz"), **z)
        if name in r.PRIMARY[1:]:
            old = r.arrays_from_bytes((ERRORS / (slug + "_flow.npz")).read_bytes())
            r.require(
                all(np.array_equal(old[k], z[k]) for k in z),
                "Existing fine source changed.",
            )
        if name == "en_test/hate":
            old = e.old_arrays(name, {})
            r.require(
                np.array_equal(old["truth"], z["truth"])
                and np.array_equal(old["predictions"], z["gated"]),
                "Existing TEST decisions changed.",
            )
    save(out / "context.json", context.summary())
    save(
        out / "source_bindings.json",
        dict(study=study.bindings, linear=linear.bindings, context=context.bindings),
    )
    r.require("torch" not in sys.modules, "Unexpected training/weight library.")
    save(
        out / "complete.json",
        dict(
            status="complete",
            models=len(models),
            endpoints=len(views),
            archive_bytes_read=saved_bytes,
            existing_primary_arrays_unchanged=True,
            code_sha256=r.sha(Path(__file__)),
            outputs={
                str(p.relative_to(out)): r.sha(p) for p in out.rglob("*") if p.is_file()
            },
            training_calls=0,
            qwen_calls=0,
            weights_loaded=0,
        ),
    )
    print(
        "Extraction complete: eight frozen English endpoints, all 306 model views; original primary arrays unchanged.",
        flush=True,
    )


def load(name):
    out = BASE / "extraction"
    slug = m.slug(name)
    rec = r.read(out / "complete.json")
    for suffix in ("_metadata.json", "_arrays.npz"):
        r.checked_file(out / (slug + suffix), rec["outputs"][slug + suffix])
    return r.read(out / (slug + "_metadata.json")), r.arrays_from_bytes(
        (out / (slug + "_arrays.npz")).read_bytes()
    )


def binary(y, p):
    counts = e.binary_counts(y, p)
    metrics = e.binary_metrics(counts)

    return metrics, dict(
        fpr=not bool((np.asarray(y) == 0).any()),
        fnr=not bool((np.asarray(y) == 1).any()),
    )


def aggregate(models, values):
    return c.aggregate(models, np.asarray(values))[1]


def bootstrap_binary(models, y, p, clusters):

    unique, inv = np.unique(clusters, return_inverse=True)
    terms = []
    for j in range(len(p)):
        terms.extend(
            np.bincount(inv, weights=mask, minlength=len(unique))
            for mask in (
                p[j].astype(bool) & y.astype(bool),
                p[j].astype(bool) & ~y.astype(bool),
                ~p[j].astype(bool) & y.astype(bool),
                ~p[j].astype(bool) & ~y.astype(bool),
            )
        )
    terms = np.asarray(terms).T
    out = {k: [] for k in ("macro_f1", "accuracy", "fpr", "fnr")}
    rng = np.random.default_rng(20261020)
    digest = hashlib.sha256()
    for first in range(0, 10000, 64):
        picks = rng.integers(
            len(unique), size=(min(64, 10000 - first), len(unique)), dtype=np.int64
        )
        digest.update(picks.astype("<i8").tobytes())
        weights = np.asarray(
            [np.bincount(x, minlength=len(unique)) for x in picks], float
        )
        cc = (weights @ terms).reshape(len(picks), len(p), 4)
        metrics = e.binary_metrics(cc)
        for key in out:
            _, bb = c.aggregate(models, metrics[key])
            out[key].append(np.stack([bb[x] for x in sorted(bb)], axis=1))
    return {k: np.concatenate(v) for k, v in out.items()}, dict(
        keys=sorted(bb),
        clusters=len(unique),
        draws=10000,
        picks_sha256=digest.hexdigest(),
    )


if __name__ == "__main__":
    with r.offline(), threadpool_limits(limits=2):
        if sys.argv[1:] == ["freeze"]:
            freeze()
        elif sys.argv[1:] == ["extract"]:
            extract()
        else:
            raise SystemExit("Use freeze or extract.")
