import argparse
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import itertools
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.broad import metrics as main

reader = main.reader
c = reader.c
r = reader.r
scoring = reader.scoring
ROOT = reader.ROOT
BASE = ROOT / "experiments/astra/sq2_broad_robustness_review_20261104_v1"
PRIOR = ROOT / "experiments/astra/sq2_broad_context_review_20261104_v1"
CONDITIONS = reader.CONDITIONS
DRAWS = 10000
SEED = 20261008
require = reader.require
bindings = reader.bindings


def safe(v):
    if isinstance(v, np.ndarray):
        return safe(v.tolist())
    if isinstance(v, np.generic):
        return safe(v.item())
    if isinstance(v, dict):
        return {str(k): safe(w) for k, w in v.items()}
    if isinstance(v, (tuple, list)):
        return [safe(w) for w in v]
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()), "Output outside Part 7")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(c.encoded(safe(value)))


def array(path, **values):
    require(path.resolve().is_relative_to(BASE.resolve()), "Array outside Part 7")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def sources():
    paths = [
        Path(__file__),
        ROOT / "Thesis_Log/astra_result/astra_sq2_broad_analysis_plan_20261104_v1.md",
        ROOT
        / "Thesis_Log/astra_result/astra_remaining_work_before_writing_20261104_v149.md",
        PRIOR / "part6_complete.json",
        PRIOR / "final_review.json",
        PRIOR / "analysis_complete.json",
        reader.BASE / "part1_complete.json",
        reader.BASE / "reader_review.json",
        reader.BASE / "predictions.jsonl",
        reader.BASE / "endpoint_setup.json",
        reader.BASE / "comparison_map.json",
        reader.BASE / "item_bindings.json",
        reader.BASE / "population_setup.json",
        main.BASE / "part2_complete.json",
        main.BASE / "scores_complete.json",
        c.CORE / "dev_inputs.jsonl",
        c.PREP / "broad_roster.jsonl",
        c.PREP / "plan.json",
        reader.REPLAY,
        c.BASE / "scoring_contract_v1.json",
        r.BASE / "live/complete.json",
        r.PLAN,
        Path(reader.__file__),
        Path(scoring.__file__),
        Path(scoring.original.__file__),
        Path(main.__file__),
    ]
    for p in (main.BASE / "endpoints").iterdir():
        paths += [p / "encoded.npz", p / "results.json"]
    return bindings(paths)


def freeze():
    save(
        BASE / "analysis_contract.json",
        dict(
            status="part7_scope_recorded_before_sensitivity_scoring",
            created_utc=c.utc(),
            sources=sources(),
            prior_exposure="Parts 1–6 and their original outcomes are known. Metadata-only inspection found 163 larger-DEV inputs overlapping the original 1,295-input core. This is an analysis-time scope, not new preregistration.",
            required=[
                "All 40 original contrasts: natural source-stratified whole-cluster sensitivity or HateCheck item sensitivity.",
                "Common prospective 27-key replay exclusion for EVERY compared system including zero and encoders, fixed full-roster macro labels/family; unchanged endpoints explicitly retained.",
                "Separate one-ASCII-space German TEST hate exclusion, never substituted for replay removal.",
                "Own/common validity, original full-roster failure bounds, partial vote counts, parsing/repair history and fixed outer/inner variation.",
            ],
            added_sensitivities=[
                "All 40 contrasts restricted symmetrically to inputs with three valid inner responses in every compared Qwen outer repeat; encoders impose no vote requirement.",
                "All 40 contrasts restricted symmetrically to complete vote sets with zero repair calls in all compared Qwen repeats. Selection changes the population; no causal repair benefit claim.",
                "Core-overlap exclusion by original core key OR language plus normalized exact text, shared across every compared system. Only affected endpoints require new scoring; this does not make remaining data untouched.",
            ],
            statistics=dict(
                draws=DRAWS,
                seed=SEED,
                primary_family=4,
                primary_confidence=0.9875,
                supporting_confidence=0.95,
                fixed_labels="Original full-roster label sets, including zero-denominator labels, never recalculated after exclusions or draws.",
                pairing="Common-valid across all compared fixed settings; keep text/component dependence. Repeat F1s averaged, never pooled as observations.",
                multiplicity="Original primary/supporting status remains. Four-question adjusted intervals retained within each primary sensitivity; the collection of exploratory sensitivities is not a new jointly corrected confirmatory family.",
                reuse="Unchanged exclusions or one-stratum source resampling reuse original draws with explicit equivalence checks; no needless recomputation.",
            ),
            repeat_checks="All three own/common scores, all nine cross-repeat differences, leave-one-repeat-out means, field/gate-aware exact stability. Fixed repeat ranges are descriptive, not sampling intervals.",
            inner_checks="Agreement only among at least two valid responses, all fields and phases; separately retain one/zero-vote counts. Single-final-inner-response diagnostic uses fixed slots 0/1/2 on the all-condition/all-repeat complete-vote population. Responses may have been repaired; do not call this a measured one-call system.",
            rationale="Compare aggregated fields with the first valid response agreeing with the aggregate gate, whose rationale was copied. Field differences show that a copied rationale need not accompany identical fine labels; they do not measure rationale truth.",
            raw_scope="Reuse all 181 fixed Part 1 raw-review selections. Verify their original result hashes, reconstruct repair-chain validity/error categories and model metadata. Sample error proportions are not population estimates; do not audit the whole raw tree.",
            exposure="Original-core overlap, previously exposed DEV/TEST/HateCheck, existing exact/near-duplicate qualifications, fixed KB/model/repeats and server alias versus immutable weights.",
            prohibited=[
                "network",
                "new Qwen calls",
                "retrieval/index queries",
                "model loading",
                "training",
                "new inference",
                "source edits",
                "server actions",
                "gold relabeling",
                "writer edits",
            ],
            deferred=[
                "Actual token/time/resource ledger (Part 8)",
                "Standalone and combined SQ2 synthesis",
            ],
        ),
    )


def canonical(v, field):
    value = v[field]
    return tuple(sorted(set(value))) if isinstance(value, list) else value


def agree(votes, field):
    vals = [canonical(v, field) for v in votes if v is not None]
    n = len(vals)
    return dict(
        valid=n,
        informative=n >= 2,
        unanimous=(len(set(vals)) == 1) if n >= 2 else None,
        agreeing_pairs=sum(a == b for a, b in itertools.combinations(vals, 2)),
        pairs=n * (n - 1) // 2,
    )


def compact_metadata():
    contract = c.read(BASE / "analysis_contract.json")
    r.verify(contract["sources"])
    groups, _ = scoring.scientific_rosters("broad")
    roster = {v["key"]: v for rr in groups.values() for v in rr}
    core = list(reader.rows(c.CORE / "dev_inputs.jsonl"))
    corekeys = {v["key"] for v in core}
    coretexts = {(v["lang"], reader.gold.normalized(v["text"])) for v in core}
    overlapping = [
        dict(
            key=v["key"],
            phase=v["phase"],
            key_match=v["key"] in corekeys,
            text_match=(v["lang"], reader.gold.normalized(v["text"])) in coretexts,
        )
        for v in roster.values()
        if v["key"] in corekeys
        or (v["lang"], reader.gold.normalized(v["text"])) in coretexts
    ]
    require(roster[c.EMPTY_KEY]["text"] == " ", "Whitespace input changed")
    keys = list(roster)
    ki = {k: i for i, k in enumerate(keys)}
    settings = [(co, j) for co in CONDITIONS for j in range(3)]
    si = {v: i for i, v in enumerate(settings)}
    nv = np.full((9, len(keys)), -1, np.int8)
    repairs = np.zeros_like(nv)
    ties = np.zeros_like(nv, bool)
    normal = np.zeros_like(nv, np.int16)
    gates = np.zeros_like(nv, bool)
    cells = defaultdict(Counter)
    missing = defaultdict(Counter)
    agreement = defaultdict(Counter)
    rationale = defaultdict(Counter)
    nullids = []
    partialids = []
    whitespace = []
    epmap = c.read(reader.BASE / "endpoint_setup.json")
    vote_arrays = {}
    for name, ep in epmap.items():
        vote_arrays[name] = dict(
            ids=ep["scorable_ids"],
            labels=ep["all_labels"],
            macro=ep["macro_labels"],
            settings=settings,
            predicted=np.zeros(
                (9, 3, len(ep["scorable_ids"]), len(ep["all_labels"])), np.int8
            ),
            valid=np.zeros((9, 3, len(ep["scorable_ids"])), bool),
            gate=np.zeros((9, 3, len(ep["scorable_ids"])), bool),
        )
    memberships = defaultdict(list)
    for name, v in vote_arrays.items():
        for j, key in enumerate(v["ids"]):
            memberships[key].append((name, j))
    total = Counter()
    n = 0
    for row in reader.rows(reader.BASE / "predictions.jsonl"):
        k = ki[row["key"]]
        s = si[row["condition"], row["repeat"]]
        require(nv[s, k] == -1, "Duplicate cached identity")
        nv[s, k] = row["n_valid"]
        repairs[s, k] = row["repairs"]
        ties[s, k] = row["uncertain"]
        normal[s, k] = row["gate_normalised"]
        gates[s, k] = bool(row["result"] and row["result"]["hate"])
        counts = dict(
            predictions=1,
            nulls=int(row["n_valid"] == 0),
            partial=int(row["n_valid"] in (1, 2)),
            **{"valid_" + str(row["n_valid"]): 1},
            repaired_predictions=int(row["repairs"] > 0),
            repairs=row["repairs"],
            parse_failures=row["parse_failures"],
            gate_ties=int(row["uncertain"]),
            gate_normalised=row["gate_normalised"],
            normalisations=row["normalisations"]
        )
        cells[row["phase"], row["condition"], row["repeat"]].update(counts)
        total.update(counts)
        truth = roster[row["key"]]["gate"]
        stratum = (row["phase"], row["source"], row["condition"], str(truth))
        missing[stratum].update(counts)
        if row["n_valid"] == 0:
            nullids.append(
                dict(
                    key=row["key"],
                    condition=row["condition"],
                    repeat=row["repeat"],
                    phase=row["phase"],
                )
            )
        elif row["n_valid"] < 3:
            partialids.append(
                dict(
                    key=row["key"],
                    condition=row["condition"],
                    repeat=row["repeat"],
                    n_valid=row["n_valid"],
                )
            )
        if row["key"] == c.EMPTY_KEY:
            whitespace.append(row)
        for field in reader.FIELDS:
            a = agree(row["votes"], field)
            co = agreement[row["phase"], row["condition"], field]
            co.update(
                predictions=1,
                informative=int(a["informative"]),
                unanimous=int(a["unanimous"] is True),
                agreeing_pairs=a["agreeing_pairs"],
                pairs=a["pairs"],
                **{"valid_" + str(a["valid"]): 1}
            )
        if row["result"] is not None:
            chosen = next(
                v
                for v in row["votes"]
                if v is not None and v["hate"] == row["result"]["hate"]
            )
            diff = [
                f
                for f in reader.FIELDS
                if canonical(chosen, f) != canonical(row["result"], f)
            ]
            rationale[row["phase"], row["condition"]].update(
                valid_predictions=1,
                any_field_differs=int(bool(diff)),
                **{f: 1 for f in diff}
            )
        for name, j in memberships[row["key"]]:
            ep = epmap[name]
            a = vote_arrays[name]
            for vindex, vote in enumerate(row["votes"]):
                if vote is None:
                    continue
                a["valid"][s, vindex, j] = True
                a["gate"][s, vindex, j] = vote["hate"]
                a["predicted"][s, vindex, j] = scoring.original.bits(
                    vote[ep["dimension"]], ep["dimension"], ep["all_labels"]
                )
        n += 1
    require(n == 194643 and (nv >= 0).all(), "Metadata matrix incomplete")
    old = c.read(reader.BASE / "reader_review.json")["counts"]
    for field in (
        "predictions",
        "nulls",
        "repairs",
        "parse_failures",
        "gate_ties",
        "gate_normalised",
        "normalisations",
    ):
        require(total[field] == old[field], "Audit total differs " + field)
    for j in range(4):
        require(
            total["valid_" + str(j)] == old["valid_votes_" + str(j)],
            "Vote histogram differs",
        )
    array(
        BASE / "prediction_metadata.npz",
        n_valid=nv,
        repairs=repairs,
        ties=ties,
        gate_normalised=normal,
        gate=gates,
    )
    save(
        BASE / "metadata_index.json",
        dict(
            keys=keys,
            settings=settings,
            core_overlap=overlapping,
            core_inputs=len(core),
            replay=c.read(reader.REPLAY)["exclusion_roster"],
            whitespace_key=c.EMPTY_KEY,
        ),
    )
    save(
        BASE / "validity_inventory.json",
        dict(
            total=total,
            cells=[
                dict(phase=k[0], condition=k[1], repeat=k[2], counts=v)
                for k, v in sorted(cells.items())
            ],
            source_gold=[
                dict(phase=k[0], source=k[1], condition=k[2], gold=k[3], counts=v)
                for k, v in sorted(missing.items())
            ],
            null_predictions=nullids,
            null_distinct_inputs=len({v["key"] for v in nullids}),
            partial_predictions=partialids,
            inner_agreement=[
                dict(phase=k[0], condition=k[1], field=k[2], counts=v)
                for k, v in sorted(agreement.items())
            ],
            rationale_source_field_differences=[
                dict(phase=k[0], condition=k[1], counts=v)
                for k, v in sorted(rationale.items())
            ],
        ),
    )
    save(
        BASE / "whitespace_saved_answers.json",
        dict(gold=roster[c.EMPTY_KEY], predictions=whitespace),
    )
    outputs = []
    for name, a in vote_arrays.items():
        p = BASE / "inner_votes" / (main.slug(name) + ".npz")
        array(p, predicted=a["predicted"], valid=a["valid"], gate=a["gate"])
        outputs.append(p)
    save(
        BASE / "metadata_complete.json",
        dict(
            status="saved_vote_metadata_assembled",
            sources=bindings(
                [
                    BASE / "analysis_contract.json",
                    reader.BASE / "predictions.jsonl",
                    c.CORE / "dev_inputs.jsonl",
                ]
            ),
            outputs=bindings(
                outputs
                + [
                    BASE / n
                    for n in (
                        "prediction_metadata.npz",
                        "metadata_index.json",
                        "validity_inventory.json",
                        "whitespace_saved_answers.json",
                    )
                ]
            ),
            new_Qwen_calls=0,
        ),
    )
    print(
        "Saved metadata:",
        n,
        "predictions;",
        len(overlapping),
        "core overlaps;",
        total["nulls"],
        "invalid predictions.",
        flush=True,
    )


def endpoint(name):
    result = c.read(main.BASE / "endpoints" / main.slug(name) / "results.json")
    ep = result["endpoint"]
    with np.load(ROOT / result["encoded_path"], allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    settings = [tuple(v) for v in result["settings"]]
    terms = {}
    for j, ident in enumerate(settings):
        terms[ident] = dict(
            ids=ep["scorable_ids"],
            labels=ep["all_labels"],
            macro_indices=[ep["all_labels"].index(v) for v in ep["macro_labels"]],
            gold=arrays["gold"].astype(np.int64),
            pred=arrays["predicted"][j].astype(np.int64),
            counts=arrays["counts"][j].astype(np.int64),
            valid=arrays["valid"][j],
        )
    return result, terms


def restrict(v, keep):
    require(
        keep.dtype == bool and len(keep) == len(v["ids"]), "Invalid sensitivity mask"
    )
    return v | dict(
        ids=[key for key, on in zip(v["ids"], keep) if on],
        **{k: v[k][keep] for k in ("gold", "pred", "counts", "valid")}
    )


def capture(
    left,
    right,
    clusters,
    sources,
    *,
    stratify=False,
    family=None,
    draws=DRAWS,
    seed=SEED
):
    captured = []
    cache = {}

    def quantile(v, q, *args, **kwargs):
        a = np.asarray(v, float)
        if captured:
            require(np.array_equal(a, captured[0]), "Intervals use different draws")
        else:
            captured.append(a.copy())
        return np.quantile(a, q, *args, **kwargs)

    def weighted(expr, w, t, **kwargs):
        require(
            expr == "bg,sgk->bsk"
            and w.dtype == t.dtype == np.int64
            and w.min() >= 0
            and t.min() >= 0,
            "Unexpected frozen count sum",
        )
        require(
            int(w.sum(1).max()) * int(t.max()) < 2**52, "Exact float64 bound exceeded"
        )
        if cache.get("terms") is not t:
            cache.update(terms=t, floating=t.astype(float))
        out = np.einsum(expr, w.astype(float), cache["floating"], optimize=True)
        require(np.array_equal(out, np.rint(out)), "Noninteger weighted sum")
        return out.astype(np.int64)

    proxy = SimpleNamespace(**dict(vars(np), quantile=quantile, einsum=weighted))
    result = reader.clone(scoring.contrast, np=proxy)(
        left,
        right,
        clusters,
        sources=sources,
        stratify=stratify,
        family=family,
        draws=draws,
        seed=seed,
    )
    values = captured[0] if captured else np.empty(0, float)
    require(len(values) in (0, draws), "Incomplete draw capture")
    return result, values


def interval_relation(v):
    return (
        "not_scorable"
        if v is None
        else (
            "above_zero"
            if v[0] > 0
            else "below_zero" if v[1] < 0 else "crosses_or_touches_zero"
        )
    )


def score():
    contract = c.read(BASE / "analysis_contract.json")
    r.verify(contract["sources"])
    meta = c.read(BASE / "metadata_complete.json")
    r.verify(meta["outputs"])
    test = c.read(BASE / "tests.json")
    require(test["status"] == "passed", "Tests required")
    r.verify(test["sources"])
    index = c.read(BASE / "metadata_index.json")
    ki = {v: i for i, v in enumerate(index["keys"])}
    si = {tuple(v): i for i, v in enumerate(index["settings"])}
    with np.load(BASE / "prediction_metadata.npz", allow_pickle=False) as z:
        nvalid = z["n_valid"]
        repairs = z["repairs"]
    replay = {v["key"] for v in index["replay"]}
    core = {v["key"] for v in index["core_overlap"]}
    results = []
    repetitions = []
    own = []
    draw_count = 0
    for name in sorted(c.read(reader.BASE / "endpoint_setup.json")):
        original, terms = endpoint(name)
        ep = original["endpoint"]
        ids = ep["scorable_ids"]
        clusters = ep["row_clusters"]
        src = ep["row_sources"]
        positions = [ki[k] for k in ids]
        for co, rows in original["own_valid"].items():
            own.append(
                dict(
                    endpoint=name,
                    condition=co,
                    summary=rows,
                    common_nine=original["three_condition_common"]["per_condition"].get(
                        co
                    ),
                )
            )
        for pair in original["contrasts"]:
            left = pair["left"]
            right = pair["right"]
            ident = [(co, j) for co in (left, right) for j in range(3)]
            allterms = [terms[v] for v in ident]
            common = np.logical_and.reduce([v["valid"] for v in allterms])
            scores = [main.f1_on_mask(v, common) for v in allterms]
            differences = (
                None
                if not ep["macro_labels"]
                else np.subtract.outer(scores[:3], scores[3:])
            )
            repetitions.append(
                dict(
                    endpoint=name,
                    left=left,
                    right=right,
                    role=pair["role"],
                    n_common_valid=int(common.sum()),
                    scores=scores,
                    differences=differences,
                    positive_pairs=(
                        None if differences is None else int((differences > 0).sum())
                    ),
                    negative_pairs=(
                        None if differences is None else int((differences < 0).sum())
                    ),
                    range=(
                        None
                        if differences is None
                        else [float(differences.min()), float(differences.max())]
                    ),
                    leave_one_each=(
                        None
                        if differences is None
                        else [
                            [
                                float(
                                    np.mean([scores[k] for k in range(3) if k != i])
                                    - np.mean(
                                        [scores[3 + k] for k in range(3) if k != j]
                                    )
                                )
                                for j in range(3)
                            ]
                            for i in range(3)
                        ]
                    ),
                )
            )
            qindices = [si[v] for v in ident if v[0] in CONDITIONS]
            complete = (nvalid[qindices][:, positions] == 3).all(0)
            unrepaired = complete & (repairs[qindices][:, positions] == 0).all(0)
            scenarios = [
                ("required_sampling", np.ones(len(ids), bool)),
                ("replay_exclusion", np.array([k not in replay for k in ids])),
                ("complete_votes", complete),
                ("unrepaired_complete", unrepaired),
                ("core_exclusion", np.array([k not in core for k in ids])),
            ]
            if name == "de_test/hate":
                scenarios.append(
                    ("whitespace_exclusion", np.array([k != c.EMPTY_KEY for k in ids]))
                )
            for scenario, keep in scenarios:
                selected = [restrict(v, keep) for v in allterms]
                cc = [k for k, on in zip(clusters, keep) if on]
                ss = [k for k, on in zip(src, keep) if on]
                stratify = (
                    scenario == "required_sampling"
                    and ep["phase"] != "en_hatecheck_eval"
                )
                if (
                    scenario == "required_sampling"
                    and ep["phase"] == "en_hatecheck_eval"
                ):
                    cc = [k for k, on in zip(ids, keep) if on]
                equivalent = bool(keep.all()) and (
                    scenario != "required_sampling" or (stratify and len(set(src)) == 1)
                )
                if equivalent:
                    value = {
                        k: v
                        for k, v in pair.items()
                        if k not in ("interval_relation", "failure_bound_relation")
                    }
                    draws = None
                    value["source_stratified"] = stratify
                    if stratify and "cluster_strata" in value:
                        value["cluster_strata"] = {str(src[0]): value["n_clusters"]}
                else:
                    value, draws = capture(
                        selected[:3],
                        selected[3:],
                        cc,
                        ss,
                        stratify=stratify,
                        family=pair["family_size"],
                    )
                entry = dict(
                    endpoint=name,
                    left=left,
                    right=right,
                    role=pair["role"],
                    scenario=scenario,
                    original_difference=pair["mean_difference"],
                    original_interval=(
                        pair["family_interval"]
                        if pair["role"] == "primary"
                        else pair["ci95"]
                    ),
                    original_common_valid=pair["n_common_valid"],
                    removed_eligible_ids=[k for k, on in zip(ids, keep) if not on],
                    remaining_eligible=int(keep.sum()),
                    removed_common_valid=int((common & ~keep).sum()),
                    fixed_macro_labels=ep["macro_labels"],
                    remaining_gold_support=dict(
                        zip(ep["all_labels"], selected[0]["gold"].sum(0))
                    ),
                    result=value,
                    reused_original_draws=equivalent,
                    selection_note="All conditions share the same filtered input roster. Original results remain primary; vote/repair restrictions select different inputs.",
                )
                if draws is not None and len(draws):
                    path = (
                        BASE
                        / "draws"
                        / main.slug(name)
                        / (left + "_minus_" + right + "__" + scenario + ".npy")
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as f:
                        np.save(f, draws, allow_pickle=False)
                    entry["draws_path"] = c.rel(path)
                    entry["draws_sha256"] = c.sha(path)
                    draw_count += len(draws)
                elif equivalent and "draws_path" in pair:
                    entry.update(
                        draws_path=pair["draws_path"], draws_sha256=pair["draws_sha256"]
                    )
                ci = (
                    value["family_interval"]
                    if pair["role"] == "primary"
                    else value["ci95"]
                )
                entry["interval_relation"] = interval_relation(ci)
                entry["original_relation"] = interval_relation(
                    entry["original_interval"]
                )
                entry["difference_change"] = (
                    None
                    if value["mean_difference"] is None
                    else value["mean_difference"] - pair["mean_difference"]
                )
                results.append(entry)
            print("Sensitivity checks:", name, left, "minus", right, flush=True)
    save(BASE / "sensitivities.json", results)
    save(BASE / "repeat_comparisons.json", repetitions)
    save(BASE / "original_coverage_and_bounds.json", own)
    outputs = [
        BASE / n
        for n in (
            "sensitivities.json",
            "repeat_comparisons.json",
            "original_coverage_and_bounds.json",
        )
    ] + list((BASE / "draws").rglob("*.npy"))
    save(
        BASE / "scores_complete.json",
        dict(
            status="part7_sensitivities_scored_review_pending",
            contrasts=len(repetitions),
            sensitivity_entries=len(results),
            new_draws=draw_count,
            sources=bindings(
                [
                    BASE / "analysis_contract.json",
                    BASE / "metadata_complete.json",
                    BASE / "tests.json",
                    Path(__file__),
                ]
            ),
            outputs=bindings(outputs),
            new_Qwen_calls=0,
        ),
    )
    print(
        "Sensitivity computation complete:",
        len(results),
        "entries,",
        draw_count,
        "new draws.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["freeze", "metadata", "score"], required=True
    )
    args = parser.parse_args()
    with r.original.offline(), threadpool_limits(limits=2):
        {"freeze": freeze, "metadata": compact_metadata, "score": score}[args.mode]()
