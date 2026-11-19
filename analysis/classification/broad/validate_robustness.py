from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from analysis.classification.broad import robustness as x

checks = 0
maximum = 0.0


def equal(a, b, where=""):
    global checks, maximum
    a = x.safe(a)
    b = x.safe(b)
    if isinstance(a, dict):
        x.require(
            isinstance(b, dict) and a.keys() == b.keys(), "Different keys " + where
        )
        for k in a:
            equal(a[k], b[k], where + "/" + k)
    elif isinstance(a, list):
        x.require(isinstance(b, list) and len(a) == len(b), "Different list " + where)
        for j, (u, v) in enumerate(zip(a, b)):
            equal(u, v, where + "/" + str(j))
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        x.require(
            isinstance(b, (float, int)) and not isinstance(b, bool),
            "Different numeric type " + where,
        )
        d = abs(a - b)
        maximum = max(maximum, d)
        checks += 1
        x.require(
            d <= 1e-12, "Numerical mismatch " + where + ": " + str(a) + " / " + str(b)
        )
    else:
        checks += 1
        x.require(a == b, "Different value " + where + ": " + str(a) + " / " + str(b))


def macro(tp, pred, actual, idx):
    denom = pred + actual
    f = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return f[..., idx].mean(-1) if idx else np.full(f.shape[:-1], np.nan)


def fscore(y, p, idx):
    return macro((p * y).sum(-2), p.sum(-2), np.broadcast_to(y, p.shape).sum(-2), idx)


def failure_bounds(y, p, valid, idx):
    if not idx:
        return [None, None]
    yy = y[valid]
    pp = p[valid]
    tp = (yy * pp).sum(0)
    pc = pp.sum(0)
    ac = yy.sum(0)
    unknown = y[~valid]
    low = macro(tp, pc + (1 - unknown).sum(0), ac + unknown.sum(0), idx)
    added = unknown.sum(0)
    high = macro(tp + added, pc + added, ac + added, idx)
    return [float(low), float(high)]


def draws_independent(y, p, idx, clusters, sources, stratify):
    unique = sorted(set(clusters))
    mapping = {k: i for i, k in enumerate(unique)}
    moments = np.stack((p * y, p, np.broadcast_to(y, p.shape)), axis=-1)
    grouped = np.zeros((len(unique), p.shape[0], p.shape[2], 3), np.int64)
    np.add.at(grouped, [mapping[k] for k in clusters], moments.transpose(1, 0, 2, 3))
    strata = defaultdict(list)
    for i, k in enumerate(unique):
        strata[sources[k] if stratify else ("all",)].append(i)
    flat = grouped.reshape(len(unique), -1).astype(float)
    rng = np.random.default_rng(x.SEED)
    out = []
    for start in range(0, x.DRAWS, 64):
        n = min(64, x.DRAWS - start)
        sample = np.concatenate(
            [
                rng.choice(v, (n, len(v)), replace=True)
                for _, v in sorted(strata.items())
            ],
            axis=1,
        )
        w = np.stack([np.bincount(v, minlength=len(unique)) for v in sample])
        sums = (w @ flat).reshape(n, p.shape[0], p.shape[2], 3)
        ff = macro(sums[..., 0], sums[..., 1], sums[..., 2], idx)
        out.append(ff[:, :3].mean(1) - ff[:, 3:].mean(1))
    return np.concatenate(out), {"|".join(k): len(v) for k, v in strata.items()}


def run():
    done = x.c.read(x.BASE / "scores_complete.json")
    x.r.verify(done["sources"])
    x.r.verify(done["outputs"])
    meta = x.c.read(x.BASE / "metadata_index.json")
    keyi = {k: i for i, k in enumerate(meta["keys"])}
    seti = {tuple(k): i for i, k in enumerate(meta["settings"])}
    with np.load(x.BASE / "prediction_metadata.npz", allow_pickle=False) as z:
        nv = z["n_valid"]
        rep = z["repairs"]
    replay = {v["key"] for v in x.c.read(x.reader.REPLAY)["exclusion_roster"]}
    originalcore = list(x.reader.rows(x.c.CORE / "dev_inputs.jsonl"))
    corekeys = {v["key"] for v in originalcore}
    coretexts = {(v["lang"], x.reader.gold.normalized(v["text"])) for v in originalcore}
    roster = {
        v["key"]: v
        for rr in x.scoring.scientific_rosters("broad")[0].values()
        for v in rr
    }
    core = {
        k
        for k, v in roster.items()
        if k in corekeys
        or (v["lang"], x.reader.gold.normalized(v["text"])) in coretexts
    }
    equal(core, {v["key"] for v in meta["core_overlap"]}, "Core members")
    items = x.c.read(x.BASE / "sensitivities.json")
    verified = []
    drawtotal = 0
    byendpoint = defaultdict(list)
    for v in items:
        byendpoint[v["endpoint"]].append(v)
    for name, rows in sorted(byendpoint.items()):
        old, terms = x.endpoint(name)
        ep = old["endpoint"]
        ids = ep["scorable_ids"]
        positions = [keyi[k] for k in ids]
        idx = [ep["all_labels"].index(v) for v in ep["macro_labels"]]
        for row in rows:
            selected = [(co, j) for co in (row["left"], row["right"]) for j in range(3)]
            q = [seti[k] for k in selected if k[0] in x.CONDITIONS]
            full_y = terms[selected[0]]["gold"]
            full_p = np.stack([terms[k]["pred"] for k in selected])
            full_valid = np.stack([terms[k]["valid"] for k in selected])
            original_common = full_valid.all(0)
            scenario = row["scenario"]
            if scenario == "required_sampling":
                keep = np.ones(len(ids), bool)
            elif scenario == "replay_exclusion":
                keep = np.array([k not in replay for k in ids])
            elif scenario == "core_exclusion":
                keep = np.array([k not in core for k in ids])
            elif scenario == "whitespace_exclusion":
                keep = np.array([k != x.c.EMPTY_KEY for k in ids])
            elif scenario == "complete_votes":
                keep = (nv[q][:, positions] == 3).all(0)
            elif scenario == "unrepaired_complete":
                keep = ((nv[q][:, positions] == 3) & (rep[q][:, positions] == 0)).all(0)
            else:
                raise ValueError("Unexpected scenario")
            equal(
                row["removed_eligible_ids"],
                [k for k, on in zip(ids, keep) if not on],
                name + "/" + scenario + "/excluded",
            )
            equal(row["remaining_eligible"], int(keep.sum()))
            equal(row["removed_common_valid"], int((original_common & ~keep).sum()))
            equal(row["fixed_macro_labels"], ep["macro_labels"])
            y = full_y[keep]
            p = full_p[:, keep]
            valid = full_valid[:, keep]
            keptids = [k for k, on in zip(ids, keep) if on]
            common = valid.all(0)
            out = row["result"]
            equal(row["remaining_gold_support"], dict(zip(ep["all_labels"], y.sum(0))))
            equal(out["common_valid_ids"], [k for k, on in zip(keptids, common) if on])
            equal(out["excluded_ids"], [k for k, on in zip(keptids, common) if not on])
            equal(out["n_common_valid"], int(common.sum()))
            bounds = np.array(
                [failure_bounds(y, p[j], valid[j], idx) for j in range(6)]
            )
            expectedbounds = (
                [None, None]
                if not idx
                else [
                    float(bounds[:3, 0].mean() - bounds[3:, 1].mean()),
                    float(bounds[:3, 1].mean() - bounds[3:, 0].mean()),
                ]
            )
            equal(
                out["full_roster_failure_difference_bounds"],
                expectedbounds,
                name + "/" + scenario + "/failure bounds",
            )
            if not idx or not common.any():
                equal(out["mean_difference"], None)
                continue
            fs = fscore(y[common], p[:, common], idx)
            point = float(fs[:3].mean() - fs[3:].mean())
            equal(out["common_f1_by_setting"], fs)
            equal(out["mean_difference"], point)
            primary = row["role"] == "primary"
            family = 4 if primary else None
            equal(out["family_size"], family)
            unitids = (
                ids
                if scenario == "required_sampling"
                and ep["phase"] == "en_hatecheck_eval"
                else ep["row_clusters"]
            )
            retained = [k for k, on in zip(unitids, keep) if on]
            cc = [k for k, on in zip(retained, common) if on]
            stratify = (
                scenario == "required_sampling" and ep["phase"] != "en_hatecheck_eval"
            )
            members = defaultdict(set)
            for k, source, on in zip(unitids, ep["row_sources"], keep):
                if on:
                    members[k].add(source)
            sourcegroups = {k: tuple(sorted(v)) for k, v in members.items()}
            equal(out["n_clusters"], len(set(cc)))
            equal(out["source_stratified"], stratify)
            if row["reused_original_draws"]:
                x.require(
                    keep.all()
                    and (
                        scenario != "required_sampling"
                        or stratify
                        and len(set(ep["row_sources"])) == 1
                    ),
                    "Invalid reuse shortcut",
                )
                x.require(
                    x.c.sha(x.ROOT / row["draws_path"]) == row["draws_sha256"],
                    "Reused original draws changed",
                )
                draws = np.load(x.ROOT / row["draws_path"], allow_pickle=False)
            else:
                draws, strata = draws_independent(
                    y[common], p[:, common], idx, cc, sourcegroups, stratify
                )
                saved = np.load(x.ROOT / row["draws_path"], allow_pickle=False)
                equal(draws, saved, name + "/" + scenario + "/all draws")
                equal(out["cluster_strata"], strata)
                drawtotal += len(draws)
            equal(out["ci95"], np.percentile(draws, [2.5, 97.5]))
            equal(
                out["family_interval"],
                np.percentile(draws, [0.625, 99.375]) if primary else None,
            )
            chosen = out["family_interval"] if primary else out["ci95"]
            equal(row["interval_relation"], x.interval_relation(chosen))
            equal(row["difference_change"], point - row["original_difference"])
            verified.append(
                dict(
                    endpoint=name,
                    right=row["right"],
                    scenario=scenario,
                    new_draws=not row["reused_original_draws"],
                    n=int(common.sum()),
                )
            )
        print("Independently checked sensitivity samples/draws:", name, flush=True)
    equal(drawtotal, done["new_draws"])
    equal(len(items), done["sensitivity_entries"])
    repeats = x.c.read(x.BASE / "repeat_comparisons.json")
    for v in repeats:
        original, terms = x.endpoint(v["endpoint"])
        selected = [(co, j) for co in (v["left"], v["right"]) for j in range(3)]
        mask = np.stack([terms[k]["valid"] for k in selected]).all(0)
        idx = terms[selected[0]]["macro_indices"]
        yy = terms[selected[0]]["gold"][mask]
        pp = np.stack([terms[k]["pred"][mask] for k in selected])
        scores = fscore(yy, pp, idx)
        equal(v["scores"], scores)
        d = np.subtract.outer(scores[:3], scores[3:]) if idx else None
        equal(v["differences"], d)
        if idx:
            equal(v["range"], [d.min(), d.max()])
            equal(v["positive_pairs"], int((d > 0).sum()))
            equal(v["negative_pairs"], int((d < 0).sum()))
            equal(
                v["leave_one_each"],
                [
                    [
                        np.delete(scores[:3], i).mean()
                        - np.delete(scores[3:], j).mean()
                        for j in range(3)
                    ]
                    for i in range(3)
                ],
            )
    x.save(
        x.BASE / "independent_sensitivity_verification.json",
        dict(
            status="all_sensitivity_masks_points_bounds_families_and_new_draws_passed",
            checked_values=checks,
            maximum_difference=maximum,
            new_draws_verified=drawtotal,
            sensitivity_entries=len(items),
            compared_repeat_grids=len(repeats),
            details=verified,
            sources=x.bindings([Path(__file__), x.BASE / "scores_complete.json"]),
            new_Qwen_calls=0,
        ),
    )


if __name__ == "__main__":
    with x.r.original.offline(), x.threadpool_limits(limits=2):
        run()
