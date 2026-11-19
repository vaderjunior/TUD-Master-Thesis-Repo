from collections import Counter
from pathlib import Path
import itertools
import numpy as np
from analysis.classification.broad import robustness as x


def outcome(y, p, g, kind, macro):
    yy = y.astype(bool)
    pp = p.astype(bool)
    counts = np.stack((pp & yy, pp & ~yy, ~pp & yy), axis=-1).sum(-3)
    exact = (pp == yy).all(-1)
    if kind not in ("hate", "legal"):
        exact &= g
    return dict(f1=x.scoring.original.macro(counts, macro), exact=exact)


def diagnostics():
    done = x.c.read(x.BASE / "metadata_complete.json")
    x.r.verify(done["outputs"])
    index = x.c.read(x.BASE / "metadata_index.json")
    ki = {k: i for i, k in enumerate(index["keys"])}
    with np.load(x.BASE / "prediction_metadata.npz", allow_pickle=False) as z:
        meta = {k: z[k].copy() for k in z.files}
    output = []
    strata = []
    core = []
    for name, ep in sorted(x.c.read(x.reader.BASE / "endpoint_setup.json").items()):
        old, terms = x.endpoint(name)
        ii = [ki[k] for k in ep["scorable_ids"]]
        macro = [ep["all_labels"].index(v) for v in ep["macro_labels"]]
        kind = ep["dimension"]
        keys = [(co, j) for co in x.CONDITIONS for j in range(3)]
        y = terms[keys[0]]["gold"]
        p = np.stack([terms[k]["pred"] for k in keys])
        valid = np.stack([terms[k]["valid"] for k in keys])
        g = meta["gate"][:, ii]
        with np.load(
            x.BASE / "inner_votes" / (x.main.slug(name) + ".npz"), allow_pickle=False
        ) as z:
            vp = z["predicted"]
            vv = z["valid"]
            vg = z["gate"]
        common = valid.all(0)
        complete = (meta["n_valid"][:, ii] == 3).all(0)
        x.require(
            np.array_equal(complete, vv.all((0, 1))) and np.all(~complete | common),
            "Inner validity alignment differs",
        )
        overlap = {v["key"] for v in index["core_overlap"]}
        om = np.array([k in overlap for k in ep["scorable_ids"]])
        core.append(
            dict(
                endpoint=name,
                eligible=len(ii),
                core_overlap=int(om.sum()),
                overlap_common_nine=int((om & common).sum()),
                keys=[k for k, on in zip(ep["scorable_ids"], om) if on],
            )
        )
        for j, co in enumerate(x.CONDITIONS):
            sel = slice(j * 3, j * 3 + 3)
            a = outcome(y[common], p[sel][:, common], g[sel][:, common], kind, macro)
            exact = a["exact"]
            identical = (p[sel][:, common] == p[j * 3, common]).all((0, 2))
            if kind not in ("hate", "legal"):
                identical &= (g[sel][:, common] == g[j * 3, common]).all(0)
            stable = dict(
                n_common_nine=int(common.sum()),
                same_answer_all_repeats=int(identical.sum()),
                same_wrong_answer=int((identical & ~exact.any(0)).sum()),
                all_repeats_correct=int(exact.all(0).sum()),
                all_repeats_wrong=int((~exact.any(0)).sum()),
                variable_correctness=int((exact.any(0) & ~exact.all(0)).sum()),
                correct_repeats_hist=dict(Counter(exact.sum(0).tolist())),
                per_repeat_f1=a["f1"],
                per_repeat_exact=exact.mean(1),
            )
            one = outcome(
                y[complete],
                vp[sel][:, :, complete],
                vg[sel][:, :, complete],
                kind,
                macro,
            )
            agg = outcome(
                y[complete], p[sel][:, complete], g[sel][:, complete], kind, macro
            )
            synth = (vp[sel][:, :, complete] != p[sel][:, None, complete]).any(-1)
            if kind not in ("hate", "legal"):
                synth |= vg[sel][:, :, complete] != g[sel][:, None, complete]
            gained = agg["exact"] & ~one["exact"].any(1)
            lost = ~agg["exact"] & one["exact"].all(1)
            x.require(not lost.any(), "All correct votes yielded wrong delivered field")
            single = dict(
                n_complete_all_conditions=int(complete.sum()),
                removed_from_common_nine=int((common & ~complete).sum()),
                fixed_slot_f1=one["f1"],
                mean_single_f1=float(np.mean(one["f1"])) if macro else None,
                fixed_slot_exact=one["exact"].mean(2),
                mean_single_exact=float(one["exact"].mean()),
                aggregate_f1=agg["f1"],
                mean_aggregate_f1=float(np.mean(agg["f1"])) if macro else None,
                aggregate_exact=agg["exact"].mean(1),
                mean_aggregate_exact=float(agg["exact"].mean()),
                aggregate_correct_all_votes_wrong_per_repeat=gained.sum(1),
                aggregate_wrong_all_votes_correct_per_repeat=lost.sum(1),
                aggregate_answer_matches_no_single_vote_per_repeat=synth.all(1).sum(1),
                note="Each cell is one fixed saved final inner response, possibly after repairs. Same complete 27-response population across all three conditions; no new one-call system or pooled-vote inference.",
            )
            output.append(
                dict(endpoint=name, condition=co, outer=stable, single_inner=single)
            )
            for family, values in [
                ("valid_votes", (0, 1, 2, 3)),
                ("repairs", ("none", "some")),
                ("gate_tie", (False, True)),
                ("gate_normalized", (False, True)),
            ]:
                for stratum in values:
                    rr = []
                    for repeat in range(3):
                        s = j * 3 + repeat
                        if family == "valid_votes":
                            mask = meta["n_valid"][s, ii] == stratum
                        elif family == "repairs":
                            mask = (
                                (meta["repairs"][s, ii] == 0)
                                if stratum == "none"
                                else (meta["repairs"][s, ii] > 0)
                            )
                        elif family == "gate_tie":
                            mask = meta["ties"][s, ii] == stratum
                        else:
                            mask = (meta["gate_normalised"][s, ii] > 0) == stratum
                        m = mask & valid[s]
                        n = int(m.sum())
                        vvv = (
                            outcome(y[m], p[s : s + 1, m], g[s : s + 1, m], kind, macro)
                            if n
                            else None
                        )
                        row = dict(
                            repeat=repeat,
                            eligible=int(mask.sum()),
                            valid=n,
                            invalid=int((mask & ~valid[s]).sum()),
                            gold_positive_label_support=y[m].sum(0),
                            f1=None if not n or not macro else float(vvv["f1"][0]),
                            gate_aware_exact=(
                                None if not n else float(vvv["exact"].mean())
                            ),
                        )
                        if kind == "hate":
                            yy = y[m, 1].astype(bool)
                            pp = p[s, m, 1].astype(bool)
                            row.update(
                                fp=int((pp & ~yy).sum()),
                                fn=int((~pp & yy).sum()),
                                gold_hate=int(yy.sum()),
                                gold_nonhate=int((~yy).sum()),
                            )
                        rr.append(row)
                    strata.append(
                        dict(
                            endpoint=name,
                            condition=co,
                            family=family,
                            stratum=stratum,
                            repeats=rr,
                            note="Different input subsets, descriptive only; do not attribute performance changes causally to repair, validity or agreement.",
                        )
                    )
    x.save(x.BASE / "repeat_and_inner_diagnostics.json", output)
    x.save(x.BASE / "validity_outcome_strata.json", strata)
    x.save(x.BASE / "core_overlap_by_endpoint.json", core)
    x.save(
        x.BASE / "diagnostics_complete.json",
        dict(
            status="outer_inner_and_validity_diagnostics_complete",
            sources=x.bindings([Path(__file__), x.BASE / "metadata_complete.json"]),
            outputs=x.bindings(
                [
                    x.BASE / n
                    for n in (
                        "repeat_and_inner_diagnostics.json",
                        "validity_outcome_strata.json",
                        "core_overlap_by_endpoint.json",
                    )
                ]
            ),
            new_Qwen_calls=0,
        ),
    )
    print(
        "All 33 endpoint/condition stability and fixed-inner diagnostics complete.",
        flush=True,
    )


def raw_review():
    selections = x.c.read(x.reader.BASE / "reader_review.json")["raw_string_checks"]
    old = x.c.read(x.reader.BASE / "item_bindings.json")
    parser = x.r.e.isolated.adapter(x.ROOT / "config/taxonomy.yaml").bundle.parser
    entries = []
    counter = Counter()
    errors = Counter()
    models = Counter()
    bound = {}
    for entry in selections:
        path = entry["result_path"]
        x.require(
            path in old and x.c.sha(path) == old[path],
            "Selected original result binding differs",
        )
        bound[path] = old[path]
        v = x.c.read(path)
        chains = []
        x.require(
            v["active_model"] == v["requested_model"] == "Qwen/Qwen3-VL-32B-Instruct",
            "Selected model alias differs",
        )
        for i, (chain, saved) in enumerate(zip(v["raw_runs"], v["run_predictions"])):
            steps = []
            for j, raw in enumerate(chain):
                parsed, n, error = parser.parse_response(raw)
                last = parsed.model_dump() if parsed is not None else None
                category = (
                    "valid"
                    if parsed is not None
                    else (
                        "invalid_json"
                        if error.startswith("invalid JSON:")
                        else (
                            "schema_violation"
                            if error.startswith("schema violation:")
                            else "wrong_json_type"
                        )
                    )
                )
                counter[category] += 1
                if error:
                    errors[error] += 1
                x.require(
                    parsed is None or j == len(chain) - 1,
                    "Valid response followed by repair",
                )
                steps.append(
                    dict(
                        step=j,
                        category=category,
                        error=error,
                        normalisations=n,
                        raw_sha256=hashlib.sha256(raw.encode("utf8")).hexdigest(),
                    )
                )
            x.require(
                x.reader.gold.decision(last) == x.reader.gold.decision(saved),
                "Focused saved final vote differs",
            )
            first_valid = steps[0]["category"] == "valid"
            final_valid = steps[-1]["category"] == "valid"
            counter["votes"] += 1
            counter["rescued_after_repair"] += int(not first_valid and final_valid)
            counter["exhausted"] += int(not final_valid)
            counter["first_response_valid"] += int(first_valid)
            chains.append(
                dict(
                    vote=i, steps=steps, final_valid=final_valid, repairs=len(chain) - 1
                )
            )
        for event in v["events"]:
            mm = event.get("model_response_metadata") or {}
            models[json.dumps(mm, sort_keys=True, ensure_ascii=False)] += 1
        x.require(
            sum(len(z["steps"]) for z in chains) == entry["saved_strings_checked"],
            "Prior raw sample length changed",
        )
        entries.append(
            dict(
                selection=entry,
                chains=chains,
                requested_model=v["requested_model"],
                active_model=v["active_model"],
            )
        )
    live = x.c.read(x.r.BASE / "live/complete.json")
    x.save(
        x.BASE / "raw_history_sample.json",
        dict(
            status="all_prior_181_stratified_raw_selections_reparsed",
            selection_policy="Original Part 1 first-two-by-phase/condition/validity/repair/tie/gate-normalization plus both repeated-label cases. No outcome-based selection.",
            sample_counts=counter,
            error_messages=errors,
            model_response_metadata=models,
            entries=entries,
            all_run_model_reporting=live["accounting"],
            population_counts=x.c.read(x.reader.BASE / "reader_review.json")["counts"],
            limits="These 181 selected records deliberately oversample failures and repairs. Sample error categories and rescue rates are not population estimates. Saved aliases/provider model fields do not pin immutable server weights or certify unchanged backend configuration.",
            sources=bound
            | x.bindings(
                [
                    Path(__file__),
                    x.reader.BASE / "reader_review.json",
                    x.r.BASE / "live/complete.json",
                ]
            ),
        ),
    )
    print(
        "Reparsed all",
        len(entries),
        "previously selected original records:",
        dict(counter),
        flush=True,
    )


if __name__ == "__main__":
    with x.r.original.offline(), x.threadpool_limits(limits=2):
        diagnostics()
        raw_review()
