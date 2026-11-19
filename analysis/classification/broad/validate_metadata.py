from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from analysis.classification.broad import validate_robustness as v

x = v.x
eq = v.equal


def bits(value, field, labels):
    chosen = (
        {"hate" if value["hate"] else "not_hate"}
        if field == "hate"
        else (
            {value[field]}
            if field == "severity" and value[field] is not None
            else set(value[field] or []) if field != "severity" else set()
        )
    )
    return [int(label in chosen) for label in labels]


def exact(y, p, gate, kind):
    correct = np.equal(p, y).all(-1)
    return correct if kind in ("hate", "legal") else correct & gate


def run():
    index = x.c.read(x.BASE / "metadata_index.json")
    ki = {k: i for i, k in enumerate(index["keys"])}
    si = {tuple(k): i for i, k in enumerate(index["settings"])}
    epmap = x.c.read(x.reader.BASE / "endpoint_setup.json")
    with np.load(x.BASE / "prediction_metadata.npz", allow_pickle=False) as z:
        metadata = {k: z[k].copy() for k in z.files}
    iv = {}
    members = defaultdict(list)
    for name, ep in epmap.items():
        with np.load(
            x.BASE / "inner_votes" / (x.main.slug(name) + ".npz"), allow_pickle=False
        ) as z:
            iv[name] = {k: z[k].copy() for k in z.files}
        for j, k in enumerate(ep["scorable_ids"]):
            members[k].append((name, j))
    inventory = x.c.read(x.BASE / "validity_inventory.json")
    counts = Counter()
    cells = defaultdict(Counter)
    source = defaultdict(Counter)
    agree = defaultdict(Counter)
    rationales = defaultdict(Counter)
    seen = set()
    nulls = []
    partials = []
    roster = {
        z["key"]: z
        for rr in x.scoring.scientific_rosters("broad")[0].values()
        for z in rr
    }
    for row in x.reader.rows(x.reader.BASE / "predictions.jsonl"):
        ident = row["condition"], row["repeat"]
        s = si[ident]
        i = ki[row["key"]]
        key = (s, i)
        x.require(key not in seen, "Duplicate cache record")
        seen.add(key)
        nv = sum(a is not None for a in row["votes"])
        eq(nv, row["n_valid"])
        eq(int(metadata["n_valid"][s, i]), nv)
        eq(int(metadata["repairs"][s, i]), row["repairs"])
        eq(bool(metadata["ties"][s, i]), row["uncertain"])
        eq(int(metadata["gate_normalised"][s, i]), row["gate_normalised"])
        eq(bool(metadata["gate"][s, i]), bool(row["result"] and row["result"]["hate"]))
        co = Counter(
            predictions=1,
            nulls=int(nv == 0),
            partial=int(0 < nv < 3),
            repaired_predictions=int(row["repairs"] > 0),
            repairs=row["repairs"],
            parse_failures=row["parse_failures"],
            gate_ties=int(row["uncertain"]),
            gate_normalised=row["gate_normalised"],
            normalisations=row["normalisations"],
        )
        co["valid_" + str(nv)] += 1
        counts.update(co)
        cells[row["phase"], row["condition"], row["repeat"]].update(co)
        source[
            row["phase"],
            row["source"],
            row["condition"],
            str(roster[row["key"]]["gate"]),
        ].update(co)
        if nv == 0:
            nulls.append(
                dict(
                    key=row["key"],
                    condition=row["condition"],
                    repeat=row["repeat"],
                    phase=row["phase"],
                )
            )
        elif nv < 3:
            partials.append(
                dict(
                    key=row["key"],
                    condition=row["condition"],
                    repeat=row["repeat"],
                    n_valid=nv,
                )
            )
        for field in x.reader.FIELDS:
            values = [
                tuple(sorted(set(a[field]))) if isinstance(a[field], list) else a[field]
                for a in row["votes"]
                if a is not None
            ]
            frequencies = Counter(values)
            a = agree[row["phase"], row["condition"], field]
            a.update(
                predictions=1,
                informative=int(nv > 1),
                unanimous=int(nv > 1 and len(frequencies) == 1),
                agreeing_pairs=sum(n * (n - 1) // 2 for n in frequencies.values()),
                pairs=nv * (nv - 1) // 2,
            )
            a["valid_" + str(nv)] += 1
        if row["result"] is not None:
            first = [
                a
                for a in row["votes"]
                if a is not None and a["hate"] == row["result"]["hate"]
            ][0]
            rr = rationales[row["phase"], row["condition"]]
            rr["valid_predictions"] += 1
            different = []
            for field in x.reader.FIELDS:
                a, b = first[field], row["result"][field]
                same = set(a) == set(b) if isinstance(a, list) else a == b
                if not same:
                    rr[field] += 1
                    different.append(field)
            rr["any_field_differs"] += bool(different)
        for name, j in members[row["key"]]:
            ep = epmap[name]
            for slot, a in enumerate(row["votes"]):
                eq(bool(iv[name]["valid"][s, slot, j]), a is not None)
                eq(
                    iv[name]["predicted"][s, slot, j],
                    (
                        np.zeros(len(ep["all_labels"]), int)
                        if a is None
                        else bits(a, ep["dimension"], ep["all_labels"])
                    ),
                )
                eq(bool(iv[name]["gate"][s, slot, j]), bool(a and a["hate"]))
    eq(len(seen), 194643)
    eq(dict(counts), inventory["total"])
    eq(nulls, inventory["null_predictions"])
    eq(partials, inventory["partial_predictions"])
    eq(len({a["key"] for a in nulls}), inventory["null_distinct_inputs"])
    for a in inventory["cells"]:
        eq(dict(cells[a["phase"], a["condition"], a["repeat"]]), a["counts"])
    for a in inventory["source_gold"]:
        eq(
            dict(source[a["phase"], a["source"], a["condition"], a["gold"]]),
            a["counts"],
        )
    for a in inventory["inner_agreement"]:
        eq(dict(agree[a["phase"], a["condition"], a["field"]]), a["counts"])
    for a in inventory["rationale_source_field_differences"]:
        eq(dict(rationales[a["phase"], a["condition"]]), a["counts"])
    print(
        "All cached vote arrays, validity strata, agreement and rationale-source differences match.",
        flush=True,
    )
    results = x.c.read(x.BASE / "repeat_and_inner_diagnostics.json")
    strata = x.c.read(x.BASE / "validity_outcome_strata.json")
    for name, ep in sorted(epmap.items()):
        old, terms = x.endpoint(name)
        keys = [(co, j) for co in x.CONDITIONS for j in range(3)]
        y = terms[keys[0]]["gold"]
        p = np.stack([terms[k]["pred"] for k in keys])
        valid = np.stack([terms[k]["valid"] for k in keys])
        ii = [ki[k] for k in ep["scorable_ids"]]
        g = metadata["gate"][:, ii]
        kind = ep["dimension"]
        idx = [ep["all_labels"].index(k) for k in ep["macro_labels"]]
        cm = valid.all(0)
        complete = iv[name]["valid"].all((0, 1))
        eq(complete, (metadata["n_valid"][:, ii] == 3).all(0))
        for co in x.CONDITIONS:
            j = x.CONDITIONS.index(co)
            positions = list(range(3 * j, 3 * j + 3))
            saved = next(
                a for a in results if a["endpoint"] == name and a["condition"] == co
            )
            pp = p[positions][:, cm]
            gg = g[positions][:, cm]
            ex = exact(y[cm], pp, gg, kind)
            ident = (pp[0] == pp[1]).all(1) & (pp[1] == pp[2]).all(1)
            if kind not in ("hate", "legal"):
                ident &= (gg[0] == gg[1]) & (gg[1] == gg[2])
            expected = dict(
                n_common_nine=int(cm.sum()),
                same_answer_all_repeats=int(ident.sum()),
                same_wrong_answer=int((ident & ~ex.any(0)).sum()),
                all_repeats_correct=int(ex.all(0).sum()),
                all_repeats_wrong=int((~ex.any(0)).sum()),
                variable_correctness=int((ex.any(0) & ~ex.all(0)).sum()),
                correct_repeats_hist=dict(Counter(ex.sum(0).tolist())),
                per_repeat_f1=v.fscore(y[cm], pp, idx),
                per_repeat_exact=ex.sum(1) / cm.sum(),
            )
            eq(expected, saved["outer"], name + "/" + co + "/outer")
            eq(
                expected["per_repeat_f1"],
                old["three_condition_common"]["per_condition"][co]["repeat_macro_f1"],
            )
            vp = iv[name]["predicted"][positions][:, :, complete]
            vg = iv[name]["gate"][positions][:, :, complete]
            ap = p[positions][:, complete]
            ag = g[positions][:, complete]
            yy = y[complete]
            ve = exact(yy, vp, vg, kind)
            ae = exact(yy, ap, ag, kind)
            vf = v.fscore(yy, vp, idx)
            af = v.fscore(yy, ap, idx)
            matches = (vp == ap[:, None]).all(-1)
            if kind not in ("hate", "legal"):
                matches &= vg == ag[:, None]
            expected = dict(
                n_complete_all_conditions=int(complete.sum()),
                removed_from_common_nine=int((cm & ~complete).sum()),
                fixed_slot_f1=vf,
                mean_single_f1=float(vf.mean()) if idx else None,
                fixed_slot_exact=ve.mean(2),
                mean_single_exact=float(ve.mean()),
                aggregate_f1=af,
                mean_aggregate_f1=float(af.mean()) if idx else None,
                aggregate_exact=ae.mean(1),
                mean_aggregate_exact=float(ae.mean()),
                aggregate_correct_all_votes_wrong_per_repeat=(ae & ~ve.any(1)).sum(1),
                aggregate_wrong_all_votes_correct_per_repeat=(~ae & ve.all(1)).sum(1),
                aggregate_answer_matches_no_single_vote_per_repeat=(
                    ~matches.any(1)
                ).sum(1),
            )
            eq(
                expected,
                {k: a for k, a in saved["single_inner"].items() if k != "note"},
                name + "/" + co + "/inner",
            )
        for row in [a for a in strata if a["endpoint"] == name]:
            start = 3 * x.CONDITIONS.index(row["condition"])
            family = row["family"]
            stratum = row["stratum"]
            for a in row["repeats"]:
                s = start + a["repeat"]
                if family == "valid_votes":
                    m = metadata["n_valid"][s, ii] == stratum
                elif family == "repairs":
                    m = (metadata["repairs"][s, ii] > 0) == (stratum == "some")
                elif family == "gate_tie":
                    m = metadata["ties"][s, ii] == stratum
                else:
                    m = (metadata["gate_normalised"][s, ii] > 0) == stratum
                mask = m & valid[s]
                n = int(mask.sum())
                eq(a["eligible"], int(m.sum()))
                eq(a["valid"], n)
                eq(a["invalid"], int((m & ~valid[s]).sum()))
                eq(a["gold_positive_label_support"], y[mask].sum(0))
                eq(
                    a["f1"],
                    float(v.fscore(y[mask], p[s, mask], idx)) if n and idx else None,
                )
                eq(
                    a["gate_aware_exact"],
                    (
                        float(exact(y[mask], p[s, mask], g[s, mask], kind).mean())
                        if n
                        else None
                    ),
                )
                if kind == "hate":
                    yy = y[mask, 1]
                    pp = p[s, mask, 1]
                    eq(a["fp"], int(((pp == 1) & (yy == 0)).sum()))
                    eq(a["fn"], int(((pp == 0) & (yy == 1)).sum()))
                    eq(a["gold_hate"], int(yy.sum()))
                    eq(a["gold_nonhate"], n - int(yy.sum()))
        print("Independent repeat/inner/stratum metrics:", name, flush=True)
    raw = x.c.read(x.BASE / "raw_history_sample.json")
    previous = x.c.read(x.reader.BASE / "reader_review.json")
    eq([z["selection"] for z in raw["entries"]], previous["raw_string_checks"])
    t = Counter()
    errs = Counter()
    for row in raw["entries"]:
        for ch in row["chains"]:
            cats = [z["category"] for z in ch["steps"]]
            t.update(cats)
            t["votes"] += 1
            t["rescued_after_repair"] += cats[0] != "valid" and cats[-1] == "valid"
            t["exhausted"] += cats[-1] != "valid"
            t["first_response_valid"] += cats[0] == "valid"
            eq(ch["repairs"], len(cats) - 1)
            for z in ch["steps"]:
                if z["error"]:
                    errs[z["error"]] += 1
    eq(dict(t), raw["sample_counts"])
    eq(dict(errs), raw["error_messages"])
    x.r.verify(raw["sources"])
    x.save(
        x.BASE / "independent_metadata_verification.json",
        dict(
            status="all_cache_vote_metadata_outer_inner_strata_and_raw_sample_checks_passed",
            checked_values=v.checks,
            maximum_difference=v.maximum,
            cached_predictions=len(seen),
            endpoint_conditions=len(results),
            validity_strata=len(strata),
            raw_selected_records=len(raw["entries"]),
            sources=x.bindings(
                [
                    Path(__file__),
                    Path(v.__file__),
                    x.BASE / "metadata_complete.json",
                    x.BASE / "diagnostics_complete.json",
                    x.BASE / "raw_history_completion.json",
                ]
            ),
            new_Qwen_calls=0,
        ),
    )


if __name__ == "__main__":
    with x.r.original.offline(), x.threadpool_limits(limits=2):
        run()
