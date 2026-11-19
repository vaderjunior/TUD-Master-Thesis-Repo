"""Policy verify robustness."""

from collections import Counter, defaultdict
import numpy as np
from analysis.knowledge.policy import robustness as p

c = p.c


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def values(row, field, gold=False):
    if field == "hate":
        return {"hate" if row["gate" if gold else "hate"] else "not_hate"}
    if field == "severity":
        return set() if row["severity"] is None else {row["severity"]}
    key = (
        {"hate_type": "hate_types", "target_group": "target_groups", "legal": "legal"}[field]
        if gold
        else field
    )
    return set(row[key])


def scores(counts):
    result = []
    pooled = []
    for cell in counts:
        f = []
        for tp, fp, fn in cell:
            f.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        tp, fp, fn = cell.sum(axis=0)
        result.append(sum(f) / len(f))
        pooled.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return (np.array(result), np.array(pooled))


def run():
    c.verify(c.read(p.BASE / "analysis_contract.json")["inputs"])
    roster, arrays, rows = p.load()
    index = {(r["key"], r["condition"], r["repeat"]): r for r in rows}
    linked = {
        r["key"]
        for r in c.read(p.P4 / "context_followup_checks.json")["gahd_direct_contrastive_links"][
            "cases"
        ]
    }
    checked_cells = bootstrap_values = 0
    records = []
    all_counts = {}
    direct_arrays = {}
    for endpoint, meta in p.load_endpoints().items():
        keys = meta["keys"]
        labels = meta["all_labels"]
        field = meta["dimension"]
        ar = arrays[endpoint]
        shape = (3, 3, len(keys), len(labels), 3)
        contrib = np.zeros(shape, np.int64)
        states = np.empty((3, 3, len(keys)), object)
        exact = np.zeros((3, 3, len(keys)), bool)
        for i, key in enumerate(keys):
            y = values(roster[key], field, True)
            close(ar["gold"][i], [label in y for label in labels])
            for a, cond in enumerate(p.NAMES):
                for r in range(3):
                    row = index[key, cond, r]
                    v = row["result"]
                    assert bool(ar["valid"][a, r, i]) == (v is not None)
                    assert (
                        ar["n_valid"][a, r, i] == row["n_valid"]
                        and ar["repairs"][a, r, i] == row["repairs"]
                    )
                    if v is None:
                        states[a, r, i] = None
                        continue
                    pred = values(v, field)
                    close(ar["predicted"][a, r, i], [label in pred for label in labels])
                    fine = field not in ("hate", "legal")
                    states[a, r, i] = (bool(v["hate"]) if fine else None, tuple(sorted(pred)))
                    exact[a, r, i] = pred == y and (not fine or v["hate"])
                    for j, label in enumerate(labels):
                        contrib[a, r, i, j] = [
                            label in y and label in pred,
                            label not in y and label in pred,
                            label in y and label not in pred,
                        ]
        assert np.array_equal(exact, ar["correct"])
        all_counts[endpoint] = contrib
        direct_arrays[endpoint] = (states, exact)
        for view in (
            x for x in c.read(p.BASE / "sensitivity_index.json") if x["endpoint"] == endpoint
        ):
            report = c.read(c.ROOT / view["path"])
            name = view["mask"]
            mask = p.mask_set(endpoint, ar, linked)[name]
            wanted = [k for k, on in zip(keys, mask) if on]
            assert wanted == report["composition"]["keys"]
            assert report["composition"]["sources"] == dict(
                Counter((roster[k]["source"] for k in wanted))
            )
            point = report["point"]
            counts = contrib[:, :, mask].sum(axis=2)
            close(counts, point["counts"])
            macro, micro = scores(counts.reshape(9, len(labels), 3))
            close(macro.reshape(3, 3), point["macro_f1_repeats"])
            if "micro_f1_repeats" in point:
                close(micro.reshape(3, 3), point["micro_f1_repeats"])
            close(exact[:, :, mask].mean(axis=2), point["exact_repeats"])
            agreements = np.stack(
                [states[:, a, mask] == states[:, b, mask] for a, b in ((0, 1), (0, 2), (1, 2))],
                axis=1,
            )
            close(agreements.mean(axis=(1, 2)), point["repeatability"]["pairwise_agreement"])
            checked_cells += 9
            selected_indices = [i for i, on in enumerate(mask) if on]
            selected_indices.sort(key=lambda i: roster[keys[i]]["text_cluster"])
            assert len({roster[keys[i]]["text_cluster"] for i in selected_indices}) == len(
                selected_indices
            )
            source_groups = defaultdict(list)
            for j, i in enumerate(selected_indices):
                source_groups[roster[keys[i]]["source"],].append(j)
            rng = np.random.default_rng(20261114)
            weights = np.zeros((256, len(selected_indices)), int)
            for source, indices in sorted(source_groups.items()):
                weights[:, indices] = rng.multinomial(
                    len(indices), [1 / len(indices)] * len(indices), size=256
                )
            c.require(
                c.sha(c.ROOT / report["bootstrap_source"]) == report["bootstrap_sha256"],
                "Bootstrap artifact changed",
            )
            boot = dict(np.load(c.ROOT / report["bootstrap_source"], allow_pickle=False))
            counts_selected = contrib[:, :, selected_indices]
            for draw in range(8):
                totals = (counts_selected * weights[draw][None, None, :, None, None]).sum(axis=2)
                ma, mi = scores(totals.reshape(9, len(labels), 3))
                close(ma.reshape(3, 3).mean(axis=1), boot["macro_f1"][draw])
                close(mi.reshape(3, 3).mean(axis=1), boot["micro_f1_all_labels"][draw])
                agr = np.stack(
                    [
                        states[:, a, selected_indices] == states[:, b, selected_indices]
                        for a, b in ((0, 1), (0, 2), (1, 2))
                    ],
                    axis=1,
                ).mean(axis=1)
                close(
                    (agr * weights[draw]).sum(axis=1) / weights[draw].sum(),
                    boot["mean_pairwise_repeat_agreement"][draw],
                )
                bootstrap_values += 9
            for contrast in report["contrasts"]:
                a = p.NAMES.index(contrast["left"])
                b = p.NAMES.index(contrast["right"])
                delta = boot[contrast["metric"]][:, a] - boot[contrast["metric"]][:, b]
                confidence = contrast["family_confidence"]
                tail = (1 - confidence) / 2
                close(
                    np.quantile(delta, [tail, 1 - tail], method="linear"),
                    contrast["family_interval"],
                )
                assert len(delta) == 50000 and np.isfinite(delta).all()
            records.append(
                dict(
                    endpoint=endpoint,
                    mask=name,
                    n=len(wanted),
                    score_cells=9,
                    bootstrap_values=72,
                    passed=True,
                )
            )
    missing = c.read(p.BASE / "missing_output_scores_bounds.json")
    for endpoint, ar in arrays.items():
        counts = all_counts[endpoint]
        meta = p.load_endpoints()[endpoint]
        for ci, cond in enumerate(p.NAMES):
            for r in range(3):
                valid = ar["valid"][ci, r]
                cell = counts[ci, r, valid].sum(axis=0)
                own = missing[endpoint]["conditions"][cond]["own_valid_repeats"][r]
                ma, mi = scores(cell[None])
                close(ma[0], own["macro_f1"])
                close(cell, own["counts"])
                if "micro_f1" in own:
                    close(mi[0], own["micro_f1"])
                b = missing[endpoint]["conditions"][cond]["full_roster_bounds_repeats"][r]
                assert b["missing"] == int((~valid).sum())
                assert b["macro_f1"][0] <= b["macro_f1"][1] + 1e-12
                if valid.all():
                    close(b["macro_f1"], [own["macro_f1"]] * 2)
    original = c.read(p.load_run() / "contrasts.json")
    current = [
        r
        for r in c.read(p.BASE / "sensitivity_contrasts.json")
        if r["mask"] == "original_common_nine"
    ]
    current_index = {
        tuple((r[k] for k in ("endpoint", "metric", "left", "right"))): r for r in current
    }
    for before in original:
        after = current_index[tuple((before[k] for k in ("endpoint", "metric", "left", "right")))]
        for key in ("endpoint", "metric", "left", "right"):
            assert before[key] == after[key]
        for key in ("estimate", "interval95"):
            close(before[key], after[key])
        if before["family_interval"] is None:
            assert after["family"] is None
            close(before["interval95"], after["family_interval"])
        else:
            close(before["family_interval"], after["family_interval"])
    old = {(r["key"], r["repeat"]): r for r in c.rows(p.BASE / "historical_matched_rows.jsonl")}
    hist = c.read(p.BASE / "historical_comparison.json")
    history_cells = 0
    for endpoint, meta in p.load_endpoints().items():
        keys = meta["keys"]
        field = meta["dimension"]
        labels = meta["all_labels"]
        wanted = [
            k
            for k in keys
            if all(
                (
                    old[k, r]["result"] is not None
                    and index[k, p.NAMES[0], r]["result"] is not None
                    for r in range(3)
                )
            )
        ]
        assert len(wanted) == hist["populations"][endpoint]["n"]
        for ci in range(2):
            for r in range(3):
                counts = np.zeros((len(labels), 3), int)
                for key in wanted:
                    y = values(roster[key], field, True)
                    v = (old[key, r] if ci == 0 else index[key, p.NAMES[0], r])["result"]
                    pr = values(v, field)
                    for j, label in enumerate(labels):
                        counts[j] += [
                            label in y and label in pr,
                            label not in y and label in pr,
                            label in y and label not in pr,
                        ]
                ma, mi = scores(counts[None])
                close(ma[0], hist["populations"][endpoint]["scores_repeats"]["macro_f1"][ci][r])
                if "micro_f1" in hist["populations"][endpoint]["scores_repeats"]:
                    close(mi[0], hist["populations"][endpoint]["scores_repeats"]["micro_f1"][ci][r])
                history_cells += 1
    p.save(
        "independent_checks.json",
        dict(
            status="independent_checks_passed",
            score_cells=checked_cells,
            actual_bootstrap_values=bootstrap_values,
            historical_score_cells=history_cells,
            original_contrasts_unchanged=len(current),
            records=records,
            methods="Direct sets/counts, exact states, source masks, independently weighted first eight draws and all saved quantiles. Eighteen separate exhaustive missing-fill boundary tests.",
            sources=c.bindings([__file__, p.__file__]),
        ),
    )
    print(
        f"PASS: {checked_cells} actual score cells, {bootstrap_values} bootstrap values, {history_cells} historical cells.",
        flush=True,
    )


if __name__ == "__main__":
    with c.offline():
        run()
