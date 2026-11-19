from analysis.feedback import micro_f1 as p
from collections import Counter
from pathlib import Path
import hashlib
import itertools
import numpy as np

c, h = p.c, p.h
SECTIONS = {
    "feedback": "analysis_v2",
    "relevance": "analysis_v2",
    "explanations": "analysis_v3",
    "balanced": "analysis_v3",
    "reserved": "analysis_v4",
    "transfer": "analysis_v2",
    "rounds": "analysis_v3",
    "human": "analysis_v2",
    "extras": "analysis_v2",
}


def indexes():
    return [c.read(p.BASE / att / sec / "index.json") for sec, att in SECTIONS.items()]


def paths():
    return [c.ROOT / path for i in indexes() for path in i["outputs"]]


def calc(tp, pp, gg, ix):
    den = pp + gg
    f = np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0)
    full = den.sum(-1)
    rest = den[..., ix].sum(-1)
    return np.stack(
        [
            f[..., ix].mean(-1),
            np.divide(
                2 * tp.sum(-1),
                full,
                out=np.zeros_like(full, dtype=float),
                where=full != 0,
            ),
            np.divide(
                2 * tp[..., ix].sum(-1),
                rest,
                out=np.zeros_like(rest, dtype=float),
                where=rest != 0,
            ),
        ],
        -1,
    )


def aggregate(r, setting):
    g = np.stack([setting[..., js, :].mean(-2) for js in r["groups"].values()], -2)
    names = list(r["groups"])
    v = np.stack(
        [
            sum(w * g[..., names.index(k), :] for k, w in col.items())
            for col in r["columns"].values()
        ],
        -2,
    )
    return g, v


def fixtures():
    checked = 0

    y = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], bool)
    x = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], bool)
    valid = np.array([1, 0, 0], bool)
    values = []
    for bits in itertools.product([False, True], repeat=6):
        z = x.copy()
        z[~valid] = np.array(bits).reshape(2, 3)
        values.append(calc((z & y).sum(0), z.sum(0), y.sum(0), [0, 1]))
    summary = h.setting_summary(y, x, valid, ["a", "b", "c"], ["a", "b"])
    for j, m in enumerate(h.METHODS):
        h.equal(
            summary["full_roster_missing_bounds"][m],
            [np.min(values, 0)[j], np.max(values, 0)[j]],
            "Exhaustive missing-output fixture",
        )
        checked += 1
    c.require(
        summary["n_eligible"] == 3 and summary["n_invalid"] == 2,
        "Missing not empty fixture",
    )
    checked += 1

    z = calc(np.array([1, 0]), np.array([1, 1]), np.array([1, 0]), [0])
    h.equal(z, [1, 2 / 3, 1], "Label-scope fixture")
    checked += 1

    tp = np.array([[1], [9]])
    pp = np.array([[1], [99]])
    gold = np.array([[1], [9]])
    c.require(
        abs(
            calc(tp, pp, gold, [0])[:, 1].mean()
            - calc(tp.sum(0), pp.sum(0), gold.sum(0), [0])[1]
        )
        > 0.1,
        "No pooled-runs fixture",
    )
    checked += 1

    r = {
        "groups": {"reference": [0, 1, 2], "update": list(range(3, 12))},
        "columns": {"change": {"update": 1, "reference": -1}},
    }
    s = np.arange(36).reshape(12, 3) / 36
    g, v = aggregate(r, s)
    h.equal(v[0], s[3:].mean(0) - s[:3].mean(0), "Actual reference weighting fixture")
    checked += 1
    return checked


def main():
    counters = Counter(fixtures=fixtures())
    seen = set()
    questions = []
    raw_total = 0
    stored_values = 0
    old_values = 0
    sources = {}
    indices = indexes()
    accepted_paths = paths()
    c.require(len(accepted_paths) == 201, "Incomplete accepted scope")
    for idx in indices:
        c.require(idx["status"] == "passed", "Incomplete section")
        for path, sha in idx["source_sha256"].items():
            c.require(c.sha(c.ROOT / path) == sha, "Changed input since calculation")
            sources[path] = sha
    original_reserved = c.read(
        p.B / "sq3_reserved_part5_results_20261020_v1/missing_output_bounds.json"
    )
    for count, path in enumerate(accepted_paths, 1):
        r = c.read(path)
        c.require(r["id"] not in seen, "Duplicate accepted view")
        seen.add(r["id"])
        raw_total += r["raw_records_checked"]
        a = h.read_arrays(path.with_name(path.stem + "_data.npz"))
        g = a["gold"]
        pred = a["pred"]
        valid = a["valid"]
        mask = a["mask"]
        idx = r["macro_indices"]
        S, N, L = pred.shape
        c.require(
            [x for i, x in enumerate(r["ids"]) if mask[i]] == r["common_ids"],
            "Saved common ID identity",
        )
        c.require(
            mask.sum() == r["n_common"] and (~valid[:, mask]).sum() == 0,
            "Saved common validity",
        )
        tp = (pred & g[None]).astype(np.int64)
        pp = pred.astype(np.int64)
        gg = np.broadcast_to(g, pred.shape).astype(np.int64)
        points = calc(tp[:, mask].sum(1), pp[:, mask].sum(1), gg[:, mask].sum(1), idx)
        group, cols = aggregate(r, points)
        h.equal(
            cols,
            [
                [v["metrics"][m]["value"] for m in h.METHODS]
                for v in r["columns_results"].values()
            ],
            "Independent stored point replay",
        )
        counters["point_columns"] += len(cols)
        for j, summ in enumerate(r["own_settings"]):
            own = calc(
                tp[j, valid[j]].sum(0),
                pp[j, valid[j]].sum(0),
                gg[j, valid[j]].sum(0),
                idx,
            )
            h.equal(
                own,
                [summ["scores"][m] for m in h.METHODS],
                "Independent own-valid replay",
            )
            counters["own_settings"] += 1
            for m in h.METHODS:
                bound = summ["full_roster_missing_bounds"][m]
                c.require(bound[0] <= bound[1] + 1e-14, "Reversed bounds")
            h.equal(
                sum(t["full_micro_contribution"] for t in summ["per_label"]),
                own[1],
                "Micro contribution sum",
            )
            h.equal(
                sum(
                    t["restricted_minus_macro_contribution"] for t in summ["per_label"]
                ),
                own[2] - own[0],
                "Weighting contribution sum",
            )
        for name, rec in r["columns_results"].items():
            if rec["new_micro_question"]:
                questions.append(rec["new_micro_question"])
            if r["study"] == "reserved":
                key = r["view"] + "/ordinary_valid"
                field = "opposing_difference" if name == "treatment-control" else name
                h.equal(
                    rec["full_roster_missing_bounds"]["macro"],
                    original_reserved[key]["metrics"]["type_macro_f1_fixed7"][field],
                    "Original reserved missing bounds",
                )
                counters["reserved_bounds"] += 1
        if r["bootstrap"]:
            saved = h.read_arrays(path.with_name(path.stem + "_draws.npz"))
            pos = np.flatnonzero(mask)
            labels = (
                list(range(len(pos)))
                if r["sampling"] == "item"
                else sorted({r["clusters"][i] for i in pos})
            )
            loc = {k: i for i, k in enumerate(labels)}
            membership = (
                np.arange(len(pos))
                if r["sampling"] == "item"
                else np.array([loc[r["clusters"][i]] for i in pos])
            )

            cube = np.stack([tp[:, pos], pp[:, pos], gg[:, pos]], -1).transpose(
                1, 0, 2, 3
            )
            total = np.zeros((len(labels), S, L, 3), np.int64)
            np.add.at(total, membership, cube)
            flat = total.reshape(len(labels), -1).astype(float)
            rng = np.random.Generator(np.random.PCG64(r["seed"]))
            dig = hashlib.sha256()
            for start in range(0, 10000, 128):
                size = min(128, 10000 - start)
                chosen = rng.integers(
                    0, len(labels), size=(size, len(labels)), dtype=np.int64
                )
                dig.update(chosen.astype("<i4").tobytes())
                weights = np.zeros((size, len(labels)))
                np.add.at(weights, (np.arange(size)[:, None], chosen), 1)
                z = (weights @ flat).reshape(size, S, L, 3)
                sc = calc(z[..., 0], z[..., 1], z[..., 2], idx)
                gs, cs = aggregate(r, sc)
                h.equal(
                    gs,
                    saved["groups"][start : start + size],
                    "Independent stored group bootstrap replay",
                )
                h.equal(
                    cs,
                    saved["columns"][start : start + size],
                    "Independent stored column bootstrap replay",
                )
                stored_values += gs.size + cs.size
            c.require(
                dig.hexdigest() == r["bootstrap"]["pick_stream_sha256"],
                "Resampling stream differs",
            )
            counters["bootstrap_views"] += 1
            for name, rec in r["columns_results"].items():
                j = list(r["columns"]).index(name)
                for mi, m in enumerate(h.METHODS):
                    metric = rec["metrics"][m]
                    h.equal(
                        np.quantile(saved["columns"][:, j, mi], [0.025, 0.975]),
                        metric["ci95"],
                        "Exported CI95",
                    )
                    if metric["adjusted_interval"] is not None:
                        alpha = 1 - metric["adjusted_confidence"]
                        h.equal(
                            np.quantile(
                                saved["columns"][:, j, mi], [alpha / 2, 1 - alpha / 2]
                            ),
                            metric["adjusted_interval"],
                            "Exported adjusted CI",
                        )
            if r["original_draw_path"]:
                old_values += 10000 * len(r["original_draw_columns"])
        if count % 20 == 0 or count == len(accepted_paths):
            print(
                "Independent verification", count, "/", len(accepted_paths), flush=True
            )
    planned = [
        q
        for f in c.read(p.BASE / "analysis_plan.json")["families"].values()
        for q in f["members"]
    ]
    c.require(
        sorted(questions) == sorted(planned) and len(questions) == 16,
        "Question coverage changed",
    )
    c.require(
        {u["id"] for u in c.read(p.BASE / "analysis_plan.json")["units"]}
        <= {c.read(path)["unit_id"] for path in accepted_paths},
        "A registered unit is missing",
    )

    byid = {c.read(path)["id"]: path for path in accepted_paths}
    partial_checked = 0
    for path in (p.BASE / "analysis_v2").glob("*/failure.json"):
        failed = c.read(path)
        for name in failed.get("completed", []):
            prior = c.read(c.ROOT / name)
            current = c.read(byid[prior["id"]])
            h.equal(
                [
                    [v["metrics"][m]["value"] for m in h.METHODS]
                    for v in prior["columns_results"].values()
                ],
                [
                    [v["metrics"][m]["value"] for m in h.METHODS]
                    for v in current["columns_results"].values()
                ],
                "Earlier completed values preserved",
            )
            partial_checked += 1
    p.save(
        p.BASE / "independent_verification.json",
        dict(
            status="passed",
            views=len(accepted_paths),
            registered_units=67,
            questions=questions,
            checks=dict(counters),
            raw_records_checked=raw_total,
            independently_replayed_stored_values=stored_values,
            original_bootstrap_values_replayed=old_values,
            partial_completed_views_unchanged=partial_checked,
            numerical_checks=dict(h.CHECKS),
            max_error=dict(h.ERRORS),
            source_sha256={**sources, **c.BINDINGS},
        ),
    )
    p.save(
        p.BASE / "accepted_analysis_index.json",
        dict(
            status="all_offline_sections_verified",
            sections=SECTIONS,
            outputs=[c.rel(x) for x in accepted_paths],
            artifacts={
                c.rel(x): c.sha(x)
                for x in p.BASE.rglob("*")
                if x.is_file()
                and x.name not in ["handover_before.md", "session_before.md"]
            },
            source_sha256={**sources, **c.BINDINGS},
        ),
    )
    print(
        "VERIFIED",
        len(accepted_paths),
        "views;",
        stored_values,
        "stored metric values;",
        old_values,
        "original macro/micro draw values;",
        len(questions),
        "frozen questions",
        flush=True,
    )


if __name__ == "__main__":
    main()
