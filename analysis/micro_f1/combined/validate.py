from analysis.micro_f1.combined import prepare as p
from collections import Counter
from pathlib import Path
import csv
import json
import numpy as np

c = p.c


def equal(a, b, message):
    if a is None or b is None:
        c.require(a is None and b is None, message)
        return
    c.require(np.allclose(a, b, atol=2e-14, rtol=0, equal_nan=False), message)


def classify(interval):
    if interval is None:
        return "not_tested"
    if min(interval) > 0:
        return "positive"
    if max(interval) < 0:
        return "negative"
    return "includes_zero"


def metric_values(tp, pp, gg, indices):
    den = pp + gg
    f = np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0)

    def score(ix):
        num = 2 * tp[..., ix].sum(-1)
        d = den[..., ix].sum(-1)
        return np.divide(num, d, out=np.zeros_like(num, dtype=float), where=d != 0)

    return np.stack(
        [f[..., indices].mean(-1), score(list(range(tp.shape[-1]))), score(indices)], -1
    )


def main():
    rows = c.read(p.BASE / "comparison_ledger.json")
    registry = c.read(p.ROOTS[1] / "comparison_registry_v2.json")
    expected = [
        q for f in registry["new_micro_families"].values() for q in f["members"]
    ]
    c.require(
        Counter(r["question_id"] for r in rows if r["question_id"])
        == Counter(expected),
        "Independent family coverage",
    )
    lookup = {}
    scalar_checks = 0
    for row in rows:
        path = c.ROOT / row["source"]
        if row["source"] not in lookup:
            lookup[row["source"]] = c.read(path)
        src = lookup[row["source"]]
        for token in row["source_pointer"].split("/")[1:]:
            src = src[int(token)] if isinstance(src, list) else src[token]
        original = src.get("metrics", src)
        for name, metric in row["metrics"].items():
            value = original[name]
            equal(
                metric["value"],
                value.get("value", value.get("difference")),
                "Source scalar differs",
            )
            equal(metric["ci95"], value.get("ci95"), "Source nominal interval differs")
            c.require(
                metric["class95"] == classify(value.get("ci95")),
                "Source classification differs",
            )
            if name == "macro" and row["part"] == 2:
                ai = src.get("original_macro_adjusted_interval")
                ac = src.get("original_macro_adjusted_confidence")
            else:
                ai = value.get(
                    "original_adjusted_interval", value.get("adjusted_interval")
                )
                ac = value.get(
                    "original_adjusted_confidence", value.get("adjusted_confidence")
                )
            equal(metric["adjusted_interval"], ai, "Source adjusted interval differs")
            equal(metric["adjusted_confidence"], ac, "Source confidence differs")
            scalar_checks += 1
        a, b, d = (
            row["metrics"][k]["value"] for k in ["macro", "micro", "micro_macro_labels"]
        )
        equal(
            [d - a, b - d],
            [row["averaging_component"], row["label_scope_component"]],
            "Weighting/scope decomposition",
        )
    csv_checks = 0
    for filename in [
        "comparison_ledger.csv",
        "prelisted_questions.csv",
        "metric_sensitive_inventory.csv",
    ]:
        with c.bind(p.BASE / filename).open(encoding="utf8", newline="") as f:
            export = list(csv.DictReader(f))
        byid = {r["id"]: r for r in rows}
        for row in export:
            original = byid[row["id"]]
            for name in ["macro", "micro", "micro_macro_labels"]:
                c.require(
                    float(row[name + "_value"]) == original["metrics"][name]["value"],
                    "CSV precision loss",
                )
                csv_checks += 1

    idx = c.read(p.BASE / "broad_strict/index.json")
    strict = []
    values_checked = 0
    for f in idx["outputs"]:
        path = c.ROOT / f
        r = c.read(path)
        with np.load(c.bind(path.parent / "arrays.npz"), allow_pickle=False) as file:
            a = {k: file[k] for k in file.files}
        with np.load(c.bind(path.parent / "draws.npz"), allow_pickle=False) as file:
            saved = file["draws"]
        g = a["gold"]
        pred = a["pred"]
        mask = a["mask"]
        labels = r["labels"]
        ix = [labels.index(x) for x in r["macro_labels"]]
        c.require(
            np.array_equal(mask, a["valid"].all(0) & (a["nvalid"] == 3).all(0)),
            "Strict mask changed",
        )
        c.require(
            [k for k, v in zip(r["ids"], mask) if v] == r["common_ids"],
            "Strict identities changed",
        )
        names = sorted(
            {cluster for cluster, valid in zip(r["clusters"], mask) if valid}
        )
        loc = {k: i for i, k in enumerate(names)}
        membership = np.array([loc[r["clusters"][i]] for i in np.flatnonzero(mask)])
        S, N, L = pred.shape
        tp = (pred[:, mask] & g[mask]).astype(np.int64)
        pp = pred[:, mask].astype(np.int64)
        gg = np.broadcast_to(g[mask], tp.shape).astype(np.int64)
        cube = np.stack([tp, pp, gg], -1).transpose(1, 0, 2, 3)
        total = np.zeros((len(names), S, L, 3), np.int64)
        np.add.at(total, membership, cube)
        flat = total.reshape(len(names), -1).astype(float)
        rng = np.random.default_rng(r["seed"])
        nleft = len(r["left_settings"])
        for start in range(0, 10000, 128):
            n = min(128, 10000 - start)
            draw = rng.choice(np.arange(len(names)), (n, len(names)), replace=True)
            counts = np.zeros((n, len(names)))
            np.add.at(counts, (np.arange(n)[:, None], draw), 1)
            z = (counts @ flat).reshape(n, S, L, 3)
            score = metric_values(z[..., 0], z[..., 1], z[..., 2], ix)
            left = score[:, :nleft].mean(1)
            right = score[:, nleft:].mean(1)
            new = np.stack([left, right, left - right], 1)
            equal(
                new, saved[start : start + n], "Independent strict bootstrap mismatch"
            )
            values_checked += new.size
        for mi, name in enumerate(["macro", "micro", "micro_macro_labels"]):
            equal(
                np.quantile(saved[:, 2, mi], [0.025, 0.975]),
                r["metrics"][name]["ci95"],
                "Strict exported 95% interval",
            )
            if name != "macro":
                equal(
                    np.quantile(saved[:, 2, mi], [0.0025, 0.9975]),
                    r["metrics"][name]["adjusted_interval"],
                    "Strict exported family interval",
                )
        strict.append(
            dict(
                question_id=r["question_id"],
                n_main=r["n_original_common"],
                n_strict=r["n_common"],
                original_micro=r["original_metrics"]["micro"],
                strict_micro=r["metrics"]["micro"],
                strict_macro=r["metrics"]["macro"],
                main95=classify(r["original_metrics"]["micro"]["ci95"]),
                strict95=classify(r["metrics"]["micro"]["ci95"]),
                main_adjusted=classify(
                    r["original_metrics"]["micro"]["adjusted_interval"]
                ),
                strict_adjusted=classify(r["metrics"]["micro"]["adjusted_interval"]),
                repairs_remain=r["repairs_remain"],
                excluded_texts=r["excluded_texts"],
                gold_support=r["group_gold_support"],
            )
        )
    c.require(
        len(strict) == 10 and values_checked == 900000, "Strict verification coverage"
    )
    report = dict(
        status="passed",
        normalized_rows=len(rows),
        source_metric_rows=scalar_checks,
        csv_scalar_checks=csv_checks,
        registered_questions=len(expected),
        strict_independent_draw_values=values_checked,
        strict_comparisons=strict,
        source_sha256=c.BINDINGS,
        adapter_note="An initial normalization identifier used an extra primary segment for four matched-information questions; the registry guard rejected it before any ledger was saved. Corrected IDs match the frozen manifest exactly.",
    )
    p.save("independent_verification.json", report)
    print(
        "Verified",
        len(rows),
        "normalized contrasts;",
        scalar_checks,
        "source metric rows;",
        values_checked,
        "new strict-bootstrap values.",
        flush=True,
    )
    for row in strict:
        v = row["strict_micro"]
        print(
            row["question_id"],
            row["n_strict"],
            [
                round(v["difference"] * 100, 3),
                [round(x * 100, 3) for x in v["adjusted_interval"]],
            ],
            row["main_adjusted"],
            "->",
            row["strict_adjusted"],
            flush=True,
        )


if __name__ == "__main__":
    main()
