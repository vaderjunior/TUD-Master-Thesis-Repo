from analysis.knowledge.micro_f1 import components_guidelines as p
from pathlib import Path
from itertools import combinations
import numpy as np
import pyarrow.parquet as pq


def run():
    idx = p.c.read(p.BASE / "analysis_v2/analysis_index.json")
    results = [p.c.read(p.c.ROOT / x) for x in idx["outputs"]]
    allcounts = 0
    goldrows = 0
    overlaps = []
    subsets = []
    unique = {r["unit"]["id"]: r for r in results}
    for (x, a), (y, b) in combinations(unique.items(), 2):
        if (
            a["unit"]["study"] == b["unit"]["study"]
            and a["unit"]["dimension"] == b["unit"]["dimension"]
        ):
            overlaps.append(
                dict(left=x, right=y, shared_ids=len(set(a["ids"]) & set(b["ids"])))
            )
    for r in results:
        if r["unit"]["study"] != "german_guidelines":
            continue
        u = r["unit"]
        phase = u["endpoint"].split("/")[0]
        a = p.h.read_arrays(p.c.ROOT / r["arrays"])
        original = p.h.read_arrays(p.c.ROOT / u["prediction_path"])
        rows = pq.read_table(
            p.c.bind(p.c.ROOT / "data/processed" / f"{phase}.parquet")
        ).to_pylist()
        bykey = {
            x["lang"] + "/" + str(x["id"]): x for x in rows if x["legal"] is not None
        }
        assert set(bykey) == set(r["ids"])
        gold = np.array(
            [
                [label in bykey[key]["legal"] for label in r["labels"]]
                for key in r["ids"]
            ],
            bool,
        )
        p.h.equal(gold, a["gold"], "Direct German parquet gold")
        goldrows += len(gold)
        design = p.c.read(p.c.ROOT / u["resampling_source"])
        indices = np.flatnonzero(a["common"])
        members = [indices[x] for x in design["cluster_rows"]]
        full = []
        for ix in members:
            y = gold[ix]
            x = a["pred"][:, ix]
            full.append(
                np.stack(
                    [
                        (~x & ~y).sum(1),
                        (x & ~y).sum(1),
                        (~x & y).sum(1),
                        (x & y).sum(1),
                    ],
                    -1,
                )
            )
        full = np.array(full)
        matrix = full.reshape(len(full), -1)
        rng = np.random.default_rng(design["seed"])
        for start in range(0, 10000, 64):
            size = min(64, 10000 - start)
            picks = np.concatenate(
                [
                    rng.choice(
                        s["cluster_indices"],
                        (size, len(s["cluster_indices"])),
                        replace=True,
                    )
                    for s in design["strata"]
                ],
                1,
            )
            freq = np.array([np.bincount(x, minlength=len(full)) for x in picks])
            count = (freq @ matrix).reshape(size, 3, 3, 3, 4)
            p.h.equal(
                count,
                original["bootstrap_confusions"][start : start + size],
                "All four saved German confusion entries",
            )
            allcounts += count.size
        for subset, q in r["quality"].items():
            ranks = {
                metric: sorted(
                    q["groups"],
                    key=lambda k: q["groups"][k]["scores"][metric],
                    reverse=True,
                )
                for metric in ["macro", "micro"]
            }
            subsets.append(
                dict(
                    population=phase,
                    subset=subset,
                    n=q["n"],
                    rankings=ranks,
                    ranking_differs=ranks["macro"] != ranks["micro"],
                )
            )
    assert allcounts == 2160000 and goldrows == 352
    p.c.bind(Path(__file__))
    p.save(
        p.BASE / "additional_checks.json",
        dict(
            status="passed",
            original_german_gold_rows=goldrows,
            all_four_original_bootstrap_confusion_values=allcounts,
            within_study_endpoint_overlaps=overlaps,
            german_quality_rankings=subsets,
            checks=dict(p.h.CHECKS),
            source_sha256=p.c.BINDINGS,
        ),
    )
    print(
        "Additional checks passed:",
        goldrows,
        "original German gold rows;",
        allcounts,
        "original TN/FP/FN/TP entries",
        flush=True,
    )


if __name__ == "__main__":
    run()
