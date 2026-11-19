from analysis.micro_f1.combined import prepare as p
from analysis.common import micro_f1_resampling as h
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

c = p.c
OUT = p.BASE / "broad_strict"


def main():
    c.read(p.BASE / "targeted_diagnostic_addendum.json")
    c.require(not OUT.exists(), "Preserve prior strict run")
    OUT.mkdir()
    registry = c.read(p.ROOTS[1] / "comparison_registry_v2.json")
    index = c.read(p.ROOTS[2] / "accepted_analysis_index.json")
    data = {}
    tasks = defaultdict(list)
    for f in index["broad"]["outputs"]:
        path = c.ROOT / f
        r = c.read(path)
        if not r["macro_labels"]:
            continue
        unit = next(
            (
                u
                for u in registry["units"]
                if u["study"] == "broad"
                and u["endpoint"].replace("/", "__") == r["endpoint"]
            )
        )
        old = c.read(c.ROOT / unit["source"])
        ep = old["endpoint"]
        a = h.read_arrays(path.parent / "analysis_arrays.npz")
        settings = [tuple(x["setting"]) for x in r["own_valid"]["runs"]]
        c.require(r["ids"] == ep["scorable_ids"], "Original ID order mismatch")
        nvalid = np.full(a["valid"].shape, -1, np.int8)
        repairs = np.full_like(nvalid, -1)
        data[r["endpoint"]] = dict(
            r=r,
            a=a,
            ep=ep,
            settings=settings,
            nvalid=nvalid,
            repairs=repairs,
            path=path,
        )
        for j, (condition, repeat) in enumerate(settings):
            if condition not in ["rag_full", "rag_small", "zero_shot"]:
                continue
            for i, key in enumerate(r["ids"]):
                tasks[ep["phase"], key, condition, repeat].append((r["endpoint"], j, i))
    cache = (
        c.ROOT
        / "experiments/astra/sq2_broad_reader_review_20261104_v1/predictions.jsonl"
    )
    assembly = c.read(cache.parent / "assembly.json")
    c.require(c.sha(cache) == assembly["outputs"][c.rel(cache)], "Cache not sealed")
    seen = set()
    checks = 0
    for row in c.lines(cache):
        key = (row["phase"], row["key"], row["condition"], row["repeat"])
        if key not in tasks:
            continue
        c.require(key not in seen, "Duplicate cache answer")
        seen.add(key)
        c.require(
            sum((v is not None for v in row["votes"])) == row["n_valid"],
            "Vote count metadata mismatch",
        )
        for ep, j, i in tasks[key]:
            d = data[ep]
            r = d["r"]
            a = d["a"]
            field = r["dimension"]
            answer = row["result"]
            ok = answer is not None and answer.get(field) is not None
            c.require(bool(a["valid"][j, i]) == ok, "Cached validity differs")
            if ok:
                h.equal(
                    a["predictions"][j, i],
                    [x in answer[field] for x in r["labels"]],
                    "Cached strict prediction identity",
                )
                if field != "legal":
                    c.require(
                        bool(a["gates"][j, i]) == answer["hate"], "Hate gate identity"
                    )
            d["nvalid"][j, i] = row["n_valid"]
            d["repairs"][j, i] = row["repairs"]
            checks += 1
    c.require(seen == set(tasks), "Missing cached answers")
    outputs = []
    for endpoint, d in data.items():
        r = d["r"]
        a = d["a"]
        g = a["gold"]
        settings = d["settings"]
        ix = [r["labels"].index(x) for x in r["macro_labels"]]
        for old in r["contrasts"]:
            if not old.get("question_id"):
                continue
            keys = [tuple(x) for x in old["left_settings"] + old["right_settings"]]
            js = [settings.index(x) for x in keys]
            nleft = len(old["left_settings"])
            pred = a["predictions"][js]
            valid = a["valid"][js]
            nv = d["nvalid"][js]
            rp = d["repairs"][js]
            c.require((nv >= 0).all(), "Missing vote metadata")
            ordinary = valid.all(0)
            mask = ordinary & (nv == 3).all(0)
            c.require(
                [key for key, keep in zip(r["ids"], ordinary) if keep]
                == old["common_ids"],
                "Original pair IDs changed",
            )
            plain = h.values(
                h.encode(g, pred, np.broadcast_to(ordinary, valid.shape)).sum(1), ix
            )
            h.equal(
                plain[:nleft].mean(0) - plain[nleft:].mean(0),
                [old["metrics"][m]["difference"] for m in h.METHODS],
                "Original paired point reproduction",
            )
            bs = h.bootstrap(
                g,
                pred,
                np.broadcast_to(mask, valid.shape),
                ix,
                d["ep"]["row_clusters"],
                nleft,
                old["seed"],
            )
            rec = h.contrast_summary(
                bs, old["left_settings"], old["right_settings"], 10
            )
            scores = []
            for j in range(len(keys)):
                sk = [
                    f1_score(
                        g[mask][:, ix],
                        pred[j, mask][:, ix],
                        average="macro",
                        zero_division=0,
                    ),
                    f1_score(g[mask], pred[j, mask], average="micro", zero_division=0),
                    f1_score(
                        g[mask][:, ix],
                        pred[j, mask][:, ix],
                        average="micro",
                        zero_division=0,
                    ),
                ]
                h.equal(sk, bs["point"][j], "Independent strict sklearn scores")
                scores.append(sk)
            name = endpoint + "__" + old["right_settings"][0][0]
            folder = OUT / name
            folder.mkdir()
            for filename, arrays in [
                (
                    "arrays.npz",
                    dict(
                        gold=g, pred=pred, valid=valid, nvalid=nv, repairs=rp, mask=mask
                    ),
                ),
                ("draws.npz", dict(draws=bs["draws"])),
            ]:
                with (folder / filename).open("xb") as f:
                    np.savez_compressed(f, **arrays)
            rec.update(
                question_id=old["question_id"],
                endpoint=endpoint,
                source=c.rel(d["path"]),
                labels=r["labels"],
                macro_labels=r["macro_labels"],
                ids=r["ids"],
                clusters=d["ep"]["row_clusters"],
                common_ids=[x for x, k in zip(r["ids"], mask) if k],
                n_original_common=int(ordinary.sum()),
                original_metrics=old["metrics"],
                repairs_remain=int((rp[:, mask] > 0).sum()),
                excluded_texts=int((ordinary & ~mask).sum()),
                group_gold_support=dict(
                    main=g[ordinary].sum(0).tolist(), strict=g[mask].sum(0).tolist()
                ),
                point_by_setting=scores,
                limitation="Selection on valid votes; repaired votes remain allowed. No causal repair effect or replacement of the original population.",
            )
            p.save(str((folder / "results.json").relative_to(p.BASE)), rec)
            outputs.append(c.rel(folder / "results.json"))
            print(
                endpoint,
                old["right_settings"][0][0],
                int(ordinary.sum()),
                "->",
                int(mask.sum()),
                "strict micro",
                rec["metrics"]["micro"]["difference"],
                flush=True,
            )
    c.require(len(outputs) == 10, "All ten strict comparisons required")
    p.save(
        "broad_strict/index.json",
        dict(
            status="calculated; independent verification pending",
            outputs=outputs,
            raw_answer_keys=len(seen),
            endpoint_answer_checks=checks,
            numerical_checks=dict(h.CHECKS),
            max_errors=dict(h.ERRORS),
            source_sha256=c.BINDINGS,
            new_model_calls=0,
        ),
    )


if __name__ == "__main__":
    main()
