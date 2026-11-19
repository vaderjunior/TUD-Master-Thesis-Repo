import numpy as np
from analysis.classification.combined import metrics as p


def run():
    cache = {}
    bge = {}
    checks = 0
    comparisons = 0

    def get(path):
        path = p.ROOT / path
        if path not in cache:
            cache[path] = p.load(path)
        return cache[path]

    def equal(x, y, label):
        nonlocal checks, comparisons
        if np.asarray(x).shape != np.asarray(y).shape or not np.array_equal(x, y):
            raise AssertionError(label)
        checks += np.asarray(x).size
        comparisons += 1

    master = p.a.read(p.a.g.LINEAR / "data/inference.json")
    order = {r["key"]: j for j, r in enumerate(master)}
    broad = p.a.read(p.a.g.LINEAR / "data/scopes.json")["broad"]
    border = {r["key"]: j for j, r in enumerate(broad)}
    labels = {
        h: v["labels"] for h, v in p.a.read(p.TUNE / "data/train.json")["heads"].items()
    }
    labels["hate"] = ["not_hate", "hate"]
    gm = {m["id"]: j for j, m in enumerate(p.a.read(p.a.GPU / "models.json"))}
    qgates = {
        (r["phase"], r["key"], r["condition"], r["repeat"]): (
            None if r["result"] is None else int(r["result"]["hate"])
        )
        for r in p.a.rows(p.a.BROAD / "predictions.jsonl")
    }
    for e in p.a.read(p.BASE / "extraction_complete.json")["endpoints"]:
        ep = e["endpoint"]
        slug = ep.replace("/", "__")
        phase, head = ep.split("/")
        folder = p.BASE / "endpoints" / slug
        md = p.a.read(folder / "metadata.json")
        z = p.load(folder / "aligned.npz")
        ids = md["ids"]
        labs = md["labels"]
        old = p.a.read(p.a.MAIN / "endpoints" / slug / "results.json")
        for j, m in enumerate(md["models"]):
            group = m["group"]
            source = m["source"]
            gate = None
            if group.startswith(("qwen/", "historical_")):
                k = old["settings"].index([m["original_method"], m["seed"]])
                v = get(source)["predicted"][k].astype(np.uint8)
                equal(z["valid"][j], get(source)["valid"][k], ep + group + "/valid")
                if group.startswith("qwen/"):
                    gate = np.array(
                        [
                            qgates[(phase, key, m["original_method"], m["seed"])] or 0
                            for key in ids
                        ],
                        dtype=np.uint8,
                    )
                elif head == "hate":
                    gate = v[:, 1]
                elif m["gate_available"]:
                    h = p.a.read(
                        p.a.MAIN / "endpoints" / (phase + "__hate") / "results.json"
                    )
                    hr = p.a.read(p.a.BROAD / "endpoint_setup.json")[phase + "/hate"][
                        "scorable_ids"
                    ]
                    rows = {key: i for i, key in enumerate(hr)}
                    k = h["settings"].index([m["original_method"], m["seed"]])
                    gate = get(h["encoded_path"])["predicted"][
                        k, [rows[key] for key in ids], 1
                    ]
            elif group.startswith("gpu/") and "paired_full_predictions" not in source:
                k = gm[m["original_method"]]
                v = get(source)["gated"][k]
                gate = get(source)["gate"][k]
            elif "paired_full_predictions" in source:
                policy = group.rsplit("/", 1)[1]
                i = m["seed"] - 10
                ix = [order[key] for key in ids]
                cols = [labels[head].index(l) for l in labs]
                v = get(source)[policy + "__new__" + head][i][np.ix_(ix, cols)]
                gate = get(source)[policy + "__new__hate"][i, ix, 1]
            elif group.startswith("cpu/"):
                policy = group.rsplit("/", 1)[1]
                ix = [(border if "full_neural" in group else order)[key] for key in ids]
                cols = [labels[head].index(l) for l in labs]
                v = get(source)[policy + "_" + head][np.ix_(ix, cols)]
                gate = get(source)[policy + "_hate"][ix, 1]
            elif group.startswith(("curated_original/", "curated_expanded/")):
                ix = [order[key] for key in ids]
                cols = [labels[head].index(l) for l in labs]
                v = get(source)[head][np.ix_(ix, cols)]
                gate = get(source)["hate"][ix, 1]
            elif group == "curated_bge/selected":
                if source not in bge:
                    bge[source] = {
                        r["key"]: r["result"] for r in p.a.rows(p.ROOT / source)
                    }
                rr = bge[source]
                v = []
                gate = []
                for key in ids:
                    r = rr[key]
                    gate.append(int(r["hate"]))
                    chosen = (
                        {"hate" if r["hate"] else "not_hate"}
                        if head == "hate"
                        else (
                            {r[head]}
                            if head == "severity" and r[head] is not None
                            else set() if r[head] is None else set(r[head])
                        )
                    )
                    v.append([int(label in chosen) for label in labs])
                v = np.array(v, dtype=np.uint8)
                gate = np.array(gate, dtype=np.uint8)
            else:
                raise AssertionError("Unrecognised group " + group)
            equal(
                z["predictions"][j],
                v,
                ep + group + "/" + str(m["seed"]) + "/prediction",
            )
            if gate is not None and m["gate_available"]:
                equal(
                    z["gate"][j], np.asarray(gate, dtype=np.uint8), ep + group + "/gate"
                )
            if not group.startswith("qwen/"):
                assert z["valid"][j].all()
        print("Direct source cells checked:", ep, flush=True)
    p.save(
        "alignment_verification.json",
        dict(
            status="passed",
            created_utc=p.a.utc(),
            array_comparisons=comparisons,
            exact_source_cells=checks,
            source="Explicit canonical input ID joins, original class-column lookup and all model-source/policy mappings. Saved predictions only.",
            verifier_sha256=p.a.g.sha(__file__),
        ),
    )


if __name__ == "__main__":
    with p.a.g.offline():
        run()
