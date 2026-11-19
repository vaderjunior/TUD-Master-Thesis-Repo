import argparse
import copy
from pathlib import Path
from training.roberta import common as c


def prepare():

    import pyarrow.dataset, pyarrow.parquet, pandas as pd
    from training.roberta import tokenizer as guard
    from training.frozen_features import compact as old

    if (c.BASE / "preparation.json").exists():
        validate_preparation()
        print("Existing TRAIN/text-only preparation verified; no model fit.")
        return
    original = c.read(c.BGE / "plan.json")
    c.verify(
        {k: v for k, v in original["prepared"].items() if k.endswith("train.json")}
    )
    data = copy.deepcopy(c.read(c.BGE / "train.json"))
    representation = c.read(c.FEATURES / "plan.json")
    bindings = {
        c.rel(c.BGE / "plan.json"): c.sha(c.BGE / "plan.json"),
        c.rel(c.BGE / "train.json"): c.sha(c.BGE / "train.json"),
        c.rel(c.FEATURES / "plan.json"): c.sha(c.FEATURES / "plan.json"),
    }
    for name in ("feature_inputs.json", "scope_mapping.json"):
        key = c.rel(c.FEATURES / name)
        c.require(
            c.sha(c.FEATURES / name) == representation["inputs"][key],
            "Representation text/mapping changed",
        )
        bindings[key] = representation["inputs"][key]
    features = {r["key"]: r for r in c.read(c.FEATURES / "feature_inputs.json")}
    mapping = c.read(c.FEATURES / "scope_mapping.json")
    train_map = {r["source_id"]: r["key"] for r in mapping["train"]}
    c.require(
        set(train_map) == set(data["gate_ids"]) and len(train_map) == 147,
        "TRAIN pool differs",
    )
    data["texts"] = [features[train_map[rid]]["text"] for rid in data["gate_ids"]]
    folds = data["folds"]["42"]
    c.require(len(folds) == 147 and set(folds) == set(range(5)), "Wrong TRAIN folds")
    fields = dict(
        hate="gate",
        hate_type="hate_types",
        target_group="target_groups",
        severity="severity",
    )
    support = {}
    for head in c.HEADS:
        spec = data["heads"][head]
        meta_path = c.CORE / f"encoder_data/kb/{head}_meta.json"
        meta = c.read(meta_path)
        source = {}
        c.require(
            c.sha(meta_path) == original["inputs"][c.rel(meta_path)],
            "Original head metadata changed",
        )
        for part in ("train", "val"):
            path = c.CORE / f"encoder_data/kb/{head}_{part}.parquet"
            key = c.rel(path)
            c.require(
                c.sha(path) == original["inputs"][key],
                "Original TRAIN partition changed",
            )
            bindings[key] = original["inputs"][key]
            frame = pd.read_parquet(path)
            c.require(
                len(frame) == meta["counts"][part], "Wrong original partition size"
            )
            for row in frame.to_dict("records"):
                rid = str(row["id"])
                value = row[fields[head]]
                y = (
                    int(value)
                    if head == "hate"
                    else (
                        meta["labels"].index(value)
                        if head == "severity"
                        else [int(v in value) for v in meta["labels"]]
                    )
                )
                c.require(rid not in source, "Repeated TRAIN identity")
                source[rid] = dict(text=row["text"], target=y, part=part)
        bindings[c.rel(meta_path)] = c.sha(meta_path)
        c.require(
            len(spec["rows"]) == c.protocol()["training_counts"][head]
            and set(source) == {r["id"] for r in spec["rows"]},
            "Head annotation budget mismatch",
        )
        c.require(
            spec["labels"] == meta["labels"] and spec["kind"] == meta["kind"],
            "Head semantics differ",
        )
        for row in spec["rows"]:
            text = data["texts"][row["feature_row"]]
            src = source[row["id"]]
            c.require(
                text == src["text"]
                and row["target"] == src["target"]
                and row["original_partition"] == src["part"],
                "TRAIN text/target/partition mismatch",
            )
            c.require(
                row["text_hash"] == c.signature(text)
                and row["cluster"] == c.signature(old.norm(text)),
                "TRAIN text signature mismatch",
            )
        support[head] = []
        for fold in range(5):
            fit = [r for r in spec["rows"] if folds[r["feature_row"]] != fold]
            held = [r for r in spec["rows"] if folds[r["feature_row"]] == fold]
            c.require(
                fit
                and held
                and not {r["cluster"] for r in fit} & {r["cluster"] for r in held},
                "Empty/overlapping fold",
            )
            support[head].append(
                dict(
                    fold=fold,
                    fit=len(fit),
                    held=len(held),
                    class_weights=c.class_weights(
                        head, [r["target"] for r in fit], len(spec["labels"]), True
                    ),
                )
            )
    all_train_clusters = {r["cluster"] for r in data["heads"]["hate"]["rows"]}
    for fold in range(5):
        gate_fit = {
            r["cluster"]
            for r in data["heads"]["hate"]["rows"]
            if folds[r["feature_row"]] != fold
        }
        for head in c.HEADS:
            held = {
                r["cluster"]
                for r in data["heads"][head]["rows"]
                if folds[r["feature_row"]] == fold
            }
            c.require(not gate_fit & held, "Fine-head held group leaked through gate")
    keys = sorted({r["key"] for name in ("core", "broad") for r in mapping[name]})
    inputs = [features[key] for key in keys]
    c.require(
        len(inputs) == 19082
        and len(mapping["core"]) == 970
        and len(mapping["broad"]) == 18275,
        "Changed inference universe",
    )
    c.require(
        not all_train_clusters & {c.signature(old.norm(r["text"])) for r in inputs},
        "TRAIN/inference normalized-text overlap",
    )
    positions = {k: i for i, k in enumerate(keys)}
    scopes = {
        scope: [
            dict(source_id=r["source_id"], position=positions[r["key"]], key=r["key"])
            for r in mapping[scope]
        ]
        for scope in ("core", "broad")
    }

    c.save(c.BASE / "protocol.json", c.protocol())
    c.save(c.BASE / "data/train.json", data)
    c.save(c.BASE / "data/inference.json", inputs)
    c.save(c.BASE / "data/scopes.json", scopes)
    c.save(c.BASE / "data/fold_support.json", support)
    cache = c.read(c.CORE / "plan.json")["base_caches"]["roberta-base"]
    for name, digest in cache["files"].items():
        c.require(
            c.sha(Path(cache["snapshot"]) / name) == digest, "Pinned base cache differs"
        )
    bindings[c.rel(guard.REFERENCE)] = guard.REFERENCE_SHA
    products = {
        c.rel(c.BASE / p): c.sha(c.BASE / p)
        for p in (
            "protocol.json",
            "data/train.json",
            "data/inference.json",
            "data/scopes.json",
            "data/fold_support.json",
        )
    }
    c.save(
        c.BASE / "preparation.json",
        dict(
            created_utc=c.utc(),
            environment=c.environment(),
            inputs=bindings,
            products=products,
            base_cache=cache,
            train_input_count=147,
            inference_input_count=19082,
            search_fits=len(c.search_jobs()),
            max_final_fits=60,
            evaluation_gold_read=False,
            original_partition_counts={
                h: dict(
                    train=sum(
                        r["original_partition"] == "train"
                        for r in data["heads"][h]["rows"]
                    ),
                    val=sum(
                        r["original_partition"] == "val"
                        for r in data["heads"][h]["rows"]
                    ),
                )
                for h in c.HEADS
            },
        ),
    )
    validate_preparation()
    print(
        "Prepared 147 TRAIN texts, 720 CV fits, at most60 final fits and19082 text-only inference inputs.",
        flush=True,
    )


def validate_preparation(check_cache=True):
    p = c.read(c.BASE / "preparation.json")
    c.verify(p["inputs"])
    c.verify(p["products"])
    c.require(
        c.read(c.BASE / "protocol.json") == c.ready(c.protocol()),
        "Scientific protocol changed",
    )
    c.require(p["environment"] == c.environment(), "Package environment changed")
    if check_cache:
        for name, digest in p["base_cache"]["files"].items():
            c.require(
                c.sha(Path(p["base_cache"]["snapshot"]) / name) == digest,
                "Base checkpoint changed",
            )
    return p


def seal():
    p = validate_preparation()
    sources = c.source_hashes()
    test = c.read(c.BASE / "development/tests.json")
    pilot = c.read(c.BASE / "development/pilot_complete.json")
    c.require(
        test["passed"] and test["sources"] == sources,
        "Tests need rerunning after source changes",
    )
    c.require(
        pilot["passed"] and pilot["sources"] == sources,
        "Numerical pilot needs rerunning after source changes",
    )
    c.verify(pilot["artifacts"])
    path = c.BASE / "plan.json"
    if path.exists():
        validate()
        print("Existing sealed execution verified.")
        return
    c.save(
        path,
        dict(
            created_utc=c.utc(),
            status="frozen_before_scientific_fitting",
            protocol_sha256=c.sha(c.BASE / "protocol.json"),
            inputs={
                **p["inputs"],
                **p["products"],
                **sources,
                c.rel(c.BASE / "preparation.json"): c.sha(c.BASE / "preparation.json"),
                c.rel(c.BASE / "development/tests.json"): c.sha(
                    c.BASE / "development/tests.json"
                ),
                c.rel(c.BASE / "development/pilot_complete.json"): c.sha(
                    c.BASE / "development/pilot_complete.json"
                ),
            },
            environment=p["environment"],
            base_cache=p["base_cache"],
            jobs=c.search_jobs(),
            max_final_jobs=60,
            pilot=pilot["summary"],
            resources=c.protocol()["storage"],
        ),
    )
    validate()
    print("Execution sealed. No scientific fitting started.", flush=True)


def validate(check_cache=True):
    p = c.read(c.BASE / "plan.json")
    c.verify(p["inputs"])
    c.require(
        p["environment"] == c.environment(), "Runtime package environment changed"
    )
    c.require(p["jobs"] == c.search_jobs(), "Search schedule changed")
    if check_cache:
        for name, digest in p["base_cache"]["files"].items():
            c.require(
                c.sha(Path(p["base_cache"]["snapshot"]) / name) == digest,
                "Base cache changed",
            )
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "seal", "validate"])
    mode = parser.parse_args().mode
    {"prepare": prepare, "seal": seal, "validate": validate}[mode]()
