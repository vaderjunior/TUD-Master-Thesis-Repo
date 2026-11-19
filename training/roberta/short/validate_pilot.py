import math
from pathlib import Path
import numpy as np
from training.roberta.short import common as s

c = s.c


def main():
    receipt_path = s.development_receipt("pilot")
    receipt = c.read(receipt_path)
    c.require(
        receipt["passed"] and receipt["excluded_from_scientific_results"],
        "Pilot not passed/excluded",
    )
    c.require(receipt["sources"] == s.hashes(s.source_paths()), "Pilot sources changed")
    c.verify(receipt["artifacts"])
    c.require(
        receipt["draft_sha256"] == c.sha(s.BASE / "draft.json"), "Pilot draft changed"
    )
    tests = c.read(s.development_receipt("tests"))
    c.require(
        tests["passed"] and tests["sources"] == receipt["sources"], "Tests changed"
    )
    data = s.read_data()
    rows, metas = [], {}
    for name in sorted(receipt["artifacts"]):
        if not name.endswith("/training.json"):
            continue
        path = c.ROOT / name
        meta = c.read(path)
        job, cfg = meta["job"], meta["configuration"]
        spec = data["heads"][job["head"]]
        fit = [
            r
            for r in spec["rows"]
            if data["folds"]["42"][r["feature_row"]] != job["fold"]
        ]
        held = [
            r
            for r in spec["rows"]
            if data["folds"]["42"][r["feature_row"]] == job["fold"]
        ]
        c.require(meta["pilot_only"] and job["fold"] == 0, "Unexpected pilot identity")
        c.require(
            meta["fit_ids"] == [r["id"] for r in fit]
            and meta["held_ids"] == [r["id"] for r in held],
            "Pilot membership differs",
        )
        c.require(
            not {r["cluster"] for r in fit} & {r["cluster"] for r in held},
            "Pilot group leakage",
        )
        c.require(
            meta["class_weights"]
            == c.class_weights(
                job["head"],
                [r["target"] for r in fit],
                len(spec["labels"]),
                cfg["loss"] == "balanced",
            ),
            "Fold weights differ",
        )
        total = math.ceil(len(fit) / 16) * cfg["epochs"]
        c.require(
            meta["planned_steps"] == total
            and meta["attempted_steps"] == len(meta["steps"]) == min(8, total),
            "Step count differs",
        )
        successful = positive = skipped = 0
        warmup = math.ceil(0.1 * total)
        for index, step in enumerate(meta["steps"], 1):
            factor = (
                successful / max(1, warmup)
                if successful < warmup
                else max(0.0, (total - successful) / max(1, total - warmup))
            )
            c.require(
                step["step"] == index
                and abs(step["lr"] - cfg["learning_rate"] * factor) < 1e-14,
                "Learning-rate trace differs",
            )
            c.require(
                step["skipped"] == (step["scale_after"] < step["scale_before"]),
                "AMP trace differs",
            )
            c.require(
                math.isfinite(step["scale_after"]) and step["scale_after"] > 0,
                "Invalid AMP scale",
            )
            if step["skipped"]:
                skipped += 1
            else:
                c.require(
                    math.isfinite(float(step["grad_norm"])),
                    "Nonfinite accepted gradient",
                )
                successful += 1
                positive += int(step["lr"] > 0)
        c.require(
            successful == meta["successful_steps"]
            and skipped == meta["skipped_steps"]
            and positive > 0,
            "Ineffective pilot",
        )
        c.require(
            all(math.isfinite(v) and v > 0 for v in meta["changes"].values()),
            "Unchanged/nonfinite watched parameters",
        )
        z = np.load(path.parent / "held_logits.npy", allow_pickle=False)
        probe = np.load(path.parent / "probe_logits.npy", allow_pickle=False)
        c.require(
            z.shape == (len(held), len(spec["labels"])) and np.isfinite(z).all(),
            "Invalid held logits",
        )
        c.require(
            probe.shape == (min(3, len(held)), len(spec["labels"]))
            and np.isfinite(probe).all(),
            "Invalid probe logits",
        )
        errors = []
        epoch_paths = sorted((path.parent / "epoch_logits").glob("*.npy"))

        if "/parity/original/" not in name:
            c.require(len(epoch_paths) == len(meta["epochs"]), "Missing epoch logits")
            y = np.asarray([r["target"] for r in held])
            for e in meta["epochs"]:
                q = np.load(
                    path.parent / "epoch_logits" / f"{e['epoch']:03d}.npy",
                    allow_pickle=False,
                ).astype(float)
                c.require(
                    q.shape == z.shape and np.isfinite(q).all(), "Invalid epoch logits"
                )
                if job["head"] in ("hate_type", "target_group"):
                    loss = float(np.mean(np.logaddexp(0, q) - y * q))
                else:
                    maximum = q.max(1)
                    lse = maximum + np.log(np.exp(q - maximum[:, None]).sum(1))
                    loss = float(np.mean(lse - q[np.arange(len(y)), y]))
                error = abs(loss - e["held_unweighted_loss"])
                c.require(error < 2e-5, "Independent epoch loss differs")
                errors.append(error)
            c.require(
                np.array_equal(z, np.load(epoch_paths[-1], allow_pickle=False)),
                "Final epoch/held logits differ",
            )
        export = meta["export_check"]
        if export:
            c.require(
                export["clean_reload"]
                and export["max_abs_logit_difference"] <= export["tolerance"] == 1e-4,
                "CPU reload failed",
            )
        rows.append(
            dict(
                path=name,
                head=job["head"],
                loss=cfg["loss"],
                fit_rows=len(fit),
                held_rows=len(held),
                attempted_steps=len(meta["steps"]),
                successful_steps=successful,
                positive_learning_rate_updates=positive,
                skipped_steps=skipped,
                epoch_losses_checked=len(errors),
                largest_epoch_loss_error=max(errors, default=None),
                CPU_export_check=export,
            )
        )
        metas[name] = meta
    c.require(len(rows) == 10, "Pilot coverage differs")
    original = next(n for n in metas if "/parity/original/" in n)
    derivative = next(n for n in metas if "/parity/new/" in n)
    fields = [
        "job",
        "configuration",
        "fit_ids",
        "held_ids",
        "class_weights",
        "planned_steps",
        "attempted_steps",
        "successful_steps",
        "skipped_steps",
        "steps",
        "epochs",
        "changes",
    ]
    c.require(
        all(metas[original][k] == metas[derivative][k] for k in fields),
        "Old/new numerical traces differ",
    )
    for filename in ["held_logits.npy", "probe_logits.npy"]:
        c.require(
            np.array_equal(
                np.load((c.ROOT / original).parent / filename, allow_pickle=False),
                np.load((c.ROOT / derivative).parent / filename, allow_pickle=False),
            ),
            "Old/new logit parity failed",
        )
    c.require(
        sum(r["attempted_steps"] for r in rows)
        == receipt["summary"]["attempted_steps"],
        "Pilot total differs",
    )
    c.require(
        sum(r["skipped_steps"] for r in rows) == receipt["summary"]["skipped_steps"],
        "Pilot skip total differs",
    )
    result = dict(
        status="passed",
        source_sha256=c.sha(Path(__file__)),
        pilot_receipt_sha256=c.sha(receipt_path),
        test_receipt_sha256=c.sha(s.development_receipt("tests")),
        fit_checks=rows,
        exact_old_new_trace_and_logits=True,
        all_pilots_have_positive_learning_rate_updates=True,
        scientific_observations=0,
        new_training_or_inference=False,
        Qwen_calls=0,
        interpretation="One recorded AMP gradient overflow was skipped; loss scaling fell and subsequent finite updates resumed. All ten fits changed watched encoder/head parameters. This technical check establishes no performance improvement.",
    )
    s.save(s.BASE / "development/pilot_review_20261012_v1.json", result)
    print(
        c.encoded(
            dict(
                status=result["status"],
                fits=len(rows),
                attempts=sum(r["attempted_steps"] for r in rows),
                positive_lr_updates=sum(
                    r["positive_learning_rate_updates"] for r in rows
                ),
                skipped=sum(r["skipped_steps"] for r in rows),
                independently_checked_epoch_losses=sum(
                    r["epoch_losses_checked"] for r in rows
                ),
                maximum_epoch_loss_error=max(
                    r["largest_epoch_loss_error"] or 0 for r in rows
                ),
                exact_old_new_parity=True,
            )
        ).decode()
    )


if __name__ == "__main__":
    with s.isolated():
        main()
