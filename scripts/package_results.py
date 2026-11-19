import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from analysis.common import study as c
from analysis.common.api import save_new

BASE = c.ROOT / "backups/astra_20261008_v1"
CAP = BASE / "capsule_v2"
FB = c.ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        f.write("\n")


def main():
    c.require(
        not CAP.exists(), "Preserve existing capsule; use its manifest to recheck"
    )
    CAP.mkdir(parents=True)
    origins = {}

    def copy(source, expected=None):
        source = Path(source)
        digest = c.sha(source)
        if expected is not None:
            c.require(
                digest == expected, "Source differs from original completion receipt"
            )
        dest = CAP / "originals" / c.rel(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(source, dest)
        c.require(c.sha(dest) == digest, "Copied bytes differ")
        origins[c.rel(source)] = {
            "sha256": digest,
            "capsule_path": dest.relative_to(CAP).as_posix(),
            "bytes": source.stat().st_size,
        }
        return dest.relative_to(CAP).as_posix()

    import pyarrow.dataset, pyarrow.parquet
    import pandas as pd

    summary = c.read(
        c.ROOT
        / "Thesis_Log/astra_result/evidence/core_completion_summary_20260920_v1.json"
    )
    primary = c.read(c.CORE / "primary_results_v1/complete.json")
    copy(c.CORE / "primary_results_v1/complete.json")
    copy(
        c.ROOT
        / "Thesis_Log/astra_result/evidence/core_completion_summary_20260920_v1.json"
    )
    gold_path = c.ROOT / "data/processed/en_dev_eval_main.parquet"
    copy(gold_path)
    rows = pd.read_parquet(gold_path).to_dict("records")
    gold = [
        {"id": str(r["id"]), "labels": ["hate" if bool(r["gate"]) else "not_hate"]}
        for r in rows
    ]
    write(CAP / "gold/english_hate.json", gold)
    arms = {}
    expected = {}
    for arm in ("rag_small", "rag_full", "encoder_kb", "encoder_full"):
        files = []
        for rep in range(3):
            path = (
                c.CORE
                / f"primary_results_v1/assembled/en_dev_eval_main/{arm}_rep{rep}.jsonl"
            )
            files.append(copy(path, primary["assembled"][c.rel(path)]["sha256"]))
        arms[arm] = files
        expected[arm] = summary["endpoints"]["en_dev_eval_main"]["conditions"][arm][
            "values"
        ]
    cases = [
        {
            "id": "SQ1_SQ2_English_hate_core",
            "dimension": "hate",
            "n": 299,
            "gold": "gold/english_hate.json",
            "labels": ["not_hate", "hate"],
            "arms": arms,
            "expected": expected,
        }
    ]
    science = c.read(FB / "scientific_plan_v1.json")
    result = c.read(FB / "results_v1/transfer_b04.json")
    baseline = c.read(FB / "baseline_review_v1/complete.json")
    updates = c.read(FB / "updates_review_v1/complete.json")
    for p in [
        FB / "scientific_plan_v1.json",
        FB / "results_v1/transfer_b04.json",
        FB / "baseline_review_v1/complete.json",
        FB / "updates_review_v1/complete.json",
        FB / "feasibility_v1/roster.jsonl",
    ]:
        copy(p)
    byid = {r["id"]: r for r in c.rows(FB / "feasibility_v1/roster.jsonl")}
    ids = science["rosters"]["transfer"]
    c.require(
        set(ids) == set(result["common_valid_ids"]) and len(ids) == 350,
        "Final SQ3 primary mask changed",
    )
    gold = [{"id": rid, "labels": byid[rid]["gold"]["hate_type"]} for rid in ids]
    c.require(
        all((byid[rid]["gold"]["hate"] is True for rid in ids)),
        "SQ3 transfer not all gold hateful",
    )
    write(CAP / "gold/sq3_transfer.json", gold)
    arms = {}
    expected = {}
    for arm in ("correct", "error"):
        files = []
        scores = []
        for seed in (202609281, 202609282, 202609283):
            condition = f"{arm}_s{seed}_b04"
            for rep in range(3):
                p = FB / f"updates_review_v1/assembled/{condition}_rep{rep}.jsonl"
                files.append(copy(p, updates["outputs"][c.rel(p)]))
                scores.append(
                    result["conditions"][condition]["common_valid_runs"][rep][
                        "macro_f1"
                    ]
                )
        arms[arm] = files
        expected[arm] = scores
    files = []
    for rep in range(3):
        p = FB / f"baseline_review_v1/assembled/reference_rep{rep}.jsonl"
        files.append(copy(p, baseline["outputs"][c.rel(p)]))
    arms["reference"] = files
    expected["reference"] = [
        r["macro_f1"] for r in result["conditions"]["reference"]["common_valid_runs"]
    ]
    cases.append(
        {
            "id": "SQ3_final40_transfer",
            "dimension": "hate_type",
            "n": 350,
            "gold": "gold/sq3_transfer.json",
            "labels": science["scoring"]["labels"],
            "arms": arms,
            "expected": expected,
        }
    )
    write(CAP / "cases.json", cases)
    write(CAP / "source_origins.json", origins)
    shutil.copyfile(c.ROOT / "scripts/replay_results.py", CAP / "replay.py")
    (CAP / "README.md").write_text(
        "# Private selected-results replay capsule\n\nRun `python -I -B replay.py` from this folder. Only the Python standard library is needed.\n\nThis capsule replays 33 original per-run point scores from saved predictions: SQ1 full/small KB English hate detection, SQ2 full/compact encoder English hate detection, and SQ3 final-budget correction/control/reference transfer. It includes original selected prediction files, frozen gold exports, original source tables/rosters and source hashes.\n\nThis is a bounded reproduction package, not the complete project backup. It does not rerun models, regenerate confidence intervals, reparse raw API replies, archive all checkpoints or include the currently running balanced study. Do not publish it as an unrestricted dataset: original texts/model explanations and their existing access conditions remain.\n",
        encoding="utf8",
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "selected-score-replay-private",
        "files": {
            p.relative_to(CAP).as_posix(): {
                "sha256": c.sha(p),
                "bytes": p.stat().st_size,
            }
            for p in sorted(CAP.rglob("*"))
            if p.is_file()
        },
    }
    write(CAP / "manifest.json", manifest)
    archive = BASE / "astra_selected_replay_20261008_v2.zip"
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as z:
        for p in sorted(CAP.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(CAP).as_posix())
    restored = BASE / "restored_check_v2"
    restored.mkdir()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            c.require(
                (restored / name).resolve().is_relative_to(restored.resolve()),
                "Unsafe archive path",
            )
        c.require(z.testzip() is None, "Archive CRC failure")
        z.extractall(restored)
    replay_output = BASE / "restore_verification_v2.json"
    invocation = [
        sys.executable,
        "-I",
        "-B",
        str(restored / "replay.py"),
        "--output",
        str(replay_output),
    ]
    response = subprocess.run(
        invocation, cwd=restored, capture_output=True, text=True, encoding="utf8"
    )
    c.require(response.returncode == 0, response.stderr[-2000:])
    result = c.read(replay_output)
    c.require(result["individual_scores"] == 33, "Wrong replay scope")
    target = restored / "gold/english_hate.json"
    original = target.read_bytes()
    target.write_bytes(original + b" ")
    failed = subprocess.run(
        [sys.executable, "-I", "-B", str(restored / "replay.py")],
        cwd=restored,
        capture_output=True,
        text=True,
    )
    c.require(
        failed.returncode != 0 and "Capsule file changed" in failed.stderr,
        "Tamper was not detected",
    )
    target.write_bytes(original)
    for rel, rec in manifest["files"].items():
        c.require(c.sha(restored / rel) == rec["sha256"], "Restored file differs")
    for name, rec in origins.items():
        c.require(
            c.sha(c.ROOT / name) == rec["sha256"], "Source changed during packaging"
        )
    receipt = {
        "created_utc": c.utc(),
        "status": "local_selected_restore_and_replay_pass",
        "files": len(manifest["files"]),
        "source_files": len(origins),
        "individual_scores_reproduced": 33,
        "tamper_rejection_pass": True,
        "archive": c.rel(archive),
        "archive_sha256": c.sha(archive),
        "archive_bytes": archive.stat().st_size,
        "manifest_sha256": c.sha(CAP / "manifest.json"),
        "restore_receipt_sha256": c.sha(replay_output),
        "separate_working_directory": True,
        "isolated_python_flag": "-I",
        "third_party_replay_dependencies": [],
        "external_backup_done": False,
        "full_project_backup_done": False,
        "running_study_included": False,
        "outputs": {
            c.rel(p): c.sha(p) for p in (archive, CAP / "manifest.json", replay_output)
        },
        "scripts": {
            c.rel(p): c.sha(p)
            for p in (Path(__file__), c.ROOT / "scripts/replay_results.py")
        },
    }
    save_new(
        c.ROOT
        / "Thesis_Log/astra_result/evidence/astra_repro_package_20261008_v1.json",
        receipt,
    )
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ["outputs", "scripts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    with c.offline():
        main()
