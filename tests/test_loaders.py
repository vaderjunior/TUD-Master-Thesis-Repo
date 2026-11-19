"""Synthetic source files exercise mappings and preparation without corpus downloads."""
import hashlib
import json
import sqlite3

import pandas as pd
import pytest

from src.hsrag.data.botox import load as load_botox
from src.hsrag.data.detox import P_COLS, load_detox
from src.hsrag.data.gahd import load_gahd
from src.hsrag.data.hatexplain import load_hatexplain
from src.hsrag.data.implicit_hate import load_implicit_hate
from src.hsrag.data.mhs import TARGET_MAP, load_mhs
from scripts.make_splits import split_one_language


def test_hatexplain_votes_targets_and_unknown_annotations(tmp_path):
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("source_label,unified_label\nWomen,gender\nMuslims,religion\n")
    data = {"one": {"post_tokens": ["a", "text"], "rationales": [],
                    "annotators": [{"label": "hatespeech", "target": ["Women"]},
                                   {"label": "hatespeech", "target": ["Muslims"]},
                                   {"label": "normal", "target": []}]},
            "two": {"post_tokens": ["rude"], "rationales": [],
                    "annotators": [{"label": "offensive", "target": ["Women"]}] * 3},
            "tie": {"post_tokens": ["tie"], "rationales": [],
                    "annotators": [{"label": label, "target": []} for label in ("hatespeech", "normal", "offensive")]}}
    source, splits = tmp_path / "dataset.json", tmp_path / "splits.json"
    source.write_text(json.dumps(data)); splits.write_text('{"train":["one","two","tie"]}')
    rows, stats = load_hatexplain(source, splits, mapping)
    assert len(rows) == 2 and stats["dropped_tie"] == 1
    assert rows[0].gate and rows[0].target_groups == ["gender", "religion"]
    assert not rows[1].gate and rows[1].target_groups is None
    assert all(r.hate_types is None and r.legal is None for r in rows)


def test_implicit_types_and_unmatched_stage_two_are_preserved(tmp_path):
    a, b = tmp_path / "stage1.tsv", tmp_path / "stage2.tsv"
    pd.DataFrame({"post": ["direct", "coded", "benign"],
                  "class": ["explicit_hate", "implicit_hate", "not_hate"]}).to_csv(a, sep="\t", index=False)
    pd.DataFrame({"post": ["coded", "unmatched"], "implicit_class": ["white_grievance", "other"],
                  "extra_implicit_class": ["inferiority", ""]}).to_csv(b, sep="\t", index=False)
    rows, stats = load_implicit_hate(a, b)
    assert [r.hate_types for r in rows] == [["explicit"], ["grievance", "inferiority"], None, None]
    assert rows[-1].gate and stats["stg2_unmatched"] == 1


def test_local_mhs_aggregation_and_severity_boundaries(tmp_path):
    source = tmp_path / "mhs.csv"
    rows = []
    for comment, score in enumerate([0.5, 1.0, 1.63, 1.64]):
        for annotator in [0, 1]:
            row = {"comment_id": comment, "text": f"text {comment}", "annotator_id": annotator,
                   "hate_speech_score": score, **{k: False for k in TARGET_MAP}}
            row["target_gender"] = annotator == 1
            rows.append(row)
    pd.DataFrame(rows).to_csv(source, index=False)
    records, _ = load_mhs(source)
    assert [r.severity for r in records] == [None, "low", "medium", "high"]
    assert records[0].target_groups is None and records[1].target_groups == ["gender"]
    assert all(r.raw["n_annotators"] == 2 and r.hate_types is None for r in records)


def test_detox_is_read_only_and_missing_database_is_not_created(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError):
        load_detox(missing)
    assert not missing.exists()
    source, mapping = tmp_path / "detox.sqlite3", tmp_path / "mapping.csv"
    mapping.write_text("source_column,unified_label\ndiscrim_gender,gender\n")
    rows = [{"c_id": i, "c_text": f"text {i}", "hate_speech": score, "discrim_gender": 0.6,
             "criminal_rel": 0.0, "threat": 0.0, "extrem": 0.0, "toxi": 1.0, "nb_annotators": 3,
             **{k: 0.0 for k in P_COLS}} for i, score in enumerate([0.5, 0.6, None])]
    with sqlite3.connect(source) as con:
        pd.DataFrame(rows).to_sql("Goldstandard", con, index=False)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    records, stats = load_detox(source, mapping)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert [r.gate for r in records] == [False, True]
    assert records[1].target_groups == ["gender"] and records[1].legal is None
    assert stats["dropped_nan_score"] == 1


def test_botox_legal_is_independent_and_heldout_texts_are_filtered(tmp_path):
    source = tmp_path / "botox.csv"
    pd.DataFrame({"index": [1, 2, 3], "text": ["legal", "benign", " Excluded TEXT "],
                  "class_1": [0, 0, 1], "class_2": [0, 0, 0], "class_3": [1, 0, 0]}).to_csv(source, sep=";", encoding="latin-1", index=False)
    processed = tmp_path / "processed"; processed.mkdir()
    pd.DataFrame({"text": ["excluded text"]}).to_parquet(processed / "de_dev.parquet")
    records = load_botox(source, processed_dir=processed)
    assert [r.legal for r in records] == [["insult_defamation"], []]
    assert all(r.gate is None and r.target_groups is None for r in records)


def test_general_split_is_deterministic_and_disjoint():
    frame = pd.DataFrame({"id": list(range(100)), "gate": [True] * 50 + [False] * 50})
    first, second = split_one_language(frame), split_one_language(frame)
    assert [len(first[k]) for k in ("train", "dev", "test")] == [69, 16, 15]
    assert all(first[k]["id"].tolist() == second[k]["id"].tolist() for k in first)
    sets = [set(first[k]["id"]) for k in first]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_prepare_command_uses_local_sources_and_preserves_outputs(tmp_path, run_cli):
    raw = tmp_path / "raw/gahd"; raw.mkdir(parents=True)
    pd.DataFrame({"gahd_id": list(range(100)), "text": [f"comment {i}" for i in range(100)],
                  "label": [1] * 50 + [0] * 50, "round": 1, "split": "train",
                  "contrastive_gahd_id": None}).to_csv(raw / "gahd.csv", index=False)
    output = tmp_path / "processed"
    args = ("prepare", "splits", "--sources", "gahd", "--raw-dir", str(raw.parent), "--output-dir", str(output))
    result = run_cli(tmp_path, *args)
    assert result.returncode == 0, result.stderr
    assert [len(pd.read_parquet(output / f"de_{s}.parquet")) for s in ("train", "dev", "test")] == [69, 16, 15]
    before = (output / "de_train.parquet").read_bytes()
    repeat = run_cli(tmp_path, *args)
    assert repeat.returncode == 2 and (output / "de_train.parquet").read_bytes() == before
    assert json.loads((output / "split_summary.json").read_text())["seed"] == 42


def test_guidance_only_preparation_has_no_dataset_examples(tmp_path, run_cli):
    output = tmp_path / "knowledge.jsonl"
    result = run_cli(tmp_path, "prepare", "knowledge", "--guidance-only", "--output", str(output))
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert sum(r["kind"] == "definition" for r in records) == 21
    assert sum(r["kind"] == "guideline" for r in records) == 19
    assert not any(r["kind"] == "example" for r in records)


def test_legal_preparation_keeps_its_original_stratum_rounding(tmp_path, run_cli):
    raw = tmp_path / "raw/botox"; raw.mkdir(parents=True)
    pd.DataFrame({"index": list(range(100)), "text": [f"legal comment {i}" for i in range(100)],
                  "class_1": [1] * 50 + [0] * 50, "class_2": 0, "class_3": 0}).to_csv(raw / "botox.csv", sep=";", index=False)
    output = tmp_path / "prepared"
    result = run_cli(tmp_path, "prepare", "legal", "--raw-dir", str(raw.parent),
                     "--output-dir", str(output), "--allow-missing-heldout")
    assert result.returncode == 0, result.stderr
    frames = [pd.read_parquet(output / f"de_legal_{s}.parquet") for s in ("train", "dev", "test")]
    assert [len(f) for f in frames] == [70, 14, 16]
    assert pd.concat(frames)["gate"].isna().all()


def test_scoring_ignores_manifest_header_and_uses_selected_gold(tmp_path, run_cli):
    from src.hsrag.classify import Arms, classify
    from src.hsrag.client import MockClient

    gold_dir = tmp_path / "gold"; gold_dir.mkdir()
    pd.DataFrame({"id": [str(i) for i in range(30)], "text": "sample", "lang": "en",
                  "source": "fixture", "gate": [True] * 15 + [False] * 15,
                  "target_groups": None, "hate_types": None, "severity": None, "legal": None,
                  "raw": "{}"}).to_parquet(gold_dir / "fixture.parquet", index=False)
    predictions, output = tmp_path / "predictions.jsonl", tmp_path / "scores.json"
    values = [classify(str(i), "sample", "en", "zero_shot", MockClient(), Arms()).to_json() for i in range(30)]
    predictions.write_text('{"_manifest":{"name":"fixture"}}\n' + "\n".join(values))
    result = run_cli(tmp_path, "score", "--file", str(predictions), "--subset", "fixture",
                     "--processed-dir", str(gold_dir), "--gate-mapping", "strict", "--output", str(output))
    assert result.returncode == 0, result.stderr
    gate = json.loads(output.read_text())["zero_shot"]["gate"]["strict"]
    assert gate["n_scored"] == 30 and gate["macro_f1"] == pytest.approx(1 / 3)
