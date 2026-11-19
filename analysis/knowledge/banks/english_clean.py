"""Banks english clean."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata
from analysis.knowledge.banks import build as u

ROOT = u.ROOT
BASE = ROOT / "experiments/astra/english_kb_cleanup_20260914_v1"
AUDIT = ROOT / "experiments/astra/english_kb_audit_20260914_v1"
RESULT = ROOT / "Thesis_Log/astra_result/evidence/english_kb_results_20260914_v1.json"
REVIEW = ROOT / "Thesis_Log/astra_result/evidence/english_kb_review_supplement_20260914_v1.json"


def put(name, value):
    path = BASE / name
    assert path.resolve().is_relative_to(BASE.resolve())
    data = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, "Preserve different existing artifact"
    else:
        with path.open("xb") as f:
            f.write(data)


def body_norm(text):
    return u.norm(re.sub("(?<![\\w@])@[A-Za-z0-9_]+", "@user", text))


def wrapper_key(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub("https?://\\S+|www\\.\\S+", " URLTOKEN ", text)
    text = re.sub("(?<![\\w@])@[A-Za-z0-9_]+", " MENTIONTOKEN ", text)
    text = re.sub("<user>", " MENTIONTOKEN ", text)
    text = re.sub("(?<!\\w)#(?=\\w)", "", text)
    text = re.sub("(?<!\\w)url(?!\\w)", " URLTOKEN ", text)
    text = "".join((" " if unicodedata.category(c).startswith("P") else c for c in text))
    words = text.split()
    while words and words[0] == "rt":
        words.pop(0)
    suffixes = [
        ["read", "more"],
        ["full", "story"],
        ["watch", "video"],
        ["more"],
        ["via"],
        ["URLTOKEN"],
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if len(words) >= len(suffix) and words[-len(suffix) :] == suffix:
                del words[-len(suffix) :]
                changed = True
                break
    return " ".join(words) if len(words) >= 3 else None


def survey():
    assert not (BASE / "survey.json").exists(), "Survey already frozen"
    result = u.read_json(RESULT)
    review = u.read_json(REVIEW)
    for name, h in {**result["input_hashes"], **review["input_hashes"]}.items():
        assert u.digest(ROOT / name) == h, name
    train = u.read_lines(AUDIT / "train_texts.jsonl")
    held = u.read_lines(AUDIT / "heldout_texts.jsonl")
    excluded = set(u.read_json(AUDIT / "report.json")["proposed_excluded_ids"])
    oldsmall = set(u.read_json(AUDIT / "preparation.json")["previous_small_ids"])
    nearest = {x["train_id"]: x for x in u.read_lines(AUDIT / "nearest_train.jsonl")}
    heldkeys = defaultdict(list)
    for i, h in enumerate(held):
        key = wrapper_key(h["text"])
        if key:
            heldkeys[key].append(i)
    candidates = []
    for t in train:
        key = wrapper_key(t["text"])
        if t["id"] not in excluded and key in heldkeys:
            candidates.append(
                {
                    "train_id": t["id"],
                    "train_text": t["text"],
                    "key": key,
                    "max_cosine": nearest[t["id"]]["max_cosine"],
                    "in_previous_small": t["id"] in oldsmall,
                    "matches": [{"heldout_text_index": i, **held[i]} for i in heldkeys[key]],
                }
            )
    manual = {x["train_id"] for x in review["manual_review"]}
    paths = [
        Path(__file__),
        RESULT,
        REVIEW,
        AUDIT / "report.json",
        AUDIT / "complete.json",
        AUDIT / "plan.json",
        AUDIT / "preparation.json",
        AUDIT / "train_texts.jsonl",
        AUDIT / "heldout_texts.jsonl",
        AUDIT / "nearest_train.jsonl",
        ROOT / "analysis/knowledge/banks/build.py",
    ]
    hashes = {u.rel(p): u.digest(p) for p in paths}
    put(
        "survey.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "wrapper_rule": wrapper_key.__doc__,
            "wrapper_candidates": candidates,
            "counts": {
                "retained_wrapper_matches": len(candidates),
                "beyond_manual_band": sum((x["train_id"] not in manual for x in candidates)),
                "in_previous_small": sum((x["in_previous_small"] for x in candidates)),
            },
            "input_hashes": hashes,
            "status": "survey_for_policy_review_not_applied",
        },
    )
    print(
        json.dumps(
            {
                "path": u.rel(BASE / "survey.json"),
                "counts": u.read_json(BASE / "survey.json")["counts"],
            },
            indent=2,
        )
    )


def freeze():
    raise RuntimeError(
        "Survey must be reviewed and an explicit policy added before freeze is implemented."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("survey", "freeze"))
    args = parser.parse_args()
    {"survey": survey, "freeze": freeze}[args.mode]()
