import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

RECORDS = Path("kb/records.jsonl")


VALID_KINDS = {"definition", "guideline", "example", "feedback"}
REQUIRED = {"id", "kind", "dimension", "label", "lang", "text", "source", "meta"}
VALID_LANGS = {"en", "de"}


def validate_line(obj: dict, n: int) -> list[str]:
    errors = []

    missing = REQUIRED - obj.keys()
    if missing:
        errors.append(f"line {n}: missing fields {missing}")
        return errors

    if obj["kind"] not in VALID_KINDS:
        errors.append(f"line {n} ({obj['id']}): bad kind '{obj['kind']}'")

    if obj["lang"] not in VALID_LANGS:
        errors.append(f"line {n} ({obj['id']}): bad lang '{obj['lang']}'")

    if not obj["text"] or not str(obj["text"]).strip():
        errors.append(f"line {n} ({obj['id']}): empty text")

    if not isinstance(obj["meta"], dict):
        errors.append(f"line {n} ({obj['id']}): meta is not a dict")

    if obj["kind"] == "definition":
        if not obj["dimension"] or not obj["label"]:

            if not (obj["dimension"] == "hate" and obj["label"] is None):
                errors.append(
                    f"line {n} ({obj['id']}): definition needs dimension+label"
                )

    if obj["kind"] in ("example", "feedback"):
        m = obj["meta"]
        if "gate" not in m:
            errors.append(f"line {n} ({obj['id']}): {obj['kind']} meta missing 'gate'")
        if "illustrative_only" not in m:
            errors.append(
                f"line {n} ({obj['id']}): {obj['kind']} meta missing "
                f"'illustrative_only'"
            )

    if obj["kind"] == "feedback":
        for key in ("round", "origin_item_id", "criterion", "feedback_arm"):
            if key not in obj["meta"]:
                errors.append(f"line {n} ({obj['id']}): feedback meta missing '{key}'")

    return errors


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RECORDS))
    records_path = Path(ap.parse_args().file)

    if not records_path.exists():
        print(f"{records_path} does not exist yet (build it in 3.2)")
        sys.exit(0)

    all_errors = []
    ids = set()
    by_text = defaultdict(list)

    counts = {k: 0 for k in sorted(VALID_KINDS)}
    n_parsed = 0

    with open(records_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"line {n}: bad JSON ({e})")
                continue

            all_errors.extend(validate_line(obj, n))

            if obj.get("id") in ids:
                all_errors.append(f"line {n}: duplicate id '{obj['id']}'")
            ids.add(obj.get("id"))

            if obj.get("text"):
                key = " ".join(str(obj["text"]).lower().split())
                by_text[key].append(obj.get("id"))

            n_parsed += 1
            if obj.get("kind") in counts:
                counts[obj["kind"]] += 1

    dupes = {t: rec_ids for t, rec_ids in by_text.items() if len(rec_ids) > 1}
    for rec_ids in list(dupes.values())[:10]:
        all_errors.append(f"duplicate text shared by ids {rec_ids}")

    print(f"Records: {n_parsed}")
    for kind, n in counts.items():
        print(f"  {kind:12} {n}")
    print(f"  distinct texts {len(by_text)}")

    if sum(counts.values()) != n_parsed:
        all_errors.append(
            f"per-kind counts sum to {sum(counts.values())} but {n_parsed} "
            f"records were parsed - an unknown kind is present"
        )

    if all_errors:
        print(f"\n{len(all_errors)} ERRORS:")
        for e in all_errors[:30]:
            print(f"  {e}")
        sys.exit(1)

    print("\nSchema valid.")


if __name__ == "__main__":
    main()
