import argparse
import hashlib
import json
from pathlib import Path

import yaml

CONFIG = Path("config")
KB = Path("kb")
TAXONOMY = CONFIG / "taxonomy.yaml"
BACKUP = CONFIG / "taxonomy_backup.yaml"
RECORDS = KB / "records.jsonl"
DIM = "hate_type"


def strip_label(text: str, label: str) -> str:

    lines = text.splitlines(keepends=True)
    hits = [
        i
        for i, l in enumerate(lines)
        if l.strip().startswith(f"{label}:") and "definition" in l
    ]
    assert len(hits) == 1, (
        f"expected exactly one flow-style line for '{label}', found "
        f"{len(hits)}. If this label is written as a nested block rather than "
        f"a one-liner, edit it by hand and skip this script."
    )
    return "".join(l for i, l in enumerate(lines) if i != hits[0])


def build(label: str) -> None:
    original = TAXONOMY.read_text(encoding="utf-8")
    tax = yaml.safe_load(original)
    labels = list(tax["dimensions"][DIM]["labels"])
    assert label in labels, f"'{label}' is not a {DIM} label: {labels}"

    edited = strip_label(original, label)
    after = yaml.safe_load(edited)
    assert list(after["dimensions"][DIM]["labels"]) == [
        l for l in labels if l != label
    ], "label set is not as intended"

    for d in tax["dimensions"]:
        if d == DIM:
            continue
        assert (
            after["dimensions"][d] == tax["dimensions"][d]
        ), f"dimension '{d}' changed; only {DIM} may differ"

    out_tax = CONFIG / f"taxonomy_no_{label}.yaml"
    out_tax.write_text(edited, encoding="utf-8")

    records = [
        json.loads(l)
        for l in RECORDS.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    def_id = f"def-{DIM}-{label}"
    dropped_def = [r for r in records if r["id"] == def_id]
    dropped_ex = [
        r
        for r in records
        if r["kind"] == "example" and label in (r.get("meta", {}).get(f"{DIM}s") or [])
    ]
    keep = [r for r in records if r["id"] != def_id and r not in dropped_ex]

    out_rec = KB / f"records_no_{label}.jsonl"
    with open(out_rec, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    h = hashlib.sha256(edited.encode()).hexdigest()[:8]
    print(f"\n{'=' * 74}\nLABEL REMOVAL: {DIM}/{label}")
    print(f"\n  taxonomy  {len(labels)} -> {len(labels) - 1} {DIM} labels")
    print(f"            {out_tax}")
    print(f"            taxonomy_version when swapped in: taxonomy-{h}")
    print(f"\n  KB        {len(records)} -> {len(keep)} records")
    print(f"            definition dropped: {len(dropped_def)}")
    print(f"            examples dropped:   {len(dropped_ex)}")
    for r in dropped_ex[:5]:
        print(f"              {r['id']:34} {r['meta'].get(f'{DIM}s')}")
    if len(dropped_ex) > 5:
        print(f"              ... and {len(dropped_ex) - 5} more")
    print(f"            {out_rec}")
    print(
        f"\n  NOTE the rag arm therefore differs from the restored state in "
        f"THREE\n  ways: the label is absent from the label space, its "
        f"definition is gone,\n  and {len(dropped_ex)} examples are gone. "
        f"zero_shot differs in ONE way, since it\n  receives no examples "
        f"at all - that is the clean pathway test."
    )
    print(f"\n  next:")
    print(f"    python -m scripts.build_kb_alt {out_rec} " f"kb/chroma_no_{label}")
    print(f"    python -m scripts.make_label_removal --label {label} --check")
    print(f"{'=' * 74}\n")


def check(label: str) -> None:

    from src.hsrag.prompt import render_label_space, taxonomy_version

    current = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    labels = list(current["dimensions"][DIM]["labels"])
    swapped = label not in labels

    print(f"\n{'=' * 74}\nCHECK: is the removed taxonomy currently in place?")
    print(f"  config/taxonomy.yaml {DIM} labels: {labels}")
    print(f"  '{label}' present: {not swapped}")
    print(f"  taxonomy_version:   {taxonomy_version()}")
    print(f"  backup exists:      {BACKUP.exists()}  ({BACKUP})")

    if not swapped:
        print(f"\n  The ORIGINAL taxonomy is in place. To swap:")
        print(f"    Copy-Item {TAXONOMY} {BACKUP}")
        print(
            f"    Copy-Item {CONFIG / f'taxonomy_no_{label}.yaml'} "
            f"{TAXONOMY} -Force"
        )
        print(f"{'=' * 74}\n")
        return

    print(f"\n  the removed taxonomy IS in place. Rendered label space:\n")
    for line in render_label_space().splitlines():
        print(f"    {line}")

    from src.hsrag.schema import HateType

    vals = [e.value for e in HateType]
    print(f"\n  Pydantic HateType enum: {vals}")
    assert label not in vals, "the enum still carries the removed label"
    print(
        f"  -> a prediction of '{label}' now fails validation. Expect parse "
        f"failures\n     to rise; count them, they are part of the result."
    )

    print(f"\n  RESTORE when the runs are done:")
    print(f"    Copy-Item {BACKUP} {TAXONOMY} -Force")
    print(f"    python -m scripts.make_label_removal --label {label} --check")
    print(f"  and confirm taxonomy_version returns to its pre-swap value.")
    print(f"{'=' * 74}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    check(args.label) if args.check else build(args.label)


if __name__ == "__main__":
    main()
