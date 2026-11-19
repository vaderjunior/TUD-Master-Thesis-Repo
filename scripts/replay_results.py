import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes capsule")
    return path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def f1(gold, pred, labels):

    paired = [(y, p) for y, p in zip(gold, pred) if p is not None]
    if not paired:
        return None
    values = []
    for label in labels:
        tp = sum(label in y and label in p for y, p in paired)
        fp = sum(label not in y and label in p for y, p in paired)
        fn = sum(label in y and label not in p for y, p in paired)
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return statistics.mean(values)


def failure_bounds(gold, pred, labels):
    wrong = [p if p is not None else set(labels) - y for y, p in zip(gold, pred)]
    right = [p if p is not None else y for y, p in zip(gold, pred)]
    return [f1(gold, wrong, labels), f1(gold, right, labels)]


def evaluate(root):
    manifest = read(root / "manifest.json")
    for relative, record in manifest["files"].items():
        path = checked(root, relative)
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Capsule file changed: " + relative)
    cases = read(root / "cases.json")
    results = []
    count = 0
    for case in cases:
        gold = read(checked(root, case["gold"]))
        if len(gold) != case["n"] or len({r["id"] for r in gold}) != len(gold):
            raise ValueError("Wrong or duplicated evaluation roster")
        groups = {}
        for arm, files in case["arms"].items():
            values = []
            coverage = []
            for filename in files:
                source = checked(root, filename)
                rows = [
                    json.loads(s)
                    for s in source.read_text(encoding="utf-8").splitlines()
                    if s.strip()
                ]
                mapping = {str(r.get("id", r.get("item_id"))): r for r in rows}
                if len(mapping) != len(rows):
                    raise ValueError("Duplicated prediction identity")
                actual = []
                for row in gold:
                    item = mapping[row["id"]]
                    p = item["result"]
                    if (p is None) != (item.get("n_valid", 1) == 0):
                        raise ValueError("Contradictory saved validity")
                    if p is None:
                        actual.append(None)
                        continue
                    if case["dimension"] == "hate":
                        positive = ["hate" if p["hate"] else "not_hate"]
                    else:
                        positive = p[case["dimension"]] if p["hate"] else []
                    actual.append(set(positive))
                values.append(
                    f1([set(r["labels"]) for r in gold], actual, case["labels"])
                )
                count += 1
                missing = [r["id"] for r, p in zip(gold, actual) if p is None]
                coverage.append(
                    {
                        "n_total": len(gold),
                        "n_valid": len(gold) - len(missing),
                        "invalid_ids": missing,
                        "failure_bounds": failure_bounds(
                            [set(r["labels"]) for r in gold], actual, case["labels"]
                        ),
                    }
                )
            expected = case["expected"][arm]
            if len(values) != len(expected) or any(
                abs(x - y) > 1e-12 for x, y in zip(values, expected)
            ):
                raise ValueError("Recomputed score differs: " + case["id"] + "/" + arm)
            groups[arm] = {
                "values": values,
                "mean": statistics.mean(values),
                "coverage_by_run": coverage,
            }
        results.append({"case": case["id"], "n": case["n"], "scores": groups})
    return {
        "status": "selected_scores_reproduced",
        "individual_scores": count,
        "cases": results,
        "scope": "Point scores and recorded per-run means from saved predictions; not fresh inference, raw-response reparsing or interval regeneration.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.root.resolve())
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
