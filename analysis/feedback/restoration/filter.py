from functools import lru_cache
import json
from types import FunctionType
from analysis.feedback.restoration import study as s


@lru_cache(maxsize=1)
def originals():
    return s.c.lines(s.CORE / "kb/full/records.jsonl")


def eligibility_filter(label, records):
    codes = sorted(
        {
            s.flat_metadata(r)["hate_types"]
            for r in records
            if r["kind"] == "example" and label in (r["meta"].get("hate_types") or [])
        }
    )
    s.c.require(0 < len(codes) <= 128, "Unexpected withdrawn annotation combinations")
    return {
        "$or": [
            {"kind": "guideline"},
            {
                "$and": [
                    {"kind": "definition"},
                    {
                        "$or": [
                            {"dimension": {"$ne": "hate_type"}},
                            {"label": {"$ne": label}},
                        ]
                    },
                ]
            },
            {"$and": [{"kind": "example"}, {"hate_types": {"$nin": codes}}]},
        ]
    }


class FilteredCollection:
    def __init__(self, collection, records, filtered):
        self.collection = collection
        self.metadata = collection.metadata
        self.allowed = {r["id"] for r in records}
        self.filtered = filtered
        self.where = None
        if filtered:
            full = originals()
            removed = {r["id"] for r in full} - self.allowed
            matches = []
            for label in s.LABELS:
                manifest = s.c.read(s.PACKET / f"taxonomy_manifests/{label}.json")
                if removed == set(
                    manifest["removed_example_ids"] + manifest["removed_definition_ids"]
                ):
                    matches.append(label)
            s.c.require(
                len(matches) == 1,
                "Filter must match exactly one frozen withdrawal manifest",
            )
            self.label = matches[0]
            self.where = eligibility_filter(self.label, full)
            actual = set(collection.get(where=self.where, include=[])["ids"])
            s.c.require(
                actual == self.allowed,
                "Compact filter does not select exactly the frozen retained records",
            )

    def query(self, **kwargs):
        s.c.require("ids" not in kwargs, "Caller supplied a competing ID filter")
        if self.filtered:
            s.c.require("where" in kwargs, "Original kind/language filter required")
            kwargs["where"] = {"$and": [kwargs["where"], self.where]}
        result = self.collection.query(**kwargs)
        s.c.require(
            all(set(group) <= self.allowed for group in result["ids"]),
            "Withdrawn or unknown ANN candidate",
        )
        return result


def make_retriever(*args, **kwargs):
    factory = FunctionType(
        s.make_retriever.__code__,
        {**s.make_retriever.__globals__, "FilteredCollection": FilteredCollection},
        argdefs=s.make_retriever.__defaults__,
    )
    return factory(*args, **kwargs)


def synthetic_checks():

    def match(value, where):
        if "$and" in where:
            return all(match(value, w) for w in where["$and"])
        if "$or" in where:
            return any(match(value, w) for w in where["$or"])
        assert len(where) == 1
        key, wanted = next(iter(where.items()))
        got = value.get(key)
        if isinstance(wanted, dict):
            op, operand = next(iter(wanted.items()))
            if op == "$ne":
                return got != operand
            if op == "$nin":
                return got not in operand
            raise AssertionError(op)
        return got == wanted

    cases = []
    annotations = [
        None,
        [],
        ["irony"],
        ["grievance"],
        ["irony", "grievance"],
        ["grievance", "irony"],
        ["explicit"],
    ]
    for i, types in enumerate(annotations):
        cases.append(
            {
                "id": str(i),
                "kind": "example",
                "lang": "en",
                "source": "fixture",
                "dimension": None,
                "label": None,
                "meta": {"hate_types": types, "gate": True},
            }
        )
    for dim, label in [
        ("hate_type", "irony"),
        ("hate_type", "grievance"),
        ("legal", "irony"),
    ]:
        cases.append(
            {
                "id": dim + label,
                "kind": "definition",
                "lang": "en",
                "source": "fixture",
                "dimension": dim,
                "label": label,
                "meta": {},
            }
        )
    cases.append(
        {
            "id": "guide",
            "kind": "guideline",
            "lang": "en",
            "source": "fixture",
            "dimension": None,
            "label": None,
            "meta": {},
        }
    )
    where = eligibility_filter("irony", cases)
    actual = {r["id"] for r in cases if match(s.flat_metadata(r), where)}
    expected = {
        r["id"]
        for r in cases
        if not (
            (r["kind"] == "example" and "irony" in (r["meta"].get("hate_types") or []))
            or (
                r["kind"] == "definition"
                and r["dimension"] == "hate_type"
                and r["label"] == "irony"
            )
        )
    }
    assert actual == expected
    assert {"0", "1", "3", "6", "legalirony", "guide", "hate_typegrievance"} == actual
    return {
        "synthetic_records": len(cases),
        "retained": len(actual),
        "multi_label_orders_rejected": True,
        "missing_annotation_and_empty_retained": True,
        "same_name_other_dimension_retained": True,
    }
