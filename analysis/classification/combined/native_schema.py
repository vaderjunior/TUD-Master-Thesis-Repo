from analysis.classification.combined import prepare as a


def native_row(row, bank, head):
    ident = str(row["id"])
    canonical = ident if ident in bank else "ex-" + ident
    a.require(canonical in bank, "Native source ID cannot be resolved to its KB record")
    source = bank[canonical]
    a.require(
        row["text"] == source["text"], "Canonical native source has different text"
    )
    out = row | {"id": canonical}
    if head != "hate":
        a.require(
            source["meta"]["gate"] is True, "Fine training source is not gold hateful"
        )
        a.require(
            "gate" not in row or row["gate"] is True,
            "Conflicting native gate annotation",
        )
        out["gate"] = source["meta"]["gate"]
    return out
