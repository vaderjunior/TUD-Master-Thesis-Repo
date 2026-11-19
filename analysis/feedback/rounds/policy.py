from collections import Counter
from analysis.common import runtime as c

POLICIES = ("adaptive", "static", "random")


def order(seed, round_no, policy, gate, source_id):
    return c.text_sha(f"A17|{seed}|{round_no}|{policy}|{int(gate)}|{source_id}")


def partition(rows, seed, rounds=4):
    c.require(len({r["id"] for r in rows}) == len(rows), "Source identities repeat")
    blocks = [[] for _ in range(rounds)]
    for gate in (False, True):
        part = sorted(
            (r for r in rows if r["gold"]["hate"] is gate),
            key=lambda r: order(seed, 0, "source", gate, r["id"]),
        )
        c.require(
            part and len(part) % rounds == 0, "Source classes cannot be split evenly"
        )
        width = len(part) // rounds
        for i in range(rounds):
            blocks[i].extend((r["id"] for r in part[i * width : (i + 1) * width]))
    return [sorted(v) for v in blocks]


def gate(votes):
    c.require(
        len(votes) == 3 and all((v is None or type(v) is bool for v in votes)),
        "Need three binary/invalid repeats",
    )
    return None if any((v is None for v in votes)) else sum(votes) >= 2


def select(rows, initial, current, previous, seed, round_no, cap=5):

    by = {r["id"]: r for r in rows}
    c.require(len(by) == len(rows), "Repeated candidate ID")
    c.require(
        set(initial) == set(current) == set(by), "Screened candidate identities differ"
    )
    c.require(set(previous) == set(POLICIES), "Missing cumulative policy")
    c.require(
        all(
            (not set(v) & set(by) and len(v) == len(set(v)) for v in previous.values())
        ),
        "Candidate already in earlier pool",
    )
    old = {i: gate(initial[i]) for i in by}
    new = {i: gate(current[i]) for i in by}
    valid = sorted((i for i in by if old[i] is not None and new[i] is not None))
    sets = {}
    for pol in POLICIES:
        for g in (False, True):
            eligible = [
                i
                for i in valid
                if by[i]["gold"]["hate"] is g
                and (pol == "random" or (new if pol == "adaptive" else old)[i] is not g)
            ]
            rank_policy = "mistake" if pol in ("adaptive", "static") else pol
            sets[pol, g] = sorted(
                eligible, key=lambda i: order(seed, round_no, rank_policy, g, i)
            )
    k = min(
        [cap]
        + [len(sets[pol, g]) for pol in ("adaptive", "static") for g in (False, True)]
    )
    additions = {
        pol: sorted(sets[pol, False][:k] + sets[pol, True][:k]) for pol in POLICIES
    }
    cumulative = {pol: sorted(previous[pol] + additions[pol]) for pol in POLICIES}
    c.require(
        len({len(v) for v in cumulative.values()}) == 1
        and all((len(v) <= 40 for v in cumulative.values())),
        "Unequal/excess cumulative budgets",
    )
    return {
        "seed": seed,
        "round": round_no,
        "screened_ids": sorted(by),
        "eligible_ids": valid,
        "invalid_ids": sorted(set(by) - set(valid)),
        "per_class_budget": k,
        "eligible_counts": {
            pol: {str(g): len(sets[pol, g]) for g in (False, True)} for pol in POLICIES
        },
        "additions": additions,
        "cumulative": cumulative,
        "shortfall": k < cap,
        "annotation_opportunities": len(rows),
    }


def independent_select(rows, initial, current, previous, seed, round_no, cap=5):

    groups = {p: {0: [], 1: []} for p in POLICIES}
    missing = []
    for r in rows:
        i = r["id"]
        a = initial[i]
        b = current[i]
        if None in a or None in b:
            missing.append(i)
            continue
        g = int(r["gold"]["hate"])
        ov = sum((x == True for x in a)) > 1
        nv = sum((x == True for x in b)) > 1
        groups["random"][g].append(i)
        if ov != bool(g):
            groups["static"][g].append(i)
        if nv != bool(g):
            groups["adaptive"][g].append(i)
    k = min(cap, *(len(groups[p][g]) for p in ("adaptive", "static") for g in (0, 1)))
    additions = {}
    for p in POLICIES:
        chosen = []
        for g in (0, 1):
            rank_policy = "mistake" if p != "random" else p
            ranked = sorted(
                groups[p][g],
                key=lambda i: c.text_sha(
                    "A17|%s|%s|%s|%s|%s" % (seed, round_no, rank_policy, g, i)
                ),
            )
            chosen.extend(ranked[:k])
        additions[p] = sorted(chosen)
    return (
        k,
        additions,
        {p: sorted(previous[p] + additions[p]) for p in POLICIES},
        sorted(missing),
    )


def verified_selection(*args, **kwargs):
    a = select(*args, **kwargs)
    k, added, pools, missing = independent_select(*args, **kwargs)
    c.require(
        (a["per_class_budget"], a["additions"], a["cumulative"], a["invalid_ids"])
        == (k, added, pools, missing),
        "Independent selection differs",
    )
    return a
