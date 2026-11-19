"""Policy family sensitivity."""

from collections import defaultdict
import csv
import numpy as np
from analysis.knowledge.policy import robustness as p

c, m = (p.c, p.m)


def run():
    path = c.ROOT / "data/raw/gahd/gahd.csv"
    p.save(
        "exploratory_family_contract.json",
        dict(
            created_utc=c.a.utc(),
            inputs=c.bindings([path, p.P4 / "context_followup_checks.json"]),
            rationale="Part 4 identified directly related GAHD examples. In addition to excluding their 97 exposed queries, examine whether official contrastive families among evaluation texts affect the uncertainty calculation.",
            prior_exposure="Original and prespecified Part 5 point/interval results are already known. This is an additional exploratory sensitivity.",
            method="Build undirected connected components of every official contrastive_gahd_id link in the raw CSV, without inspecting outcomes. Replace GAHD normalized-text clusters by these family components. Other sources retain original text clusters. Keep all original common-nine inputs and labels, paired settings and source strata. 50000 draws, seed 20261114.",
            limits="Only documented GAHD links are grouped. This does not eliminate training-context overlap or detect all paraphrases/authors/templates. Original main inference remains unchanged.",
        ),
    )
    graph = defaultdict(set)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        a, b = (find(x), find(y))
        if a != b:
            parent[max(a, b)] = min(a, b)

    with path.open(encoding="utf8", newline="") as f:
        for row in csv.DictReader(f):
            x = row["gahd_id"]
            y = row["contrastive_gahd_id"].strip()
            find(x)
            if y:
                union(x, y)
                graph[x].add(y)
                graph[y].add(x)
    roster, arrays, _ = p.load()
    updated = {k: dict(v) for k, v in roster.items()}
    groups = defaultdict(list)
    meta = p.load_endpoints()["de_hate"]
    ar = arrays["de_hate"]
    for k in meta["keys"]:
        if roster[k]["source"] == "gahd":
            ident = k.split("gahd-", 1)[1]
            assert ident in parent
            updated[k]["text_cluster"] = "GAHD-family/" + find(ident)
            groups[find(ident)].append(k)
    for ident in parent:
        if ident not in graph:
            continue
        if ident != find(ident):
            continue
        stack = [ident]
        seen = set()
        while stack:
            item = stack.pop()
            if item in seen:
                continue
            seen.add(item)
            stack.extend(graph[item] - seen)
        assert all((find(item) == ident for item in seen))
    feat, strata, _ = m.cluster_features(meta, ar, updated)
    boot = m.bootstrap_from_features(feat, strata, len(meta["all_labels"]), 50000, 20261114)
    p.npzsave("exploratory_family_bootstrap.npz", boot)
    point = m.point_metrics(meta, ar)
    contrasts = p.contrasts("de_hate", point, boot, "exploratory_GAHD_family_clusters")
    grouped = defaultdict(list)
    for i, k in enumerate(meta["keys"]):
        if ar["common"][i]:
            grouped[updated[k]["text_cluster"]].append(i)
    order = sorted(grouped)
    rng = np.random.default_rng(20261114)
    w = np.zeros((256, len(order)), int)
    positions = defaultdict(list)
    for j, key in enumerate(order):
        positions[tuple(sorted({roster[meta["keys"][i]]["source"] for i in grouped[key]}))].append(
            j
        )
    for _, idx in sorted(positions.items()):
        w[:, idx] = rng.multinomial(len(idx), [1 / len(idx)] * len(idx), size=256)
    from analysis.knowledge.policy.verify_robustness import scores, close

    for draw in range(8):
        selected = [
            i
            for j, cluster in enumerate(order)
            for _ in range(w[draw, j])
            for i in grouped[cluster]
        ]
        counts = m.contributions(ar["gold"][selected], ar["predicted"][:, :, selected]).sum(axis=2)
        macro, micro = scores(counts.reshape(9, 2, 3))
        close(macro.reshape(3, 3).mean(axis=1), boot["macro_f1"][draw])
        close(micro.reshape(3, 3).mean(axis=1), boot["micro_f1_all_labels"][draw])
    p.save(
        "exploratory_family_results.json",
        dict(
            endpoint="de_hate",
            n=int(ar["common"].sum()),
            normalized_text_clusters_before=int(ar["common"].sum()),
            family_clusters_after=len(feat),
            gahd_selected_inputs=sum(map(len, groups.values())),
            gahd_families=len(groups),
            selected_families_with_multiple_inputs={k: v for k, v in groups.items() if len(v) > 1},
            contrasts=contrasts,
            independent_first_draw_values_checked=48,
            sources=c.bindings(
                [
                    __file__,
                    p.BASE / "exploratory_family_contract.json",
                    p.BASE / "exploratory_family_bootstrap.npz",
                ]
            ),
        ),
    )
    print(
        f"Exploratory family check: {sum(map(len, groups.values()))} GAHD inputs, {len(groups)} families; 48 draw values verified.",
        flush=True,
    )


if __name__ == "__main__":
    with c.offline():
        run()
