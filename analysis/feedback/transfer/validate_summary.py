from analysis.feedback.transfer import summary as p
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path
import math
import numpy as np

a = p.a
OUT = p.OUT
LABELS = {
    "hate_type": (
        "explicit",
        "incitement",
        "inferiority",
        "irony",
        "stereotyping",
        "threatening",
        "grievance",
    ),
    "target_group": (
        "race",
        "religion",
        "gender",
        "sexual_orientation",
        "disability",
        "national_origin",
        "other",
    ),
    "severity": ("low", "medium", "high"),
}
GOLD = {
    "hate_type": "gold_hate_types",
    "target_group": "gold_target_groups",
    "severity": "gold_severity",
}
ELIG = {
    "hate_type": "hate_types",
    "target_group": "target_groups",
    "severity": "severity",
}


def need_near(x, y, label):
    a.need(abs(float(x) - float(y)) < 2e-12, "Point/interval mismatch " + label)


def fraction(n, d):
    return F(int(n), int(d)) if d else F(0)


def mean(values):
    return sum(values, F(0)) / len(values)


def interval(values, confidence):
    values = sorted(float(v) for v in values)
    out = []
    for q in ((1 - confidence) / 2, 1 - (1 - confidence) / 2):
        pos = (len(values) - 1) * q
        lo = math.floor(pos)
        hi = math.ceil(pos)
        out.append(values[lo] + (pos - lo) * (values[hi] - values[lo]))
    return out


def main():
    a.need(not (OUT / "verification.json").exists(), "Preserve previous proof")
    contract = a.read(OUT / "analysis_contract.json")
    p.bindings_check(contract["input_bindings"])
    r2 = a.read(p.PARTS[2] / "rates.json")
    r4 = a.read(p.PARTS[4] / "quality_results.json")
    r8 = a.read(p.PARTS[8] / "rates.json")
    families = a.read(p.PARTS[8] / "families.json")
    qwen = np.full((15758, 4, 3), -1, dtype=np.int8)
    valid = np.zeros_like(qwen)
    gold = np.zeros(15758, dtype=np.int8)
    populations = []
    clusters = []
    source = []
    keys = []
    common_ids = defaultdict(list)
    head_ids = defaultdict(list)
    full_support = defaultdict(Counter)
    conf = defaultdict(lambda: np.zeros((4, 3, 2, 2), dtype=np.int64))
    exact = defaultdict(lambda: np.zeros((4, 3), dtype=np.int64))
    label_counts = defaultdict(lambda: np.zeros((4, 3, 7, 3), dtype=np.int64))
    provided = defaultdict(lambda: np.zeros((4, 3), dtype=np.int64))
    no_all_fine = 0
    for i, row in enumerate(a.rows(p.PARTS[1] / "matrix.jsonl")):
        a.need(row["input_index"] == i, "Ordered original matrix")
        pop = row["population"]
        g = bool(row["gold_hate"])
        gold[i] = g
        populations.append(pop)
        clusters.append(row["inference_cluster"])
        source.append(row["source"])
        keys.append(row["key"])
        all_valid = all(
            v["result"] is not None for arm in row["predictions"] for v in arm
        )
        a.need(all_valid == row["masks"]["common_valid"], "Independent common mask")
        a.need(row["gold_legal"] is None, "No invented legal gold")
        no_all_fine += not all(row["eligible_fine_gold"][k] for k in ELIG.values())
        for head in LABELS:
            if row["eligible_fine_gold"][ELIG[head]]:
                v = row[GOLD[head]]
                full_support[(pop, head)].update([v] if head == "severity" else v)
                if all_valid:
                    head_ids[(pop, head)].append(i)
        if all_valid:
            common_ids[pop].append(i)
        for c, arm in enumerate(row["predictions"]):
            for rep, slot in enumerate(arm):
                pred = slot["result"]
                valid[i, c, rep] = slot["n_valid"]
                if pred is None:
                    continue
                qwen[i, c, rep] = int(pred["hate"])
                if not all_valid:
                    continue
                conf[pop][c, rep, int(g), int(pred["hate"])] += 1
                known = pred["hate"] == g
                for head, labels in LABELS.items():
                    if not row["eligible_fine_gold"][ELIG[head]]:
                        continue
                    actual = row[GOLD[head]]
                    predicted = pred[head]
                    gs = {actual} if head == "severity" else set(actual)
                    ps = (
                        ({predicted} if predicted is not None else set())
                        if head == "severity"
                        else set(predicted)
                    )
                    ok = ps == gs
                    exact[(pop, head)][c, rep] += ok
                    known = known and ok
                    if head == "severity":
                        provided[pop][c, rep] += predicted is not None
                    for j, label in enumerate(labels):
                        label_counts[(pop, head)][c, rep, j] += [
                            int(label in gs and label in ps),
                            int(label not in gs and label in ps),
                            int(label in gs and label not in ps),
                        ]
                exact[(pop, "known_answer")][c, rep] += known
    a.need(
        len(keys) == 15758 and len(set(keys)) == 15758 and no_all_fine == 15758,
        "Original population and annotation boundary",
    )
    a.need(
        np.count_nonzero(qwen < 0) == 276
        and np.count_nonzero((qwen < 0).any(axis=(1, 2))) == 150,
        "Null census",
    )
    z = np.load(p.PARTS[8] / "arrays.npz", allow_pickle=False)
    a.need(
        np.array_equal(qwen, z["qwen_decisions"])
        and np.array_equal(valid, z["valid_votes"]),
        "Classifier comparison uses exact original Qwen slots",
    )
    l = z["linear_decisions"]
    a.need(
        np.array_equal(
            l,
            z["probabilities"][:, :, None, :]
            > np.asarray([0.6, 0.5])[None, None, :, None],
        ),
        "Strict fixed thresholds",
    )
    lerr = l != gold[:, None, None, None]
    a.need(np.array_equal(lerr, z["linear_errors"]), "Classifier error bits")
    qerr = np.where(qwen < 0, -1, qwen != gold[:, None, None])
    a.need(np.array_equal(qerr, z["qwen_errors"]), "Qwen original error bits")
    primary = []
    classifier = []
    interval_cells = 0
    for j, ep in enumerate(p.ENDPOINTS):
        pop, kind = ep.rsplit("_", 1)
        ids = [i for i in common_ids[pop] if bool(gold[i]) == (kind == "FNR")]
        v = r2["main/" + ep]
        a.need(ids == v["common_indices"], "Fresh endpoint cohort")
        n = len(ids)
        counts = qerr[ids].sum(axis=(0, 2))
        ref = fraction(counts[0], 3 * n)
        treat = fraction(sum(counts[1:]), 9 * n)
        delta = treat - ref
        for name, x in [
            ("reference_rate", ref),
            ("treatment_rate", treat),
            ("difference", delta),
        ]:
            need_near(x, v[name], ep + "/" + name)
        a.need(
            v["reference_error_count"] == int(counts[0])
            and v["treatment_error_count"] == int(sum(counts[1:])),
            "Real output error denominators",
        )
        a.need(v["bootstrap"]["seed"] == 2026101216 + j, "Frozen primary seed")
        draws = np.load(p.ROOT / v["draw_file"], allow_pickle=False)
        ci = interval(draws[:, -1], 0.9875)
        for x, y in zip(ci, v["interval98_75"]):
            need_near(x, y, "Primary percentile")
            interval_cells += 1
        primary.append(
            dict(
                endpoint=ep,
                common_n=n,
                clusters=len({clusters[i] for i in ids}),
                reference=float(ref),
                feedback=float(treat),
                difference=float(delta),
                reference_exact=str(ref),
                feedback_exact=str(treat),
                difference_exact=str(delta),
                interval98_75=ci,
                reference_error_outputs=int(counts[0]),
                feedback_error_outputs=int(sum(counts[1:])),
                reference_output_count=3 * n,
                feedback_output_count=9 * n,
                source=p.a.rel(p.PARTS[2] / "rates.json"),
            )
        )
        fam = families["cohort/main/" + ep]
        a.need(
            fam["common_indices"] == ids and fam["bootstrap"]["seed"] == 2026101224 + j,
            "Classifier same cohort/frozen seed",
        )
        lc = lerr[ids, 0, 0].sum(axis=0)
        lb = fraction(lc[0], n)
        lu = fraction(sum(lc[1:]), 3 * n)
        ld = lu - lb
        dd = ld - delta
        gap = lu - treat
        v8 = r8["all_permitted/cutoff_06/cohort/main/" + ep]
        values = dict(
            linear_baseline=lb,
            linear_updated=lu,
            qwen_reference=ref,
            qwen_updated=treat,
            linear_change=ld,
            qwen_change=delta,
            difference_in_change=dd,
            updated_absolute_difference=gap,
        )
        for name, x in values.items():
            need_near(x, v8["estimates"][name], ep + "/" + name)
        draws = np.load(p.ROOT / fam["draw_file"], allow_pickle=False)
        ci = interval(draws[:, 0, 6], 0.9875)
        for x, y in zip(ci, v8["primary_interval98_75"]):
            need_near(x, y, "Classifier percentile")
            interval_cells += 1
        classifier.append(
            dict(
                endpoint=ep,
                common_n=n,
                estimates={k: float(x) for k, x in values.items()},
                exact={k: str(x) for k, x in values.items()},
                primary_interval98_75=ci,
                own_change_intervals95={
                    k: v8["intervals95"][k] for k in ("linear_change", "qwen_change")
                },
                source=a.rel(p.PARTS[8] / "rates.json"),
            )
        )
    quality = []

    def record(pop, head, metric, run):
        q = r4[f"population/{pop}/{head}/main"]
        m = q["metrics"][metric]
        ref = mean(run[0])
        treat = mean([v for arm in run[1:] for v in arm])
        diff = treat - ref
        for k, x in [("reference", ref), ("treatment", treat), ("difference", diff)]:
            need_near(x, m[k], "Quality " + pop + "/" + head + "/" + metric + "/" + k)
        quality.append(
            dict(
                population=pop,
                head=head,
                metric=metric,
                common_n=q["common_n"],
                reference=float(ref),
                feedback=float(treat),
                difference=float(diff),
                interval95=m["interval95"],
                interval_role="supporting_prior_independent_proof",
                reference_exact=str(ref),
                feedback_exact=str(treat),
                source=a.rel(p.PARTS[4] / "quality_results.json"),
            )
        )

    for pop, ids in common_ids.items():
        n = len(ids)
        q = r4[f"population/{pop}/binary/main"]
        a.need(q["common_indices"] == ids, "Binary main quality mask")
        runs = {k: [] for k in ("macro_f1", "balanced_accuracy")}
        for c in range(4):
            for k in runs:
                runs[k].append([])
            for rep in range(3):
                tn, fp, fn, tp = conf[pop][c, rep].ravel()
                runs["macro_f1"][c].append(
                    mean(
                        [
                            fraction(2 * tp, 2 * tp + fp + fn),
                            fraction(2 * tn, 2 * tn + fp + fn),
                        ]
                    )
                )
                runs["balanced_accuracy"][c].append(
                    mean([fraction(tp, tp + fn), fraction(tn, tn + fp)])
                )
        for metric, run in runs.items():
            record(pop, "binary", metric, run)
        record(
            pop,
            "known_answer",
            "known_answer_exact",
            [[fraction(x, n) for x in arm] for arm in exact[(pop, "known_answer")]],
        )
        for head, labels in LABELS.items():
            q = r4[f"population/{pop}/{head}/main"]
            ids = head_ids[(pop, head)]
            a.need(q["common_indices"] == ids, "Fine eligibility mask")
            if not ids:
                a.need(
                    q["status"] == "unscorable_no_known_gold",
                    "No invented fine quality",
                )
                continue
            n = len(ids)
            mask = [
                j
                for j, label in enumerate(labels)
                if full_support[(pop, head)][label] >= 10
            ]
            a.need(
                q["population_macro_labels"] == [labels[j] for j in mask],
                "Full roster fixed macro support",
            )
            run = []
            for c in range(4):
                arm = []
                for rep in range(3):
                    arm.append(
                        mean(
                            [
                                fraction(
                                    2 * label_counts[(pop, head)][c, rep, j, 0],
                                    2 * label_counts[(pop, head)][c, rep, j, 0]
                                    + label_counts[(pop, head)][c, rep, j, 1]
                                    + label_counts[(pop, head)][c, rep, j, 2],
                                )
                                for j in mask
                            ]
                        )
                    )
                run.append(arm)
            record(pop, head, "macro_f1", run)
            record(
                pop,
                head,
                "exact_set",
                [[fraction(x, n) for x in arm] for arm in exact[(pop, head)]],
            )
            if head == "severity":
                record(
                    pop,
                    head,
                    "severity_provided",
                    [[fraction(x, n) for x in arm] for arm in provided[pop]],
                )
    a.need(len(quality) == 15, "All main scored quality quantities")

    for view in ("representative", "representative_replay_excluded"):
        a.need(
            r2[view + "/en_hatecheck_eval_FNR"]["interval98_75"][1] == 0,
            "Exact-zero HateCheck sensitivity",
        )
    a.need(
        classifier[3]["exact"]["difference_in_change"] == "0",
        "Exact-zero classifier change contrast",
    )
    addition = a.read(p.PARTS[9] / "resource_addition.json")
    old = a.read(p.ROOT / addition["previous_ledger"])
    a.need(
        old["programmes"] == addition["preserved_previous_programmes"],
        "Eleven preserved ledger rows",
    )
    rows = [*old["programmes"], addition["addition"]]
    fields = tuple(old["totals"])
    totals = {k: sum(q[k] for q in rows) for k in fields}
    a.need(
        totals == addition["prospective_twelve_programme_totals"],
        "Exact-once ledger total",
    )
    a.need(
        len(rows) == 12 and totals["reported_tokens"] == 1860042318,
        "Twelve remote programmes",
    )
    ledger = {
        **old,
        "previous_ledger": addition["previous_ledger"],
        "previous_ledger_sha256": a.sha(p.ROOT / addition["previous_ledger"]),
        "new_programme_source": a.rel(p.PARTS[9] / "resource_groups.json"),
        "new_programme_source_sha256": a.sha(p.PARTS[9] / "resource_groups.json"),
        "programmes": rows,
        "totals": totals,
        "unknown_remote_usage_preserved": True,
        "whole_thesis_cost": False,
        "capacity_diagnostics_excluded": True,
        "classifier_fits_accounted_separately": True,
        "inherited_screening_counted_once": True,
        "scientifically_reviewed_transfer": True,
    }
    p.save(OUT / "partial_programme_ledger.json", ledger)
    p.save(
        OUT / "display_values.json",
        dict(
            primary=primary,
            classifier=classifier,
            quality=quality,
            resources=a.read(p.PARTS[9] / "resource_summary.json"),
            prior_draw_proofs=contract["reused_resampling_proofs"],
        ),
    )
    p.save(
        OUT / "verification.json",
        dict(
            status="part10_fresh_main_arithmetic_and_bound_prior_proofs_passed",
            checked_utc=a.utc(),
            script_sha256=a.sha(Path(__file__)),
            contract_sha256=a.sha(OUT / "analysis_contract.json"),
            immutable_bindings_checked=len(contract["input_bindings"]),
            prior_parts=9,
            current_scientific_figures=25,
            original_matrix_rows=15758,
            original_Qwen_slots=189096,
            headlines_freshly_reconstructed=4,
            classifier_contrasts_freshly_reconstructed=4,
            main_quality_quantities_freshly_reconstructed=len(quality),
            headline_interval_endpoints_rechecked=interval_cells,
            prior_independently_regenerated_bootstrap_sets=335,
            prior_independently_verified_scalars=77220000,
            new_bootstrap_draws=0,
            new_raw_response_replay=False,
            exact_zero_boundaries_retained=True,
            partial_remote_programmes=12,
            resource_row_added_once=True,
            artifacts={
                a.rel(OUT / name): a.sha(OUT / name)
                for name in ("display_values.json", "partial_programme_ledger.json")
            },
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_annotations=0,
            new_primary_families=0,
        ),
    )
    print(
        "Part 10 verification passed: four primary effects, four classifier contrasts, fifteen quality quantities and exact-once resource extension; prior full proofs retained."
    )


if __name__ == "__main__":
    with a.offline():
        main()
