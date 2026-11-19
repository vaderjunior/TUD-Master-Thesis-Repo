from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
import numpy as np

TYPES = (
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
)
MASKS = ("ordinary_valid", "all_three_votes", "one_per_cluster")
COHORTS = ("full", "same_six_excluded")


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A15-support-v1|" + name).encode()).digest()[:8], "little"
    )


def bitset(labels):
    need(
        len(labels) == len(set(labels)) and set(labels) <= set(TYPES),
        "Invalid type set",
    )
    return sum(1 << TYPES.index(t) for t in labels)


def specs(plan):

    need(
        plan["bootstrap_draws"] == 10000 and plan["decoder_repeats"] == 3,
        "Draw/repeat plan",
    )
    pools = plan["pool_seeds"]
    need(pools == [2026100441, 2026100442, 2026100443], "Pool plan")
    groups = sorted(
        {
            c.rsplit("_s", 1)[0]
            for c in plan["conditions"]
            if not c.startswith("reference")
        }
    )
    need(len(groups) == 24 and len(plan["conditions"]) == 74, "Condition inventory")

    def arm(group):
        return (
            [group]
            if group.startswith("reference")
            else [f"{group}_s{s}" for s in pools]
        )

    out = {}

    def add(name, family, t, c, ep, confidence, rng):
        out[name] = dict(
            id=name,
            family=family,
            treatment_conditions=arm(t),
            control_conditions=arm(c),
            endpoint=ep,
            confidence=confidence,
            seed=rng,
            draws=10000,
        )

    for family in ("primary", "mechanism"):
        p = plan[family]
        need(
            p["confidence"] == (0.975 if family == "primary" else 1 - 0.05 / 6),
            "Family confidence",
        )
        controls = [p["control"]] if family == "primary" else p["controls"]
        for i, c in enumerate(controls):
            for j, ep in enumerate(("FPR", "FNR")):
                add(
                    f"{family}/{c}/{ep}",
                    family,
                    p["treatment"],
                    c,
                    ep,
                    p["confidence"],
                    p["bootstrap_seeds"][2 * i + j],
                )
    for g in sorted(groups + ["reference4"]):
        for ep in ("FPR", "FNR"):
            label = f"{g}-reference5/{ep}"
            add(
                "supporting/" + label,
                "supporting",
                g,
                "reference5",
                ep,
                0.95,
                seed(label),
            )
    for mode in ("natural5", "reserved5", "extended6", "masked5"):
        pairs = [
            (a + "_" + composition, b + "_" + composition)
            for composition in ("positive", "balanced")
            for a, b in (
                ("error", "correct"),
                ("error", "random"),
                ("correct", "random"),
            )
        ]
        pairs += [
            (s + "_balanced", s + "_positive") for s in ("error", "correct", "random")
        ]
        for left, right in pairs:
            for ep in ("FPR", "FNR"):
                label = f"part4-direct-v1/{mode}/{left}-minus-{right}/{ep}"
                add(
                    label,
                    "additional_part4_supporting",
                    mode + "__" + left,
                    mode + "__" + right,
                    ep,
                    0.95,
                    seed(label),
                )
    need(len(out) == 130, "Contrast inventory")
    return out


def representative_indices(indices, roster):
    groups = defaultdict(list)
    for i in indices:
        groups[(roster[i]["gold"]["hate"], roster[i]["cluster"])].append(i)
    chosen = {
        min(
            g,
            key=lambda i: hashlib.sha256(
                ("A15-representative|" + roster[i]["key"]).encode()
            ).digest(),
        )
        for g in groups.values()
    }
    return np.asarray([i for i in indices if i in chosen], dtype=int)


def frac(n, d, zero=F(0)):
    return F(int(n), int(d)) if d else zero


def mean(values):
    return (
        sum(values, F(0)) / len(values)
        if values and all(v is not None for v in values)
        else None
    )


def score(gold, predicted, fine):

    n = len(gold)
    if fine:
        per = []
        for k in range(7):
            g, p = (gold & (1 << k)) != 0, (predicted & (1 << k)) != 0
            tp = int((g & p).sum())
            per.append(frac(2 * tp, int(g.sum() + p.sum())))
        return {
            "type_macro_f1_fixed7": mean(per) if n else None,
            "type_exact_set": frac((gold == predicted).sum(), n, None),
        }
    tp = int(((gold == 1) & (predicted == 1)).sum())
    tn = int(((gold == 0) & (predicted == 0)).sum())
    fp = int(((gold == 0) & (predicted == 1)).sum())
    fn = int(((gold == 1) & (predicted == 0)).sum())
    return {
        "binary_macro_f1": (
            (frac(2 * tp, 2 * tp + fp + fn) + frac(2 * tn, 2 * tn + fp + fn)) / 2
            if n
            else None
        ),
        "balanced_accuracy": (
            (F(tp, tp + fn) + F(tn, tn + fp)) / 2 if tp + fn and tn + fp else None
        ),
    }


class Data:
    def __init__(self, roster, conditions, records):
        self.roster, self.conditions = roster, conditions
        self.keys = [r["key"] for r in roster]
        self.positions = {k: i for i, k in enumerate(self.keys)}
        self.columns = {
            (c, r): 3 * i + r for i, c in enumerate(conditions) for r in range(3)
        }
        shape = (len(roster), 3 * len(conditions))
        self.hate = np.full(shape, -1, dtype=np.int8)
        self.types = np.zeros(shape, dtype=np.int16)
        self.valid = np.full(shape, -1, dtype=np.int8)
        self.gold = np.asarray([r["gold"]["hate"] for r in roster], dtype=np.int8)
        self.gold_types = np.asarray(
            [
                (
                    bitset(r["gold"]["hate_type"])
                    if r["gold"]["hate_type"] is not None
                    else -1
                )
                for r in roster
            ],
            dtype=np.int16,
        )
        self.resource = Counter()
        self.batch = {}
        for r in records:
            i, j = self.positions[r["key"]], self.columns[r["condition"], r["repeat"]]
            need(self.valid[i, j] == -1, "Duplicate prediction")
            v, p = r["n_valid"], r["result"]
            need(
                type(v) is int and 0 <= v <= 3 and ((p is None) == (v == 0)),
                "Invalid prediction",
            )
            self.valid[i, j] = v
            if p is not None:
                need(
                    type(p["hate"]) is bool and (p["hate"] or not p["hate_type"]),
                    "Invalid gate",
                )
                self.hate[i, j] = p["hate"]
                self.types[i, j] = bitset(p["hate_type"])
            need(
                r["key"] not in self.batch or self.batch[r["key"]] == r["stage"],
                "Split batch input",
            )
            self.batch[r["key"]] = r["stage"]
            for group in (
                "accounting",
                "observed_usage",
                "ordinary_usage",
                "repair_usage",
            ):
                self.resource.update(
                    {group + "/" + k: int(v) for k, v in r.get(group, {}).items()}
                )
            self.resource["administrative_unknown_requests"] += r[
                "administrative_unknown_requests"
            ]
        need((self.valid >= 0).all(), "Missing planned prediction")

    def arm(self, spec, side):
        return [
            self.columns[c, r] for c in spec[side + "_conditions"] for r in range(3)
        ]

    def selected(
        self, excluded=(), representatives=False, endpoint=None, fine=False, source=None
    ):
        excluded = set(excluded)
        indices = np.asarray(
            [
                i
                for i, r in enumerate(self.roster)
                if r["key"] not in excluded
                and (source is None or r["source"] == source)
                and (endpoint is None or bool(self.gold[i]) == (endpoint == "FNR"))
                and (not fine or self.gold_types[i] >= 0)
            ],
            dtype=int,
        )
        return (
            representative_indices(indices, self.roster) if representatives else indices
        )

    def mask(
        self,
        spec,
        *,
        excluded=(),
        mask="ordinary_valid",
        endpoint=None,
        fine=False,
        source=None,
    ):
        ix = self.selected(excluded, mask == "one_per_cluster", endpoint, fine, source)
        arms = [self.arm(spec, side) for side in ("treatment", "control")]
        threshold = 3 if mask == "all_three_votes" else 1
        oks = [self.valid[np.ix_(ix, a)] >= threshold for a in arms]
        common = ix[np.logical_and(oks[0].all(axis=1), oks[1].all(axis=1))]
        return ix, common, arms, oks

    def rate(self, spec, **kwargs):
        ix, common, arms, oks = self.mask(spec, endpoint=spec["endpoint"], **kwargs)
        truth = int(spec["endpoint"] == "FNR")
        counts = [int((self.hate[np.ix_(common, a)] != truth).sum()) for a in arms]
        values = [frac(n, len(common) * len(a), None) for n, a in zip(counts, arms)]
        missing = [int((~ok).sum()) for ok in oks]
        bounds = []
        for a, ok, m in zip(arms, oks, missing):
            errors = int(((self.hate[np.ix_(ix, a)] != truth) & ok).sum())
            bounds.append(
                [
                    frac(errors, len(ix) * len(a), None),
                    frac(errors + m, len(ix) * len(a), None),
                ]
            )
        diff = values[0] - values[1] if len(common) else None
        bd = (
            [bounds[0][0] - bounds[1][1], bounds[0][1] - bounds[1][0]]
            if len(ix)
            else None
        )
        return dict(
            eligible_n=len(ix),
            common_n=len(common),
            common_ids=[self.keys[i] for i in common],
            rates=values,
            difference=diff,
            real_error_counts=counts,
            outputs_per_input=list(map(len, arms)),
            missing_real_outputs=missing,
            bounds=bd,
            indices=common,
            arms=arms,
        )

    def quality(self, spec, fine, **kwargs):
        ix, common, arms, _ = self.mask(spec, fine=fine, **kwargs)
        gold = self.gold_types if fine else self.gold
        pred = self.types if fine else self.hate
        metrics = tuple(score(gold[:0], gold[:0], fine))
        vals = [
            [score(gold[common], pred[common, col], fine) for col in arm]
            for arm in arms
        ]
        result = {}
        for metric in metrics:
            t, c = [mean([v[metric] for v in arm]) for arm in vals]
            result[metric] = {
                "treatment": t,
                "control": c,
                "difference": t - c if t is not None and c is not None else None,
            }
        return dict(
            eligible_n=len(ix),
            common_n=len(common),
            common_ids=[self.keys[i] for i in common],
            metrics=result,
            indices=common,
            arms=arms,
        )

    def quality_bounds(self, spec, fine, **kwargs):
        ix, _, arms, oks = self.mask(spec, fine=fine, **kwargs)
        gold = (self.gold_types if fine else self.gold)[ix]
        predicted = self.types if fine else self.hate
        metrics = tuple(score(gold[:0], gold[:0], fine))
        arm_bounds = []
        for arm, ok in zip(arms, oks):
            results = []
            for best in (False, True):
                runs = []
                for j, col in enumerate(arm):
                    p = predicted[ix, col].copy()
                    p[~ok[:, j]] = (
                        gold[~ok[:, j]]
                        if best
                        else (gold[~ok[:, j]] ^ (127 if fine else 1))
                    )
                    runs.append(score(gold, p, fine))
                results.append({m: mean([r[m] for r in runs]) for m in metrics})
            arm_bounds.append(results)
        out = {}
        for m in metrics:
            t, c = [[r[m] for r in arm] for arm in arm_bounds]
            out[m] = dict(
                treatment=t,
                control=c,
                opposing_difference=(
                    [t[0] - c[1], t[1] - c[0]]
                    if all(v is not None for v in t + c)
                    else [None, None]
                ),
            )
        return dict(
            eligible_n=len(ix),
            missing_real_outputs_by_arm=[int((~ok).sum()) for ok in oks],
            metrics=out,
        )


def omit(spec, pool):
    result = dict(spec)
    for side in ("treatment", "control"):
        field = side + "_conditions"
        result[field] = [c for c in spec[field] if not c.endswith("_s" + str(pool))]
        need(result[field], "Empty omitted arm")
    return result


def exact_quantile(ordered, q):
    at = q * (len(ordered) - 1)
    lo = at.numerator // at.denominator
    part = at - lo
    return ordered[lo] * (1 - part) + ordered[min(lo + 1, len(ordered) - 1)] * part


def rate_draws(data, spec, point):

    common, arms = point["indices"], point["arms"]
    labels = sorted({data.roster[i]["cluster"] for i in common})
    pos = {c: i for i, c in enumerate(labels)}
    unit = math.lcm(*map(len, arms))
    numerators = np.zeros(len(labels), dtype=np.int64)
    sizes = np.zeros(len(labels), dtype=np.int64)
    truth = int(spec["endpoint"] == "FNR")
    for i in common:
        j = pos[data.roster[i]["cluster"]]
        sizes[j] += 1
        numerators[j] += sum(
            (1 if side == 0 else -1)
            * (unit // len(a))
            * int((data.hate[i, a] != truth).sum())
            for side, a in enumerate(arms)
        )
    if not labels:
        return np.empty(0), None, {}
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    exact = []
    for start in range(0, spec["draws"], 200):
        chosen = rng.integers(
            0, len(labels), size=(min(200, spec["draws"] - start), len(labels))
        )
        num = numerators[chosen].sum(axis=1)
        den = sizes[chosen].sum(axis=1) * unit
        exact.extend(F(int(a), int(b)) for a, b in zip(num, den))
    saved_order = np.asarray([float(v) for v in exact])
    confidence = F(spec["confidence"]).limit_denominator(10000)
    alpha = (1 - confidence) / 2
    ordered = sorted(exact)
    interval = [exact_quantile(ordered, q) for q in (alpha, 1 - alpha)]
    return (
        saved_order,
        interval,
        dict(
            clusters=len(labels),
            integer_unit=unit,
            exact_zero_draws=exact.count(F(0)),
            confidence=str(confidence),
            quantiles=[str(alpha), str(1 - alpha)],
        ),
    )


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
