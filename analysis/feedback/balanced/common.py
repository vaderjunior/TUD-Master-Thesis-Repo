from pathlib import Path
import json
from analysis.common import feedback_inputs as old

ROOT = old.ROOT
BASE = ROOT / "experiments/astra/sq3_balanced_corrections_20261006_v1"
SELECTION = (
    ROOT / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection"
)
CANDIDATES = old.BASE / "benign_selection"
CORE = old.CORE
TAXONOMY = old.OLD_TAX
SEEDS = [2026100441, 2026100442, 2026100443]
GROUP = "Additional labelled examples"
require, read, rows, sha, sig, rel, utc, verify, bindings = (
    old.require,
    old.read,
    old.rows,
    old.sha,
    old.sig,
    old.rel,
    old.utc,
    old.verify,
    old.bindings,
)


def write(path, value):
    path = Path(path)
    require(
        path.resolve().is_relative_to(BASE.resolve()), "Write outside balanced study"
    )
    data = (
        value
        if isinstance(value, str)
        else json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        require(path.read_bytes() == data, "Preserve existing artifact: " + rel(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def jsonlines(path, values):
    write(
        path,
        "".join(
            json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for v in values
        ),
    )


def forbid(*args, **kwargs):
    raise AssertionError("Offline preparation: network/model access is disabled")
