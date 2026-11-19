"""Check saved-matrix replay and output preservation."""

import json
import os
import subprocess
import sys
import numpy as np
import pytest
from analysis.knowledge.guidelines.replay import analyze
from analysis.knowledge.guidelines.design import CONDITIONS


def matrix():
    values = np.zeros((3, 5, 3), dtype=int)
    values[0, 0] = 1
    values[2, 1, 2] = -1
    return {
        "conditions": list(CONDITIONS),
        "repeats": [0, 1, 2],
        "missing_value": -1,
        "items": [
            {
                "key": str(i),
                "gate": False,
                "functionality": "fixture",
                "templ_id": "a" if i < 2 else "b",
            }
            for i in range(3)
        ],
        "values": values.tolist(),
    }


def test_missing_predictions_use_one_shared_mask():
    report = analyze(matrix(), 20)
    assert report["scorable_items"] == 2 and report["missing_predictions"] == 1
    summary = report["scopes"]["all"]
    assert summary["common_templates"] == 1 and summary["excluded_keys"] == ["2"]
    assert summary["conditions"]["no_guidelines"]["common"]["values"] == [0.5, 0.5, 0.5]
    assert sum(c["primary_family_interval"] is not None for c in summary["contrasts"].values()) == 4
    assert all(
        c["primary_family_interval"] is None
        for c in report["scopes"]["fixture"]["contrasts"].values()
    )


def test_wrong_roster_and_prediction_layout_rejected():
    changed = matrix()
    changed["items"][0]["gate"] = True
    with pytest.raises(ValueError):
        analyze(changed, 20)
    changed = matrix()
    changed["items"][0]["key"] = "1"
    with pytest.raises(ValueError):
        analyze(changed, 20)
    with pytest.raises(ValueError):
        analyze(matrix(), 0)


def test_command_outside_checkout_preserves_output(tmp_path):
    inputs, output = tmp_path / "matrix.json", tmp_path / "results.json"
    inputs.write_text(json.dumps(matrix()), encoding="utf-8")
    args = [
        sys.executable,
        "-B",
        "-m",
        "analysis.knowledge.guidelines.replay",
        "--matrix",
        str(inputs),
        "--output",
        str(output),
        "--draws",
        "20",
    ]
    env = {**os.environ, "PYTHON_DOTENV_DISABLED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    first = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    before = output.read_bytes()
    assert json.loads(before)["scorable_items"] == 2
    second = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert second.returncode != 0 and output.read_bytes() == before
