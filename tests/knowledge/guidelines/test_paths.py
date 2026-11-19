"""Read moved evaluation bytes through their original logical path."""

import hashlib
from src.hsrag import paths
from analysis.common import api


def test_moved_input_retains_original_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    new = tmp_path / "data/evaluations/english_guidelines.json"
    new.parent.mkdir(parents=True)
    raw = b'{"fixture": 1}\n'
    new.write_bytes(raw)
    old = (
        tmp_path
        / "experiments/astra/sq1_full_kb_guideline_oracle_20260925_v1/analysis_results_v1/gate_matrix.json"
    )
    assert api.read(old) == {"fixture": 1}
    assert api.digest(old) == hashlib.sha256(raw).hexdigest()
    old.parent.mkdir(parents=True)
    old.write_bytes(b'{"fixture": 2}')
    assert api.read(old) == {"fixture": 2}
