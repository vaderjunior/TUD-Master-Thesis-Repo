"""Locate project assets independently of the current working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluation_path(path: str | Path) -> Path:
    """Resolve the moved guideline input when its original path is absent."""
    path = project_path(path)
    if path.exists():
        return path
    try:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path
    original = "experiments/astra/sq1_full_kb_guideline_oracle_20260925_v1/analysis_results_v1/gate_matrix.json"
    return (
        project_path("data/evaluations/english_guidelines.json") if relative == original else path
    )
