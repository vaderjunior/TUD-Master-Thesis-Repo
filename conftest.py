import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import os
import subprocess
import pytest


@pytest.fixture
def run_cli():

    def run(cwd, *args):
        env = {
            **os.environ,
            "PYTHON_DOTENV_DISABLED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        return subprocess.run(
            [sys.executable, "-B", "-m", "src.hsrag", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    return run


@pytest.fixture(scope="session", autouse=True)
def disposable_study_outputs():
    """Remove mock request files after the test session."""
    import shutil
    import tempfile
    from src.hsrag.paths import PROJECT_ROOT

    parent = PROJECT_ROOT / ".test-output"
    parent.mkdir(exist_ok=True)
    test_root = Path(tempfile.mkdtemp(prefix="t_", dir=parent)).resolve()
    if not test_root.is_relative_to(PROJECT_ROOT.resolve()):
        raise RuntimeError("Test output directory is outside the checkout")
    study_paths = {}

    def compact(relative, prefix, consumed):
        parts = relative.parts
        key = (prefix, parts[:consumed])
        if key not in study_paths:
            study_paths[key] = format(len(study_paths), "02x")
        return Path(prefix) / study_paths[key] / Path(*parts[consumed:])

    changes = []
    previous_output = os.environ.get("SQ2_TEST_OUTPUT")
    os.environ["SQ2_TEST_OUTPUT"] = str(test_root / "full_data")
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        attributes = (
            (
                "BASE",
                "OUTPUT",
                "OUT",
                "DERIVED",
                "LOCAL",
                "LAYOUT",
                "PLAN",
                "SOURCE_BASE",
                "SCHEDULE",
                "SHARED",
                "OLD",
                "A15",
            )
            if name.startswith(("analysis.", "training."))
            else ("TEST_ROOT",) if name.startswith("tests.") else ()
        )
        for attribute in attributes:
            current = getattr(module, attribute, None)
            if not isinstance(current, Path):
                continue
            try:
                relative = current.relative_to(PROJECT_ROOT)
                if relative.parts[:2] == ("experiments", "astra"):
                    relative = compact(relative, "e", 3)
            except ValueError:
                external = Path(current.drive.rstrip(":") or "root") / Path(
                    *current.parts[1:]
                )
                relative = compact(external, "x", min(3, len(external.parts)))
            changes.append((module, attribute, current))
            setattr(module, attribute, test_root / relative)
    cpu_module = sys.modules.get("training.cpu.common")
    if cpu_module is not None:
        qa = cpu_module.BASE / "qa"
        if not qa.resolve().is_relative_to(test_root):
            raise RuntimeError("CPU test output is outside the disposable directory")
        qa.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        if previous_output is None:
            os.environ.pop("SQ2_TEST_OUTPUT", None)
        else:
            os.environ["SQ2_TEST_OUTPUT"] = previous_output
        for module, attribute, original in reversed(changes):
            setattr(module, attribute, original)
        if not test_root.is_relative_to(PROJECT_ROOT.resolve()):
            raise RuntimeError("Test cleanup directory is outside the checkout")
        cleanup_path = str(test_root)
        if os.name == "nt":
            cleanup_path = "\\\\?\\" + cleanup_path
        shutil.rmtree(cleanup_path)
        if parent.exists() and (not any(parent.iterdir())):
            parent.rmdir()


def pytest_configure(config):
    if os.name == "nt" and config.option.basetemp is None:
        import tempfile
        import uuid

        parent = Path(tempfile.gettempdir()).resolve()
        target = parent / ("hsrag_" + uuid.uuid4().hex[:8])
        if (
            not target.resolve().is_relative_to(parent)
            or target == parent
            or target.exists()
        ):
            raise RuntimeError("Unexpected temporary test directory")
        config.option.basetemp = str(target)
