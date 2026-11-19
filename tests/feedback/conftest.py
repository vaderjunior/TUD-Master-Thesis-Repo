import copy
from pathlib import Path
import shutil
import tempfile
import pytest
import yaml
from src.hsrag.paths import PROJECT_ROOT


@pytest.fixture(scope="session")
def taxonomy_fixture_paths(disposable_study_outputs):
    parent = PROJECT_ROOT / ".test-output"
    folder = Path(tempfile.mkdtemp(prefix="taxonomy_", dir=parent)).resolve()
    assert folder.is_relative_to(parent.resolve())
    full = yaml.safe_load(
        (PROJECT_ROOT / "config/taxonomy.yaml").read_text(encoding="utf-8")
    )
    labels = list(full["dimensions"]["hate_type"]["labels"])
    for name in ["full", *labels]:
        value = copy.deepcopy(full)
        if name != "full":
            value["dimensions"]["hate_type"]["labels"].pop(name)
        (folder / (name + ".yaml")).write_text(
            yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
        )
    try:
        yield folder
    finally:
        assert folder.is_relative_to(parent.resolve())
        shutil.rmtree(folder)


@pytest.fixture(autouse=True)
def synthetic_taxonomy_variants(taxonomy_fixture_paths, monkeypatch):
    from analysis.common import schema

    original = schema.adapter

    def adapter(path):
        source = Path(str(path).removeprefix(chr(92) * 2 + "?" + chr(92))).resolve()
        if (
            source.parent.name == "taxonomies"
            and source.parent.parent.name == "sq3_full_kb_taxonomy_20261001_v1"
        ):
            return original(taxonomy_fixture_paths / source.name)
        return original(path)

    monkeypatch.setattr(schema, "adapter", adapter)
