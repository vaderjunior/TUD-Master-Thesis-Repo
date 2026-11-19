"""Use the analysis offline contexts when collecting their tests with pytest."""

import importlib
import pytest

CONTEXTS = {
    "tests.knowledge.german_guidelines.test_context": "analysis.knowledge.german_guidelines.context",
    "tests.knowledge.german_guidelines.test_hate": "analysis.knowledge.german_guidelines.hate",
    "tests.knowledge.german_guidelines.test_legal": "analysis.knowledge.german_guidelines.legal",
    "tests.knowledge.german_guidelines.test_reader": "analysis.knowledge.german_guidelines.reader",
}


@pytest.fixture(autouse=True)
def analysis_offline_context(request):
    module = CONTEXTS.get(request.module.__name__)
    if module:
        with importlib.import_module(module).offline():
            yield
    else:
        yield
