"""Runnable CLI, prompt contracts and optional indexing dependencies."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.hsrag.classify import Arms, classify, text_hash
from src.hsrag.client import MockClient
from src.hsrag.kb import build, unsentinel_meta
from src.hsrag.prompt import PromptContext, _render_gold, build_prompt
from src.hsrag.retrieve import Hit

@pytest.mark.parametrize('meta,expected', [({'gate': None, 'legal': []}, 'legal=[]'), ({'gate': None, 'legal': ['insult_defamation']}, 'legal=[insult_defamation]'), ({'gate': False, 'legal': ['insult_defamation']}, 'hate=false, legal=[insult_defamation]'), ({'gate': True, 'hate_types': [], 'severity': None}, 'hate=true, hate_type=[]'), ({'gate': None, 'astra_aux_stgb': '130'}, '\n  Auxiliary DeTox paragraph flags (illustrative, not legal gold): §130')])
def test_only_observed_example_annotations_are_rendered(meta, expected):
    assert _render_gold(meta) == expected

def test_public_and_saved_corrected_rendering_match():
    from analysis.common.prompt import render_messages
    h = Hit('example', 'example', 'Ein Text', {'gate': None, 'legal': [], 'astra_aux_stgb': '130'}, 0.0, 'dense')
    ctx = PromptContext(example_groups=[('Labelled examples', [h])])
    stored = {'definitions': [], 'guidelines': [], 'example_groups': [['Labelled examples', [vars(h)]]]}
    assert build_prompt('Neue Aussage', 'de', ctx) == render_messages('Neue Aussage', 'de', stored, 'corrected')

def test_legacy_rendering_remains_available():
    from analysis.common.prompt import module
    assert module('legacy')._render_gold({'gate': None, 'legal': []}) == 'hate=false'
    assert module('corrected')._render_gold({'gate': None, 'legal': []}) == 'legal=[]'

def test_mock_classification_from_another_directory(tmp_path, run_cli):
    path = tmp_path / 'predictions.jsonl'
    result = run_cli(tmp_path, 'classify', '--provider', 'mock', '--text', 'A sample.', '--lang', 'en', '--output', str(path))
    assert result.returncode == 0, result.stderr
    record = json.loads(path.read_text())
    assert record['n_runs'] == record['n_valid'] == 3
    assert record['active_model'] == 'mock' and len(record['raw_runs']) == 3
    assert json.loads(result.stdout) == record['result']

def test_jsonl_input_and_existing_output_are_handled(tmp_path, run_cli):
    inputs, output = (tmp_path / 'inputs.jsonl', tmp_path / 'predictions.jsonl')
    inputs.write_text(json.dumps({'id': 'one', 'lang': 'de', 'text': 'Schönes Wetter.'}) + '\n', encoding='utf-8-sig')
    first = run_cli(tmp_path, 'classify', '--provider', 'mock', '--input', str(inputs), '--output', str(output))
    assert first.returncode == 0, first.stderr
    assert json.loads(output.read_text(encoding='utf-8'))['text_hash'] == text_hash('Schönes Wetter.')
    before = output.read_bytes()
    second = run_cli(tmp_path, 'classify', '--provider', 'mock', '--input', str(inputs), '--output', str(output))
    assert second.returncode == 2 and output.read_bytes() == before

def test_import_and_mock_do_not_load_retrieval_or_credentials(tmp_path):
    code = "import sys; from src.hsrag.client import MockClient; from src.hsrag.classify import Arms, classify; from src.hsrag import client; client.load_dotenv=lambda *a, **k: (_ for _ in ()).throw(AssertionError('dotenv read')); classify('x','sample','en','zero_shot',MockClient(),Arms()); assert not {'torch','sentence_transformers','chromadb'} & set(sys.modules)"
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

def test_invalid_vote_count_fails_before_requesting():
    with pytest.raises(ValueError, match='votes'):
        classify('x', 'text', 'en', 'zero_shot', MockClient(), Arms(), n_votes=0)

def test_index_ingestion_respects_batch_size_and_metadata(tmp_path, monkeypatch):
    records = [{'id': str(i), 'kind': 'example', 'lang': 'de', 'text': f'text {i}', 'source': 'fixture', 'dimension': None, 'label': None, 'meta': {'gate': None, 'legal': []}} for i in range(5)]
    source = tmp_path / 'records.jsonl'
    source.write_text('\n'.join((json.dumps(r) for r in records)))
    batches = []

    class Collection:

        def add(self, **kwargs):
            assert len(kwargs['ids']) <= 2
            batches.append(kwargs)

        def count(self):
            return sum((len(x['ids']) for x in batches))

    class Client:

        def get_max_batch_size(self):
            return 2

        def delete_collection(self, name):
            pass

        def create_collection(self, name, metadata):
            assert metadata['hnsw:space'] == 'cosine' and metadata['embedding_model'] == 'fixture'
            return Collection()

    class Model:

        def __init__(self, name):
            assert name == 'fixture'

        def encode(self, texts, **kwargs):
            assert kwargs['normalize_embeddings']
            return np.asarray([[i, 1] for i in range(len(texts))])
    monkeypatch.setitem(sys.modules, 'chromadb', SimpleNamespace(PersistentClient=lambda **kw: Client()))
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Model))
    build(source, tmp_path / 'index', 'fixture')
    assert [i for batch in batches for i in batch['ids']] == [r['id'] for r in records]
    assert all((unsentinel_meta(meta)['gate'] is None and unsentinel_meta(meta)['legal'] == [] for batch in batches for meta in batch['metadatas']))
