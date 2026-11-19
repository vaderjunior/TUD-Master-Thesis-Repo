"""Policy classification."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
from pathlib import Path
import time
import json
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import reader as reader
from analysis.knowledge.policy import metrics as m

def close(actual, expected, message):
    c.require(np.allclose(actual, expected, atol=1e-12, rtol=0), message)

def verify_scores(meta, arrays, point):
    mask = arrays['common']
    y = arrays['gold'][mask]
    p = arrays['predicted'][:, :, mask]
    tested = 0
    for ci in range(3):
        for repeat in range(3):
            pred = p[ci, repeat]
            close(point['macro_f1_repeats'][ci, repeat], f1_score(y, pred, average='macro', zero_division=0), 'Independent sklearn macro differs')
            close(point['per_label_f1_repeats'][ci, repeat], f1_score(y, pred, average=None, zero_division=0), 'Independent label F1 differs')
            if meta['dimension'] in ('hate_type', 'target_group', 'legal'):
                close(point['micro_f1_repeats'][ci, repeat], f1_score(y, pred, average='micro', zero_division=0), 'Independent micro differs')
                close(point['jaccard_repeats'][ci, repeat], jaccard_score(y, pred, average='samples', zero_division=1), 'Independent Jaccard differs')
            tested += 1
        close(point['precision'][ci], np.mean([precision_score(y, p[ci, r], average=None, zero_division=0) for r in range(3)], axis=0), 'Precision differs')
        close(point['recall'][ci], np.mean([recall_score(y, p[ci, r], average=None, zero_division=0) for r in range(3)], axis=0), 'Recall differs')
        pairs = []
        stable_right = stable_wrong = 0
        for i in range(len(y)):
            gold = tuple(np.flatnonzero(y[i]).tolist())
            states = []
            correct = []
            for repeat in range(3):
                labels = tuple(np.flatnonzero(p[ci, repeat, i]).tolist())
                fine = meta['dimension'] not in ('hate', 'legal')
                gate = bool(arrays['gates'][ci, repeat, mask][i])
                states.append((gate, labels) if fine else labels)
                correct.append(labels == gold and (gate or not fine))
            pairs.append(sum((states[a] == states[b] for a, b in m.PAIRS)) / 3)
            stable_right += all(correct)
            stable_wrong += len(set(states)) == 1 and (not any(correct))
        close(np.mean(pairs), point['repeatability']['pairwise_agreement'][ci], 'Scalar pair agreement differs')
        close(stable_right / len(y), point['repeatability']['stable_correct'][ci], 'Stable correct differs')
        close(stable_wrong / len(y), point['repeatability']['stable_wrong'][ci], 'Stable wrong differs')
    return tested

def joint_results(endpoints, all_arrays, roster):
    cohorts = (('hate_and_types', 'en_hate_type', ['en_hate_type']), ('hate_and_targets', 'en_target_group', ['en_target_group']), ('hate_targets_severity', 'en_severity', ['en_target_group', 'en_severity']))
    out = {}
    for name, reference, fields in cohorts:
        keys = endpoints[reference]['keys']
        common = np.ones(len(keys), bool)
        correct = np.ones((3, 3, len(keys)), bool)
        for endpoint in fields:
            positions = {k: i for i, k in enumerate(endpoints[endpoint]['keys'])}
            selected = np.array([positions[k] for k in keys])
            data = all_arrays[endpoint]
            common &= data['common'][selected]
            correct &= data['correct'][:, :, selected]
        c.require(all((roster[k]['gate'] is True for k in keys)), 'Joint cohort not positive gate gold')
        correct = correct[:, :, common]
        out[name] = dict(eligible=len(keys), n=int(common.sum()), endpoint_fields=fields, exact=correct.mean(axis=(1, 2)).tolist(), exact_repeats=correct.mean(axis=2).tolist(), all_three_correct=correct.all(axis=1).mean(axis=1).tolist())
    return out

def run():
    base = reader.BASE
    receipt = c.read(base / 'reader_complete.json')
    c.require(receipt['status'] == 'all_saved_results_verified_and_replayed', 'Reader incomplete')
    c.verify(receipt['reader_files'])
    c.verify(receipt['reader_sources'])
    c.verify(c.read(base / 'analysis_contract.json')['original_bindings'])
    registry = c.read(c.SETUP / 'comparison_registry.json')
    c.require(registry['draws'] == 50000 and registry['seed'] == 20261114, 'Frozen inference differs')
    tests = c.read(base / 'synthetic_checks.json')
    c.require(tests['status'] == 'passed' and tests['metric_source_sha256'] == c.sha(Path(m.__file__)), 'Synthetic tests absent/stale')
    extra = c.read(base / 'extra_checks.json')
    c.require(extra['status'] == 'passed' and extra['metric_source_sha256'] == c.sha(Path(m.__file__)), 'Independent bootstrap tests absent/stale')
    roster = {r['key']: r for r in c.rows(c.SETUP / 'roster.jsonl')}
    rows = [row for name in receipt['reader_files'] for row in c.read(c.ROOT / name)]
    index = {(r['key'], r['condition'], r['repeat']): r for r in rows}
    endpoints = c.read(c.SETUP / 'endpoints.json')
    run_root = base / 'runs' / f'run_{time.time_ns()}'
    run_root.mkdir(parents=True)
    sources = c.bindings([Path(__file__), Path(m.__file__), base / 'reader_complete.json', base / 'synthetic_checks.json', base / 'extra_checks.json', c.SETUP / 'endpoints.json', c.SETUP / 'comparison_registry.json'])
    reader.save(run_root / 'start.json', dict(created_utc=c.a.utc(), sources=sources, conditions=list(c.CONDITIONS), bootstrap_rng='numpy default_rng / PCG64; each endpoint starts seed 20261114; source strata sorted; cluster IDs sorted', chunk=256, rounding='No rounding before contrasts; display only at final export'))
    analyses = {}
    all_arrays = {}
    contrasts = []
    verification = {}
    for endpoint in ('en_hate', 'en_hate_type', 'de_hate', 'de_legal', 'en_target_group', 'en_severity'):
        meta = endpoints[endpoint]
        c.require(meta['all_labels'] == meta['macro_labels'], 'Unexpected label scope')
        arrays = m.endpoint_arrays(meta, roster, index, c.CONDITIONS)
        all_arrays[endpoint] = arrays
        n = int(arrays['common'].sum())
        c.require(n == receipt['coverage'][endpoint]['common_nine'], 'Reader coverage differs')
        point = m.point_metrics(meta, arrays)
        verification[endpoint] = verify_scores(meta, arrays, point)
        source_points = {}
        for source in sorted(meta['sources']):
            mask = arrays['common'] & np.array([roster[k]['source'] == source for k in meta['keys']])
            source_points[source] = m.point_metrics(meta, arrays, mask)
        features, strata, membership = m.cluster_features(meta, arrays, roster)
        print(f"{endpoint}: {n}/{meta['n']} common texts; independent scores passed; starting 50000 draws", flush=True)
        boot = m.bootstrap_from_features(features, strata, len(meta['all_labels']), registry['draws'], registry['seed'], progress=lambda done: print(f'{endpoint}: {done}/50000 paired draws', flush=True))
        for entry in (r for r in registry['rows'] if r['endpoint'] == endpoint):
            metric = entry['metric']
            left = c.CONDITIONS.index(entry['left'])
            right = c.CONDITIONS.index(entry['right'])
            if metric == 'macro_f1':
                point_values = point['macro_f1']
            elif metric == 'micro_f1_all_labels':
                point_values = point['micro_f1']
            else:
                point_values = point['repeatability']['pairwise_agreement']
            difference = boot[metric][:, left] - boot[metric][:, right]
            family = registry['families'].get(entry['family'])
            contrasts.append(dict(**entry, n=n, estimate=float(point_values[left] - point_values[right]), left_value=float(point_values[left]), right_value=float(point_values[right]), interval95=m.interval(difference, 0.95), family_confidence=None if family is None else family['confidence'], family_interval=None if family is None else m.interval(difference, family['confidence'])))
        result = dict(endpoint=endpoint, meta={k: v for k, v in meta.items() if k != 'keys'}, common_nine=n, excluded_keys=[k for k, keep in zip(meta['keys'], arrays['common']) if not keep], point=m.serial(point), sources=m.serial(source_points), bootstrap=dict(draws=registry['draws'], seed=registry['seed'], clusters=len(features), strata=[dict(sources=list(k), clusters=len(v)) for k, v in sorted(membership.items())]), descriptive95={metric: [m.interval(values[:, ci], 0.95) for ci in range(3)] for metric, values in boot.items() if metric != 'micro_f1_all_labels' or meta['dimension'] in ('hate_type', 'target_group', 'legal')})
        folder = run_root / endpoint
        folder.mkdir()
        reader.save(folder / 'results.json', result)
        with (folder / 'encoded.npz').open('xb') as handle:
            np.savez_compressed(handle, **arrays, keys=np.array(meta['keys']))
        with (folder / 'bootstrap.npz').open('xb') as handle:
            np.savez_compressed(handle, **boot)
        analyses[endpoint] = result
    families = Counter((v['family'] for v in contrasts if v['family']))
    c.require(dict(families) == {k: v['size'] for k, v in registry['families'].items()}, 'Comparison family size differs')
    reader.save(run_root / 'contrasts.json', contrasts)
    reader.save(run_root / 'joint_correctness.json', joint_results(endpoints, all_arrays, roster))
    c.verify(sources)
    verification = dict(status='passed', independent_setting_checks=verification, total_setting_checks=sum(verification.values()), tools='sklearn independent macro/micro/Jaccard/per-label P/R/F1, plus scalar set/gate agreement and stable correct/wrong checks', family_counts=dict(families), registered_rows=len(contrasts), source_preservation='All frozen analysis inputs rechecked after calculations')
    reader.save(run_root / 'independent_score_checks.json', verification)
    artifacts = c.bindings(sorted((p for p in run_root.rglob('*') if p.is_file())))
    reader.save(run_root / 'complete.json', dict(status='part3_metrics_and_frozen_intervals_complete', created_utc=c.a.utc(), artifacts=artifacts))
    reader.save(base / 'analysis_complete.json', dict(run=c.ref(run_root), completion_sha256=c.sha(run_root / 'complete.json'), status='metrics_verified_report_pending'))
    print('Part 3 metric calculations and independent checks complete:', run_root, flush=True)
if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description='Recalculate complete-policy classification from saved reader records.').parse_args()
    with c.offline():
        run()
