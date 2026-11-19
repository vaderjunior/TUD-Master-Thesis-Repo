# Knowledge selection

The SQ1 code is in analysis/knowledge. Shared scoring, retrieval and saved-output readers are in analysis/common. The tests use synthetic inputs and keep archive-dependent checks separate.

## Replay the included results

The repository includes the English guideline study's prediction matrix. It has 693 known-benign HateCheck cases, five conditions and three repeats. The matrix contains reference gates, case and template IDs, functionality tags, text hashes and predicted gates. It contains no comment text.

~~~powershell
python -m pip install -e '.[analysis]'
python -m analysis.knowledge.guidelines.replay --output results/guideline_replay.json
~~~

The command uses one shared valid mask across all 15 settings. It preserves the single missing prediction, leaving 692 texts and 302 templates. It resamples complete templates, weights texts, keeps the three repeats fixed and uses 10,000 draws with seed 20260925. The four primary comparisons use 98.75% intervals; group results use descriptive 95% intervals. Existing outputs are preserved.

The saved result is in [English guideline results](../results/knowledge/english_guidelines.json). The [prediction matrix](../data/evaluations/english_guidelines.json) is unchanged from the original export. The replay recovers the thesis's overall false-positive rates and the mapped-minus-secondary interval. It does not reconstruct prompt delivery, latency or token measurements; those need the original records.

## Study code and inputs

| Study | Code in analysis/knowledge | Required records |
|---|---|---|
| Full and compact banks | banks/primary.py, banks/secondary.py, banks/severity.py | Gold rosters, fixed labels, repeated outputs and full/compact memberships |
| Curated and sampled banks | sampled_banks/results.py | All 33 bank/repeat settings, nested memberships and original first-pass prompts |
| Dense, BM25 and hybrid retrieval | strategies/results.py | Nine strategy/repeat settings, candidate ranks, delivered records and outputs |
| Examples, definitions and guidelines | components/results.py | Eight factorial arms and the separate fresh prompt reference |
| Definition allocation | allocation/results.py | Three policies, repeated outputs and delivered definition IDs |
| English guideline selection | guidelines/replay.py, guidelines/results.py | Included matrix for gate scores; complete prompt/output archives for the wider audit |
| Complete policy | policy/classification.py, policy/labels.py, policy/robustness.py, policy/resources.py | Three-policy stores, rosters, clusters, repeats and resource records |
| German guideline allocation | german_guidelines | Saved conditions, predictions, source clusters and request records |
| Supplementary micro-F1 | micro_f1 | Original count arrays, masks, comparison registries and resampling designs |

The full/compact core retains its paired item bootstrap and original four-comparison family. The sampled-bank scores use all 33 settings and equal weighting of the three random banks. Strategy scores use all nine settings. The component study keeps its 24-setting factorial mask separate from the six-setting bridge. Definition allocation keeps its nine-setting mask. Later policy and German analyses retain their declared cluster units and confidence families.

Observed empty labels and missing annotations remain distinct. Missing model outputs remain missing. Legal gold is independent of the hate gate. Fine-head scoring keeps its original gold-gate rules and fixed label support.

The bank folder also contains English/German cleaning and audit code, the original bank helpers and preparation of the core design. Source-specific policies, exclusions and final memberships are needed to reproduce the full and compact banks. Current prototype knowledge preparation does not identify every historical membership.

## Historical records

Frozen plans remain under their original study IDs in experiments. [Source changes](source-changes.json) maps the old source names to the cleaned files and records both hashes. The original bound sources are preserved outside the public checkout for the examiner archive.

Historical generation and full audit commands require their original preparation records, environment, tokenizer assets, run layout and source bindings. Their saved dates and execution guards are retained. They describe the completed studies; the current classifier commands in the README are the starting point for new use.

The ZIP still lacks most gold/prediction stores, corpus exports, bank memberships and resource records. Their absence limits empirical replay. The included matrix supports the English guideline gate analysis; it does not establish reproduction of the other studies.
