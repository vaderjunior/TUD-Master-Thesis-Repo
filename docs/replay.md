# Result replay

The standard-library replayer is scripts/replay_results.py. It checks a selected result package's file sizes and hashes, then recomputes per-run macro-F1, mean scores, validity coverage and logical missing-output bounds.

~~~powershell
python -m scripts.replay_results --help
python -m scripts.replay_results --root path/to/result_package --output results/replay_check.json
~~~

The package must contain manifest.json, cases.json, the named gold records and every named prediction file. The case records declare the dimension, fixed labels, roster size, arms and expected per-run scores. Duplicate IDs, changed files, invalid/missing-state inconsistencies and mismatched scores are rejected. Existing output files are preserved.

This command covers point-score replay. The study-specific analyses regenerate or inspect their own paired intervals, resampling populations and confidence families.

## Build the original selected package

scripts/package_results.py retains the original assembly of selected core and feedback results, including 33 expected individual point scores. It requires the matching prepared gold, completed predictions, original completion receipts and numerical summaries. Its output is a selected result package, not a complete thesis archive. It also checks a separate restored copy and rejects a changed file.

The required inputs were absent from the downloaded ZIP. The builder has not been run during cleanup. Restore its complete matching source records before executing it.

## Inspect retrieval

For a single explicit query, scripts/inspect_retrieval.py displays the retrieved definitions, guidelines and examples, including record IDs, scores and channel positions. It uses the configured retrieval settings and accepts explicit record/index paths.

~~~powershell
python -m scripts.inspect_retrieval --help
~~~

A query requires the retrieval dependencies, local records/index and embedding model. Help loads none of those assets. This utility does not regenerate a historical prediction store.

The [reproduction notes](reproduction.md) list the broader archive requirements. [Source changes](source-changes.json) retains the original identities and cleaned source mappings.
