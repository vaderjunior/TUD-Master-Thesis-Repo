# Submission and data availability

This repository is the editable programming-code part of my thesis. The code archive extracts into the same runnable folder. It includes the classifier, preparation and analysis code, tests, label definitions/mappings, prompt template, frozen study plans, original environment references and selected numerical exports.

## Run the code archive

Use Python 3.11 or newer. From the extracted folder:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m src.hsrag classify --provider mock --lang en --text "A short comment."
~~~

On Linux/macOS, activate with `source .venv/bin/activate`. The mock is a fixed synthetic program check. Model access, retrieval, preparation and scoring are described in the [README](../README.md).

For the maintained tests:

~~~powershell
python -m pip install -e '.[analysis,test]'
python -m pytest -q
~~~

Checks requiring original data or optional model dependencies skip when those inputs are absent. The original experiment runners retain their recorded integrity checks and execution contracts.

## Empirical records

| Material | In this repository | Separate empirical package |
|---|---|---|
| Codebook and source mappings | config, prompts and data/mappings | Same exact label/mapping versions |
| Experiment settings | Frozen plans/manifests and study-to-code guides | Matching original protocols and amendments |
| Gold, predictions and evaluation masks | Selected English guideline matrix and aggregate exports | Complete rosters, reference annotations, per-run responses, validity masks and source/pool/repeat IDs |
| Statistical analyses | Editable analysis code and selected final tables | Matching unrounded results, paired populations, cluster identities and saved draws where required |
| Corpus preparation and knowledge banks | Preparation code and access instructions | Original releases, frozen memberships, exclusions and permitted prepared banks |
| Training/model records | Recipes, original settings and dependency pins | Matching checkpoints/tokenizers, fit receipts, selected cutoffs and prediction archives |
| Human review | Editable instrument/response-processing code | Exact original guides/forms, assignments, genuine returns and participation/access records |
| Historical source verification | Original/current path and hash map | The unmodified bound source versions identified by those original hashes |
| Resource and exposure records | Reconciliation code and recorded boundaries | Original request/usage/fit ledgers and earlier exposure memberships |

The large copied archives are being assembled separately and have not yet been reviewed. This code repository is ready for source review; complete empirical replication still depends on that package. Corpus and participant material must follow the original access and release arrangements.

[Data inputs](data.md), [reproduction notes](reproduction.md), [result replay](replay.md) and the study guides describe the required records. The selected standard-library replayer covers point scores, coverage and logical missing-output bounds; it does not supply a complete empirical archive.

The original study IDs, form namespaces and historical source hashes remain in the records. Current code hashes identify this cleaned source version. The repository keeps those original identities so saved records can be matched.
