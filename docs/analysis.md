# Supplementary analyses

The main study guides are [Knowledge selection](knowledge.md), [Classification](classification.md) and [Feedback](feedback.md). This page covers the shared supplementary checks and exports.

| Work | Source | Required inputs |
|---|---|---|
| Micro-F1 comparison registry | analysis/micro_f1/setup | The original comparison registry, endpoint metadata and paired-contrast manifest |
| Retrieval and classifier micro-F1 | analysis/classification/micro_f1 | Saved predictions, gold masks, model/pool/seed identities and original macro draws |
| SQ1 micro-F1 | analysis/knowledge/micro_f1 | Saved bank/strategy/component/guideline answers and original confidence families |
| SQ3 micro-F1 | analysis/feedback/micro_f1.py | Saved correction assignments, predictions, validity masks and original cohort IDs |
| Combined metric diagnostics | analysis/micro_f1/combined | Verified outputs from the separate analyses and their original label scopes |
| Data provenance and exposure | analysis/data | Prepared partitions, source memberships, bank exclusions and recorded earlier prediction IDs |
| Recorded resources | analysis/resources | Original request/training ledgers and metadata inventories |
| Figures | analysis/figures | The matching unrounded numerical exports |

Install the analysis and test dependencies before using these modules:

~~~powershell
python -m pip install -e '.[analysis,test]'
python -m analysis.classification.micro_f1.retrieval --help
python -m analysis.knowledge.policy.synthesis --help
~~~

Most modules retain their original mode arguments or expose callable functions. Use the matching study records; a plan alone does not supply predictions, gold labels or bootstrap draws. Imports and help for the registry readers do not require the missing archives.

## Metric scope

The code keeps original macro-F1, micro-F1 over all scoreable labels, and micro-F1 restricted to the original macro label set separate. It retains missing versus observed-empty annotations, gate rules, own-valid/shared-valid masks, repeat/pool/seed averaging, cluster units and random streams.

The supplementary registry contains 77 prelisted questions in 17 separate families. Original macro families keep their own confidence levels. Same-confidence and label-scope diagnostics explain changes in support; they do not choose models or cutoffs from micro-F1. Comparisons from different studies are not pooled into one F1 value or one significance family. Repeated evaluation IDs document text reuse, not independent replication.

## Resource and exposure limits

Resource reconciliation uses recorded attempts, repairs, retries, response usage and fit receipts. Selected programme totals remain partial. Concurrent request-duration sums differ from elapsed wall time. Successful-response token totals omit unknown failed or interrupted remote work. Reused fits, offline QA and human/curation time retain their original accounting boundaries.

Data provenance reports preserve the minimum documented earlier DEV/TEST exposure. They do not certify that unmatched IDs or unavailable earlier records were untouched. Frozen source memberships and cleaning exclusions remain necessary for replication.

The original study IDs and source bindings are kept in [Source changes](source-changes.json). Current source names and hashes identify the cleaned code. Historical hash checks require the original bound versions.

The large copied archives have not been inspected. Full empirical replay remains unverified.
