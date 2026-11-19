# Classification comparisons

The SQ2 code is in analysis/classification. It compares saved answers from full-bank retrieval, compact-bank retrieval and prompting without retrieval, then relates those answers to the supervised classifiers.

| Thesis section | Code | Saved inputs needed |
|---|---|---|
| 4.2.1 Retrieval versus no retrieval | broad/reader.py, broad/metrics.py | Gold rosters, endpoint contracts, original clusters and all condition/repeat predictions |
| 4.2.2 Example information | matched | The thirteen conditions, exact example assignments, saved prompts and outputs |
| 4.2.2 Neighbour-label vote | neighbour_vote.py | The five original neighbours, observed annotations and paired output masks |
| 4.2.3 Supervised comparisons | supervised, combined | Completed classifier archives, probabilities, decisions, fit identities and training memberships |
| 4.2.4 Complete answers | broad/quality.py, supervised/quality.py, label_scores.py | Gate states, known reference fields and the original common-valid populations |
| 4.2.5 Context and explanations | broad/contexts.py, combined/quality.py | Delivered records, saved messages and response provenance |
| 4.2.6 Repeated answers | repeatability.py | Three outer aggregate predictions, inner-slot validity and original clusters |

Shared scoring and saved-output readers are in analysis/common. Training methods are described in [Training](training.md).

## Scoring rules

Keep missing annotations distinct from observed empty sets. Missing answers remain missing. Legal annotations are independent of the hate gate. Type and target exactness include the gate rules declared for each endpoint.

The analyses retain their original label support, valid-output masks, clusters, seeds, resampling draws and confidence families. Own-valid scores, shared-valid comparisons and missing-output bounds have different denominators. Model fits and the three decoder repeats retain their original averaging rules.

Repeatability measures agreement between the three saved aggregate answers. It is separate from correctness and the inner-response voting comparison. Its post-hoc agreement intervals use 50,000 whole-cluster draws with seed 20261113 and the separate eleven-endpoint full/no-retrieval family.

The neighbour baseline uses only observed neighbour annotations. Its declared YAML label-order correction is integrated into the retained reader; missing annotations can still produce abstentions.

## Saved results and replay

[Final SQ2 tables](../experiments/astra/sq2_combined_final_20261104_v1/publication_tables.json) and [figure values](../experiments/astra/sq2_combined_final_20261104_v1/figure_values.json) are retained with their original bytes and study IDs.

Most gold rosters, saved response stores, fit archives and prepared corpora were absent from the downloaded ZIP. The copied external archives will be checked separately. This part verifies code and synthetic boundary cases; it does not establish full empirical replay. The original recovery policy and most fit, tokenizer and response records are still required.

Historical preparation and execution modules retain their study-specific source checks, original environments and dates. They need the completed original archive layout. Shorter comments and moved source files have new hashes; [Source changes](source-changes.json) records that relationship. Use preserved original sources when verifying historical source bindings.

## Commands

The maintained program commands are in the [README](../README.md). Individual analysis modules expose their original preparation, scoring or validation functions. Run only the stage matching the recovered archive; a plan file alone is insufficient to recalculate the study.

Run the offline regression checks from the checkout:

~~~powershell
python -m pip install -e '.[test,analysis]'
python -m pytest -q
~~~

The tests use synthetic records and disposable outputs. Checks needing real corpus, tokenizer or archive files state their prerequisites.
