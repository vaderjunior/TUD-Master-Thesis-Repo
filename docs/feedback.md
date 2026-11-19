# Feedback studies

The feedback code is in analysis/feedback. Most studies use dataset labels as simulated corrections while keeping the generator weights fixed. The human walkthrough and classifier refits are separate comparisons.

| Thesis section | Folder | Main saved inputs |
|---|---|---|
| 4.3.1 Familiar categories | restoration | Withdrawn/restored definitions and examples, category rosters, original schemas and predictions |
| 4.3.2 Correction uptake | source | Screening answers, fixed source pools, correction assignments, source and transfer predictions |
| 4.3.2 Gold-assisted relevance | relevance | Coverage/disjoint-label rules and the saved natural and matched corrections |
| 4.3.3 Explanations and selective delivery | explanations, initial, supplements | Source explanations, fallback masks, TRAIN-calibrated cutoffs and exact messages |
| 4.3.4 Balanced corrections | balanced | Hateful-only and balanced binary pools, eligibility and delivered records |
| 4.3.5 Ordinary insertion | insertion | Copied bank identities, candidate/fusion traces, prompts and replay checks |
| 4.3.6 Reserved slot | reserved | The 675-text DEV roster, five/four/six-example references, label-hidden controls and saved votes |
| 4.3.7 Broader transfer | transfer | English TEST/HateCheck rosters, gold fields, prompts, correction identities and response records |
| 4.3.8 Successive rounds | rounds | Three source-order streams, all four rounds, source screening, selections and fixed trajectory cohorts |
| 4.3.9 Novice review | human | The 24 selected texts, both guides, assignments, original form fields and responses |
| 4.3.10 Classifier feedback | classifiers, training/feedback | The same correction labels, TRAIN memberships, feature identities, frozen cutoffs and saved fits/predictions |

The earlier prototype checks in relevance/prototype_* use different rosters and type-coverage targets. They remain separate from the later thesis comparisons. Supplementary micro-F1 is in micro_f1.py and micro_f1_inputs.py.

## Keep the evaluation targets separate

Source uptake evaluates the corrected source text. Transfer evaluates a different text; the source correction does not supply that query's reference label. Gold-assisted relevance uses evaluation information and is a diagnostic control.

The early "correct-selected" condition uses type coverage across the original three screening answers. It does not require exact type-set correctness. Later binary correction records supply hate alone; other fields remain unknown.

Ordinary insertion lets corrections compete for the usual five examples. Added delivery retains five ordinary examples and appends a correction. Reserved delivery keeps the first four and replaces the fifth. The label-hidden control retains the correction text. These contexts have different purposes and remain distinct.

Four-round selection keeps the current-error, initial-error and balanced-random policies, three streams and five sources per class per round. The stored pool grows through 10, 20, 30 and 40 records, while each query receives one correction. Each round has a fresh unchanged reference. Fixed all-round cohorts differ from final-round cohorts.

The analyses retain missing versus observed-empty annotations, own-valid/shared-valid masks, strict-vote sensitivities, cluster units, label support, pool/repeat averaging, seeds and interval families. Complete-known-field agreement does not fill unannotated fields. Case examples remain descriptive and keep their selection rules.

The two novice reviewers assessed the same 24 deliberately selected texts under both guides in opposite orders. Their four returns are repeated observations from two people. Human and dataset targets remain separate, with no adjudication or benchmark relabelling. Original form-field UUID namespaces are kept because saved responses refer to them. Editable forms and offline response decoding are in human/forms and human/response_audit.py.

Classifier updates use the same supplied correction labels and the original feature, class-weight and cutoff recipes. Static and cumulative updates remain distinguishable. Model fitting is separate from reading the saved classifier results.

## Commands and required records

Run the analyses from the checkout. Their preparation and validation also use the independent checks in tests/feedback.

~~~powershell
python -m pip install -e '.[analysis,test]'
python -m pytest -q tests/feedback
python -m analysis.feedback.source.run --help
python -m analysis.feedback.reserved.run --help
python -m analysis.feedback.rounds.effects --help
~~~

Most preparation, scoring and validation modules expose their original functions or mode arguments. Restore the complete matching study records before using a scoring mode. A plan file alone does not supply the rosters, predictions or source bindings. Retrieval/index preparation also needs the retrieval dependencies and original assets.

The editable prototype utilities remain available:

~~~powershell
python -m scripts.make_sq3_subset --help
python -m scripts.make_sq3_sentinel --help
python -m scripts.apply_feedback --help
~~~

Use the feedback and control arms' own previous results after round one. A shared previous-results stem is only appropriate when the arms share that state. The dry-run options still read the required local inputs; they do not reproduce missing experiments.

Original study IDs, saved field names, form UUIDs and frozen JSON records are unchanged. [Source changes](source-changes.json) records the renamed code and its new hashes. Original hash checks still require the preserved bound source versions. The old execution dates, worker limits, transport amendments and integrity checks remain part of their historical contracts. Recovery source matching tolerates formatting while retaining exact replacement counts and the existing amendment rules.

The downloaded ZIP lacked most response stores, source/query memberships, original tokenizer/index assets and human returns. The large copied archives have not been inspected in this part. Full empirical replay remains unverified; the code checks use synthetic records and disposable outputs.
