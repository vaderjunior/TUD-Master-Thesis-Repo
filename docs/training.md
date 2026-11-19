# Training

The training code is in training. The methods remain separate because the thesis compared different representations, labelled-data budgets and fitting recipes.

| Folder | Method |
|---|---|
| encoder.py | Editable one-head RoBERTa/GBERT trainer for the prepared full, compact and German partitions |
| original | Aligned historical encoder fits and their corrected-tokenizer audit |
| frozen_features | BGE and frozen RoBERTa features, compact-bank controls and TRAIN-only linear fitting |
| cpu | Lexical and frozen-feature classifiers with the original selection and memory controls |
| roberta | Compact RoBERTa search, refits, selected cutoffs and saved-output analysis |
| roberta/short | The distinct shorter training schedules and epoch-logit checks |
| matched_annotations | Equal type-annotation pools |
| text_budgets | Whole-text budgets and the extended training study |
| full_data | One canonical implementation of the final server training, selection, inference and archive kernels |
| type_update | The five final hate-type refits and their independent checks |

## Train one head

Prepare the head-specific data first. Each arm directory contains head_train.parquet, head_val.parquet and head_meta.json. The preparation command is:

~~~powershell
hsrag prepare encoder --arm full
~~~

Install the training dependencies, then run the trainer with explicit paths:

~~~powershell
python -m pip install -e '.[training]'
python -m training.encoder --arm full --head hate_type --seed 0 --data-dir data/encoder --output-dir models/encoder
~~~

Use --arm kb for the compact partition and --arm de for the German partition. --model selects a model name or original local checkpoint. --local-files-only uses the supplied local assets. --taxonomy selects an explicit vocabulary file. The default paths resolve from the checkout, so launching from another directory does not change them.

Use the saved heads to classify the prepared evaluation files:

~~~powershell
python -m training.predict --arm full --seed 0 --models-dir models/encoder --processed-dir data/processed --output-dir results/encoder_predictions
~~~

The predictor uses local checkpoint files. It retains the original head borrowing, label order, saved probabilities and gate clearing; legal predictions remain independent of the gate.

--dry-run performs two training steps and skips the final model export. It still loads a model and uses training compute. It is different from a command/help or synthetic test check.

## Preserved methods

The one-head trainer keeps taxonomy label order and the missing/empty annotation rules from preparation. Its default multilabel cutoff is 0.5. The compact arm has fixed epochs; the larger arms use TRAIN-internal validation for checkpoint selection. Its validation metric is separate from the thesis evaluation scorer.

The later controls preserve their own class weights, selected cutoffs, folds, seeds, optimizer schedules and precision settings. Selection uses the declared TRAIN-only records. The shorter trainer shares loading, tokenization, loss and inference helpers with the compact trainer while keeping its own update loop and epoch records.

The corrected English tokenizer and the earlier German outputs remain distinguishable. The final type update retains its separate model identities and original input/prediction preservation checks.

The final two-step AMP validation is in recovery/update_rules.py and recovery/state.py. It distinguishes a fully proven initialization-only outcome from a trained fit. The canonical trainer and reader use that existing amendment directly. Its original recovery_policy.json is still absent and remains a checksum-checked prerequisite for historical execution.

## Historical environment and inputs

[Recorded dependency pins](../deploy/environment/requirements.in) and the adjacent requirements.lock retain the exact original versions. [Resolver and source identities](../deploy/environment/provenance.json) record the Windows 3.12 environment in which the dependency report was produced. The frozen server plan retains its own Linux/CUDA requirements; cleanup uses a separate offline verification environment.

The final server code reads its frozen package inputs from deploy/full_training by default. HSRAG_TRAINING_PACKAGE selects a complete original package elsewhere. HSRAG_TYPE_UPDATE_PACKAGE selects the separate original type-update package. The historical runners still check their original mounted paths, service limits, stop dates, package inventories and source hashes. Restoring only data/configs.json is insufficient for a historical training run. Historical source checks in the tests can read an unmodified original package through HSRAG_ORIGINAL_TRAINING_PACKAGE.

The final archive reader keeps its original E:/SQ2_Backup/sq2_full_20261029_v1 location and volume checks. Portable reading of the incoming copies will be addressed after their manifests are verified.

No model fitting, inference or live service call is performed as part of repository cleanup. The code checks use synthetic inputs and mocks. Full training or result reproduction remains dependent on the separate archives and appropriate model/environment access.
