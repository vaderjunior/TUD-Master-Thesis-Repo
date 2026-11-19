# Data inputs and annotation rules

Raw corpora are not included in this checkout. Keep their source releases, versions and access terms with the empirical submission. Redistribution depends on each source's terms.

| Source | Expected local input | Mapped fields |
|---|---|---|
| HateXplain | `data/raw/hatexplain/dataset.json` and `post_id_divisions.json` | Hate and targets |
| Measuring Hate Speech | `data/raw/mhs/mhs.parquet` or `mhs.csv`, with original annotator rows | Hate, targets, severity |
| Implicit Hate Corpus | `data/raw/implicit_hate/implicit_hate_v1_stg1_posts.tsv` and `implicit_hate_v1_stg2_posts.tsv` | Hate and available types |
| GAHD | `data/raw/gahd/gahd.csv` | Hate |
| DeTox | `data/raw/detox/DeTox-Dataset_complete.sqlite3` | Hate, targets; auxiliary paragraph flags |
| BoTox | `data/raw/botox/botox.csv` | Grouped legal annotations |
| English HateCheck | `data/raw/hatecheck/hatecheck.csv` or an explicit input export | Hate and diagnostic targets |

`hsrag prepare splits --raw-dir PATH` accepts another source location. MHS accepts `--mhs-file PATH` and an explicit revision for requested downloads. Recover the original export for historical reproduction where acquisition revisions were not recorded.

## Codebook

The vocabulary and definitions are in `config/taxonomy.yaml`. Source mappings are in `data/mappings/*.csv`.

- A missing reference is `None`/`null`. An observed empty label set is `[]`. Missing annotations do not supply negative reference labels.
- HateXplain uses majority class votes. Offensive maps to non-hate; three-way ties are excluded. Hateful targets use the mapped union of annotator targets.
- MHS is aggregated to one row per comment. Hate uses a mean score above 0.5; targets use any positive annotator flag. Hateful scores are grouped at 1.00 and 1.63. [Recorded thresholds](../data/mappings/mhs_thresholds.md) describe the bands.
- Implicit Hate Corpus types are joined by source text. Secondary types and unmatched stage-two texts are retained. Unmapped types supply no type reference.
- GAHD maps label 1 to hate. DeTox uses scores above 0.5 for hate and target flags; its paragraph flags are not BoTox legal gold.
- BoTox maps positive classes to propaganda, incitement/public peace and insult/defamation. It supplies no hate reference. An observed legal negative is an empty list.
- A negative model hate decision clears predicted targets, types and severity. Legal predictions stay independent. Missing model lists default to empty predictions; missing reference annotations remain unknown.

General partitions use seed 42 and a two-step hate-stratified split with nominal 70/15/15 proportions. Each step rounds; a 100-row fixture produces 69/16/15. BoTox uses separate strata for legal positives/negatives, floor allocation and a TEST remainder. Frozen membership files identify the historical populations.

Exact duplicate texts keep the first source record. The prototype knowledge builder removes TRAIN example matches with available DEV/TEST parents. The thesis banks also have their own cleaning and consolidation policies.

## Commands

~~~powershell
hsrag prepare splits --sources gahd --raw-dir PATH --output-dir PATH
hsrag prepare hatecheck --input PATH --output data/processed/en_hatecheck_eval.parquet
hsrag prepare knowledge --guidance-only --output kb/guidance.jsonl
hsrag prepare encoder --arm full --dry-run
~~~

Prepare general partitions before legal partitions so BoTox filtering can use held-out texts. `--allow-missing-heldout` explicitly permits standalone legal preparation with absent general parents; it changes the available overlap checks.

Exact human-review instruments, original judgments, aliases and consent/access records belong to the empirical package. Provide permitted blank forms and guides to reviewers.

Source references: HateXplain (Mathew et al., 2021); Measuring Hate Speech (Sachdeva et al., 2022); Implicit Hate Corpus (ElSherief et al., 2021); GAHD (Goldzycher et al., 2024, CC BY 4.0); DeTox (Demus et al., 2022); BoTox (Kums et al., 2025); HateCheck (Röttger et al., 2021).

## Included evaluation records

The English guideline matrix is in [data/evaluations/english_guidelines.json](../data/evaluations/english_guidelines.json). It preserves the original reference gates, IDs, template groups, functionality tags, text hashes and all 10,395 gate predictions, including one missing result. These are the gold-selected, known-benign diagnostic cases from the thesis, not a general hate-speech evaluation set. [Knowledge-selection notes](knowledge.md) describe the replay and other required archives.
