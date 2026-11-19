# Original training inputs

Editable training code is in ../training. The [training guide](../docs/training.md) describes the distinct methods and commands.

full_training contains the original final configuration grid, plan, package manifest and an example storage record. history retains the distinct earlier manifests; identical grids and plans were consolidated. type_update contains the separate update protocol, manifest and recorded storage inputs. These JSON files retain their original bytes.

environment contains the original requirements.in and requirements.lock. provenance.json records the resolver environment and original helper identities. The full original dependency/AST dumps remain with the cleanup review outside this checkout.

HSRAG_TRAINING_PACKAGE and HSRAG_TYPE_UPDATE_PACKAGE select complete matching packages elsewhere. Historical source verification can use HSRAG_ORIGINAL_TRAINING_PACKAGE. The included references do not supply the original training memberships, checkpoints or complete bound source trees.
