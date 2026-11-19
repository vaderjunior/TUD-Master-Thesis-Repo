# Exact values for the final SQ3 synthesis

These are transcribed from completed reviews. Different rows have different comparators, cohorts and prespecified interval families. They are not a meta-analysis. F1 changes use the 0–1 scale; error changes below use percentage points (100 × rate difference).

| ID / study | Endpoint and contrast | N | Difference | Interval | Confidence / role |
|---|---|---:|---:|---|---|
| fixed_40 / Fixed full-KB correction pools | seven-type macro F1; mistake-selected − already-correct | 350 | +0.007031 | [-0.011796, +0.025596] | 95%; prespecified primary |
| relevance / Gold-assisted correction relevance | seven-type macro F1; relevant full minus natural | 342 | +0.059167 | [+0.038421, +0.081452] | 98.75%; prespecified primary |
| explanation / Source-specific assistant explanation | seven-type macro F1; specific always minus plain always | 349 | -0.004826 | [-0.017813, +0.007775] | 98.75%; prespecified primary |
| selectivity / Fixed selective application | seven-type macro F1; plain selective minus plain always | 350 | -0.006201 | [-0.026904, +0.014160] | 98.75%; prespecified primary |
| taxonomy / Familiar-category restoration | mean restored-target F1; full package − schema-only | 466 | +0.173031 | [+0.130841, +0.214359] | 98.75%; prespecified primary |
| factorial_example / Examples × definitions supplement | mean restored-target F1; example block main effect | 193 | +0.136877 | [+0.050732, +0.236596] | 97.5%; prespecified primary |
| factorial_definition / Examples × definitions supplement | mean restored-target F1; definition block main effect | 193 | +0.055186 | [-0.001342, +0.108824] | 97.5%; prespecified primary |
| dataset_meaning / Existing dataset human implied meanings | seven-type macro F1; human meaning − plain correction | 350 | +0.000131 | [-0.010668, +0.011311] | 95%; prespecified primary |
| reserved_FPR / Reserved correction, exposed DEV | FPR; 4 ordinary + correction − 5 ordinary | 204 | +3.540 pp | [+0.436, +6.863] pp | 97.5%; prespecified primary |
| reserved_FNR / Reserved correction, exposed DEV | FNR; 4 ordinary + correction − 5 ordinary | 467 | -1.665 pp | [-3.341, -0.024] pp | 97.5%; prespecified primary |
| broad_en_test_FPR / Broader fixed-policy transfer | en_test_FPR; 4 ordinary + correction − 5 ordinary | 8291 | +3.181 pp | [+2.634, +3.728] pp | 98.75%; prespecified primary |
| static_en_test_FPR / Same-feedback static classifier | en_test_FPR; (linear update − linear base) − (Qwen feedback − Qwen reference) | 8291 | -3.230 pp | [-3.760, -2.694] pp | 98.75%; prespecified primary |
| broad_en_test_FNR / Broader fixed-policy transfer | en_test_FNR; 4 ordinary + correction − 5 ordinary | 3593 | -1.358 pp | [-2.000, -0.718] pp | 98.75%; prespecified primary |
| static_en_test_FNR / Same-feedback static classifier | en_test_FNR; (linear update − linear base) − (Qwen feedback − Qwen reference) | 3593 | +1.228 pp | [+0.557, +1.913] pp | 98.75%; prespecified primary |
| broad_en_hatecheck_eval_FPR / Broader fixed-policy transfer | en_hatecheck_eval_FPR; 4 ordinary + correction − 5 ordinary | 1164 | +1.394 pp | [+0.301, +2.624] pp | 98.75%; prespecified primary |
| static_en_hatecheck_eval_FPR / Same-feedback static classifier | en_hatecheck_eval_FPR; (linear update − linear base) − (Qwen feedback − Qwen reference) | 1164 | -1.308 pp | [-2.517, -0.230] pp | 98.75%; prespecified primary |
| broad_en_hatecheck_eval_FNR / Broader fixed-policy transfer | en_hatecheck_eval_FNR; 4 ordinary + correction − 5 ordinary | 2560 | -0.039 pp | [-0.332, +0.258] pp | 98.75%; prespecified primary |
| static_en_hatecheck_eval_FNR / Same-feedback static classifier | en_hatecheck_eval_FNR; (linear update − linear base) − (Qwen feedback − Qwen reference) | 2560 | +0.000 pp | [-0.306, +0.317] pp | 98.75%; prespecified primary |
| primary_en_test_FNR_adaptive-random / Four successive rounds | FNR; adaptive − random | 317 | -1.122 pp | [-2.642, +0.243] pp | 98.75%; prespecified primary |
| primary_en_test_FNR_adaptive-static / Four successive rounds | FNR; adaptive − static | 317 | -0.280 pp | [-1.477, +0.879] pp | 98.75%; prespecified primary |
| primary_en_test_FPR_adaptive-random / Four successive rounds | FPR; adaptive − random | 688 | +0.291 pp | [-0.791, +1.365] pp | 98.75%; prespecified primary |
| primary_en_test_FPR_adaptive-static / Four successive rounds | FPR; adaptive − static | 688 | -0.032 pp | [-0.888, +0.840] pp | 98.75%; prespecified primary |
| cumulative_en_hatecheck_eval_FPR / Same-feedback cumulative classifier, round 4 | en_hatecheck_eval_FPR; (linear update − linear base) − (Qwen feedback − Qwen reference) | 308 | -1.479 pp | [-3.289, +0.206] pp | 95%; supporting, nominal 95%; no new primary family |
| cumulative_en_hatecheck_eval_FNR / Same-feedback cumulative classifier, round 4 | en_hatecheck_eval_FNR; (linear update − linear base) − (Qwen feedback − Qwen reference) | 736 | -0.060 pp | [-0.543, +0.369] pp | 95%; supporting, nominal 95%; no new primary family |
| cumulative_en_test_FPR / Same-feedback cumulative classifier, round 4 | en_test_FPR; (linear update − linear base) − (Qwen feedback − Qwen reference) | 688 | -1.760 pp | [-3.165, -0.420] pp | 95%; supporting, nominal 95%; no new primary family |
| cumulative_en_test_FNR / Same-feedback cumulative classifier, round 4 | en_test_FNR; (linear update − linear base) − (Qwen feedback − Qwen reference) | 317 | -0.701 pp | [-2.637, +1.153] pp | 95%; supporting, nominal 95%; no new primary family |
