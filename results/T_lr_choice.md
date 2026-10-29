| Model | Learning rate | Validation AUC (fold mean ± SD) | Chosen |
|---|---|---|---|
| MLP | 1e-3 | 0.846 ± 0.020 | yes |
| MLP | 1e-4 | 0.841 ± 0.020 |  |
| BrainNetCNN | 1e-3 | 0.859 ± 0.014 | yes |
| BrainNetCNN | 1e-4 | 0.855 ± 0.018 |  |
| BNT | 1e-3 | 0.846 ± 0.014 | yes |
| BNT | 1e-4 | 0.837 ± 0.014 |  |

Pooled 5-fold CV, five seeds per fold; the rate with the higher mean validation AUC was used for all evaluations (test AUC not used).
