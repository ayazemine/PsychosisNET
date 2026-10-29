| Stage | Run | Finished | Change | Label | n test | Same folds as final | Seeds | Pretrained | Val AUC | Test AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 | v2_depth4 | 2026-10-12 | Transformer on fixed tangent-space input, depth 4 | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.775 ± 0.059 | 0.738 ± 0.058 |
| v2 | v2_depth4_skip | 2026-10-12 | + wide linear path | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.838 ± 0.043 | 0.819 ± 0.045 |
| v2 | v2_depth4_drop05 | 2026-10-12 | stronger dropout (0.5 / path 0.3 / edge 0.4), no wide path | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.781 ± 0.060 | 0.730 ± 0.055 |
| v2 | v2_depth6_drop05_skip | 2026-10-12 | depth 6, stronger dropout, wide path | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.832 ± 0.042 | 0.790 ± 0.055 |
| v3 | v3_diff_skip | 2026-10-12 | differential attention, wide path | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.842 ± 0.050 | 0.824 ± 0.049 |
| v3 | v3_spd_skip | 2026-10-12 | learnable SPD tangent layer, wide path | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.849 ± 0.050 | 0.835 ± 0.039 |
| v3 | v3_spd_diff_skip | 2026-10-12 | SPD layer + differential attention, wide path | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.852 ± 0.050 | 0.827 ± 0.047 |
| v3 | v3_spd_diff | 2026-10-12 | SPD layer + differential attention, no wide path | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.806 ± 0.062 | 0.761 ± 0.048 |
| v4 | v4_rank20 | 2026-10-12 | low-rank (20) wide path, fixed tangent input | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.802 ± 0.054 | 0.744 ± 0.065 |
| v4 | v4_rank10_drop05 | 2026-10-12 | low-rank (10) wide path, stronger dropout | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.812 ± 0.042 | 0.770 ± 0.064 |
| v4 | v4_rank10 | 2026-10-12 | low-rank (10) wide path | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.795 ± 0.056 | 0.748 ± 0.047 |
| v4 | v4_spd_rank10 | 2026-10-12 | SPD layer + low-rank (10) wide path | psychosis (HC/SZ/BP) | 308 | no | 1 | no | 0.813 ± 0.055 | 0.748 ± 0.017 |
| v3 | v3_spd_skip_szhc | 2026-10-12 | SPD + wide path; SZ vs HC, COBRE and UCLA CNP only | SZ vs HC | 267 | no | 1 | no | 0.862 ± 0.041 | 0.804 ± 0.057 |
| v5 | v5_spd_skip_szhc_3site | 2026-10-12 | as above + SZ-Smoking (3 datasets) | SZ vs HC | 348 | no | 1 | no | 0.846 ± 0.024 | 0.810 ± 0.022 |
| v6 | v6_spd_skip_szhc_7site | 2026-10-12 | as above + SRPBS sites (SZ vs HC) | SZ vs HC | 924 | no | 1 | no | 0.864 ± 0.024 | 0.850 ± 0.029 |
| v7 | v7_spd_skip_psychosis_8site | 2026-10-13 | psychosis label, HC/SZ/BP, all classification sites | psychosis (HC/SZ/BP) | 965 | yes | 1 | no | 0.863 ± 0.021 | 0.850 ± 0.016 |
| v8 | v8_base | 2026-10-14 | 60 epochs, patience 15, one seed | psychosis (HC/SZ/BP) | 965 | yes | 1 | no | 0.855 ± 0.017 | 0.849 ± 0.017 |
| v8 | v8_delay | 2026-10-14 | + wide path frozen 10 epochs, learning-rate scale 0.1 | psychosis (HC/SZ/BP) | 965 | yes | 1 | no | 0.853 ± 0.014 | 0.849 ± 0.025 |
| v8 | v8_siteadv | 2026-10-14 | + site-adversarial head (0.1) | psychosis (HC/SZ/BP) | 965 | yes | 1 | no | 0.863 ± 0.022 | 0.847 ± 0.020 |
| v8 | v8_views | 2026-10-14 | + band-limited and lagged connectivity views | psychosis (HC/SZ/BP) | 965 | yes | 1 | no | 0.853 ± 0.034 | 0.837 ± 0.017 |
| v8 | v8_aux | 2026-10-14 | + auxiliary age/sex head (0.1) | psychosis (HC/SZ/BP) | 965 | yes | 1 | no | 0.852 ± 0.022 | 0.853 ± 0.031 |
| v8 | v8_all | 2026-10-14 | delay + site-adversarial + views + auxiliary head | psychosis (HC/SZ/BP) | 965 | yes | 1 | no | 0.868 ± 0.031 | 0.854 ± 0.023 |
| v8 | v8_ens5 | 2026-10-14 | five-seed ensemble | psychosis (HC/SZ/BP) | 965 | yes | 5 | no | 0.857 ± 0.017 | 0.856 ± 0.022 |
| v9 | v9_pre_lr1e-3 | 2026-10-14 | + self-supervised pretrained encoder, lr 1e-3 | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.866 ± 0.017 | 0.866 ± 0.020 |
| v9 | v9_pre_lr3e-4 | 2026-10-14 | pretrained encoder, lr 3e-4 | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.855 ± 0.017 | 0.855 ± 0.017 |
| v10 | v10_e2e | 2026-10-16 | + edge-to-edge layer (8 filters) | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.864 ± 0.021 | 0.854 ± 0.028 |
| v10 | v10_ocread | 2026-10-16 | + orthonormal clustering readout (4 clusters) | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.868 ± 0.017 | 0.861 ± 0.029 |
| v10 | v10_e2e_ocread | 2026-10-16 | + edge-to-edge layer and clustering readout | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.853 ± 0.019 | 0.851 ± 0.026 |
| v11 | v11_l2_1e-5 | 2026-10-16 | simplified code; Adam + explicit L2 1e-5, re-pretrained encoder | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.853 ± 0.021 | 0.853 ± 0.026 |
| v11 | v11_l2_1e-4 | 2026-10-16 | Adam + explicit L2 1e-4 | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.854 ± 0.022 | 0.853 ± 0.025 |
| v11 | v11_l2_1e-3 | 2026-10-17 | Adam + explicit L2 1e-3 | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.856 ± 0.021 | 0.852 ± 0.022 |
| v13 | v13_base | 2026-10-17 | simplified code, AdamW (weight decay 0.05), pretrained encoder | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.868 ± 0.017 | 0.869 ± 0.019 |
| v13 | v13_small | 2026-10-17 | + smaller encoder (depth 2, dim 32), own pretraining | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.863 ± 0.019 | 0.860 ± 0.026 |
| v13 | v13_freeze | 2026-10-17 | + frozen SPD congruence matrix | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.866 ± 0.019 | 0.862 ± 0.022 |
| v13 | v13_ema | 2026-10-17 | + weight moving average (0.99) | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.864 ± 0.017 | 0.867 ± 0.021 |
| v13 | v13_crop | 2026-10-17 | + random-segment cropping (0.8--1 of the run) | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.868 ± 0.013 | 0.868 ± 0.024 |
| final | final_psychosisnet | 2026-10-19 | v13_crop configuration, final run used for all reported results | psychosis (HC/SZ/BP) | 965 | yes | 5 | yes | 0.868 ± 0.013 | 0.868 ± 0.024 |
