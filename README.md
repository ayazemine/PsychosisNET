# Explainable classification of psychosis from resting-state connectivity: cross-site and external evaluation

Code for a multi-site study that classifies psychosis (schizophrenia / schizoaffective disorder and
bipolar disorder with psychotic features) against participants without psychosis from resting-state
functional connectivity, with a connectivity-aware transformer (PsychosisNET) and five comparison
models (linear SVM, XGBoost, MLP, BrainNetCNN, Brain Network Transformer).

Evaluation: pooled 5-fold cross-validation (8 sites, n = 965, 318 with psychosis), leave-one-site-out,
single-site cross-validation, and an independent external site (Transdiagnostic Connectome Project)
that is never used for training, pretraining or model selection.

## Data

The imaging data are not redistributed here. Obtain them from the original sources and place them
under `datasets/` as listed in `configs/default.yaml`:

| Label in figures | Dataset | Source |
|---|---|---|
| COBRE | Center for Biomedical Research Excellence, NIAK-preprocessed release | INDI |
| UCLA CNP | UCLA Consortium for Neuropsychiatric Phenomics LA5c (ds000030) | OpenNeuro, `scripts/download_ds000030.py` |
| SZ-Smoking | rsfMRI comorbidity – Smoking and Schizophrenia (ds001461) | OpenNeuro, `scripts/download_ds001461.py` |
| VSC SZ/BP | The Neural Basis of Visual Shape Completion in Schizophrenia and Bipolar Disorder (ds005073) | OpenNeuro (folder `datasets/dataset1`) |
| SRPBS-KTT / KUT / SWA / UTO | SRPBS Multi-disorder MRI Dataset (unrestricted) | ATR, `scripts/unpack_srpbs.py` |
| TCP (external) | Transdiagnostic Connectome Project (ds005237) | OpenNeuro, `scripts/download_ds005237.py` |

COBRE and SRPBS are distributed under data use agreements; participant identifiers and participant-level
tables are therefore not part of this repository.

## Installation

Python 3.12; tested on Windows 11 with one NVIDIA RTX 3090 (CUDA 12.4).

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Linux / macOS: .venv/bin/pip
.venv/Scripts/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Pipeline

All settings are in `configs/default.yaml`; every methodological step is documented in the
supplementary methods of the article.

```
python -m psikoz.cli organize        # participant table and run manifest
python -m psikoz.cli preprocess      # motion correction, registration to MNI152 2009a (ANTs), confounds, QC
python -m psikoz.cli features        # ROI time series (Schaefer-100 + 14 subcortical), denoising, connectivity
python -m psikoz.cli splits          # subject-level splits: pooled 5-fold CV, leave-one-site-out, external
python -m psikoz.splits --single-site
python -m psikoz.pretrain --atlas schaefer100s_gsr --name pretrain_schaefer100s_gsr   # label-free pretraining
bash run_final.sh "--pretrained data/runs/pretrain_schaefer100s_gsr/encoder.pt --crop 0.8"
```

`run_final.sh` trains PsychosisNET and the five comparison models on identical folds (pooled CV,
leave-one-site-out, external test, single-site CV), saves every model
(`data/runs/<run>/fold<k>_seed<s>.pt` or `.joblib`) and finally calls `make_results.sh`.

## Figures and tables without retraining

```
bash make_results.sh                # all figures and tables from the saved predictions
bash make_results.sh --rescore      # first recompute every prediction from the saved model files
bash make_results.sh --plots-only   # result figures and tables only (no imaging data needed)
```

| Step | Module | Output |
|---|---|---|
| Rescoring from saved models | `psikoz.evaluate` | `data/runs/<run>/rescored/` |
| Grad-CAM, Integrated Gradients | `psikoz.interpret` | `data/runs/final_psychosisnet/interpret/` |
| Cohort tables, QC | `psikoz.cohort` | `results/T_demographics`, `T_acquisition`, `F_qc` |
| Performance, ROC, confusion matrices, learning curves, Grad-CAM, site-level analyses | `psikoz.report` | `results/T_*`, `results/F_*` |
| External test | `psikoz.external` | `results/T_external`, `F_external` |
| Methodology and architecture figures | `scripts/make_figures.py` | `figures/` |

Every figure is written as PDF, SVG and 300 dpi PNG; every table as CSV and Markdown.

## Pretraining ablation and supplementary exports

```
bash run_ablation_pretraining.sh                 # trains nopre_psychosisnet* on the same splits, then compares
python scripts/ablation_pretraining.py           # results/T_ablation_pretraining, F_ablation_pretraining (paired DeLong, Holm)
python scripts/export_dev_history.py             # results/T_dev_history (development runs of PsychosisNET)
python scripts/export_supplementary_checks.py    # results/T_lr_choice, T_gradcam_checks, T_network_ig
python scripts/fig_cohort_composition.py         # results/F_cohort_composition
```

These scripts read the saved runs in `data/runs/` (not distributed; see Data).

Trained models are not distributed: every reported model was trained with SRPBS data, whose terms of use do
not permit redistribution of derived data.

## Repository layout

```
psikoz/            pipeline package (preprocessing, features, models, training, evaluation, reporting)
scripts/           data download, SRPBS unpacking, manuscript figures, supplementary exports
configs/           settings
run_final.sh       all final training runs
run_external.sh    external-test runs
run_ablation_pretraining.sh  PsychosisNET without the pretrained encoder (pretraining ablation)
make_results.sh    figures and tables from saved models
figures/, results/ figures and aggregate tables of the article
```

Modules used only for the ablations and baselines reported in the supplement: `benchmark.py`,
`stack.py`, `combine.py`, `calibrate.py`, `lso.py`, `engineer.py`, `augment.py`, `embed_classify.py`.
