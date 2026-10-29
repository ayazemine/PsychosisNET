#!/bin/bash
# Every result figure and table from the SAVED models and predictions only: nothing is trained here.
# Run after run_final.sh, and again whenever a figure or table is changed.
#   bash make_results.sh              figures / tables from the saved predictions of every run
#   bash make_results.sh --rescore    first recompute all predictions from the saved model files
#                                     (psikoz.evaluate -> data/runs/<run>/rescored/, with a check against
#                                     the stored predictions), then build everything from those
#   bash make_results.sh --plots-only only psikoz.report: needs data/runs, data/splits, the participants
#                                     table and the feature index, but no images or feature files (for a
#                                     machine without the datasets; the Grad-CAM maps are read from
#                                     data/runs/final_psychosisnet/interpret)
# Steps (all inference / plotting):
#   evaluate   (optional) saved models -> predictions                      data/runs/<run>/rescored/
#   interpret  Grad-CAM, Integrated Gradients, attention, embeddings       data/runs/final_psychosisnet/interpret/
#   cohort     demographics, acquisition, QC                               results/T_demographics, T_acquisition, F_qc
#   report     performance, CV, confusion, ROC, learning curves, Grad-CAM,
#              site-level, single- vs multi-site, LOSO                     results/T_*, F_*
#   external   independent external site (TCP)                             results/T_external, F_external
#   figures    manuscript figures 1, 2, S1, S2                             figures/
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${PY:-$([ -x .venv/Scripts/python ] && echo .venv/Scripts/python || echo .venv/bin/python)}
mkdir -p data/logs results
SITES="cobre ds000030 ds001461 ds005073 srpbs-KTT srpbs-KUT srpbs-SWA srpbs-UTO"
SINGLE="cobre ds000030 ds001461 srpbs-KTT srpbs-KUT srpbs-SWA srpbs-UTO"
step() { name=$1; shift; "$@" > data/logs/results_$name.log 2>&1; echo "== $name exit $?"; }

# learning rate of each deep comparison model, chosen by run_final.sh on the mean validation AUC
best_lr() { $PY -c "
import json
v = {lr: json.load(open(f'data/runs/cmp_$1_lr{lr}/summary.json'))['summary']['val_auc'][0] for lr in ('1e-3', '1e-4')}
print(max(v, key=v.get))"; }
LR_BRAINNETCNN=$(best_lr brainnetcnn); LR_MLP=$(best_lr mlp); LR_BNT=$(best_lr bnt)
echo "== learning rates: BrainNetCNN $LR_BRAINNETCNN, MLP $LR_MLP, BNT $LR_BNT"

PRED=""
if [ "$1" = "--rescore" ]; then
  RUNS="final_psychosisnet cmp_svm cmp_xgboost cmp_mlp_lr$LR_MLP cmp_brainnetcnn_lr$LR_BRAINNETCNN cmp_bnt_lr$LR_BNT"
  for s in $SITES external_ds005237; do
    for p in final_psychosisnet cmp_svm cmp_xgboost cmp_mlp cmp_brainnetcnn cmp_bnt; do RUNS="$RUNS ${p}_$s"; done
  done
  for s in $SINGLE; do
    for p in final_psychosisnet cmp_svm cmp_xgboost cmp_mlp cmp_brainnetcnn cmp_bnt; do
      [ -d data/runs/${p}_site_$s ] && RUNS="$RUNS ${p}_site_$s"
    done
  done
  step evaluate $PY -m psikoz.evaluate --runs $RUNS
  grep -h "max_abs_diff" data/logs/results_evaluate.log | tail -3
  PRED="--predictions rescored"
fi

PLOTS_ONLY=$([ "$1" = "--plots-only" ] && echo 1)
if [ -z "$PLOTS_ONLY" ]; then
  step interpret $PY -m psikoz.interpret --run final_psychosisnet
  step cohort $PY -m psikoz.cohort --out results
fi
step report $PY -m psikoz.report --final final_psychosisnet $PRED \
    --compare "Linear SVM=cmp_svm" "XGBoost=cmp_xgboost" "MLP=cmp_mlp_lr$LR_MLP" \
              "BrainNetCNN=cmp_brainnetcnn_lr$LR_BRAINNETCNN" "BNT=cmp_bnt_lr$LR_BNT" \
    --lso "PsychosisNET=final_psychosisnet" "Linear SVM=cmp_svm" "XGBoost=cmp_xgboost" "MLP=cmp_mlp" \
          "BrainNetCNN=cmp_brainnetcnn" "BNT=cmp_bnt" \
    --single "PsychosisNET=final_psychosisnet" "Linear SVM=cmp_svm" "XGBoost=cmp_xgboost" "MLP=cmp_mlp" \
             "BrainNetCNN=cmp_brainnetcnn" "BNT=cmp_bnt" \
    --out results
[ -n "$PLOTS_ONLY" ] && exit 0
step external $PY -m psikoz.external --run final_psychosisnet_external_ds005237 --out results
step figures $PY scripts/make_figures.py final_psychosisnet
