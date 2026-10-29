#!/bin/bash
# Final training and reporting: the selected PsychosisNET configuration and five comparison models on the
# same folds. Start only after the final configuration has been fixed from run_v13.sh (validation AUC).
#   bash run_final.sh "<final PsychosisNET options>"
#   e.g. bash run_final.sh "--pretrained data/runs/pretrain_schaefer100s_gsr/encoder.pt"
# Stage 1  pooled 5-fold CV: PsychosisNET; SVM, XGBoost; MLP, BrainNetCNN, BNT at two learning rates each
# Stage 2  learning rate of every deep comparison model = higher mean validation AUC (test not looked at)
# Stage 3  leave-one-site-out (8 sites) and external test (TCP) for all six models
# Stage 3b single-site 5-fold CV (split site_cv_<site>, 7 sites with both classes) for all six models,
#          same settings and learning rates; compared with the multi-site result in make_results.sh
# Stage 4  make_results.sh: every figure and table from the saved models (no training; can be rerun alone)
# Every model is saved: fold<k>_seed<s>.pt (deep models) or fold<k>_seed<s>.joblib (SVM, XGBoost).
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=.venv/Scripts/python
FINAL="$1"
SITES="cobre ds000030 ds001461 ds005073 srpbs-KTT srpbs-KUT srpbs-SWA srpbs-UTO"
log() { echo "== $1 exit $2"; grep -E "test_auc" data/logs/$1.log | tail -1; }
net() { name=$1; shift; $PY -m psikoz.train $FINAL "$@" --name $name > data/logs/$name.log 2>&1; log $name $?; }
cmp() { name=$1; model=$2; shift 2; $PY -m psikoz.baselines --model $model "$@" --name $name > data/logs/$name.log 2>&1; log $name $?; }

# ---- stage 1
(
  net final_psychosisnet
  for lr in 1e-3 1e-4; do cmp cmp_brainnetcnn_lr$lr brainnetcnn --lr $lr; done
  for lr in 1e-3 1e-4; do cmp cmp_mlp_lr$lr mlp --lr $lr; done
) &
(
  for lr in 1e-3 1e-4; do cmp cmp_bnt_lr$lr bnt --lr $lr; done
  cmp cmp_svm svm
  cmp cmp_xgboost xgboost
) &
wait

# ---- stage 2
best_lr() { $PY -c "
import json
v = {lr: json.load(open(f'data/runs/cmp_$1_lr{lr}/summary.json'))['summary']['val_auc'][0] for lr in ('1e-3', '1e-4')}
print(max(v, key=v.get))"; }
LR_BRAINNETCNN=$(best_lr brainnetcnn); LR_MLP=$(best_lr mlp); LR_BNT=$(best_lr bnt)
echo "== learning rates (validation AUC): BrainNetCNN $LR_BRAINNETCNN, MLP $LR_MLP, BNT $LR_BNT" | tee data/logs/final_lr_choice.log

# ---- stage 3
(
  for s in $SITES external_ds005237; do
    split=$([ "$s" = external_ds005237 ] && echo $s || echo leave_out_$s)
    net final_psychosisnet_$s --split $split
  done
  for s in $SITES external_ds005237; do
    split=$([ "$s" = external_ds005237 ] && echo $s || echo leave_out_$s)
    cmp cmp_brainnetcnn_$s brainnetcnn --lr $LR_BRAINNETCNN --split $split
  done
) &
(
  for s in $SITES external_ds005237; do
    split=$([ "$s" = external_ds005237 ] && echo $s || echo leave_out_$s)
    cmp cmp_bnt_$s bnt --lr $LR_BNT --split $split
    cmp cmp_mlp_$s mlp --lr $LR_MLP --split $split
    cmp cmp_svm_$s svm --split $split
    cmp cmp_xgboost_$s xgboost --split $split
  done
) &
wait

# ---- stage 3b
SINGLE="cobre ds000030 ds001461 srpbs-KTT srpbs-KUT srpbs-SWA srpbs-UTO"
(
  for s in $SINGLE; do net final_psychosisnet_site_$s --split site_cv_$s; done
  for s in $SINGLE; do cmp cmp_brainnetcnn_site_$s brainnetcnn --lr $LR_BRAINNETCNN --split site_cv_$s; done
) &
(
  for s in $SINGLE; do
    cmp cmp_bnt_site_$s bnt --lr $LR_BNT --split site_cv_$s
    cmp cmp_mlp_site_$s mlp --lr $LR_MLP --split site_cv_$s
    cmp cmp_svm_site_$s svm --split site_cv_$s
    cmp cmp_xgboost_site_$s xgboost --split site_cv_$s
  done
) &
wait

# ---- stage 4 (no training)
bash make_results.sh
