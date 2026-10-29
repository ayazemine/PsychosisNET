#!/bin/bash
# Completes the external test (TCP, ds005237) after run_final.sh, whose six external runs failed because
# the TCP features had not been extracted (empty test set). Nothing else is retrained.
#   bash run_external.sh "<final PsychosisNET options>"     (the same options as run_final.sh)
# 1  features of ds005237 only; their rows are added to every index.csv, all other files and rows unchanged
# 2  the six external runs (split external_ds005237; learning rates chosen in run_final.sh)
# 3  make_results.sh: every figure and table again from the saved models (no training)
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=.venv/Scripts/python
FINAL="$1"
log() { echo "== $1 exit $2"; grep -E "test_auc" data/logs/$1.log | tail -1; }
net() { name=$1; shift; $PY -m psikoz.train $FINAL "$@" --name $name > data/logs/$name.log 2>&1; log $name $?; }
cmp() { name=$1; model=$2; shift 2; $PY -m psikoz.baselines --model $model "$@" --name $name > data/logs/$name.log 2>&1; log $name $?; }

OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 $PY -m psikoz.cli features --datasets ds005237 > data/logs/features_ds005237.log 2>&1
echo "== features ds005237 exit $?"
$PY -c "
import pandas as pd
i = pd.read_csv('data/features/schaefer100s_gsr/index.csv', low_memory=False)
t = i[i.dataset == 'ds005237']
print('ds005237 runs in index:', len(t), ' pass QC:', int(t.qc_pass.sum()))
assert t.qc_pass.sum() > 0, 'no usable TCP run'
" || exit 1

best_lr() { $PY -c "
import json
v = {lr: json.load(open(f'data/runs/cmp_$1_lr{lr}/summary.json'))['summary']['val_auc'][0] for lr in ('1e-3', '1e-4')}
print(max(v, key=v.get))"; }
LR_BRAINNETCNN=$(best_lr brainnetcnn); LR_MLP=$(best_lr mlp); LR_BNT=$(best_lr bnt)
S=external_ds005237
(
  net final_psychosisnet_$S --split $S
  cmp cmp_brainnetcnn_$S brainnetcnn --lr $LR_BRAINNETCNN --split $S
) &
(
  cmp cmp_bnt_$S bnt --lr $LR_BNT --split $S
  cmp cmp_mlp_$S mlp --lr $LR_MLP --split $S
  cmp cmp_svm_$S svm --split $S
  cmp cmp_xgboost_$S xgboost --split $S
) &
wait

bash make_results.sh
