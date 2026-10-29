#!/bin/bash
# Ablation of self-supervised pretraining: the final PsychosisNET options without --pretrained, on the
# pooled CV, the 8 leave-one-site-out splits and the external TCP split (run 28-29 Oct 2026, GPU 3).
# Compared with final_psychosisnet* by paired DeLong tests in scripts/ablation_pretraining.py.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=.venv/Scripts/python
OPTS="--crop 0.8"
run() { name=$1; shift; $PY -m psikoz.train $OPTS "$@" --name $name > data/logs/$name.log 2>&1; echo "== $name exit $? $(date +%H:%M)"; grep -E "test_auc" data/logs/$name.log | tail -1; }
run nopre_psychosisnet
for s in srpbs-UTO srpbs-KUT srpbs-SWA ds000030 cobre ds001461 ds005073 srpbs-KTT; do run nopre_psychosisnet_$s --split leave_out_$s; done
run nopre_psychosisnet_external_ds005237 --split external_ds005237
.venv/Scripts/python scripts/ablation_pretraining.py
