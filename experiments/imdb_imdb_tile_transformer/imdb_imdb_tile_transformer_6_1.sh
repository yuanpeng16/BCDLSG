#!/usr/bin/env bash
if [ $# -gt 0 ]; then
  RANDOM_SEED=$1
else
  RANDOM_SEED=1
fi

if [ $# -gt 1 ]; then
  GPU_ID=$2
else
  GPU_ID=0
fi

ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

MYDIR=logs/${ID}_${RANDOM_SEED}
mkdir -p ${MYDIR}
cp ${ABS_PATH} ${MYDIR}

CUDA_VISIBLE_DEVICES=${GPU_ID} \
python3 -u mnist.py \
--parameter_random_seed ${RANDOM_SEED} \
--merge_type text \
--model_type transformer \
--log_interval 100 \
--batch_size 64 \
--lr 0.001 \
--steps 1000 \
--n_hidden_nodes 64 \
--n_common_layers 6 \
--n_separate_layers 1 \
--dataset1 imdb \
--dataset2 imdb \
--label_split tile \
--log_dir ${MYDIR} \
| tee ${MYDIR}/log.txt