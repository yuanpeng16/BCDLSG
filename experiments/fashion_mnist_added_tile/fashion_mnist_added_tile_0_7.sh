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
--merge_type added \
--model_type dnn \
--log_interval 200 \
--batch_size 512 \
--lr 0.001 \
--steps 2000 \
--n_hidden_nodes 512 \
--n_common_layers 0 \
--n_separate_layers 7 \
--dataset1 fashion_mnist \
--dataset2 mnist \
--label_split tile \
--log_dir ${MYDIR} \
| tee ${MYDIR}/log.txt