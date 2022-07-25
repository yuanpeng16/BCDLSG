#!/usr/bin/env bash

if [ $# = 1 ]; then
  GPU_ID=$1
else
  GPU_ID=0
fi

LOG_DIR=/tmp
N_COMMON_LAYERS=7
N_SEPARATE_LAYERS=0
RANDOM_SEED=1

echo \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python3 -u main.py \
  --parameter_random_seed "${RANDOM_SEED}" \
  --data_random_seed "${RANDOM_SEED}" \
  --merge_type added \
  --model_type dnn \
  --evaluator_type partition \
  --log_interval 1 \
  --batch_size 512 \
  --lr 0.001 \
  --steps 2000 \
  --n_hidden_nodes 512 \
  --n_common_layers "${N_COMMON_LAYERS}" \
  --n_separate_layers "${N_SEPARATE_LAYERS}" \
  --dataset1 fashion_mnist \
  --dataset2 mnist \
  --label_split diagonal \
  --log_dir "${LOG_DIR}"
