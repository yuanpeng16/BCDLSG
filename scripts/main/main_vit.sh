#!/usr/bin/env bash

if [ $# = 5 ]; then
  MY_DIR=$1
  N_COMMON_LAYERS=$2
  N_SEPARATE_LAYERS=$3
  RANDOM_SEED=$4
  GPU_ID=$5
else
  echo log_dir, n_common_layers, n_separate_layers, random_seed, and gpu_id.
  exit
fi

echo \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python3 -u main.py \
  --parameter_random_seed "${RANDOM_SEED}" \
  --merge_type added \
  --model_type vision_transformer \
  --log_interval 200 \
  --batch_size 512 \
  --lr 0.001 \
  --steps 10000 \
  --n_hidden_nodes 64 \
  --n_common_layers "${N_COMMON_LAYERS}" \
  --n_separate_layers "${N_SEPARATE_LAYERS}" \
  --dataset1 cifar10 \
  --dataset2 fashion_mnist \
  --label_split diagonal \
  --log_dir "${MY_DIR}"