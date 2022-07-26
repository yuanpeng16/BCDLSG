#!/usr/bin/env bash

if [ $# = 5 ]; then
  LOG_DIR=$1
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
  --data_random_seed "${RANDOM_SEED}" \
  --merge_type zeroshot_sun \
  --model_type cnn \
  --log_interval 100 \
  --batch_size 256 \
  --lr 0.001 \
  --steps 1000 \
  --n_hidden_nodes 64 \
  --n_common_layers "${N_COMMON_LAYERS}" \
  --n_separate_layers "${N_SEPARATE_LAYERS}" \
  --label_split tile \
  --combined_labels 1 \
  --log_dir "${LOG_DIR}"
