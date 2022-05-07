#!/usr/bin/env bash

ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

MYDIR=logs/${ID}
mkdir -p ${MYDIR}
cp ${ABS_PATH} ${MYDIR}

CUDA_VISIBLE_DEVICES=0 \
python3 -u mnist.py \
--parameter_random_seed 7 \
--merge_type stacked \
--model_type cnn \
--n_common_layers 5 \
--n_separate_layers 1 \
--n_hidden_nodes 32 \
--steps 200 \
| tee ${MYDIR}/log.txt