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
--merge_type paired \
--steps 200 \
--mask_input 2 \
| tee ${MYDIR}/log.txt