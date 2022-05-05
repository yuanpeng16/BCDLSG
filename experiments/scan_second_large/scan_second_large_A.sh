#!/usr/bin/env bash

ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

MYDIR=logs/${ID}
mkdir -p ${MYDIR}
cp ${ABS_PATH} ${MYDIR}

CUDA_VISIBLE_DEVICES= \
python3 -u main.py \
--experiment_id ${ID} \
--data_name uniform_scan \
--random_seed 8 \
--batch_size 256 \
--switch_temperature 0.1 \
--attention_temperature 1 \
--num_units 64 \
--epochs 50 \
--learning_rate 0.01 \
--max_gradient_norm 1.0 \
--use_embedding \
--embedding_size 32 \
--bidirectional_encoder \
--decay_steps 100 \
--content_noise_coe 0.1 \
--sample_wise_content_noise \
--masked_attention \
--random_random \
--single_representation \
--use_decoder_input \
| tee ${MYDIR}/log.txt
