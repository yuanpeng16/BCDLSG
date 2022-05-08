#!/usr/bin/env bash
if [ $# = 1 ]; then
  RANDOM_SEED=$1
else
  RANDOM_SEED=1
fi

ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

MYDIR=logs/${ID}_${RANDOM_SEED}
mkdir -p ${MYDIR}
cp ${ABS_PATH} ${MYDIR}

CUDA_VISIBLE_DEVICES=0 \
python3 -u mnist.py \
--parameter_random_seed ${RANDOM_SEED} \
--merge_type added \
--model_type cnn \
--log_interval 100 \
--batch_size 512 \
--lr 0.001 \
--steps 5000 \
--n_hidden_nodes 64 \
--n_common_layers 1 \
--n_separate_layers 5 \
--dataset1 cifar10 \
--dataset2 fashion_mnist \
--label_split tile \
--rotate_second_input \
--log_dir ${MYDIR} \
| tee ${MYDIR}/log.txt