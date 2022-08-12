#!/bin/bash
if [ $# -gt 0 ]; then
  SCRIPT=$1
else
  echo Script is not given.
  exit
fi

if [ $# -gt 1 ]; then
  GPU_ID=$2
else
  GPU_ID=0
fi

ID=$(basename "${SCRIPT}" | sed "s/.sh$//g")
if [ "$(echo "${ID}" | grep -o resnet)" = resnet ]; then
  DEPTH=5
elif [ "$(echo "${ID}" | grep -o lstm-1)" = lstm-1 ]; then
  DEPTH=2
else
  DEPTH=7
fi

if [ "$(echo "${ID}" | grep -o partition)" = partition ]; then
  LAYERS="0 ${DEPTH}"
else
  LAYERS=$(seq 0 ${DEPTH})
fi

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

for RANDOM_SEED in $(seq 5); do
  for N_SHARED_LAYERS in ${LAYERS}; do
    N_INDIVIDUAL_LAYERS=$((DEPTH - N_SHARED_LAYERS))
    sh "${SCRIPT_DIR}"/wrapper.sh \
      "${SCRIPT}" \
      "${N_SHARED_LAYERS}" \
      "${N_INDIVIDUAL_LAYERS}" \
      "${RANDOM_SEED}" \
      "${GPU_ID}"
  done
done
