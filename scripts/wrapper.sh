#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  SCRIPT=$1
else
  echo Script is not given.
  exit
fi

if [ $# -gt 1 ]; then
  N_SHARED_LAYERS=$2
else
  N_SHARED_LAYERS=0
fi

if [ $# -gt 2 ]; then
  N_INDIVIDUAL_LAYERS=$3
else
  N_INDIVIDUAL_LAYERS=7
fi

if [ $# -gt 3 ]; then
  RANDOM_SEED=$4
else
  RANDOM_SEED=1
fi

if [ $# -gt 4 ]; then
  GPU_ID=$5
else
  GPU_ID=0
fi

ID=$(basename "${SCRIPT}" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f "${SCRIPT}")
cd "$(dirname "$(dirname "$(dirname "${ABS_PATH}")")")" || exit

LOG_DIR="logs/${ID}/${ID}_${N_SHARED_LAYERS}_${N_INDIVIDUAL_LAYERS}_${RANDOM_SEED}"
mkdir -p "${LOG_DIR}"
cp "${ABS_PATH}" "${LOG_DIR}"

command=$(sh "${SCRIPT}" \
  "${LOG_DIR}" \
  "${N_SHARED_LAYERS}" \
  "${N_INDIVIDUAL_LAYERS}" \
  "${RANDOM_SEED}" \
  "${GPU_ID}")

echo "${command}" |
  sed 's/python/\\\n  python/g' |
  sed 's/--/\\\n  --/g' |
  tee "${LOG_DIR}"/command.txt

date >"${LOG_DIR}"/time.txt
eval "${command}" | tee "${LOG_DIR}"/log.txt
date >>"${LOG_DIR}"/time.txt
