#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  MY_SCRIPT=$1
else
  echo Needs script.
  exit
fi

if [ $# -gt 1 ]; then
  N_COMMON_LAYERS=$2
else
  N_COMMON_LAYERS=0
fi

if [ $# -gt 2 ]; then
  N_SEPARATE_LAYERS=$3
else
  N_SEPARATE_LAYERS=7
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

ID=$(basename "${MY_SCRIPT}" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f "${MY_SCRIPT}")
cd "$(dirname "$(dirname "$(dirname"${ABS_PATH}")")")" || exit

MY_DIR="logs/${ID}/${ID}_${N_COMMON_LAYERS}_${N_SEPARATE_LAYERS}_${RANDOM_SEED}"
mkdir -p "${MY_DIR}"
cp "${ABS_PATH}" "${MY_DIR}"

command=$(sh "${MY_SCRIPT}" \
  "${MY_DIR}" \
  "${N_COMMON_LAYERS}" \
  "${N_SEPARATE_LAYERS}" \
  "${RANDOM_SEED}" \
  "${GPU_ID}")

echo "${command}" |
  sed 's/python/\\\n  python/g' |
  sed 's/--/\\\n  --/g' |
  tee "${MY_DIR}"/command.txt

date >"${MY_DIR}"/time.txt
eval "${command}" | tee "${MY_DIR}"/log.txt
date >>"${MY_DIR}"/time.txt
