#!/bin/bash
if [ $# -gt 0 ]; then
  exp_id=$1
else
  exp_id="partition-f_main_dnn"
fi

log_dir="logs/${exp_id}"
temp_dir="/tmp/${log_dir}"
mkdir -p "${temp_dir}"

output_dir="outputs/${exp_id}"
mkdir -p "${output_dir}"

prefix="${log_dir}/${exp_id}"

for COL in $(seq 15); do
  for i in "${prefix}_0_7_1" "${prefix}_7_0_1"; do
    temp_file="/tmp/${i}.txt"
    grep -v final <"${i}/log.txt" | cut -d' ' -f"${COL}" >"${temp_file}"
  done

  paste "${temp_dir}/${exp_id}_0_7_1.txt" "${temp_dir}/${exp_id}_7_0_1.txt" |
    python3 "${HOME}"/Work/repositories/tools/plot.py \
      --filename "${output_dir}/${COL}.pdf"
done
