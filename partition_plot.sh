#!/bin/bash
if [ $# -gt 0 ]; then
  COL=$1
else
  COL=8
fi

log_dir="logs/partition_main_cnn"
output_dir="/tmp/${log_dir}"
mkdir -p "${output_dir}"

for i in $(ls -d ${log_dir}/*); do
  tmp_dir="/tmp/${i}.txt"
  cat "${i}/log.txt" | grep -v final | cut -d' ' -f"${COL}" >"${tmp_dir}"
done

paste $(ls ${output_dir}/*)
