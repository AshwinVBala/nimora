#!/usr/bin/env bash
set -u

echo "== Identity =="
hostname
id

echo
echo "== Accessible partitions =="
sinfo -o '%P|%a|%l|%D|%G|%c|%m'

echo
echo "== Community GPU inventory =="
sinfo -p mit_normal_gpu -O Partition,Nodes,StateCompact,CPUs,Memory,Gres

echo
echo "== Associations and QOS =="
sacctmgr -n -P show assoc user="$USER" \
  format=Cluster,Account,User,Partition,QOS,GrpTRES,MaxJobs 2>/dev/null || true

echo
echo "== Apptainer modules =="
module -t avail apptainer 2>&1 | tail -n 12

echo
echo "== Miniforge modules =="
module -t avail miniforge 2>&1 | tail -n 12

echo
echo "== Nimora storage candidates =="
for path in "$HOME/orcd/scratch" "$HOME/orcd/pool" "/scratch/$USER"; do
  if [[ -e "$path" ]]; then
    echo "$path"
    df -h "$path" | tail -n 1
  else
    echo "$path (missing)"
  fi
done

echo
echo "== Recent jobs =="
sacct -S today -u "$USER" -o JobID,JobName%24,Partition,State,Elapsed,AllocTRES \
  2>/dev/null || true
