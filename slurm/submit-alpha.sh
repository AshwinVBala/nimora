#!/usr/bin/env bash
set -euo pipefail

nimora_root="${NIMORA_ROOT:-$(git rev-parse --show-toplevel)}"
nimora_env="${NIMORA_ENV:-$HOME/orcd/scratch/nimora/env}"
nimora_cache="${NIMORA_CACHE:-$HOME/orcd/scratch/nimora/cache}"
run_name="${NIMORA_RUN_NAME:-alpha-$(date -u +%Y%m%d-%H%M%S)}"
run_dir="${NIMORA_RUN_DIR:-$HOME/orcd/scratch/nimora/runs/$run_name}"
config="${NIMORA_CONFIG:-$nimora_root/configs/lora-qwen3-4b-orcd.yaml}"
eval_file="${NIMORA_EVAL_FILE:-$nimora_root/data/trajectories/alpha-v0.0.1/eval.jsonl}"
partition="${NIMORA_PARTITION:-mit_normal_gpu}"
gpus="${NIMORA_GPUS:-l40s:1}"
train_time="${NIMORA_TRAIN_TIME:-06:00:00}"
eval_time="${NIMORA_EVAL_TIME:-02:00:00}"
allow_dirty="${NIMORA_ALLOW_DIRTY:-0}"

command -v sbatch >/dev/null || { echo "sbatch is unavailable" >&2; exit 2; }
[[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
[[ -f "$eval_file" ]] || { echo "Missing eval data: $eval_file" >&2; exit 2; }
[[ -x "$nimora_env/bin/python" ]] || {
  echo "Missing environment: $nimora_env; submit slurm/bootstrap.sbatch first" >&2
  exit 2
}
if [[ "$allow_dirty" != "1" ]] && [[ -n "$(git -C "$nimora_root" status --porcelain)" ]]; then
  echo "Refusing to submit a dirty tree; commit it or set NIMORA_ALLOW_DIRTY=1" >&2
  exit 2
fi

mkdir -p "$run_dir/slurm"

train_export="ALL,NIMORA_ROOT=$nimora_root,NIMORA_ENV=$nimora_env,NIMORA_CACHE=$nimora_cache,NIMORA_CONFIG=$config,NIMORA_RUN_DIR=$run_dir,NIMORA_RESUME=auto,NIMORA_ALLOW_DIRTY=$allow_dirty"
train_job_raw=$(sbatch --parsable \
  --partition="$partition" \
  --gpus="$gpus" \
  --time="$train_time" \
  --output="$run_dir/slurm/train-%j.out" \
  --export="$train_export" \
  "$nimora_root/slurm/train-lora.sbatch")
train_job="${train_job_raw%%;*}"

eval_export="ALL,NIMORA_ROOT=$nimora_root,NIMORA_ENV=$nimora_env,NIMORA_CACHE=$nimora_cache,NIMORA_ADAPTER=$run_dir/final-adapter,NIMORA_EVAL_FILE=$eval_file,NIMORA_EVAL_OUTPUT=$run_dir/evaluation-report.json,NIMORA_BASE_MODEL=Qwen/Qwen3-4B,NIMORA_MODEL_REVISION=1cfa9a7208912126459214e8b04321603b3df60c"
eval_job_raw=$(sbatch --parsable \
  --dependency="afterok:$train_job" \
  --partition="$partition" \
  --gpus="$gpus" \
  --time="$eval_time" \
  --output="$run_dir/slurm/eval-%j.out" \
  --export="$eval_export" \
  "$nimora_root/slurm/eval-adapter.sbatch")
eval_job="${eval_job_raw%%;*}"

echo "Run directory: $run_dir"
echo "Training job: $train_job"
echo "Evaluation job: $eval_job (afterok:$train_job)"
echo "Monitor: squeue -j $train_job,$eval_job"
