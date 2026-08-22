#!/usr/bin/env bash
#SBATCH --job-name=nimora-train
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=nimora-train-%j.out

set -euo pipefail

nimora_root="${NIMORA_ROOT:-${SLURM_SUBMIT_DIR:?submit from the Nimora repository}}"
nimora_env="${NIMORA_ENV:-$HOME/orcd/scratch/nimora/env}"
nimora_cache="${NIMORA_CACHE:-$HOME/orcd/scratch/nimora/cache}"
nimora_config="${NIMORA_CONFIG:-$nimora_root/configs/lora-qwen3-4b-orcd.yaml}"
nimora_run_dir="${NIMORA_RUN_DIR:?NIMORA_RUN_DIR is required}"
nimora_resume="${NIMORA_RESUME:-auto}"
miniforge_module="${NIMORA_MINIFORGE_MODULE:-miniforge}"

module load "$miniforge_module"
if [[ ! -x "$nimora_env/bin/python" ]]; then
  echo "Missing environment: $nimora_env; submit slurm/bootstrap.sbatch first" >&2
  exit 2
fi

mkdir -p "$nimora_cache/huggingface" "$nimora_cache/torch" "$nimora_run_dir"
export HF_HOME="$nimora_cache/huggingface"
export TORCH_HOME="$nimora_cache/torch"
export PYTHONNOUSERSITE=True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$nimora_root"
nvidia-smi
"$nimora_env/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the SLURM allocation")
print("CUDA device:", torch.cuda.get_device_name(0), flush=True)
print("CUDA memory bytes:", torch.cuda.get_device_properties(0).total_memory, flush=True)
PY

train_args=(
  "$nimora_root/scripts/slurm_train_lora.py"
  --config "$nimora_config"
  --run-dir "$nimora_run_dir"
  --resume "$nimora_resume"
)
if [[ "${NIMORA_ALLOW_DIRTY:-0}" == "1" ]]; then
  train_args+=(--allow-dirty)
fi

child_pid=""
handle_usr1() {
  echo "Received USR1; stopping at the most recent Trainer checkpoint and requeueing."
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap handle_usr1 USR1

srun --unbuffered "$nimora_env/bin/python" "${train_args[@]}" &
child_pid=$!
set +e
wait "$child_pid"
status=$?
set -e
exit "$status"
