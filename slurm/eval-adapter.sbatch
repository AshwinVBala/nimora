#!/usr/bin/env bash
#SBATCH --job-name=nimora-eval
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=nimora-eval-%j.out

set -euo pipefail

nimora_root="${NIMORA_ROOT:-${SLURM_SUBMIT_DIR:?submit from the Nimora repository}}"
nimora_env="${NIMORA_ENV:-$HOME/orcd/scratch/nimora/env}"
nimora_cache="${NIMORA_CACHE:-$HOME/orcd/scratch/nimora/cache}"
nimora_adapter="${NIMORA_ADAPTER:?NIMORA_ADAPTER is required}"
nimora_eval_file="${NIMORA_EVAL_FILE:?NIMORA_EVAL_FILE is required}"
nimora_eval_output="${NIMORA_EVAL_OUTPUT:?NIMORA_EVAL_OUTPUT is required}"
miniforge_module="${NIMORA_MINIFORGE_MODULE:-miniforge}"

module load "$miniforge_module"
if [[ ! -x "$nimora_env/bin/python" ]]; then
  echo "Missing environment: $nimora_env; submit slurm/bootstrap.sbatch first" >&2
  exit 2
fi

mkdir -p "$nimora_cache/huggingface" "$nimora_cache/torch" \
  "$(dirname "$nimora_eval_output")"
export HF_HOME="$nimora_cache/huggingface"
export TORCH_HOME="$nimora_cache/torch"
export PYTHONNOUSERSITE=True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$nimora_root"
nvidia-smi

eval_args=(
  "$nimora_root/scripts/evaluate_adapter.py"
  --adapter "$nimora_adapter"
  --eval-file "$nimora_eval_file"
  --output "$nimora_eval_output"
  --quantization 4bit
)
if [[ -n "${NIMORA_BASE_MODEL:-}" ]]; then
  eval_args+=(--base-model "$NIMORA_BASE_MODEL")
fi
if [[ -n "${NIMORA_MODEL_REVISION:-}" ]]; then
  eval_args+=(--model-revision "$NIMORA_MODEL_REVISION")
fi

srun --unbuffered "$nimora_env/bin/python" "${eval_args[@]}"
