#!/usr/bin/env bash
#SBATCH --job-name=nimora-bootstrap
#SBATCH --partition=mit_normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=nimora-bootstrap-%j.out

set -euo pipefail

nimora_root="${NIMORA_ROOT:-${SLURM_SUBMIT_DIR:?submit from the Nimora repository}}"
nimora_env="${NIMORA_ENV:-$HOME/orcd/scratch/nimora/env}"
nimora_cache="${NIMORA_CACHE:-$HOME/orcd/scratch/nimora/cache}"
miniforge_module="${NIMORA_MINIFORGE_MODULE:-miniforge}"

module load "$miniforge_module"
mkdir -p "$nimora_cache/pip" "$(dirname "$nimora_env")"
export PIP_CACHE_DIR="$nimora_cache/pip"
export PYTHONNOUSERSITE=True

if [[ ! -x "$nimora_env/bin/python" ]]; then
  python -m venv "$nimora_env"
fi

"$nimora_env/bin/python" -m pip install --upgrade pip wheel
"$nimora_env/bin/python" -m pip install \
  --requirement "$nimora_root/slurm/requirements-orcd.txt"
"$nimora_env/bin/python" -m pip install --no-deps --editable "$nimora_root"
"$nimora_env/bin/python" -m pip freeze > "$nimora_env/requirements.lock.txt"

"$nimora_env/bin/python" - <<'PY'
import torch
import transformers
import peft
import accelerate
import bitsandbytes

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("accelerate", accelerate.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("CUDA build", torch.version.cuda)
PY

echo "Nimora environment ready: $nimora_env"
