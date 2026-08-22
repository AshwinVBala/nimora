# Nimora

Nimora is an open training stack for an action-native software-engineering system.
It has two deliberately separate model lanes:

- **Nimora Controller** is pretrained from random initialization. It selects tools,
  manages context, follows permission policy, recovers from failures, and drives Git/PR
  workflows.
- **Nimora Code** adapts `Qwen/Qwen3-4B` with LoRA. It writes, explains, repairs, and
  reviews code when the controller requests deeper code intelligence.

The default configurations target one AMD Radeon RX 9060 XT with 16 GB VRAM. No model
weights or datasets are included, and nothing trains merely by installing this package.

## Repository status

This repository contains training code and configuration only. Before a real run, it
still needs licensed training data, held-out evaluations, and a workstation with the
correct ROCm build of PyTorch.

## Architecture

```text
Developer request
       |
       v
Nimora Controller (~114M)
  | inspect/search/test/git/policy
  | compact state and recovery decisions
  v
Nimora Code (Qwen3-4B + LoRA)
  | patches/reviews/explanations
  v
Controller verifies evidence and continues
```

The controller is a decoder-only Transformer with RMSNorm, RoPE, grouped-query
attention, SwiGLU, tied embeddings, PyTorch scaled-dot-product attention, selective
loss masking, gradient checkpointing, and atomic resumable checkpoints.

## Nimora Runtime

The repository also contains a model-neutral coding-agent runtime. It works with an
OpenAI-compatible chat endpoint such as a local vLLM or llama.cpp server, so the agent
loop can be evaluated and used to collect trajectories before Nimora's own weights are
trained.

The loop exposes typed workspace, shell, local Git, and remote change-management tools.
Every action is schema-validated and policy-checked. Existing files require the SHA-256
returned by the prior read before they can be replaced. Git branches, commits, and pushes
require the expected HEAD SHA. Remote approval and merge require the runtime to fetch the
change, diff, and non-empty passing checks first, and the final mutation remains bound to
that exact head revision.

Review `configs/runtime-policy.yaml`, then run against a compatible endpoint:

```bash
nimora agent-run \
  --workspace /absolute/path/to/repository \
  --policy configs/runtime-policy.yaml \
  --endpoint http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-4B \
  --task "Find and fix the parser regression"
```

The checked-in policy permits concurrency-safe workspace edits but disables shell, Git
mutation, push, remote reads, PR creation, approval, and merge. Enable each capability
deliberately. `WorkspaceBoundary` prevents path traversal and symlink escapes, but it is
not an operating-system sandbox. Run shell-enabled agents inside a disposable container
or VM because tests and build tools execute repository code.

### Git providers

Copy `configs/provider.example.yaml`, choose `github`, `gitlab`, `gitea`, or `forgejo`,
and set the named token environment variable. A Forgejo-specific example is available at
`configs/provider-forgejo.example.yaml`. Tokens are read from the environment and are
never stored in the configuration or trajectory metadata. Pass the file with
`--provider-config`. Gitea and Forgejo require their instance base URL (or a URL already
ending in `/api/v1`). Forgejo is a distinct adapter because its compatibility with Gitea
is not guaranteed across releases. Verify the instance's `/api/swagger` documentation
when targeting a new Forgejo release. Custom providers can implement the `GitProvider`
protocol and register the same provider tools.

Trajectory recording is opt-in through `--record trajectories/sessions.jsonl`. Common
secret patterns are redacted and strings are bounded, but logs must still be reviewed
before entering a training corpus.

### Coding evaluations

Evaluation cases use pinned local Git repositories and explicit command checks. Nimora
clones each case into a disposable directory, runs the agent, executes checks, and writes
a JSONL result with status and a working-tree fingerprint:

```bash
nimora eval-run \
  --cases evals/cases.jsonl \
  --policy configs/runtime-policy.yaml \
  --endpoint http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-4B \
  --output evals/results/qwen3-4b.jsonl
```

Evaluation check commands are trusted code and must use an isolated container or VM.
Start from `evals/examples/cases.jsonl`; do not treat the placeholder case as runnable.

## Operating system and PyTorch

Use a ROCm-supported Linux release. Install AMD's Radeon/ROCm software and its matching
PyTorch wheel by following the current AMD instructions for the exact ROCm release.
Do not install a CUDA PyTorch wheel on the Radeon workstation.

After PyTorch is installed:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,lora,dev]'
nimora runtime-info
```

`runtime-info` only reports detected versions and hardware metadata. Training commands
refuse to proceed on CPU or when `torch.version.hip` is absent.

## Training data

Each JSONL line is either general pretraining text:

```json
{"text":"Licensed code, documentation, or structured software-engineering text."}
```

or an agent trajectory:

```json
{
  "messages": [
    {"role":"system","content":"Choose small, verifiable actions."},
    {"role":"user","content":"Fix the parser regression."},
    {"role":"assistant","decision":{"plan":"Locate the failing test first.","action":{"name":"workspace.search","arguments":{"query":"parser","path":"tests"}}}},
    {"role":"tool","name":"workspace.search","content":"tests/test_parser.py:14"},
    {"role":"assistant","decision":{"result":"Fixed and verified with tests."}}
  ]
}
```

For trajectory records, controller loss is applied only to assistant planning, actions,
and results. User, system, and tool-observation tokens remain visible as context but are
masked from the loss. Raw `text` records learn on every token.

Never train on private repositories, prompts, or interaction logs without explicit
authorization. Preserve source, license, commit, and deduplication metadata in the
dataset-production system even if the packed training shards omit it.

## Build the licensed source corpus

Nimora does not scrape repositories or infer that publicly accessible code is safe to
train on. Copy `data/examples/sources.jsonl` to the ignored `data/sources.jsonl` and
edit it. Every row must describe a clean local Git checkout, explicitly assert
authorization, pin the full
commit SHA, declare an allowlisted SPDX license, and identify a tracked license-evidence
file. Use at least two repositories so validation can remain repository-disjoint.

Review `configs/corpus-policy.yaml`, then build the corpus:

```bash
nimora build-corpus \
  --sources data/sources.jsonl \
  --policy configs/corpus-policy.yaml \
  --output-dir data/corpus
```

The builder reads only Git-tracked files and refuses dirty checkouts, revision drift,
unapproved licenses, missing license evidence, and an existing output directory. It
filters binary, oversized, generated, vendored, secret-pattern-bearing, and
email-address-bearing files; performs exact and near-duplicate removal; and assigns
entire repositories to a stable train or validation split.

Its outputs are:

- `train.jsonl` and `validation.jsonl`: normalized text plus source URL, revision,
  SPDX declaration, license-evidence hash, repository path, content hash, and split.
- `audit.jsonl`: one inclusion or rejection decision per tracked file, without rejected
  file contents.
- `manifest.json`: policy digest, immutable source lock, counts, and rejection summary.

The license declaration remains your responsibility; recording a license file and hash
is provenance evidence, not an automated legal conclusion.

## Prepare controller data

Train the 32K byte-level BPE tokenizer on a representative licensed mixture:

```bash
nimora train-tokenizer \
  --input data/corpus/train.jsonl \
  --input data/source/trajectories.jsonl \
  --output data/tokenizer/tokenizer.json \
  --vocab-size 32768
```

Create separate train and validation shards:

```bash
nimora prepare \
  --input data/corpus/train.jsonl \
  --input data/source/train-trajectories.jsonl \
  --tokenizer data/tokenizer/tokenizer.json \
  --output-dir data/processed/train

nimora prepare \
  --input data/corpus/validation.jsonl \
  --input data/source/validation-trajectories.jsonl \
  --tokenizer data/tokenizer/tokenizer.json \
  --output-dir data/processed/validation
```

Packing produces memory-mapped `uint16` token shards, `uint8` loss masks, and a
`metadata.json` manifest. Split by repository before packing to prevent train/eval
leakage.

## Pretrain Nimora Controller

Review `configs/controller-120m.yaml` first. Then, only when the ROCm workstation is
ready:

```bash
nimora estimate --config configs/controller-120m.yaml
nimora train-controller --config configs/controller-120m.yaml
```

The default model is approximately 114M parameters. FP16 model weights are small, but
AdamW state, gradients, activations, and the attention workspace determine actual VRAM
usage. The conservative defaults are sequence length 2048, micro-batch 1, gradient
accumulation 32, and gradient checkpointing enabled.

Resume by setting `training.resume_from` to a `training-state.pt` checkpoint. Each
checkpoint includes the model, optimizer, scaler, step, configuration, and random-number
generator states.

## Train the Qwen3-4B LoRA adapter

The LoRA lane consumes trajectory JSONL directly and masks loss to assistant turns:

```bash
nimora train-lora --config configs/lora-qwen3-4b.yaml
```

The default configuration uses ordinary FP16 LoRA for the ROCm workstation. Set
`quantization.enabled: true` to use 4-bit NF4 QLoRA on a supported CUDA runtime, such as
a Colab T4. The default rank is 16 and targets attention and SwiGLU projection layers.
Trajectories are indexed by byte offset and tokenized lazily, so the complete corpus is
never materialized in RAM. The final adapter and tokenizer are written under the
configured output directory.

## Run QLoRA on MIT ORCD

The ORCD lane defaults to one L40S in `mit_normal_gpu`, uses scratch storage for the
Python environment and model cache, saves resumable Trainer checkpoints, and submits
evaluation only after training succeeds. It never publishes weights automatically.

From an ORCD Open OnDemand terminal, clone or update the repository and inspect the
resources available to your account:

```bash
cd ~/orcd/scratch
git clone https://github.com/AshwinVBala/nimora.git nimora
cd nimora
bash slurm/discover-orcd.sh
```

Build the pinned environment once as a CPU job:

```bash
sbatch --export=ALL,NIMORA_ROOT="$PWD" slurm/bootstrap.sbatch
```

After that job completes successfully, submit the resumable train-to-evaluation chain:

```bash
bash slurm/submit-alpha.sh
```

The submission helper requires a clean Git tree so every run records an exact source
revision. It prints both job IDs and places logs, resolved configuration, provenance,
checkpoints, the final adapter, and `evaluation-report.json` under
`~/orcd/scratch/nimora/runs/<run-name>/`. Monitor with `squeue --me`, inspect completed
jobs with `sacct -j <job-id>`, and follow the printed log paths with `tail -f`.

Resource choices are environment overrides. For example, after confirming account
access with the discovery script:

```bash
NIMORA_GPUS=h200:1 bash slurm/submit-alpha.sh
NIMORA_PARTITION=mit_preemptable NIMORA_TRAIN_TIME=12:00:00 bash slurm/submit-alpha.sh
```

The preemptable queue can interrupt a job. The training batch script requests requeue,
handles the advance signal, and resumes from the newest complete `checkpoint-*`
directory. Keep Hugging Face tokens out of batch files; the pinned public Qwen base does
not need one. Review the evaluation report and publication gate before separately
uploading an adapter.

## Validation without training

Static checks do not allocate a model or touch a GPU:

```bash
python -m compileall -q src tests
pytest -q tests/test_serialization.py tests/test_config.py
pytest -q tests/test_corpus.py
ruff check src tests
```

Tests that import PyTorch automatically skip when it is unavailable. No automated test
in this repository launches a training run.

## Planned training stages

1. General code, documentation, and structured action-language pretraining.
2. Supervised tool selection and context-control trajectories.
3. Failure recovery, rollback, and evidence-based completion.
4. Issue-to-PR workflows against reproducible repositories.
5. Independent review, SHA-bound approval, and policy escalation.
6. Preference or reinforcement training using executable task outcomes.

## License

Apache-2.0. Model weights and datasets must document their own licenses separately.
