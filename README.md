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
    {"role":"assistant","channel":"plan","content":"Locate the failing test first."},
    {"role":"assistant","action":{"name":"search","arguments":{"query":"parser","path":"tests"}}},
    {"role":"tool","name":"search","content":"tests/test_parser.py:14"},
    {"role":"assistant","channel":"result","content":"Fixed and verified with tests."}
  ]
}
```

For trajectory records, controller loss is applied only to assistant planning, actions,
and results. User, system, and tool-observation tokens remain visible as context but are
masked from the loss. Raw `text` records learn on every token.

Never train on private repositories, prompts, or interaction logs without explicit
authorization. Preserve source, license, commit, and deduplication metadata in the
dataset-production system even if the packed training shards omit it.

## Prepare controller data

Train the 32K byte-level BPE tokenizer on a representative licensed mixture:

```bash
nimora train-tokenizer \
  --input data/source/code.jsonl \
  --input data/source/trajectories.jsonl \
  --output data/tokenizer/tokenizer.json \
  --vocab-size 32768
```

Create separate train and validation shards:

```bash
nimora prepare \
  --input data/source/train.jsonl \
  --tokenizer data/tokenizer/tokenizer.json \
  --output-dir data/processed/train

nimora prepare \
  --input data/source/validation.jsonl \
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

It intentionally uses ordinary FP16 LoRA—not QLoRA—to avoid making the first training
run depend on a ROCm quantization stack. The default rank is 16 and targets attention and
SwiGLU projection layers. Trajectories are indexed by byte offset and tokenized lazily, so
the complete corpus is never materialized in RAM. The final adapter and tokenizer are
written under the configured output directory.

## Validation without training

Static checks do not allocate a model or touch a GPU:

```bash
python -m compileall -q src tests
pytest -q tests/test_serialization.py tests/test_config.py
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
