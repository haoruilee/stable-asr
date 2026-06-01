# Cloud Machine Runbook

This is the shortest path from a fresh rented GPU machine to a Stable-ASR final
run. The script is intentionally phase-based so missing corpora, external
prediction exports, or upstream ASR weights are reported before expensive jobs
start.

## Machine

Recommended first run:

```text
1 x RTX 4090 24GB or RTX A6000 48GB
128GB RAM
2TB-4TB NVMe
Ubuntu 22.04/24.04
CUDA-capable PyTorch environment
```

## Setup

```bash
git clone git@github.com:haoruilee/stable-asr.git
cd stable-asr
bash scripts/run_final_machine.sh setup
```

If the provider image already has a Python environment, override the venv path:

```bash
VENV_DIR=/workspace/stable-asr-venv bash scripts/run_final_machine.sh setup
```

## Stage Inputs

Place corpora and external outputs at the paths configured in
`configs/final/paper_final.json`:

```text
data/librispeech/LibriSpeech/dev-clean
data/aishell1/data_aishell
data/wenetspeech/WenetSpeech
data/common_voice/en
runs/final/external/*_raw.jsonl
runs/final/asr_commands/*.jsonl
runs/final/voiceworld_real_metadata.tsv
```

Then inspect missing files:

```bash
bash scripts/run_final_machine.sh status
```

## Run Phases

```bash
bash scripts/run_final_machine.sh prepare-inputs
bash scripts/run_final_machine.sh data-layer
bash scripts/run_final_machine.sh train
bash scripts/run_final_machine.sh evaluate
bash scripts/run_final_machine.sh asr-adapters
bash scripts/run_final_machine.sh bundle
```

After all inputs are staged, the full sequence is:

```bash
bash scripts/run_final_machine.sh final
```

The data-layer phase uses correctness-checked speed benchmarks. Audio windows
are reloaded from source WAV and log-mel feature caches are recomputed from
source audio before a speedup is accepted.
