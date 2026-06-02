# Large-Scale Data Runbook

This runbook is for preparing data before moving work to a training platform.
The goal is to do all CPU-friendly work first: source collection planning,
manifest normalization, weak turn-window derivation, leakage-audited splits,
external turn-data conversion, VoiceWorld scenario normalization, and
Parquet/Lance data-layer checks.

## Setup

Base install:

```bash
bash scripts/prepare_large_scale_data.sh setup
```

Install data backends for Parquet and Lance:

```bash
bash scripts/prepare_large_scale_data.sh setup-data
```

Install all optional dependencies, including Torch, when you also want to build
log-mel feature caches locally:

```bash
bash scripts/prepare_large_scale_data.sh setup-all
```

## Inspect Inputs

```bash
bash scripts/prepare_large_scale_data.sh status
```

By default the script looks under `data/`:

```text
data/librispeech/LibriSpeech
data/aishell1/data_aishell
data/wenetspeech/WenetSpeech
data/common_voice/en
data/voiceworld/metadata.tsv
data/voiceworld/audio
```

Override paths with environment variables:

```bash
OUTPUT_DIR=/mnt/asr-prep \
DATA_ROOT=/mnt/corpora \
LIBRISPEECH_DIR=/mnt/corpora/LibriSpeech \
AISHELL1_DIR=/mnt/corpora/data_aishell \
WENETSPEECH_DIR=/mnt/corpora/WenetSpeech \
COMMON_VOICE_DIR=/mnt/corpora/common_voice/en \
bash scripts/prepare_large_scale_data.sh status
```

## Prepare ASR Manifests

Prepare every configured public corpus that exists locally, then combine them
into one ASR manifest:

```bash
bash scripts/prepare_large_scale_data.sh manifests
```

Useful split filters:

```bash
LIBRISPEECH_SPLIT=train-clean-100 bash scripts/prepare_large_scale_data.sh manifests
AISHELL1_SPLIT=train bash scripts/prepare_large_scale_data.sh manifests
WENETSPEECH_SPLIT=train bash scripts/prepare_large_scale_data.sh manifests
COMMON_VOICE_SPLIT=validated bash scripts/prepare_large_scale_data.sh manifests
```

For custom metadata tables:

```bash
ASR_METADATA=/mnt/corpora/my_asr.tsv \
ASR_AUDIO_ROOT=/mnt/corpora/audio \
ASR_LANGUAGE=zh \
ASR_SOURCE=my_asr_v1 \
bash scripts/prepare_large_scale_data.sh manifests
```

## Derive Weak Turn Splits

```bash
bash scripts/prepare_large_scale_data.sh turn
```

This writes:

```text
runs/large_data/turn/turn_manifest.jsonl
runs/large_data/turn/splits/turn_train.jsonl
runs/large_data/turn/splits/turn_dev.jsonl
runs/large_data/turn/splits/turn_test.jsonl
```

The script groups split assignment by `metadata.asr_record_id` by default, so
the complete and incomplete windows derived from the same utterance cannot leak
across train/dev/test.

## Convert Turn Datasets

If you have raw JSONL exports for turn datasets:

```bash
EASYTURN_INPUT=/mnt/turn/easyturn.jsonl \
FULL_DUPLEX_BENCH_INPUT=/mnt/turn/full_duplex_bench.jsonl \
SMART_TURN_INPUT=/mnt/turn/smart_turn.jsonl \
bash scripts/prepare_large_scale_data.sh external
```

## Prepare VoiceWorld Scenarios

```bash
VOICEWORLD_METADATA=/mnt/voiceworld/metadata.tsv \
VOICEWORLD_AUDIO_ROOT=/mnt/voiceworld/audio \
bash scripts/prepare_large_scale_data.sh voiceworld
```

## Data Layer

Run format conversion and random-sampling checks:

```bash
bash scripts/prepare_large_scale_data.sh data-layer
```

When Parquet and Lance are installed, the script automatically includes them.
The output JSON report is written to:

```text
runs/large_data/reports/data_benchmark.json
```

## Feature Cache

Feature-cache preparation is CPU-friendly but can be slow on large corpora. Run
it explicitly before training if you want to reduce training startup cost:

```bash
CACHE_DATASET=runs/large_data/voiceworld/voiceworld_real.jsonl \
CACHE_AUDIO_ROOT=data/voiceworld/audio \
SAMPLE_COUNT=10000 \
CORRECTNESS_SAMPLE_COUNT=10000 \
bash scripts/prepare_large_scale_data.sh cache
```

For a bounded local check:

```bash
MAX_RECORDS=1000 SAMPLE_COUNT=1000 CORRECTNESS_SAMPLE_COUNT=256 \
bash scripts/prepare_large_scale_data.sh cache
```

## One-Pass CPU Preparation

Run all non-training data preparation:

```bash
bash scripts/prepare_large_scale_data.sh all
```

Run everything plus log-mel feature-cache checks:

```bash
bash scripts/prepare_large_scale_data.sh full
```

Use strict mode when missing corpora or external files should fail instead of
being skipped:

```bash
REQUIRE_CORPORA=1 REQUIRE_EXTERNAL=1 bash scripts/prepare_large_scale_data.sh all
```

## Training Handoff

After data preparation, copy or mount `OUTPUT_DIR` on the target training
environment. The training phase should consume:

```text
<OUTPUT_DIR>/turn/splits/turn_train.jsonl
<OUTPUT_DIR>/turn/splits/turn_dev.jsonl
<OUTPUT_DIR>/turn/splits/turn_test.jsonl
<OUTPUT_DIR>/voiceworld/voiceworld_real.jsonl
<OUTPUT_DIR>/reports/*.json
```

This script prepares data; it does not make the dataset strong enough by
itself. Strong NanoTurn training still requires real `backchannel`, `wait`,
`interruption`, `side_conversation`, and `ambient_speech` labels at scale.
