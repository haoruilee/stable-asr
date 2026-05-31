# Data Schema

The first stable data format is a JSONL turn/action manifest. Each line
represents one training or evaluation window.

## Required Fields

```json
{
  "id": "turn_000001",
  "audio": "audio/turn_000001.wav",
  "sample_rate": 16000,
  "start": 0.0,
  "end": 2.0,
  "turn_label": "complete",
  "action_label": "take_turn",
  "assistant_speaking": false,
  "overlap": false,
  "language": "en",
  "source": "example"
}
```

## Optional Fields

```json
{
  "text": "what is the weather",
  "asr_text": "what is the weather",
  "scenario": "normal_question",
  "metadata": {
    "pause_ms": 850,
    "snr_db": 20,
    "reverb": "none",
    "speaking_rate": 1.0
  }
}
```

## Registered Backends

- JSONL: built in
- Parquet: install `stable-asr[data]`
- Lance: install `stable-asr[lance]`

## VoiceWorld Scenario Names

The seedable synthetic suite currently includes:

- `normal_question`
- `incomplete_pause`
- `backchannel`
- `wait_stop`
- `user_interruption`
- `side_conversation`
- `ambient_speech`
- `noisy_farfield`
- `code_switching`

The scenario suite is also versioned as machine-readable JSON:

```bash
stable-asr scenario-suite --suite configs/scenarios/stable_asr_voiceworld_v0.json --validate-only
stable-asr scenario-suite --output runs/SCENARIO_SUITE.md
```

This file records expected turn/action labels, assistant-speaking state, overlap
state, factors of variation, and paper-facing metrics for each VoiceWorld
scenario.

## Turn Labels

- `complete`
- `incomplete`
- `backchannel`
- `wait`

## Action Labels

- `take_turn`
- `keep_listening`
- `continue_speaking`
- `stop_tts_and_listen`
- `hold`
- `ignore`
- `light_ack`

## External Conversions

Stable-ASR currently supports JSONL conversion for:

- EasyTurn-style manifests
- Full-Duplex-Bench-style manifests
- SmartTurn-style manifests

## Train/Dev/Test Splits

Use `split-turn-data` to create deterministic NanoTurn training splits from any
registered turn manifest backend:

```bash
stable-asr split-turn-data \
  --input examples/data/turn_demo.jsonl \
  --output-dir runs/splits \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 0
```

By default the command stratifies by `turn_label`. Repeat `--stratify-by` to
add fields such as `scenario`, use `--group-by metadata.conversation_id` to
keep dialogue windows together, or pass `--no-stratify` for a plain shuffled
split.

## Data Source Registry

Stable-ASR keeps a machine-readable source registry at
`configs/datasets/stable_asr_sources.json`. It separates implemented sources
from planned public-corpus recipes and can be rendered or validated:

```bash
stable-asr data-sources --registry configs/datasets/stable_asr_sources.json --validate-only
stable-asr data-sources --output runs/DATA_SOURCES.md
```

## ASR Corpus Manifest

Public ASR corpora usually start as utterance metadata rather than turn/action
windows. Stable-ASR therefore has a separate ASR manifest schema for local
corpus preparation:

```json
{
  "id": "asr_demo_0001",
  "audio": "audio/asr_demo_0001.wav",
  "sample_rate": 16000,
  "text": "what is the weather",
  "language": "en",
  "source": "librispeech",
  "duration": 2.1,
  "split": "dev",
  "speaker_id": "spk_a",
  "metadata": {
    "domain": "assistant_query"
  }
}
```

Create and validate this manifest from TSV/CSV/JSONL metadata:

```bash
stable-asr prepare-asr-manifest \
  --input examples/data/asr_metadata.tsv \
  --output runs/asr_manifest.jsonl \
  --audio-root examples/data \
  --sample-rate 16000

stable-asr validate-asr-manifest runs/asr_manifest.jsonl
stable-asr inspect-asr-manifest runs/asr_manifest.jsonl
stable-asr asr-to-turn --input runs/asr_manifest.jsonl --output runs/asr_turn.jsonl --include-incomplete
stable-asr audit-audio --kind asr --manifest runs/asr_manifest.jsonl
```

Supported input aliases include `utt_id`/`key`, `audio_path`/`wav`,
`transcript`/`reference`, `duration_sec`, `speaker_id`, `split`, `source`, and
`language`. This recipe is the v0 bridge from public corpora such as
LibriSpeech, AISHELL-1, WenetSpeech, and Common Voice into the paper data layer.

## ASR-to-Turn Weak Labels

Public ASR corpora normally do not contain full-duplex turn labels. Use
`asr-to-turn` to bootstrap weak complete windows from utterance-level ASR
manifests:

```bash
stable-asr asr-to-turn \
  --input runs/asr_manifest.jsonl \
  --output runs/asr_turn.jsonl \
  --window-sec 2.0
```

Add `--include-incomplete` to emit truncated incomplete negatives from each
utterance:

```bash
stable-asr asr-to-turn \
  --input runs/asr_manifest.jsonl \
  --output runs/asr_turn_with_negatives.jsonl \
  --include-incomplete \
  --incomplete-ratio 0.65
```

These records are explicitly marked with `metadata.derived_from=asr_manifest`
and `scenario=asr_weak_complete` or `asr_weak_incomplete`. They are useful for
bootstrapping endpointing baselines, not a substitute for real interruption,
backchannel, or wait annotations.

For the common case, `bootstrap-turn-data` runs metadata preparation,
ASR-to-turn conversion, and train/dev/test splitting in one command:

```bash
stable-asr bootstrap-turn-data \
  --input examples/data/asr_metadata.tsv \
  --output-dir runs/bootstrap_turn \
  --audio-root examples/data \
  --sample-rate 16000 \
  --include-incomplete
```

It writes `asr_manifest.jsonl`, `turn_manifest.jsonl`, split manifests under
`splits/`, `bootstrap_summary.json`, and `BOOTSTRAP_TURN_DATA.md` with next
commands for validation and NanoTurn training.

## Audio Audit

Field validation does not prove the referenced audio exists. Use `audit-audio`
before training or benchmarking:

```bash
stable-asr audit-audio \
  --kind turn \
  --manifest runs/splits/turn_train.jsonl \
  --audio-root data/turn_audio

stable-asr audit-audio \
  --kind asr \
  --manifest runs/asr_manifest.jsonl \
  --duration-tolerance-sec 0.10
```

The built-in reader inspects WAV files for sample-rate and duration mismatch.
Non-WAV files such as FLAC are checked for existence by default; add
`--require-inspectable` when every record must be inspectable by the built-in
WAV reader.

## Streaming ASR Fixture Schema

`eval-streaming-asr` accepts JSONL records with final transcript, partial
hypotheses, endpoint timing, and optional word timestamps:

```json
{
  "id": "utt_001",
  "reference": "what is the weather",
  "final_text": "what is the weather",
  "audio_duration": 2.0,
  "processing_time": 0.5,
  "speech_end_time": 1.8,
  "endpoint_time": 2.1,
  "reference_word_timestamps": [
    {"word": "what", "start": 0.10, "end": 0.35}
  ],
  "word_timestamps": [
    {"word": "what", "start": 0.12, "end": 0.36}
  ],
  "partials": [
    {"time": 0.4, "text": "what"},
    {"time": 2.1, "text": "what is the weather", "is_final": true}
  ]
}
```

`compare-streaming-asr` wraps these files in `TranscriptJSONLAdapter` objects.
`convert-asr-transcript` normalizes external transcript JSONL into this schema:

```bash
stable-asr convert-asr-transcript \
  --schema whisper \
  --input tests/fixtures/whisper_transcript_sample.jsonl \
  --output /tmp/stable-asr-whisper-streaming.jsonl

stable-asr convert-asr-transcript \
  --schema funasr \
  --input tests/fixtures/funasr_transcript_sample.jsonl \
  --output /tmp/stable-asr-funasr-streaming.jsonl
```

Supported external schemas currently include Whisper-style `segments`/`words`
and FunASR-style `sentence_info`/`timestamp` rows. Future WeNet, NeMo, and
ESPnet adapters should implement the same `StreamingASRAdapter.load_records()`
evaluation protocol.
