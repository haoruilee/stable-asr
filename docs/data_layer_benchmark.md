# Stable-ASR Data Layer Benchmark

Generated on 2026-06-01.

## stable-worldmodel Pattern

stable-worldmodel's data layer has three relevant design choices:

- one format registry for record, load, and conversion paths
- Lance as the default training-oriented backend for fast indexed reads
- reproducible throughput and storage benchmarks across local and remote sources

Stable-ASR mirrors this at two levels:

- turn manifest registry: JSONL, Parquet, Lance
- audio-window cache: source WAV baseline, Parquet materialized windows, Lance materialized windows

## Local Results

Commands:

```bash
.venv/bin/python -m stable_asr.cli benchmark-data \
  --dataset runs/final/turn_train.jsonl \
  --output-dir runs/final/data_bench \
  --formats jsonl parquet lance \
  --sample-count 10000 \
  --json-output runs/final/reports/data_benchmark.json

.venv/bin/python -m stable_asr.cli benchmark-audio-windows \
  --dataset runs/final/voiceworld_real.jsonl \
  --output-dir runs/final/audio_window_bench_correctness \
  --formats source_wav parquet lance \
  --sample-count 10000 \
  --correctness-sample-count 10000 \
  --json-output runs/final/reports/audio_window_benchmark_correctness.json

.venv/bin/python -m stable_asr.cli benchmark-train-features \
  --dataset runs/final/voiceworld_real.jsonl \
  --output-dir runs/final/train_feature_bench \
  --formats source_audio source_audio_file_cache parquet lance \
  --sample-count 10000 \
  --correctness-sample-count 10000 \
  --json-output runs/final/reports/train_feature_benchmark.json

.venv/bin/python -m stable_asr.cli benchmark-train-features \
  --dataset runs/final/voiceworld_real.jsonl \
  --output-dir runs/final/train_feature_bench_10k_correctness \
  --formats source_audio source_audio_file_cache parquet lance \
  --sample-count 10000 \
  --correctness-sample-count 10000 \
  --json-output runs/final/reports/train_feature_benchmark_10k_correctness.json

.venv/bin/python -m stable_asr.cli benchmark-train-features \
  --dataset runs/final/voiceworld_real.jsonl \
  --output-dir runs/final/train_feature_bench_100k_cached_correctness \
  --formats parquet lance \
  --sample-count 100000 \
  --correctness-sample-count 10000 \
  --json-output runs/final/reports/train_feature_benchmark_100k_cached_correctness.json
```

Manifest-only benchmark on `runs/final/turn_train.jsonl`:

| format | records | random samples/s | size |
| --- | ---: | ---: | ---: |
| JSONL | 4906 | 276707.4 | 5065096 bytes |
| Parquet | 4906 | 112482.3 | 578659 bytes |
| Lance | 4906 | 37288.6 | 1512012 bytes |

This metadata-only case is too small to show Lance's strengths; the JSONL path loads the full file once and samples from memory.

Audio-window benchmark on `runs/final/voiceworld_real.jsonl`:

| format | records | random samples/s | speedup vs source WAV | correctness samples | max abs error | size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source WAV | 180 | 2720.2 | 1.0x | 0 | 0.0 | 5883120 bytes |
| Parquet cache | 180 | 213919.0 | 78.6x | 10000 | 0.0 | 2359584 bytes |
| Lance cache | 180 | 27349.4 | 10.1x | 10000 | 0.0 | 647193 bytes |

Interpretation:

- For small local audio-window data, Parquet is fastest because it can read one compact column into memory and select rows there.
- Lance still removes per-sample WAV open/decode cost and is 10.1x faster than the source WAV baseline while using the smallest cache footprint in this run.
- The next paper-grade benchmark should repeat this on larger real ASR/turn corpora, with cold-cache and warm-cache variants, because stable-worldmodel's Lance advantage is most relevant for larger random-access and remote-storage workloads.
- Parquet and Lance both reloaded 10000 sampled source WAV windows and matched
  the materialized cache with `max_abs_error=0.0`.

Training log-mel feature benchmark on the same VoiceWorld records:

| format | records | sample-count | random samples/s | speedup vs source audio | correctness samples | max abs error | size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source audio | 180 | 10000 | 383.2 | 1.0x | 0 | 0.0 | n/a |
| source audio with file cache | 180 | 10000 | 515.9 | 1.3x | 0 | 0.0 | n/a |
| Parquet log-mel cache | 180 | 10000 | 384066.7 | 1002.3x | 10000 | 0.0 | 54751 bytes |
| Lance log-mel cache | 180 | 10000 | 1499475.6 | 3913.2x | 10000 | 0.0 | 76185 bytes |

The train-time cache stores the 32-dimensional NanoTurn log-mel vector by
record id. This is the most aggressive v0 acceleration path because repeated
training runs skip audio open, decode, window slicing, and STFT. The cached
feature path is now exposed through `train-turn --feature-cache`. The correctness run
also removes Python row materialization from cached reads by converting Arrow
columns directly into NumPy arrays before constructing the Torch tensor.
`sample_seconds` measures the training-time random read path; one-time cache
write cost and correctness recomputation are reported separately so the speedup
does not hide preprocessing cost or skip correctness checks.

Cached-only stress run:

| format | records | sample-count | random samples/s | correctness samples | max abs error | size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parquet log-mel cache | 180 | 100000 | 1449091.4 | 10000 | 0.0 | 54751 bytes |
| Lance log-mel cache | 180 | 100000 | 2007069.0 | 10000 | 0.0 | 76185 bytes |

The 10k result is the main paper-facing local result because it includes the
uncached source-audio baseline. On this workload, cached Lance training windows
are `3913.2x` faster than repeatedly opening and decoding source audio. The
100k cached-only run checks that the cached path stays stable under heavier
random sampling; it does not report a source-audio speedup because source
decoding was intentionally skipped to keep the local run bounded.
