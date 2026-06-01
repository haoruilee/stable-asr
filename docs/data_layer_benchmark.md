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
  --output-dir runs/final/audio_window_bench \
  --formats source_wav parquet lance \
  --sample-count 10000 \
  --json-output runs/final/reports/audio_window_benchmark.json

.venv/bin/python -m stable_asr.cli benchmark-train-features \
  --dataset runs/final/voiceworld_real.jsonl \
  --output-dir runs/final/train_feature_bench \
  --formats source_audio source_audio_file_cache parquet lance \
  --sample-count 1000 \
  --json-output runs/final/reports/train_feature_benchmark.json

.venv/bin/python -m stable_asr.cli benchmark-train-features \
  --dataset runs/final/voiceworld_real.jsonl \
  --output-dir runs/final/train_feature_bench_10k \
  --formats source_audio source_audio_file_cache parquet lance \
  --sample-count 10000 \
  --json-output runs/final/reports/train_feature_benchmark_10k.json

.venv/bin/python -m stable_asr.cli benchmark-train-features \
  --dataset runs/final/voiceworld_real.jsonl \
  --output-dir runs/final/train_feature_bench_100k_cached \
  --formats parquet lance \
  --sample-count 100000 \
  --json-output runs/final/reports/train_feature_benchmark_100k_cached.json
```

Manifest-only benchmark on `runs/final/turn_train.jsonl`:

| format | records | random samples/s | size |
| --- | ---: | ---: | ---: |
| JSONL | 4906 | 276707.4 | 5065096 bytes |
| Parquet | 4906 | 112482.3 | 578659 bytes |
| Lance | 4906 | 37288.6 | 1512012 bytes |

This metadata-only case is too small to show Lance's strengths; the JSONL path loads the full file once and samples from memory.

Audio-window benchmark on `runs/final/voiceworld_real.jsonl`:

| format | records | random samples/s | speedup vs source WAV | size |
| --- | ---: | ---: | ---: | ---: |
| source WAV | 180 | 2612.8 | 1.0x | 5883120 bytes |
| Parquet cache | 180 | 215109.8 | 82.3x | 2359584 bytes |
| Lance cache | 180 | 26616.1 | 10.2x | 647193 bytes |

Interpretation:

- For small local audio-window data, Parquet is fastest because it can read one compact column into memory and select rows there.
- Lance still removes per-sample WAV open/decode cost and is 10.2x faster than the source WAV baseline while using the smallest cache footprint in this run.
- The next paper-grade benchmark should repeat this on larger real ASR/turn corpora, with cold-cache and warm-cache variants, because stable-worldmodel's Lance advantage is most relevant for larger random-access and remote-storage workloads.

Training log-mel feature benchmark on the same VoiceWorld records:

| format | records | sample-count | random samples/s | speedup vs source audio | size |
| --- | ---: | ---: | ---: | ---: | ---: |
| source audio | 180 | 1000 | 91.2 | 1.0x | n/a |
| source audio with file cache | 180 | 1000 | 219.6 | 2.4x | n/a |
| Parquet log-mel cache | 180 | 1000 | 35188.2 | 385.8x | 54751 bytes |
| Lance log-mel cache | 180 | 1000 | 104082.5 | 1141.1x | 76184 bytes |
| source audio | 180 | 10000 | 381.3 | 1.0x | n/a |
| source audio with file cache | 180 | 10000 | 705.0 | 1.8x | n/a |
| Parquet log-mel cache | 180 | 10000 | 113104.4 | 296.7x | 54751 bytes |
| Lance log-mel cache | 180 | 10000 | 144444.3 | 378.9x | 76185 bytes |

The train-time cache stores the 32-dimensional NanoTurn log-mel vector by
record id. This is the most aggressive v0 acceleration path because repeated
training runs skip audio open, decode, window slicing, and STFT. The cached
feature path is now exposed through `train-turn --feature-cache`.

Cached-only stress run:

| format | records | sample-count | random samples/s | size |
| --- | ---: | ---: | ---: | ---: |
| Parquet log-mel cache | 180 | 100000 | 125620.1 | 54751 bytes |
| Lance log-mel cache | 180 | 100000 | 131653.2 | 76185 bytes |

The 10k result is the main paper-facing local result because it includes the
uncached source-audio baseline. On this workload, cached Lance training windows
are `378.9x` faster than repeatedly opening and decoding source audio. The 100k
cached-only run checks that the cached path stays stable under heavier random
sampling; it does not report a source-audio speedup because source decoding was
intentionally skipped to keep the local run bounded.
