# Data API

Core data entry points:

- `stable_asr.data.manifest.TurnManifestRecord`
- `stable_asr.data.asr_manifest.ASRManifestRecord`
- `stable_asr.data.registry.load_turn_records`
- `stable_asr.data.registry.write_turn_records`
- `stable_asr.data.registry.convert_turn_manifest`
- `stable_asr.data.recipes.prepare_voiceworld_manifest`
- `stable_asr.data.benchmark.benchmark_data_formats`
- `stable_asr.data.audio_window_cache.materialize_audio_windows`
- `stable_asr.data.audio_window_cache.benchmark_audio_window_formats`
- `stable_asr.train.feature_cache.write_logmel_feature_cache`
- `stable_asr.train.feature_cache.benchmark_train_feature_cache`

Supported turn manifest backends:

- JSONL: zero-dependency core backend
- Parquet: `stable-asr[data]`
- Lance: `stable-asr[lance]`

Example:

```python
from stable_asr.data.registry import load_turn_records
from stable_asr.data.benchmark import benchmark_data_formats

records = load_turn_records("examples/data/turn_demo.jsonl")
rows = benchmark_data_formats(
    records,
    output_dir="runs/data_bench",
    formats=["jsonl", "parquet", "lance"],
)
```

Audio-window cache benchmark:

```python
from stable_asr.data.audio_window_cache import benchmark_audio_window_formats

records = load_turn_records("runs/final/voiceworld_real.jsonl")
rows = benchmark_audio_window_formats(
    records,
    output_dir="runs/final/audio_window_bench",
    formats=["source_wav", "parquet", "lance"],
    sample_count=5000,
)
```

`source_wav` is the baseline that opens the original WAV file for every random
sample. `parquet` and `lance` first materialize fixed turn windows into a
columnar cache and then benchmark random row retrieval. Benchmark rows include
`correctness_sample_count`, `max_abs_error_vs_source`, and
`allclose_to_source` so speedups are tied to source-window equivalence checks.

Training feature cache:

```python
from stable_asr.train.feature_cache import benchmark_train_feature_cache

rows = benchmark_train_feature_cache(
    records,
    output_dir="runs/final/train_feature_bench",
    formats=["source_audio", "source_audio_file_cache", "parquet", "lance"],
    sample_count=1000,
)
```

This cache stores the 32-dimensional NanoTurn log-mel vector by record id, so
subsequent training runs can skip audio open, decode, slicing, and STFT work.
Benchmark rows also include a cached-feature correctness check:
`correctness_sample_count`, `max_abs_error_vs_source`, and
`allclose_to_source`.
