# Data API

Core data entry points:

- `stable_asr.data.manifest.TurnManifestRecord`
- `stable_asr.data.asr_manifest.ASRManifestRecord`
- `stable_asr.data.registry.load_turn_records`
- `stable_asr.data.registry.write_turn_records`
- `stable_asr.data.registry.convert_turn_manifest`
- `stable_asr.data.benchmark.benchmark_data_formats`

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
