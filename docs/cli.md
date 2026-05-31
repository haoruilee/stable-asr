# CLI

Stable-ASR exposes one command: `stable-asr`.

## Health And Roadmap

```bash
stable-asr doctor
stable-asr doctor --check-final-files
stable-asr roadmap-status --roadmap configs/roadmap/stable_asr_roadmap.json
```

## Turn Data

```bash
stable-asr validate-manifest examples/data/turn_demo.jsonl
stable-asr inspect-manifest examples/data/turn_demo.jsonl
stable-asr split-turn-data --input examples/data/turn_demo.jsonl --output-dir runs/splits
stable-asr audit-turn-splits --train runs/splits/turn_train.jsonl --dev runs/splits/turn_dev.jsonl --test runs/splits/turn_test.jsonl
stable-asr benchmark-data --dataset examples/data/turn_demo.jsonl --output-dir runs/data_bench --formats jsonl parquet lance
```

## Baselines And NanoTurn

```bash
stable-asr eval-turn --dataset examples/data/turn_demo.jsonl --baseline vad_pause
stable-asr compare-turn --dataset examples/data/turn_demo.jsonl --baseline vad_pause --baseline text_turn
stable-asr train-turn --dataset examples/data/turn_demo.jsonl --output-dir runs/nanoturn
stable-asr eval-turn --dataset examples/data/turn_demo.jsonl --checkpoint runs/nanoturn/checkpoint.pt
stable-asr export-turn-onnx --checkpoint runs/nanoturn/checkpoint.pt --output runs/nanoturn/nanoturn.onnx
```

## Streaming ASR

```bash
stable-asr eval-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl
stable-asr compare-streaming-asr --input balanced=tests/fixtures/streaming_asr_sample.jsonl --input fast_unstable=tests/fixtures/streaming_asr_fast_unstable_sample.jsonl
stable-asr sweep-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl --chunks-ms 160 320 640 --lookahead-ms 0 160
stable-asr eval-asr-command --name my_asr --command "python your_export.py --output {output}" --output runs/my_asr.jsonl
```

## Paper Artifacts

```bash
stable-asr reproduce-paper --config configs/paper/paper_smoke.json
stable-asr paper-bundle --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts
stable-asr final-config --config configs/final/paper_final.json --prepare-corpora
stable-asr final-config --config configs/final/paper_final.json --check-files
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke
stable-asr paper-release-audit --repo-root . --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts
```
