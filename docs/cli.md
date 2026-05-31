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
stable-asr prepare-voiceworld --input examples/data/voiceworld_metadata.tsv --output runs/voiceworld_demo.jsonl
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
stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --validate-only --require-input-manifest --min-adapters 2
stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --json-output runs/final/reports/asr_command_compare.json
```

## References

```bash
stable-asr model-registry --registry configs/models/stable_asr_models.json --validate-only
stable-asr model-registry --output runs/MODELS.md
stable-asr asr-collections --registry configs/references/asr_collections.json --validate-only
stable-asr asr-collections --output runs/ASR_COLLECTIONS.md
stable-asr asr-collections --format paper-markdown --output runs/ASR_REFERENCES.md
stable-asr asr-collections --format bibtex --output runs/ASR_REFERENCES.bib
stable-asr asr-collections --audit-coverage --require-priority p0 --require-priority p1
stable-asr asr-collections --audit-readiness --output runs/ASR_COLLECTION_READINESS.md
```

## Paper Artifacts

```bash
stable-asr reproduce-paper --config configs/paper/paper_smoke.json
stable-asr paper-bundle --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts
stable-asr final-config --config configs/final/paper_final.json --prepare-inputs
stable-asr final-config --config configs/final/paper_final.json --prepare-corpora
stable-asr final-config --config configs/final/paper_final.json --prepare-asr-eval-manifest
stable-asr final-config --config configs/final/paper_final.json --bootstrap-turn-splits
stable-asr final-config --config configs/final/paper_final.json --prepare-external-predictions
stable-asr final-config --config configs/final/paper_final.json --prepare-voiceworld-real
stable-asr final-config --config configs/final/paper_final.json --audit-voiceworld-real --scenario-suite configs/scenarios/stable_asr_voiceworld_v0.json
stable-asr final-config --config configs/final/paper_final.json --audit-asr-commands
stable-asr final-config --config configs/final/paper_final.json --prepare-asr-transcript-conversions
stable-asr final-config --config configs/final/paper_final.json --check-files
stable-asr final-config --config configs/final/paper_final.json --plan-missing --output runs/final/FINAL_RUN_ACTION_PLAN.md
stable-asr final-inputs --registry configs/final/input_collections.json --output runs/final/FINAL_INPUT_COLLECTIONS.md
stable-asr paper-evidence-matrix --output runs/final/FINAL_EVIDENCE_MATRIX.md
stable-asr make-card model --input configs/models/stable_asr_models.json --model-id nanoturn_pico --metrics runs/final/nanoturn/metrics.json --output runs/final/MODEL_CARD.md
stable-asr eval-scenario --dataset runs/final/voiceworld_real.jsonl --checkpoint runs/final/nanoturn/checkpoint.pt --json-output runs/final/reports/scenarios.json
stable-asr final-results --config configs/final/paper_final.json --output runs/final/paper_results.json
stable-asr leaderboard-export --results runs/final/paper_results.json --output runs/final/leaderboard.jsonl
stable-asr leaderboard-validate --input runs/final/leaderboard.jsonl --require-complete-suite --output runs/final/LEADERBOARD_VALIDATION.md
stable-asr leaderboard-report --input runs/final/leaderboard.jsonl --require-complete-suite --output runs/final/LEADERBOARD_REPORT.md
stable-asr paper-artifact-integrity --manifest runs/paper/smoke/artifacts/artifact_hashes.json --root runs/paper/smoke/artifacts
stable-asr benchmark-suite --suite runs/paper/smoke/artifacts/benchmark_suite.json --artifacts-dir runs/paper/smoke/artifacts --validate-only
stable-asr paper-archive --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/artifacts.tar.gz
stable-asr paper-archive-verify --archive runs/paper/smoke/artifacts.tar.gz
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke
stable-asr paper-release-audit --repo-root . --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts
```
