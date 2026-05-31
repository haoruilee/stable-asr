# Stable-ASR Documentation

Stable-ASR is a reproducible research platform for real-time ASR systems and
full-duplex turn-taking evaluation.

The repository is intentionally scoped around the system layer between
streaming ASR and voice-agent control:

- turn/action manifests
- endpointing and turn-taking baselines
- NanoTurn checkpoint training and evaluation
- VoiceWorld scenario evaluation
- failure-case mining for interaction case studies
- streaming ASR metrics
- streaming ASR failure mining
- utterance-level ASR corpus manifests and metadata-table recipes
- audio file audits for turn and ASR manifests
- adapter registry for baselines, converters, command-backed ASR wrappers, and external-system templates
- curated upstream ASR reference collections for adapter and benchmark planning
- policy search
- paper parity audit for stable-worldmodel-style platform-paper gaps
- final-scale experiment runbook for real paper execution
- final-run config template for paper-scale directory, corpus, and artifact planning
- final-run scaffold for directories and README hints without fake data
- final-run file audit for missing real paper inputs
- machine-readable roadmap status for milestone evidence and planned final-scale work
- repository doctor for environment, config, and final-input readiness checks
- paper tables, figures, bundles, audits, and drafts

Stable-ASR is not intended to replace full ASR toolkits such as ESPnet, FunASR,
WeNet, or NeMo. It provides the missing reproducibility layer for comparing
real-time behavior, latency, policies, and interaction failures.

## Core Commands

```bash
stable-asr doctor
stable-asr doctor --check-release-env
stable-asr doctor --check-final-files
stable-asr roadmap-status --roadmap configs/roadmap/stable_asr_roadmap.json
stable-asr validate-manifest examples/data/turn_demo.jsonl
stable-asr convert examples/data/turn_demo.jsonl runs/turn_demo.lance
stable-asr profile-turn-data --dataset examples/data/turn_demo.jsonl --report runs/turn_profile.md
stable-asr benchmark-data --dataset examples/data/turn_demo.jsonl --output-dir runs/data_bench --formats jsonl parquet lance --sample-count 16
stable-asr audit-turn-splits --train runs/splits/turn_train.jsonl --dev runs/splits/turn_dev.jsonl --test runs/splits/turn_test.jsonl
stable-asr data-sources --registry configs/datasets/stable_asr_sources.json --validate-only
stable-asr adapter-registry --registry configs/adapters/stable_asr_adapters.json --validate-only
stable-asr asr-collections --registry configs/references/asr_collections.json --validate-only
stable-asr asr-collections --audit-coverage
stable-asr scenario-suite --suite configs/scenarios/stable_asr_voiceworld_v0.json --validate-only
stable-asr prepare-asr-manifest --input examples/data/asr_metadata.tsv --output runs/asr_manifest.jsonl --audio-root examples/data --sample-rate 16000
stable-asr prepare-public-asr --corpus librispeech --input-dir data/librispeech/LibriSpeech/dev-clean --output runs/final/librispeech_dev_clean/asr_manifest.jsonl
stable-asr prepare-public-asr --corpus wenetspeech --input-dir data/wenetspeech/WenetSpeech --split dev --output runs/final/wenetspeech_dev/asr_manifest.jsonl
stable-asr prepare-public-asr --corpus common_voice --input-dir data/common_voice/en --split dev --output runs/final/common_voice_en_dev/asr_manifest.jsonl
stable-asr validate-asr-manifest runs/asr_manifest.jsonl
stable-asr inspect-asr-manifest runs/asr_manifest.jsonl
stable-asr asr-to-turn --input runs/asr_manifest.jsonl --output runs/asr_turn.jsonl --include-incomplete
stable-asr bootstrap-turn-data --input examples/data/asr_metadata.tsv --output-dir runs/bootstrap_turn --audio-root examples/data --sample-rate 16000 --include-incomplete
stable-asr audit-audio --kind asr --manifest runs/asr_manifest.jsonl
stable-asr train-turn --dataset examples/data/turn_demo.jsonl --output-dir runs/nanoturn
stable-asr eval-turn --dataset examples/data/turn_demo.jsonl --baseline vad_pause
stable-asr predict-turn --dataset examples/data/turn_demo.jsonl --baseline text_turn --output runs/text_turn_predictions.jsonl
stable-asr validate-turn-predictions --dataset examples/data/turn_demo.jsonl --predictions runs/text_turn_predictions.jsonl
stable-asr compare-turn --dataset examples/data/turn_demo.jsonl --baseline vad_pause --baseline text_turn --predictions oracle=tests/fixtures/turn_predictions_sample.jsonl --report runs/turn_compare.md
stable-asr compare-turn-splits --train runs/splits/turn_train.jsonl --dev runs/splits/turn_dev.jsonl --test runs/splits/turn_test.jsonl --baseline vad_pause --baseline text_turn --report runs/turn_split_compare.md
stable-asr eval-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl
stable-asr compare-streaming-asr --input balanced=tests/fixtures/streaming_asr_sample.jsonl --input fast_unstable=tests/fixtures/streaming_asr_fast_unstable_sample.jsonl
stable-asr sweep-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl --chunks-ms 160 320 640 --lookahead-ms 0 160
stable-asr convert-asr-transcript --schema whisper --input tests/fixtures/whisper_transcript_sample.jsonl --output runs/whisper_streaming.jsonl
stable-asr eval-asr-command --name my_asr --command "python your_asr_export.py --output {output}" --output runs/my_asr_streaming.jsonl
stable-asr compare-asr-commands --config examples/configs/asr_command_compare_demo.json --report runs/asr_command_compare.md
stable-asr eval-scenario --episodes 21 --seed 0 --baseline vad_pause
stable-asr reproduce-paper --config configs/paper/paper_smoke.json
stable-asr paper-bundle --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts
stable-asr paper-status --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts
stable-asr paper-case-studies --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts
stable-asr paper-claim-audit --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output-dir runs/paper/smoke/artifacts
stable-asr paper-parity-audit --checklist configs/paper/paper_parity_checklist.json --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts
stable-asr final-experiments --registry configs/paper/final_experiments.json --output runs/paper/smoke/artifacts/FINAL_EXPERIMENTS.md
stable-asr final-config --config configs/final/paper_final.json --output runs/paper/smoke/artifacts/FINAL_RUN_CONFIG.md
stable-asr final-config --config configs/final/paper_final.json --scaffold
stable-asr final-config --config configs/final/paper_final.json --prepare-inputs
stable-asr final-config --config configs/final/paper_final.json --prepare-corpora
stable-asr final-config --config configs/final/paper_final.json --prepare-asr-eval-manifest
stable-asr final-config --config configs/final/paper_final.json --bootstrap-turn-splits
stable-asr final-config --config configs/final/paper_final.json --prepare-external-predictions
stable-asr final-config --config configs/final/paper_final.json --audit-voiceworld-real --scenario-suite configs/scenarios/stable_asr_voiceworld_v0.json
stable-asr final-config --config configs/final/paper_final.json --audit-asr-commands
stable-asr paper-release-audit --repo-root . --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts
stable-asr benchmark-suite --suite configs/benchmarks/stable_asr_v0.json --validate-only
stable-asr benchmark-suite --suite configs/benchmarks/stable_asr_v0.json --results runs/paper/smoke/paper_results.json --validate-only
```

## Documents

- [Quick start](quick_start.md)
- [CLI](cli.md)
- [Baselines](baselines.md)
- [Paper pipeline](paper_pipeline.md)
- [Release gates](release_gates.md)
- [Data schema](schema.md)
- [ASR reference collections](asr_collections.md)
- API: [Data](api/data.md), [Turn](api/turn.md), [Scenarios](api/scenarios.md), [Paper](api/paper.md)
- Guides: [External ASR adapters](guides/adapters.md), [Release smoke](guides/release_smoke.md)
