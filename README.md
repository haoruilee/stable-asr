# Stable-ASR

Stable-ASR is a reproducible research platform for real-time ASR systems.

The project starts with turn-taking and endpointing: the missing control layer
between streaming ASR and full-duplex voice agents. NanoTurn is the first
built-in model family.

The long-term target is a stable-worldmodel-style platform paper:

```text
Stable-ASR: A Platform for Reproducible Real-Time ASR and
Full-Duplex Turn-Taking Research and Evaluation
```

## Scope

Stable-ASR is not another general ASR toolkit. The first releases focus on:

- standardized ASR and turn-taking manifests
- NanoTurn baseline models
- endpointing and turn-taking evaluation
- streaming ASR metrics beyond WER/CER
- scenario-based robustness testing
- latency and deployment reports
- paper-ready benchmark scripts, tables, and figures

## Documentation

See [docs/index.md](docs/index.md) for the project documentation, including the
paper pipeline, release gates, and manifest schema.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -e ".[train]"  # optional: NanoTurn training
python -m pip install -e ".[lance]"  # optional: Lance data backend
stable-asr doctor
stable-asr doctor --check-final-files
stable-asr validate-manifest examples/data/turn_demo.jsonl
stable-asr labels
stable-asr eval-turn --dataset examples/data/turn_demo.jsonl --baseline vad_pause
stable-asr eval-turn --dataset examples/data/turn_demo.jsonl --baseline text_turn
stable-asr eval-turn --dataset examples/data/turn_demo.jsonl --predictions tests/fixtures/turn_predictions_sample.jsonl
stable-asr predict-turn --dataset examples/data/turn_demo.jsonl --baseline text_turn --output /tmp/stable-asr-text-turn-predictions.jsonl
stable-asr validate-turn-predictions --dataset examples/data/turn_demo.jsonl --predictions /tmp/stable-asr-text-turn-predictions.jsonl
stable-asr compare-turn --dataset examples/data/turn_demo.jsonl --baseline vad_pause --baseline text_turn --predictions oracle=tests/fixtures/turn_predictions_sample.jsonl --report /tmp/stable-asr-turn-compare.md
stable-asr compare-turn-splits --train /tmp/stable-asr-splits/turn_train.jsonl --dev /tmp/stable-asr-splits/turn_dev.jsonl --test /tmp/stable-asr-splits/turn_test.jsonl --baseline vad_pause --baseline text_turn --report /tmp/stable-asr-turn-splits.md
stable-asr make-synthetic-turn-data --output /tmp/stable-asr-synth.jsonl --episodes 10 --seed 42 --write-audio
stable-asr inspect-manifest examples/data/turn_demo.jsonl
stable-asr profile-turn-data --dataset examples/data/turn_demo.jsonl --report /tmp/stable-asr-turn-profile.md
stable-asr split-turn-data --input examples/data/turn_demo.jsonl --output-dir /tmp/stable-asr-splits --train-ratio 0.5 --dev-ratio 0.25 --test-ratio 0.25 --seed 7
stable-asr audit-turn-splits --train /tmp/stable-asr-splits/turn_train.jsonl --dev /tmp/stable-asr-splits/turn_dev.jsonl --test /tmp/stable-asr-splits/turn_test.jsonl
stable-asr convert examples/data/turn_demo.jsonl /tmp/stable-asr-copy.jsonl
stable-asr convert examples/data/turn_demo.jsonl /tmp/stable-asr-copy.parquet
stable-asr convert examples/data/turn_demo.jsonl /tmp/stable-asr-copy.lance
stable-asr convert-external --schema easyturn --input tests/fixtures/easyturn_sample.jsonl --output /tmp/stable-asr-easyturn.jsonl
stable-asr convert-external --schema full_duplex_bench --input tests/fixtures/full_duplex_bench_sample.jsonl --output /tmp/stable-asr-fdb.jsonl
stable-asr convert-external --schema smart_turn --input tests/fixtures/smart_turn_manifest_sample.jsonl --output /tmp/stable-asr-smartturn.jsonl
stable-asr convert-predictions --schema easyturn --input tests/fixtures/easyturn_predictions_sample.jsonl --output /tmp/stable-asr-easyturn-predictions.jsonl
stable-asr convert-asr-transcript --schema whisper --input tests/fixtures/whisper_transcript_sample.jsonl --output /tmp/stable-asr-whisper-streaming.jsonl
stable-asr convert-asr-transcript --schema funasr --input tests/fixtures/funasr_transcript_sample.jsonl --output /tmp/stable-asr-funasr-streaming.jsonl
stable-asr benchmark-data --dataset examples/data/turn_demo.jsonl --output-dir /tmp/stable-asr-data-bench --formats jsonl parquet lance --sample-count 16
stable-asr data-sources --registry configs/datasets/stable_asr_sources.json --validate-only
stable-asr data-sources --output /tmp/stable-asr-paper/DATA_SOURCES.md
stable-asr adapter-registry --registry configs/adapters/stable_asr_adapters.json --validate-only
stable-asr adapter-registry --output /tmp/stable-asr-paper/ADAPTERS.md
stable-asr asr-collections --registry configs/references/asr_collections.json --validate-only
stable-asr asr-collections --output /tmp/stable-asr-paper/ASR_COLLECTIONS.md
stable-asr asr-collections --audit-coverage --output /tmp/stable-asr-paper/ASR_COLLECTION_COVERAGE.md
stable-asr scenario-suite --suite configs/scenarios/stable_asr_voiceworld_v0.json --validate-only
stable-asr scenario-suite --output /tmp/stable-asr-paper/SCENARIO_SUITE.md
stable-asr prepare-asr-manifest --input examples/data/asr_metadata.tsv --output /tmp/stable-asr-asr-manifest.jsonl --audio-root examples/data --sample-rate 16000
stable-asr validate-asr-manifest /tmp/stable-asr-asr-manifest.jsonl
stable-asr inspect-asr-manifest /tmp/stable-asr-asr-manifest.jsonl
stable-asr asr-to-turn --input /tmp/stable-asr-asr-manifest.jsonl --output /tmp/stable-asr-asr-turn.jsonl --include-incomplete
stable-asr bootstrap-turn-data --input examples/data/asr_metadata.tsv --output-dir /tmp/stable-asr-bootstrap --audio-root examples/data --sample-rate 16000 --include-incomplete
stable-asr audit-audio --kind turn --manifest /tmp/stable-asr-synth.jsonl
stable-asr benchmark-turn --dataset examples/data/turn_demo.jsonl --baseline text_turn --warmup 0 --repeat 3 --report /tmp/stable-asr-turn-benchmark.md
stable-asr train-turn --dataset examples/data/turn_demo.jsonl --output-dir /tmp/stable-asr-nanoturn --epochs 20
stable-asr eval-turn --dataset examples/data/turn_demo.jsonl --checkpoint /tmp/stable-asr-nanoturn/checkpoint.pt
stable-asr export-turn-onnx --checkpoint /tmp/stable-asr-nanoturn/checkpoint.pt --output /tmp/stable-asr-nanoturn/nanoturn.onnx
stable-asr train-turn --dataset /tmp/stable-asr-synth.jsonl --output-dir /tmp/stable-asr-audio --feature-source audio --epochs 5
stable-asr reproduce-paper --output-dir /tmp/stable-asr-paper --episodes 12 --seed 5
python scripts/reproduce_paper.py --config configs/paper/paper_smoke.json --skip-train
stable-asr paper-table baselines --results /tmp/stable-asr-paper/paper_results.json
stable-asr paper-table turn_benchmark --results /tmp/stable-asr-paper/paper_results.json
stable-asr paper-table data --results /tmp/stable-asr-paper/paper_results.json
stable-asr paper-table asr_manifest_recipe --results /tmp/stable-asr-paper/paper_results.json
stable-asr paper-table failure_cases --results /tmp/stable-asr-paper/paper_results.json
stable-asr eval-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl
stable-asr compare-streaming-asr --input balanced=tests/fixtures/streaming_asr_sample.jsonl --input fast_unstable=tests/fixtures/streaming_asr_fast_unstable_sample.jsonl
stable-asr sweep-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl --chunks-ms 160 320 640 --lookahead-ms 0 160
stable-asr eval-asr-command --name my_asr --command "python your_asr_export.py --output {output}" --output /tmp/stable-asr-command-transcript.jsonl
stable-asr compare-asr-commands --config examples/configs/asr_command_compare_demo.json --report /tmp/stable-asr-command-compare.md
stable-asr eval-scenario --episodes 15 --seed 3 --baseline vad_pause --report /tmp/stable-asr-scenario.md
stable-asr optimize-policy --dataset examples/data/turn_demo.jsonl --baseline vad_pause --output /tmp/stable-asr-policy.json
stable-asr paper-table streaming --results /tmp/stable-asr-paper/paper_results.json
stable-asr paper-table streaming_failures --results /tmp/stable-asr-paper/paper_results.json
stable-asr paper-table streaming_sweep --results /tmp/stable-asr-paper/paper_results.json
stable-asr paper-table asr_transcript_conversions --results /tmp/stable-asr-paper/paper_results.json
stable-asr paper-table scenarios --results /tmp/stable-asr-paper/paper_results.json
stable-asr paper-table policy --results /tmp/stable-asr-paper/paper_results.json
stable-asr paper-figure architecture --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/architecture.svg
stable-asr paper-figure api_flow --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/api_flow.svg
stable-asr paper-figure data_registry --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/data_registry.svg
stable-asr paper-figure voiceworld_timeline --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/voiceworld_timeline.svg
stable-asr paper-figure policy_state_machine --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/policy_state_machine.svg
stable-asr paper-figure robustness_heatmap --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/robustness_heatmap.svg
stable-asr paper-figure latency_quality_pareto --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/latency_quality_pareto.svg
stable-asr paper-figure baselines --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/baselines.svg
stable-asr paper-figure latency --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/latency.svg
stable-asr paper-figure data --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/data.svg
stable-asr paper-figure streaming --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/streaming.svg
stable-asr paper-figure scenarios --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/scenarios.svg
stable-asr paper-figure policy --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/policy.svg
stable-asr paper-bundle --results /tmp/stable-asr-paper/paper_results.json --output-dir /tmp/stable-asr-paper/artifacts
stable-asr paper-status --results /tmp/stable-asr-paper/paper_results.json --artifacts-dir /tmp/stable-asr-paper/artifacts --output /tmp/stable-asr-paper/artifacts/PAPER_STATUS.md
stable-asr paper-case-studies --results /tmp/stable-asr-paper/paper_results.json --output-dir /tmp/stable-asr-paper/artifacts
stable-asr paper-claim-audit --results /tmp/stable-asr-paper/paper_results.json --artifacts-dir /tmp/stable-asr-paper/artifacts --output-dir /tmp/stable-asr-paper/artifacts
stable-asr paper-parity-audit --checklist configs/paper/paper_parity_checklist.json --results /tmp/stable-asr-paper/paper_results.json --artifacts-dir /tmp/stable-asr-paper/artifacts --output /tmp/stable-asr-paper/artifacts/PAPER_PARITY.md
stable-asr final-experiments --registry configs/paper/final_experiments.json --output /tmp/stable-asr-paper/artifacts/FINAL_EXPERIMENTS.md
stable-asr final-config --config configs/final/paper_final.json --output /tmp/stable-asr-paper/artifacts/FINAL_RUN_CONFIG.md
stable-asr final-config --config configs/final/paper_final.json --scaffold
stable-asr leaderboard-export --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/leaderboard.jsonl --format jsonl
stable-asr benchmark-suite --suite configs/benchmarks/stable_asr_v0.json --validate-only
stable-asr benchmark-suite --suite configs/benchmarks/stable_asr_v0.json --results /tmp/stable-asr-paper/paper_results.json --validate-only
stable-asr benchmark-suite --output /tmp/stable-asr-paper/BENCHMARK_SUITE.md
stable-asr adapter-registry --registry configs/adapters/stable_asr_adapters.json --validate-only
stable-asr paper-parity-audit --checklist configs/paper/paper_parity_checklist.json --validate-only
stable-asr final-experiments --registry configs/paper/final_experiments.json --validate-only
stable-asr final-config --config configs/final/paper_final.json --validate-only
stable-asr paper-audit --results /tmp/stable-asr-paper/paper_results.json --artifacts-dir /tmp/stable-asr-paper/artifacts
stable-asr paper-draft --results /tmp/stable-asr-paper/paper_results.json --artifacts-dir /tmp/stable-asr-paper/artifacts --output /tmp/stable-asr-paper/PAPER_DRAFT.md
stable-asr paper-latex --results /tmp/stable-asr-paper/paper_results.json --artifacts-dir /tmp/stable-asr-paper/artifacts --output /tmp/stable-asr-paper/paper.tex
stable-asr make-card dataset --input examples/data/turn_demo.jsonl --output /tmp/stable-asr-dataset-card.md
stable-asr make-card experiment --input /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-experiment-card.md
stable-asr paper-release-audit --repo-root . --results /tmp/stable-asr-paper/paper_results.json --artifacts-dir /tmp/stable-asr-paper/artifacts --markdown-draft /tmp/stable-asr-paper/PAPER_DRAFT.md --latex-draft /tmp/stable-asr-paper/paper.tex --dataset-card /tmp/stable-asr-dataset-card.md --experiment-card /tmp/stable-asr-experiment-card.md
python -m pytest
```

Current M0 functionality:

- installable package
- `stable-asr` CLI
- `doctor` command for environment, optional dependency, config, and final-input readiness checks
- JSONL turn manifest validation
- canonical turn/action labels
- core turn dataclasses
- minimal turn metrics
- threshold/hysteresis turn policy
- Markdown report helper
- `eval-turn` baseline evaluation command
- `predict-turn` for exporting baseline/checkpoint predictions in the shared prediction manifest schema
- `validate-turn-predictions` for checking prediction schema, ID coverage, duplicate IDs, and extra predictions before benchmark submission
- `compare-turn` for same-dataset turn baseline/checkpoint/prediction comparison reports
- `compare-turn-splits` for train/dev/test turn benchmark comparison reports
- rule endpoint and VAD pause baselines
- text-only turn baseline via `TextTurnBaseline`
- external turn prediction JSONL adapter via `--predictions`
- seedable synthetic turn scenario manifest generation for incomplete pauses, backchannels, waits, interruptions, side conversations, ambient speech, noisy far-field speech, and code-switching
- deterministic synthetic WAV generation
- WAV loading and pooled log-spectral audio features
- data format registry with JSONL backend
- optional Parquet backend via `stable-asr[data]`
- optional Lance backend via `stable-asr[lance]`
- `inspect-manifest` and `convert` CLI commands
- `profile-turn-data` for label/scenario/duration distribution reports and training-readiness warnings
- `split-turn-data` for seedable train/dev/test turn manifest splits
- `audit-turn-splits` for detecting record, audio, or ASR/conversation group leakage across splits
- `convert-external` for EasyTurn-style, Full-Duplex-Bench-style, and SmartTurn-style JSONL manifests
- `convert-predictions` for generic, SmartTurn-style, and EasyTurn-style prediction JSONL
- `convert-asr-transcript` for Whisper-style and FunASR-style transcript JSONL into the normalized streaming ASR schema
- `benchmark-data` for paper-facing data-layer tables
- machine-readable data source registry in `configs/datasets/stable_asr_sources.json`
- `data-sources` for validating and rendering data source registries
- machine-readable adapter registry in `configs/adapters/stable_asr_adapters.json`
- `adapter-registry` for validating and rendering baseline, converter, command-backed, and external-system adapter entries
- machine-readable upstream ASR reference collection in `configs/references/asr_collections.json`
- `asr-collections` for validating and rendering top ASR project references
- `asr-collections --audit-coverage` for checking P0 reference coverage in the adapter registry
- machine-readable VoiceWorld scenario suite in `configs/scenarios/stable_asr_voiceworld_v0.json`
- `scenario-suite` for validating and rendering scenario suite definitions
- utterance-level ASR manifest schema and metadata-table recipe via `prepare-asr-manifest`
- `validate-asr-manifest` and `inspect-asr-manifest` for public ASR corpus manifests
- `asr-to-turn` for weakly labeled complete/incomplete turn windows from ASR utterance manifests
- `bootstrap-turn-data` for one-command ASR metadata to weak turn manifest and train/dev/test splits
- `audit-audio` for turn/ASR manifest file existence, WAV sample-rate, and WAV duration checks
- `benchmark-turn` for latency, throughput, RTF, and artifact-size reports
- optional NanoTurnPico/NanoTurnNano PyTorch models
- `train-turn` checkpoint and metrics generation
- `train-turn --feature-source audio` for synthetic WAV manifests
- `eval-turn --checkpoint` for NanoTurn checkpoints
- `export-turn-onnx` for NanoTurn checkpoints
- `reproduce-paper` smoke experiment bundle with JSON/Markdown artifacts and optional NanoTurn baseline rows
- config-driven paper smoke run via `configs/paper/paper_smoke.json`
- `scripts/reproduce_paper.py` wrapper
- `paper-table` extraction for baseline, turn benchmark, data benchmark, ASR manifest recipe, failure-case, streaming, streaming failure, streaming sweep, ASR transcript conversion, scenario, and policy tables
- failure-case mining for false completes, premature responses, missed interruptions, backchannel breaks, wait violations, and missed completes
- streaming ASR failure mining for word errors, endpoint delay, partial revisions, timestamp drift, first partial latency, stable-prefix loss, and slow RTF
- streaming ASR transcript fixture adapter and metrics
- `StreamingASRAdapter` protocol and `TranscriptJSONLAdapter` implementation
- `eval-streaming-asr` for WER/CER/RTF, partial stability, endpoint delay, and timestamp drift metrics
- `compare-streaming-asr` for multi-adapter streaming ASR comparison
- `sweep-streaming-asr` for chunk-size and lookahead sensitivity
- `eval-asr-command` for dependency-light external ASR command adapters
- `compare-asr-commands` for JSON-configured multi-system command-backed ASR comparisons
- paper smoke conversion checks for external Whisper/FunASR transcript schemas
- seedable VoiceWorld scenario evaluation with per-scenario breakdowns
- threshold policy search and cost-sensitive interaction objective
- `paper-table streaming` extraction
- `paper-table streaming_sweep` extraction
- `paper-table asr_transcript_conversions` extraction
- `paper-table scenarios` extraction
- `paper-table policy` extraction
- `paper-figure` SVG generation for platform diagrams plus baseline, latency, data, streaming, scenario, and policy figures
- `paper-bundle` generation for paper tables, figures, artifact index, and artifact manifest
- `paper-case-studies` for JSON/Markdown failure case studies linked to manifest and transcript records
- `paper-claim-audit` for mapping platform-paper claims to concrete files, result keys, commands, and artifacts
- `paper-status` for a single-page summary of smoke, structural, final-input, and final-paper readiness
- machine-readable stable-worldmodel-style paper parity checklist in `configs/paper/paper_parity_checklist.json`
- `paper-parity-audit` for separating smoke-level structural evidence from final-scale paper gaps
- machine-readable final-scale experiment runbook in `configs/paper/final_experiments.json`
- `final-experiments` for rendering final paper experiment inputs, commands, metrics, artifacts, and success criteria
- final run configuration template in `configs/final/paper_final.json`
- `final-config` for validating final paper run directories, corpora, splits, adapter config, and artifact paths
- `final-config --scaffold` for creating final-run directories and README hints without fabricating data
- `final-config --check-files` for reporting which final paper inputs are still missing before expensive runs
- `leaderboard-export` for JSONL/CSV metric rows
- machine-readable benchmark suite definition in `configs/benchmarks/stable_asr_v0.json`
- `benchmark-suite` for validating and rendering benchmark suite definitions
- `paper-audit` checks for paper result sections and bundled table/figure artifacts
- `paper-release-audit` checks stricter platform-paper release gates and reports remaining gaps
- `paper-draft` generation for editable Markdown preprint drafts
- `paper-latex` generation for arXiv-style LaTeX preprint drafts
- dataset and experiment card generation via `make-card`
- `CITATION.cff` and project documentation under `docs/`
- fixture dataset and tests
- GitHub Actions test workflow

Verification status:

```text
python3 -m pytest -q
passed with 1 skipped test
```

See [ROADMAP.md](ROADMAP.md) for the full plan.
