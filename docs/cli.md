# CLI

Stable-ASR exposes one command: `stable-asr`.

## Health And Roadmap

```bash
stable-asr doctor
stable-asr doctor --check-final-files
stable-asr roadmap-status --roadmap configs/roadmap/stable_asr_roadmap.json
stable-asr roadmap-status --require-final-ready
stable-asr platform-parity --registry configs/platform/stable_worldmodel_parity.json --validate-only
stable-asr platform-parity --output runs/PLATFORM_PARITY.md
```

`doctor` reports final input readiness as `NOT_CHECKED` unless
`--check-final-files` is supplied. Use `--check-release-env` separately for the
optional Torch/Lance dependency gate needed by strict release smoke.

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
stable-asr turn-submission --dataset examples/data/turn_demo.jsonl --predictions tests/fixtures/turn_predictions_sample.jsonl --system oracle_fixture --output-dir runs/submissions/oracle_fixture
stable-asr train-turn --dataset examples/data/turn_demo.jsonl --output-dir runs/nanoturn
stable-asr eval-turn --dataset examples/data/turn_demo.jsonl --checkpoint runs/nanoturn/checkpoint.pt
stable-asr export-turn-onnx --checkpoint runs/nanoturn/checkpoint.pt --output runs/nanoturn/nanoturn.onnx
```

## Contributor Benchmark Pack

```bash
stable-asr benchmark-pack --output-dir runs/benchmark_pack
cd runs/benchmark_pack
bash commands.sh
```

The generated `commands.sh` validates schemas, builds turn and streaming submission packages, then indexes the `submissions/` directory into `leaderboard/SUBMISSIONS.md` and a merged `leaderboard/leaderboard.jsonl` with validation and ranked Markdown reports.

## External ASR Adapter Pack

```bash
stable-asr adapter-pack --output-dir runs/adapter_pack
cd runs/adapter_pack
bash commands.sh
```

## VoiceWorld Scenario Pack

```bash
stable-asr scenario-pack --output-dir runs/scenario_pack
cd runs/scenario_pack
bash commands.sh
```

## Final Run Pack

```bash
stable-asr final-pack --output-dir runs/final_pack
cd runs/final_pack
bash commands.sh
```

The generated pack collects the final run config, final input collection plan,
final experiment runbook, ASR references, VoiceWorld suite, evidence matrix,
file audit, action plan, and scaffold directories. It does not create real
corpora, prediction manifests, checkpoints, or final benchmark results.

## Final Acquisition Pack

```bash
stable-asr final-acquisition-pack --output-dir runs/final_acquisition_pack
stable-asr final-assignment-audit --input runs/final_acquisition_pack/acquisition/assignments.json --output runs/final_acquisition_pack/reports/FINAL_ASSIGNMENT_AUDIT.md
stable-asr final-handoff-template --output runs/final_acquisition_pack/acquisition/handoff_template.json
stable-asr final-handoff-checksums --input runs/final_acquisition_pack/acquisition/handoff_template.json --repo-root . --output runs/final_acquisition_pack/acquisition/handoff_template.json
stable-asr validate-schema-file --input runs/final_acquisition_pack/acquisition/handoff_template.json --schema-id stable_asr.final_handoff.v0 --output runs/final_acquisition_pack/reports/FINAL_HANDOFF_SCHEMA_VALIDATION.md
stable-asr final-handoff-audit --input runs/final_acquisition_pack/acquisition/handoff_template.json --require-checksums --output runs/final_acquisition_pack/reports/FINAL_HANDOFF_AUDIT.md
cd runs/final_acquisition_pack
bash commands.sh
```

The generated pack converts `configs/final/input_collections.json` into a
staging checklist, source URL index, owner assignment tracker,
license/consent review sheet, VoiceWorld recording checklist, and structured
final input handoff template. The assignment audit checks owner, due-date, and
release-blocker status before handoff. The handoff audit checks staged paths,
owner metadata, license/consent notes, verification outputs, and optional
checksums before M5 final runs.

## Contributor Pack

```bash
stable-asr contributor-pack --output-dir runs/contributor_pack
cd runs/contributor_pack
bash commands.sh
```

The contributor pack gathers benchmark, adapter, VoiceWorld, final-run, and
final acquisition starter packs into one workspace, then copies the GitHub issue
and pull request templates so contributors can choose the right track before
opening a PR.

## Streaming ASR

```bash
stable-asr eval-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl
stable-asr compare-streaming-asr --input balanced=tests/fixtures/streaming_asr_sample.jsonl --input fast_unstable=tests/fixtures/streaming_asr_fast_unstable_sample.jsonl
stable-asr streaming-submission --input tests/fixtures/streaming_asr_sample.jsonl --system streaming_fixture --output-dir runs/submissions/streaming_fixture
stable-asr sweep-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl --chunks-ms 160 320 640 --lookahead-ms 0 160
stable-asr eval-asr-command --name my_asr --command "python your_export.py --output {output}" --output runs/my_asr.jsonl
stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --validate-only --require-input-manifest --min-adapters 4
stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --json-output runs/final/reports/asr_command_compare.json
```

## References

```bash
stable-asr model-registry --registry configs/models/stable_asr_models.json --validate-only
stable-asr model-registry --output runs/MODELS.md
stable-asr schema-registry --registry configs/schemas/stable_asr_schemas.json --validate-only
stable-asr schema-registry --output runs/SCHEMAS.md
stable-asr schema-registry --schema-id stable_asr.streaming_asr_record.v0 --json
stable-asr schema-registry --schema-id stable_asr.final_handoff.v0 --json
stable-asr validate-schema-file --input examples/data/turn_demo.jsonl --schema-id stable_asr.turn_manifest_record.v0
stable-asr validate-schema-file --input tests/fixtures/streaming_asr_sample.jsonl --schema-id stable_asr.streaming_asr_record.v0 --output runs/STREAMING_SCHEMA_VALIDATION.md
stable-asr asr-collections --registry configs/references/asr_collections.json --validate-only
stable-asr asr-collections --output runs/ASR_COLLECTIONS.md
stable-asr asr-collections --format paper-markdown --output runs/ASR_REFERENCES.md
stable-asr asr-collections --format bibtex --output runs/ASR_REFERENCES.bib
stable-asr asr-collections --format acquisition-markdown --output runs/ASR_COLLECTION_ACQUISITION.md
stable-asr asr-collections --audit-coverage --require-priority p0 --require-priority p1
stable-asr asr-collections --audit-readiness --output runs/ASR_COLLECTION_READINESS.md
stable-asr asr-collections --audit-licenses --output runs/ASR_COLLECTION_LICENSE_REVIEW.md
stable-asr turn-collections --registry configs/references/turn_collections.json --validate-only
stable-asr turn-collections --output runs/TURN_COLLECTIONS.md
stable-asr turn-collections --audit-coverage --require-priority p0 --require-priority p1 --output runs/TURN_COLLECTION_COVERAGE.md
stable-asr turn-collections --format acquisition-markdown --output runs/TURN_COLLECTION_ACQUISITION.md
```

## Paper Artifacts

```bash
stable-asr reproduce-paper --config configs/paper/paper_smoke.json
stable-asr paper-bundle --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts
stable-asr final-pack --output-dir runs/final_pack
stable-asr final-acquisition-pack --output-dir runs/final_acquisition_pack
stable-asr contributor-pack --output-dir runs/contributor_pack
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
stable-asr final-assignment-audit --input runs/final_acquisition_pack/acquisition/assignments.json --require-owner --require-due-date --require-ready --output runs/final/FINAL_ASSIGNMENT_AUDIT.md
stable-asr final-handoff-template --output runs/final/FINAL_INPUT_HANDOFF.json
stable-asr final-handoff-checksums --input runs/final/FINAL_INPUT_HANDOFF.json --repo-root . --output runs/final/FINAL_INPUT_HANDOFF.json
stable-asr validate-schema-file --input runs/final/FINAL_INPUT_HANDOFF.json --schema-id stable_asr.final_handoff.v0 --output runs/final/FINAL_HANDOFF_SCHEMA_VALIDATION.md
stable-asr final-handoff-audit --input runs/final/FINAL_INPUT_HANDOFF.json --repo-root . --require-checksums --output runs/final/FINAL_HANDOFF_AUDIT.md
stable-asr paper-evidence-matrix --output runs/final/FINAL_EVIDENCE_MATRIX.md
stable-asr make-card model --input configs/models/stable_asr_models.json --model-id nanoturn_pico --metrics runs/final/nanoturn/metrics.json --output runs/final/MODEL_CARD.md
stable-asr eval-scenario --dataset runs/final/voiceworld_real.jsonl --checkpoint runs/final/nanoturn/checkpoint.pt --json-output runs/final/reports/scenarios.json
stable-asr final-results --config configs/final/paper_final.json --output runs/final/paper_results.json
stable-asr leaderboard-export --results runs/final/paper_results.json --output runs/final/leaderboard.jsonl
stable-asr leaderboard-validate --input runs/final/leaderboard.jsonl --require-complete-suite --output runs/final/LEADERBOARD_VALIDATION.md
stable-asr leaderboard-report --input runs/final/leaderboard.jsonl --require-complete-suite --output runs/final/LEADERBOARD_REPORT.md
stable-asr submission-index --root runs/submissions --output-dir runs/final/community_leaderboard
stable-asr leaderboard-merge --input runs/submissions/oracle_fixture/leaderboard.jsonl --input runs/submissions/streaming_fixture/leaderboard.jsonl --output runs/final/community_leaderboard.jsonl --validation-output runs/final/COMMUNITY_LEADERBOARD_VALIDATION.md --report-output runs/final/COMMUNITY_LEADERBOARD_REPORT.md
stable-asr paper-artifact-integrity --manifest runs/paper/smoke/artifacts/artifact_hashes.json --root runs/paper/smoke/artifacts
stable-asr benchmark-suite --suite runs/paper/smoke/artifacts/benchmark_suite.json --artifacts-dir runs/paper/smoke/artifacts --validate-only
stable-asr paper-archive --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/artifacts.tar.gz
stable-asr paper-archive-verify --archive runs/paper/smoke/artifacts.tar.gz
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke
stable-asr paper-release-smoke --require-final-ready
stable-asr paper-status --release-dir runs/paper/release_smoke
stable-asr paper-release-audit --repo-root . --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts
stable-asr paper-release-audit --repo-root . --require-final-ready
```
