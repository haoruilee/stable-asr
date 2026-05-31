<h1 align="center">Stable-ASR</h1>

<p align="center"><i>A reproducible research platform for real-time ASR systems and full-duplex turn-taking.</i></p>

<p align="center">
  <a href="docs/index.md"><img alt="Documentation" src="https://img.shields.io/badge/Docs-blue.svg"/></a>
  <a href="https://github.com/haoruilee/stable-asr/actions"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/haoruilee/stable-asr/tests.yml?label=Tests"/></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"/></a>
  <a href="ROADMAP.md"><img alt="Roadmap" src="https://img.shields.io/badge/Roadmap-Stable--ASR-orange.svg"/></a>
</p>

<p align="center">
  <a href="#installation"><b>Installation</b></a> ·
  <a href="#quick-start"><b>Quick Start</b></a> ·
  <a href="#data-formats"><b>Data Formats</b></a> ·
  <a href="#voiceworld-scenarios"><b>VoiceWorld</b></a> ·
  <a href="#baselines-and-adapters"><b>Baselines</b></a> ·
  <a href="#paper-and-release-smoke"><b>Paper Pipeline</b></a> ·
  <a href="#documentation"><b>Documentation</b></a>
</p>

---

Stable-ASR provides a single interface for the system layer between streaming
ASR and voice-agent control: **data manifests**, **turn-taking baselines**,
**scenario simulation**, **streaming ASR metrics**, **policy search**, and
**paper-ready evaluation artifacts**.

The first model family is **NanoTurn**, a lightweight turn/action model for
endpointing, incomplete pauses, backchannels, wait commands, interruptions, and
side speech.

## Scope

Stable-ASR is not another general ASR toolkit. The first releases focus on:

- standardized ASR and turn-taking manifests
- NanoTurn baseline models
- endpointing and turn-taking evaluation
- streaming ASR metrics beyond WER/CER
- scenario-based robustness testing
- latency and deployment reports
- paper-ready benchmark scripts, tables, and figures

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -e ".[train]"   # optional NanoTurn training
python -m pip install -e ".[lance]"   # optional Parquet/Lance data backends
python -m pip install -e ".[lance,train]"  # paper-release-smoke READY path
```

## Quick Start

```bash
# 1. Inspect data and compare turn-taking baselines
stable-asr doctor
stable-asr validate-manifest examples/data/turn_demo.jsonl
stable-asr compare-turn \
  --dataset examples/data/turn_demo.jsonl \
  --baseline vad_pause \
  --baseline text_turn \
  --report runs/turn_compare.md

# 2. Train and evaluate NanoTurn
stable-asr train-turn --dataset examples/data/turn_demo.jsonl --output-dir runs/nanoturn
stable-asr eval-turn --dataset examples/data/turn_demo.jsonl --checkpoint runs/nanoturn/checkpoint.pt

# 3. Generate stable-worldmodel-style paper evidence
stable-asr doctor --check-release-env
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke
```

## Data Formats

Stable-ASR uses a small format registry for turn manifests. JSONL is the
zero-dependency core format; Parquet and Lance are opt-in data-layer backends for
paper-facing throughput and random-sampling benchmarks.

| Format | Install | Best for |
| --- | --- | --- |
| `jsonl` | base package | inspection, fixtures, simple interchange |
| `parquet` | `stable-asr[data]` or `stable-asr[lance]` | columnar corpus manifests and compact benchmark artifacts |
| `lance` | `stable-asr[lance]` | random indexed reads, paper data-layer benchmark rows |

```bash
stable-asr convert examples/data/turn_demo.jsonl runs/turn_demo.parquet
stable-asr convert examples/data/turn_demo.jsonl runs/turn_demo.lance
stable-asr benchmark-data \
  --dataset examples/data/turn_demo.jsonl \
  --output-dir runs/data_bench \
  --formats jsonl parquet lance
```

## VoiceWorld Scenarios

VoiceWorld is the speech interaction counterpart of stable-worldmodel's
environment suite. The default suite is defined in
`configs/scenarios/stable_asr_voiceworld_v0.json`.

| Scenario | Expected action |
| --- | --- |
| normal question | `take_turn` |
| incomplete pause | `keep_listening` |
| listener backchannel | `continue_speaking` |
| wait or hold command | `hold` |
| user interruption | `stop_tts_and_listen` |
| side conversation | `ignore` |
| ambient speech | `ignore` |
| noisy far-field speech | `take_turn` |
| code switching | `take_turn` |

Factors of variation include pause length, SNR, reverb, speaking rate, overlap
offset, network jitter, far-field distance, and code-switch ratio.

```bash
stable-asr scenario-suite --suite configs/scenarios/stable_asr_voiceworld_v0.json --validate-only
stable-asr eval-scenario --episodes 21 --seed 0 --baseline vad_pause --report runs/scenario.md
```

## Baselines And Adapters

| System | Type | Purpose |
| --- | --- | --- |
| `rule_endpoint` | baseline | lowest endpointing baseline |
| `vad_pause` | baseline | industrial pause-threshold endpointing |
| `text_turn` | baseline | semantic text-only turn baseline |
| `prediction_manifest` | adapter | SmartTurn/EasyTurn/VAP-style prediction bridge |
| `nanoturn_pico` | model | trainable lightweight turn/action model |
| command ASR adapters | adapter | evaluate Whisper, FunASR, WeNet, NeMo, ESPnet, SpeechBrain, icefall, sherpa-onnx, FireRedASR2S, Qwen3-ASR, whisper.cpp, WhisperX, Moonshine, and HF exports without vendoring them |

Reference coverage is tracked in `configs/references/asr_collections.json`,
`configs/references/turn_collections.json`, and `configs/adapters/stable_asr_adapters.json`.

```bash
stable-asr adapter-registry --registry configs/adapters/stable_asr_adapters.json --validate-only
stable-asr asr-collections --audit-coverage --require-priority p0 --require-priority p1
stable-asr asr-collections --audit-readiness --output runs/ASR_COLLECTION_READINESS.md
stable-asr asr-collections --format acquisition-markdown --output runs/ASR_COLLECTION_ACQUISITION.md
stable-asr asr-collections --format bibtex --output runs/ASR_REFERENCES.bib
stable-asr turn-collections --audit-coverage --output runs/TURN_COLLECTION_COVERAGE.md
stable-asr turn-collections --format acquisition-markdown --output runs/TURN_COLLECTION_ACQUISITION.md
stable-asr compare-asr-commands --config examples/configs/asr_vendor_adapter_demo.json --report runs/asr_vendor_adapter.md
stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --validate-only --require-input-manifest --min-adapters 2
```

## Paper And Release Smoke

Stable-ASR treats paper artifacts as part of the platform, not a separate
afterthought.

```bash
stable-asr reproduce-paper --config configs/paper/paper_smoke.json
stable-asr paper-bundle --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke
stable-asr paper-release-audit \
  --repo-root . \
  --results runs/paper/smoke/paper_results.json \
  --artifacts-dir runs/paper/smoke/artifacts
```

`paper-release-smoke` writes `paper_results.json`, tables, figures, starter packs, registry
artifacts, a copied `paper_results.json`, artifact hash manifests, provenance
manifests, case studies, final-run audits/action plans, final input collection
status, final evidence matrix, claim evidence, roadmap status, `PAPER_DRAFT.md`, `paper.tex`,
dataset/experiment/model cards, `artifacts.tar.gz`, and
`RELEASE_AUDIT.md`.

Use `stable-asr doctor --check-release-env` before strict release smoke. A
READY smoke audit needs both the optional Lance backend and NanoTurn training
dependencies:

```bash
python -m pip install -e ".[lance,train]"
stable-asr doctor --check-release-env
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke --strict
```

## Documentation

See [docs/index.md](docs/index.md) for the full documentation. A MkDocs config
is provided in `mkdocs.yaml`.

## Current M0 Functionality

- installable package
- `stable-asr` CLI
- `doctor` command for environment, optional dependency, config, and final-input readiness checks
- machine-readable roadmap registry in `configs/roadmap/stable_asr_roadmap.json`
- `roadmap-status` for validating and rendering current milestone evidence without claiming final-scale completion
- JSONL turn manifest validation
- canonical turn/action labels
- core turn dataclasses
- minimal turn metrics
- threshold/hysteresis turn policy
- Markdown report helper
- `eval-turn` baseline evaluation command
- `predict-turn` for exporting baseline/checkpoint predictions in the shared prediction manifest schema
- `validate-turn-predictions` for checking prediction schema, ID coverage, duplicate IDs, and extra predictions before benchmark submission
- `turn-submission` for packaging external turn predictions with schema validation, coverage validation, evaluation, and leaderboard rows
- `submission-index` for discovering submission packages, generating a community index, and merging leaderboard rows
- `benchmark-pack` for generating a contributor starter kit with schemas, benchmark suite metadata, fixture data, and runnable submission commands
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
- `prepare-voiceworld` for converting real VoiceWorld scenario annotation tables into canonical turn manifests
- `convert-external` for EasyTurn-style, Full-Duplex-Bench-style, and SmartTurn-style JSONL manifests
- `convert-predictions` for generic, SmartTurn-style, and EasyTurn-style prediction JSONL
- `convert-asr-transcript` for Whisper, FunASR, Qwen3-ASR, FireRedASR2S, WhisperX, whisper.cpp, SenseVoice, Moonshine, and WhisperKit-style transcript JSONL into the normalized streaming ASR schema
- `benchmark-data` for paper-facing data-layer tables
- CI smoke coverage for optional JSONL/Parquet/Lance data backend benchmarks
- machine-readable data source registry in `configs/datasets/stable_asr_sources.json`
- `data-sources` for validating and rendering data source registries
- machine-readable adapter registry in `configs/adapters/stable_asr_adapters.json`
- `adapter-registry` for validating and rendering baseline, converter, command-backed, and external-system adapter entries
- `adapter-pack` for generating an external ASR adapter starter kit with registries, command config, fixtures, and script templates
- machine-readable JSON Schema registry in `configs/schemas/stable_asr_schemas.json`
- `schema-registry` for validating and rendering public data, prediction, streaming, leaderboard, model, and final-input contracts
- `validate-schema-file` for checking JSON/JSONL files against those public contracts before publishing or submitting artifacts
- machine-readable upstream ASR reference collection in `configs/references/asr_collections.json`
- `asr-collections` for validating and rendering top ASR project references
- `asr-collections --format paper-markdown|bibtex` for paper-ready related-work reference artifacts
- `asr-collections --format acquisition-markdown` for turning the upstream ASR registry into a concrete collection and evidence-staging plan
- `asr-collections --audit-coverage` for checking P0 reference coverage in the adapter registry
- `asr-collections --audit-readiness` for checking review freshness, P0/P1 adapter evidence, action plans, and license-review warnings
- machine-readable turn/full-duplex reference collection in `configs/references/turn_collections.json`
- `turn-collections` for validating Smart Turn, Easy Turn, VAP, Full-Duplex-Bench, VAD, and voice-agent framework references
- machine-readable VoiceWorld scenario suite in `configs/scenarios/stable_asr_voiceworld_v0.json`
- `scenario-suite` for validating and rendering scenario suite definitions
- `scenario-pack` for generating a VoiceWorld scenario contribution kit with suite metadata, editable annotations, and runnable evaluation commands
- `final-pack` for generating a final-scale run starter kit with final configs, input collection plans, experiment runbooks, evidence audits, and scaffold directories
- `final-acquisition-pack` for generating a collaborator-facing final input staging checklist, license/consent review sheet, VoiceWorld recording checklist, and handoff template
- `contributor-pack` for generating all public contribution starter packs plus copied issue and PR templates in one onboarding workspace
- utterance-level ASR manifest schema and metadata-table recipe via `prepare-asr-manifest`
- `validate-asr-manifest` and `inspect-asr-manifest` for public ASR corpus manifests
- `prepare-public-asr` recipes for local LibriSpeech, AISHELL-1, Common Voice, and WenetSpeech directories
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
- `streaming-submission` for packaging external streaming ASR traces with schema validation, metrics, and leaderboard rows
- `sweep-streaming-asr` for chunk-size and lookahead sensitivity
- `eval-asr-command` for dependency-light external ASR command adapters
- `compare-asr-commands` for JSON-configured multi-system command-backed ASR comparisons
- `compare-asr-commands --validate-only` for auditing adapter commands, shared ASR manifests, output placeholders, and required raw exports before executing heavyweight systems
- paper smoke conversion checks for external Whisper, FunASR, Qwen3-ASR, and FireRedASR2S transcript schemas
- seedable VoiceWorld scenario evaluation with per-scenario breakdowns
- threshold policy search and cost-sensitive interaction objective
- `paper-table streaming` extraction
- `paper-table streaming_sweep` extraction
- `paper-table asr_transcript_conversions` extraction
- `paper-table scenarios` extraction
- `paper-table policy` extraction
- `paper-figure` SVG generation for platform diagrams plus baseline, latency, data, streaming, scenario, and policy figures
- `paper-bundle` generation for paper results, tables, figures, artifact index, and artifact manifest
- contributor benchmark, adapter, VoiceWorld scenario, final-run, final-input acquisition, and unified contributor starter packs in `paper-bundle`
- schema registry JSON/Markdown artifacts in `paper-bundle`
- ASR and turn/full-duplex reference collection artifacts in `paper-bundle`
- paper bundle sha256 integrity manifests and `paper-artifact-integrity` verification
- paper bundle provenance manifests that record Stable-ASR version, git commit, input result hashes, and config hashes
- `paper-archive` and `paper-archive-verify` for publishable tar.gz artifact archives with SHA256 and embedded bundle checks
- ASR and turn/full-duplex reference collection, coverage, readiness, acquisition, paper-reference, and BibTeX artifacts in `paper-bundle`
- final evidence matrix artifacts in `paper-bundle`
- `paper-case-studies` for JSON/Markdown failure case studies linked to manifest and transcript records
- `paper-claim-audit` for mapping platform-paper claims to concrete files, result keys, commands, and artifacts
- `paper-status` for a single-page summary of smoke, structural, final-input, and final-paper readiness
- `paper-evidence-matrix` for mapping final experiments to blockers, commands, expected artifacts, and success criteria
- machine-readable stable-worldmodel-style paper parity checklist in `configs/paper/paper_parity_checklist.json`
- `paper-parity-audit` for separating smoke-level structural evidence from final-scale paper gaps
- roadmap status JSON/Markdown artifacts in `paper-bundle`
- machine-readable final-scale experiment runbook in `configs/paper/final_experiments.json`
- `final-experiments` for rendering final paper experiment inputs, commands, metrics, artifacts, and success criteria
- final run configuration template in `configs/final/paper_final.json`
- `final-config` for validating final paper run directories, corpora, splits, adapter config, and artifact paths
- `final-config --scaffold` for creating final-run directories and README hints without fabricating data
- `final-config --check-files` for reporting which final paper inputs are still missing before expensive runs
- `final-config --prepare-inputs` for running final corpus, ASR eval manifest, weak split, prediction, VoiceWorld, ASR-command, and file-audit preparation in sequence
- `final-config --prepare-corpora` for preparing configured public ASR manifests from local corpus directories
- `final-config --prepare-asr-eval-manifest` for combining prepared ASR corpus manifests into the shared final streaming-ASR evaluation manifest
- `final-config --bootstrap-turn-splits` for creating weak train/dev/test turn splits from prepared final ASR manifests
- `final-config --prepare-external-predictions` for normalizing configured SmartTurn/EasyTurn-style prediction exports
- `final-config --prepare-voiceworld-real` for preparing and auditing configured real VoiceWorld metadata/audio inputs
- `final-config --audit-voiceworld-real` for checking final real VoiceWorld scenario and factor coverage
- `final-config --audit-asr-commands` for checking final command-backed ASR comparison inputs without executing the adapters
- `final-config --prepare-asr-transcript-conversions` for turning configured ASR adapter outputs into the final transcript-conversion result input
- `final-config --plan-missing` for turning the final-run file audit into an actionable data-staging and experiment runbook
- `final-inputs` for validating and rendering the final-scale input collection plan in `configs/final/input_collections.json`
- `final-results` for assembling audited final-scale JSON outputs into `runs/final/paper_results.json`
- `--json-output` on core evaluators so final runbooks can write machine-readable result inputs without shell redirection
- `leaderboard-export` for JSONL/CSV metric rows
- `leaderboard-validate` for checking external leaderboard JSONL submissions against the benchmark suite schema
- `leaderboard-report` for generating ranked per-task/per-metric Markdown or JSON leaderboard reports
- `leaderboard-merge` for combining multiple external submission leaderboards into one validated ranked report
- machine-readable benchmark suite definition in `configs/benchmarks/stable_asr_v0.json`
- `benchmark-suite` for validating and rendering benchmark suite definitions and required artifact coverage
- machine-readable built-in model registry in `configs/models/stable_asr_models.json`
- `model-registry` for validating and rendering built-in model/baseline metadata
- `paper-audit` checks for paper result sections and bundled table/figure artifacts
- `paper-release-audit` checks stricter platform-paper release gates and reports remaining gaps
- `paper-release-smoke` runs the smoke pipeline, generates bundle/drafts/cards, and writes release audit files
- `paper-draft` generation for editable Markdown preprint drafts
- `paper-latex` generation for arXiv-style LaTeX preprint drafts
- dataset, experiment, and model card generation via `make-card`
- `CITATION.cff` and project documentation under `docs/`
- fixture dataset and tests
- GitHub Actions test workflow

Verification status:

```text
python3 -m pytest -q
passed with 1 skipped test
```

See [ROADMAP.md](ROADMAP.md) for the full plan.
