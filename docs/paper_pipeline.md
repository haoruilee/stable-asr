# Paper Pipeline

The paper pipeline is designed to make the future platform paper reproducible
from checked-in code and configs.

## Smoke Run

```bash
stable-asr reproduce-paper --config configs/paper/paper_smoke.json
```

The smoke run writes:

- `paper_results.json`
- `reports/paper_smoke.md`
- synthetic turn manifests
- machine-readable data source registry artifacts
- machine-readable adapter registry artifacts
- ASR metadata TSV -> ASR manifest recipe fixture
- external conversion fixtures
- JSONL, Parquet, and Lance write/read/size/random-sampling benchmark rows when optional dependencies are installed
- multi-adapter streaming ASR comparison fixtures
- config-driven command-backed ASR comparison demo under `examples/configs/asr_command_compare_demo.json`
- chunk-size and lookahead streaming schedule sweep rows
- command-backed streaming ASR adapter fixture
- external Whisper, FunASR, Qwen3-ASR, and FireRedASR2S transcript conversion fixtures and streaming metrics
- stable-worldmodel-style paper parity checklist and gap audit
- final-scale experiment runbook for real paper execution
- final-run config template for corpus, split, adapter, and artifact paths
- final-run scaffold for directories and README hints without fake data
- final-run file audit showing which real paper inputs are still missing
- optional NanoTurn checkpoint and metrics
- streaming ASR fixture metrics

The smoke run is not a final paper result. It fixes the artifact shape that
larger experiments must follow.

## Tables And Figures

```bash
stable-asr paper-table baselines --results runs/paper/smoke/paper_results.json
stable-asr paper-table asr_manifest_recipe --results runs/paper/smoke/paper_results.json
stable-asr paper-table failure_cases --results runs/paper/smoke/paper_results.json
stable-asr paper-table streaming_failures --results runs/paper/smoke/paper_results.json
stable-asr paper-table asr_transcript_conversions --results runs/paper/smoke/paper_results.json
stable-asr paper-figure architecture --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/figures/architecture.svg
stable-asr paper-bundle --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts
stable-asr paper-status --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/artifacts/PAPER_STATUS.md
stable-asr paper-case-studies --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts
stable-asr paper-claim-audit --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output-dir runs/paper/smoke/artifacts
stable-asr paper-parity-audit --checklist configs/paper/paper_parity_checklist.json --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts
stable-asr final-experiments --registry configs/paper/final_experiments.json --output runs/paper/smoke/artifacts/FINAL_EXPERIMENTS.md
stable-asr final-config --config configs/final/paper_final.json --output runs/paper/smoke/artifacts/FINAL_RUN_CONFIG.md
stable-asr final-config --config configs/final/paper_final.json --scaffold
stable-asr final-config --config configs/final/paper_final.json --prepare-inputs
stable-asr final-config --config configs/final/paper_final.json --prepare-corpora
stable-asr final-config --config configs/final/paper_final.json --bootstrap-turn-splits
stable-asr final-config --config configs/final/paper_final.json --prepare-external-predictions
# Expected to report NOT_READY until final corpora, splits, and external predictions exist.
stable-asr final-config --config configs/final/paper_final.json --check-files
stable-asr leaderboard-export --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/leaderboard.jsonl
stable-asr benchmark-suite --suite configs/benchmarks/stable_asr_v0.json --validate-only
stable-asr data-sources --registry configs/datasets/stable_asr_sources.json --validate-only
stable-asr adapter-registry --registry configs/adapters/stable_asr_adapters.json --validate-only
stable-asr scenario-suite --suite configs/scenarios/stable_asr_voiceworld_v0.json --validate-only
stable-asr benchmark-suite --suite configs/benchmarks/stable_asr_v0.json --results runs/paper/smoke/paper_results.json --validate-only
```

The bundle includes Markdown tables and SVG figures for platform architecture,
data registry, VoiceWorld timelines, policy state transitions, baseline quality,
latency, ASR manifest recipe summaries, failure-case taxonomy, streaming metrics,
streaming failure taxonomy, external ASR transcript conversion, scenario
robustness, policy search, leaderboard-ready JSONL/CSV metric rows, a
machine-readable benchmark suite definition, a data source registry, a
baseline/adapter registry, ASR reference collection and coverage artifacts, a VoiceWorld scenario suite definition, and
case-study JSON/Markdown artifacts that link failure examples back to source
records, a paper parity audit that separates smoke-level structural evidence
from final-scale paper gaps, a final-scale experiment runbook, a final-run
configuration template, a final-run file audit, a roadmap status report, and a claim evidence matrix
that links platform-paper claims to files, result keys, artifacts, and
reproduction commands. `PAPER_STATUS.md` summarizes these signals in one page.

## Audits And Drafts

```bash
stable-asr paper-audit --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke
stable-asr paper-draft --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/PAPER_DRAFT.md
stable-asr paper-latex --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/paper.tex
```

`paper-audit` checks artifact shape. `paper-release-audit` checks whether the
repository has enough evidence for a platform paper release. `paper-parity-audit`
checks whether each stable-worldmodel-style paper element has structural
evidence and lists the remaining final-scale experiment requirements.
`paper-release-smoke` runs the full smoke pipeline, creates drafts and cards,
and writes `release_audit.json` plus `RELEASE_AUDIT.md` in one command. Use
`--skip-train` for a faster structural check that does not train NanoTurn, and
use `--strict` when the environment has optional Lance dependencies and
final-scale inputs installed and a NOT_READY audit should fail the command.
