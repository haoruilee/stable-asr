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
- final-run action plan showing the remaining data-staging and experiment commands
- final evidence matrix linking final experiments to blockers, commands, success criteria, and artifacts
- paper bundle provenance recording git commit, Stable-ASR version, input result hash, and config hashes
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
stable-asr final-pack --output-dir runs/final_pack
stable-asr final-acquisition-pack --output-dir runs/final_acquisition_pack
stable-asr contributor-pack --output-dir runs/contributor_pack
stable-asr paper-status --release-dir runs/paper/smoke --output runs/paper/smoke/artifacts/PAPER_STATUS.md
stable-asr paper-case-studies --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts
stable-asr paper-claim-audit --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output-dir runs/paper/smoke/artifacts
stable-asr paper-parity-audit --checklist configs/paper/paper_parity_checklist.json --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts
stable-asr platform-parity --registry configs/platform/stable_worldmodel_parity.json --output runs/paper/smoke/artifacts/PLATFORM_PARITY.md
stable-asr reference-workqueue --output runs/paper/smoke/artifacts/REFERENCE_WORKQUEUE.md
stable-asr reference-workqueue --format evidence-markdown --output runs/paper/smoke/artifacts/REFERENCE_EVIDENCE_TEMPLATES.md
stable-asr final-experiments --registry configs/paper/final_experiments.json --output runs/paper/smoke/artifacts/FINAL_EXPERIMENTS.md
stable-asr final-config --config configs/final/paper_final.json --output runs/paper/smoke/artifacts/FINAL_RUN_CONFIG.md
stable-asr final-config --config configs/final/paper_final.json --scaffold
stable-asr final-config --config configs/final/paper_final.json --prepare-inputs
stable-asr final-config --config configs/final/paper_final.json --prepare-corpora
stable-asr final-config --config configs/final/paper_final.json --prepare-asr-eval-manifest
stable-asr final-config --config configs/final/paper_final.json --bootstrap-turn-splits
stable-asr final-config --config configs/final/paper_final.json --prepare-external-predictions
stable-asr final-config --config configs/final/paper_final.json --prepare-voiceworld-real
stable-asr final-config --config configs/final/paper_final.json --audit-voiceworld-real --scenario-suite configs/scenarios/stable_asr_voiceworld_v0.json
stable-asr final-config --config configs/final/paper_final.json --audit-asr-commands
stable-asr final-config --config configs/final/paper_final.json --prepare-asr-transcript-conversions
# Expected to report NOT_READY until final corpora, splits, external predictions, real VoiceWorld, and raw ASR exports exist.
stable-asr final-config --config configs/final/paper_final.json --check-files
stable-asr final-config --config configs/final/paper_final.json --plan-missing --output runs/final/FINAL_RUN_ACTION_PLAN.md
stable-asr final-assignment-audit --input runs/final_acquisition_pack/acquisition/assignments.json --require-owner --require-due-date --require-ready --output runs/final/FINAL_ASSIGNMENT_AUDIT.md
stable-asr final-handoff-template --output runs/final/FINAL_INPUT_HANDOFF.json
stable-asr final-handoff-checksums --input runs/final/FINAL_INPUT_HANDOFF.json --repo-root . --output runs/final/FINAL_INPUT_HANDOFF.json
stable-asr validate-schema-file --input runs/final/FINAL_INPUT_HANDOFF.json --schema-id stable_asr.final_handoff.v0 --output runs/final/FINAL_HANDOFF_SCHEMA_VALIDATION.md
stable-asr final-handoff-audit --input runs/final/FINAL_INPUT_HANDOFF.json --repo-root . --require-checksums --output runs/final/FINAL_HANDOFF_AUDIT.md
stable-asr final-inputs --registry configs/final/input_collections.json --output runs/final/FINAL_INPUT_COLLECTIONS.md
stable-asr paper-evidence-matrix --output runs/final/FINAL_EVIDENCE_MATRIX.md
stable-asr final-results --config configs/final/paper_final.json --output runs/final/paper_results.json
stable-asr leaderboard-export --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/leaderboard.jsonl
stable-asr leaderboard-validate --input runs/paper/smoke/leaderboard.jsonl --output runs/paper/smoke/LEADERBOARD_VALIDATION.md
stable-asr submission-index --root runs/submissions --output-dir runs/paper/smoke/community_leaderboard
stable-asr leaderboard-merge --input runs/submissions/oracle_fixture/leaderboard.jsonl --input runs/submissions/streaming_fixture/leaderboard.jsonl --output runs/paper/smoke/community_leaderboard.jsonl --validation-output runs/paper/smoke/COMMUNITY_LEADERBOARD_VALIDATION.md --report-output runs/paper/smoke/COMMUNITY_LEADERBOARD_REPORT.md
stable-asr paper-artifact-integrity --manifest runs/paper/smoke/artifacts/artifact_hashes.json --root runs/paper/smoke/artifacts
stable-asr benchmark-suite --suite runs/paper/smoke/artifacts/benchmark_suite.json --artifacts-dir runs/paper/smoke/artifacts --validate-only
stable-asr paper-archive --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/artifacts.tar.gz
stable-asr paper-archive-verify --archive runs/paper/smoke/artifacts.tar.gz
stable-asr benchmark-suite --suite configs/benchmarks/stable_asr_v0.json --validate-only
stable-asr data-sources --registry configs/datasets/stable_asr_sources.json --validate-only
stable-asr adapter-registry --registry configs/adapters/stable_asr_adapters.json --validate-only
stable-asr model-registry --registry configs/models/stable_asr_models.json --validate-only
stable-asr scenario-suite --suite configs/scenarios/stable_asr_voiceworld_v0.json --validate-only
stable-asr benchmark-suite --suite configs/benchmarks/stable_asr_v0.json --results runs/paper/smoke/paper_results.json --validate-only
```

The `paper-status` output reports the final assignment and handoff gates
separately from final input paths, so missing owners, due dates, release
blockers, missing audit files, or invalid handoff metadata stay visible before
the final handoff is accepted.

The bundle includes Markdown tables and SVG figures for platform architecture,
data registry, VoiceWorld timelines, policy state transitions, baseline quality,
latency, ASR manifest recipe summaries, failure-case taxonomy, streaming metrics,
streaming failure taxonomy, external ASR transcript conversion, scenario
robustness, policy search, leaderboard-ready JSONL/CSV metric rows, a
leaderboard validation report, ranked leaderboard reports, a copied `paper_results.json`, sha256 artifact
integrity manifests, provenance manifests for git/config/result traceability,
machine-readable benchmark suite definition, a data source registry, a
baseline/adapter registry, ASR and turn/full-duplex reference collections,
paper-reference notes, BibTeX, source manifests, a unified reference work
queue, reference evidence templates, reference evidence audit, reference assignment tracker, coverage artifacts, readiness artifacts,
acquisition plans, a VoiceWorld scenario suite definition, and
case-study JSON/Markdown artifacts that link failure examples back to source
records, a paper parity audit that separates smoke-level structural evidence
from final-scale paper gaps, a repository-level stable-worldmodel platform
parity audit, a final-scale experiment runbook, a final-run
configuration template, a final-run file audit/action plan, a final evidence matrix, a roadmap status report, and a claim evidence matrix
that links platform-paper claims to files, result keys, artifacts, and
reproduction commands. `PAPER_STATUS.md` summarizes these signals in one page.
For final-scale release gates, fill and audit the acquisition assignment
tracker before handoff, then fill `runs/final/FINAL_INPUT_HANDOFF.json` with
real owner, license/consent, verification, staged path, and checksum evidence
by running `final-handoff-checksums`, `validate-schema-file --schema-id stable_asr.final_handoff.v0`,
and `final-handoff-audit --require-checksums`.

## Audits And Drafts

```bash
stable-asr paper-audit --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke
stable-asr paper-draft --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/PAPER_DRAFT.md
stable-asr paper-latex --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/paper.tex
```

`paper-audit` checks artifact shape, provenance files, and verifies the bundle hash manifest.
`paper-artifact-integrity` can be run directly to re-check `artifact_hashes.json`
after moving or publishing a bundle. `paper-archive` writes a publishable
`tar.gz` plus SHA256 sidecar after those gates pass, and `paper-archive-verify`
checks the sidecar digest, archive path safety, embedded hash manifest, and
benchmark artifact requirements. `paper-release-audit` checks whether the
repository has enough evidence for a platform paper release. `paper-parity-audit`
checks whether each stable-worldmodel-style paper element has structural
evidence and lists the remaining final-scale experiment requirements.
`platform-parity` checks whether the repository has the stable-worldmodel-style
platform surfaces expected by Stable-ASR: install/CLI, data formats,
VoiceWorld scenarios, baseline/solver zoo, release artifacts, contributor
packs, and reference collections.
`paper-release-smoke` runs the full smoke pipeline, creates drafts and cards,
and writes `release_audit.json` plus `RELEASE_AUDIT.md` in one command. Use
`--skip-train` for a faster structural check that does not train NanoTurn, and
use `--strict` when the environment has optional Lance dependencies and
final-scale inputs installed and a NOT_READY audit should fail the command.
