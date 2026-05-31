# Release Gates

Stable-ASR uses two levels of paper readiness checks.

## Artifact Audit

`paper-audit` verifies that one run contains the required result sections,
tables, figures, artifact index, manifest, sha256 artifact integrity manifest,
and provenance manifest.

This is a structural check. Passing it does not mean the platform paper is
ready.

## Release Audit

`paper-release-audit` checks stricter platform-paper gates:

- source package and CI exist
- `MANIFEST.in` exists and includes platform configs, docs, examples, scripts, and fixtures for source distributions
- wheel data-file metadata installs platform configs, docs, examples, scripts, and fixtures under `share/stable-asr`
- CI builds a wheel, installs it into a clean virtual environment, and runs platform commands from an empty working directory
- CI installs `stable-asr[lance]` and runs JSONL/Parquet/Lance data backend smoke benchmarks plus a Lance-enabled release smoke pass
- `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md` exist
- A pull request template exists for evidence commands, data/license notes, and final-scale impact checks
- GitHub issue templates exist for final data acquisition, ASR adapters, VoiceWorld scenarios, and benchmark submissions
- paper config and reproduction script exist
- roadmap registry exists and `roadmap-status` separates current milestone evidence from final-scale readiness; `roadmap-status --require-final-ready` must fail until real paper-scale inputs and artifacts exist
- platform parity registry exists and `platform-parity` can audit stable-worldmodel-style repository shape
- `stable-asr doctor` reports required config schemas as OK
- paper parity checklist JSON exists in `configs/paper/paper_parity_checklist.json`
- final-scale experiment registry JSON exists in `configs/paper/final_experiments.json`
- final-run config JSON exists in `configs/final/paper_final.json`
- final ASR command comparison config exists in `configs/final/asr_command_compare.json` and `compare-asr-commands --validate-only --require-input-manifest` audits shared manifests, raw exports, output placeholders, and at least two adapters
- final Whisper and FunASR raw-export bridge scripts exist under `scripts/` and validate manifest coverage before writing normalized streaming rows
- `final-config --prepare-asr-transcript-conversions` can convert configured normalized ASR outputs into the final transcript-conversion result input
- final results assembly is available through `stable-asr final-results` so final-scale JSON outputs have one audited path into `paper_results.json`
- benchmark suite JSON exists in `configs/benchmarks/stable_asr_v0.json`
- benchmark suite task/system/metric coverage is verified against leaderboard rows
- benchmark suite required artifacts are verified against the generated bundle
- data source registry JSON exists in `configs/datasets/stable_asr_sources.json`
- adapter registry JSON exists in `configs/adapters/stable_asr_adapters.json`
- ASR reference collection JSON exists in `configs/references/asr_collections.json`
- required P0 and P1 ASR references have adapter or bridge coverage evidence
- ASR reference collection readiness checks review freshness, Stable-ASR action plans, adapter evidence, and license-review warnings
- ASR reference collection exports paper Markdown notes and BibTeX attribution artifacts
- ASR reference collection exports an acquisition plan that maps upstream projects to evidence targets
- turn/full-duplex reference collection JSON exists in `configs/references/turn_collections.json`
- required P0 turn/full-duplex references have data-source or adapter coverage evidence
- VoiceWorld scenario suite JSON exists in `configs/scenarios/stable_asr_voiceworld_v0.json`
- ASR manifest schema and metadata-table recipe exist
- data benchmark sections exist
- data benchmark rows include random sampling throughput
- at least three external data source conversions are represented
- ASR manifest recipe results validate at least one utterance-level corpus manifest
- baseline, latency, scenario, policy, and streaming metrics exist
- baseline failure-case mining exists for paper case studies
- streaming ASR comparison includes at least two adapter rows
- streaming ASR failure mining exists for real-time case studies
- streaming ASR schedule sweep includes chunk/lookahead rows
- command-backed streaming ASR adapter fixture exists
- external ASR transcript conversion includes at least two schemas
- NanoTurn is included as a checkpoint-backed baseline
- paper bundle, Markdown draft, and LaTeX draft exist
- `paper_results.json` is copied into the artifact bundle and matches the audited source results file
- publishable `artifacts.tar.gz` and `.sha256` sidecar are generated from an audited artifact bundle
- `paper-archive-verify` validates the archive SHA256 sidecar, tar path safety, embedded bundle hashes, and benchmark artifact requirements
- dataset, experiment, and model cards are provided
- leaderboard-ready JSONL/CSV exports are included in the artifact bundle
- leaderboard validation JSON/Markdown files are included in the artifact bundle
- ranked leaderboard report JSON/Markdown files are included in the artifact bundle
- artifact integrity JSON/Markdown files are included and pass sha256 verification
- provenance JSON/Markdown files record git, version, result hash, and config hashes
- benchmark suite JSON/Markdown files are included in the artifact bundle
- benchmark, external ASR adapter, VoiceWorld scenario, final-run, final-input acquisition, and unified contributor starter packs are included in the artifact bundle
- data source registry JSON/Markdown files are included in the artifact bundle
- adapter registry JSON/Markdown files are included in the artifact bundle
- model registry JSON/Markdown and NanoTurn model card JSON/Markdown files are included in the artifact bundle
- ASR reference collection JSON/Markdown, paper reference, BibTeX, coverage audit, and readiness audit files are included in the artifact bundle
- turn/full-duplex reference collection JSON/Markdown, acquisition plan, and coverage audit files are included in the artifact bundle
- scenario suite JSON/Markdown files are included in the artifact bundle
- case-study JSON/Markdown files are included in the artifact bundle
- paper parity JSON/Markdown files are included in the artifact bundle
- final experiment JSON/Markdown files are included in the artifact bundle
- final input collection JSON/status/Markdown files are included in the artifact bundle
- final-run config JSON/Markdown files are included in the artifact bundle
- final-run file audit JSON/Markdown files are included in the artifact bundle
- final-run action plan JSON/Markdown files are included in the artifact bundle
- final evidence matrix JSON/Markdown files are included in the artifact bundle
- roadmap status JSON/Markdown files are included in the artifact bundle
- claim evidence JSON/Markdown files are included in the artifact bundle
- `CITATION.cff` exists
- `docs/` exists

The release audit is expected to remain `NOT_READY` for runs that skip
NanoTurn training or omit the optional Lance data benchmark. A paper-facing run
should install `stable-asr[lance]`, include a checkpoint-backed NanoTurn row,
and then scale the smoke fixtures into the final benchmark suite.

Use `doctor --check-release-env` to fail early when the local environment cannot
produce a READY smoke audit:

```bash
python -m pip install -e ".[lance,train]"
stable-asr doctor --check-release-env
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke --strict
```

## Typical Command

```bash
stable-asr paper-release-audit \
  --repo-root . \
  --results runs/paper/smoke/paper_results.json \
  --artifacts-dir runs/paper/smoke/artifacts \
  --markdown-draft runs/paper/smoke/PAPER_DRAFT.md \
  --latex-draft runs/paper/smoke/paper.tex \
  --dataset-card runs/paper/smoke/DATASET_CARD.md \
  --experiment-card runs/paper/smoke/EXPERIMENT_CARD.md \
  --model-card runs/paper/smoke/MODEL_CARD.md
```

For a one-command smoke pass that generates the bundle, drafts, cards, and audit
files:

```bash
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke
```
