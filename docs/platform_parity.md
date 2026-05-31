# Platform Parity

Stable-ASR tracks repository-level parity with the stable-worldmodel platform
shape. The goal is not to copy world-model tasks, but to keep the ASR platform
complete enough to support reproducible data, baselines, scenarios, solvers,
release artifacts, documentation, and contributor workflows.

The registry lives at `configs/platform/stable_worldmodel_parity.json`.

```bash
stable-asr platform-parity --registry configs/platform/stable_worldmodel_parity.json --validate-only
stable-asr platform-parity --output runs/PLATFORM_PARITY.md
stable-asr platform-parity --json
```

## What It Checks

- repository identity: README, install, quick start, docs, citation, roadmap
- installable CLI surface and CI smoke coverage
- data format registry with JSONL, Parquet, and Lance backends
- VoiceWorld scenario suite and factors of variation
- baseline, adapter, NanoTurn, and policy solver surface
- paper/release pipeline and final-run configs
- contributor extension packs and GitHub templates
- ASR and turn/full-duplex reference collections

This is a repository-shape audit. It complements `paper-parity-audit`, which
checks paper-claim evidence and final-scale gaps, and `paper-release-audit`,
which checks paper artifact bundles and release gates.
