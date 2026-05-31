# Contributing

Stable-ASR is being built as a reproducible research platform. Contributions
should keep APIs small, tests deterministic, and experiment outputs traceable
to configs and seeds.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest
```

## Current Priorities

1. Keep the turn manifest schema stable and well tested.
2. Expand baseline evaluation before adding larger neural models.
3. Add scenario generators with explicit factors of variation.
4. Add data backends through the format registry instead of one-off loaders.
5. Make every paper-facing experiment reproducible from a config.

## Contribution Tracks

Use the GitHub issue templates so each contribution carries the right evidence:

- final data acquisition: corpora, VoiceWorld records, external predictions,
  NanoTurn final artifacts, and paper bundle inputs
- external ASR adapters: command-backed ASR exports, transcript converters, and
  corpus bridges
- VoiceWorld scenarios: scenario records, factor coverage, consent, and
  validation commands
- benchmark submissions: turn, streaming ASR, data-layer, scenario, and policy
  leaderboard rows

Before opening final-scale data or adapter work, generate the relevant starter
pack locally:

```bash
stable-asr contributor-pack --output-dir runs/contributor_pack
stable-asr final-acquisition-pack --output-dir runs/final_acquisition_pack
stable-asr adapter-pack --output-dir runs/adapter_pack
stable-asr scenario-pack --output-dir runs/scenario_pack
stable-asr benchmark-pack --output-dir runs/benchmark_pack
```

## Pull Request Expectations

- Add tests for new behavior.
- Keep examples small enough to run quickly.
- Avoid introducing heavyweight dependencies into the base install.
- Document new CLI commands in `README.md`.
- Update `ROADMAP.md` when a milestone changes status.
- Link the relevant issue template and include the commands that generated or
  validated the submitted artifact.
- Fill out `.github/PULL_REQUEST_TEMPLATE.md`, including data/license notes and
  final-scale impact.
