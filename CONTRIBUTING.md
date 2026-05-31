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

## Pull Request Expectations

- Add tests for new behavior.
- Keep examples small enough to run quickly.
- Avoid introducing heavyweight dependencies into the base install.
- Document new CLI commands in `README.md`.
- Update `ROADMAP.md` when a milestone changes status.

