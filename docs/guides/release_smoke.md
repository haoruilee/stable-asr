# Release Smoke

`paper-release-smoke` is the one-command path for checking whether the repository
is approaching a stable-worldmodel-style platform release.

```bash
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke
```

It runs the smoke experiment, generates the artifact bundle, writes paper
drafts/cards, and then runs `paper-release-audit`.

Expected outputs:

- `paper/paper_results.json`
- `artifacts/` with tables, figures, hash manifests, and provenance manifests
- `PAPER_DRAFT.md`
- `paper.tex`
- `DATASET_CARD.md`
- `EXPERIMENT_CARD.md`
- `release_audit.json`
- `RELEASE_AUDIT.md`

Useful variants:

```bash
stable-asr doctor --check-release-env
stable-asr paper-release-smoke --skip-train
stable-asr paper-release-smoke --strict
```

Use `--skip-train` for fast structural checks. Use `--strict` in final
environments where missing Lance rows, NanoTurn checkpoints, or final-scale
inputs should fail CI.

## READY Smoke Environment

The paper-facing smoke path can only reach `READY` when both optional pieces are
available:

- Torch, for the checkpoint-backed NanoTurn baseline row.
- Lance, for the data-layer benchmark row required by the platform paper gate.

Prepare that environment with:

```bash
python -m pip install -e ".[lance,train]"
stable-asr doctor --check-release-env
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke --strict
```

Without those optional dependencies, `paper-release-smoke` should still generate
artifacts, but the release audit will stay `NOT_READY` and point to the missing
NanoTurn or Lance gates.
