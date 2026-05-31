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
- `artifacts/`
- `PAPER_DRAFT.md`
- `paper.tex`
- `DATASET_CARD.md`
- `EXPERIMENT_CARD.md`
- `release_audit.json`
- `RELEASE_AUDIT.md`

Useful variants:

```bash
stable-asr paper-release-smoke --skip-train
stable-asr paper-release-smoke --strict
```

Use `--skip-train` for fast structural checks. Use `--strict` in final
environments where missing Lance rows, NanoTurn checkpoints, or final-scale
inputs should fail CI.
