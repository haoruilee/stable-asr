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
- `artifacts/` with copied paper results, tables, figures, starter packs, hash manifests, and provenance manifests
- `PAPER_DRAFT.md`
- `paper.tex`
- `DATASET_CARD.md`
- `EXPERIMENT_CARD.md`
- `MODEL_CARD.md`
- `artifacts.tar.gz`
- `artifacts.tar.gz.sha256`
- `archive_verification.json`
- `ARCHIVE_VERIFICATION.md`
- `artifacts/PAPER_STATUS.md`
- `release_audit.json`
- `RELEASE_AUDIT.md`

Useful variants:

```bash
stable-asr doctor --check-release-env
stable-asr paper-release-smoke --skip-train
stable-asr paper-release-smoke --strict
stable-asr paper-release-smoke --require-final-ready
stable-asr paper-archive-verify --archive runs/paper/release_smoke/artifacts.tar.gz
```

Use `--skip-train` for fast structural checks. Use `--strict` in final
environments where missing Lance rows, NanoTurn checkpoints, or final-scale
inputs should fail CI. Use `--require-final-ready` when a job must fail until
the release audit is READY and real paper-scale corpora, external predictions,
and final artifact evidence are present. Use `paper-archive-verify` after moving the generated archive or
before attaching it to a release.

The command prints `final_inputs_ready`, `final_assignment_ready`, and
`final_handoff_ready`. Final scale remains `NO` until the data paths exist, the
assignment tracker has owners and due dates with no release blockers, the
strict `FINAL_ASSIGNMENT_AUDIT.md` evidence file is present, and the filled
handoff passes `final-handoff-audit --require-checksums`.

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
