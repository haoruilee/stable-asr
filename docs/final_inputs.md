# Final Inputs

Stable-ASR separates structural smoke evidence from final-scale evidence. The
final paper run needs real corpora, real or explicitly composed VoiceWorld
records, external turn-model exports, command-backed ASR outputs, NanoTurn
artifacts, and a final paper bundle.

The machine-readable collection plan lives at:

```text
configs/final/input_collections.json
```

Validate it:

```bash
stable-asr final-inputs --registry configs/final/input_collections.json --validate-only
```

Render the local collection status:

```bash
stable-asr final-inputs \
  --registry configs/final/input_collections.json \
  --config configs/final/paper_final.json \
  --output runs/final/FINAL_INPUT_COLLECTIONS.md
```

The report does not create placeholder data. It checks the configured local
paths and prints the commands needed to stage, normalize, verify, and package
the final evidence.

Expected P0 collection groups:

- LibriSpeech dev-clean and AISHELL-1 dev corpus directories.
- Leakage-audited turn train/dev/test splits derived from prepared ASR manifests.
- Real VoiceWorld annotations and audio.
- SmartTurn, EasyTurn, and VAP raw prediction exports.
- Command-backed ASR output reports for external ASR systems.
- NanoTurn final checkpoint, metrics, ONNX export, and model card.
- Filled `FINAL_ASSIGNMENT_AUDIT.md`, `FINAL_INPUT_HANDOFF.json`, and
  `FINAL_HANDOFF_AUDIT.md` proving owner, due-date, release-blocker,
  license/consent, verification, staged path, and checksum evidence.
- Final `paper_results.json`, artifact bundle, archive, and release gates.

Use this together with:

```bash
stable-asr final-pack --output-dir runs/final_pack
stable-asr final-acquisition-pack --output-dir runs/final_acquisition_pack
stable-asr contributor-pack --output-dir runs/contributor_pack
stable-asr final-config --config configs/final/paper_final.json --plan-missing
stable-asr paper-status --repo-root .
```

`paper-status` includes the strict final assignment and handoff gates, so a
final run stays `NOT_READY` until the owner assignment tracker exists, passes
`final-assignment-audit --require-owner --require-due-date --require-ready`,
the `FINAL_ASSIGNMENT_AUDIT.md` evidence file is present, and the filled
`FINAL_INPUT_HANDOFF.json` passes `final-handoff-audit`.

`final-acquisition-pack` is the collaborator-facing version of the final input
plan. It writes a TSV/JSON staging checklist, owner assignment tracker, license
and consent review sheet, VoiceWorld recording checklist, and structured
handoff template so real corpora and external model outputs can be collected
without inventing placeholder evidence. Use `final-assignment-audit` before
handoff, then use `final-handoff-audit` on a filled handoff JSON before treating
the staged inputs as final release evidence. `paper-release-audit
--require-final-ready` now checks that assignment and handoff evidence exist
and audit cleanly.

Before handoff, use `final-assignment-audit` on the generated
`acquisition/assignments.json` file. The default audit reports unassigned
owners, missing due dates, and release blockers; strict mode can require
owners, due dates, and no blocking collections:

```bash
stable-asr final-assignment-audit \
  --input runs/final_acquisition_pack/acquisition/assignments.json \
  --require-owner \
  --require-due-date \
  --require-ready
```
