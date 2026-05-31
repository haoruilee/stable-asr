## Summary

- What changed:
- Why it matters:
- Linked issue:

## Contribution Track

Check one or more:

- [ ] final data acquisition
- [ ] external ASR adapter or transcript converter
- [ ] VoiceWorld scenario
- [ ] benchmark submission or leaderboard artifact
- [ ] core platform code
- [ ] docs, CI, packaging, or release gates

## Evidence

Commands run:

```bash

```

Attach or list generated artifacts:

- 

## Data, License, And Provenance

- [ ] No placeholder data is claimed as final-scale evidence.
- [ ] Upstream license, model terms, or data consent are documented where relevant.
- [ ] External code is not vendored unless the copied scope and license are explicitly documented.
- [ ] Generated files can be traced back to configs, seeds, commands, or source manifests.

## Final-Scale Impact

- [ ] This PR does not affect M5 final-scale evidence.
- [ ] This PR stages or verifies an M5 input and links the relevant `final-acquisition-pack` checklist row.
- [ ] This PR changes final-run configs, schemas, metrics, or release gates and updates docs/tests accordingly.

## Required Local Checks

Run the narrow checks for your change plus any applicable commands below:

```bash
python3 -m pytest -q
python3 -m stable_asr.cli roadmap-status --roadmap configs/roadmap/stable_asr_roadmap.json --validate-only
python3 -m stable_asr.cli paper-release-smoke --output-dir /tmp/stable-asr-release-smoke
```

For contribution packs:

```bash
stable-asr benchmark-pack --output-dir /tmp/stable-asr-benchmark-pack
stable-asr adapter-pack --output-dir /tmp/stable-asr-adapter-pack
stable-asr scenario-pack --output-dir /tmp/stable-asr-scenario-pack
stable-asr final-pack --output-dir /tmp/stable-asr-final-pack
stable-asr final-acquisition-pack --output-dir /tmp/stable-asr-final-acquisition-pack
```
