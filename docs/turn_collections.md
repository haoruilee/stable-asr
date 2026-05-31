# Turn And Full-Duplex Reference Collections

Stable-ASR tracks turn-taking and full-duplex references separately from general
ASR toolkits. The curated registry lives at
`configs/references/turn_collections.json` and can be validated or rendered:

```bash
stable-asr turn-collections --registry configs/references/turn_collections.json --validate-only
stable-asr turn-collections --output runs/TURN_COLLECTIONS.md
stable-asr turn-collections --audit-coverage --output runs/TURN_COLLECTION_COVERAGE.md
stable-asr turn-collections --format acquisition-markdown --output runs/TURN_COLLECTION_ACQUISITION.md
```

The registry is not a vendoring list. It records which upstream projects should
shape Stable-ASR's turn/action labels, VoiceWorld scenarios, external prediction
adapters, and endpointing baselines.

## Initial Coverage

- Turn detection and state models: Smart Turn and Easy Turn.
- Full-duplex benchmark references: Full-Duplex-Bench.
- Turn prediction objectives: Voice Activity Projection.
- Voice-agent integration references: Pipecat.
- VAD endpointing baselines: Silero VAD and WebRTC VAD.

## Project Rule

Every substantial turn-taking adapter, endpointing baseline, VoiceWorld
scenario, or full-duplex benchmark feature should either link to this registry
or add a new entry first.

`turn-collections --audit-coverage` checks whether required P0 references have
evidence in the data-source registry or adapter registry. For v0 this means
Smart Turn, Easy Turn, Full-Duplex-Bench, and VAP must have a converter,
prediction adapter, scenario bridge, or template.

`turn-collections --format acquisition-markdown` turns the same registry into a
collection plan with P0 acquisition order, evidence targets, and license-review
flags. It is useful for final-scale work because it distinguishes real external
predictions and scenario bridges from registry-only intent.
