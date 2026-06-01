# Turn And Full-Duplex Reference Collections

Stable-ASR tracks turn-taking and full-duplex references separately from general
ASR toolkits. The curated registry lives at
`configs/references/turn_collections.json` and can be validated or rendered:

```bash
stable-asr turn-collections --registry configs/references/turn_collections.json --validate-only
stable-asr turn-collections --output runs/TURN_COLLECTIONS.md
stable-asr turn-collections --audit-coverage --require-priority p0 --require-priority p1 --output runs/TURN_COLLECTION_COVERAGE.md
stable-asr turn-collections --format acquisition-markdown --output runs/TURN_COLLECTION_ACQUISITION.md
stable-asr turn-collections --format source-manifest --output runs/TURN_COLLECTION_SOURCE_MANIFEST.json
stable-asr reference-workqueue --output runs/REFERENCE_WORKQUEUE.md
stable-asr reference-workqueue --format evidence-markdown --output runs/REFERENCE_EVIDENCE_TEMPLATES.md
stable-asr reference-workqueue --audit-evidence --output runs/REFERENCE_EVIDENCE_AUDIT.md
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

`turn-collections --audit-coverage` checks whether required references have
evidence in the data-source registry or adapter registry. Use
`--require-priority p0 --require-priority p1` for release-facing coverage across
Smart Turn, Easy Turn, Full-Duplex-Bench, VAP, Pipecat, Silero VAD, and WebRTC
VAD. Each required item must have a converter, prediction adapter, scenario
bridge, runtime bridge, or endpointing template.

`turn-collections --format acquisition-markdown` turns the same registry into a
collection plan with P0 acquisition order, evidence targets, and license-review
flags. It is useful for final-scale work because it distinguishes real external
predictions and scenario bridges from registry-only intent.

`turn-collections --format source-manifest` writes the same source/docs URLs,
license policy, license review targets, acquisition tracks, evidence targets,
and Stable-ASR action queues as JSON, so turn adapters and VoiceWorld bridges
can be collected without relying on a manually copied checklist.

`reference-workqueue` merges the turn source manifest with the ASR source
manifest into a single contributor queue. It is useful when collecting external
turn predictions, scenario bridges, command ASR exports, and license reviews in
one release plan. Use `--format assignments-tsv` or
`--format assignments-markdown` when the queue needs owners, due dates, and
release-blocker status, then run `reference-assignment-audit` to surface
unassigned owners, missing due dates, missing evidence, and unresolved license
reviews.

`reference-workqueue --format evidence-markdown` renders per-reference evidence
templates for the same queue, including required version, input, command,
output, metric, failure-note, and license-decision sections.

Run `reference-workqueue --audit-evidence` when you need a direct readiness
answer from the generated queue. It checks evidence targets and license-review
files on disk, so registry entries and source manifests cannot be mistaken for
completed collection work.
