# OpenAI Whisper Stable-ASR Reference Evidence

## Upstream version and source

- reference_id: `whisper`
- source_url: https://github.com/openai/whisper
- docs_url: https://github.com/openai/whisper/blob/main/model-card.md
- acquisition_track: `ASR command adapter`
- priority: `p0`
- local_review_date: `2026-06-01`

## Inputs used

Stable-ASR uses this project as an external reference or command-adapter target. The local final run keeps raw model outputs in normalized Stable-ASR manifests under `runs/final/` and does not vendor upstream checkpoints, training corpora, or source trees.

## Command, script, or bridge implementation notes

Stable-ASR action plan:
- maintain_transcript_converter
- benchmark_timestamp_stability
- track_streaming_limitations

Adapter policy: `permissive_with_notice`. The intended integration is a command-backed or prediction-manifest bridge that converts upstream outputs into Stable-ASR schemas before evaluation.

## Output paths and schema or validation commands

- evidence_target: `docs/references/collections/whisper/COMMAND_ADAPTER.md`
- license_review_target: `docs/references/collections/whisper/LICENSE_REVIEW.md`
- validation: `stable-asr reference-workqueue --audit-evidence --require-content --repo-root .`
- final bundle target: `runs/final/artifacts/REFERENCE_WORKQUEUE.md`

## Metrics, examples, or failure notes relevant to Stable-ASR

Reference use: Study model cards, transcription API shape, timestamp behavior, and robust multilingual baseline expectations.

For final local evidence, outputs are evaluated through normalized benchmark artifacts rather than by redistributing upstream implementation files.

## License and redistribution decision

- upstream_license_field: `MIT`
- redistribution_decision: link or command adapter only unless a separate license review approves vendoring.
- blocked_by: none
