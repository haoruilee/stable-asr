# Smart Turn Stable-ASR Reference Evidence

## Upstream version and source

- reference_id: `smart_turn`
- source_url: https://github.com/pipecat-ai/smart-turn
- docs_url: https://docs.pipecat.ai/pipecat-cloud/guides/smart-turn
- acquisition_track: `turn prediction export`
- priority: `p0`
- local_review_date: `2026-06-01`

## Inputs used

Stable-ASR uses this project as an external reference or command-adapter target. The local final run keeps raw model outputs in normalized Stable-ASR manifests under `runs/final/` and does not vendor upstream checkpoints, training corpora, or source trees.

## Command, script, or bridge implementation notes

Stable-ASR action plan:
- maintain_prediction_converter
- compare_turn_completion_probabilities
- audit_voice_agent_endpointing_errors

Adapter policy: `link_or_command_adapter_until_reviewed`. The intended integration is a command-backed or prediction-manifest bridge that converts upstream outputs into Stable-ASR schemas before evaluation.

## Output paths and schema or validation commands

- evidence_target: `docs/references/collections/smart_turn/PREDICTION_EXPORT.md`
- license_review_target: `docs/references/collections/smart_turn/LICENSE_REVIEW.md`
- validation: `stable-asr reference-workqueue --audit-evidence --require-content --repo-root .`
- final bundle target: `runs/final/artifacts/REFERENCE_WORKQUEUE.md`

## Metrics, examples, or failure notes relevant to Stable-ASR

Reference use: Use as an audio-only external turn detector baseline and prediction-export target.

For final local evidence, outputs are evaluated through normalized benchmark artifacts rather than by redistributing upstream implementation files.

## License and redistribution decision

- upstream_license_field: `see_upstream`
- redistribution_decision: link or command adapter only unless a separate license review approves vendoring.
- blocked_by: license_review_before_vendoring
