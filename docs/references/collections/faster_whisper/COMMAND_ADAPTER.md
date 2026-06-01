# faster-whisper Stable-ASR Reference Evidence

## Upstream version and source

- reference_id: `faster_whisper`
- source_url: https://github.com/SYSTRAN/faster-whisper
- docs_url: https://github.com/SYSTRAN/faster-whisper
- acquisition_track: `runtime command adapter`
- priority: `p1`
- local_review_date: `2026-06-01`

## Inputs used

Stable-ASR uses this project as an external reference or command-adapter target. The local final run keeps raw model outputs in normalized Stable-ASR manifests under `runs/final/` and does not vendor upstream checkpoints, training corpora, or source trees.

## Command, script, or bridge implementation notes

Stable-ASR action plan:
- plan_command_adapter_recipe
- compare_rtf
- track_quantized_inference

Adapter policy: `permissive_with_notice`. The intended integration is a command-backed or prediction-manifest bridge that converts upstream outputs into Stable-ASR schemas before evaluation.

## Output paths and schema or validation commands

- evidence_target: `docs/references/collections/faster_whisper/COMMAND_ADAPTER.md`
- license_review_target: `docs/references/collections/faster_whisper/LICENSE_REVIEW.md`
- validation: `stable-asr reference-workqueue --audit-evidence --require-content --repo-root .`
- final bundle target: `runs/final/artifacts/REFERENCE_WORKQUEUE.md`

## Metrics, examples, or failure notes relevant to Stable-ASR

Reference use: Study practical Whisper deployment, batched inference, quantization, and latency-sensitive transcription.

For final local evidence, outputs are evaluated through normalized benchmark artifacts rather than by redistributing upstream implementation files.

## License and redistribution decision

- upstream_license_field: `MIT`
- redistribution_decision: link or command adapter only unless a separate license review approves vendoring.
- blocked_by: none
