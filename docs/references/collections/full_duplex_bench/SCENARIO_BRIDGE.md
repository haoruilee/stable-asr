# Full-Duplex-Bench Stable-ASR Reference Evidence

## Upstream version and source

- reference_id: `full_duplex_bench`
- source_url: https://github.com/DanielLin94144/Full-Duplex-Bench
- docs_url: https://github.com/DanielLin94144/Full-Duplex-Bench
- acquisition_track: `scenario benchmark bridge`
- priority: `p0`
- local_review_date: `2026-06-01`

## Inputs used

Stable-ASR uses this project as an external reference or command-adapter target. The local final run keeps raw model outputs in normalized Stable-ASR manifests under `runs/final/` and does not vendor upstream checkpoints, training corpora, or source trees.

## Command, script, or bridge implementation notes

Stable-ASR action plan:
- maintain_external_manifest_converter
- align_voiceworld_scenario_taxonomy
- compare_overlap_handling_metrics

Adapter policy: `link_or_command_adapter_until_reviewed`. The intended integration is a command-backed or prediction-manifest bridge that converts upstream outputs into Stable-ASR schemas before evaluation.

## Output paths and schema or validation commands

- evidence_target: `docs/references/collections/full_duplex_bench/SCENARIO_BRIDGE.md`
- license_review_target: `docs/references/collections/full_duplex_bench/LICENSE_REVIEW.md`
- validation: `stable-asr reference-workqueue --audit-evidence --require-content --repo-root .`
- final bundle target: `runs/final/artifacts/REFERENCE_WORKQUEUE.md`

## Metrics, examples, or failure notes relevant to Stable-ASR

Reference use: Use as a benchmark-design reference for VoiceWorld scenario coverage and overlap handling.

For final local evidence, outputs are evaluated through normalized benchmark artifacts rather than by redistributing upstream implementation files.

## License and redistribution decision

- upstream_license_field: `see_upstream`
- redistribution_decision: link or command adapter only unless a separate license review approves vendoring.
- blocked_by: license_review_before_vendoring
