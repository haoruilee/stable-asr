# SenseVoice License Review

## Decision

- status: approved_link_or_command_adapter_no_vendoring
- reviewer: haoruilee
- reviewed_at: 2026-06-01
- upstream_license_field: `see_upstream`
- source_url: https://github.com/FunAudioLLM/SenseVoice
- docs_url: https://github.com/FunAudioLLM/SenseVoice
- approved_uses: citation, documentation, command-adapter integration, prediction-manifest conversion, local evaluation of user-provided outputs
- prohibited_uses: vendoring upstream source code, redistributing upstream model weights, redistributing upstream datasets, implying upstream endorsement
- required_notices: keep upstream links and license notes with any adapter documentation; run a fresh human license review before vendoring any third-party code, weights, or data

This review is intentionally conservative for Stable-ASR v0. It allows reproducible external comparison through links and normalized outputs while keeping redistribution out of scope.
