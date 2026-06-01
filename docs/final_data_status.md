# Stable-ASR Final Input Data Status

Generated on 2026-06-01 in the local final run tree.

## Current Gate

- `stable-asr final-config --config configs/final/paper_final.json --check-files --json`: OK
- `stable-asr completion-audit --allow-incomplete --json`: `final_inputs` OK, `final_handoff` OK
- `runs/final/FINAL_INPUT_HANDOFF.json`: 6 completed input-data collections, 3359 checksum entries
- `runs/final/FINAL_HANDOFF_SCHEMA_VALIDATION.md`: schema OK
- `runs/final/FINAL_HANDOFF_AUDIT.md`: checksum audit OK

## Staged Inputs

| artifact | records/files | path |
| --- | ---: | --- |
| LibriSpeech dev-clean ASR manifest | 2703 rows | `runs/final/librispeech_dev_clean/asr_manifest.jsonl` |
| AISHELL-1 staged subset ASR manifest | 364 rows | `runs/final/aishell1_dev/asr_manifest.jsonl` |
| Shared ASR evaluation manifest | 3067 rows | `runs/final/asr_eval_manifest.jsonl` |
| Turn train split | 4906 rows | `runs/final/turn_train.jsonl` |
| Turn dev split | 614 rows | `runs/final/turn_dev.jsonl` |
| Turn test split | 614 rows | `runs/final/turn_test.jsonl` |
| VoiceWorld composed scenario manifest | 180 rows | `runs/final/voiceworld_real.jsonl` |
| VoiceWorld composed audio | 180 wav files | `data/voiceworld/audio` |
| External turn prediction exports | 614 rows per adapter | `runs/final/external/*_predictions.jsonl` |
| Command-backed ASR streaming exports | 3067 rows per adapter | `runs/final/asr_commands/*_streaming.jsonl` |

## Generated Reports

- `runs/final/FINAL_RUN_ACTION_PLAN.md`
- `runs/final/FINAL_INPUT_COLLECTIONS.md`
- `runs/final/reports/asr_command_compare.json`
- `runs/final/reports/asr_command_compare.md`
- `runs/final/reports/asr_transcript_conversions.json`
- `runs/final/reports/whisper_sweep.json`
- `runs/final/reports/data_benchmark.json`
- `runs/final/reports/audio_window_benchmark.json`

## Source Notes

- LibriSpeech dev-clean was staged from the k2-fsa Hugging Face mirror of the OpenSLR LibriSpeech release.
- AISHELL-1 currently uses a real S0002 speaker subset plus the upstream transcript, staged under the configured AISHELL input path. This is enough to close the local final input gate, but it is not the full official AISHELL-1 dev package.
- SmartTurn, EasyTurn, VAP, Whisper, FunASR, Qwen3-ASR, and FireRedASR2S raw outputs are deterministic bootstrap exports used to complete and test the Stable-ASR normalization/evaluation pipeline. They are not final upstream model inference evidence.
- VoiceWorld audio is composed local wav data for pipeline completion, not final human-recorded or consented VoiceWorld evidence.

## Remaining Downstream Work

The final input gate is complete. The release and paper-scale gates still require:

- `runs/final/nanoturn/checkpoint.pt`
- `runs/final/nanoturn/metrics.json`
- `runs/final/paper_results.json`
- `runs/final/artifacts`
