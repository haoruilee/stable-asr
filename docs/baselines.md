# Baselines

Stable-ASR separates model outputs from interaction policies. A turn model emits
probabilities or labels; a policy decides whether the voice agent should speak,
listen, hold, or stop TTS.

## Implemented Baselines

| Baseline | Interface | Input | Output | Purpose |
| --- | --- | --- | --- | --- |
| `rule_endpoint` | `TurnPredictor` | manifest metadata | turn probabilities | lowest endpointing baseline |
| `vad_pause` | `TurnPredictor` | pause and overlap metadata | turn probabilities | industrial pause-threshold baseline |
| `text_turn` | `TurnPredictor` | reference or ASR text | turn probabilities | semantic text-only baseline |
| `prediction_manifest` | `TurnPredictor` | external prediction JSONL | turn probabilities | bridge for SmartTurn/EasyTurn/VAP-style outputs |
| `nanoturn_pico` | checkpoint predictor | manifest/audio-derived features | turn probabilities | trainable lightweight baseline |
| `nanoturn_nano` | checkpoint predictor | manifest/audio-derived features | turn probabilities | larger trainable NanoTurn baseline with audited config |

## External Systems

External ASR or turn-taking systems should not be vendored into Stable-ASR.
Instead, wrap them through:

- `convert-predictions` for SmartTurn, EasyTurn, and VAP-style future activity turn outputs
- `convert-asr-transcript` for Whisper, FunASR, Qwen3-ASR, FireRedASR2S, WhisperX, whisper.cpp, SenseVoice, Moonshine, and WhisperKit transcript exports
- `eval-asr-command` and `compare-asr-commands` for command-backed ASR systems
- adapter registry entries in `configs/adapters/stable_asr_adapters.json`

## Policy Search

```bash
stable-asr optimize-policy \
  --dataset examples/data/turn_demo.jsonl \
  --baseline vad_pause \
  --output runs/policy.json
```

The default objective is cost-sensitive: false completes and missed interrupts
cost more than small latency changes.
