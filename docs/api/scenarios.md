# Scenario API

VoiceWorld-style scenario evaluation is the Stable-ASR counterpart to
environment suites in stable-worldmodel.

Core entry points:

- `stable_asr.scenarios.synthetic_turn.generate_synthetic_turn_records`
- `stable_asr.scenarios.synthetic_turn.write_synthetic_turn_manifest`
- `stable_asr.scenarios.voice_world.evaluate_voice_world`
- `stable_asr.scenarios.World`
- `stable_asr.scenarios.suites.load_scenario_suite`

The default scenario suite lives at:

```text
configs/scenarios/stable_asr_voiceworld_v0.json
```

High-level world API:

```python
import stable_asr as sasr

world = sasr.World("sdx/zh-full-duplex-mini-v1", num_envs=8, seed=0)
records = world.sample(episodes=32)
report = world.evaluate(baseline="vad_pause", episodes=32)
```

It covers incomplete pauses, backchannels, wait/hold utterances, interruptions,
side speech, ambient speech, noisy far-field speech, and code-switching.
