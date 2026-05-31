# VoiceWorld

VoiceWorld is the Stable-ASR scenario layer for full-duplex voice-agent
interaction. It plays the same platform role that environments play in
stable-worldmodel: a versioned suite with controllable factors of variation and
shared evaluation commands.

The default suite is `configs/scenarios/stable_asr_voiceworld_v0.json`.

```bash
stable-asr scenario-suite --suite configs/scenarios/stable_asr_voiceworld_v0.json --validate-only
stable-asr eval-scenario --episodes 21 --seed 0 --baseline vad_pause
stable-asr scenario-pack --output-dir runs/scenario_pack
```

Python API:

```python
import stable_asr as sasr
from stable_asr.models.baselines import TextTurnBaseline

world = sasr.World("stable_asr_voiceworld_v0", num_envs=4, seed=0)
records = world.collect("runs/voiceworld_demo.jsonl", episodes=100)
report = world.evaluate(TextTurnBaseline(), dataset="runs/voiceworld_demo.jsonl")

print(world.spec.to_dict())
print(report.to_markdown())
```

`World(...)` is the Stable-ASR counterpart to stable-worldmodel's environment
entrypoint. It gives researchers one object for scenario sampling, manifest
collection, and policy/model evaluation without replacing lower-level dataset
or evaluator APIs.

## Scenario Coverage

- normal question
- incomplete pause
- listener backchannel
- wait or hold command
- user interruption
- side conversation
- ambient speech
- noisy far-field speech
- code switching

## Factors Of Variation

The v0 suite tracks factors such as pause length, SNR, reverb, speaking rate,
overlap offset, network jitter, far-field distance, assistant speaking state,
and code-switch ratio. Final-scale evidence requires real or licensed audio
examples, not only synthetic fixtures.
