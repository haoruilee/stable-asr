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
