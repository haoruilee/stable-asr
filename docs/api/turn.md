# Turn API

Core turn-taking entry points:

- `stable_asr.turn.labels.TURN_LABELS`
- `stable_asr.turn.labels.ACTION_LABELS`
- `stable_asr.turn.policy.TurnPolicy`
- `stable_asr.models.baselines.RuleEndpointBaseline`
- `stable_asr.models.baselines.VADPauseBaseline`
- `stable_asr.models.baselines.TextTurnBaseline`
- `stable_asr.train.turn_trainer.train_nanoturn`
- `stable_asr.train.framework.NanoTurnRunConfig`
- `stable_asr.train.framework.fit_nanoturn`

Example:

```python
from stable_asr.data.registry import load_turn_records
from stable_asr.eval.turn_eval import evaluate_turn_records
from stable_asr.models.baselines import VADPauseBaseline

records = load_turn_records("examples/data/turn_demo.jsonl")
report = evaluate_turn_records(records, VADPauseBaseline())
print(report.to_dict())
```

Training example:

```python
from stable_asr.data.manifest import load_manifest
from stable_asr.train.framework import NanoTurnRunConfig, fit_nanoturn

records = load_manifest("examples/data/turn_demo.jsonl")
result = fit_nanoturn(
    records,
    output_dir="runs/nanoturn",
    config=NanoTurnRunConfig(
        epochs=5,
        batch_size=2,
        validation_split=0.25,
        optimizer="adamw",
        checkpoint_interval=1,
    ),
)
print(result.metrics["final_accuracy"])
```
