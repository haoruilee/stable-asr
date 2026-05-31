# Turn API

Core turn-taking entry points:

- `stable_asr.turn.labels.TURN_LABELS`
- `stable_asr.turn.labels.ACTION_LABELS`
- `stable_asr.turn.policy.TurnPolicy`
- `stable_asr.models.baselines.RuleEndpointBaseline`
- `stable_asr.models.baselines.VADPauseBaseline`
- `stable_asr.models.baselines.TextTurnBaseline`
- `stable_asr.train.turn_trainer.train_nanoturn`

Example:

```python
from stable_asr.data.registry import load_turn_records
from stable_asr.eval.turn_eval import evaluate_turn_records
from stable_asr.models.baselines import VADPauseBaseline

records = load_turn_records("examples/data/turn_demo.jsonl")
report = evaluate_turn_records(records, VADPauseBaseline())
print(report.to_dict())
```
