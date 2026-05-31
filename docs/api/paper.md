# Paper API

Stable-ASR treats paper artifacts as first-class reproducibility outputs.

Core entry points:

- `stable_asr.paper.experiments.run_paper_smoke`
- `stable_asr.paper.artifacts.paper_artifact_bundle`
- `stable_asr.paper.release_smoke.run_paper_release_smoke`
- `stable_asr.paper.audit.audit_paper_release`
- `stable_asr.paper.parity.audit_paper_parity`
- `stable_asr.paper.claims.audit_claims`

Example:

```python
from stable_asr.paper.release_smoke import run_paper_release_smoke

result = run_paper_release_smoke("runs/paper/release_smoke")
print(result.to_text())
```
