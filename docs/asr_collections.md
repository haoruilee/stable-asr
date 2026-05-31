# ASR Reference Collections

Stable-ASR should be built with explicit awareness of the strongest open-source
ASR ecosystem projects. The curated registry lives at
`configs/references/asr_collections.json` and can be validated or rendered:

```bash
stable-asr asr-collections --registry configs/references/asr_collections.json --validate-only
stable-asr asr-collections --output runs/ASR_COLLECTIONS.md
stable-asr asr-collections --format paper-markdown --output runs/ASR_REFERENCES.md
stable-asr asr-collections --format bibtex --output runs/ASR_REFERENCES.bib
stable-asr asr-collections --audit-coverage --output runs/ASR_COLLECTION_COVERAGE.md
stable-asr asr-collections --audit-coverage --require-priority p0 --require-priority p1
```

The registry is not a vendoring list. It records what each upstream project is
useful for, how Stable-ASR should learn from it, and which adapters or
benchmarks deserve priority.

`--format paper-markdown` and `--format bibtex` turn the same curated registry
into related-work drafting artifacts, so the paper can cite upstream projects
without maintaining a separate manual reference list.

## Initial Coverage

- Classic ASR toolkit references: Kaldi.
- Research training toolkits: ESPnet, NeMo Speech, SpeechBrain, icefall.
- Production and industrial ASR references: WeNet, FunASR, FireRedASR2S.
- Data-layer references: Lhotse.
- Deployment/runtime references: sherpa-onnx, whisper.cpp, faster-whisper, WhisperKit.
- Model-family and hub references: OpenAI Whisper, Qwen3-ASR, SenseVoice, Moonshine, Hugging Face Transformers ASR.
- Timestamp/alignment references: WhisperX.

## Project Rule

Every substantial adapter, benchmark, data recipe, or deployment feature should
either link to an entry in this registry or add a new entry first. That keeps
Stable-ASR grounded in the existing ASR ecosystem instead of becoming an
isolated toolkit.

`asr-collections --audit-coverage` checks whether required reference priorities
have evidence in the adapter registry. By default it requires P0 references such
as FunASR, WeNet, Lhotse, sherpa-onnx, and Whisper to have at least an
implemented converter, command template, or bridge template. For release review,
require both P0 and P1 so Kaldi, ESPnet, NeMo Speech, SpeechBrain, icefall,
SenseVoice, WhisperX, Moonshine, faster-whisper, and Hugging Face Transformers
ASR also have explicit adapter or bridge plans.
