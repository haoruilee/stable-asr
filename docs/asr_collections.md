# ASR Reference Collections

Stable-ASR should be built with explicit awareness of the strongest open-source
ASR ecosystem projects. The curated registry lives at
`configs/references/asr_collections.json` and can be validated or rendered:

```bash
stable-asr asr-collections --registry configs/references/asr_collections.json --validate-only
stable-asr asr-collections --output runs/ASR_COLLECTIONS.md
stable-asr asr-collections --format paper-markdown --output runs/ASR_REFERENCES.md
stable-asr asr-collections --format bibtex --output runs/ASR_REFERENCES.bib
stable-asr asr-collections --format acquisition-markdown --output runs/ASR_COLLECTION_ACQUISITION.md
stable-asr asr-collections --format source-manifest --output runs/ASR_COLLECTION_SOURCE_MANIFEST.json
stable-asr asr-collections --audit-coverage --output runs/ASR_COLLECTION_COVERAGE.md
stable-asr asr-collections --audit-coverage --require-priority p0 --require-priority p1
stable-asr asr-collections --audit-readiness --output runs/ASR_COLLECTION_READINESS.md
stable-asr asr-collections --audit-licenses --output runs/ASR_COLLECTION_LICENSE_REVIEW.md
stable-asr asr-collections --audit-licenses --require-license-reviewed
stable-asr reference-workqueue --output runs/REFERENCE_WORKQUEUE.md
```

The registry is not a vendoring list. It records what each upstream project is
useful for, how Stable-ASR should learn from it, and which adapters or
benchmarks deserve priority.

`--format paper-markdown` and `--format bibtex` turn the same curated registry
into related-work drafting artifacts, so the paper can cite upstream projects
without maintaining a separate manual reference list.

`--format acquisition-markdown` turns the registry into a concrete collection
plan: P0 acquisition order, adapter or bridge track, license-review flag, and
the expected evidence artifact for each upstream reference.

`--format source-manifest` writes a machine-readable source manifest with
source/docs URLs, license policy, license review targets, acquisition tracks,
evidence targets, citation keys, and Stable-ASR action queues. Use it as the
work queue for adapter, transcript export, and collection-review tasks.

`reference-workqueue` merges this ASR source manifest with the turn/full-duplex
source manifest into one P0/P1 contributor queue. The generated JSON/Markdown
names the next action, evidence target, and license-review blocker for each
upstream reference without claiming that the collection work is already done.
Use `--format assignments-tsv` or `--format assignments-markdown` to create an
owner-fillable tracker for collection, evidence, and license-review work.

`--audit-licenses` renders the reuse policy for each reference and names the
manual review file to fill before copying upstream code, weights, fixtures, or
long snippets. Without `--require-license-reviewed`, this is an advisory report
that keeps `see_upstream` projects usable through link-only notes or command
adapters. With `--require-license-reviewed`, unresolved P0/P1 reviews fail the
gate for final release or vendoring decisions.

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

`asr-collections --audit-readiness` is the release-facing variant. It checks
the registry review date, P0/P1 adapter evidence, Stable-ASR action plans, and
license-review warnings such as `see_upstream` entries. The warnings are kept
visible in paper bundles so adapters can interoperate with upstream projects
without accidentally implying that Stable-ASR vendors or relicenses them.

`asr-collections --audit-licenses` is narrower: it records whether each entry is
currently permissive-with-notice or link/command-adapter-only until a human
license review is staged at `runs/collections/<reference>/LICENSE_REVIEW.md`.
