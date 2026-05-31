# External ASR Adapters

Stable-ASR should interoperate with strong ASR projects instead of copying them.

## Command Adapter Contract

An external ASR command should write Stable-ASR streaming transcript JSONL:

```json
{
  "id": "utt_001",
  "reference": "hello world",
  "hypothesis": "hello world",
  "duration": 1.2,
  "rtf": 0.15,
  "partials": [{"time": 0.4, "text": "hello"}],
  "words": [{"word": "hello", "start": 0.1, "end": 0.4}]
}
```

Evaluate it with:

```bash
stable-asr eval-asr-command \
  --name funasr_export \
  --command "python export_funasr.py --output {output}" \
  --output runs/funasr_streaming.jsonl
```

To start a new integration without guessing the file layout, generate an
adapter pack:

```bash
stable-asr adapter-pack --output-dir runs/adapter_pack
cd runs/adapter_pack
bash commands.sh
```

The generated pack includes an ASR manifest fixture, normalized streaming ASR
fixtures, adapter/reference registries, a command-comparison config, and
`scripts/export_streaming_template.py`. Replace the fixture copy in that script
with the upstream ASR call, then keep the same `--input-manifest` and
`--output` contract.

## Vendor Transcript Normalizer

For upstream systems that already export JSONL transcripts, normalize the
transcript before comparison:

```bash
python3 examples/commands/convert_vendor_transcript.py \
  --schema qwen3_asr \
  --input tests/fixtures/qwen3_asr_transcript_sample.jsonl \
  --output runs/qwen3_asr_streaming.jsonl

stable-asr eval-streaming-asr --input runs/qwen3_asr_streaming.jsonl
```

Supported transcript schemas include `whisper`, `funasr`, `whisper_cpp`,
`whisperx`, `qwen3_asr`, `firered_asr2s`, `sensevoice`, `moonshine`, and
`whisperkit`.

The demo config below runs two command-backed adapters through the same
comparison path without importing either upstream ASR package:

```bash
stable-asr compare-asr-commands \
  --config examples/configs/asr_vendor_adapter_demo.json \
  --report runs/asr_vendor_adapter/report.md
```

Final-scale command configs can be audited before any heavyweight ASR job is
executed:

```bash
stable-asr compare-asr-commands \
  --config configs/final/asr_command_compare.json \
  --validate-only \
  --require-input-manifest \
  --min-adapters 2
```

The checked-in `scripts/export_whisper_streaming.py` and
`scripts/export_funasr_streaming.py` bridges normalize precomputed upstream raw
exports, enrich missing references from `runs/final/asr_eval_manifest.jsonl`,
and fail when record IDs do not cover the shared manifest.

## Reference Coverage

The curated reference collection is stored in
`configs/references/asr_collections.json`. Release audit requires P0 and P1
references to have adapter, converter, command-template, or bridge-template
evidence in `configs/adapters/stable_asr_adapters.json`.
Run `stable-asr asr-collections --audit-readiness` before adding a new adapter
to confirm the upstream reference has a current review, explicit Stable-ASR
actions, and visible license-review notes.

Current command-template coverage includes classic ASR toolkits, Chinese-first
industrial systems, modern speech-LLM ASR families, timestamp/alignment tools,
and edge runtimes such as FireRedASR2S, Qwen3-ASR, SenseVoice, whisper.cpp,
WhisperX, Moonshine, and WhisperKit.
