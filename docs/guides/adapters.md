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

## Reference Coverage

The curated reference collection is stored in
`configs/references/asr_collections.json`. Release audit requires P0 and P1
references to have adapter, converter, command-template, or bridge-template
evidence in `configs/adapters/stable_asr_adapters.json`.

Current command-template coverage includes classic ASR toolkits, Chinese-first
industrial systems, modern speech-LLM ASR families, timestamp/alignment tools,
and edge runtimes such as FireRedASR2S, Qwen3-ASR, SenseVoice, whisper.cpp,
WhisperX, Moonshine, and WhisperKit.
