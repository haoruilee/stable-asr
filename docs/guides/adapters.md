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

## Reference Coverage

The curated reference collection is stored in
`configs/references/asr_collections.json`. Release audit requires P0 and P1
references to have adapter, converter, command-template, or bridge-template
evidence in `configs/adapters/stable_asr_adapters.json`.

Current command-template coverage includes classic ASR toolkits, Chinese-first
industrial systems, modern speech-LLM ASR families, timestamp/alignment tools,
and edge runtimes such as FireRedASR2S, Qwen3-ASR, SenseVoice, whisper.cpp,
WhisperX, Moonshine, and WhisperKit.
