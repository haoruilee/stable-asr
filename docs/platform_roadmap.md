# Stable-ASR Platform Expansion Roadmap

> **Status (2026-06-03): Superseded as the v1 paper plan.** The v1 paper
> direction is locked in [`ROADMAP.md`](../ROADMAP.md) §"Paper Direction
> (locked 2026-06-03)" — an *empirical-findings paper on LDC multilingual
> conversational corpora*, targeted at Interspeech 2026.
>
> This document is kept for reference. Sections 4–8 below (Data / Training
> / Inference / Deployment / Acceleration) describe the longer-term
> platform substrate that v2+ papers can build on. The v1 paper does **not**
> require the P1 / P2 deliverables here (streaming Conformer-T training,
> deployment subpackage, distributed training, runtime backend zoo). Those
> stay in this document as a future work-list, not as v1 dependencies.
>
> If a milestone here conflicts with the v1 plan in `ROADMAP.md`, the v1
> plan wins.

---

# Stable-ASR Platform Expansion Roadmap (original v0)

**Document status:** Planning. Authored 2026-06-03. Companion to `ROADMAP.md`
(turn-taking milestones M0–M5) and `configs/platform/stable_worldmodel_parity.json`.

This document scopes the work needed to lift Stable-ASR from a turn-taking
research toolkit into a *stable-worldmodel-class* real-time ASR **platform
paper** with credible coverage across four dimensions:

1. **Data** — corpora adapters, manifests, format registry
2. **Training** — model zoo, distributed training, SSL/transformer backbones
3. **Inference** — batch, streaming chunk decoder, ONNX, runtime backends
4. **Deployment** — quantization, server, edge, mobile, latency budget

Each dimension has a target acceleration story so the paper can claim
measurable speedups end-to-end.

---

## 1. Vision and Paper Target

### Paper one-liner

> *Stable-ASR: A Reproducible Platform for Real-Time ASR Systems and Full-Duplex
> Turn-Taking, with a Unified Data, Model, Inference, and Deployment Stack.*

### What "platform-class" means here

stable-worldmodel succeeded because it shipped: (a) a *registry* of data
formats, environments, and solvers; (b) reference baselines for every
registered task; (c) a reproducible paper-release pipeline; (d) measurable
performance numbers across configurations. Stable-ASR already mirrors (a),
(c), and parts of (d) — see `configs/platform/stable_worldmodel_parity.json`.
The remaining gap is (b) at *ASR-system* depth: a model zoo, streaming
runtime, and deployment story comparable to NeMo / icefall / WeNet.

### Non-goals

- Beating SOTA WER. Stable-ASR is a *platform* paper, not a model paper.
  Reference checkpoints exist to validate the platform plumbing.
- Re-implementing Conformer / Zipformer / Whisper from scratch. We adopt
  trained checkpoints from upstream and integrate them as first-class adapters.
- Production multi-tenant serving. Deployment story stops at
  *single-replica latency-correct* runtimes (Triton config, FastAPI, ONNX
  Runtime, TensorRT) — not autoscaling.

---

## 2. Reference Repositories

Every dimension below is calibrated against well-known projects so reviewers
can place Stable-ASR on a familiar map. None of these are dependencies; we
either *integrate via adapter* or *take the design as inspiration*.

### Streaming ASR engines

| Repo | What we adopt | Integration mode |
|---|---|---|
| [`k2-fsa/icefall`](https://github.com/k2-fsa/icefall) | Zipformer / Streaming Conformer-T recipes; manifest layout | adapter + recipe parity |
| [`k2-fsa/sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx) | Production-grade ONNX streaming runtime, multi-language | runtime backend |
| [`wenet-e2e/wenet`](https://github.com/wenet-e2e/wenet) | U2++ chunk-based streaming, LibTorch/ONNX/TRT runtimes | adapter + runtime ref |
| [`espnet/espnet`](https://github.com/espnet/espnet) | Recipe coverage, kaldi-style data prep | recipe parity |
| [`NVIDIA/NeMo`](https://github.com/NVIDIA/NeMo) | FastConformer, cache-aware streaming, Triton export | adapter + Triton template |
| [`alibaba-damo-academy/FunASR`](https://github.com/modelscope/FunASR) | Paraformer streaming, SenseVoice multilingual | adapter (already wired) |
| [`speechbrain/speechbrain`](https://github.com/speechbrain/speechbrain) | Recipe modularity, HuggingFace integration | adapter |

### Offline / encoder reference

| Repo | What we adopt |
|---|---|
| [`openai/whisper`](https://github.com/openai/whisper) | Whisper baseline + word-timestamp adapter (already wired via `scripts/run_whisper_streaming.py`) |
| [`SYSTRAN/faster-whisper`](https://github.com/SYSTRAN/faster-whisper) | CTranslate2 INT8/FP16 reference for offline acceleration numbers |
| [`m-bain/whisperX`](https://github.com/m-bain/whisperX) | Forced-alignment timestamps for streaming-eval reference traces |
| [`ggerganov/whisper.cpp`](https://github.com/ggerganov/whisper.cpp) | CPU/edge baseline (already wired) |
| [`huggingface/transformers`](https://github.com/huggingface/transformers) | Wav2Vec2 / HuBERT / WavLM / Whisper checkpoints for SSL turn models |

### Turn-taking / full-duplex

| Repo | Role |
|---|---|
| [`ErikEkstedt/VoiceActivityProjection`](https://github.com/ErikEkstedt/VoiceActivityProjection) | VAP reference (currently bootstrap predictions; replace with real inference — `models/baselines/vap.py:232`) |
| [`pipecat-ai/smart-turn`](https://github.com/pipecat-ai/smart-turn) | Smart-turn manifest converter (already wired in `data/converters/external.py`) |
| [`Full-Duplex-Bench`](https://github.com/IntelLabs/Full-Duplex-Bench) | Full-duplex eval converter (already wired) |
| [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad) | VAD reference for `vad_pause` baseline |

### Deployment & acceleration

| Repo | Role |
|---|---|
| [`microsoft/onnxruntime`](https://github.com/microsoft/onnxruntime) | Default cross-platform runtime, CUDA + CPU + DirectML EP |
| [`NVIDIA/TensorRT`](https://github.com/NVIDIA/TensorRT) | INT8/FP16 GPU latency reference |
| [`triton-inference-server/server`](https://github.com/triton-inference-server/server) | Multi-backend serving template + ensemble graph |
| [`pytorch/pytorch`](https://github.com/pytorch/pytorch) | `torch.compile`, `torch.export`, dynamic quantization |
| [`OpenVINO`](https://github.com/openvinotoolkit/openvino) | CPU edge inference (Intel) |
| [`intel/neural-compressor`](https://github.com/intel/neural-compressor) | Static quantization + AWQ for encoder weights |
| [`apple/coremltools`](https://github.com/apple/coremltools) | iOS / Apple Silicon path |
| [`tensorflow/tensorflow`](https://github.com/tensorflow/tensorflow) (TFLite) | Android / embedded reference |
| [`huggingface/optimum`](https://github.com/huggingface/optimum) | Transformers → ONNX/TRT/OpenVINO bridge |
| [`Lightning-AI/pytorch-lightning`](https://github.com/Lightning-AI/pytorch-lightning) / [`huggingface/accelerate`](https://github.com/huggingface/accelerate) | Distributed training reference patterns |

### Data corpora (already in `configs/datasets/stable_asr_sources.json`)

LibriSpeech, AISHELL-1, AISHELL-4, AMI IHM, ICSI, MagicData-RAMC, GigaSpeech,
MLS, TED-LIUM 3, VoxPopuli, CommonVoice, WenetSpeech, Switchboard / Fisher /
CallHome (LDC).

---

## 3. Phase Plan

The platform expansion is split into three phases targeting paper submission.
Dates are absolute and assume the new training machine is online by 2026-06-15.

| Phase | Window | Theme | Exit criterion |
|---|---|---|---|
| **P0 — Foundation parity** | 2026-06-03 → 2026-07-15 | Close stable-worldmodel parity gaps | All `configs/platform/stable_worldmodel_parity.json` items pass `--validate-only`; real VAP predictions; LDC + AMI in registry |
| **P1 — Platform depth** | 2026-07-15 → 2026-09-30 | Streaming ASR zoo, distributed training, deployment substrate | At least 3 ASR backbones run end-to-end (data → train → eval → ONNX → server); 5+ acceleration backends benchmarked |
| **P2 — Paper polish** | 2026-10-01 → 2026-11-30 | Full-scale runs, leaderboard, ablations, paper draft | Submission-ready paper artifacts; reproducible from clean clone in ≤ 4 hours on single 4080-class GPU for smoke; full-scale runs documented |

### P0 milestones (2026-06 → 2026-07)

| ID | Deliverable | Reference repo cue |
|---|---|---|
| P0.1 | Real VAP inference replaces bootstrap predictions | ErikEkstedt/VoiceActivityProjection |
| P0.2 | LDC adapter (commit `0b9a72c`) registered in `data/sources.py` + `recipes/public_corpora.py` + `configs/datasets/` | icefall LDC recipes |
| P0.3 | AMI IHM promoted from `runs/` to first-class `recipes/public_corpora.py` recipe | espnet AMI recipe |
| P0.4 | License/badge consistency (README MIT badge → Research Non-Commercial) | — |
| P0.5 | `paper/` 30-module surface compressed (merge {audit, integrity, provenance, completion}, {claims, evidence}, {final_*}); keep public CLI stable | — |
| P0.6 | `streaming/sweep.py` augmented with stateful chunk decoder reference (not just offline metric scan) | wenet U2++ chunk inference |

### P1 milestones (2026-07 → 2026-09)

| ID | Deliverable | Reference repo cue |
|---|---|---|
| P1.1 | `stable_asr/asr/` subpackage: 1 streaming ASR baseline (Streaming Conformer-T, LibriSpeech 100h) | k2-fsa/icefall |
| P1.2 | Add Whisper / Paraformer / Wav2Vec2 / HuBERT adapters as `stable_asr/models/asr/*.py` (checkpoint adapters, not retrains) | huggingface/transformers, FunASR |
| P1.3 | Distributed training (DDP + accelerate) on `train/framework.py` | huggingface/accelerate |
| P1.4 | `stable_asr/deploy/` subpackage: ONNX, TensorRT export, dynamic INT8, FastAPI server, Triton config template | sherpa-onnx, NVIDIA NeMo |
| P1.5 | `stable_asr/asr/streaming_decoder.py` — stateful chunk decoder loop with cache state, endpoint-triggered finalization | wenet U2++, NeMo cache-aware streaming |
| P1.6 | SSL turn baselines: HuBERT-frozen + linear head, WavLM + small adapter | huggingface/transformers |
| P1.7 | `bench_acceleration.py` extended: PyTorch / `torch.compile` / ONNX-CUDA / ONNX-CPU / TensorRT / OpenVINO / faster-whisper-ct2 | faster-whisper, onnxruntime |
| P1.8 | Edge benchmark: i9 CPU-only, RPi-class, Apple Silicon (via CoreML export) | whisper.cpp, coremltools |

### P2 milestones (2026-10 → 2026-11)

| ID | Deliverable |
|---|---|
| P2.1 | Full-scale AMI + LibriSpeech + LDC training runs, all checkpoints uploaded to HF |
| P2.2 | Cross-lingual transfer table (en → zh AMI/RAMC) |
| P2.3 | Latency-accuracy Pareto figure across all backends |
| P2.4 | Public leaderboard JSON merged into `paper/leaderboard.py` |
| P2.5 | `paper-release-smoke` reproducibility: cold clone → full pipeline ≤ 4 h on 4080 |
| P2.6 | Paper draft + supplementary, AAAI / Interspeech format |

---

## 4. Dimension 1 — Data

### Current state

- Format registry: `data/registry.py` (JSONL / Parquet / Lance — already at parity with stable-worldmodel data layer)
- Sources registry: `configs/datasets/stable_asr_sources.json` (12 entries)
- Recipes implemented: LibriSpeech, AISHELL-1, CommonVoice, WenetSpeech (`data/recipes/public_corpora.py`)
- Converters: smart-turn, easyturn, full-duplex-bench, whisper, funasr, vendor (`data/converters/external.py`)
- LDC adapter: just added (commit `0b9a72c`) but not yet registered

### Target

A *coverage matrix* on par with icefall's recipe list and FunASR's adapter set.

| Corpus | License | Languages | Recipe target | Reference impl |
|---|---|---|---|---|
| LibriSpeech | CC-BY-4.0 | en | ✅ done | icefall, espnet |
| AISHELL-1 | Apache | zh | ✅ done | icefall, FunASR |
| AISHELL-4 | CC-BY-4.0 | zh | P0 | espnet/aishell4 |
| AMI IHM | CC-BY-4.0 | en | P0 (promote from runs) | espnet/ami |
| CommonVoice | CC0 | multi | ✅ done | huggingface/datasets |
| WenetSpeech | CC-BY | zh | ✅ stub, P1 expand | wenet |
| GigaSpeech | apply | en | P1 | speechcolab/GigaSpeech |
| MLS | CC-BY | multi | P1 | facebook/mls |
| TED-LIUM 3 | CC-BY-NC-ND | en | P1 | espnet/tedlium3 |
| VoxPopuli | CC0 | multi | P1 | facebook/voxpopuli |
| MagicData-RAMC | RAIL | zh | P1 (cross-lingual transfer experiment) | magicdatatech/MagicData-RAMC |
| Switchboard / Fisher / CallHome | LDC | en | P1 (real conversational baseline) | LDC + custom (commit 0b9a72c starts this) |
| ICSI Meeting | CC-BY | en | P2 | espnet/icsi |
| KsponSpeech | KAIST | ko | P2 stretch | kosp2e/KsponSpeech |

### Action items

1. Register LDC + AMI in `data/sources.py` and `recipes/public_corpora.py` (P0.2, P0.3).
2. Add GigaSpeech / MLS / TED-LIUM3 / VoxPopuli recipes following the icefall layout (P1).
3. Add a `data/streaming/` partial-hypothesis schema upgrade so streaming
   transcripts carry per-chunk timing metadata, matching `whisperX` alignment output.
4. Sample-rate audit: enforce 16 kHz mono in all recipes; add `data/audio_audit.py`
   regression coverage for stereo and 8 kHz Switchboard.

---

## 5. Dimension 2 — Training Framework

### Current state

- `train/framework.py` (718 lines): single-GPU, AMP optional, cosine schedule, early stopping, TensorBoard
- `train/feature_cache.py` (462 lines): logmel cache to Parquet/Lance
- `train/turn_trainer.py` (293 lines): turn-task-specific dataset / loss
- 5 NanoTurn variants + 1 NanoTurnMicro TCN

### Target

Distributed-capable trainer with a model zoo that includes at least one
self-trained streaming ASR model and SSL-encoder turn baselines.

### Action items (P1)

1. **Distributed training.** Wire `accelerate` (preferred) or native DDP into
   `framework.py`. Acceptance: 2-GPU training of NanoTurnMicro on AMI gives
   1.7×+ throughput vs single-GPU. Reference: `huggingface/accelerate` examples.
2. **`torch.compile` integration** in `NanoTurnTrainer.__init__` behind a config
   flag. Acceptance: compile mode `reduce-overhead` measurable speedup on
   audio_seq path.
3. **ASR subpackage `stable_asr/asr/`**:
   - `asr/streaming_conformer.py` — Streaming Conformer-T (CTC + RNN-T heads),
     adapted from icefall's published recipe, ≤ 30M params
   - `asr/whisper_adapter.py` — wrap `openai/whisper` and `faster-whisper` checkpoints
   - `asr/wav2vec2_ctc.py` — HuggingFace Wav2Vec2-CTC adapter
   - `asr/paraformer_adapter.py` — FunASR Paraformer wrapper
4. **SSL turn models** in `stable_asr/turn/`:
   - `turn/nanoturn_hubert.py` — HuBERT-base frozen + linear head
   - `turn/nanoturn_wavlm.py` — WavLM-base + adapter
5. **Mixed-precision matrix.** Currently AMP is 0.72× on nano (Tensor Core
   underutilization, see `runs/bench_accel/acceleration_results.json`). After
   P1.1 backbones land, re-run the matrix; expect 1.5–2× on Conformer-class
   models.

### Reference patterns

- DDP launcher: `accelerate launch --multi_gpu`
- Mixed precision policy from icefall: bf16 on A100/H100, fp16 elsewhere
- Gradient accumulation: support to compensate for 16 GB 4080 VRAM during
  Conformer training
- Checkpointing: `torch.distributed.checkpoint` for sharded saves at scale

---

## 6. Dimension 3 — Inference Framework

### Current state

- Offline batch inference: `predict_batch()` measured at **10.8× single
  predict** on 614 records (memory.md verified)
- Streaming evaluation: `streaming/metrics.py`, `streaming/sweep.py` —
  but these consume *pre-computed* partial-hypothesis traces; **there is no
  real chunk-by-chunk decoder loop in-repo**
- ONNX export: `train/export.py` (single-shot, no caching)

### Target

A real streaming inference path with state caching, parity with wenet U2++ and
NeMo cache-aware streaming, plus a runtime backend matrix.

### Action items

1. **Stateful chunk decoder** (`stable_asr/asr/streaming_decoder.py`, P1.5):
   - Maintains encoder cache state across chunks
   - Produces incremental partial hypotheses with stable-prefix logic
     (already evaluated by `streaming/metrics.py:110`)
   - Endpoint trigger calls into a turn predictor → finalization
   - Reference: wenet `wenet/transformer/asr_model.py:forward_chunk`
2. **Runtime backend registry** (`stable_asr/inference/backends/`):
   - `pytorch_eager`
   - `pytorch_compiled` (torch.compile)
   - `onnx_runtime` (CPU + CUDA EP)
   - `tensorrt`
   - `openvino`
   - `ctranslate2` (faster-whisper)
   - `coreml`
   Each backend implements a `Backend.predict(batch)` interface; the unified
   `bench_acceleration.py` benchmark walks all of them.
3. **Streaming server** (`stable_asr/inference/server/`):
   - WebSocket audio-chunk streaming endpoint
   - Reference: sherpa-onnx websocket server, NeMo Riva
4. **Latency budget tooling** (`stable_asr/inference/latency.py`):
   - Per-chunk p50/p95/p99
   - First-token-latency, last-token-latency, finalization-latency
   - Reference: WhisperLive timing report

---

## 7. Dimension 4 — Deployment

### Current state

`stable_asr/train/export.py` (46 lines) → ONNX export only. No quantization,
no server, no Triton, no edge.

### Target

Subpackage `stable_asr/deploy/` covering: format export, quantization, serving,
edge profile.

```
stable_asr/deploy/
├── export/
│   ├── onnx.py                 # ✅ exists in train/export.py, move here
│   ├── tensorrt.py             # P1
│   ├── coreml.py               # P1 stretch
│   ├── tflite.py               # P2
│   └── openvino.py             # P1
├── quantize/
│   ├── dynamic_int8.py         # torch + onnx dynamic quant — P1
│   ├── static_int8.py          # calibration-based — P1
│   └── awq.py                  # weight-only quant — P2 stretch
├── serve/
│   ├── fastapi_server.py       # offline batch service — P1
│   ├── websocket_server.py     # streaming audio chunks — P1
│   ├── triton/
│   │   ├── config.pbtxt        # config template — P1
│   │   └── ensemble.pbtxt      # encoder + decoder ensemble — P2
│   └── docker/
│       └── Dockerfile          # P1
└── edge/
    ├── cpu_only.py             # i9-14900 / Ryzen reference — P1
    ├── rpi.py                  # ARM-class — P2 stretch
    └── apple_silicon.py        # CoreML reference — P2 stretch
```

### Reference repos

- **Triton config template** lifted from `triton-inference-server/server` examples
- **TensorRT export** patterns from NVIDIA NeMo (`nemo.export.tensorrt_llm`)
- **CTranslate2 path** mirrors `faster-whisper` to validate INT8 numbers
- **CoreML** uses `coremltools.convert` on the ONNX intermediate
- **OpenVINO** uses `optimum-intel` for HF model conversion

### Acceptance for P1.4

- Single command exports a NanoTurnMicro checkpoint to all 5 runtimes
- Single command spins up FastAPI + WebSocket server reading any of the runtimes
- Triton config validates with `triton-inference-server` `model-analyzer`
- Reproducible Docker image under 2 GB

---

## 8. Dimension 5 — Acceleration Benchmarks

### Current state

`runs/bench_accel/acceleration_results.json` (committed). Headline numbers:

| Track | Best speedup | Notes |
|---|---|---|
| Training (nano MLP, AMI 217k) | 1.25× (cosine + early stopping) | AMP gives 0.72× — model too small |
| Training (micro TCN, AMI 7k cached) | ~1.0× (all configs) | bottleneck is data, not compute |
| Inference (nano, batched) | 10.8× | best single number to date |
| Inference (nano, ONNX) | slower than batched PT-CUDA | expected for tiny model on GPU |

This is **not strong enough** for a platform paper — the numbers are
honest but small because the models are tiny. The fix is to land
Conformer-class backbones (P1.1) and re-run the matrix.

### Target benchmark surface (P1.7)

| Axis | Variants |
|---|---|
| Model | NanoTurnMicro, Streaming Conformer-T, Whisper-tiny, Wav2Vec2-base |
| Hardware | RTX 4080 Super (current), i9-14900KS CPU-only, Apple M-series (stretch) |
| Backend | PyTorch eager, torch.compile, ONNX-CUDA, ONNX-CPU, TensorRT FP16, TensorRT INT8, OpenVINO CPU, CTranslate2 INT8 |
| Batch | single, 4, 16, 64 |
| Sequence length | 1 s, 5 s, 30 s |
| Metric | throughput (rec/s), latency p50/p95/p99 (ms), peak memory (MB), accuracy delta vs FP32 |

### Headline claims to support in the paper

These are the claims the benchmark surface must produce; they are *targets*,
not yet measured:

1. **Training:** 1.5–2× from AMP+compile on Conformer-class; 1.7×+ scaling
   from 1 → 2 GPU
2. **Offline inference (encoder):** 5–8× from FP32 PyTorch → INT8 TensorRT
3. **Streaming inference (chunked):** real-time factor (RTF) ≤ 0.1 on RTX 4080
   for Streaming Conformer-T at 320 ms chunk
4. **Edge:** RTF ≤ 1.0 on i9-14900KS CPU-only for Whisper-base via CTranslate2
5. **Latency budget:** p95 first-token < 200 ms, p95 finalization < 350 ms

### Reference numbers (external, for paper comparison table)

- faster-whisper paper / blog: tiny INT8 ~12× over openai/whisper-tiny on CPU
- sherpa-onnx published benchmarks: Streaming Zipformer RTF ~0.05 on T4
- NeMo cache-aware streaming: latency ~80 ms on A100 for 80 ms chunk

---

## 9. Repository Surface After Expansion

```
stable_asr/
├── asr/                        # NEW (P1) — streaming ASR backbones + adapters
│   ├── streaming_conformer.py
│   ├── streaming_decoder.py
│   ├── whisper_adapter.py
│   ├── paraformer_adapter.py
│   └── wav2vec2_ctc.py
├── data/                       # existing — extended recipes + LDC
├── deploy/                     # NEW (P1) — export / quantize / serve / edge
│   ├── export/
│   ├── quantize/
│   ├── serve/
│   └── edge/
├── eval/                       # existing
├── inference/                  # NEW (P1) — runtime registry + latency tooling
│   ├── backends/
│   ├── latency.py
│   └── server/
├── models/                     # existing — baselines
│   ├── adapters/
│   └── baselines/
├── paper/                      # existing — slim to ≤15 modules (P0.5)
├── references/                 # existing
├── scenarios/                  # existing
├── streaming/                  # existing — eval-side
├── train/                      # existing — DDP + compile (P1.3)
└── turn/                       # existing — extended with HuBERT/WavLM (P1.6)
```

---

## 10. Definition of Done (paper submission readiness)

A reviewer cloning Stable-ASR at submission time should be able to run:

```bash
# 1. Clean clone, environment up
git clone https://github.com/haoruilee/stable-asr && cd stable-asr
python -m pip install -e ".[all]"
stable-asr doctor --check-release-env

# 2. Smoke (≤ 30 min on 4080-class GPU, no large data needed)
bash scripts/run_quickstart.sh setup smoke data
stable-asr paper-release-smoke --output-dir runs/paper/release_smoke

# 3. Full-scale reproduction (≤ 4 h on 4080 + AMI/LibriSpeech subsets)
bash scripts/run_final_machine.sh full
bash scripts/run_ablations.sh
bash scripts/run_eval.sh

# 4. Acceleration benchmark across all backends
python scripts/bench_acceleration.py --all-backends \
    --output runs/paper/acceleration

# 5. Deployment validation
stable-asr deploy export --checkpoint runs/.../checkpoint.pt --all-runtimes
stable-asr deploy serve --runtime onnx_cuda --port 8000
stable-asr deploy bench --runtime triton --config deploy/serve/triton/config.pbtxt
```

and produce, in `runs/paper/`:

- A trained Streaming Conformer-T checkpoint + its 4 quantized variants
- 5+ runtime-backend latency tables
- Cross-lingual transfer results (en → zh)
- Pareto figure (latency vs accuracy across backends)
- Leaderboard JSON
- Paper-ready Markdown tables and figures
- Audit report passing `paper-release-audit`

---

## 11. Risks and Open Questions

| Risk | Mitigation |
|---|---|
| 4080 16 GB VRAM insufficient for Conformer-T training | use gradient accumulation; quote results on subsets; reserve full-scale for new machine |
| LDC license blocks public release of Switchboard-derived predictions | release only the *adapter code* + scripts; require user to bring their own LDC tarball |
| External adapter (Whisper/Paraformer/FunASR) version drift breaks reproducibility | pin exact upstream commit + checksum in `configs/adapters/stable_asr_adapters.json` |
| Triton / TensorRT setup non-trivial on CI | gate runtime tests behind `pytest -m gpu` and `pytest -m triton`; CI runs only CPU/ONNX path |
| Paper module surface (30 files in `paper/`) reads as overengineered | P0.5 consolidation step before submission |

---

## 12. Cross-Reference Index

- Existing turn-taking roadmap: [`ROADMAP.md`](../ROADMAP.md)
- Stable-worldmodel parity checklist: [`configs/platform/stable_worldmodel_parity.json`](../configs/platform/stable_worldmodel_parity.json)
- Current data sources: [`configs/datasets/stable_asr_sources.json`](../configs/datasets/stable_asr_sources.json)
- Current model registry: [`configs/models/stable_asr_models.json`](../configs/models/stable_asr_models.json)
- Acceleration baseline numbers: [`runs/bench_accel/acceleration_results.json`](../runs/bench_accel/acceleration_results.json)
- Memory snapshot: `~/.claude/projects/-home-li-stable-asr/memory/project_status.md`

---

*Edits to this roadmap should land alongside an update to
`configs/platform/stable_worldmodel_parity.json` whenever a milestone changes
the required-paths or required-commands surface.*
