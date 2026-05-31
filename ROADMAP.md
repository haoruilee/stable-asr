# Stable-ASR Roadmap

Stable-ASR is a reproducible research platform for real-time ASR systems.

It starts with turn-taking and endpointing: the control layer between streaming
ASR and full-duplex voice agents. NanoTurn is the first built-in model family.

Final research target:

```text
Stable-ASR: A Platform for Reproducible Real-Time ASR and
Full-Duplex Turn-Taking Research and Evaluation
```

The goal is to produce a stable-worldmodel-style platform paper, not only a
software package. The paper contribution should be a unified platform that
standardizes data, baselines, scenarios, policies, and evaluation protocols for
real-time ASR and voice-agent interaction research.

Reference target: https://arxiv.org/abs/2605.21800

## Positioning

Stable-ASR is not another all-in-one ASR toolkit. ESPnet, FunASR, WeNet, NeMo,
and related projects already cover general ASR training and deployment well.

Stable-ASR focuses on the missing system layer:

- standardized ASR and turn-taking manifests
- turn-taking, endpointing, interruption, and backchannel baselines
- streaming ASR evaluation beyond WER/CER
- scenario-based robustness tests for real-time voice products
- policy and threshold optimization for voice-agent actions
- deployment reports for latency, RTF, memory, ONNX, and CPU inference

The first release is deliberately narrow:

```text
Stable-ASR Turn Suite v0.1
  NanoTurn baseline
  turn/action data format
  endpointing policy
  interaction-level metrics
  ONNX export
```

The final platform paper should make the same type of claim that
stable-worldmodel makes for world-model research, but in the real-time ASR and
full-duplex voice-agent domain:

```text
Real-time ASR system research is fragmented by one-off data formats,
inconsistent streaming protocols, hard-to-compare endpointing policies, and
non-standard full-duplex interaction benchmarks.

Stable-ASR provides one reproducible framework for collecting, converting,
training, adapting, evaluating, and stress-testing these systems.
```

## Stable-WorldModel Analogy

The target paper should mirror the platform structure of stable-worldmodel while
remaining specific to speech systems.

| stable-worldmodel | Stable-ASR target |
| --- | --- |
| World models | Real-time ASR systems and turn-action models |
| `World(...)` environments | `VoiceWorld(...)` scenarios |
| Lance-based video/robot data layer | Lance/Parquet/JSONL/HF audio and stream-trace data layer |
| MP4, HDF5, LeRobot conversion | WAV, FLAC, WebDataset, HF, EasyTurn, Full-Duplex-Bench conversion |
| World model baselines | NanoTurn, VAD endpointing, SmartTurn/EasyTurn/VAP adapters, ASR adapters |
| Planning solvers | Turn policy solvers, threshold search, hysteresis, calibration, barge-in policy |
| Visual/geometric/physical factors | Noise, reverb, accent, speaking rate, overlap, ASR error, TTS voice, network jitter |
| MPC evaluation | Real-time voice-agent interaction evaluation |
| OOD generalization | New speakers, languages, accents, noise, overlap patterns, domains |

The paper should claim three concrete contributions:

1. A high-performance speech data layer for ASR windows, turn episodes, and
   streaming traces, with conversion tools across common formats.
2. Clean, tested baselines and policy solvers for turn-taking, endpointing,
   interruption, and streaming ASR evaluation.
3. A suite of controllable full-duplex voice scenarios for reproducible
   robustness, latency, and out-of-distribution evaluation.

## Stable-WorldModel Paper Parity Plan

The end target is not to imitate the world-model task, but to match the paper
shape: a platform paper whose claims are backed by code, reproducible scripts,
benchmarks, tables, and figures.

| stable-worldmodel paper element | Stable-ASR paper requirement |
| --- | --- |
| Fragmentation problem statement | show that ASR, VAD, endpointing, turn-taking, streaming metrics, and voice-agent control are evaluated in separate pipelines |
| Data bottleneck evidence | benchmark JSONL/audio-folder, Parquet, Lance, and remote/object-storage reads for random speech windows and stream traces |
| Unified abstraction | define `TurnManifestRecord`, `VoiceWorld`, `TurnPolicy`, ASR adapters, and paper result artifacts |
| Baseline zoo | run rule/VAD baselines, NanoTurn, external turn adapters, and streaming ASR adapters through the same interface and registry |
| Solver layer | show cost-sensitive threshold search, hysteresis, calibration, and barge-in policies improve interaction metrics |
| Controllable FoV suite | evaluate noise, reverb, speaking rate, overlap offset, language, accent, ASR error, assistant state, and network jitter |
| Case study | demonstrate that a model with acceptable window metrics can fail under interruption, backchannel, or side-speech scenarios |
| Appendix-level usability | document install, CLI, schemas, scenarios, adapters, configs, extension points, one-command reproduction, a paper-parity audit, a final-experiment runbook, and final-run config templates |

The first arXiv-ready version should contain one strong central result:

```text
WER/CER and static turn accuracy are insufficient for real-time ASR systems.
Stable-ASR exposes interaction failures through standardized streaming metrics,
turn-action policies, and controllable VoiceWorld scenarios.
```

This keeps the paper differentiated from ASR toolkits while making the same kind
of platform contribution that stable-worldmodel makes for world-model research.

## Paper-First Research Questions

The roadmap should be judged by whether it can answer these research questions
with tables, figures, and reproducible scripts.

### RQ1: Data Layer

Can Stable-ASR load and sample real-time ASR training windows faster and more
reproducibly than ad hoc JSONL/audio-folder pipelines?

Required experiments:

- JSONL + audio folder vs Parquet vs Lance
- ASR metadata TSV/CSV/JSONL -> canonical ASR manifest recipe
- local NVMe vs object storage
- random 2-second window sampling throughput
- episode-contiguous turn trace loading
- multi-worker dataloader utilization
- storage size and conversion time

Paper artifacts:

- throughput table
- storage table
- ASR corpus manifest recipe table, smoke implementation via `paper-table asr_manifest_recipe`
- data format registry diagram, smoke implementation via `paper-figure data_registry`
- paper parity checklist artifact, implemented with `stable-asr paper-parity-audit`
- final-scale experiment runbook, implemented with `stable-asr final-experiments`
- final-run config template, implemented with `stable-asr final-config`
- reproducible benchmark script

### RQ2: Baseline Reproducibility

Can Stable-ASR make common endpointing and turn-taking baselines directly
comparable under one interface?

Required baselines:

- `RuleEndpointBaseline`
- `VADPauseBaseline`
- `NanoTurnPico`
- `NanoTurnNano`
- `SmartTurnAdapter`
- `EasyTurnAdapter`
- `VAPBaseline`
- `TextTurnBaseline`

Required experiments:

- classification metrics
- interaction metrics
- CPU latency
- ONNX latency
- model size
- failure case taxonomy

Paper artifacts:

- baseline comparison table
- adapter registry artifact, implemented with `stable-asr adapter-registry`
- latency/quality Pareto plot, smoke implementation via `paper-figure latency_quality_pareto`
- confusion matrices
- representative failure examples, smoke implementation via `paper-table failure_cases`

### RQ3: Policy Matters

Do explicit turn policies and policy solvers improve system-level behavior beyond
raw model probabilities or fixed pause thresholds?

Required policies:

- fixed pause threshold
- probability threshold
- hysteresis policy
- barge-in policy
- calibration-aware policy
- cost-sensitive threshold solver

Required experiments:

- false complete vs latency trade-off
- missed interruption vs false interruption trade-off
- backchannel precision under assistant-speaking state
- policy transfer across scenarios

Paper artifacts:

- cost matrix
- policy search curves
- threshold sensitivity plots
- scenario-level interaction table

### RQ4: Controllable Scenario Evaluation

Can Stable-ASR reveal robustness failures that static WER/CER or window-level
accuracy cannot show?

Required scenarios:

- incomplete pause
- backchannel
- wait / stop
- user interruption
- side conversation
- ambient speech
- noisy far-field
- code-switching
- versioned suite definition, implemented as `configs/scenarios/stable_asr_voiceworld_v0.json`

Required factors of variation:

- SNR
- reverb
- speaking rate
- accent tag
- overlap offset
- assistant TTS voice
- assistant speaking state
- ASR word error rate
- network jitter
- language

Paper artifacts:

- scenario suite table
- scenario suite registry artifact, implemented with `stable-asr scenario-suite`
- factor-of-variation table
- zero-shot robustness heatmap
- interaction failure breakdown

### RQ5: Real-Time ASR Evaluation Beyond WER

Do models with similar WER/CER produce different real-time user experiences?

Required metrics:

- WER
- CER
- RTF
- first partial latency
- finalization latency
- endpoint delay
- partial revision rate
- stable prefix length
- timestamp drift
- turn-action error rate

Required experiments:

- at least two ASR adapters under the same streaming evaluator
- chunk-size and lookahead sweeps
- partial transcript stability analysis
- endpoint delay vs recognition accuracy

Paper artifacts:

- streaming metrics table
- streaming failure taxonomy table, smoke implementation via `paper-table streaming_failures`
- partial revision plot
- timestamp drift plot
- WER-vs-interaction-quality scatter plot

## Target Paper Outline

The final paper should be planned from the beginning.

```text
1. Introduction
   - real-time ASR systems are evaluated inconsistently
   - WER/CER alone misses endpointing, streaming, and turn-action failures
   - Stable-ASR unifies data, baselines, scenarios, policies, and evaluation

2. Related Work
   - ASR toolkits
   - streaming ASR evaluation
   - VAD and endpointing
   - turn-taking and full-duplex benchmarks
   - speech data preparation frameworks

3. Platform Overview
   - architecture
   - data layer
   - model and adapter interfaces
   - policy layer
   - evaluation/reporting layer

4. Data Layer
   - manifest schema
   - TurnEpisode and TurnWindow
   - streaming traces
   - conversion tools
   - Lance/Parquet/JSONL/HF benchmarks

5. Baselines and Solvers
   - rule and VAD baselines
   - NanoTurn models
   - external model adapters
   - policy solvers

6. VoiceWorld Scenario Suite
   - scenario definitions
   - controllable factors of variation
   - seed reproducibility
   - synthetic and real audio composition

7. Experiments
   - data throughput
   - baseline comparison
   - policy optimization
   - scenario robustness
   - streaming ASR evaluation

8. Limitations
   - dataset licensing
   - synthetic scenario realism
   - language coverage
   - full-duplex user-study gap

9. Conclusion
```

## Target Figures and Tables

Required figures:

- platform architecture diagram implemented through `paper-figure architecture`
- three-stage API flow implemented through `paper-figure api_flow`
- data format registry diagram implemented through `paper-figure data_registry`
- VoiceWorld scenario timeline diagram implemented through `paper-figure voiceworld_timeline`
- policy decision state machine implemented through `paper-figure policy_state_machine`
- scenario robustness heatmap implemented through `paper-figure robustness_heatmap`
- latency/quality Pareto plot implemented through `paper-figure latency_quality_pareto`
- smoke-run SVG figures for baseline quality, latency, data size, streaming
  metrics, scenario accuracy, and policy search objective implemented through
  `stable-asr paper-figure`

Required tables:

- data format comparison
- ASR corpus manifest recipe summary
- data throughput benchmark
- baseline zoo summary
- failure-case taxonomy summary
- scenario suite summary
- factor-of-variation summary
- turn-taking metrics comparison
- streaming ASR metrics comparison
- ablation table for policy solvers

## Paper Release Gates

The project should not claim paper readiness until these gates pass.

### Gate 1: Software Reproducibility

- package install works from source
- CI test workflow added
- docs build
- examples run end to end
- configs pin random seeds
- benchmark scripts generate the paper tables

### Gate 2: Data Reproducibility

- public demo dataset available
- conversion scripts for at least three external data sources
- metadata-table ASR manifest recipe for public corpora such as LibriSpeech,
  AISHELL-1, WenetSpeech, and Common Voice
- data cards for every bundled dataset or manifest
- license notes for all sources
- deterministic train/valid/test splits

### Gate 3: Baseline Reproducibility

- all baselines share one interface
- baseline configs are versioned
- checkpoints or training scripts are available
- latency scripts run on CPU and GPU
- ONNX export works for NanoTurn

### Gate 4: Scenario Reproducibility

- scenarios are seedable
- factors of variation are inspectable from CLI
- scenario configs are versioned, implemented with `configs/scenarios/stable_asr_voiceworld_v0.json`
- generated manifests can be regenerated
- reports include scenario-level breakdowns

### Gate 5: Paper Reproducibility

- one command or script family regenerates all paper tables
- one command or script family regenerates all paper figures, implemented for
  smoke artifacts with `stable-asr paper-figure`
- one command bundles all smoke paper artifacts, implemented with
  `stable-asr paper-bundle`
- one command audits required result sections and bundled artifacts, implemented
  with `stable-asr paper-audit`
- one command audits release-readiness gates and remaining paper gaps, implemented
  with `stable-asr paper-release-audit`
- one command maps platform-paper claims to concrete files, result keys,
  commands, and artifacts, implemented with `stable-asr paper-claim-audit`
- one command generates an arXiv-style LaTeX draft from the same result bundle,
  implemented with `stable-asr paper-latex`
- experiment logs are archived
- case-study artifacts link failure categories to manifest/transcript records
- claim-evidence artifacts map each platform contribution to concrete proof
- exact code commit is tagged
- README, docs, and citation are ready for the smoke-stage repository

## Product Thesis

Traditional ASR evaluation answers:

```text
audio -> text
WER / CER
```

Real-time voice systems also need to answer:

```text
audio stream -> text + timestamp + endpoint + turn state + action
```

Stable-ASR should make these questions reproducible:

- Did the user finish speaking?
- Is the user pausing mid-utterance?
- Is the user only backchanneling?
- Is the user interrupting the assistant?
- Should the assistant speak, keep listening, stop TTS, hold, or ignore?
- How do these decisions change under noise, reverb, accents, overlap, and ASR errors?

## Non-Goals

Stable-ASR should avoid becoming a broad ASR clone.

- Do not compete directly with ESPnet, FunASR, WeNet, or NeMo as a full ASR stack.
- Do not start by training large general ASR models.
- Do not replace Pipecat or other voice-agent orchestration frameworks.
- Do not build a full ASR/TTS/LLM product stack in v0.x.
- Do not optimize only for WER while ignoring latency and interaction quality.

## Package Structure

Target repository layout:

```text
stable-asr/
  stable_asr/
    data/
      asr_manifest.py
      manifest.py
      dataset.py
      audio.py
      registry.py
      recipes/
        asr_folder.py
      converters/
        easyturn.py
        full_duplex_bench.py
        hf.py
        webdataset.py
      formats/
        jsonl.py
        parquet.py
        lance.py

    turn/
      labels.py
      nanoturn.py
      policy.py
      endpointing.py
      interruption.py

    models/
      adapters/
        whisper.py
        funasr.py
        wenet.py
        nemo.py
        espnet.py
      baselines/
        vad_pause.py
        rule_endpoint.py

    streaming/
      chunker.py
      partials.py
      latency.py

    eval/
      wer.py
      cer.py
      rtf.py
      turn_metrics.py
      timestamp_metrics.py
      report.py
      paper_tables.py
      paper_figures.py

    scenarios/
      voice_world.py
      incomplete_pause.py
      backchannel.py
      interruption.py
      wait_stop.py
      side_speech.py
      ambient_speech.py

    train/
      trainer.py
      losses.py
      checkpoint.py

    paper/
      experiments.py
      tables.py
      figures.py
      latex.py

  configs/
    nanoturn_nano.yaml
    scenarios/
      zh_turn_mini_v0.yaml
    paper/
      data_layer_benchmark.yaml
      baseline_comparison.yaml
      policy_ablation.yaml
      scenario_robustness.yaml
      streaming_asr_eval.yaml

  scripts/
    train_turn.py
    eval_turn.py
    export_turn_onnx.py
    convert_easyturn.py
    convert_full_duplex_bench.py
    make_synthetic_turn_data.py
    reproduce_paper.py

  examples/
    01_train_nanoturn.py
    02_eval_endpointing.py
    03_streaming_asr_with_turn.py
```

## Core Data Model

The first stable format is a JSONL manifest. Each line is one training or
evaluation window.

```json
{
  "id": "zh_turn_000001",
  "audio": "audio/000001.flac",
  "sample_rate": 16000,
  "start": 0.0,
  "end": 2.0,
  "text": "我想问一下今天北京的天气",
  "asr_text": "我想问一下今天北京的天气",
  "turn_label": "complete",
  "action_label": "take_turn",
  "assistant_speaking": false,
  "overlap": false,
  "scenario": "normal_question",
  "language": "zh",
  "source": "synthetic_v0"
}
```

The v0.1 turn labels are:

```text
complete
incomplete
backchannel
wait
```

The v0.1 action labels are:

```text
take_turn
keep_listening
continue_speaking
stop_tts_and_listen
hold
ignore
light_ack
```

Future ASR evaluation fields:

```json
{
  "reference": "我想问一下今天北京的天气",
  "hypothesis": "我想问一下今天北京天气",
  "word_timestamps": [
    {"word": "我", "start": 0.10, "end": 0.18}
  ],
  "partial_hypotheses": [
    {"time": 0.5, "text": "我想"},
    {"time": 1.0, "text": "我想问一下"}
  ]
}
```

## Model Interfaces

Turn models should expose a small interface:

```python
class TurnModel:
    def predict(self, audio_window, context=None):
        ...
```

Prediction output:

```python
@dataclass
class TurnPrediction:
    probs: dict[str, float]
    timestamp: float
    embedding: np.ndarray | None = None
```

Policy output:

```python
@dataclass
class TurnAction:
    action: Literal[
        "take_turn",
        "keep_listening",
        "continue_speaking",
        "stop_tts_and_listen",
        "ignore",
        "hold",
        "light_ack",
    ]
    confidence: float
    reason: str | None = None
```

ASR adapters should expose:

```python
class ASRModel:
    def transcribe(self, audio) -> ASRResult:
        ...

    def stream(self, audio_chunks) -> Iterator[PartialASRResult]:
        ...
```

Current implementation status:

- `ASRModel`, `ASRResult`, and `PartialASRResult` protocol/data containers implemented
- `StreamingASRAdapter` evaluation protocol implemented
- `TranscriptJSONLAdapter` implemented as the first adapter-compatible streaming fixture backend
- `compare_streaming_adapters(...)` runs adapter objects through one evaluator

## 90-Day Execution Plan

### Weeks 1-2: Foundation

Build the smallest installable project.

- package skeleton
- manifest schema
- JSONL validation
- example turn dataset
- CLI entrypoint
- unit tests for schema and metrics

Target outcome:

```text
stable-asr validate-manifest examples/data/turn_demo.jsonl
```

### Weeks 3-4: Metrics and Baselines

Make evaluation useful before model training becomes complicated.

- rule endpoint baseline
- VAD pause baseline
- turn classification metrics
- endpointing interaction metrics
- Markdown report writer
- small fixture-based regression tests

Target outcome:

```text
stable-asr eval-turn --baseline vad_pause --dataset examples/data/turn_demo.jsonl
```

### Weeks 5-7: NanoTurn v0

Train the first built-in model.

- log-mel frontend
- NanoTurnPico
- NanoTurnNano
- training loop
- checkpoint save/load
- evaluation loop
- config file

Target outcome:

```text
stable-asr train-turn --config configs/nanoturn_nano.yaml
stable-asr eval-turn --model runs/nanoturn/best.pt --dataset examples/data/turn_demo.jsonl
```

### Weeks 8-9: Policy and Export

Turn probabilities into deployable decisions.

- threshold policy
- hysteresis policy
- cost-weighted metric summary
- ONNX export
- CPU latency benchmark

Target outcome:

```text
stable-asr export-turn-onnx --checkpoint runs/nanoturn/best.pt
stable-asr benchmark-turn --dataset data/eval.jsonl --checkpoint runs/nanoturn/best.pt --artifact runs/nanoturn/model.onnx
```

### Weeks 10-12: Scenario Mini-Suite

Move from static classification to interaction evaluation.

- incomplete pause scenario
- backchannel scenario
- user interruption scenario
- side conversation scenario
- ambient speech scenario
- noisy far-field scenario
- Chinese-English code-switching scenario
- simple synthetic audio composition
- scenario-level report sections

Target outcome:

```text
stable-asr eval-scenario --policy runs/policy.json --suite zh_turn_mini_v0
```

### Day-90 Release Target

Release `Stable-ASR Turn Suite v0.1` with:

- installable package
- working CLI
- manifest schema
- NanoTurn baseline
- VAD and rule baselines
- core turn metrics
- ONNX export
- CPU latency report
- nine interaction scenarios
- reproducible examples

This is an engineering release, not a paper release. It creates the first
vertical slice needed for the future platform paper.

## Milestones

### M0: Project Skeleton

Goal: make the repo installable and establish stable public APIs.

Current status:

```text
implemented
```

Deliverables:

- `pyproject.toml`
- `stable_asr` package
- basic CLI entrypoint: `stable-asr`
- JSONL manifest loader
- typed dataclasses for turn windows, predictions, and actions
- minimal test suite
- contributor-facing docs
- fixture turn manifest
- canonical turn/action labels
- threshold/hysteresis `TurnPolicy`
- Markdown report helper

Exit criteria:

- `pip install -e .` works
- `pytest` works
- sample JSONL manifest can be loaded and validated

Verified locally:

```text
python3 -m pytest
97 passed

python3 -m stable_asr validate-manifest examples/data/turn_demo.jsonl
OK: examples/data/turn_demo.jsonl contains 4 valid record(s).

/tmp/stable-asr-venv/bin/stable-asr validate-manifest examples/data/turn_demo.jsonl
OK: examples/data/turn_demo.jsonl contains 4 valid record(s).
```

### v0.1: Stable-ASR Turn Suite

Goal: deliver the first useful slice: turn-taking and endpointing.

Current status:

```text
in progress
```

Deliverables:

- `NanoTurnPico`
- `NanoTurnPico` implemented with optional PyTorch dependency
- `NanoTurnNano` implemented with optional PyTorch dependency
- `VADPauseBaseline` implemented
- `RuleEndpointBaseline` implemented
- `TextTurnBaseline` implemented
- external turn prediction JSONL adapter implemented through `eval-turn --predictions`
- external prediction conversion implemented for generic, SmartTurn-style, and EasyTurn-style JSONL
- turn dataset loader with windowing
- four-class turn training implemented for metadata-feature v0
- audio-feature training implemented for synthetic WAV manifests
- endpointing policy with thresholds and hysteresis
- ONNX export for NanoTurn implemented
- Markdown report generation implemented for baseline evaluation
- `eval-turn` CLI implemented for pause baselines
- `train-turn` CLI implemented for NanoTurn checkpoints
- `eval-turn --checkpoint` implemented for NanoTurn checkpoints
- `benchmark-turn` CLI implemented for baseline, prediction manifest, and NanoTurn checkpoint latency benchmarks

Metrics:

- macro F1
- complete precision
- incomplete recall
- false complete rate
- premature response rate
- backchannel precision
- wait recall
- decision latency
- CPU latency
- model size

Exit criteria:

- train NanoTurn on a small local manifest
- evaluate on held-out windows partially implemented for baselines
- export ONNX
- produce a reproducible report implemented for baselines

Verified locally:

```text
python3 -m stable_asr eval-turn --dataset examples/data/turn_demo.jsonl --baseline vad_pause
accuracy: 0.5000
macro_f1: 0.3333
false_complete_rate: 0.3333
premature_response_rate: 0.3333
missed_interrupt_rate: 0.0000

python3 -m stable_asr eval-turn --dataset examples/data/turn_demo.jsonl --baseline text_turn
accuracy: 1.0000
macro_f1: 1.0000
false_complete_rate: 0.0000
premature_response_rate: 0.0000
missed_interrupt_rate: 0.0000

python3 -m stable_asr eval-turn --dataset examples/data/turn_demo.jsonl --predictions tests/fixtures/turn_predictions_sample.jsonl
accuracy: 1.0000
macro_f1: 1.0000
false_complete_rate: 0.0000
premature_response_rate: 0.0000
missed_interrupt_rate: 0.0000

python3 -m stable_asr convert-predictions --schema easyturn --input tests/fixtures/easyturn_predictions_sample.jsonl --output /tmp/stable-asr-easyturn-preds.jsonl
converted 4 prediction record(s) from tests/fixtures/easyturn_predictions_sample.jsonl to /tmp/stable-asr-easyturn-preds.jsonl

python3 -m stable_asr eval-turn --dataset examples/data/turn_demo.jsonl --predictions /tmp/stable-asr-easyturn-preds.jsonl
accuracy: 1.0000
macro_f1: 1.0000
false_complete_rate: 0.0000
premature_response_rate: 0.0000
missed_interrupt_rate: 0.0000

python3 -m stable_asr benchmark-turn --dataset examples/data/turn_demo.jsonl --baseline text_turn --warmup 0 --repeat 3 --report /tmp/stable-asr-turn-benchmark.md
records: 4
predictions: 12
avg_latency_ms: 0.0120
p50_latency_ms: 0.0026
p95_latency_ms: 0.0061
throughput_predictions_per_sec: 82194.60
rtf: 0.000007
report: /tmp/stable-asr-turn-benchmark.md

python3 -m stable_asr train-turn --dataset examples/data/turn_demo.jsonl --output-dir /tmp/stable-asr-nanoturn --model nanoturn_pico --epochs 20 --seed 0
checkpoint: /tmp/stable-asr-nanoturn/checkpoint.pt
metrics: /tmp/stable-asr-nanoturn/metrics.json
final_loss: 1.131062
final_accuracy: 0.7500

python3 -m stable_asr eval-turn --dataset examples/data/turn_demo.jsonl --checkpoint /tmp/stable-asr-nanoturn/checkpoint.pt
accuracy: 0.7500
macro_f1: 0.6667
false_complete_rate: 0.0000
premature_response_rate: 0.0000
missed_interrupt_rate: 0.0000

python3 -m stable_asr export-turn-onnx --checkpoint /tmp/stable-asr-nanoturn/checkpoint.pt --output /tmp/stable-asr-nanoturn/nanoturn.onnx
onnx: /tmp/stable-asr-nanoturn/nanoturn.onnx

python3 -m stable_asr make-synthetic-turn-data --output /tmp/stable-asr-audio/synthetic.jsonl --episodes 8 --seed 11 --write-audio
wrote 8 record(s) to /tmp/stable-asr-audio/synthetic.jsonl

python3 -m stable_asr train-turn --dataset /tmp/stable-asr-audio/synthetic.jsonl --output-dir /tmp/stable-asr-audio/run --feature-source audio --epochs 5 --seed 0
checkpoint: /tmp/stable-asr-audio/run/checkpoint.pt
metrics: /tmp/stable-asr-audio/run/metrics.json
final_loss: 1.360426
final_accuracy: 0.3750

python3 -m pytest
97 passed
```

### v0.2: Scenario Evaluation

Goal: evaluate turn models as interaction controllers, not just classifiers.

Current status:

```text
in progress
```

Scenarios:

- incomplete pause seedable synthetic manifest implemented
- backchannel seedable synthetic manifest implemented
- wait / stop seedable synthetic manifest implemented
- user interruption seedable synthetic manifest implemented
- side conversation seedable synthetic manifest implemented
- ambient speech seedable synthetic manifest implemented
- noisy far-field seedable synthetic manifest implemented
- Chinese-English code-switching seedable synthetic manifest implemented

Factors:

- SNR implemented in synthetic metadata
- reverb implemented in synthetic metadata
- speaking rate implemented in synthetic metadata
- overlap offset implemented in synthetic metadata
- assistant speaking state
- ASR error rate
- language
- accent tag implemented in synthetic metadata
- far-field distance implemented in synthetic metadata
- code-switch ratio implemented in synthetic metadata
- network jitter implemented in synthetic metadata

Metrics:

- missed interrupt rate
- stop latency
- unnecessary stop rate
- side speech rejection
- ambient speech rejection
- flow continuity score

Exit criteria:

- scenario configs are reproducible by seed partially implemented
- deterministic synthetic WAV audio generation implemented
- per-scenario VoiceWorld evaluation implemented
- scenario robustness paper table implemented
- policies can be evaluated across multiple scenarios
- reports include failure examples and scenario breakdowns

Verified locally:

```text
python3 -m stable_asr make-synthetic-turn-data --output /tmp/stable-asr-synth/synthetic.jsonl --episodes 18 --seed 42
wrote 18 record(s) to /tmp/stable-asr-synth/synthetic.jsonl

python3 -m stable_asr validate-manifest /tmp/stable-asr-synth/synthetic.jsonl
OK: /tmp/stable-asr-synth/synthetic.jsonl contains 18 valid record(s).

python3 -m stable_asr eval-scenario --episodes 18 --seed 3 --baseline vad_pause --report /tmp/stable-asr-scenario/scenario.md
suite: zh_turn_mini_v0
records: 18
accuracy: 0.4444
macro_f1: 0.2708
scenarios include noisy_farfield and code_switching
```

### v0.3: Paper-Grade Data Layer

Goal: add the data-system contribution needed for a platform paper.

Current status:

```text
started
```

Deliverables:

- data format registry implemented
- JSONL manifest backend implemented
- JSONL conversion CLI implemented
- manifest inspection CLI implemented
- JSONL/audio-folder backend
- Parquet backend implemented as optional `pyarrow` data extra
- Lance backend implemented as optional `pylance` data extra
- machine-readable data source registry implemented in `configs/datasets/stable_asr_sources.json`
- `data-sources` CLI implemented for source registry validation/rendering
- HF datasets adapter
- WebDataset adapter
- stream-trace schema
- conversion scripts for EasyTurn, Full-Duplex-Bench, and SmartTurn-style manifests
- EasyTurn-style JSONL converter implemented
- Full-Duplex-Bench-style JSONL converter implemented
- SmartTurn-style JSONL converter implemented
- prediction converters for generic, SmartTurn-style, and EasyTurn-style JSONL implemented
- data benchmark scripts with random sampling rows

Experiments:

- random window sampling throughput implemented for manifest records
- episode trace loading throughput
- conversion speed
- local storage size
- remote/object-storage readiness if available

Exit criteria:

- paper-ready data throughput table can be regenerated
- format conversion works from CLI for JSONL
- format conversion works from CLI for Parquet when `pyarrow` is installed
- format conversion works from CLI for Lance when `pylance` is installed
- `benchmark-data` outputs JSONL/Parquet/Lance write/read/size/random-sampling rows
- the default training loader can use Lance or Parquet without code changes

Verified locally:

```text
python3 -m stable_asr convert examples/data/turn_demo.jsonl /tmp/stable-asr-convert/turn_demo.jsonl
converted 4 record(s) from examples/data/turn_demo.jsonl to /tmp/stable-asr-convert/turn_demo.jsonl

python3 -m stable_asr inspect-manifest /tmp/stable-asr-convert/turn_demo.jsonl
records: 4

python3 -m stable_asr convert examples/data/turn_demo.jsonl /tmp/stable-asr-data/turn_demo.parquet
converted 4 record(s) from examples/data/turn_demo.jsonl to /tmp/stable-asr-data/turn_demo.parquet

python3 -m stable_asr convert examples/data/turn_demo.jsonl /tmp/stable-asr-data/turn_demo.lance
converted 4 record(s) from examples/data/turn_demo.jsonl to /tmp/stable-asr-data/turn_demo.lance

python3 -m stable_asr benchmark-data --dataset examples/data/turn_demo.jsonl --output-dir /tmp/stable-asr-data/bench --formats jsonl parquet lance --sample-count 16 --json
[
  {"format": "jsonl", "sample_count": 16, "...": "..."},
  {"format": "parquet", "sample_count": 16, "...": "..."},
  {"format": "lance", "sample_strategy": "lance_take", "...": "..."}
]

python3 -m stable_asr data-sources --registry configs/datasets/stable_asr_sources.json --validate-only
OK: stable_asr_sources_v0 (10 source(s))

python3 -m stable_asr convert-external --schema easyturn --input tests/fixtures/easyturn_sample.jsonl --output /tmp/stable-asr-external/easyturn.jsonl
converted 3 external record(s) from tests/fixtures/easyturn_sample.jsonl to /tmp/stable-asr-external/easyturn.jsonl

python3 -m stable_asr convert-external --schema full_duplex_bench --input tests/fixtures/full_duplex_bench_sample.jsonl --output /tmp/stable-asr-external/fdb.jsonl
converted 3 external record(s) from tests/fixtures/full_duplex_bench_sample.jsonl to /tmp/stable-asr-external/fdb.jsonl

python3 -m stable_asr convert-external --schema smart_turn --input tests/fixtures/smart_turn_manifest_sample.jsonl --output /tmp/stable-asr-external/smart_turn.jsonl
converted 3 external record(s) from tests/fixtures/smart_turn_manifest_sample.jsonl to /tmp/stable-asr-external/smart_turn.jsonl
```

### v0.4: Streaming ASR Evaluator

Goal: compare existing streaming ASR systems under product-relevant metrics.

Current status:

```text
started
```

Adapters:

- transcript JSONL fixture adapter implemented
- multi-adapter transcript comparison implemented through `compare-streaming-asr`
- command-backed external ASR adapter implemented through `eval-asr-command`
- config-driven multi-command ASR comparison implemented through `compare-asr-commands`
- Whisper-style transcript converter implemented through `convert-asr-transcript --schema whisper`
- FunASR-style transcript converter implemented through `convert-asr-transcript --schema funasr`
- WeNet
- NeMo
- ESPnet

Metrics:

- WER implemented
- CER implemented
- RTF implemented
- first partial latency implemented
- finalization latency implemented
- endpoint delay implemented
- partial revision rate implemented
- stable prefix ratio implemented
- timestamp drift implemented for word timestamps
- chunk-size and lookahead sensitivity implemented through `sweep-streaming-asr`

Exit criteria:

- at least two ASR adapters run through the same evaluator implemented for transcript fixtures
- streaming and offline metrics are reported separately
- partial hypothesis behavior is captured in reports
- chunk-size and lookahead sweep rows are included in paper artifacts
- Whisper/FunASR transcript conversion rows are included in paper smoke results and release gates

Verified locally:

```text
python3 -m stable_asr eval-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl
records: 2
wer: 0.1250
cer: 0.1081
rtf: 0.2500
first_partial_latency: 0.4500
final_latency: 2.2750
endpoint_delay: 0.3750
partial_revision_rate: 0.2500
stable_prefix_ratio: 0.2481
timestamp_drift: 0.1079

python3 -m stable_asr compare-streaming-asr --input balanced=tests/fixtures/streaming_asr_sample.jsonl --input fast_unstable=tests/fixtures/streaming_asr_fast_unstable_sample.jsonl
adapter=balanced records=2 wer=0.1250 rtf=0.2500 endpoint_delay=0.3750 timestamp_drift=0.1079
adapter=fast_unstable records=2 wer=0.3750 rtf=0.1477 endpoint_delay=0.1000 timestamp_drift=0.2867

python3 -m stable_asr compare-asr-commands --config examples/configs/asr_command_compare_demo.json --report /tmp/stable-asr-command-compare.md
adapter=balanced_command records=2 wer=0.1250 rtf=0.2500 endpoint_delay=0.3750 timestamp_drift=0.1079
adapter=fast_unstable_command records=2 wer=0.3750 rtf=0.1477 endpoint_delay=0.1000 timestamp_drift=0.2867
report: /tmp/stable-asr-command-compare.md

python3 -m stable_asr sweep-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl --chunks-ms 160 320 --lookahead-ms 0 160
chunk_ms=160 lookahead_ms=0 wer=0.1250 first_partial_latency=0.5600 endpoint_delay=0.5000
chunk_ms=160 lookahead_ms=160 wer=0.1250 first_partial_latency=0.7200 endpoint_delay=0.6600

python3 -m stable_asr convert-asr-transcript --schema whisper --input tests/fixtures/whisper_transcript_sample.jsonl --output /tmp/stable-asr-whisper-streaming.jsonl
converted 2 ASR transcript record(s) from tests/fixtures/whisper_transcript_sample.jsonl to /tmp/stable-asr-whisper-streaming.jsonl

python3 -m stable_asr convert-asr-transcript --schema funasr --input tests/fixtures/funasr_transcript_sample.jsonl --output /tmp/stable-asr-funasr-streaming.jsonl
converted 2 ASR transcript record(s) from tests/fixtures/funasr_transcript_sample.jsonl to /tmp/stable-asr-funasr-streaming.jsonl
```

### v0.5: Minimal ASR Recipe

Goal: provide a small, hackable ASR baseline without competing with full ASR frameworks.

Models:

- MiniConformer-CTC
- MiniConformer-CTC+AED

Datasets:

- LibriSpeech small recipe
- AISHELL-1 recipe
- optional WenetSpeech small split recipe

Deliverables:

- BPE tokenizer pipeline
- log-mel frontend
- SpecAugment
- CTC training
- optional AED decoder
- WER/CER evaluation

Exit criteria:

- users can train a small ASR model end to end
- recipes are documented as educational baselines
- the module does not become the primary project identity

### v0.6: Robustness Benchmark

Goal: evaluate real-time ASR behavior under controlled distribution shifts.

Scenarios:

- noisy near-field
- noisy far-field
- room reverb
- overlapped speech
- Chinese-English code-switching
- regional accents
- long-form audio
- meeting-like turn shifts
- voice assistant interruption

Exit criteria:

- benchmark manifests are versioned
- metrics cover accuracy, latency, timestamp quality, and turn actions
- reports compare ASR adapter + endpointing + turn policy combinations

### v0.7: Paper Experiment Suite

Goal: freeze the experiments needed for the first arXiv paper.

Current status:

```text
started
```

Deliverables:

- `stable-asr reproduce-paper` smoke bundle implemented
- `configs/paper/paper_smoke.json` implemented
- `scripts/reproduce_paper.py` implemented
- JSON result artifact implemented for smoke bundle
- Markdown report artifact implemented for smoke bundle
- NanoTurn checkpoint predictor included in paper baseline and latency tables when `reproduce-paper` trains the model
- external turn prediction manifest is included as `prediction_manifest` in smoke baseline comparison
- table generators implemented for baseline, turn benchmark, data benchmark, streaming, streaming sweep, ASR transcript conversion, scenario, and policy Markdown tables
- SVG figure generators implemented for platform architecture, API flow, data registry, VoiceWorld timeline, policy state machine, scenario robustness heatmap, latency/quality Pareto, baseline, latency, data, streaming, scenario, and policy smoke artifacts
- paper artifact bundler implemented for tables, figures, leaderboard exports, benchmark suite files, index, and manifest
- paper artifact audit implemented for result sections, tables, figures, leaderboard exports, benchmark suite files, index, and manifest
- release-readiness audit implemented for software, data, baseline, scenario, policy, streaming, and paper gates
- Markdown paper draft generator implemented for editable preprint drafts
- LaTeX paper draft generator implemented for arXiv-style preprint drafts
- `CITATION.cff` implemented
- initial docs site implemented under `docs/`
- model cards for baselines
- data cards for benchmark manifests partially implemented through `make-card dataset`
- experiment cards implemented through `make-card experiment`
- ablation configs
- failure case browser or report section

Required experiments:

- data-layer benchmark
- baseline comparison
- policy ablation
- scenario robustness
- streaming ASR comparison
- NanoTurn model-size/latency ablation

Exit criteria:

- all paper figures and tables can be regenerated from checked-in configs
- results are stored in a structured format under `runs/paper/`
- every plotted result links back to a command, config, and seed

Verified locally:

```text
python3 -m stable_asr reproduce-paper --output-dir /tmp/stable-asr-paper --episodes 12 --seed 5
results: /tmp/stable-asr-paper/paper_results.json
report: /tmp/stable-asr-paper/reports/paper_smoke.md

The generated Markdown report includes baseline comparison and data benchmark
tables plus an external conversion section.

python3 scripts/reproduce_paper.py --config configs/paper/paper_smoke.json --skip-train
results: runs/paper/smoke/paper_results.json
report: runs/paper/smoke/reports/paper_smoke.md

python3 -m stable_asr paper-table baselines --results /tmp/stable-asr-paper/paper_results.json
| baseline | accuracy | macro_f1 | false_complete_rate | missed_interrupt_rate |

python3 -m stable_asr paper-table turn_benchmark --results /tmp/stable-asr-paper/paper_results.json
| baseline | avg_latency_ms | p50_latency_ms | p95_latency_ms | throughput | rtf | artifact_bytes |

python3 -m stable_asr paper-table data --results /tmp/stable-asr-paper/paper_results.json
| format | records | write_seconds | read_seconds | size_bytes |

python3 -m stable_asr paper-table streaming --results /tmp/stable-asr-paper/paper_results.json
| records | wer | cer | rtf | first_partial_latency | final_latency | partial_revision_rate | stable_prefix_ratio |

python3 -m stable_asr paper-table streaming_sweep --results /tmp/stable-asr-paper/paper_results.json
| chunk_ms | lookahead_ms | first_partial_latency | final_latency | endpoint_delay | partial_revision_rate | timestamp_drift |

python3 -m stable_asr paper-table asr_transcript_conversions --results /tmp/stable-asr-paper/paper_results.json
| schema | records | wer | cer | rtf | endpoint_delay | partial_revision_rate | timestamp_drift |

python3 -m stable_asr paper-table scenarios --results /tmp/stable-asr-paper/paper_results.json
| scenario | records | accuracy | macro_f1 | false_complete_rate | missed_interrupt_rate |

python3 -m stable_asr paper-table policy --results /tmp/stable-asr-paper/paper_results.json
| score | complete_threshold | backchannel_threshold | wait_threshold | interrupt_min_confidence | trials |

python3 -m stable_asr paper-figure architecture --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/architecture.svg
figure: /tmp/stable-asr-paper/figures/architecture.svg

python3 -m stable_asr paper-figure api_flow --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/api_flow.svg
figure: /tmp/stable-asr-paper/figures/api_flow.svg

python3 -m stable_asr paper-figure data_registry --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/data_registry.svg
figure: /tmp/stable-asr-paper/figures/data_registry.svg

python3 -m stable_asr paper-figure voiceworld_timeline --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/voiceworld_timeline.svg
figure: /tmp/stable-asr-paper/figures/voiceworld_timeline.svg

python3 -m stable_asr paper-figure policy_state_machine --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/policy_state_machine.svg
figure: /tmp/stable-asr-paper/figures/policy_state_machine.svg

python3 -m stable_asr paper-figure robustness_heatmap --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/robustness_heatmap.svg
figure: /tmp/stable-asr-paper/figures/robustness_heatmap.svg

python3 -m stable_asr paper-figure latency_quality_pareto --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/latency_quality_pareto.svg
figure: /tmp/stable-asr-paper/figures/latency_quality_pareto.svg

python3 -m stable_asr paper-figure baselines --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/baselines.svg
figure: /tmp/stable-asr-paper/figures/baselines.svg

python3 -m stable_asr paper-figure latency --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/latency.svg
figure: /tmp/stable-asr-paper/figures/latency.svg

python3 -m stable_asr paper-figure data --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/data.svg
figure: /tmp/stable-asr-paper/figures/data.svg

python3 -m stable_asr paper-figure streaming --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/streaming.svg
figure: /tmp/stable-asr-paper/figures/streaming.svg

python3 -m stable_asr paper-figure scenarios --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/scenarios.svg
figure: /tmp/stable-asr-paper/figures/scenarios.svg

python3 -m stable_asr paper-figure policy --results /tmp/stable-asr-paper/paper_results.json --output /tmp/stable-asr-paper/figures/policy.svg
figure: /tmp/stable-asr-paper/figures/policy.svg

python3 -m stable_asr paper-bundle --results /tmp/stable-asr-paper/paper_results.json --output-dir /tmp/stable-asr-paper/artifacts
artifact_index: /tmp/stable-asr-paper/artifacts/ARTIFACT_INDEX.md
manifest: /tmp/stable-asr-paper/artifacts/artifact_manifest.json
tables: 8
figures: 13
leaderboards: 2

python3 -m stable_asr paper-audit --results /tmp/stable-asr-paper/paper_results.json --artifacts-dir /tmp/stable-asr-paper/artifacts
paper_audit: OK

python3 -m stable_asr paper-draft --results /tmp/stable-asr-paper/paper_results.json --artifacts-dir /tmp/stable-asr-paper/artifacts --output /tmp/stable-asr-paper/PAPER_DRAFT.md
draft: /tmp/stable-asr-paper/PAPER_DRAFT.md

python3 -m stable_asr paper-latex --results /tmp/stable-asr-paper/paper_results.json --artifacts-dir /tmp/stable-asr-paper/artifacts --output /tmp/stable-asr-paper/paper.tex
latex: /tmp/stable-asr-paper/paper.tex

python3 -m stable_asr make-card dataset --input examples/data/turn_demo.jsonl --output /tmp/stable-asr-cards/DATASET_CARD.md
card: /tmp/stable-asr-cards/DATASET_CARD.md

python3 -m stable_asr make-card experiment --input /tmp/stable-asr-cards/paper/paper_results.json --output /tmp/stable-asr-cards/EXPERIMENT_CARD.md
card: /tmp/stable-asr-cards/EXPERIMENT_CARD.md

python3 -m stable_asr paper-release-audit --repo-root . --results /tmp/stable-asr-paper/paper_results.json --artifacts-dir /tmp/stable-asr-paper/artifacts --markdown-draft /tmp/stable-asr-paper/PAPER_DRAFT.md --latex-draft /tmp/stable-asr-paper/paper.tex --dataset-card /tmp/stable-asr-cards/DATASET_CARD.md --experiment-card /tmp/stable-asr-cards/EXPERIMENT_CARD.md
paper_release_audit: READY
```

### v0.8: Preprint Candidate

Goal: prepare the first stable-worldmodel-style paper draft.

Current status:

```text
started
```

Deliverables:

- editable Markdown paper draft generated from `paper_results.json` and artifact bundles implemented
- arXiv-style LaTeX paper draft generated from the same results and artifacts implemented
- polished README
- full docs site
- release tag
- install instructions
- citation
- benchmark data release notes
- limitations and ethics section

Exit criteria:

- external user can reproduce the quick-start example
- at least one external baseline adapter is validated
- paper claims are backed by scripts and logs
- code, docs, and draft use consistent naming

### v1.0: Stable-ASR Platform Paper Release

Goal: provide a coherent public benchmark for real-time ASR systems.

Scope:

- ASR accuracy
- streaming latency
- timestamp stability
- endpointing quality
- turn-taking quality
- interruption handling
- deployment efficiency

Exit criteria:

- stable dataset schema
- stable CLI and Python APIs
- reproducible benchmark configs
- baseline zoo
- public leaderboard-ready report format implemented through `leaderboard-export`
- machine-readable benchmark suite definition implemented in `configs/benchmarks/stable_asr_v0.json`
- benchmark suite validation/rendering implemented through `benchmark-suite`
- benchmark suite coverage audit implemented against generated leaderboard rows

Verified leaderboard export:

```text
stable-asr leaderboard-export --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/leaderboard.jsonl --format jsonl
leaderboard: runs/paper/smoke/leaderboard.jsonl

stable-asr benchmark-suite --suite configs/benchmarks/stable_asr_v0.json --validate-only
OK: stable_asr_v0 (7 task(s))

stable-asr benchmark-suite --suite configs/benchmarks/stable_asr_v0.json --results runs/paper/smoke/paper_results.json --validate-only
OK: stable_asr_v0 (7 task(s)); coverage=OK
```

Paper claim:

```text
Stable-ASR is an open-source platform for standardized and reproducible
real-time ASR and full-duplex turn-taking research and evaluation.
It unifies data conversion, training, ASR adapters, turn-action baselines,
policy solvers, scenario simulation, and product-relevant streaming metrics.
```

## Baseline Zoo

Initial baselines:

- `RuleEndpointBaseline`
- `VADPauseBaseline`
- `TextTurnBaseline`
- `NanoTurnPico`
- `NanoTurnNano`

Generic adapters:

- `TurnPredictionManifestAdapter`
- `convert-predictions --schema generic|smart_turn|easyturn`
- `StreamingASRAdapter`
- `TranscriptJSONLAdapter`
- `CommandStreamingASRAdapter`
- `compare-streaming-asr`
- `convert-asr-transcript --schema whisper|funasr`

Planned adapters:

- `SmartTurnAdapter`
- `EasyTurnAdapter`
- `VAPBaseline`
- live `WhisperAdapter`
- live `FunASRAdapter`
- `WeNetAdapter`
- `NeMoAdapter`
- `ESPnetAdapter`

## Policy Layer

Model probabilities are not system actions. Stable-ASR should make policy
decisions explicit and tunable.

Example:

```python
policy = TurnPolicy(
    model=nanoturn,
    thresholds={
        "complete": 0.75,
        "backchannel": 0.70,
        "wait": 0.60,
    },
    hysteresis={
        "complete_windows": 2,
        "interrupt_min_ms": 120,
    },
)
```

Policy solvers:

- threshold search implemented through `optimize-policy`
- hysteresis search
- calibration
- cost-sensitive policy search partially implemented through weighted threshold objective

Cost-sensitive objective examples:

```text
false complete: high cost
missed interruption: high cost
backchannel mistaken as new user request: medium cost
extra 300 ms wait: low cost
```

## CLI Roadmap

v0.1 commands:

```bash
stable-asr validate-manifest data/turn_train.jsonl
stable-asr train-turn --config configs/nanoturn_nano.yaml
stable-asr eval-turn --model runs/nanoturn/best.pt --dataset data/turn_test.jsonl
stable-asr export-turn-onnx --checkpoint runs/nanoturn/best.pt
```

v0.2 commands:

```bash
stable-asr eval-scenario --policy runs/policy.json --scenario incomplete_pause
stable-asr make-synthetic-turn-data --config configs/scenarios/zh_turn_v0.yaml
```

v0.3 commands:

```bash
stable-asr convert data/easyturn --dest data/easyturn.lance --dest-format lance
stable-asr benchmark-data --config configs/paper/data_layer_benchmark.yaml
```

v0.4 commands:

```bash
stable-asr eval-streaming-asr --adapter funasr --dataset data/streaming_eval.jsonl
stable-asr compare-asr --config configs/eval/streaming_compare.yaml
```

paper commands:

```bash
stable-asr reproduce-paper --config configs/paper/paper_smoke.json --skip-train
stable-asr paper-table baselines --results runs/paper/smoke/paper_results.json
stable-asr paper-figure architecture --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/figures/architecture.svg
stable-asr paper-figure api_flow --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/figures/api_flow.svg
stable-asr paper-figure data_registry --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/figures/data_registry.svg
stable-asr paper-figure voiceworld_timeline --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/figures/voiceworld_timeline.svg
stable-asr paper-figure policy_state_machine --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/figures/policy_state_machine.svg
stable-asr paper-figure robustness_heatmap --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/figures/robustness_heatmap.svg
stable-asr paper-figure latency_quality_pareto --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/figures/latency_quality_pareto.svg
stable-asr paper-figure scenarios --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/figures/scenarios.svg
stable-asr paper-bundle --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts
stable-asr paper-audit --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts
stable-asr paper-draft --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/PAPER_DRAFT.md
stable-asr paper-latex --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/paper.tex
stable-asr paper-release-audit --repo-root . --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --markdown-draft runs/paper/smoke/PAPER_DRAFT.md --latex-draft runs/paper/smoke/paper.tex
```

## Example API Target

```python
import stable_asr as sasr

dataset = sasr.data.load_turn_dataset(
    "data/turn_train.jsonl",
    window_sec=2.0,
    sample_rate=16000,
)

model = sasr.turn.NanoTurnNano(
    labels=["complete", "incomplete", "backchannel", "wait"]
)

trainer = sasr.Trainer(model=model, batch_size=128, lr=3e-4)
trainer.fit(dataset.train, valid_dataset=dataset.valid)

policy = sasr.turn.TurnPolicy.from_model(
    model,
    complete_threshold=0.75,
    wait_threshold=0.60,
    backchannel_threshold=0.70,
    complete_hysteresis=2,
)

report = sasr.eval.evaluate_turn(
    policy=policy,
    dataset=dataset.valid,
    metrics=[
        "macro_f1",
        "false_complete_rate",
        "missed_interrupt_rate",
        "backchannel_precision",
        "decision_latency_ms",
    ],
)

report.save_markdown("reports/turn_eval.md")
```

## First Implementation Backlog

1. Create installable Python package skeleton.
2. Define manifest schema and validators.
3. Implement JSONL loader and fixture dataset.
4. Implement `TurnWindow`, `TurnPrediction`, and `TurnAction`.
5. Implement label normalization.
6. Implement rule endpoint baseline.
7. Implement VAD pause baseline.
8. Implement turn metrics.
9. Implement Markdown report writer.
10. Add NanoTurn log-mel frontend.
11. Add NanoTurn CNN-TCN or CNN-GRU model.
12. Add training loop.
13. Add checkpoint save/load.
14. Add ONNX export.
15. Add CLI commands for train/eval/export.

## Definition of Done for v0.1

Stable-ASR v0.1 is done when a user can:

```bash
pip install -e .
stable-asr validate-manifest examples/data/turn_demo.jsonl
stable-asr train-turn --config configs/nanoturn_nano.yaml
stable-asr eval-turn --model runs/nanoturn/best.pt --dataset examples/data/turn_demo.jsonl
stable-asr export-turn-onnx --checkpoint runs/nanoturn/best.pt
```

and receive:

- a trained NanoTurn checkpoint
- turn classification metrics
- endpointing interaction metrics
- CPU latency numbers
- an ONNX model
- a Markdown report
