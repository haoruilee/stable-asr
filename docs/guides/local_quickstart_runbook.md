# Local Quickstart Runbook

This runbook is for any user who wants to try Stable-ASR on their own machine
without staging external corpora or model outputs.

## Install

Base install:

```bash
git clone https://github.com/haoruilee/stable-asr.git
cd stable-asr
bash scripts/run_quickstart.sh setup
```

Install all optional local extras, including Torch, ONNX, PyArrow, and Lance:

```bash
bash scripts/run_quickstart.sh setup-all
```

## Run

Smoke checks that require no external data:

```bash
bash scripts/run_quickstart.sh smoke
```

Synthetic audio, data-layer conversion, and correctness-checked cache
benchmarks:

```bash
bash scripts/run_quickstart.sh data
```

NanoTurn demo training, if Torch is installed:

```bash
bash scripts/run_quickstart.sh train
```

Fixture-backed ASR adapter demos:

```bash
bash scripts/run_quickstart.sh adapters
```

Everything together:

```bash
bash scripts/run_quickstart.sh all
```

Outputs are written under `runs/quickstart` by default. To use another
directory:

```bash
RUN_DIR=/tmp/stable-asr-demo bash scripts/run_quickstart.sh all
```

The script automatically uses optional formats when the dependency is
installed. For example, Parquet is enabled when PyArrow is present, and Lance is
enabled when the `lance` Python package is present.

The script resolves the repository root from its own path, so it can also be
called from outside the checkout:

```bash
/path/to/stable-asr/scripts/run_quickstart.sh smoke
```
