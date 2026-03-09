# gpu-compute-test

Unified smoke test script for validating Wav2Vec2 inference across CPU, CUDA,
ROCm, and Intel XPU environments.

## Quick start

Install dependencies:

```bash
uv sync
```

Run a basic smoke test (auto-select best available accelerator):

```bash
uv run python compute-test.py --accelerator auto --max-samples 2
```

## Install PyTorch for your hardware

PyTorch dependencies are intentionally **not pinned in** `pyproject.toml`.
GPU-specific wheels vary by platform and index URL, so install PyTorch stack
explicitly for your environment before running the script.

Examples (adjust versions as needed):

CPU:

```bash
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
```

CUDA 11.8:

```bash
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.4:

```bash
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
```

ROCm 6.4:

```bash
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

Intel XPU:

```bash
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/xpu
```

Then install project dependencies:

```bash
uv sync
```

## Unified CLI

Main script: `compute-test.py`

```bash
uv run python compute-test.py [options]
```

Key flags:

- `--accelerator {auto,cpu,cuda,xpu,rocm}` choose accelerator preference
- `--runtime {any,rocm,cuda11,cuda12}` assert expected torch runtime family
- `--strict` fail instead of CPU fallback when request is unavailable/mismatched
- `--max-samples N` run only first N samples (fast smoke path)
- `--print-transcriptions` print decoded output
- `--profile` enable torch profiler trace collection
- `--profile-dir DIR` output directory for profiler traces
- `--model-id`, `--dataset-id`, `--dataset-config`, `--split` override defaults

## Examples

CPU:

```bash
uv run python compute-test.py --accelerator cpu --max-samples 2
```

ROCm (strict):

```bash
uv run python compute-test.py --accelerator rocm --runtime rocm --strict
```

CUDA 12 (strict runtime check):

```bash
uv run python compute-test.py --accelerator cuda --runtime cuda12 --strict
```

Intel XPU:

```bash
uv run python compute-test.py --accelerator xpu --strict
```

Profiler run:

```bash
uv run python compute-test.py --accelerator auto --profile --max-samples 5
```

Profiler output files are Chrome/Perfetto trace JSON files.

View traces with:

```bash
# Open in browser and drag trace_step_*.json files
https://ui.perfetto.dev

# Or use Chrome's built-in trace viewer
chrome://tracing
```

## Validation commands

```bash
uv run ruff check .
uv run ruff format . --check
uv run python compute-test.py --accelerator cpu --max-samples 1
```
