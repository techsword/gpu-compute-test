# AGENTS.md

Guidance for agentic coding assistants working in `gpu-compute-test`.

## 1) Repo purpose and structure
- This repository contains Python smoke-test scripts for accelerator compute.
- Primary workload: Wav2Vec2 inference on a tiny LibriSpeech dummy split.
- Current scripts are top-level files (no package structure yet):
  - `compute-test.py`
  - Legacy split scripts were consolidated into this unified runner.
- Project configuration is in `pyproject.toml`.
- Dependency management is via `uv` (`uv.lock` is present).

## 2) Cursor/Copilot policy files
Checked locations:
- `.cursor/rules/` -> not present
- `.cursorrules` -> not present
- `.github/copilot-instructions.md` -> not present
Conclusion: there are currently no additional repository-specific Cursor/Copilot rules.

## 3) Environment setup
- Required Python version: `>=3.12`.
- Preferred setup:
  - `uv sync`
- Run commands in managed environment:
  - `uv run <command>`
If using an activated virtualenv directly:
- `source .venv/bin/activate`
- `python -m pip install -U pip`

## 4) Build / sanity commands
There is no dedicated packaging build pipeline yet. Treat build as runtime sanity:
- `uv sync`
- `uv run python -m compileall .`
- `uv run python compute-test.py --accelerator cpu`
Optional future packaging command (if backend is added):
- `uv build`

## 5) Lint and format commands
Ruff is configured in `pyproject.toml`.
- Lint check: `uv run ruff check .`
- Lint autofix: `uv run ruff check . --fix`
- Format check: `uv run ruff format . --check`
- Apply formatting: `uv run ruff format .`
- Local gate: `uv run ruff check . && uv run ruff format . --check`
Configured style settings:
- Line length: `88`
- Indent width: `4`
- Quote style: `double`
- Lint selection: `E4`, `E7`, `E9`, `F`

## 6) Test commands (current state)
- There is no `tests/` directory and no pytest suite at this time.
- Validation is performed by running smoke scripts.
Smoke-test commands:
- CPU path: `uv run python compute-test.py --accelerator cpu`
- CUDA path: `uv run python compute-test.py --accelerator cuda`
- XPU path: `uv run python compute-test.py --accelerator xpu`
- ROCm path: `uv run python compute-test.py --accelerator rocm --runtime rocm`
- Auto selection: `uv run python compute-test.py --accelerator auto`
Profiling commands:
- `uv run python compute-test.py --accelerator cpu --profile`
- `uv run python compute-test.py --accelerator cuda --profile`
- `uv run python compute-test.py --accelerator xpu --profile`
- `uv run python compute-test.py --accelerator rocm --profile`
- View traces in Perfetto: `https://ui.perfetto.dev` (drag `trace_step_*.json`)
- Fallback viewer: `chrome://tracing`

### Single-test guidance
Because pytest tests do not exist yet, a "single test" means one script for one device path.
- Recommended single-run validation: `uv run python compute-test.py --accelerator cpu --max-samples 1`
If pytest is added later, use this pattern:
- `uv run pytest tests/test_file.py::test_name -q`

## 7) Code style and implementation guidelines
### Imports
- Order imports as:
  1) standard library
  2) third-party
  3) local imports
- Keep imports explicit; avoid wildcard imports.
- Remove unused imports promptly.

### Formatting
- Follow Ruff formatting; do not hand-format against tool output.
- Use 4-space indentation and no tabs.
- Use double-quoted strings by default.
- Keep lines <= 88 characters unless formatter requires otherwise.

### Types and interfaces
- Add type hints for new functions and non-trivial values.
- Prefer concrete types over `Any` when feasible.
- Keep CLI flags explicit and help text actionable.
- For dictionaries, prefer typed keys/values where practical.

### Naming conventions
- Script filenames follow existing kebab-case style (`compute-test.py`).
- Use `snake_case` for functions/variables.
- Use `UPPER_SNAKE_CASE` for constants (`DEVICE`, `LOG_DIR`).
- Avoid one-letter names except in tiny local loops.

### Error handling
- Fail fast with clear, actionable error messages.
- Keep device fallback explicit (`cuda`/`xpu` to `cpu` when unavailable).
- Add context around external operations (model load, dataset load, profiler setup).
- Do not silently swallow exceptions.

### Runtime/performance behavior
- Keep inference models in eval mode (`model.eval()`).
- Ensure tensors/models are moved to the selected device consistently.
- Keep profiling optional and isolated to profiling paths.
- Avoid repeated heavy work inside inference loops.

### Logging and outputs
- Keep output concise and deterministic.
- Preserve progress bars for long loops where useful.
- Gate verbose artifacts (like per-sample transcriptions) behind explicit flags.

### Dependencies and artifacts
- Add/modify dependencies in `pyproject.toml`.
- Avoid machine-specific absolute paths.
- Keep generated logs and traces under `log/` or another ignored directory.
- Do not commit large generated artifacts unless explicitly requested.

### Script behavior expectations
- Keep existing CLI flags stable unless a task explicitly changes interface.
- For new flags, provide defaults and helpful `argparse` help text.
- Preserve existing device selection behavior and explicit fallback to CPU.
- Keep model/dataset identifiers configurable only when required by the task.
- Avoid adding hidden environment-variable dependencies without documentation.

### Editing discipline
- Prefer minimal diffs and keep changes close to the task scope.
- Do not rewrite unrelated scripts for style-only cleanup.
- When refactoring, preserve runtime behavior and output semantics.
- Remove dead code and unused imports only in files you touch.

## 8) Agent checklist before finalizing changes
- Run lint: `uv run ruff check .`
- Run format check: `uv run ruff format . --check`
- Run at least one relevant smoke command for your change.
- If touching device-specific code, run that device path when hardware is available.
- Update `README.md` for behavior or CLI changes.
- Keep changes minimal and scoped; avoid unrelated refactors.

## 9) Future test migration note
If pytest tests are introduced:
- Add `tests/` with focused unit/integration tests.
- Document exact single-test commands in this file.
- Prefer offline-friendly tests (mock remote model/dataset calls).
