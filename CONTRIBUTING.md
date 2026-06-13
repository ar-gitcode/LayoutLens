# Contributing

This repository is a research prototype released alongside an undergraduate
thesis. It is shared primarily for transparency and reproducibility rather than
as an actively maintained library, so support and review may be limited.
Contributions and bug reports are still welcome.

## Reporting issues

When opening an issue, please include:

- the command you ran and the full error output,
- your operating system, Python version, and PyTorch / CUDA versions,
- which configuration (and dataset) you used.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"
```

Run the test suite before submitting changes:

```bash
pytest evaluations/tests
```

## Pull requests

- Keep changes focused and describe the motivation in the PR description.
- Match the style of the surrounding code.
- Do not commit datasets, checkpoints, logs, W&B runs, caches, or other large or
  private artifacts. The [`.gitignore`](.gitignore) already excludes the common
  cases; please check `git status` before committing.
- Do not introduce hard-coded local paths. Use the `ROOMENV_*` environment
  variables and the helpers in [`room_envelopes/paths.py`](room_envelopes/paths.py).

## Code of conduct

Please be respectful and constructive in all interactions.
