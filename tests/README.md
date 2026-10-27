# Running the tests

```bash
pytest tests -q          # everything the machine can run
pytest tests -q --run-slow   # adds the tests that download real models
```

On a machine with no model stack installed that gives roughly:

```
144 passed, 23 skipped
```

Skipped is the expected result there, not a problem. **Nothing should error.**

## Three tiers

| tier | needs | how it is marked |
|---|---|---|
| **pure** | nothing beyond pytest | no marker — always runs |
| **optional-dependency** | `torch`, `fastapi`, `numpy`, `pydantic` | `pytest.importorskip(...)` at the top of the module |
| **slow** | downloads real model weights, sometimes a GPU | `@pytest.mark.slow`, skipped unless `--run-slow` |

CI installs `requirements-ci.txt`, so the first two tiers all run there on every
pull request. The slow tier does not run in CI at all.

## Why the import guards exist

A test module that imports `torch` at the top used to **fail collection** on a
machine without torch. Eighteen modules did, so `pytest tests` reported
`18 errors` and exited non-zero before running anything.

That is worse than it sounds. Collection errors are indistinguishable at a
glance from genuine failures, so a real failing test hides among them — which
is exactly how a broken JSON file reached `main` in PR #50 with its guard test
already failing. Making the optional-dependency tier skip cleanly means a
non-zero exit now means something is actually wrong.

It also lets anyone run the suite before pushing. Only the Colab environment
has the full stack; whoever is working on the UI, the data layer or the
evaluation harness should not need a multi-gigabyte install to check they have
not broken something.

## Adding a test that needs a heavy dependency

Put the guard above the heavy imports, not merely above the `src.` ones — the
third-party import is usually what fails first:

```python
from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="needs optional torch; see tests/README.md")

import torch
from src.risk_model.model import ModernBertRiskClassifier
```

If the dependency is only needed by one test rather than the whole module, use
`pytest.importorskip` inside that test instead, so the rest of the file keeps
running.

For a test that downloads weights or wants a GPU, mark it `@pytest.mark.slow`
rather than guarding the import — it is not about what is installed, it is
about what the test costs to run.
