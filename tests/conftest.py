from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Replace torch.compile-wrapped GAE with its eager original. The compile cold
# start cost (~5s) dwarfs the actual GAE math on the tiny tensors used in
# tests, and the eager path exercises the same logic.
import magic_ai.returns as _returns  # noqa: E402

_orig = getattr(_returns._gae_returns_batched_compiled, "_torchdynamo_orig_callable", None)
if _orig is not None:
    _returns._gae_returns_batched_compiled = _orig  # type: ignore[assignment]
