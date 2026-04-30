"""Re-export shim — ``PPOPolicy`` and helpers moved to ``magic_ai.slot_encoder.model``.

Types that were previously defined here and have since moved to ``magic_ai.actions``
(``ParsedBatch``, ``ParsedStep``, ``PolicyStep``, ``ActionTrace``, ``TraceKind``,
``TRACE_KIND_VALUES``, ``TRACE_KIND_TO_ID``) are also re-exported here for backward
compatibility.

External code that imports from ``magic_ai.model`` continues to work; only the
definitions have moved.
"""

from magic_ai.actions import (  # noqa: F401
    TRACE_KIND_TO_ID,
    TRACE_KIND_VALUES,
    ActionTrace,
    ParsedBatch,
    ParsedStep,
    PolicyStep,
    TraceKind,
)
from magic_ai.slot_encoder.model import (  # noqa: F401
    PPOPolicy,
    _clone_detaching_buffer,
)

__all__ = [
    "TRACE_KIND_TO_ID",
    "TRACE_KIND_VALUES",
    "ActionTrace",
    "ParsedBatch",
    "ParsedStep",
    "PolicyStep",
    "PPOPolicy",
    "TraceKind",
    "_clone_detaching_buffer",
]
