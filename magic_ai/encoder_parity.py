"""Re-export shim — moved to ``magic_ai.slot_encoder.encoder_parity``.

External code that imports from ``magic_ai.encoder_parity`` continues to work;
only the definition has moved.
"""

from magic_ai.slot_encoder.encoder_parity import (  # noqa: F401
    BatchEncoder,
    EncoderBatchOutputs,
    EncoderParityCase,
    EncoderParityResult,
    assert_batch_outputs_match,
    build_sample_encoders,
    build_sample_parity_cases,
    compare_batch_outputs,
    encode_python_batch,
    encode_python_reference,
    expected_state_shapes,
    load_batch_encoder,
    priority_candidates_equal,
    run_parity_suite,
)

__all__ = [
    "BatchEncoder",
    "EncoderBatchOutputs",
    "EncoderParityCase",
    "EncoderParityResult",
    "assert_batch_outputs_match",
    "build_sample_encoders",
    "build_sample_parity_cases",
    "compare_batch_outputs",
    "encode_python_batch",
    "encode_python_reference",
    "expected_state_shapes",
    "load_batch_encoder",
    "priority_candidates_equal",
    "run_parity_suite",
]
