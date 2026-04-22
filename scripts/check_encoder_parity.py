"""Run batch encoder parity checks against the Python reference implementation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from magic_ai.encoder_parity import (  # noqa: E402
    build_sample_encoders,
    build_sample_parity_cases,
    encode_python_batch,
    load_batch_encoder,
    run_parity_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a batch encoder implementation against the Python reference path."
    )
    parser.add_argument(
        "--candidate",
        default="magic_ai.encoder_parity:encode_python_batch",
        help="batch encoder in module:callable form",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4",
        help="comma-separated batch sizes to exercise",
    )
    args = parser.parse_args()

    batch_sizes = tuple(int(part.strip()) for part in args.batch_sizes.split(",") if part.strip())
    candidate_encoder = (
        encode_python_batch
        if args.candidate == "magic_ai.encoder_parity:encode_python_batch"
        else load_batch_encoder(args.candidate)
    )
    game_state_encoder, action_encoder = build_sample_encoders()
    cases = build_sample_parity_cases()
    results = run_parity_suite(
        game_state_encoder=game_state_encoder,
        action_encoder=action_encoder,
        cases=cases,
        candidate_encoder=candidate_encoder,
        batch_sizes=batch_sizes,
    )

    failed = [result for result in results if not result.ok]
    if failed:
        for result in failed:
            names = ", ".join(result.batch_case_names)
            print(f"FAIL [{names}]", file=sys.stderr)
            for mismatch in result.mismatches:
                print(f"  {mismatch}", file=sys.stderr)
        return 1

    print(
        f"PASS candidate={args.candidate} cases={len(cases)} comparisons={len(results)}",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
