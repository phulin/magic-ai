from __future__ import annotations

import unittest

from magic_ai.slot_encoder.encoder_parity import (
    build_sample_encoders,
    build_sample_parity_cases,
    compare_batch_outputs,
    encode_python_batch,
    encode_python_reference,
    expected_state_shapes,
    load_batch_encoder,
    run_parity_suite,
)


class EncoderParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game_state_encoder, self.action_encoder = build_sample_encoders()
        self.cases = build_sample_parity_cases()

    def test_python_batch_matches_single_item_reference(self) -> None:
        states = [case.state for case in self.cases]
        pendings = [case.pending for case in self.cases]
        perspective_player_indices = [case.perspective_player_idx for case in self.cases]

        expected = encode_python_reference(
            self.game_state_encoder,
            self.action_encoder,
            states,
            pendings,
            perspective_player_indices,
        )
        actual = encode_python_batch(
            self.game_state_encoder,
            self.action_encoder,
            states,
            pendings,
            perspective_player_indices,
        )

        self.assertEqual(compare_batch_outputs(expected, actual), [])

    def test_parity_suite_exercises_multiple_batch_sizes(self) -> None:
        results = run_parity_suite(
            game_state_encoder=self.game_state_encoder,
            action_encoder=self.action_encoder,
            cases=self.cases,
            batch_sizes=(1, 2, 4),
        )

        self.assertEqual(len(results), 7)
        self.assertTrue(all(result.ok for result in results))

    def test_expected_shapes_match_batch_output(self) -> None:
        states = [case.state for case in self.cases[:2]]
        pendings = [case.pending for case in self.cases[:2]]
        perspective_player_indices = [case.perspective_player_idx for case in self.cases[:2]]
        outputs = encode_python_batch(
            self.game_state_encoder,
            self.action_encoder,
            states,
            pendings,
            perspective_player_indices,
        )

        shapes = expected_state_shapes(batch_size=2)
        self.assertEqual(tuple(outputs.parsed_state.slot_card_rows.shape), shapes["slot_card_rows"])
        self.assertEqual(tuple(outputs.parsed_state.slot_occupied.shape), shapes["slot_occupied"])
        self.assertEqual(tuple(outputs.parsed_state.slot_tapped.shape), shapes["slot_tapped"])
        self.assertEqual(tuple(outputs.parsed_state.game_info.shape), shapes["game_info"])

    def test_loader_resolves_builtin_batch_encoder(self) -> None:
        candidate = load_batch_encoder("magic_ai.slot_encoder.encoder_parity:encode_python_batch")
        self.assertIs(candidate, encode_python_batch)


if __name__ == "__main__":
    unittest.main()
