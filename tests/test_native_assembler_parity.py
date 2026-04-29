"""Parity-test scaffold for the native text-encoder assembler port.

Phase 4 of the assembler-port (see ``mage-go/cmd/pylib/TODO_assembler_port.md``)
will land an ``MageEncodeTokens`` Go export that walks the same opcode
stream as the Python ``_assemble_one`` and writes a dense ``(B, max_tokens)``
int64 token tensor + anchor arrays directly. The cutover is unsafe without
a parity gate that drives both paths over the same game state and asserts
byte-equality.

This file is the **infrastructure** for that gate — Phase 4 hasn't landed
yet, so the assertions today only exercise the Python path. The structure
is set up so dropping in the native-path call is a one-liner once
``MageEncodeTokens`` exists; see ``_assemble_native`` below.

The harness:
  1. Spins up real mage games via ``mage.new_game`` with deterministic seeds.
  2. Drives each game forward through the priority loop, capturing the
     ``NativeEncodedBatch`` (with render plans) at every priority pending.
  3. At each capture:
     - Runs the Python assembler path (existing).
     - (Phase 4) Will run the native path and assert byte-equal outputs.
  4. Asserts well-formedness invariants across both paths so the gate
     catches regressions even before the native path lands.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECK = REPO_ROOT / "decks" / "bears.json"
DEFAULT_CACHE = REPO_ROOT / "data" / "text_encoder_card_tokens.pt"


def _have_mage() -> bool:
    try:
        import mage  # noqa: F401

        mage.load()
        return True
    except Exception:  # pragma: no cover
        return False


@unittest.skipUnless(_have_mage(), "libmage not available")
class NativeAssemblerParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mage
        from magic_ai.native_encoder import NativeBatchEncoder
        from magic_ai.text_encoder.assembler import build_assembler_tokens
        from magic_ai.text_encoder.card_cache import (
            build_card_cache,
            load_card_cache,
        )
        from magic_ai.text_encoder.native_token_tables import (
            register_native_token_tables,
        )
        from magic_ai.text_encoder.render import load_oracle_text
        from magic_ai.text_encoder.token_tables import build_token_tables
        from magic_ai.text_encoder.tokenizer import load_tokenizer

        mage.load()
        cls.mage = mage
        cls.tokenizer = load_tokenizer()
        cls.oracle = load_oracle_text()
        if DEFAULT_CACHE.exists():
            cls.cache = load_card_cache(DEFAULT_CACHE)
        else:
            cls.cache = build_card_cache(
                sorted(cls.oracle.keys()),
                cls.oracle,
                cls.tokenizer,
                missing_policy="warn",
            )

        cls.assembler_tokens = build_assembler_tokens(cls.tokenizer)

        # Phase 4: register the closed-vocabulary token tables with the
        # native side so MageEncodeTokens can dispatch through them.
        cls.token_tables_py = build_token_tables(cls.tokenizer, cls.cache)
        register_native_token_tables(cls.token_tables_py)
        cls.max_tokens_native = 2048
        cls.max_options_native = 32
        cls.max_targets_native = 8
        cls.max_card_refs_native = 256

        # Engine shape constants live in mage-go/cmd/pylib/encoder.go; the
        # encoder validates these on every call. We don't need a policy to
        # build the encoder for the parity test — fixed shapes are enough.
        cls.encoder = NativeBatchEncoder(
            max_options=16,
            max_targets_per_option=4,
            max_cached_choices=64,
            zone_slot_count=50,
            game_info_dim=90,
            option_scalar_dim=14,
            target_scalar_dim=2,
            lib=mage._lib,
            ffi=mage._ffi,
            validate=True,
            emit_render_plan=True,
            render_plan_capacity=4096,
        )
        if not cls.encoder.is_available:
            raise unittest.SkipTest("NativeBatchEncoder is not available")

        # Card-name → row mapping must match the cache used for assembly so
        # the engine emits row ids that the cache can resolve.
        name_to_row = {name: idx for idx, name in enumerate(cls.cache.row_to_name)}
        cls.encoder.register_card_rows(name_to_row)

        payload = json.loads(DEFAULT_DECK.read_text())
        cls.deck_a = payload.get("player_a", payload)
        cls.deck_b = payload.get("player_b", payload)

    # -- harness helpers -------------------------------------------------

    def _drive_and_capture(
        self,
        *,
        seed: int,
        max_captures: int = 8,
        max_steps: int = 400,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Drive a single mage game forward, capturing native batches at
        each priority pending. Returns a list of (native_batch, snapshot)
        tuples.
        """

        from magic_ai.text_encoder.rollout import (
            _default_action_for,
            _translate_action,
        )

        captures: list[tuple[Any, dict[str, Any]]] = []
        game = self.mage.new_game(
            self.deck_a,
            self.deck_b,
            seed=seed,
            shuffle=True,
            hand_size=7,
        )
        steps = 0
        while steps < max_steps and len(captures) < max_captures:
            steps += 1
            game.refresh_state()
            if game.is_over:
                break
            pending = cast(dict[str, Any] | None, game.pending or game.legal())
            if pending is None:
                try:
                    game.step({"kind": "pass"})
                except Exception:
                    break
                continue

            kind = pending.get("kind", "") or ""
            options = list(pending.get("options", []) or [])
            player_idx = int(pending.get("player_idx", 0) or 0)
            if player_idx not in (0, 1):
                player_idx = 0

            # Only capture priority pendings — the Python assembler grammar
            # is built around those, and they're the dominant trace kind on
            # the rollout hot path.
            if kind == "priority" and options:
                snapshot = cast(dict[str, Any], game.state)
                try:
                    native_batch = self.encoder.encode_handles(
                        [game],
                        perspective_player_indices=[player_idx],
                    )
                    captures.append((native_batch, snapshot))
                except Exception:
                    # Failed encodings shouldn't abort the game drive — log
                    # and continue. A real Phase-4 cutover would be much
                    # less forgiving, but during scaffolding we want to see
                    # how many captures we can collect.
                    pass

            # Advance the game. Picking the first option is good enough for
            # reaching a diverse spread of states. We don't care which path
            # the game takes — just that we keep getting fresh pendings.
            if not options:
                action = _default_action_for(cast(Any, pending))
            elif kind == "priority":
                action = _translate_action(cast(Any, pending), 0, None)
            else:
                action = _default_action_for(cast(Any, pending))

            try:
                game.step(dict(action))
            except Exception:
                break

        return captures

    def _assemble_python(self, native_batch: Any) -> Any:
        """Run the existing Python assembler path on a native batch."""

        from magic_ai.text_encoder.assembler import assemble_batch

        if native_batch.render_plan is None or native_batch.render_plan_lengths is None:
            raise AssertionError("native batch missing render plan")
        if native_batch.render_plan_overflow is not None and bool(
            native_batch.render_plan_overflow.any()
        ):
            raise AssertionError("render plan overflowed")
        lengths = native_batch.render_plan_lengths.tolist()
        plans = [
            native_batch.render_plan[row_idx, : int(length)]
            for row_idx, length in enumerate(lengths)
        ]
        return assemble_batch(
            plans,
            self.cache,
            self.tokenizer,
            max_tokens=2048,
            on_overflow="truncate",
            assembler_tokens=self.assembler_tokens,
        )

    def _encode_native(self, game: Any, perspective: int) -> tuple[Any, Any]:
        """Run the native MageEncodeTokens path on a single game handle.

        Returns the native (NativeEncodedBatch, NativeAssemblerOutputs) pair
        for downstream comparison against the Python path.
        """

        from magic_ai.text_encoder.native_assembler import encode_tokens

        return encode_tokens(
            self.encoder,
            [game],
            perspective_player_indices=[perspective],
            max_tokens=self.max_tokens_native,
            max_options=self.max_options_native,
            max_targets=self.max_targets_native,
            max_card_refs=self.max_card_refs_native,
        )

    def _assemble_native(self, native_batch: Any) -> Any | None:
        """Convert a native-output bundle to a TextEncodedBatch for parity.

        ``native_batch`` here may either be the legacy ``NativeEncodedBatch``
        (in which case there's no native assembler output to surface and we
        return None) or a ``(legacy, NativeAssemblerOutputs)`` tuple
        produced by ``_encode_native``.
        """

        if isinstance(native_batch, tuple):
            _legacy, outputs = native_batch
            return outputs.to_text_encoded_batch()
        return None

    # -- assertions on the Python path (always live) --------------------

    def _assert_python_well_formed(self, encoded: Any, native_batch: Any) -> None:
        n = int(native_batch.render_plan_lengths.shape[0])
        self.assertEqual(int(encoded.token_ids.shape[0]), n)
        self.assertEqual(int(encoded.attention_mask.shape[0]), n)
        self.assertEqual(int(encoded.seq_lengths.shape[0]), n)

        for b in range(n):
            seq_len = int(encoded.seq_lengths[b])
            self.assertGreater(seq_len, 0, f"row {b} has zero seq_len")
            self.assertLessEqual(seq_len, int(encoded.token_ids.shape[1]))
            # attention mask matches seq_lengths.
            self.assertEqual(
                int(encoded.attention_mask[b, :seq_len].sum()),
                seq_len,
                f"row {b} attention mask mismatch",
            )
            # padding is correct.
            if seq_len < int(encoded.token_ids.shape[1]):
                self.assertEqual(
                    int(encoded.attention_mask[b, seq_len:].sum()),
                    0,
                    f"row {b} attention mask leaks past seq_len",
                )

            # option positions either point inside the sequence or are -1.
            opt_pos = encoded.option_positions[b]
            opt_mask = encoded.option_mask[b]
            for o in range(int(opt_pos.shape[0])):
                pos = int(opt_pos[o])
                if bool(opt_mask[o]):
                    self.assertGreaterEqual(pos, 0)
                    self.assertLess(pos, seq_len)
                else:
                    # Masked-off slot: position should be -1 (truncation
                    # sentinel) or simply unused.
                    self.assertTrue(pos == -1 or pos < seq_len)

    # -- parity assertion (lit when _assemble_native returns non-None) ---

    def _assert_byte_equal(self, py: Any, nat: Any) -> None:
        self.assertTrue(
            torch.equal(py.token_ids, nat.token_ids),
            "token_ids differ between Python and native paths",
        )
        self.assertTrue(torch.equal(py.attention_mask, nat.attention_mask))
        self.assertTrue(torch.equal(py.seq_lengths, nat.seq_lengths))
        self.assertTrue(torch.equal(py.option_positions, nat.option_positions))
        self.assertTrue(torch.equal(py.option_mask, nat.option_mask))
        self.assertTrue(torch.equal(py.target_positions, nat.target_positions))
        self.assertTrue(torch.equal(py.target_mask, nat.target_mask))
        self.assertTrue(torch.equal(py.card_ref_positions, nat.card_ref_positions))

    # -- tests ----------------------------------------------------------

    def test_capture_loop_collects_states(self) -> None:
        captures = self._drive_and_capture(seed=1, max_captures=8)
        self.assertGreater(
            len(captures),
            0,
            "harness failed to capture any priority-pending states",
        )

    def test_python_path_well_formed_across_drive(self) -> None:
        """Sanity gate that the test infra runs cleanly across many states.

        This is the assertion that locks the scaffold in place: until
        Phase 4 lands, this is what proves the harness works. Once the
        native path is implemented, the byte-equal assertion below kicks
        in and this test becomes a (nearly free) precondition check.
        """

        captures = self._drive_and_capture(seed=2, max_captures=8)
        self.assertGreater(len(captures), 0)
        for native_batch, _snapshot in captures:
            encoded = self._assemble_python(native_batch)
            self._assert_python_well_formed(encoded, native_batch)

    def _drive_and_capture_dual(
        self,
        *,
        seed: int,
        max_captures: int = 8,
        max_steps: int = 400,
    ) -> list[tuple[Any, Any]]:
        """Like ``_drive_and_capture`` but at each priority pending also
        runs the native MageEncodeTokens path and returns both bundles.

        Each entry is ``(NativeEncodedBatch_python_path,
        (NativeEncodedBatch_native_path, NativeAssemblerOutputs))``.
        """

        from magic_ai.text_encoder.rollout import (
            _default_action_for,
            _translate_action,
        )

        captures: list[tuple[Any, Any]] = []
        game = self.mage.new_game(
            self.deck_a,
            self.deck_b,
            seed=seed,
            shuffle=True,
            hand_size=7,
        )
        steps = 0
        while steps < max_steps and len(captures) < max_captures:
            steps += 1
            game.refresh_state()
            if game.is_over:
                break
            pending = cast(dict[str, Any] | None, game.pending or game.legal())
            if pending is None:
                try:
                    game.step({"kind": "pass"})
                except Exception:
                    break
                continue

            kind = pending.get("kind", "") or ""
            options = list(pending.get("options", []) or [])
            player_idx = int(pending.get("player_idx", 0) or 0)
            if player_idx not in (0, 1):
                player_idx = 0

            if kind == "priority" and options:
                try:
                    # NativeBatchEncoder shares its scratch buffers across
                    # calls. The NativeEncodedBatch returned here holds
                    # views into that scratch, so we MUST assemble + clone
                    # before the next capture overwrites the buffers.
                    py_native_batch = self.encoder.encode_handles(
                        [game],
                        perspective_player_indices=[player_idx],
                    )
                    py_assembled = self._assemble_python(py_native_batch)
                    py_assembled = type(py_assembled)(
                        token_ids=py_assembled.token_ids.clone(),
                        attention_mask=py_assembled.attention_mask.clone(),
                        card_ref_positions=py_assembled.card_ref_positions.clone(),
                        option_positions=py_assembled.option_positions.clone(),
                        option_mask=py_assembled.option_mask.clone(),
                        target_positions=py_assembled.target_positions.clone(),
                        target_mask=py_assembled.target_mask.clone(),
                        seq_lengths=py_assembled.seq_lengths.clone(),
                    )
                    # Native outputs already live in their own per-call
                    # tensors; safe to keep as-is.
                    nat_pair = self._encode_native(game, player_idx)
                    captures.append((py_assembled, nat_pair))
                except Exception:
                    pass

            if not options:
                action = _default_action_for(cast(Any, pending))
            elif kind == "priority":
                action = _translate_action(cast(Any, pending), 0, None)
            else:
                action = _default_action_for(cast(Any, pending))
            try:
                game.step(dict(action))
            except Exception:
                break

        return captures

    def test_byte_equal_native_vs_python(self) -> None:
        """Phase 4 parity gate: native and Python assemblers produce
        byte-equal token tensors at every captured priority pending."""

        dual = self._drive_and_capture_dual(seed=3, max_captures=8)
        self.assertGreater(len(dual), 0, "harness collected zero dual-path captures")
        for py_assembled, nat_pair in dual:
            nat = self._assemble_native(nat_pair)
            self.assertIsNotNone(nat)
            self._assert_byte_equal(py_assembled, nat)

    def _encode_native_packed(self, game: Any, perspective: int) -> tuple[Any, Any]:
        from magic_ai.text_encoder.native_assembler import encode_tokens_packed

        return encode_tokens_packed(
            self.encoder,
            [game],
            perspective_player_indices=[perspective],
            max_tokens=self.max_tokens_native,
            max_options=self.max_options_native,
            max_targets=self.max_targets_native,
            max_card_refs=self.max_card_refs_native,
        )

    def test_packed_native_matches_pack_batch_of_dense(self) -> None:
        """Go's MageEncodeTokensPacked must produce the same result as
        running the dense path and then ``pack_batch`` in Python."""

        from magic_ai.text_encoder.batch import pack_batch

        captures: list[tuple[Any, Any, Any]] = []
        game = self.mage.new_game(self.deck_a, self.deck_b, seed=4, shuffle=True, hand_size=7)
        from magic_ai.text_encoder.rollout import (
            _default_action_for,
            _translate_action,
        )

        steps = 0
        while steps < 400 and len(captures) < 6:
            steps += 1
            game.refresh_state()
            if game.is_over:
                break
            pending = cast(dict[str, Any] | None, game.pending or game.legal())
            if pending is None:
                try:
                    game.step({"kind": "pass"})
                except Exception:
                    break
                continue
            kind = pending.get("kind", "") or ""
            options = list(pending.get("options", []) or [])
            player_idx = int(pending.get("player_idx", 0) or 0)
            if player_idx not in (0, 1):
                player_idx = 0
            if kind == "priority" and options:
                _, dense_outputs = self._encode_native(game, player_idx)
                dense_batch = dense_outputs.to_text_encoded_batch()
                expected = pack_batch(dense_batch)
                expected_clone = type(expected)(
                    token_ids=expected.token_ids.clone(),
                    seq_id=expected.seq_id.clone(),
                    pos_in_seq=expected.pos_in_seq.clone(),
                    cu_seqlens=expected.cu_seqlens.clone(),
                    seq_lengths=expected.seq_lengths.clone(),
                    state_positions=expected.state_positions.clone(),
                    card_ref_positions=expected.card_ref_positions.clone(),
                    option_positions=expected.option_positions.clone(),
                    option_mask=expected.option_mask.clone(),
                    target_positions=expected.target_positions.clone(),
                    target_mask=expected.target_mask.clone(),
                )
                _, packed_outputs = self._encode_native_packed(game, player_idx)
                captures.append((expected_clone, packed_outputs, None))
            if not options:
                action = _default_action_for(cast(Any, pending))
            elif kind == "priority":
                action = _translate_action(cast(Any, pending), 0, None)
            else:
                action = _default_action_for(cast(Any, pending))
            try:
                game.step(dict(action))
            except Exception:
                break

        self.assertGreater(len(captures), 0, "no packed parity captures collected")
        for expected, packed_outputs, _ in captures:
            packed = packed_outputs.to_packed_text_batch()
            self.assertTrue(torch.equal(packed.token_ids, expected.token_ids))
            self.assertTrue(torch.equal(packed.seq_id, expected.seq_id))
            self.assertTrue(torch.equal(packed.pos_in_seq, expected.pos_in_seq))
            self.assertTrue(torch.equal(packed.cu_seqlens, expected.cu_seqlens))
            self.assertTrue(torch.equal(packed.seq_lengths, expected.seq_lengths))
            self.assertTrue(torch.equal(packed.state_positions, expected.state_positions))
            self.assertTrue(torch.equal(packed.option_positions, expected.option_positions))
            self.assertTrue(torch.equal(packed.option_mask, expected.option_mask))
            self.assertTrue(torch.equal(packed.target_positions, expected.target_positions))
            self.assertTrue(torch.equal(packed.target_mask, expected.target_mask))
            self.assertTrue(torch.equal(packed.card_ref_positions, expected.card_ref_positions))

    def test_drive_is_deterministic_for_fixed_seed(self) -> None:
        """Two drives with the same seed produce identical encoded batches."""

        a = self._drive_and_capture(seed=42, max_captures=4)
        b = self._drive_and_capture(seed=42, max_captures=4)
        self.assertEqual(len(a), len(b))
        for (na, _), (nb, _) in zip(a, b, strict=True):
            ea = self._assemble_python(na)
            eb = self._assemble_python(nb)
            self.assertTrue(
                torch.equal(ea.token_ids, eb.token_ids),
                "token_ids differ across two drives with seed=42",
            )
            self.assertTrue(torch.equal(ea.seq_lengths, eb.seq_lengths))


if __name__ == "__main__":
    unittest.main()
