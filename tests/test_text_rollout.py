"""Smoke tests for :mod:`magic_ai.text_encoder.rollout`.

These exercise the cache + emitter + assembler + RecurrentTextPolicy
pipeline against the live mage engine. ``mage`` is required at import
time; the tests skip if the shared library can't be loaded.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECK = REPO_ROOT / "decks" / "bears.json"
DEFAULT_CACHE = REPO_ROOT / "data" / "text_encoder_card_tokens.npz"


def _have_mage() -> bool:
    try:
        import mage  # noqa: F401

        mage.load()
        return True
    except Exception:  # pragma: no cover - environment-dependent
        return False


@unittest.skipUnless(_have_mage(), "libmage not available")
class TextRolloutSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from magic_ai.text_encoder.card_cache import (
            build_card_cache,
            load_card_cache,
        )
        from magic_ai.text_encoder.model import TextEncoderConfig
        from magic_ai.text_encoder.recurrent import (
            RecurrentTextPolicy,
            RecurrentTextPolicyConfig,
        )
        from magic_ai.text_encoder.render import load_oracle_text
        from magic_ai.text_encoder.tokenizer import load_tokenizer

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

        cfg = TextEncoderConfig(
            vocab_size=len(cls.tokenizer),
            pad_id=int(cls.tokenizer.pad_token_id or 0),
            d_model=64,
            n_layers=1,
            n_heads=2,
        )
        rcfg = RecurrentTextPolicyConfig(encoder=cfg, lstm_hidden=64)
        cls.policy = RecurrentTextPolicy(rcfg)
        cls.policy.eval()

        payload = json.loads(DEFAULT_DECK.read_text())
        cls.deck_a = payload.get("player_a", payload)
        cls.deck_b = payload.get("player_b", payload)

    def _make_worker(self, seed: int = 0):
        from magic_ai.text_encoder.rollout import TextRolloutWorker

        return TextRolloutWorker(
            policy=self.policy,
            cache=self.cache,
            tokenizer=self.tokenizer,
            max_tokens=2048,
            device="cpu",
            sampling_temperature=1.0,
            oracle=self.oracle,
            seed=seed,
        )

    def test_single_episode_completes(self) -> None:
        worker = self._make_worker(seed=0)
        cfg = {
            "deck_a": self.deck_a,
            "deck_b": self.deck_b,
            "seed": 1,
            "shuffle": True,
        }
        episode = worker.play_episode(cfg, max_turns=10)
        self.assertGreaterEqual(len(episode.steps), 1)
        for step in episode.steps:
            self.assertIn(step.reward, (-1.0, 0.0, 1.0))
        # Sum of rewards over an episode is a small integer in {-N, 0, +N}
        # depending on the ratio of winner / loser steps. Just assert finite.
        total = sum(s.reward for s in episode.steps)
        self.assertTrue(total == int(total))

    def test_action_well_formed_against_legal(self) -> None:
        worker = self._make_worker(seed=1)
        cfg = {
            "deck_a": self.deck_a,
            "deck_b": self.deck_b,
            "seed": 2,
            "shuffle": True,
        }
        episode = worker.play_episode(cfg, max_turns=10)
        for step in episode.steps:
            # The chosen option index must point inside the legal_options
            # list captured at that decision (no off-by-one in the worker).
            self.assertGreaterEqual(step.chosen_option_idx, 0)
            self.assertLess(step.chosen_option_idx, len(step.legal_options))
            chosen = step.legal_options[step.chosen_option_idx]
            # Sanity: the chosen option is a real engine record (a dict with
            # at least a ``kind``). Some kinds (e.g. ``pass``) carry no id.
            self.assertIsInstance(chosen, dict)
            self.assertIn("kind", chosen)

    def test_per_player_state_independence(self) -> None:
        """The two LSTM states are advanced independently per perspective.

        We patch the worker to record the (h, c) seen on each policy call
        and assert: between two consecutive policy calls for the same
        perspective player, exactly one LSTM call happened on that
        player's state — i.e. the *next* call's input state equals the
        *previous* call's output state, with nothing else mutating it in
        between.
        """

        from magic_ai.text_encoder import rollout as rollout_mod

        worker = self._make_worker(seed=2)

        events: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        original = rollout_mod.TextRolloutWorker._score_step
        states_ref: list[list] = []

        # Track which slot ``state`` came from by identity-comparing against
        # the live ``states`` list inside ``play_episode``. We tap that list
        # by patching ``_init_states`` to capture the freshly built list.
        original_init = rollout_mod.TextRolloutWorker._init_states

        def init_spy(self):  # type: ignore[no-untyped-def]
            states = original_init(self)
            states_ref.append(states)
            return states

        def spy(  # type: ignore[no-untyped-def]
            self, game, snapshot, legal_options, state, perspective_player_idx
        ):
            # Recover player_idx by identity-matching ``state`` against the
            # captured ``states`` list — this avoids any reliance on snapshot
            # bookkeeping fields.
            current = states_ref[-1] if states_ref else []
            player_idx = -1
            for i, s in enumerate(current):
                if s is state:
                    player_idx = i
                    break
            h_in = state.h.detach().clone()
            result = original(self, game, snapshot, legal_options, state, perspective_player_idx)
            if result is not None:
                _, _, new_state = result
                events.append((player_idx, h_in, new_state.h.detach().clone()))
            return result

        try:
            setattr(rollout_mod.TextRolloutWorker, "_score_step", spy)
            setattr(rollout_mod.TextRolloutWorker, "_init_states", init_spy)
            cfg = {
                "deck_a": self.deck_a,
                "deck_b": self.deck_b,
                "seed": 3,
                "shuffle": True,
            }
            episode = worker.play_episode(cfg, max_turns=8)
        finally:
            setattr(rollout_mod.TextRolloutWorker, "_score_step", original)
            setattr(rollout_mod.TextRolloutWorker, "_init_states", original_init)

        self.assertGreaterEqual(len(episode.steps), 1)

        # For each player, walk consecutive events and assert: the (i+1)th
        # call's h_in is bit-equal to the ith call's h_out. If a player
        # advance had been incorrectly run on the *other* player's state,
        # the input would have drifted.
        for player_idx in (0, 1):
            seq = [(h_in, h_out) for p, h_in, h_out in events if p == player_idx]
            for prev, nxt in zip(seq, seq[1:]):
                _, prev_h_out = prev
                next_h_in, _ = nxt
                self.assertTrue(
                    torch.equal(prev_h_out, next_h_in),
                    f"player {player_idx}: LSTM state was mutated between calls",
                )


if __name__ == "__main__":
    unittest.main()
