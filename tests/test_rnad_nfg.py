"""Toy normal-form-game convergence test for R-NaD.

Runs the full R-NaD outer-loop algorithm on matching pennies (the DeepNash
paper's own worked example, §26-§40) and asserts both players' policies
converge to uniform within tolerance. This catches algorithmic bugs in the
reward transform + NeuRD + outer-iteration plumbing before they cost real
MTG training compute.

Matching pennies payoff matrix (from player 1's perspective):

              P2 heads    P2 tails
    P1 heads    +1           -1
    P1 tails    -1           +1

The unique Nash equilibrium is uniform: pi_1 = pi_2 = [0.5, 0.5].

Optimizer choice: this test uses SGD, not Adam. Adam's momentum (b1 > 0)
causes persistent cycling on zero-sum NFGs — the paper explicitly sets
b1=0.0 (§199), and even then relies on the target network plus v-trace
for stabilization, neither of which is in play for a degenerate single-step
NFG. SGD on the pure NeuRD gradient is the cleanest validation that the
reward-transform + outer-iteration math is correct.
"""

from __future__ import annotations

import unittest

import torch
from magic_ai.rnad import neurd_loss


def _policy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _run_rnad_nfg(
    *,
    payoff_p1: torch.Tensor,  # (2, 2); payoff_p2 = -payoff_p1 (zero-sum)
    eta: float,
    delta_m: int,
    num_outer: int,
    lr: float,
    neurd_beta: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final (pi_1, pi_2) after the R-NaD fixed-point iteration."""

    torch.manual_seed(seed)
    # Two-action per-player logits; start at random.
    logits_1 = torch.randn(2, requires_grad=True)
    logits_2 = torch.randn(2, requires_grad=True)
    opt = torch.optim.SGD([logits_1, logits_2], lr=lr)

    # Regularization policies; start uniform.
    uniform_logp = float(-torch.tensor(2.0).log())
    logp_reg_prev_1 = torch.full((2,), uniform_logp)
    logp_reg_prev_2 = torch.full((2,), uniform_logp)
    logp_reg_cur_1 = logp_reg_prev_1.clone()
    logp_reg_cur_2 = logp_reg_prev_2.clone()

    for _ in range(num_outer):
        for step in range(delta_m):
            alpha = min(1.0, 2.0 * step / max(1, delta_m))
            blended_reg_1 = alpha * logp_reg_cur_1 + (1.0 - alpha) * logp_reg_prev_1
            blended_reg_2 = alpha * logp_reg_cur_2 + (1.0 - alpha) * logp_reg_prev_2

            pi_1 = _policy_from_logits(logits_1)
            pi_2 = _policy_from_logits(logits_2)
            logp_1 = torch.log_softmax(logits_1, dim=-1)
            logp_2 = torch.log_softmax(logits_2, dim=-1)

            # Per-action expected Q (one row = one "timestep" per player).
            # Regularized Q for player 1: E_{a_2 ~ pi_2} r_1(a_1, a_2)
            # minus the entropy-regularization term on player 1's own policy.
            q_1_base = payoff_p1 @ pi_2.detach()  # shape (2,)
            q_2_base = -payoff_p1.t() @ pi_1.detach()  # shape (2,)
            q_1 = q_1_base - eta * (logp_1.detach() - blended_reg_1)
            q_2 = q_2_base - eta * (logp_2.detach() - blended_reg_2)

            # NeuRD on each player's logits, shaped (T=1, A=2).
            mask = torch.ones(1, 2, dtype=torch.bool)
            loss_1 = neurd_loss(
                logits=logits_1.unsqueeze(0),
                q_hat=q_1.unsqueeze(0),
                legal_mask=mask,
                beta=neurd_beta,
                clip=1e6,
            )
            loss_2 = neurd_loss(
                logits=logits_2.unsqueeze(0),
                q_hat=q_2.unsqueeze(0),
                legal_mask=mask,
                beta=neurd_beta,
                clip=1e6,
            )
            opt.zero_grad(set_to_none=True)
            (loss_1 + loss_2).backward()
            opt.step()

        # Outer iteration: reg_prev <- reg_cur ; reg_cur <- current pi
        logp_reg_prev_1 = logp_reg_cur_1.clone()
        logp_reg_prev_2 = logp_reg_cur_2.clone()
        logp_reg_cur_1 = torch.log_softmax(logits_1.detach(), dim=-1)
        logp_reg_cur_2 = torch.log_softmax(logits_2.detach(), dim=-1)

    return _policy_from_logits(logits_1.detach()), _policy_from_logits(logits_2.detach())


class MatchingPenniesConvergenceTests(unittest.TestCase):
    def test_converges_to_uniform(self) -> None:
        # Player-1 payoff matrix for matching pennies.
        payoff = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        pi_1, pi_2 = _run_rnad_nfg(
            payoff_p1=payoff,
            eta=0.5,
            delta_m=300,
            num_outer=4,
            lr=0.05,
            neurd_beta=100.0,
            seed=0,
        )
        # Nash equilibrium is uniform [0.5, 0.5].
        self.assertAlmostEqual(float(pi_1[0]), 0.5, delta=0.02)
        self.assertAlmostEqual(float(pi_1[1]), 0.5, delta=0.02)
        self.assertAlmostEqual(float(pi_2[0]), 0.5, delta=0.02)
        self.assertAlmostEqual(float(pi_2[1]), 0.5, delta=0.02)

    def test_initial_nonuniform_still_converges(self) -> None:
        # Different seed, biased initial logits: make sure we still land on
        # the uniform Nash of matching pennies.
        payoff = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        pi_1, pi_2 = _run_rnad_nfg(
            payoff_p1=payoff,
            eta=0.5,
            delta_m=300,
            num_outer=4,
            lr=0.05,
            neurd_beta=100.0,
            seed=42,
        )
        self.assertAlmostEqual(float(pi_1[0]), 0.5, delta=0.02)
        self.assertAlmostEqual(float(pi_2[0]), 0.5, delta=0.02)


if __name__ == "__main__":
    unittest.main()
