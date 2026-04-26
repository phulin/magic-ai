"""LSTM per-step input-state recompute strategies (issue 2).

The R-NaD trajectory loss needs, for each replayed step ``t`` of an episode,
the LSTM input state ``(h, c)`` that the *current* policy parameters would
produce after replaying steps ``0..t-1`` from a zero state. Four strategies
are implemented here so we can compare correctness and throughput:

- ``"legacy"``: per-episode, batch=1, sequential ``T`` ``nn.LSTM(seq=1)``
  calls. Reference; one cuDNN launch per (episode, step).
- ``"pad"``: pad to ``T_max`` and run the LSTM cell across the
  ``(N_episodes, seq=1)`` batch for ``T_max`` steps. Cuts launches by
  ``N_episodes`` at the cost of doing useless work on padded positions.
- ``"gather"``: same shape as ``"pad"`` but freezes finished episodes'
  state via a per-step mask, so finished episodes don't contribute
  meaningful compute (output is sliced off anyway, so this is mostly a
  tidiness/numerics safety net rather than a throughput win over pad).
- ``"packed"``: per-sequence pointer that always advances (no GPU->CPU
  sync to compute an active prefix). At every step we run the full ``N``
  batch through the cell, gathering each sequence's input from its own
  pointer (clamped to its last valid position once finished -- the cell
  call still runs but its output is gated off). Writeback to the running
  state and to the history tensor is masked by ``t < length[i]``.

All four strategies return a list of ``(h_in, c_in)`` tensors of shape
``(num_layers, T_i, hidden)`` per episode -- the input states *before*
each step, with ``h_in[:, 0, :] = c_in[:, 0, :] = 0``.

The strategies operate on already-projected features ``(T_max, N, hidden)``
so they can be unit-tested and benchmarked without a rollout buffer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

LstmRecomputeStrategy = Literal["legacy", "pad", "gather", "packed"]
STRATEGIES: tuple[LstmRecomputeStrategy, ...] = ("legacy", "pad", "gather", "packed")


def _legacy(
    lstm: nn.LSTM,
    projected: Tensor,
    lengths: Sequence[int],
) -> list[tuple[Tensor, Tensor]]:
    num_layers = int(lstm.num_layers)
    hidden = int(lstm.hidden_size)
    device = projected.device
    dtype = projected.dtype
    out: list[tuple[Tensor, Tensor]] = []
    for i, t_i in enumerate(lengths):
        h = torch.zeros(num_layers, 1, hidden, dtype=dtype, device=device)
        c = torch.zeros_like(h)
        h_list: list[Tensor] = []
        c_list: list[Tensor] = []
        for t in range(t_i):
            h_list.append(h)
            c_list.append(c)
            step = projected[t : t + 1, i : i + 1, :].contiguous()
            _o, (h, c) = lstm(step, (h.contiguous(), c.contiguous()))
        h_stack = torch.cat(h_list, dim=1).contiguous()
        c_stack = torch.cat(c_list, dim=1).contiguous()
        out.append((h_stack, c_stack))
    return out


def _step_loop(
    lstm: nn.LSTM,
    projected: Tensor,
    lengths: Sequence[int],
    *,
    mask_finished: bool,
) -> list[tuple[Tensor, Tensor]]:
    num_layers = int(lstm.num_layers)
    hidden = int(lstm.hidden_size)
    device = projected.device
    dtype = projected.dtype
    t_max, n, _ = projected.shape
    h = torch.zeros(num_layers, n, hidden, dtype=dtype, device=device)
    c = torch.zeros_like(h)
    h_list: list[Tensor] = []
    c_list: list[Tensor] = []
    if mask_finished:
        lengths_t = torch.tensor(list(lengths), dtype=torch.long, device=device)
        active_per_step = torch.arange(t_max, device=device).unsqueeze(1) < lengths_t.unsqueeze(
            0
        )  # (T_max, N)
    for t in range(t_max):
        h_list.append(h)
        c_list.append(c)
        step = projected[t].unsqueeze(1)  # (N, 1, hidden) -- already contiguous
        _o, (h_new, c_new) = lstm(step, (h, c))
        if mask_finished:
            active = active_per_step[t].view(1, n, 1)
            h = torch.where(active, h_new, h)
            c = torch.where(active, c_new, c)
        else:
            h, c = h_new, c_new
    h_history = torch.stack(h_list, dim=0)  # (T_max, num_layers, N, hidden)
    c_history = torch.stack(c_list, dim=0)
    return _slice_history(h_history, c_history, lengths)


def _packed(
    lstm: nn.LSTM,
    projected: Tensor,
    lengths: Sequence[int],
) -> list[tuple[Tensor, Tensor]]:
    """Sync-free packed: per-sequence pointer, writeback-gated.

    At step ``t`` every sequence advances its own pointer ``p_i = min(t,
    t_i - 1)`` and reads its own input from ``projected[p_i, i]``. The cell
    call runs over the full ``N`` batch -- no ``.item()`` to compute an
    active prefix, no GPU->CPU sync. State + history writeback for finished
    sequences (``t >= t_i``) is gated off via a mask, so the "useless
    computation" they contribute is discarded. Both the per-step gather
    indices and the active mask are precomputed once outside the loop, so
    the per-step loop body is the same shape as the gather strategy.
    """
    num_layers = int(lstm.num_layers)
    hidden = int(lstm.hidden_size)
    device = projected.device
    dtype = projected.dtype
    t_max, n, _ = projected.shape
    lengths_t = torch.tensor(list(lengths), dtype=torch.long, device=device)
    arange_t = torch.arange(t_max, device=device).unsqueeze(1)  # (T_max, 1)
    ptrs = torch.minimum(arange_t, (lengths_t - 1).unsqueeze(0))  # (T_max, N)
    projected_packed = torch.gather(projected, 0, ptrs.unsqueeze(-1).expand(t_max, n, hidden))
    active_per_step = arange_t < lengths_t.unsqueeze(0)  # (T_max, N)
    h = torch.zeros(num_layers, n, hidden, dtype=dtype, device=device)
    c = torch.zeros_like(h)
    h_list: list[Tensor] = []
    c_list: list[Tensor] = []
    for t in range(t_max):
        h_list.append(h)
        c_list.append(c)
        step = projected_packed[t].unsqueeze(1)
        _o, (h_new, c_new) = lstm(step, (h, c))
        active = active_per_step[t].view(1, n, 1)
        h = torch.where(active, h_new, h)
        c = torch.where(active, c_new, c)
    h_history = torch.stack(h_list, dim=0)
    c_history = torch.stack(c_list, dim=0)
    return _slice_history(h_history, c_history, lengths)


def _slice_history(
    h_history: Tensor,
    c_history: Tensor,
    lengths: Sequence[int],
) -> list[tuple[Tensor, Tensor]]:
    out: list[tuple[Tensor, Tensor]] = []
    for i, t_i in enumerate(lengths):
        h_i = h_history[:t_i, :, i, :].permute(1, 0, 2).contiguous()
        c_i = c_history[:t_i, :, i, :].permute(1, 0, 2).contiguous()
        out.append((h_i, c_i))
    return out


def lstm_recompute_per_step_h_out(
    lstm: nn.LSTM,
    projected: Tensor,
    lengths: Sequence[int],
    *,
    chunk_size: int = 200,
    compiled_lstm: Callable[..., Any] | None = None,
) -> list[Tensor]:
    """Chunked-BPTT fused cuDNN recompute returning per-step top-layer h_out.

    Implements the DeepNash R-NaD recipe (arxiv 2206.15378 §"Full games
    learning"): pad to ``T_max``, chop along the time axis into chunks of
    ``chunk_size`` steps, and process chunk-by-chunk. Within each chunk the
    LSTM runs as one fused cuDNN call (full BPTT inside the chunk); the
    state crossing each chunk boundary is detached so gradients do not
    flow across boundaries. Pass-(a)/pass-(b) are merged here -- if the
    caller is in ``torch.no_grad`` mode the whole thing is forward-only;
    otherwise gradients are captured within each chunk.

    Default ``chunk_size=200`` matches the production
    ``--max-steps-per-game=200`` cap, so by default the whole trajectory
    is one chunk (full BPTT through the fused call). Smaller values cap
    activation memory at the cost of truncating gradient flow.

    ``projected``: ``(T_max, N, hidden)``. Returns a list of ``(T_i, hidden)``
    tensors -- the top-layer hidden output at each step.

    ``compiled_lstm`` (optional): a ``torch.compile``'d wrapper around the
    LSTM forward. Falls back to the un-compiled module when ``None``.
    """

    if not lengths:
        raise ValueError("lengths must be non-empty")
    if any(t <= 0 for t in lengths):
        raise ValueError("each length must be >= 1")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1 (got {chunk_size})")
    t_max, n, h = projected.shape
    if n != len(lengths):
        raise ValueError(f"projected dim 1 ({n}) must match len(lengths) ({len(lengths)})")
    if t_max < max(lengths):
        raise ValueError(f"projected dim 0 ({t_max}) must be >= max(lengths) ({max(lengths)})")
    if h != lstm.hidden_size:
        raise ValueError(f"projected dim 2 ({h}) must equal lstm.hidden_size ({lstm.hidden_size})")

    # nn.LSTM with batch_first=True wants (N, T, hidden). cuDNN fuses the
    # T loop internally -- one launch per chunk instead of T_chunk launches.
    inputs = projected.transpose(0, 1).contiguous()
    h_state = torch.zeros(
        lstm.num_layers, n, lstm.hidden_size, dtype=projected.dtype, device=projected.device
    )
    c_state = torch.zeros_like(h_state)
    runner = compiled_lstm if compiled_lstm is not None else lstm
    chunk_outputs: list[Tensor] = []
    for start in range(0, t_max, chunk_size):
        stop = min(start + chunk_size, t_max)
        chunk_in = inputs[:, start:stop, :].contiguous()
        chunk_out, (h_state, c_state) = runner(chunk_in, (h_state, c_state))
        chunk_outputs.append(chunk_out)
        # Detach state crossing the chunk boundary so the next chunk's
        # backward stops here. Within a chunk, BPTT is full.
        h_state = h_state.detach()
        c_state = c_state.detach()
    outputs = chunk_outputs[0] if len(chunk_outputs) == 1 else torch.cat(chunk_outputs, dim=1)
    out: list[Tensor] = []
    for i, t_i in enumerate(lengths):
        out.append(outputs[i, :t_i, :].contiguous())
    return out


@torch.no_grad()
def lstm_recompute_per_step_states(
    lstm: nn.LSTM,
    projected: Tensor,
    lengths: Sequence[int],
    *,
    strategy: LstmRecomputeStrategy = "legacy",
) -> list[tuple[Tensor, Tensor]]:
    """Compute per-step LSTM input states under ``strategy``.

    ``projected``: ``(T_max, N, hidden)`` of feature-projected inputs;
    positions beyond ``lengths[i]`` are padding and not read for that
    episode. Returns a list of ``(h_in, c_in)`` of shape
    ``(num_layers, T_i, hidden)``.
    """

    if not lengths:
        raise ValueError("lengths must be non-empty")
    if any(t <= 0 for t in lengths):
        raise ValueError("each length must be >= 1")
    t_max, n, h = projected.shape
    if n != len(lengths):
        raise ValueError(f"projected dim 1 ({n}) must match len(lengths) ({len(lengths)})")
    if t_max < max(lengths):
        raise ValueError(f"projected dim 0 ({t_max}) must be >= max(lengths) ({max(lengths)})")
    if h != lstm.hidden_size:
        raise ValueError(f"projected dim 2 ({h}) must equal lstm.hidden_size ({lstm.hidden_size})")
    if strategy == "legacy":
        return _legacy(lstm, projected, lengths)
    if strategy == "pad":
        return _step_loop(lstm, projected, lengths, mask_finished=False)
    if strategy == "gather":
        return _step_loop(lstm, projected, lengths, mask_finished=True)
    if strategy == "packed":
        return _packed(lstm, projected, lengths)
    raise ValueError(f"unknown strategy: {strategy!r}")
