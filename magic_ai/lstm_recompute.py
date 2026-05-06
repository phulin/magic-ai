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
    proj_per_episode: Sequence[Tensor],
    *,
    chunk_size: int = 200,
    compiled_lstm: Callable[..., Any] | None = None,
) -> list[Tensor]:
    """Chunked-BPTT fused cuDNN recompute returning per-step top-layer h_out.

    Takes a list of per-episode projected feature tensors of shape
    ``(T_i, hidden)`` -- already in the LSTM's input space, no padding.
    Returns a list of ``(T_i, hidden)`` top-layer hidden outputs.

    Implements the DeepNash R-NaD recipe (arxiv 2206.15378 §"Full games
    learning"): chop along the time axis into chunks of ``chunk_size``
    steps, and process chunk-by-chunk. Within each chunk the LSTM runs as
    one fused cuDNN call (full BPTT inside the chunk); the state crossing
    each chunk boundary is detached so gradients do not flow across
    boundaries.

    Each chunk is fed to ``nn.LSTM`` as a :class:`PackedSequence` built
    from the active episodes' chunk slices, so cuDNN never does work on
    padding and we never materialize a ``(T_max, N, hidden)`` padded
    tensor. Pass-(a)/pass-(b) are merged here -- if the caller is in
    ``torch.no_grad`` mode the whole thing is forward-only; otherwise
    gradients are captured within each chunk.

    Default ``chunk_size=200`` matches the production
    ``--max-steps-per-game=200`` cap, so by default the whole trajectory
    is one chunk (full BPTT through the fused call). Smaller values cap
    activation memory at the cost of truncating gradient flow.

    ``compiled_lstm`` (optional): a ``torch.compile``'d wrapper around the
    LSTM forward. Falls back to the un-compiled module when ``None``.
    """

    if not proj_per_episode:
        raise ValueError("proj_per_episode must be non-empty")
    if any(t.dim() != 2 for t in proj_per_episode):
        raise ValueError("each per-episode projection must be (T_i, hidden)")
    if any(t.shape[0] == 0 for t in proj_per_episode):
        raise ValueError("each episode must have at least one step")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1 (got {chunk_size})")
    n = len(proj_per_episode)
    hidden = int(lstm.hidden_size)
    if any(int(t.shape[1]) != hidden for t in proj_per_episode):
        raise ValueError(f"each per-episode projection must have hidden dim {hidden}")

    device = proj_per_episode[0].device
    dtype = proj_per_episode[0].dtype
    lengths = [int(t.shape[0]) for t in proj_per_episode]
    t_max = max(lengths)

    runner = compiled_lstm if compiled_lstm is not None else lstm
    h_state = torch.zeros(lstm.num_layers, n, hidden, dtype=dtype, device=device)
    c_state = torch.zeros_like(h_state)
    chunks_per_ep: list[list[Tensor]] = [[] for _ in range(n)]

    for chunk_start in range(0, t_max, chunk_size):
        chunk_stop = chunk_start + chunk_size
        active_idx = [i for i, t_i in enumerate(lengths) if t_i > chunk_start]
        if not active_idx:
            break
        chunk_seqs = [
            proj_per_episode[i][chunk_start : min(chunk_stop, lengths[i])] for i in active_idx
        ]
        active_idx_t = torch.tensor(active_idx, dtype=torch.long, device=device)
        h_in = h_state.index_select(1, active_idx_t).contiguous()
        c_in = c_state.index_select(1, active_idx_t).contiguous()
        packed = nn.utils.rnn.pack_sequence(chunk_seqs, enforce_sorted=False)
        out_packed, (h_out, c_out) = runner(packed, (h_in, c_in))
        # Detach state crossing the chunk boundary so the next chunk's
        # backward stops here. Within a chunk, BPTT is full.
        h_state.index_copy_(1, active_idx_t, h_out.detach().to(dtype=h_state.dtype))
        c_state.index_copy_(1, active_idx_t, c_out.detach().to(dtype=c_state.dtype))
        padded, chunk_lengths = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        for i_local, ep_idx in enumerate(active_idx):
            t_chunk = int(chunk_lengths[i_local])
            chunks_per_ep[ep_idx].append(padded[i_local, :t_chunk])

    return [parts[0] if len(parts) == 1 else torch.cat(parts, dim=0) for parts in chunks_per_ep]


def lstm_recompute_per_step_h_out_per_player(
    lstm: nn.LSTM,
    projected: Tensor,
    perspective: Tensor,
    ep_lengths: Sequence[int],
    *,
    chunk_size: int = 200,
    compiled_lstm: Callable[..., Any] | None = None,
) -> Tensor:
    """Per-player split LSTM recompute returning per-step top-layer h_out.

    Mirrors rollout behaviour: each player's LSTM state advances only on
    *their own* turns and resets to zero at the start of every episode.

    Each ``(episode, player)`` pair is treated as an independent virtual
    sequence.  Steps are regrouped by virtual sequence ID
    ``ep_id * 2 + player`` via a stable argsort, the fused cuDNN LSTM runs
    once over all virtual sequences via :func:`lstm_recompute_per_step_h_out`,
    and outputs are scattered back with ``h_out[sort_idx] = all_h_cat``.

    Args:
        lstm: The LSTM module.
        projected: ``[T_total, hidden]`` flat projected features (all
            episodes concatenated in order).
        perspective: ``[T_total]`` long tensor with values 0 or 1; the
            player whose turn it is at each step.
        ep_lengths: Length of each episode; ``sum(ep_lengths)`` must equal
            ``T_total``.
        chunk_size: BPTT chunk size forwarded to
            :func:`lstm_recompute_per_step_h_out`.
        compiled_lstm: Optional ``torch.compile``'d LSTM callable.

    Returns:
        ``[T_total, hidden]`` h_out at every step.
    """
    if projected.dim() != 2:
        raise ValueError("projected must be (T_total, hidden)")
    t_total = int(projected.shape[0])
    hidden = int(lstm.hidden_size)
    if int(projected.shape[1]) != hidden:
        raise ValueError(f"projected dim 1 must equal lstm.hidden_size ({hidden})")
    if perspective.dim() != 1 or int(perspective.shape[0]) != t_total:
        raise ValueError("perspective must be (T_total,) matching projected")
    if sum(ep_lengths) != t_total:
        raise ValueError(f"sum(ep_lengths)={sum(ep_lengths)} must equal T_total={t_total}")

    n_eps = len(ep_lengths)
    device = projected.device

    # Episode ID for every flat step: [T_total]
    ep_ids = torch.repeat_interleave(
        torch.arange(n_eps, device=device),
        torch.tensor(ep_lengths, dtype=torch.long, device=device),
    )

    # Virtual sequence ID: ep_id * 2 + player, values in [0, 2*n_eps)
    virt_ids = ep_ids * 2 + perspective  # [T_total]

    # Stable sort by virtual ID; within each virtual seq game order is preserved
    sort_idx = torch.argsort(virt_ids, stable=True)  # [T_total]
    feats_sorted = projected[sort_idx]  # [T_total, hidden]

    # Per-virtual-sequence lengths; sum == T_total
    virt_lengths = torch.bincount(virt_ids, minlength=2 * n_eps)  # [2*n_eps]

    # Build list of non-empty per-virtual-sequence tensors for lstm_recompute
    seqs = [s for s in torch.split(feats_sorted, virt_lengths.tolist()) if s.shape[0] > 0]

    all_h_outs = lstm_recompute_per_step_h_out(
        lstm,
        seqs,
        chunk_size=chunk_size,
        compiled_lstm=compiled_lstm,
    )

    # Unsort: all_h_cat[j] is the output for sort_idx[j] in the original flat order.
    # Gather via the inverse permutation so the result stays in the autograd graph.
    all_h_cat = torch.cat(all_h_outs, dim=0)
    return all_h_cat[torch.argsort(sort_idx)]


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
