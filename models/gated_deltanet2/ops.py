
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    "chunk_gated_delta_rule",
    "recurrent_gated_delta_rule",
    "gated_delta_rule",
    "FLA_AVAILABLE",
]

# --------------------------------------------------------------------------- #
# Optional fused backend
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on the local environment
    from fla.ops.gated_delta_rule import (  # type: ignore
        chunk_gated_delta_rule as _fla_chunk,
        fused_recurrent_gated_delta_rule as _fla_recurrent,
    )

    FLA_AVAILABLE = True
except Exception:  # pragma: no cover
    _fla_chunk, _fla_recurrent = None, None
    FLA_AVAILABLE = False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True).clamp_min(eps))


def _expand_groups(x: torch.Tensor, nheads: int) -> torch.Tensor:
    b, t, g, n = x.shape
    if g == nheads:
        return x
    assert nheads % g == 0, f"nheads={nheads} must be a multiple of ngroups={g}"
    return x.unsqueeze(3).expand(b, t, g, nheads // g, n).reshape(b, t, nheads, n)


def _prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    use_qk_l2norm: bool,
) -> Tuple[torch.Tensor, ...]:
    nheads = v.shape[2]
    q = _expand_groups(q, nheads)
    k = _expand_groups(k, nheads)
    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    return q.float(), k.float(), v.float(), g.float(), beta.float()


# --------------------------------------------------------------------------- #
# chunk-wise parallel form (training)
# --------------------------------------------------------------------------- #
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    use_fla: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    q, k, v, g, beta = _prepare(q, k, v, g, beta, use_qk_l2norm)
    B, T, H, P = v.shape
    N = k.shape[-1]

    if use_fla and FLA_AVAILABLE and q.is_cuda:  # pragma: no cover
        o, final_state = _fla_chunk(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
        )
        return o, final_state

    C = min(chunk_size, max(T, 1))
    pad = (C - T % C) % C
    if pad:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))          # log-gate 0 -> alpha = 1
        beta = F.pad(beta, (0, 0, 0, pad))    # beta 0 -> no write
    Tp = T + pad
    NC = Tp // C

    # (B, T, H, X) -> (B*H, NC, C, X): heads become independent batch elements.
    def _fold(x):
        x = rearrange(x, "b (n c) h ... -> (b h) n c ...", n=NC, c=C)
        return x

    q, k, v = _fold(q), _fold(k), _fold(v)
    g, beta = _fold(g), _fold(beta)                      # (BH, NC, C)

    BH = B * H
    # cumulative log gate inside each chunk, inclusive: g_cum[i] = sum_{j<=i} g_j
    g_cum = g.cumsum(dim=-1)                             # (BH, NC, C)
    a = g_cum.exp().unsqueeze(-1)                        # (BH, NC, C, 1)

    # pairwise decay exp(g_i - g_j) for j <= i, zero above the diagonal.
    diff = g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)     # (BH, NC, C, C)
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    decay = torch.where(causal, diff, torch.full_like(diff, float("-inf"))).exp()
    decay = torch.nan_to_num(decay, nan=0.0, posinf=0.0, neginf=0.0)

    kk = k @ k.transpose(-1, -2)                         # (BH, NC, C, C)
    qk = q @ k.transpose(-1, -2)

    beta_c = beta.unsqueeze(-1)                          # (BH, NC, C, 1)
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    # M[i, j] = beta_i * (k_i . k_j) * exp(g_i - g_j),  j < i
    M = (beta_c * kk * decay) * strict
    eye = torch.eye(C, device=q.device, dtype=q.dtype)
    A_tri = eye + M                                      # unit lower triangular

    # attention used for the output: inclusive of the diagonal.
    A_out = qk * decay

    state = initial_state
    if state is None:
        state = q.new_zeros(BH, N, P)
    else:
        state = rearrange(state.float(), "b h n p -> (b h) n p").contiguous()

    outputs = []
    for c in range(NC):
        k_c, q_c, v_c = k[:, c], q[:, c], v[:, c]        # (BH, C, N/P)
        a_c, beta_i = a[:, c], beta_c[:, c]              # (BH, C, 1)

        ks0 = k_c @ state                                # (BH, C, P)
        qs0 = q_c @ state                                # (BH, C, P)

        rhs = beta_i * (v_c - a_c * ks0)
        u_bar = torch.linalg.solve_triangular(
            A_tri[:, c], rhs, upper=False, unitriangular=True
        )                                                # (BH, C, P)

        o_c = a_c * qs0 + A_out[:, c] @ u_bar
        outputs.append(o_c)

        # S_C = a_C * S_0 + sum_j exp(g_C - g_j) k_j u_bar_j^T
        last = g_cum[:, c, -1].unsqueeze(-1)             # (BH, 1)
        w = (last - g_cum[:, c]).exp().unsqueeze(-1)     # (BH, C, 1)
        state = last.exp().unsqueeze(-1) * state + (k_c * w).transpose(-1, -2) @ u_bar

    o = torch.stack(outputs, dim=1)                      # (BH, NC, C, P)
    o = rearrange(o, "(b h) n c p -> b (n c) h p", b=B, h=H)[:, :T]
    final_state = rearrange(state, "(b h) n p -> b h n p", b=B, h=H) if output_final_state else None
    return o, final_state


# --------------------------------------------------------------------------- #
# recurrent form (inference)
# --------------------------------------------------------------------------- #
def recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
    use_fla: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    q, k, v, g, beta = _prepare(q, k, v, g, beta, use_qk_l2norm)
    B, T, H, P = v.shape
    N = k.shape[-1]

    if use_fla and FLA_AVAILABLE and q.is_cuda:  # pragma: no cover
        return _fla_recurrent(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
        )

    if initial_state is None:
        state = q.new_zeros(B, H, N, P)
    else:
        state = initial_state.float().clone()

    outs = []
    for t in range(T):
        k_t = k[:, t]                                     # (B, H, N)
        q_t = q[:, t]
        v_t = v[:, t]                                     # (B, H, P)
        alpha_t = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)
        beta_t = beta[:, t].unsqueeze(-1)                 # (B, H, 1)

        # predicted value currently stored under key k_t
        pred = torch.einsum("bhn,bhnp->bhp", k_t, state)  # (B, H, P)
        delta = beta_t * (v_t - alpha_t.squeeze(-1) * pred)
        state = alpha_t * state + torch.einsum("bhn,bhp->bhnp", k_t, delta)
        outs.append(torch.einsum("bhn,bhnp->bhp", q_t, state))

    o = torch.stack(outs, dim=1)                          # (B, T, H, P)
    return o, (state if output_final_state else None)


def gated_delta_rule(*args, mode: str = "chunk", **kwargs):
    if mode == "chunk":
        return chunk_gated_delta_rule(*args, **kwargs)
    if mode in ("recurrent", "step"):
        return recurrent_gated_delta_rule(*args, **kwargs)
    raise ValueError(f"unknown mode '{mode}', expected 'chunk' or 'recurrent'")
