
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gated_deltanet2 import (  # noqa: E402
    DeltaPathBlock,
    DeltaPathMixerModel,
    GatedDeltaNet2,
    LightGatedDeltaNet2,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# the scan itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk_size", [8, 16, 64])
@pytest.mark.parametrize("T", [1, 7, 64, 120])
def test_chunk_matches_recurrent(chunk_size, T):
    B, H, N, P, G = 2, 4, 16, 8, 2
    q = torch.randn(B, T, G, N)
    k = torch.randn(B, T, G, N)
    v = torch.randn(B, T, H, P)
    g = -torch.rand(B, T, H) * 0.5
    beta = torch.rand(B, T, H)

    o_chunk, s_chunk = chunk_gated_delta_rule(
        q, k, v, g, beta, output_final_state=True, chunk_size=chunk_size)
    o_rec, s_rec = recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=True)

    assert torch.allclose(o_chunk, o_rec, atol=1e-4, rtol=1e-4)
    assert torch.allclose(s_chunk, s_rec, atol=1e-4, rtol=1e-4)


def test_initial_state_is_honoured():
    B, T, H, N, P = 2, 40, 4, 16, 8
    q, k = torch.randn(B, T, H, N), torch.randn(B, T, H, N)
    v = torch.randn(B, T, H, P)
    g, beta = -torch.rand(B, T, H) * 0.3, torch.rand(B, T, H)

    o_full, s_full = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True, chunk_size=8)

    cut = 24
    o1, s1 = chunk_gated_delta_rule(q[:, :cut], k[:, :cut], v[:, :cut], g[:, :cut], beta[:, :cut],
                                    output_final_state=True, chunk_size=8)
    o2, s2 = chunk_gated_delta_rule(q[:, cut:], k[:, cut:], v[:, cut:], g[:, cut:], beta[:, cut:],
                                    initial_state=s1, output_final_state=True, chunk_size=8)

    assert torch.allclose(torch.cat([o1, o2], dim=1), o_full, atol=1e-4, rtol=1e-4)
    assert torch.allclose(s2, s_full, atol=1e-4, rtol=1e-4)


def test_beta_zero_reduces_to_pure_decay():
    B, T, H, N, P = 1, 12, 2, 8, 4
    q, k = torch.randn(B, T, H, N), torch.randn(B, T, H, N)
    v = torch.randn(B, T, H, P)
    g = -torch.rand(B, T, H)
    o, s = chunk_gated_delta_rule(q, k, v, g, torch.zeros(B, T, H),
                                  output_final_state=True, chunk_size=4)
    assert o.abs().max() < 1e-6
    assert s.abs().max() < 1e-6


def test_delta_rule_erases_before_writing():
    B, T, H, N, P = 1, 2, 1, 4, 3
    key = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 1, N)
    k = key.repeat(1, T, 1, 1)
    q = key.repeat(1, T, 1, 1)
    v = torch.zeros(B, T, H, P)
    v[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
    v[0, 1, 0] = torch.tensor([0.0, 1.0, 0.0])
    g = torch.zeros(B, T, H)              # no decay
    beta = torch.ones(B, T, H)            # full write strength

    o, _ = recurrent_gated_delta_rule(q, k, v, g, beta)
    # After step 2, reading with the same key returns the *new* value only.
    assert torch.allclose(o[0, 1, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_gradients_flow():
    B, T, H, N, P = 2, 20, 2, 8, 4
    q = torch.randn(B, T, H, N, requires_grad=True)
    k = torch.randn(B, T, H, N, requires_grad=True)
    v = torch.randn(B, T, H, P, requires_grad=True)
    g = (-torch.rand(B, T, H) * 0.4).requires_grad_(True)
    beta = torch.rand(B, T, H).requires_grad_(True)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=8)
    o.sum().backward()
    for t in (q, k, v, g, beta):
        assert t.grad is not None and torch.isfinite(t.grad).all()


# --------------------------------------------------------------------------- #
# the modules
# --------------------------------------------------------------------------- #
def test_gated_deltanet2_is_a_mamba2_drop_in():
    d_model, B, L = 64, 3, 33
    mixer = GatedDeltaNet2(d_model=d_model, d_state=32, d_conv=4, expand=2,
                           headdim=32, ngroups=1, chunk_size=16, layer_idx=0)
    u = torch.randn(B, L, d_model)
    assert mixer(u).shape == (B, L, d_model)
    # flattened (batch*seqlen, d) form, as used under sequence parallelism
    assert mixer(u.reshape(B * L, d_model), seqlen=L).shape == (B * L, d_model)


def test_gated_deltanet2_step_matches_forward():
    d_model, B, L = 32, 2, 9
    mixer = GatedDeltaNet2(d_model=d_model, d_state=16, headdim=16, d_inner=32,
                           chunk_size=4).eval()
    u = torch.randn(B, L, d_model)
    with torch.no_grad():
        ref = mixer(u, mode="recurrent")
        conv_state, ssm_state = mixer.allocate_inference_cache(B, L)
        outs = [mixer.step(u[:, t: t + 1], conv_state, ssm_state)[0] for t in range(L)]
    assert torch.allclose(torch.cat(outs, dim=1), ref, atol=1e-4, rtol=1e-4)


def test_deltapath_block_shapes_and_gating():
    d_model, d_state, headdim, d_inner, B, L = 256, 32, 64, 256, 2, 40
    nheads = d_inner // headdim
    block = DeltaPathBlock(d_model=d_model, d_state=d_state, headdim=headdim,
                           d_inner=d_inner, aux_feature_size=3, add_bcdt_item=True,
                           chunk_size=16)
    assert block.nheads == nheads == 4
    u = torch.randn(B, L, d_model)
    out = block(u, B=torch.randn(B, L, d_state), C=torch.randn(B, L, d_state),
                dt=torch.randn(B, L, nheads), beta=torch.randn(B, L, nheads))
    assert out.shape == (B, L, d_model)
    # Road half first, GPS half second.
    assert out[..., : d_model // 2].shape == out[..., d_model // 2:].shape


def test_padding_mask_does_not_leak():
    d_model, B, L, valid = 64, 2, 20, 12
    enc = DeltaPathMixerModel(d_model=d_model, n_layer=2, d_intermediate=0,
                              aux_feature_size=3, d_state=16, headdim=32, d_inner=64,
                              chunk_size=8, verbose=False).eval()
    mask = torch.arange(L).unsqueeze(0).expand(B, L) >= valid
    x = torch.randn(B, L, d_model)
    aux = torch.randn(B, L, 3)

    with torch.no_grad():
        out_a = enc(x, aux, padding_mask=mask)
        x2, aux2 = x.clone(), aux.clone()
        x2[:, valid:] = torch.randn_like(x2[:, valid:]) * 10
        aux2[:, valid:] = torch.randn_like(aux2[:, valid:]) * 10
        out_b = enc(x2, aux2, padding_mask=mask)
    assert torch.allclose(out_a[:, :valid], out_b[:, :valid], atol=1e-4, rtol=1e-4)


def test_light_block_shapes():
    light = LightGatedDeltaNet2(d_model=5, expand=2)
    x = torch.randn(4, 30, 5)
    assert light(x).shape == x.shape


def test_mixer_variants_run():
    for block_type in ("DeltaPath", "GatedDeltaNet2"):
        enc = DeltaPathMixerModel(
            d_model=64, n_layer=2, d_intermediate=0,
            aux_feature_size=3 if block_type == "DeltaPath" else 0,
            d_state=16, headdim=32, d_inner=64, chunk_size=8, verbose=False)
        aux = torch.randn(2, 25, 3) if block_type == "DeltaPath" else None
        assert enc(torch.randn(2, 25, 64), aux).shape == (2, 25, 64)
