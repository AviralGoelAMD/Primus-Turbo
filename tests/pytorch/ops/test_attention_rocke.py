###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""rocKE dense-attention forward backend (proof of concept): gate, routing, accuracy, gradients.

The accuracy tests alone could pass while another kernel ran, so routing is tested separately:
``_rocke_launch`` is spied and the FlyDSL / AITER forward impls are replaced by tripwires.
GPU tests skip unless the device is gfx950 and rocKE is importable (``import rocke``, or
``PRIMUS_TURBO_ROCKE_PATH`` pointing at a rocKE source tree).
"""

import math

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.attention import attention_impl, attention_rocke_impl
from primus_turbo.pytorch.ops import flash_attn_func
from primus_turbo.pytorch.ops.attention import flash_attn_interface
from tests.pytorch.test_utils import compute_snr

_PREC = PrecisionType.BF16_FP16_FP32
_HQ, _HKV, _D = 64, 8, 64

gfx950_rocke = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950() and attention_rocke_impl.rocke_available()),
    reason="rocKE attention needs a gfx950 device and an importable rocKE",
)


@pytest.fixture(autouse=True)
def _reset_backend():
    GlobalBackendManager.set_attn_backend(None, _PREC)
    yield
    GlobalBackendManager.set_attn_backend(None, _PREC)


def _pin(backend):
    GlobalBackendManager.set_attn_backend(backend, _PREC)


def _inputs(B, S, seed=0, dtype=torch.bfloat16, requires_grad=False):
    """sbhd storage handed to the op as its [b, s, h, d] view (what Primus does)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(S, B, _HQ, _D, dtype=dtype, device="cuda", generator=g)
    k = torch.randn(S, B, _HKV, _D, dtype=dtype, device="cuda", generator=g)
    v = torch.randn(S, B, _HKV, _D, dtype=dtype, device="cuda", generator=g)
    sink = torch.randn(_HQ, dtype=torch.float32, device="cuda", generator=g)
    dout = torch.randn(S, B, _HQ, _D, dtype=dtype, device="cuda", generator=g)
    if requires_grad:
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        sink.requires_grad_()
    return q, k, v, sink, dout


def _call(q, k, v, sink, window_left, return_lse=True):
    window = (window_left, 0) if window_left >= 0 else (-1, -1)
    return flash_attn_func(
        q.permute(1, 0, 2, 3),
        k.permute(1, 0, 2, 3),
        v.permute(1, 0, 2, 3),
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        window_size=window,
        return_lse=return_lse,
        sink=sink,
    )


def _reference(q, k, v, sink, window_left):
    """fp32 attention with the sink as an extra softmax-denominator column and the Primus window
    (W keeps W+1 keys). Returns out [B, S, Hq, D] and natural-log LSE [B, Hq, S]."""
    S, B = q.shape[0], q.shape[1]
    group = _HQ // _HKV
    qh = q.detach().float().permute(1, 2, 0, 3)
    kh = k.detach().float().permute(1, 2, 0, 3).repeat_interleave(group, 1)
    vh = v.detach().float().permute(1, 2, 0, 3).repeat_interleave(group, 1)
    s = qh @ kh.transpose(-1, -2) / math.sqrt(_D)
    i = torch.arange(S, device=q.device)[:, None]
    j = torch.arange(S, device=q.device)[None, :]
    keep = j <= i
    if window_left >= 0:
        keep &= j >= i - window_left
    s = s.masked_fill(~keep, float("-inf"))
    s = torch.cat([s, sink.detach().float().view(1, _HQ, 1, 1).expand(B, _HQ, S, 1)], dim=-1)
    lse = torch.logsumexp(s, dim=-1)
    p = torch.softmax(s, dim=-1)[..., :-1]
    return (p @ vh).permute(0, 2, 1, 3), lse


# --------------------------------------------------------------------------------------
# registry / gate (no rocKE or GPU needed)
# --------------------------------------------------------------------------------------
def test_rocke_is_registered_pin_only():
    entry = attention_impl._DENSE_FWD_BACKENDS[BackendType.ROCKE]
    assert entry.autotune is False
    # Last, behind AITER (superset gate): never a default and never reached as the fallback.
    assert list(attention_impl._DENSE_FWD_BACKENDS)[-1] is BackendType.ROCKE
    assert BackendType["ROCKE"] is BackendType.ROCKE


def _gate_args(**over):
    kw = dict(
        q=torch.empty(2, 256, _HQ, _D, dtype=torch.bfloat16),
        k=torch.empty(2, 256, _HKV, _D, dtype=torch.bfloat16),
        v=torch.empty(2, 256, _HKV, _D, dtype=torch.bfloat16),
        causal=True,
        window_size=(-1, -1),
        sink=torch.empty(_HQ, dtype=torch.float32),
        qkv_format="sbhd",
    )
    kw.update(over)
    return kw


@pytest.mark.parametrize(
    "over",
    [
        dict(dropout_p=0.1),
        dict(bias=torch.empty(1)),
        dict(alibi_slopes=torch.empty(1)),
        dict(return_softmax=True),
        dict(causal=False),
        dict(qkv_format="bshd"),
        dict(qkv_format="bhsd"),
        dict(sink=torch.empty(_HQ, dtype=torch.bfloat16)),
        dict(sink=torch.empty(_HQ + 1, dtype=torch.float32)),
        dict(window_size=(32, 0)),
        dict(window_size=(128, 1)),
        dict(softmax_scale=0.5),
        dict(q=torch.empty(2, 256, _HQ, _D, dtype=torch.float16)),
        dict(q=torch.empty(2, 256, _HQ, 128, dtype=torch.bfloat16)),
        dict(k=None),
    ],
    ids=lambda o: ",".join(o),
)
def test_rocke_can_handle_refuses(over, monkeypatch):
    monkeypatch.setattr(attention_impl, "get_device_compute_capability", lambda: (9, 5))
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: False)
    base = _gate_args()
    assert attention_impl.rocke_dense_unsupported_reason(**{**base, **over}) is not None
    assert attention_impl.DenseAttnFwdRockeBackend.can_handle(**{**base, **over}) is False


def test_rocke_can_handle_refuses_other_arch(monkeypatch):
    monkeypatch.setattr(attention_impl, "get_device_compute_capability", lambda: (9, 4))
    assert attention_impl.DenseAttnFwdRockeBackend.can_handle(**_gate_args()) is False


def test_rocke_missing_is_a_clear_error(monkeypatch):
    def _no_rocke(name, *a, **k):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(attention_rocke_impl, "_ROCKE", None)
    monkeypatch.setattr(attention_rocke_impl, "_import_module", _no_rocke)
    with pytest.raises(attention_rocke_impl.RockeUnavailableError, match="PRIMUS_TURBO_ROCKE_PATH"):
        attention_rocke_impl._load_rocke()
    assert attention_rocke_impl.rocke_available() is False
    monkeypatch.setattr(attention_impl, "get_device_compute_capability", lambda: (9, 5))
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: False)
    assert attention_impl.DenseAttnFwdRockeBackend.can_handle(**_gate_args()) is False


# --------------------------------------------------------------------------------------
# GPU: refusals through the real resolver, window semantics
# --------------------------------------------------------------------------------------
@gfx950_rocke
@pytest.mark.parametrize(
    "window_size,S",
    [((128, 1), 1024), ((32, 0), 1024), ((128, 0), 1000)],
    ids=["right_gt_0", "left_below_flydsl_bwd_min", "S_not_tile_multiple"],
)
def test_pinned_rocke_still_refuses_what_it_cannot_do(window_size, S):
    q, k, v, sink, _ = _inputs(1, S)
    _pin(BackendType.ROCKE)
    with pytest.raises(ValueError, match="cannot handle"):
        flash_attn_func(
            q.permute(1, 0, 2, 3), k.permute(1, 0, 2, 3), v.permute(1, 0, 2, 3),
            causal=True, window_size=window_size, sink=sink, return_lse=True,
        )  # fmt: skip


@gfx950_rocke
def test_unpinned_default_stays_flydsl_on_both_layer_types(monkeypatch):
    q, k, v, sink, _ = _inputs(1, 1024)
    seen = []
    orig = flash_attn_interface.FlashAttnFunc.apply
    monkeypatch.setattr(
        flash_attn_interface.FlashAttnFunc, "apply", staticmethod(lambda *a: (seen.append(a[-1]), orig(*a))[1])
    )
    _call(q, k, v, sink, 128)
    _call(q, k, v, sink, -1)
    assert seen == [BackendType.FLYDSL, BackendType.FLYDSL]


@gfx950_rocke
def test_pinned_rocke_serves_both_gptoss_layer_types(monkeypatch):
    """Primus GPT-OSS issues (128, 0) on even layers and full causal on odd ones: one pinned
    backend serves both, with rocKE's 129-key window on the first."""
    q, k, v, sink, _ = _inputs(1, 1024)
    calls = []
    real = attention_rocke_impl._rocke_launch
    monkeypatch.setattr(
        attention_rocke_impl, "_rocke_launch", lambda *a, **kw: (calls.append(a[0]), real(*a, **kw))[1]
    )
    for mod, name in (
        (flash_attn_interface, "flash_attn_sbhd_flydsl_forward_impl"),
        (flash_attn_interface, "attention_aiter_forward_impl"),
        (attention_impl, "flash_attn_sbhd_flydsl_forward_impl"),
    ):
        monkeypatch.setattr(mod, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} ran under pinned ROCKE"))
    _pin(BackendType.ROCKE)
    for _ in range(2):  # two "layers" of each kind, in Primus's alternating order
        _call(q, k, v, sink, 128)
        _call(q, k, v, sink, -1)
    torch.cuda.synchronize()
    assert [s.sliding_window for s in calls] == [129, 0, 129, 0]


@gfx950_rocke
def test_direct_apply_with_ineligible_inputs_raises_never_aiter(monkeypatch):
    """FlashAttnFunc.apply is reachable without the dispatcher; ROCKE must raise there too."""
    q, k, v, sink, _ = _inputs(1, 1024)
    monkeypatch.setattr(
        flash_attn_interface,
        "attention_aiter_forward_impl",
        lambda *a, **k: pytest.fail("AITER forward ran for a ROCKE call"),
    )
    with pytest.raises(ValueError, match="rocKE flash-attn cannot take this call"):
        flash_attn_interface.FlashAttnFunc.apply(
            q.permute(1, 0, 2, 3), k.permute(1, 0, 2, 3), v.permute(1, 0, 2, 3),
            0.0, None, True, (128, 1), None, None, False, True, False, True, 1, sink, "sbhd",
            BackendType.ROCKE,
        )  # fmt: skip


# --------------------------------------------------------------------------------------
# GPU: routing
# --------------------------------------------------------------------------------------
@gfx950_rocke
@pytest.mark.parametrize("window_left", [-1, 127, 128])
def test_pinned_rocke_routes_to_rocke_launch_once(window_left, monkeypatch):
    q, k, v, sink, _ = _inputs(1, 1024)
    calls = []
    real = attention_rocke_impl._rocke_launch

    def spy(*a, **kw):
        calls.append(a[0])
        return real(*a, **kw)

    monkeypatch.setattr(attention_rocke_impl, "_rocke_launch", spy)
    for mod, name in (
        (flash_attn_interface, "flash_attn_sbhd_flydsl_forward_impl"),
        (flash_attn_interface, "attention_aiter_forward_impl"),
        (attention_impl, "flash_attn_sbhd_flydsl_forward_impl"),
    ):
        monkeypatch.setattr(mod, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} ran under pinned ROCKE"))
    _pin(BackendType.ROCKE)
    out, lse = _call(q, k, v, sink, window_left)
    torch.cuda.synchronize()
    assert len(calls) == 1
    assert calls[0].emit_lse and calls[0].use_sinks
    assert calls[0].sliding_window == (window_left + 1 if window_left >= 0 else 0)
    assert out.shape == (1, 1024, _HQ, _D) and lse.shape == (1, _HQ, 1024)


@gfx950_rocke
def test_pinned_rocke_forward_backward_routes_backward_to_flydsl(monkeypatch):
    q, k, v, sink, dout = _inputs(1, 1024, requires_grad=True)
    launches, bwd = [], []
    real = attention_rocke_impl._rocke_launch
    monkeypatch.setattr(
        attention_rocke_impl, "_rocke_launch", lambda *a, **kw: (launches.append(1), real(*a, **kw))[1]
    )
    real_bwd = flash_attn_interface.flash_attn_sbhd_flydsl_backward_impl
    monkeypatch.setattr(
        flash_attn_interface,
        "flash_attn_sbhd_flydsl_backward_impl",
        lambda *a, **kw: (bwd.append(1), real_bwd(*a, **kw))[1],
    )
    monkeypatch.setattr(
        flash_attn_interface,
        "attention_aiter_backward_impl",
        lambda *a, **k: pytest.fail("AITER backward ran"),
    )
    _pin(BackendType.ROCKE)
    out = _call(q, k, v, sink, -1, return_lse=False)
    out.backward(dout.permute(1, 0, 2, 3))
    torch.cuda.synchronize()
    assert (len(launches), len(bwd)) == (1, 1)


@gfx950_rocke
def test_profiler_shows_rocke_kernel_not_flydsl_or_aiter_in_forward():
    from torch.profiler import ProfilerActivity, profile

    q, k, v, sink, _ = _inputs(2, 1024)
    _pin(BackendType.ROCKE)
    _call(q, k, v, sink, -1)  # compile outside the trace
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _call(q, k, v, sink, -1)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    spec, _ = attention_rocke_impl._dense_spec(2, 1024, 1024, _HQ, _HKV, _D, 0, True, True)
    kernel_sym = spec.kernel_name()
    assert any(kernel_sym in n for n in names), (kernel_sym, names)
    for n in names:
        low = n.lower()
        assert "flash_attn" not in low and "fmha" not in low and "aiter" not in low, n


# --------------------------------------------------------------------------------------
# GPU: accuracy and gradients
# --------------------------------------------------------------------------------------
@gfx950_rocke
@pytest.mark.parametrize("window_left", [-1, 127, 128, 255], ids=["full", "w127", "w128", "w255"])
@pytest.mark.parametrize("B", [1, 2])
def test_rocke_forward_matches_primus_reference(B, window_left):
    q, k, v, sink, _ = _inputs(B, 1024)
    out_ref, lse_ref = _reference(q, k, v, sink, window_left)
    _pin(BackendType.ROCKE)
    out, lse = _call(q, k, v, sink, window_left)
    assert torch.isfinite(out).all()
    assert compute_snr(out_ref, out) > 40.0
    # LSE: natural log, sink included, [B, Hq, S] -- the layout the FlyDSL backward reads.
    assert lse.shape == lse_ref.shape
    torch.testing.assert_close(lse.float(), lse_ref, rtol=1e-2, atol=1e-2)


@gfx950_rocke
def test_rocke_primus_window_128_is_129_keys_not_128():
    """Primus (128, 0) is 129 keys. The output matches that reference and is far from the
    128-key one (127 earlier keys), so the band is really the Primus band, not a neighbour."""
    q, k, v, sink, _ = _inputs(2, 1024)
    out129, _ = _reference(q, k, v, sink, 128)
    out128, _ = _reference(q, k, v, sink, 127)
    _pin(BackendType.ROCKE)
    out, _ = _call(q, k, v, sink, 128)
    assert compute_snr(out129, out) > 40.0
    assert compute_snr(out128, out) < 30.0


@gfx950_rocke
@pytest.mark.parametrize("window_left", [-1, 127, 128], ids=["full", "w127", "w128"])
def test_rocke_fwd_bwd_grads_match_flydsl_only(window_left):
    B, S = 2, 1024
    q, k, v, sink, dout = _inputs(B, S, requires_grad=True)

    def run(backend):
        for t in (q, k, v, sink):
            t.grad = None
        _pin(backend)
        out = _call(q, k, v, sink, window_left, return_lse=False)
        out.backward(dout.permute(1, 0, 2, 3))
        torch.cuda.synchronize()
        return (
            out.detach().float(),
            q.grad.clone().float(),
            k.grad.clone().float(),
            v.grad.clone().float(),
            sink.grad.clone().float(),
        )

    ref = run(BackendType.FLYDSL)
    got = run(BackendType.ROCKE)
    for name, r, g in zip(("out", "dq", "dk", "dv", "dsink"), ref, got):
        assert torch.isfinite(g).all(), name
        snr = compute_snr(r, g)
        assert snr >= 40.0, f"{name}: SNR {snr:.2f} dB vs the FlyDSL-only path"


@gfx950_rocke
@pytest.mark.parametrize("window_left", [-1, 127, 128], ids=["full", "w127", "w128"])
def test_hoisted_launch_is_bitwise_equal_to_rocke_runner(window_left):
    """Launches after the first skip run_attention_dense_torch's per-call validation; the result
    must be bit-identical to calling that runner directly."""
    S, B = 1024, 2
    q, k, v, sink, _ = _inputs(B, S)
    wk = window_left + 1 if window_left >= 0 else 0
    spec, _ = attention_rocke_impl._dense_spec(B, S, S, _HQ, _HKV, _D, wk, True, True)
    _, run, _ = attention_rocke_impl._load_rocke()
    qb, kb, vb = (t.permute(1, 0, 2, 3).contiguous() for t in (q, k, v))
    out_ref = torch.empty_like(qb)
    lse_ref = torch.empty(B, _HQ, S, 1, dtype=torch.float32, device="cuda")
    run(
        spec=spec, q=qb, k=kb, v=vb, out=out_ref, scale=_D**-0.5, sinks=sink, lse=lse_ref,
        stream=torch.cuda.current_stream().cuda_stream,
    )  # fmt: skip
    window = (window_left, 0) if window_left >= 0 else (-1, -1)
    for _ in range(3):  # first call prepares the fast state, the rest use it
        out_s, lse = attention_rocke_impl.flash_attn_sbhd_rocke_forward_impl(
            q, k, v, window_size=window, sink=sink
        )
    torch.cuda.synchronize()
    assert attention_rocke_impl._FAST_LAUNCH.get(spec) is not None
    assert torch.equal(out_s.permute(1, 0, 2, 3), out_ref)
    assert torch.equal(lse.view(B, S, _HQ).permute(0, 2, 1), lse_ref.view(B, _HQ, S))


@gfx950_rocke
def test_dense_spec_takes_an_fp32_sink():
    spec, why = attention_rocke_impl._dense_spec(2, 1024, 1024, _HQ, _HKV, _D, 0, True, True)
    assert spec is not None, why
    assert spec.sink_dtype == "f32"
    spec, why = attention_rocke_impl._dense_spec(2, 1024, 1024, _HQ, _HKV, _D, 0, False, True)
    assert spec is not None and spec.sink_dtype == "q", why


@gfx950_rocke
def test_fp32_sink_reaches_rocke_unrounded(monkeypatch):
    q, k, v, sink, _ = _inputs(2, 1024)
    seen = {}
    real = attention_rocke_impl._rocke_launch

    def spy(spec, *args):
        seen["spec"], seen["sink"] = spec, args[5]
        return real(spec, *args)

    monkeypatch.setattr(attention_rocke_impl, "_rocke_launch", spy)
    attention_rocke_impl.flash_attn_sbhd_rocke_forward_impl(q, k, v, sink=sink)
    assert seen["spec"].sink_dtype == "f32"
    assert seen["sink"].dtype == torch.float32 and torch.equal(seen["sink"], sink)
