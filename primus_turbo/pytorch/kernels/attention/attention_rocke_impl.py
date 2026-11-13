###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""rocKE dense attention forward for gfx950 / MI355X (proof of concept, "Option A").

Turbo imports rocKE's Python packages and launches ONE pinned rocKE spec per layer kind. It does
not use rocKE's dispatcher. Everything that touches rocKE sits in this file, and the launch itself
sits behind ``_rocke_launch``: replacing that one function (and ``_load_rocke``) with a launcher
for a precompiled .hsaco ("Option B") leaves the rest of the adapter unchanged.

Scope: forward only, bf16, head dim 64, causal, Sq == Skv, SBHD storage, optional fp32 per-head
sink (passed to rocKE as fp32, ``sink_dtype="f32"``, never rounded; needs ROCm/rocm-libraries#15721), left-only window. The
backward is FlyDSL's: this module returns the same objects
``flash_attn_sbhd_flydsl_forward_impl`` returns (``out`` sbhd-contiguous, ``lse`` [B*Sq, Hq] fp32,
natural log, sink included), so the existing FlyDSL backward consumes them unchanged.

Window semantics. Turbo/FA ``window_size=(W, 0)`` keeps the W+1 keys ``q-W <= k <= q`` (the
diagonal included). rocKE's ``sliding_window=S`` keeps S keys (``q-k < S``), so ``(W, 0)`` is
``sliding_window = W + 1`` for any W >= 0: Primus's GPT-OSS ``(128, 0)`` is ``sliding_window=129``.
This needs rocKE's gfx950 dense forward with the any-window start-tile rounding (ROCm/rocm-libraries#15708); on the
older source, where S had to be a multiple of the tile's ``block_n``, rocKE's own spec rejects the
window and the call is refused (never approximated by a neighbouring window).
"""

import dataclasses
import functools
import importlib
import math
import os
import sys
from typing import Optional, Tuple

import torch

_import_module = importlib.import_module  # one name for tests to replace

ENV_ROCKE_PATH = "PRIMUS_TURBO_ROCKE_PATH"
"""Directory holding rocKE's sources (``rocke/`` and ``library/``). Unset: use whatever
``import rocke`` / ``import kernels.gfx950.attention_dense`` already resolve."""

# Head dim the pinned specs were measured on. rocKE itself also builds 128.
ROCKE_HEAD_DIMS = (64,)

# --------------------------------------------------------------------------------------
# Pinned specs, from a rocKE tile sweep (12x4096, MI355X, rocKE 2db5a46d): dataclass
# overrides on rocKE's gfx950 dense default. Everything not listed keeps the default. Shape
# fields (batch, sequence lengths, head counts) are NOT pinned: with ``runtime_shape`` one compiled
# binary serves every shape, so the spec cache below only restates them per shape.
# Sequence lengths must be multiples of block_m and block_n; any window length is legal.
# --------------------------------------------------------------------------------------
ROCKE_SPEC_OVERRIDES = {
    # window layers (sliding_window > 0): 128x128 tile, grid body
    "window": dict(block_m=128, block_n=128, persistent=False, nonpersist_decode="qb_minor", exp_per_pv_step=1),
    # full causal layers: 256x128 tile, persistent, 512 workgroups
    "full": dict(block_m=256, block_n=128, persistent=True, num_persistent=512),
}


class RockeUnavailableError(ImportError):
    """rocKE could not be imported; the message says how to make it importable."""


_ROCKE = None
_ROCKE_ERROR: Optional[str] = None


def _load_rocke():
    """Import rocKE once and return ``(Gfx950AttentionDenseSpec, run_attention_dense_torch,
    supports_attention_dense)``. ``torch`` is imported first so that rocKE's comgr/hip resolver
    sees the copies torch already mapped."""
    global _ROCKE, _ROCKE_ERROR
    if _ROCKE is not None:
        return _ROCKE
    root = os.environ.get(ENV_ROCKE_PATH)
    try:
        if root:
            root = os.path.abspath(root)
            for path in reversed([root, os.path.join(root, "library")]):
                if path not in sys.path:
                    sys.path.insert(0, path)
        _import_module("rocke")
        mod = _import_module("kernels.gfx950.attention_dense")
        if root and not os.path.abspath(mod.__file__).startswith(root):
            raise ImportError(f"kernels resolved to {mod.__file__}, not under {ENV_ROCKE_PATH}={root}")
        if "sink_dtype" not in {f.name for f in dataclasses.fields(mod.Gfx950AttentionDenseSpec)}:
            raise ImportError("this rocKE's gfx950 dense attention has no fp32 sink (sink_dtype)")
    except ImportError as exc:
        _ROCKE_ERROR = f"{type(exc).__name__}: {exc}"
        raise RockeUnavailableError(
            "rocKE is not importable, but the ROCKE attention backend was requested. Install rocKE or "
            f"set {ENV_ROCKE_PATH} to a rocKE source tree (a directory holding rocke/ and library/). "
            f"Import error: {_ROCKE_ERROR}"
        ) from exc
    _ROCKE = (
        mod.Gfx950AttentionDenseSpec,
        mod.run_attention_dense_torch,
        mod.supports_attention_dense,
    )
    return _ROCKE


def rocke_available() -> bool:
    try:
        _load_rocke()
    except RockeUnavailableError:
        return False
    return True


@functools.lru_cache(maxsize=256)
def _dense_spec(
    batch: int,
    seqlen_q: int,
    seqlen_kv: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    window_keys: int,
    use_sinks: bool,
    emit_lse: bool,
) -> Tuple[Optional[object], str]:
    """``(spec, "")`` for a problem rocKE serves, else ``(None, reason)``. Cached per shape,
    refusals included: a rocKE spec validates itself on every construction and launch."""
    spec_cls, _, supports = _load_rocke()
    overrides = ROCKE_SPEC_OVERRIDES["window" if window_keys > 0 else "full"]
    try:
        spec = spec_cls(
            batch=batch,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            num_query_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            causal=True,
            dtype="bf16",
            sliding_window=window_keys,
            use_sinks=use_sinks,
            sink_dtype="f32" if use_sinks else "q",
            emit_lse=emit_lse,
            **overrides,
        )
    except (ValueError, NotImplementedError) as exc:
        return None, f"rocKE spec rejected: {exc}"
    ok, why = supports(spec)
    if not ok:
        return None, f"rocKE spec unsupported: {why}"
    if not spec.runtime_shape:
        # A baked-shape spec would recompile per shape; the pinned specs are runtime_shape.
        return None, "pinned rocKE spec is not runtime_shape"
    return spec, ""


def _window_keys(window_size) -> int:
    """Keys per query that Turbo's ``(W, 0)`` / ``(-1, x)`` window keeps, as rocKE's
    ``sliding_window`` (0 = no window)."""
    wl = int(window_size[0])
    return wl + 1 if wl >= 0 else 0


def rocke_attn_unsupported_reason(q, k, v, window_size, has_sink: bool, emit_lse: bool = True) -> Optional[str]:
    """Why rocKE cannot serve these SBHD ``[S, B, H, D]`` tensors, or None. This is the
    rocKE-specific part of the gate; the caller adds the FlyDSL-backward gate."""
    if not rocke_available():
        return f"rocKE is not importable ({_ROCKE_ERROR})"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return "q/k/v must be 4-D"
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        return "bf16 only"
    if not (q.is_cuda and q.device == k.device == v.device):
        return "q/k/v must be on one CUDA device"
    Sq, B, Hq, D = q.shape
    Skv, Bk, Hkv, Dk = k.shape
    if v.shape != k.shape or Bk != B or Dk != D:
        return "k/v shape does not match q"
    if D not in ROCKE_HEAD_DIMS:
        return f"head dim {D} not in the pinned set {ROCKE_HEAD_DIMS}"
    if Sq != Skv:
        return "Sq != Skv"
    if min(B, Sq, Hq, Hkv) <= 0 or Hq % Hkv != 0:
        return "empty or non-GQA shape"
    spec, why = _dense_spec(B, Sq, Skv, Hq, Hkv, D, _window_keys(window_size), has_sink, emit_lse)
    return None if spec is not None else why


# Per-spec launch state. ``run_attention_dense_torch`` re-validates the spec and rebuilds the launcher
# cache key on every call (0.6-2.6 ms of host time, measured by RockeVsFlydsl), which would starve a
# 0.4 ms kernel. The first launch of a spec goes through it (validation + compile); later launches
# reuse the launcher it cached and the spec's grid/block/runtime args. Only plain dense specs
# (no varlen/paged/kv-split) take this path; the pinned ones are all plain.
_FAST_LAUNCH = {}


def _prepare_fast_launch(spec):
    mod = _import_module("kernels.gfx950.attention_dense")
    rt = _import_module("rocke.runtime")
    launcher = mod._DENSE_LAUNCHER_CACHE[mod.attention_dense_cache_key(spec, arch="gfx950")]
    return (
        launcher,
        rt.LaunchConfig,
        mod.attention_dense_grid(spec),
        mod.attention_dense_block(spec),
        dict(mod.attention_dense_runtime_args(spec)),
    )


def _rocke_launch(spec, q, k, v, out, lse, sink, scale: float, stream: int) -> None:
    """The single call into rocKE. ``q/k/v/out`` are bshd-contiguous ``[B, S, H, D]`` bf16,
    ``lse`` is fp32 ``[B, Hq, Sq, 1]`` (or None), ``sink`` is fp32 ``[Hq]`` (or None)."""
    fast = _FAST_LAUNCH.get(spec)
    if fast is None:
        _, run, _ = _load_rocke()
        run(spec=spec, q=q, k=k, v=v, out=out, scale=scale, stream=stream, sinks=sink, lse=lse)
        if not (spec.varlen or spec.paged or spec.kv_segments > 1):
            _FAST_LAUNCH[spec] = _prepare_fast_launch(spec)
        return
    launcher, launch_config, grid, block, runtime_args = fast
    vals = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out, "scale": float(scale), **runtime_args}
    if sink is not None:
        vals["sink_ptr"] = sink
    if lse is not None:
        vals["lse_ptr"] = lse
    launcher(vals, config=launch_config(grid=grid, block=block, stream=int(stream)))


def flash_attn_sbhd_rocke_forward_impl(
    q,
    k,
    v,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    return_lse=True,
    sink=None,
):
    """SBHD forward on rocKE, same contract as ``flash_attn_sbhd_flydsl_forward_impl``.

    q ``[Sq, B, Hq, D]``, k/v ``[Skv, B, Hkv, D]`` bf16, sink fp32 ``[Hq]`` or None. Returns
    ``out`` ``[Sq, B, Hq, D]`` contiguous and, when ``return_lse``, ``lse`` ``[B*Sq, Hq]`` fp32
    natural-log (sink included), batch-major: ``lse.view(B, Sq, Hq).permute(0, 2, 1)`` is
    ``[B, Hq, Sq]``.

    rocKE reads bshd, so q/k/v are copied to bshd and ``out`` back to sbhd (no copy when B == 1).
    The fp32 sink is passed as is; rocKE loads it as fp32 (``sink_dtype='f32'``).
    """
    assert causal, "rocKE flash-attn forward is causal only"
    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    scale = 1.0 / math.sqrt(D)
    if softmax_scale is not None and abs(softmax_scale - scale) >= 1e-6:
        raise ValueError(f"rocKE flash-attn forward bakes softmax_scale=1/sqrt(D), got {softmax_scale}")
    wr = int(window_size[1])
    if wr not in (0, -1):
        raise ValueError(f"only left-window (W,0) / full (-1,-1) supported, got {tuple(window_size)}")
    if sink is not None and not (sink.dtype == torch.float32 and sink.numel() == Hq):
        raise ValueError("sink must be fp32 [Hq]")
    why = rocke_attn_unsupported_reason(q, k, v, window_size, sink is not None, bool(return_lse))
    if why is not None:
        raise ValueError(f"rocKE attention cannot serve this call: {why}")
    spec, _ = _dense_spec(B, Sq, Skv, Hq, Hkv, D, _window_keys(window_size), sink is not None, bool(return_lse))

    if q.device != torch.device("cuda", torch.cuda.current_device()):
        with torch.cuda.device(q.device):
            return flash_attn_sbhd_rocke_forward_impl(
                q, k, v, softmax_scale, causal, window_size, return_lse, sink
            )

    qb, kb, vb = (t.permute(1, 0, 2, 3).contiguous() for t in (q, k, v))
    out_b = torch.empty_like(qb)
    lse_b = torch.empty((B, Hq, Sq, 1), dtype=torch.float32, device=q.device) if return_lse else None
    sink_b = sink.contiguous() if sink is not None else None
    _rocke_launch(
        spec, qb, kb, vb, out_b, lse_b, sink_b, scale, torch._C._cuda_getCurrentRawStream(q.device.index)
    )
    out = out_b.permute(1, 0, 2, 3).contiguous()
    if not return_lse:
        return out
    # [B, Hq, Sq, 1] -> the FlyDSL kernel layout [B*Sq, Hq]. Kept physically [B, Sq, Hq] because
    # the FlyDSL backward reads this view and is fastest on that stride pattern.
    lse = lse_b.view(B, Hq, Sq).permute(0, 2, 1).contiguous().view(B * Sq, Hq)
    return out, lse


__all__ = [
    "ENV_ROCKE_PATH",
    "ROCKE_HEAD_DIMS",
    "ROCKE_SPEC_OVERRIDES",
    "RockeUnavailableError",
    "flash_attn_sbhd_rocke_forward_impl",
    "rocke_attn_unsupported_reason",
    "rocke_available",
]
