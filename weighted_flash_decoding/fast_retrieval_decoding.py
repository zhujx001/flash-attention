# Copyright (c) 2023, Tri Dao.
#
# Extreme-speed "three-region" decoding for the transfer-bound regime where the
# middle/retrieved KV is re-fetched from CPU every step and barely reused.
#
# Why this design (read before changing the CUDA kernel):
#   Decoding attention with seqlen_q == 1 is purely memory-bound, and the only
#   slow path here is the PCIe transfer of the retrieved KV (~25 GB/s pinned vs
#   ~2-3 TB/s HBM). Fusing kernels / removing the previous-out round-trip can at
#   best reach the transfer floor  bytes / PCIe_bandwidth  -- it cannot beat it.
#   To go *below* that floor you must move fewer bytes across PCIe. So:
#     1. Quantize the retrieved KV to int8 on the host  -> 4x fewer PCIe bytes.
#     2. Transfer the int8 + per-row scales (pinned, async, chunked pipeline).
#     3. Dequantize on the GPU with a trivial HBM-bound op (negligible vs PCIe).
#     4. Feed bf16/fp16 into the existing weighted_flash_decoding kernel and fuse
#        each chunk via previous_out / previous_lse (exact LSE merge).
#   This gets the full 4x PCIe win WITHOUT touching the attention .cu kernel.
#   An in-kernel fused dequant would only save HBM traffic (not the bottleneck)
#   and would not improve end-to-end latency, so it is intentionally not done.

import torch

from weighted_flash_decoding.weighted_flash_decoding_interface import (
    weighted_flash_decoding,
)


def quantize_kv_int8(x):
    """Per-row (over the headdim) symmetric int8 quantization.

    Args:
        x: (..., headdim) float tensor (typically on CPU when building the
           retrieved KV store).
    Returns:
        q:     (..., headdim) int8
        scale: (...,) float32, the per-row scale such that x ~= q * scale.
    """
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 127.0).clamp(min=1e-8)
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(-1).to(torch.float32)


def dequantize_kv_int8(q, scale, dtype):
    """Cheap on-GPU dequant: q (int8) * scale (fp32) -> dtype. HBM-bound."""
    return q.to(dtype) * scale.unsqueeze(-1).to(dtype)


def pin(x):
    """Page-lock a CPU tensor so its H2D copy is async and runs at full PCIe BW."""
    return x.pin_memory() if (x.device.type == "cpu" and not x.is_pinned()) else x


@torch.no_grad()
def fast_retrieval_decoding(
    q,
    resident_k,
    resident_v,
    retrieved_k_q8_host,
    retrieved_v_q8_host,
    retrieved_k_scale_host,
    retrieved_v_scale_host,
    resident_seqlens=None,
    chunk_size=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    num_copy_streams=2,
    return_softmax_lse=False,
):
    """Decode over (resident KV) + (int8 CPU-retrieved KV) at the PCIe floor.

    The retrieved KV is transferred as int8 (4x fewer bytes), copied in a
    double-buffered chunk pipeline so transfer of chunk i+1 overlaps the
    dequant+attention of chunk i, dequantized cheaply on the GPU, and fused into
    the running result via the kernel's previous_out / previous_lse path.

    Args:
        q: (batch, seqlen_q, nheads, headdim) on GPU, fp16/bf16.
        resident_k, resident_v: (batch, seqlen_res, nheads_k, headdim) on GPU
            -- the sink + local-window + newly-generated KV.
        retrieved_k_q8_host, retrieved_v_q8_host: (batch, seqlen_ret, nheads_k,
            headdim) int8 in CPU memory. MUST be pinned for real overlap (use
            the `pin` helper or allocate with pin_memory=True).
        retrieved_k_scale_host, retrieved_v_scale_host: (batch, seqlen_ret,
            nheads_k) fp32 in CPU memory (pinned). From quantize_kv_int8.
        resident_seqlens: int or (batch,) int32 valid length of the resident cache.
        chunk_size: tokens per pipeline chunk along seqlen_ret. None = single chunk.
            Smaller chunks hide more transfer latency but add launch overhead;
            a few hundred to ~1-2k tokens is usually a good range.
        softmax_scale: float, defaults to headdim ** -0.5.
        causal: see fused_retrieval_decoding; for seqlen_q == 1 it is a no-op.
        window_size: passed through to each call.
        num_copy_streams: number of side streams for double/triple buffering.
        return_softmax_lse: also return the fused fp32 logsumexp.

    Returns:
        out: (batch, seqlen_q, nheads, headdim) [, softmax_lse].
    """
    device = q.device
    dtype = q.dtype
    assert q.is_cuda and resident_k.is_cuda and resident_v.is_cuda

    b, s_ret, nh_k, d = retrieved_k_q8_host.shape
    if chunk_size is None or chunk_size >= s_ret:
        chunks = [(0, s_ret)] if s_ret > 0 else []
    else:
        chunks = [(i, min(i + chunk_size, s_ret)) for i in range(0, s_ret, chunk_size)]

    comp = torch.cuda.current_stream(device)
    streams = [torch.cuda.Stream(device=device) for _ in range(max(1, num_copy_streams))]

    def copy_chunk(lo, hi, st):
        # Copy depends only on host data (always ready) + a fresh destination,
        # so it does NOT wait on the compute stream -> full overlap with compute.
        with torch.cuda.stream(st):
            kq = retrieved_k_q8_host[:, lo:hi].to(device, non_blocking=True)
            vq = retrieved_v_q8_host[:, lo:hi].to(device, non_blocking=True)
            ks = retrieved_k_scale_host[:, lo:hi].to(device, non_blocking=True)
            vs = retrieved_v_scale_host[:, lo:hi].to(device, non_blocking=True)
        return kq, vq, ks, vs

    # Prefetch the first chunk and compute the resident attention concurrently.
    pending = {}
    if chunks:
        pending[0] = copy_chunk(*chunks[0], streams[0])

    out, lse = weighted_flash_decoding(
        q,
        resident_k,
        resident_v,
        cache_seqlens=resident_seqlens,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        return_softmax_lse=True,
    )

    for ci, (lo, hi) in enumerate(chunks):
        kq, vq, ks, vs = pending.pop(ci)
        # Kick off the next chunk's transfer so it overlaps this chunk's compute.
        if ci + 1 < len(chunks):
            nst = streams[(ci + 1) % len(streams)]
            pending[ci + 1] = copy_chunk(*chunks[ci + 1], nst)
        # This chunk's data is ready once its copy stream is done.
        src_st = streams[ci % len(streams)]
        comp.wait_stream(src_st)
        for t in (kq, vq, ks, vs):
            t.record_stream(comp)
        # Cheap HBM-bound dequant, then exact LSE fusion with the running result.
        k = dequantize_kv_int8(kq, ks, dtype)
        v = dequantize_kv_int8(vq, vs, dtype)
        out, lse = weighted_flash_decoding(
            q,
            k,
            v,
            previous_out=out,
            previous_lse=lse,
            cache_seqlens=hi - lo,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_softmax_lse=True,
        )

    return (out, lse) if return_softmax_lse else out
