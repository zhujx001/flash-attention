# Copyright (c) 2023, Tri Dao.
#
# Fast "three-region" decoding for sink + retrieved + local attention.
#
# Layout assumed by this helper:
#   - resident segment(s): the GPU-resident KV, e.g. the attention-sink tokens
#     (first N) together with the recent sliding window + freshly generated
#     tokens (last M). These are already on the GPU and can be attended to
#     immediately.
#   - retrieved segment: the "middle" KV that was selected/retrieved and lives
#     in CPU (host) memory. It must be copied to the GPU first, which is the
#     slow part we want to hide.
#
# Strategy (fastest with the existing kernel, no CUDA changes):
#   1. Kick off the async H2D copy of the retrieved KV on a side stream.
#   2. While the copy is in flight, compute attention over the GPU-resident KV
#      on the default stream -> (out, lse).
#   3. Once the copy lands, compute attention over the retrieved KV and fuse it
#      with (out, lse) via the kernel's previous_out / previous_lse path. The
#      fusion is an exact LSE (log-sum-exp) merge, identical to attending over
#      all three regions in a single pass.

import torch

from weighted_flash_decoding.weighted_flash_decoding_interface import (
    weighted_flash_decoding,
)


@torch.no_grad()
def fused_retrieval_decoding(
    q,
    resident_k,
    resident_v,
    retrieved_k_host,
    retrieved_v_host,
    resident_seqlens=None,
    retrieved_seqlens=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    copy_stream=None,
    return_softmax_lse=False,
):
    """Decode attention over (resident KV) + (CPU-retrieved KV) with the
    CPU->GPU transfer overlapped against the resident attention compute.

    Arguments:
        q: (batch, seqlen_q, nheads, headdim), fp16/bf16, on GPU.
        resident_k, resident_v: (batch, seqlen_res, nheads_k, headdim) on GPU.
            The sink + local-window + newly-generated KV.
        retrieved_k_host, retrieved_v_host: (batch, seqlen_ret, nheads_k, headdim)
            in CPU memory. For a real overlap these MUST be pinned
            (page-locked) tensors -- allocate with torch.empty(..., pin_memory=True)
            or call .pin_memory(). Non-pinned host tensors force a synchronous copy.
        resident_seqlens / retrieved_seqlens: int or (batch,) int32. Valid length
            of each cache. If None, the full second dim is used.
        softmax_scale: float, defaults to headdim ** -0.5.
        causal: bool. For pure decoding (seqlen_q == 1) this is a no-op; only
            relevant for multi-token queries, and note that causal ordering is
            NOT enforced *across* the two regions (each region is masked on its own).
        window_size: passed through to the local-window logic of each call.
        copy_stream: optional torch.cuda.Stream used for the H2D copy. A new one
            is created if not provided. Pass a persistent stream to avoid
            per-step stream-creation overhead.
        return_softmax_lse: if True, also return the fused logsumexp.

    Returns:
        out: (batch, seqlen_q, nheads, headdim).
        softmax_lse [if return_softmax_lse]: (batch, nheads, seqlen_q), fp32.
    """
    device = q.device
    assert q.is_cuda, "q must be on GPU"
    assert resident_k.is_cuda and resident_v.is_cuda, "resident KV must be on GPU"

    if copy_stream is None:
        copy_stream = torch.cuda.Stream(device=device)

    # (1) Launch the async H2D copy of the retrieved KV on a side stream so it
    #     overlaps with the resident attention below. non_blocking only actually
    #     overlaps when the host tensors are pinned.
    copy_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(copy_stream):
        retrieved_k = retrieved_k_host.to(device, non_blocking=True)
        retrieved_v = retrieved_v_host.to(device, non_blocking=True)

    # (2) Resident attention (sink + local + new). No dependency on the copy, so
    #     it runs concurrently with the transfer.
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

    # (3) Make the compute stream wait for the copy, then fuse the retrieved
    #     segment in via the previous_out / previous_lse path.
    current = torch.cuda.current_stream()
    current.wait_stream(copy_stream)
    # Tell the allocator these tensors are now used on the compute stream so
    # their memory isn't recycled while the copy / consumption is pending.
    retrieved_k.record_stream(current)
    retrieved_v.record_stream(current)

    out, lse = weighted_flash_decoding(
        q,
        retrieved_k,
        retrieved_v,
        previous_out=out,
        previous_lse=lse,
        cache_seqlens=retrieved_seqlens,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        return_softmax_lse=True,
    )

    return (out, lse) if return_softmax_lse else out
