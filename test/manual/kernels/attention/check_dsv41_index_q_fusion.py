"""Check byte parity and latency of 890 index-Q preparation fusion.

Run in a PPU SGLang environment, from this checkout::

    PYTHONPATH=python python test/manual/kernels/attention/check_dsv41_index_q_fusion.py

Byte parity covers the actual Sail RoPE/fake-quant -> FP4 pack chain and the
BF16 head-weight round trip. Timings cover both post-projection preparation and
synthetic BF16 GEMMs plus preparation. They are microbenchmarks, not model E2E
measurements; use checkpoint dimensions via the CLI for representative GEMMs.
Both implementations are called directly, independently of runtime dispatch.
"""

import argparse
import json
import math
import statistics

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
    index_q_rope_pack_weights,
)
from sglang.kernels.ops.quantization.fp4_indexer_quant import (
    quantize_fp4_indexer_tensor,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import _rope_fq4
from sglang.srt.utils import is_sail


def unfused(q, freqs, positions, raw_weights, weight_scale):
    rotated = _rope_fq4(q, freqs[positions], 64)
    payload, scales = quantize_fp4_indexer_tensor(rotated.flatten(0, 1), rne=True)
    return payload, scales, (raw_weights * weight_scale).float()


def fused(q, freqs, positions, raw_weights, weight_scale):
    return index_q_rope_pack_weights(
        q,
        torch.view_as_real(freqs).flatten(-2),
        positions,
        raw_weights,
        weight_scale,
    )


def assert_bytes_equal(actual, expected, case):
    for name, got, ref in zip(("payload", "scales", "weights"), actual, expected):
        if got.shape != ref.shape or got.dtype != ref.dtype:
            raise AssertionError(f"{case}: {name} shape/dtype mismatch")
        # Byte views also catch signed-zero differences in the FP32 weights.
        a, b = got.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
        mismatches = int((a != b).sum().item())
        if mismatches:
            raise AssertionError(f"{case}: {name}: {mismatches} bytes differ")


def latency_us(fn, iterations):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(samples)


def profile_kernels(fn):
    # JIT compilation and warmup must not enter the measured profile.
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name] = kernels.get(event.name, 0) + 1
    return kernels


def make_inputs(rows, heads, max_pos, pos_dtype):
    q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(max_pos, 32, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    # Non-monotonic positions distinguish a table lookup from a row lookup.
    positions = torch.randint(max_pos, (rows,), device="cuda", dtype=pos_dtype)
    return q, freqs, positions, weights, 128**-0.5 * heads**-0.5


def check_boundaries(heads):
    # Fix each block's amax at 6, making its fake-quant scale exactly 1.
    # The other entries then exercise every E2M1 tie, both signs and signed zero.
    values = torch.tensor(
        [0.0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0],
        device="cuda",
    )
    values = torch.cat((values, -values))
    q, freqs, positions, weights, scale = make_inputs(7, heads, 8, torch.int64)
    freqs.fill_(1 + 0j)
    q.copy_(
        values.repeat(math.ceil(q.numel() / values.numel()))[: q.numel()].view_as(q)
    )
    q.unflatten(-1, (4, 32))[..., -1] = 6
    for magnitude in (0.0, 1.0, 2**-16, 2**-120, 2**-130, 2**8):
        case_q = q * magnitude
        # Flash's 32-head scale is exactly 1/64. A non-power-of-two scale
        # additionally guards the BF16 rounding before widening weights to FP32.
        for weight_scale in (scale, scale * math.sqrt(2)):
            assert_bytes_equal(
                fused(case_q, freqs, positions, weights, weight_scale),
                unfused(case_q, freqs, positions, weights, weight_scale),
                f"E2M1 boundaries, magnitude={magnitude}, weight_scale={weight_scale}",
            )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[0, 1, 7, 512, 16384])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--max-pos", type=int, default=131072)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--q-lora-rank", type=int, default=1280)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--profile", action="store_true", help="include kernel names/counts"
    )
    args = parser.parse_args()
    if (
        min(args.rows) < 0
        or min(
            args.heads,
            args.max_pos,
            args.hidden_size,
            args.q_lora_rank,
            args.iterations,
        )
        <= 0
    ):
        parser.error(
            "rows must be nonnegative and other dimensions/iterations positive"
        )
    if not torch.cuda.is_available() or not is_sail():
        raise RuntimeError(
            "Run this validation on 890 with the SAIL PyTorch environment"
        )
    torch.manual_seed(88532746)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "config": vars(args),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    check_boundaries(args.heads)
    inputs = []
    # Finish *all* correctness checks before reporting any speedup.
    for rows in args.rows:
        for dtype in (torch.int32, torch.int64):
            data = make_inputs(rows, args.heads, args.max_pos, dtype)
            assert_bytes_equal(
                fused(*data), unfused(*data), f"rows={rows}, pos={dtype}"
            )
            if dtype == torch.int64:
                inputs.append(data)
    print(json.dumps({"byte_parity": "passed"}), flush=True)

    wq = torch.randn(
        args.heads * 128, args.q_lora_rank, device="cuda", dtype=torch.bfloat16
    )
    ww = torch.randn(args.heads, args.hidden_size, device="cuda", dtype=torch.bfloat16)
    wq *= args.q_lora_rank**-0.5
    ww *= args.hidden_size**-0.5
    for q, freqs, positions, weights, scale in inputs:
        rows = q.shape[0]
        if not rows:
            continue
        x = torch.randn(rows, args.hidden_size, device="cuda", dtype=torch.bfloat16)
        lora = torch.randn(rows, args.q_lora_rank, device="cuda", dtype=torch.bfloat16)

        def prepare(op):
            projected = F.linear(lora, wq).view(rows, args.heads, 128)
            return op(projected, freqs, positions, F.linear(x, ww), scale)

        assert_bytes_equal(
            prepare(fused), prepare(unfused), f"GEMMs + prep, rows={rows}"
        )
        for scope, old, new in (
            (
                "post_projection",
                lambda: unfused(q, freqs, positions, weights, scale),
                lambda: fused(q, freqs, positions, weights, scale),
            ),
            (
                "synthetic_bf16_gemms_and_prep",
                lambda: prepare(unfused),
                lambda: prepare(fused),
            ),
        ):
            old_us = latency_us(old, args.iterations)
            new_us = latency_us(new, args.iterations)
            result = {
                "rows": rows,
                "scope": scope,
                "unfused_us": old_us,
                "fused_us": new_us,
                "speedup": old_us / new_us,
            }
            if args.profile:
                result["unfused_kernels"] = profile_kernels(old)
                result["fused_kernels"] = profile_kernels(new)
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
