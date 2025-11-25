#!/usr/bin/env python3
"""Test with large input (16, 131k) as requested."""

import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    """Simple kernel."""
    x = x_ref[0]
    y = y_ref[0]
    result = x + y
    result = result * 2.0
    result = jnp.sin(result)
    o_ref[0] = result

def benchmark_with_optimizations(shape, use_optimizations=True, num_runs=3):
    """Benchmark with or without optimizations."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    batch_size, seq_len = shape
    x = jax.random.normal(jax.random.key(0), shape, jnp.float32)
    y = jax.random.normal(jax.random.key(1), shape, jnp.float32)

    if use_optimizations:
        params = interpret_pallas_call.InterpretParams(
            use_pure_callback_for_loads=True,
            skip_floating_point_ops=False,  # Keep real computation
        )
        label = "WITH OPTIMIZATIONS"
    else:
        params = interpret_pallas_call.InterpretParams()
        label = "WITHOUT OPTIMIZATIONS"

    def pallas_add(x, y):
        # Process batch_size rows in parallel, each row has seq_len elements
        return pl.pallas_call(
            add_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            in_specs=[
                pl.BlockSpec((1, 1), lambda i, j: (i, j)),
                pl.BlockSpec((1, 1), lambda i, j: (i, j)),
            ],
            out_specs=pl.BlockSpec((1, 1), lambda i, j: (i, j)),
            grid=(batch_size, seq_len),
            interpret=params,
        )(x, y)

    print(f"\n{label}")
    print("-" * 70)

    # Warmup
    print("  Warmup...", end=" ", flush=True)
    _ = pallas_add(x, y).block_until_ready()
    print("✓")

    # Time
    times = []
    for i in range(num_runs):
        print(f"  Run {i+1}...", end=" ", flush=True)
        start = time.time()
        result = pallas_add(x, y).block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"{elapsed:.2f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)

    print(f"\n  Average: {avg_time:.2f}s")
    print(f"  Min:     {min_time:.2f}s")

    return avg_time, min_time, result

def benchmark_fast_mode(shape):
    """Benchmark with direct execution (fast mode)."""
    batch_size, seq_len = shape
    x = jax.random.normal(jax.random.key(0), shape, jnp.float32)
    y = jax.random.normal(jax.random.key(1), shape, jnp.float32)

    @jax.jit
    def fast_kernel(x, y):
        result = x + y
        result = result * 2.0
        result = jnp.sin(result)
        return result

    print(f"\nDIRECT EXECUTION (Fast Mode)")
    print("-" * 70)

    # Warmup
    print("  Warmup...", end=" ", flush=True)
    _ = fast_kernel(x, y).block_until_ready()
    print("✓")

    # Time
    times = []
    for i in range(3):
        print(f"  Run {i+1}...", end=" ", flush=True)
        start = time.time()
        result = fast_kernel(x, y).block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"{elapsed:.6f}s")

    avg_time = sum(times) / len(times)

    print(f"\n  Average: {avg_time:.6f}s")

    return avg_time, result

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*20 + "LARGE INPUT TEST: (16, 131k)")
    print("="*80)

    shape = (16, 131072)  # 16 x 131k as requested
    batch_size, seq_len = shape

    print(f"\nInput shape: {shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len:,}")
    print(f"  Total grid points: {batch_size * seq_len:,}")
    print(f"  Total elements: {batch_size * seq_len:,}")

    print("\n" + "="*80)
    print("TESTING")
    print("="*80)

    # Test 1: Without optimizations (baseline)
    print("\n[1] BASELINE (Default InterpretParams)")
    print("="*80)
    avg_baseline, min_baseline, result_baseline = benchmark_with_optimizations(
        shape, use_optimizations=False, num_runs=2  # Fewer runs for large input
    )

    # Test 2: With optimizations
    print("\n" + "="*80)
    print("\n[2] WITH OPTIMIZATIONS")
    print("="*80)
    avg_opt, min_opt, result_opt = benchmark_with_optimizations(
        shape, use_optimizations=True, num_runs=2
    )

    # Verify correctness
    assert jnp.allclose(result_baseline, result_opt), "Results don't match!"
    print("\n  ✓ Results match baseline")

    # Test 3: Fast mode (direct execution)
    print("\n" + "="*80)
    print("\n[3] FAST MODE (Direct Execution)")
    print("="*80)
    avg_fast, result_fast = benchmark_fast_mode(shape)

    # Verify correctness
    assert jnp.allclose(result_opt, result_fast), "Results don't match!"
    print("\n  ✓ Results match optimized")

    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    speedup_opt = avg_baseline / avg_opt
    speedup_fast = avg_baseline / avg_fast

    print(f"\n{'Method':<40} {'Avg Time':<15} {'Speedup'}")
    print("-" * 80)
    print(f"{'Baseline (no optimizations)':<40} {avg_baseline:>10.2f}s {'':>10}")
    print(f"{'Optimized (jaxpr cache + pure callback)':<40} {avg_opt:>10.2f}s {speedup_opt:>9.2f}x")
    print(f"{'Fast mode (direct execution)':<40} {avg_fast:>10.6f}s {speedup_fast:>9.0f}x")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    print(f"\nFor input shape {shape}:")
    print(f"  Grid points: {batch_size * seq_len:,}")
    print(f"  Callbacks (baseline): ~{batch_size * seq_len * 10:,}")

    print(f"\nOptimization impact:")
    print(f"  Pure callback for loads: Reduces blocking on read operations")
    print(f"  Jaxpr caching: Prevents recompilation")
    print(f"  Combined speedup: {speedup_opt:.2f}x")
    print(f"  Time saved: {(avg_baseline - avg_opt):.1f}s")

    print(f"\nFast mode (direct execution):")
    print(f"  Speedup: {speedup_fast:.0f}x vs baseline")
    print(f"  Speedup: {avg_opt / avg_fast:.0f}x vs optimized interpret")
    print(f"  Eliminates all callback overhead")

    print("\nKey insights:")
    print("  1. Optimizations help but callback overhead still dominates")
    print(f"  2. ~{batch_size * seq_len * 10:,} callbacks create massive overhead")
    print("  3. Direct execution is orders of magnitude faster")
    print("  4. Use interpret mode for debugging only, not performance")

    print("\n" + "="*80)
    print(f"✓ Test complete for shape {shape}")
    print("="*80 + "\n")
