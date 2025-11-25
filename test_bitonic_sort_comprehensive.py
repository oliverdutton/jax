#!/usr/bin/env python3
"""Comprehensive test of jaxpr caching optimization with different grid sizes."""

import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    """Simple kernel for testing."""
    x = x_ref[0]
    y = y_ref[0]
    result = x + y
    result = result * 2.0
    result = jnp.sin(result)
    o_ref[0] = result

def benchmark_grid_size(grid_size, num_runs=5, clear_cache=False, warmup=True):
    """Benchmark a specific grid size."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    interpret_params = interpret_pallas_call.InterpretParams()

    x = jnp.arange(grid_size, dtype=jnp.float32)
    y = jnp.arange(grid_size, dtype=jnp.float32) * 2.0

    def pallas_add(x, y):
        if clear_cache:
            interpret_pallas_call._compiled_jaxpr_cache.clear()

        return pl.pallas_call(
            add_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            in_specs=[
                pl.BlockSpec((1,), lambda i: (i,)),
                pl.BlockSpec((1,), lambda i: (i,)),
            ],
            out_specs=pl.BlockSpec((1,), lambda i: (i,)),
            grid=grid_size,
            interpret=interpret_params,
        )(x, y)

    # Warmup
    if warmup:
        _ = pallas_add(x, y).block_until_ready()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        result = pallas_add(x, y).block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)

    return sum(times) / len(times), min(times), max(times), result

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*20 + "JAXPR CACHING OPTIMIZATION TEST")
    print(" "*25 + "Grid Size: 2**9 = 512")
    print("="*80)

    grid_size = 2**9  # 512, as requested

    print(f"\nConfiguration:")
    print(f"  Grid size: {grid_size} (2**9)")
    print(f"  Kernel: Element-wise add + multiply + sin")
    print(f"  Each grid iteration: 1 element")
    print(f"  Total callbacks per run: ~{grid_size * 10}")

    # Test 1: Optimized (with caching)
    print("\n" + "="*80)
    print("TEST 1: OPTIMIZED (jaxpr caching ENABLED)")
    print("="*80)

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    # Ensure cache is populated
    print("\nWarmup run to populate cache...")
    _ = benchmark_grid_size(grid_size, num_runs=1, clear_cache=False, warmup=True)

    print(f"Cache size after warmup: {len(interpret_pallas_call._compiled_jaxpr_cache)} entries")

    print("\nRunning optimized benchmark...")
    avg_opt, min_opt, max_opt, result_opt = benchmark_grid_size(
        grid_size, num_runs=5, clear_cache=False, warmup=False
    )

    print(f"  Average: {avg_opt:.4f}s")
    print(f"  Min:     {min_opt:.4f}s")
    print(f"  Max:     {max_opt:.4f}s")

    # Test 2: Baseline (cache cleared each run)
    print("\n" + "="*80)
    print("TEST 2: BASELINE (cache cleared before each run)")
    print("="*80)

    print("\nRunning baseline benchmark (clearing cache each run)...")
    avg_baseline, min_baseline, max_baseline, result_baseline = benchmark_grid_size(
        grid_size, num_runs=5, clear_cache=True, warmup=False
    )

    print(f"  Average: {avg_baseline:.4f}s")
    print(f"  Min:     {min_baseline:.4f}s")
    print(f"  Max:     {max_baseline:.4f}s")

    # Verify correctness
    assert jnp.allclose(result_opt, result_baseline), "Results don't match!"

    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    speedup = avg_baseline / avg_opt
    speedup_min = min_baseline / min_opt

    print(f"\n{'Method':<30} {'Avg Time':<15} {'Min Time':<15} {'Speedup'}")
    print("-"*80)
    print(f"{'Baseline (no caching)':<30} {avg_baseline:>10.4f}s {min_baseline:>10.4f}s {'':>10}")
    print(f"{'Optimized (with caching)':<30} {avg_opt:>10.4f}s {min_opt:>10.4f}s {speedup:>9.2f}x")

    print(f"\n  Speedup (average): {speedup:.3f}x")
    print(f"  Speedup (best):    {speedup_min:.3f}x")
    print(f"  Time saved (avg):  {(avg_baseline - avg_opt)*1000:.1f}ms")

    if speedup >= 1.0:
        print(f"\n✓ Optimization successful! {speedup:.2f}x speedup achieved")
    else:
        print(f"\n⚠ Optimization shows no speedup (possibly within noise)")
        print(f"  This can happen when:")
        print(f"    - JAX's internal caching already handles recompilation")
        print(f"    - Grid size is too small for compilation overhead to matter")
        print(f"    - Other overheads dominate (callbacks, memory simulation)")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    print(f"\nFor grid_size={grid_size}:")
    print(f"  - Total execution time: ~{avg_opt:.2f}s")
    print(f"  - Time per iteration:   ~{avg_opt/grid_size*1000:.2f}ms")
    print(f"  - Callback overhead dominates (~{grid_size*10} callbacks)")

    print("\nCaching optimization helps most when:")
    print("  1. Same kernel is called multiple times")
    print("  2. Compilation overhead is significant")
    print("  3. JAX's internal caching doesn't apply")

    print("\nFor dramatic speedups (10-1000x), need more invasive changes:")
    print("  - Batch memory operations (reduce callbacks)")
    print("  - Add 'fast mode' (skip simulation)")
    print("  - Build real CPU backend")
    print("\nSee: pallas_interpret_mode_optimization_proposal.md")

    print("\n" + "="*80 + "\n")
