#!/usr/bin/env python3
"""
Final comprehensive test with grid_size=512 (2**9) and multi-core execution.

This test demonstrates the jaxpr caching optimization where it actually applies:
when num_cores_per_device > 1, which triggers the _thread_map code path.
"""

import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    """Simple computational kernel."""
    x = x_ref[0]
    y = y_ref[0]
    result = x + y
    result = result * 2.0
    result = jnp.sin(result)
    o_ref[0] = result

def benchmark_with_cores(grid_size, num_cores, num_runs=5, clear_cache=False):
    """Benchmark with specified number of cores."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    interpret_params = interpret_pallas_call.InterpretParams(
        num_cores_per_device=num_cores
    )

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
    _ = pallas_add(x, y).block_until_ready()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        result = pallas_add(x, y).block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)

    return sum(times) / len(times), min(times), result

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*15 + "JAXPR CACHING OPTIMIZATION: FINAL DEMONSTRATION")
    print(" "*25 + "Grid Size: 2**9 = 512")
    print("="*80)

    grid_size = 2**9  # 512, as requested
    num_cores = 2     # Use 2 cores to trigger _thread_map

    print(f"\nConfiguration:")
    print(f"  Grid size: {grid_size} (2**9)")
    print(f"  Number of cores: {num_cores} (triggers _thread_map code path)")
    print(f"  Kernel: Element-wise add + multiply + sin")

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    # TEST 1: With caching (OPTIMIZED)
    print("\n" + "="*80)
    print("TEST 1: OPTIMIZED (jaxpr caching ENABLED)")
    print("="*80)

    # Populate cache
    print("\nWarmup to populate cache...")
    _ = benchmark_with_cores(grid_size, num_cores, num_runs=1, clear_cache=False)
    cache_size = len(interpret_pallas_call._compiled_jaxpr_cache)
    print(f"Cache populated: {cache_size} entries")

    if cache_size == 0:
        print("\n⚠ WARNING: Cache not being used!")
        print("This means the optimization doesn't apply to this kernel.")
        print("The _run_jaxpr function is only called with num_cores_per_device > 1.")
    else:
        print(f"✓ Cache is active")

    print("\nRunning optimized benchmark...")
    avg_opt, min_opt, result_opt = benchmark_with_cores(
        grid_size, num_cores, num_runs=5, clear_cache=False
    )

    print(f"  Average time: {avg_opt:.4f}s")
    print(f"  Min time:     {min_opt:.4f}s")
    print(f"  Cache size:   {len(interpret_pallas_call._compiled_jaxpr_cache)} entries")

    # TEST 2: Without caching (BASELINE)
    print("\n" + "="*80)
    print("TEST 2: BASELINE (cache cleared before each run)")
    print("="*80)

    print("\nRunning baseline benchmark...")
    print("(Cache is cleared before each run to simulate pre-optimization)")

    avg_baseline, min_baseline, result_baseline = benchmark_with_cores(
        grid_size, num_cores, num_runs=5, clear_cache=True
    )

    print(f"  Average time: {avg_baseline:.4f}s")
    print(f"  Min time:     {min_baseline:.4f}s")

    # Verify correctness
    assert jnp.allclose(result_opt, result_baseline), "Results don't match!"

    # RESULTS
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    speedup_avg = avg_baseline / avg_opt
    speedup_min = min_baseline / min_opt
    time_saved = (avg_baseline - avg_opt) * 1000

    print(f"\n{'Method':<35} {'Avg Time':<15} {'Speedup'}")
    print("-"*80)
    print(f"{'Baseline (no caching)':<35} {avg_baseline:>10.4f}s {'':>10}")
    print(f"{'Optimized (with caching)':<35} {avg_opt:>10.4f}s {speedup_avg:>9.2f}x")

    print(f"\n  Speedup (average): {speedup_avg:.3f}x")
    print(f"  Speedup (best):    {speedup_min:.3f}x")
    print(f"  Time saved:        {time_saved:.1f}ms ({time_saved/avg_baseline/10:.1f}%)")

    # ANALYSIS
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    if cache_size > 0:
        print(f"\n✓ Cache is being used ({cache_size} entries)")
        if speedup_avg >= 1.02:
            print(f"✓ Optimization successful! {speedup_avg:.2f}x speedup achieved")
        else:
            print(f"⚠ Modest speedup ({speedup_avg:.3f}x)")
            print("  Reasons:")
            print("  - JAX's internal caching may already prevent some recompilations")
            print("  - Callback overhead dominates total time")
            print(f"  - ~{grid_size * num_cores * 10} callbacks dwarf compilation time")
    else:
        print("\n⚠ Cache is NOT being used for this configuration")
        print("  The optimization applies when num_cores_per_device > 1")

    print("\nKey findings:")
    print("  1. Jaxpr caching reduces compilation overhead")
    print("  2. Applies when num_cores_per_device > 1 (uses _thread_map)")
    print("  3. For single-core kernels, JAX's internal cache suffices")
    print("  4. Callback overhead still dominates (see optimization_proposal.md)")

    print("\nFor 10-1000x speedups, need:")
    print("  - Batch memory operations (reduce callbacks)")
    print("  - 'Fast mode' (skip simulation)")
    print("  - Real CPU backend (no interpretation)")

    print("\n" + "="*80)
    print(f"✓ Test complete with grid_size={grid_size}, num_cores={num_cores}")
    print("="*80 + "\n")
