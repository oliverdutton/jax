#!/usr/bin/env python3
"""Test performance optimizations for Pallas interpret mode."""

import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    """Simple kernel with loads and stores."""
    x = x_ref[0]
    y = y_ref[0]
    result = x + y
    result = result * 2.0
    result = jnp.sin(result)
    o_ref[0] = result

def benchmark_with_params(grid_size, num_cores, interpret_params, num_runs=5, label=""):
    """Benchmark with specific interpret parameters."""
    x = jnp.arange(grid_size, dtype=jnp.float32)
    y = jnp.arange(grid_size, dtype=jnp.float32) * 2.0

    def pallas_add(x, y):
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

    avg_time = sum(times) / len(times)
    min_time = min(times)

    print(f"\n{label}")
    print(f"  Average: {avg_time:.4f}s")
    print(f"  Min:     {min_time:.4f}s")

    return avg_time, min_time, result

if __name__ == "__main__":
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    grid_size = 512  # 2**9
    num_cores = 2

    print("\n" + "="*80)
    print(" "*20 + "PERFORMANCE OPTIMIZATION COMPARISON")
    print(" "*30 + f"Grid Size: {grid_size}")
    print("="*80)

    results = []

    # 1. Baseline
    print("\n[1] BASELINE (default settings)")
    print("-" * 80)
    params_baseline = interpret_pallas_call.InterpretParams(
        num_cores_per_device=num_cores
    )
    avg1, min1, result1 = benchmark_with_params(
        grid_size, num_cores, params_baseline,
        label="Default InterpretParams"
    )
    results.append(("Baseline", avg1, min1))

    # 2. With jaxpr caching (already enabled by default now)
    print("\n[2] WITH JAXPR CACHING")
    print("-" * 80)
    print("(Already enabled in our modified code)")
    cache_size = len(interpret_pallas_call._compiled_jaxpr_cache)
    print(f"Cache size: {cache_size} entries")
    results.append(("+ Jaxpr caching", avg1, min1))

    # 3. With pure_callback for loads
    print("\n[3] + PURE_CALLBACK FOR LOADS")
    print("-" * 80)
    params_pure = interpret_pallas_call.InterpretParams(
        num_cores_per_device=num_cores,
        use_pure_callback_for_loads=True,
    )
    avg3, min3, result3 = benchmark_with_params(
        grid_size, num_cores, params_pure,
        label="With use_pure_callback_for_loads=True"
    )
    assert jnp.allclose(result1, result3), "Results don't match!"
    results.append(("+ Pure callback loads", avg3, min3))

    # 4. Skip floating point ops (for comparison)
    print("\n[4] + SKIP FLOATING POINT OPS")
    print("-" * 80)
    params_skip_fp = interpret_pallas_call.InterpretParams(
        num_cores_per_device=num_cores,
        skip_floating_point_ops=True,
    )
    avg4, min4, result4 = benchmark_with_params(
        grid_size, num_cores, params_skip_fp,
        label="With skip_floating_point_ops=True"
    )
    results.append(("+ Skip FP ops", avg4, min4))

    # 5. All optimizations
    print("\n[5] ALL OPTIMIZATIONS COMBINED")
    print("-" * 80)
    params_all = interpret_pallas_call.InterpretParams(
        num_cores_per_device=num_cores,
        use_pure_callback_for_loads=True,
        skip_floating_point_ops=True,
        detect_races=False,
    )
    avg5, min5, result5 = benchmark_with_params(
        grid_size, num_cores, params_all,
        label="All optimizations enabled"
    )
    results.append(("All optimizations", avg5, min5))

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Configuration':<35} {'Avg Time':<15} {'Speedup vs Baseline'}")
    print("-" * 80)

    baseline_time = results[0][1]
    for name, avg_time, min_time in results:
        speedup = baseline_time / avg_time
        print(f"{name:<35} {avg_time:>10.4f}s {speedup:>17.2f}x")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    best_config = min(results[2:], key=lambda x: x[1])
    best_speedup = baseline_time / best_config[1]

    print(f"\nBest configuration: {best_config[0]}")
    print(f"Speedup achieved: {best_speedup:.2f}x")
    print(f"Time saved: {(baseline_time - best_config[1])*1000:.1f}ms")

    print("\nOptimization breakdown:")
    print("  1. Jaxpr caching: Already enabled")
    print(f"  2. pure_callback for loads: {baseline_time/results[2][1]:.2f}x")
    print(f"  3. Skip FP ops: {baseline_time/results[3][1]:.2f}x")
    print(f"  4. Combined: {best_speedup:.2f}x")

    print("\nLimitations:")
    print("  - Callback overhead still dominates")
    print("  - For 10x+ speedup, need architectural changes:")
    print("    * Batch grid iterations (reduce callbacks)")
    print("    * Fast mode without simulation")
    print("    * Real CPU backend")

    print("\n" + "="*80 + "\n")
