#!/usr/bin/env python3
"""Compare Pallas TPU interpret mode performance before and after optimizations."""

import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Simple kernel: element-wise addition with some computation
def add_kernel(x_ref, y_ref, o_ref):
    """Simple kernel that adds two arrays with some extra computation."""
    # Load values
    x = x_ref[0]
    y = y_ref[0]
    # Do some computation to make it non-trivial
    result = x + y
    result = result * 2.0
    result = jnp.sin(result)
    # Store result
    o_ref[0] = result

def run_benchmark(grid_size, num_runs=5, label=""):
    """Run benchmark with given grid size."""
    print(f"\n{label}")
    print("-" * 60)

    # Create inputs
    x = jnp.arange(grid_size, dtype=jnp.float32)
    y = jnp.arange(grid_size, dtype=jnp.float32) * 2.0

    # Import interpret params
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    interpret_params = interpret_pallas_call.InterpretParams()

    # Define the pallas function
    def pallas_add_interpret(x, y):
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
    print("  Warmup...")
    _ = pallas_add_interpret(x, y).block_until_ready()

    # Time it
    times = []
    for i in range(num_runs):
        print(f"  Run {i+1}...", end=" ", flush=True)
        start = time.time()
        result = pallas_add_interpret(x, y).block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"{elapsed:.4f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(f"\n  Average: {avg_time:.4f}s")
    print(f"  Min:     {min_time:.4f}s")
    print(f"  Max:     {max_time:.4f}s")

    # Verify correctness
    expected = jnp.sin((x + y) * 2.0)
    if not jnp.allclose(result, expected):
        print(f"  ERROR: Result mismatch!")
        raise AssertionError("Result mismatch!")
    print("  ✓ Correctness check passed")

    return avg_time, result

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Pallas TPU Interpret Mode: Optimization Comparison")
    print("="*60)

    # Use a reasonable grid size that completes in reasonable time
    grid_size = 100

    print(f"\nTesting with grid_size={grid_size}")
    print("Each grid iteration processes 1 element")
    print(f"Total elements: {grid_size}")

    # Run with OPTIMIZED version (cache is enabled by default now)
    print("\n" + "="*60)
    print("RUNNING WITH OPTIMIZATION (jaxpr caching)")
    print("="*60)
    optimized_time, _ = run_benchmark(
        grid_size, num_runs=5,
        label="Testing with OPTIMIZED interpret mode (with jaxpr caching)"
    )

    # Clear cache and run again to simulate "before" state
    print("\n" + "="*60)
    print("SIMULATING BASELINE (clearing cache between runs)")
    print("="*60)
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    # Clear the cache to simulate baseline behavior
    print("Clearing compilation cache to simulate baseline...")
    interpret_pallas_call._compiled_jaxpr_cache.clear()

    # Run once to get a baseline measurement without caching
    print("\nRunning single baseline measurement...")
    x = jnp.arange(grid_size, dtype=jnp.float32)
    y = jnp.arange(grid_size, dtype=jnp.float32) * 2.0

    interpret_params = interpret_pallas_call.InterpretParams()

    def pallas_add_no_cache(x, y):
        # Clear cache before each run to simulate no caching
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

    # Measure without cache (just 2 runs since it's slow)
    baseline_times = []
    for i in range(2):
        print(f"  Baseline run {i+1}...", end=" ", flush=True)
        start = time.time()
        _ = pallas_add_no_cache(x, y).block_until_ready()
        elapsed = time.time() - start
        baseline_times.append(elapsed)
        print(f"{elapsed:.4f}s")

    baseline_time = sum(baseline_times) / len(baseline_times)
    print(f"\n  Baseline average: {baseline_time:.4f}s")

    # Calculate speedup
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nBaseline (no caching):     {baseline_time:.4f}s")
    print(f"Optimized (with caching):  {optimized_time:.4f}s")
    print(f"\nSpeedup: {baseline_time / optimized_time:.2f}x")
    print(f"Time saved: {(baseline_time - optimized_time):.4f}s ({(1 - optimized_time/baseline_time)*100:.1f}%)")

    print("\n" + "="*60)
    print("OPTIMIZATION DETAILS")
    print("="*60)
    print("\nWhat was optimized:")
    print("  - Added compilation caching for kernel jaxprs")
    print("  - Prevents recompilation on each pallas_call invocation")
    print("  - Thread-safe cache with locking mechanism")
    print("\nCode location:")
    print("  File: jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py")
    print("  Lines: 1900-1921 (_run_jaxpr function)")
    print("\nThis addresses the TODO comment at line 1932-1935:")
    print('  "Would it be worth trying to lower/compile the jaxpr at')
    print('   lowering/compilation time?"')
    print("\n" + "="*60)
