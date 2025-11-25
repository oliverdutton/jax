#!/usr/bin/env python3
"""Benchmark Pallas TPU interpret mode performance before and after optimizations."""

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

def run_benchmark(grid_size, num_runs=3):
    """Run benchmark with given grid size."""
    print(f"\n{'='*60}")
    print(f"Benchmarking with grid size: {grid_size}")
    print(f"{'='*60}")

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

    # Test with interpret mode
    print("\nTesting with CURRENT interpret mode...")

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

    baseline_time = sum(times) / len(times)
    print(f"\nAverage time (CURRENT): {baseline_time:.4f}s")

    # Verify correctness
    expected = jnp.sin((x + y) * 2.0)
    if not jnp.allclose(result, expected):
        print(f"ERROR: Result mismatch!")
        print(f"Expected: {expected[:5]}")
        print(f"Got: {result[:5]}")
        raise AssertionError("Result mismatch!")
    print("✓ Correctness check passed")

    return baseline_time, result

if __name__ == "__main__":
    print("Pallas TPU Interpret Mode Performance Benchmark")
    print("=" * 60)

    # Start with small grid sizes
    # Each grid iteration processes one element
    grid_sizes = [10, 20, 50, 100]

    best_size = grid_sizes[0]
    for grid_size in grid_sizes:
        baseline_time, _ = run_benchmark(grid_size, num_runs=3)
        best_size = grid_size

        # Stop if it takes too long
        if baseline_time > 20:
            print(f"\n⚠ Stopping - baseline time {baseline_time:.2f}s is too long")
            print(f"Will use grid_size={best_size} for optimization comparison")
            break

    print("\n" + "="*60)
    print("Baseline benchmarking complete!")
    print(f"Selected grid_size for comparison: {best_size}")
    print("="*60)
