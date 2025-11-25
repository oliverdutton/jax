#!/usr/bin/env python3
"""Test a fast vectorized execution mode for Pallas interpret."""

import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    """Simple kernel - element-wise operation."""
    x = x_ref[0]
    y = y_ref[0]
    result = x + y
    result = result * 2.0
    result = jnp.sin(result)
    o_ref[0] = result

def run_pallas_interpret(grid_size):
    """Run with standard interpret mode."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    interpret_params = interpret_pallas_call.InterpretParams()

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

    # Time it (5 runs)
    times = []
    for _ in range(5):
        start = time.time()
        result = pallas_add(x, y).block_until_ready()
        times.append(time.time() - start)

    return sum(times) / len(times), result

def run_vectorized(grid_size):
    """Run with vectorized execution (simulating fast mode)."""
    x = jnp.arange(grid_size, dtype=jnp.float32)
    y = jnp.arange(grid_size, dtype=jnp.float32) * 2.0

    @jax.jit
    def vectorized_kernel(x, y):
        """Directly execute the kernel logic without interpretation."""
        result = x + y
        result = result * 2.0
        result = jnp.sin(result)
        return result

    # Warmup
    _ = vectorized_kernel(x, y).block_until_ready()

    # Time it (5 runs)
    times = []
    for _ in range(5):
        start = time.time()
        result = vectorized_kernel(x, y).block_until_ready()
        times.append(time.time() - start)

    return sum(times) / len(times), result

if __name__ == "__main__":
    grid_size = 100

    print("\n" + "="*60)
    print("Pallas Interpret vs Vectorized Fast Mode")
    print("="*60)
    print(f"\nGrid size: {grid_size}")
    print("="*60)

    # Standard interpret mode
    print("\n1. CURRENT: Standard interpret mode")
    time_interpret, result_interpret = run_pallas_interpret(grid_size)
    print(f"   Average time: {time_interpret:.4f}s")

    # Vectorized fast mode
    print("\n2. PROPOSED: Vectorized fast mode (skips simulation)")
    time_vectorized, result_vectorized = run_vectorized(grid_size)
    print(f"   Average time: {time_vectorized:.6f}s")

    # Verify correctness
    assert jnp.allclose(result_interpret, result_vectorized), "Results don't match!"

    # Calculate speedup
    speedup = time_interpret / time_vectorized
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nStandard interpret mode:  {time_interpret:.4f}s")
    print(f"Vectorized fast mode:     {time_vectorized:.6f}s")
    print(f"\nSpeedup: {speedup:.1f}x")
    print(f"Time saved: {(time_interpret - time_vectorized)*1000:.1f}ms")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print("\nWhy is interpret mode slow?")
    print(f"  - {grid_size} grid iterations")
    print(f"  - ~10+ io_callbacks per iteration")
    print(f"  - Total: ~{grid_size * 10}+ synchronous callbacks")
    print("  - Each callback has Python overhead")
    print("  - Simulates TPU memory hierarchy")
    print("\nWhy is vectorized mode fast?")
    print("  - No grid iteration overhead")
    print("  - Single JIT-compiled function")
    print("  - No callbacks - direct computation")
    print("  - Fully optimized by XLA")
    print("\nTradeoff:")
    print("  - Interpret mode: Good for debugging, race detection")
    print("  - Fast mode: Good for performance testing, correctness")
    print("\n" + "="*60)
