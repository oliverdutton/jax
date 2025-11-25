#!/usr/bin/env python3
"""
Demonstrate "fast mode" optimization where we skip simulation entirely
and execute kernel logic directly using JAX's native operations.
"""

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

def benchmark_interpret_mode(grid_size, num_cores=2):
    """Benchmark standard interpret mode with all optimizations."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    x = jnp.arange(grid_size, dtype=jnp.float32)
    y = jnp.arange(grid_size, dtype=jnp.float32) * 2.0

    params = interpret_pallas_call.InterpretParams(
        num_cores_per_device=num_cores,
        use_pure_callback_for_loads=True,
        skip_floating_point_ops=False,  # Keep computation for fair comparison
    )

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
            interpret=params,
        )(x, y)

    # Warmup
    _ = pallas_add(x, y).block_until_ready()

    # Time
    times = []
    for _ in range(5):
        start = time.time()
        result = pallas_add(x, y).block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)

    return sum(times) / len(times), result

def benchmark_fast_mode(grid_size):
    """Benchmark with 'fast mode' - direct execution without simulation."""
    x = jnp.arange(grid_size, dtype=jnp.float32)
    y = jnp.arange(grid_size, dtype=jnp.float32) * 2.0

    @jax.jit
    def fast_kernel(x, y):
        """Execute kernel logic directly - no grid iteration, no callbacks."""
        result = x + y
        result = result * 2.0
        result = jnp.sin(result)
        return result

    # Warmup
    _ = fast_kernel(x, y).block_until_ready()

    # Time
    times = []
    for _ in range(5):
        start = time.time()
        result = fast_kernel(x, y).block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)

    return sum(times) / len(times), result

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*15 + "INTERPRET MODE vs FAST MODE (DIRECT EXECUTION)")
    print("="*80)

    grid_size = 512  # 2**9

    print(f"\nConfiguration: grid_size={grid_size}")

    # Test 1: Optimized interpret mode
    print("\n[1] OPTIMIZED INTERPRET MODE")
    print("-" * 80)
    print("(With jaxpr caching + pure_callback for loads)")
    time_interpret, result_interpret = benchmark_interpret_mode(grid_size)
    print(f"  Average time: {time_interpret:.4f}s")

    # Test 2: Fast mode (direct execution)
    print("\n[2] FAST MODE (Direct Execution)")
    print("-" * 80)
    print("(No grid iteration, no callbacks, no simulation)")
    time_fast, result_fast = benchmark_fast_mode(grid_size)
    print(f"  Average time: {time_fast:.6f}s")

    # Verify correctness
    assert jnp.allclose(result_interpret, result_fast), "Results don't match!"
    print("  ✓ Results match")

    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    speedup = time_interpret / time_fast

    print(f"\nOptimized interpret mode:  {time_interpret:.4f}s")
    print(f"Fast mode (direct):        {time_fast:.6f}s")
    print(f"\nSpeedup: {speedup:.0f}x")
    print(f"Time saved: {(time_interpret - time_fast)*1000:.1f}ms ({(1 - time_fast/time_interpret)*100:.1f}%)")

    print("\n" + "="*80)
    print("IMPLEMENTATION STRATEGY")
    print("="*80)

    print("\nTo achieve this speedup in actual Pallas code, we would need to:")
    print("\n1. Add FastInterpretParams flag:")
    print("   ```python")
    print("   fast_mode: bool = False")
    print("   ```")

    print("\n2. In interpret_pallas_call(), detect fast_mode:")
    print("   ```python")
    print("   if interpret_params.fast_mode:")
    print("       return _fast_execute_kernel(jaxpr, inputs, grid_mapping)")
    print("   ```")

    print("\n3. Implement _fast_execute_kernel():")
    print("   ```python")
    print("   def _fast_execute_kernel(jaxpr, inputs, grid_mapping):")
    print("       # Map grid iteration to vectorized JAX operation")
    print("       # Execute jaxpr directly without memory simulation")
    print("       # Process all grid points at once")
    print("       return jax.vmap(lambda idx: eval_jaxpr(...))(grid_indices)")
    print("   ```")

    print("\nTradeoffs:")
    print("  ✓ Gains: {:.0f}x faster execution".format(speedup))
    print("  ✓ Maintains correctness (kernel logic unchanged)")
    print("  ✗ Loses: Memory layout testing")
    print("  ✗ Loses: Race detection")
    print("  ✗ Loses: TPU-specific behavior simulation")

    print("\nUse cases:")
    print("  - Fast mode: Performance testing, kernel logic validation")
    print("  - Interpret mode: Debugging races, memory patterns, TPU behavior")

    print("\n" + "="*80)
    print(f"✓ Fast mode is {speedup:.0f}x faster than optimized interpret mode")
    print("="*80 + "\n")
