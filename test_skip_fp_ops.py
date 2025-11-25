#!/usr/bin/env python3
"""Test skip_floating_point_ops optimization."""

import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    """Simple kernel with floating point operations."""
    x = x_ref[0]
    y = y_ref[0]
    result = x + y
    result = result * 2.0
    result = jnp.sin(result)
    o_ref[0] = result

def run_test(grid_size, skip_fp=False):
    """Run test with or without skip_floating_point_ops."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    interpret_params = interpret_pallas_call.InterpretParams(
        skip_floating_point_ops=skip_fp
    )

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

    # Time it
    start = time.time()
    result = pallas_add(x, y).block_until_ready()
    elapsed = time.time() - start

    return elapsed, result

if __name__ == "__main__":
    grid_size = 100

    print("\n" + "="*60)
    print("Testing skip_floating_point_ops optimization")
    print("="*60)

    print(f"\nGrid size: {grid_size}")

    # Without optimization
    print("\n1. Without skip_floating_point_ops:")
    time_normal, result_normal = run_test(grid_size, skip_fp=False)
    print(f"   Time: {time_normal:.4f}s")

    # With optimization
    print("\n2. With skip_floating_point_ops=True:")
    time_skip, result_skip = run_test(grid_size, skip_fp=True)
    print(f"   Time: {time_skip:.4f}s")

    print(f"\n   Speedup: {time_normal / time_skip:.2f}x")
    print(f"   Time saved: {(time_normal - time_skip)*1000:.1f}ms")

    print("\nNote: This optimization skips FP computation, useful for")
    print("      testing control flow but not for actual kernel execution.")
