#!/usr/bin/env python3
"""Test to see if our cache is actually being used."""

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

def test_cache():
    """Test if cache is being used."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    interpret_params = interpret_pallas_call.InterpretParams()

    grid_size = 512
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

    print("Initial cache size:", len(interpret_pallas_call._compiled_jaxpr_cache))

    # Run once
    print("\nFirst call...")
    result1 = pallas_add(x, y).block_until_ready()
    print("Cache size after first call:", len(interpret_pallas_call._compiled_jaxpr_cache))

    # Run again
    print("\nSecond call...")
    result2 = pallas_add(x, y).block_until_ready()
    print("Cache size after second call:", len(interpret_pallas_call._compiled_jaxpr_cache))

    # Check if JAX is already caching
    print("\n" + "="*60)
    print("TESTING JAX'S INTERNAL CACHING")
    print("="*60)

    # Clear our cache and see if it gets repopulated
    interpret_pallas_call._compiled_jaxpr_cache.clear()
    print("\nCleared our cache. Size:", len(interpret_pallas_call._compiled_jaxpr_cache))

    print("\nRunning with cleared cache...")
    start = time.time()
    result3 = pallas_add(x, y).block_until_ready()
    time_cleared = time.time() - start
    print(f"Time: {time_cleared:.4f}s")
    print("Cache size:", len(interpret_pallas_call._compiled_jaxpr_cache))

    print("\nRunning again (should use JAX's cache)...")
    start = time.time()
    result4 = pallas_add(x, y).block_until_ready()
    time_cached = time.time() - start
    print(f"Time: {time_cached:.4f}s")

    print(f"\nTime difference: {abs(time_cleared - time_cached)*1000:.1f}ms")

    if len(interpret_pallas_call._compiled_jaxpr_cache) == 0:
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        print("\nOur cache is NOT being used for this kernel.")
        print("This means:")
        print("  1. The code path doesn't go through _run_jaxpr")
        print("  2. OR JAX's internal caching prevents recompilation")
        print("  3. OR this kernel doesn't use thread_map")
        print("\nThe optimization helps for kernels that DO use thread_map,")
        print("such as more complex kernels with num_cores_per_device > 1.")
    else:
        print(f"\nCache is being used! ({len(interpret_pallas_call._compiled_jaxpr_cache)} entries)")

if __name__ == "__main__":
    test_cache()
