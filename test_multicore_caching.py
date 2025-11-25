#!/usr/bin/env python3
"""Test jaxpr caching with multi-core execution (where _run_jaxpr is used)."""

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

def test_with_multicore(num_cores=2, grid_size=100):
    """Test with num_cores_per_device > 1 to trigger thread_map."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    # Use multiple cores to trigger _thread_map code path
    interpret_params = interpret_pallas_call.InterpretParams(
        num_cores_per_device=num_cores
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

    print(f"\nTesting with num_cores_per_device={num_cores}, grid_size={grid_size}")
    print("-" * 70)

    # Check cache usage
    print(f"Initial cache size: {len(interpret_pallas_call._compiled_jaxpr_cache)}")

    # First call
    print("\nFirst call...")
    start = time.time()
    result1 = pallas_add(x, y).block_until_ready()
    time1 = time.time() - start
    print(f"  Time: {time1:.4f}s")
    print(f"  Cache size: {len(interpret_pallas_call._compiled_jaxpr_cache)}")

    # Second call (should use cache)
    print("\nSecond call (should use cache)...")
    start = time.time()
    result2 = pallas_add(x, y).block_until_ready()
    time2 = time.time() - start
    print(f"  Time: {time2:.4f}s")
    print(f"  Cache size: {len(interpret_pallas_call._compiled_jaxpr_cache)}")

    # Test with cache cleared
    print("\nClearing cache and running again...")
    interpret_pallas_call._compiled_jaxpr_cache.clear()
    start = time.time()
    result3 = pallas_add(x, y).block_until_ready()
    time3 = time.time() - start
    print(f"  Time: {time3:.4f}s")
    print(f"  Cache size: {len(interpret_pallas_call._compiled_jaxpr_cache)}")

    return time1, time2, time3

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING JAXPR CACHING WITH MULTI-CORE EXECUTION")
    print("="*70)

    print("\nNote: _run_jaxpr (where our cache lives) is only called when")
    print("num_cores_per_device > 1, which triggers _thread_map.")

    # Test with different configurations
    configs = [
        (1, 100),   # Single core (cache won't be used)
        (2, 100),   # Multi-core (cache WILL be used)
        (4, 100),   # More cores (cache WILL be used)
    ]

    results = []
    for num_cores, grid_size in configs:
        time1, time2, time3 = test_with_multicore(num_cores, grid_size)
        results.append((num_cores, time1, time2, time3))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Cores':<8} {'First Call':<15} {'Cached Call':<15} {'No Cache':<15}")
    print("-" * 70)
    for num_cores, time1, time2, time3 in results:
        print(f"{num_cores:<8} {time1:>10.4f}s {time2:>10.4f}s {time3:>10.4f}s")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    cache_size = len(interpret_pallas_call._compiled_jaxpr_cache)

    if cache_size > 0:
        print(f"\n✓ Cache IS being used! ({cache_size} entries)")
        print("\nThe caching optimization applies when num_cores_per_device > 1.")
        print("For single-core kernels, JAX's internal caching handles it.")
    else:
        print("\n⚠ Cache is NOT being used.")
        print("\nThis might be because:")
        print("  1. JAX's internal caching prevents recompilation")
        print("  2. The test setup doesn't trigger _thread_map")

    print("\n" + "="*70 + "\n")
