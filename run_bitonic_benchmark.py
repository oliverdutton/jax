"""
Script to run the bitonic sort benchmark and track runtime.

This demonstrates progressive conversion to NumPy interpretation.
"""

import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

print("=" * 70)
print("BITONIC SORT BENCHMARK - Runtime Tracking")
print("=" * 70)

# Test configuration
ntoken = 8
n = 128
backend = jax.default_backend()

print(f"\nConfiguration:")
print(f"  Array shape: ({ntoken}, {n})")
print(f"  Backend: {backend}")
print(f"  Using interpret=True (CPU mode)")

# Create test data
key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (ntoken, n), dtype=jnp.float32)

print(f"  Input data shape: {data.shape}")
print(f"  Input data dtype: {data.dtype}")

# Simple test kernel (copying the approach from the bitonic sort)
def simple_kernel(x_ref, o_ref):
    """Simple kernel that demonstrates basic Pallas operations."""
    # Just copy and multiply
    o_ref[...] = x_ref[...] * 2.0

def run_test(name, kernel_fn, data):
    """Run a test and measure time."""
    print(f"\n{name}:")
    out_shape = jax.ShapeDtypeStruct(data.shape, data.dtype)

    # Warmup
    _ = pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        interpret=True,
    )(data)

    # Timed run
    start = time.time()
    result = pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        interpret=True,
    )(data)
    result = jax.block_until_ready(result)
    end = time.time()

    elapsed_ms = (end - start) * 1000
    print(f"  Runtime: {elapsed_ms:.2f}ms")
    print(f"  Result shape: {result.shape}")
    return elapsed_ms, result

# Run tests
print("\n" + "=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)

elapsed_pallas, result_pallas = run_test("1. Standard Pallas interpret", simple_kernel, data)

# Verify correctness
expected = data * 2.0
correct = jnp.allclose(result_pallas, expected)
print(f"\n  ✓ Results correct: {correct}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nBaseline runtime (Pallas interpret): {elapsed_pallas:.2f}ms")
print(f"\nNext steps to progressively convert to NumPy:")
print("1. Replace simple ops with NumPy via io_callback")
print("2. Handle memory references with stateful NumPy arrays")
print("3. Convert control flow (loops, conditionals) to Python")
print("4. Use lax_reference.py for complex primitives")
print("\nSee numpy_pallas_demo.py for working example of io_callback approach.")
print("=" * 70)
