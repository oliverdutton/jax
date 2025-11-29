#!/usr/bin/env python3
"""
JAX Threefry 2x32 Algorithm and Selective Indexing Demonstration

This script demonstrates:
1. How the threefry 2x32 PRNG algorithm works in JAX
2. How to efficiently compute random values for specific indices only
3. Verification that selective indexing produces identical values
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax._src.prng import threefry2x32_p
from jax._src import dtypes as jax_dtypes

print("=" * 80)
print("JAX THREEFRY 2X32 ALGORITHM - How It Works")
print("=" * 80)

print("""
THREEFRY is a counter-based PRNG (Pseudo-Random Number Generator):

KEY COMPONENTS:
1. Key: A pair of uint32 values [k1, k2]
2. Counter: Position-dependent value(s)
3. Hash Function: Threefry2x32 applies mixing rounds with rotations, XORs, additions

ALGORITHM FOR uniform(key, (N,)):
1. Extract key components: k1, k2 = key[0], key[1]
2. Create counter pairs for each position i in 0..N-1:
   - counter_hi = 0 (for small arrays)
   - counter_lo = i (the index)
3. Apply hash: bits1, bits2 = threefry2x32(k1, k2, counter_hi, counter_lo)
4. Combine: bits = bits1 XOR bits2
5. Convert to float [0,1):
   a) Right-shift to keep mantissa bits (23 bits for float32)
   b) OR with exponent bits to represent 1.0
   c) Bitcast to float and subtract 1.0

KEY INSIGHT:
Each output position depends ONLY on (key, counter).
The counter for position i is simply (0, i) for arrays with < 2^32 elements.
This means we can compute ANY index independently!
""")

print("=" * 80)
print("DEMONSTRATION: Full vs Selective Indexing")
print("=" * 80)

# Create a PRNG key
seed = 42
key = random.key(seed)
key_data = random.key_data(key)
print(f"\nSeed: {seed}")
print(f"Key data (uint32 pair): {key_data}")

# Generate full array
array_size = 128
target_indices = [7, 9]

print(f"\nGenerating full array of shape ({array_size},)...")
full_array = random.uniform(key, (array_size,))
print(f"Full array computed: {array_size} float32 values (~{array_size * 4} bytes)")
print(f"\nValues at target indices {target_indices}:")
for idx in target_indices:
    print(f"  [{idx}] = {full_array[idx]:.10f}")


def selective_uniform_f32(key, indices):
    """
    Compute uniform random float32 values for specific indices only.

    This produces EXACTLY the same values as:
        random.uniform(key, (max(indices)+1,))[indices]

    But only computes the requested indices, saving memory and computation.

    Args:
        key: PRNG key
        indices: List or array of indices to compute

    Returns:
        Array of uniform random float32 values for the specified indices
    """
    # Extract key components
    key_data = random.key_data(key)
    k1, k2 = key_data[0], key_data[1]

    # Create counter values
    # For arrays with < 2^32 elements:
    #   counter_hi = 0
    #   counter_lo = index
    indices_arr = jnp.array(indices, dtype=jnp.uint32)
    counter_hi = jnp.zeros_like(indices_arr)
    counter_lo = indices_arr

    # Apply threefry hash
    bits1, bits2 = threefry2x32_p.bind(k1, k2, counter_hi, counter_lo)

    # XOR to combine (as done in threefry_random_bits partitionable mode)
    bits = bits1 ^ bits2

    # Convert uint32 bits to float32 in range [0, 1)
    # This matches the algorithm in _uniform() from jax/_src/random.py

    dtype = jnp.float32
    finfo = jax_dtypes.finfo(dtype)
    nbits, nmant = finfo.bits, finfo.nmant  # 32 bits, 23 mantissa bits
    uint_dtype = np.dtype('uint32')

    # Right-shift to keep only mantissa bits
    float_bits = jax.lax.shift_right_logical(
        bits, jnp.array(nbits - nmant, uint_dtype))

    # OR with exponent bits representing 1.0
    float_bits = jax.lax.bitwise_or(
        float_bits,
        jnp.asarray(np.array(1.0, dtype).view(uint_dtype), dtype=uint_dtype)
    )

    # Bitcast to float and subtract 1.0 to get [0, 1)
    floats = jax.lax.bitcast_convert_type(float_bits, dtype) - jnp.array(1., dtype)

    return floats


print("\n" + "=" * 80)
print("SELECTIVE INDEXING")
print("=" * 80)

print(f"\nComputing ONLY indices {target_indices} (not the full array)...")
selective_values = selective_uniform_f32(key, target_indices)
print(f"Selective computation: {len(target_indices)} float32 values (~{len(target_indices) * 4} bytes)")
print(f"\nSelective results:")
for idx, val in zip(target_indices, selective_values):
    print(f"  [{idx}] = {val:.10f}")

print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

print(f"\nComparing selective vs full array:")
print(f"{'Index':<8} {'Selective':<15} {'Full Array':<15} {'Match':<10}")
print("-" * 60)
for idx, sel_val in zip(target_indices, selective_values):
    full_val = full_array[idx]
    match = jnp.allclose(sel_val, full_val, rtol=1e-7)
    print(f"{idx:<8} {sel_val:<15.10f} {full_val:<15.10f} {str(match):<10}")

all_match = jnp.allclose(
    selective_values,
    jnp.array([full_array[i] for i in target_indices]),
    rtol=1e-7
)
print(f"\n✓ All values match: {all_match}")

print("\n" + "=" * 80)
print("EFFICIENCY ANALYSIS")
print("=" * 80)

print("""
Memory and Computation Comparison:

Full array approach (N=128, selecting 2 indices):
  - Generate: 128 float32 values
  - Memory: 512 bytes
  - Operations: 128 × threefry_hash + 128 × bit_conversion
  - Return: Extract 2 values from 128

Selective indexing (2 indices):
  - Generate: 2 float32 values
  - Memory: 8 bytes
  - Operations: 2 × threefry_hash + 2 × bit_conversion
  - Return: 2 values directly

Speedup: ~64x less memory, ~64x fewer operations

For larger arrays (e.g., N=1,000,000, selecting 10 indices):
  - Full: 4 MB memory, 1M operations
  - Selective: 40 bytes, 10 operations
  - Speedup: ~100,000x less memory, ~100,000x fewer operations!
""")

print("=" * 80)
print("USE CASES")
print("=" * 80)

print("""
Selective indexing is valuable for:

1. **Sparse Sampling**: Need few values from large probability distributions
   Example: Sample 100 random values from indices in range [0, 1_000_000)

2. **Memory-Constrained Environments**: Can't fit full array in memory
   Example: Generate random values for specific positions in huge simulation

3. **Dynamic Indexing**: Indices determined at runtime
   Example: Random initialization of specific neural network weights

4. **Distributed Computing**: Each worker computes its assigned indices
   Example: Parallel simulation where each worker needs different random values

5. **Reproducibility with Efficiency**: Get exact same values as full array
   Example: Debugging or testing specific indices without full generation
""")

print("\n" + "=" * 80)
print("EXTENDED TEST: More Indices")
print("=" * 80)

# Test with more indices
test_indices = [0, 7, 9, 15, 31, 63, 127]
print(f"\nTesting with indices: {test_indices}")

selective_test = selective_uniform_f32(key, test_indices)
full_test = jnp.array([full_array[i] for i in test_indices])

print(f"\nResults:")
print(f"{'Index':<8} {'Selective':<15} {'Full':<15} {'Difference':<15}")
print("-" * 65)
for idx, sel, full in zip(test_indices, selective_test, full_test):
    diff = abs(sel - full)
    print(f"{idx:<8} {sel:<15.10f} {full:<15.10f} {diff:<15.2e}")

max_diff = jnp.max(jnp.abs(selective_test - full_test))
print(f"\nMaximum absolute difference: {max_diff:.2e}")
print(f"All match (rtol=1e-7): {jnp.allclose(selective_test, full_test, rtol=1e-7)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
✓ JAX's threefry 2x32 is a counter-based PRNG
✓ Each random value depends only on (key, counter_hi, counter_lo)
✓ For small arrays: counter = (0, index)
✓ Selective indexing computes EXACT same values as full array
✓ Massive efficiency gains for sparse sampling scenarios
✓ Enables scalable random number generation for large-scale applications

The counter-based nature of threefry makes it perfect for:
  - Parallel computation (no shared state needed)
  - Reproducible selective sampling
  - Memory-efficient random generation
""")
