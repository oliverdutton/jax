#!/usr/bin/env python3
"""
Test if we can achieve selective indexing using ONLY:
- jax.random.bits()
- jax.random.uniform()
- jax.random.key()
- jax.random.fold_in()
- jax.random.wrap_key_data()
"""

import jax
import jax.numpy as jnp
import jax.random as random

print("=" * 80)
print("Testing Selective Indexing with Public JAX APIs")
print("=" * 80)

# Generate reference
key = random.key(42)
full_array = random.uniform(key, (128,))
target_indices = [7, 9]

print(f"\nReference values from full array:")
print(f"  full_array[7] = {full_array[7]:.10f}")
print(f"  full_array[9] = {full_array[9]:.10f}")

print("\n" + "=" * 80)
print("Approach 1: fold_in with index")
print("=" * 80)

def try_fold_in(key, index):
    """Try folding in the index to get value at that position."""
    indexed_key = random.fold_in(key, index)
    return random.uniform(indexed_key, shape=())

val_7 = try_fold_in(key, 7)
val_9 = try_fold_in(key, 9)

print(f"fold_in(key, 7) then uniform(shape=()):")
print(f"  Result: {val_7:.10f}")
print(f"  Expected: {full_array[7]:.10f}")
print(f"  Match: {jnp.allclose(val_7, full_array[7])}")

print(f"\nfold_in(key, 9) then uniform(shape=()):")
print(f"  Result: {val_9:.10f}")
print(f"  Expected: {full_array[9]:.10f}")
print(f"  Match: {jnp.allclose(val_9, full_array[9])}")

print("\n" + "=" * 80)
print("Approach 2: Investigate wrap_key_data")
print("=" * 80)

# Get the key data
key_data = random.key_data(key)
print(f"\nOriginal key data: {key_data}")
print(f"Shape: {key_data.shape}, dtype: {key_data.dtype}")

# Try wrapping the same data
wrapped = random.wrap_key_data(key_data)
print(f"\nWrapped key equals original: {jnp.array_equal(random.key_data(wrapped), key_data)}")

# Test if wrapped key gives same results
test_val = random.uniform(wrapped, shape=())
original_val = random.uniform(key, shape=())
print(f"uniform(wrapped) = {test_val:.10f}")
print(f"uniform(original) = {original_val:.10f}")
print(f"Match: {jnp.allclose(test_val, original_val)}")

print("\n" + "=" * 80)
print("Approach 3: Can we modify key_data to encode index?")
print("=" * 80)

# For threefry, the key is shape (2,) of uint32
# The counter for index i is (0, i) in the threefry2x32 call
# But the KEY is separate from the counter

# What if we try to construct key data that includes the index?
# This doesn't make sense because key != counter

print("""
The key insight is that threefry separates KEY from COUNTER:
  - KEY: The PRNG key (what we have)
  - COUNTER: Position-dependent value (0, index)

wrap_key_data() wraps KEY data, not counter data.
The counter is determined internally by the shape parameter.

So we cannot control the counter through wrap_key_data.
""")

print("\n" + "=" * 80)
print("Approach 4: Using bits() instead of uniform()")
print("=" * 80)

# Try with bits
val_7_bits = random.bits(random.fold_in(key, 7), shape=(), dtype=jnp.uint32)
val_9_bits = random.bits(random.fold_in(key, 9), shape=(), dtype=jnp.uint32)

print(f"bits(fold_in(key, 7)): {val_7_bits}")
print(f"bits(fold_in(key, 9)): {val_9_bits}")
print("\nBut these are raw bits, not uniform floats, and still don't match")
print("the sequential generation pattern.")

print("\n" + "=" * 80)
print("Approach 5: Deep dive into fold_in semantics")
print("=" * 80)

print("""
Let's understand what fold_in actually does:

From jax/_src/prng.py, threefry_fold_in:
  fold_in(key, data) = threefry_2x32(key, threefry_seed(data))

Where threefry_seed(data) converts data to a key pair [hi, lo].

For sequential generation (uniform with shape):
  value[i] = threefry_2x32(key, counter_i)
  where counter_i = (0, i) for small arrays

For fold_in:
  fold_in(key, i) = threefry_2x32(key, threefry_seed(i))
  threefry_seed(i) = [i >> 32, i & 0xFFFFFFFF] = [0, i] for i < 2^32

So theoretically they SHOULD be the same!

But there's a subtle difference:
  - Sequential: applies hash once with counter (0, i)
  - fold_in: creates a NEW KEY from hash(key, seed(i)),
             then that new key is used for generation

This is the key difference!
""")

print("\n" + "=" * 80)
print("Testing the Theory")
print("=" * 80)

# Sequential generation is:
# bits = threefry_2x32(key, (0, index))
# uniform_value = convert_to_float(bits)

# fold_in is:
# new_key = threefry_2x32(key, threefry_seed(index))
# Then we call uniform(new_key, shape=())
# Which does: bits = threefry_2x32(new_key, (0, 0))  # counter for shape=()
# uniform_value = convert_to_float(bits)

print("""
FOUND THE ISSUE!

Sequential generation for index i:
  1. hash(original_key, counter=(0, i)) -> bits
  2. convert bits to uniform float

fold_in approach:
  1. hash(original_key, seed(i)) -> new_key
  2. hash(new_key, counter=(0, 0)) -> bits  [because shape=()]
  3. convert bits to uniform float

These are DIFFERENT because:
  - Sequential: ONE hash with counter=i
  - fold_in: TWO hashes, first creates new key, second uses counter=0
""")

print("\n" + "=" * 80)
print("Can we use bits() to get closer?")
print("=" * 80)

# What if we generate bits for a 1D array and look at position 0?
key_folded_7 = random.fold_in(key, 7)
bits_array = random.bits(key_folded_7, shape=(1,), dtype=jnp.uint32)
print(f"\nbits(fold_in(key, 7), shape=(1,)): {bits_array}")
print("This gives us one element, but it's from counter=0 of the folded key")
print("Still not the same as sequential generation with counter=7")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"""
Using ONLY the public APIs you listed:
  - jax.random.bits()
  - jax.random.uniform()
  - jax.random.key()
  - jax.random.fold_in()
  - jax.random.wrap_key_data()

Answer: NO, you CANNOT achieve true selective indexing that matches
sequential generation exactly.

Reason:
  - Sequential generation: hash(key, counter=index) -> value
  - All public APIs lead to: hash(modified_key, counter=0) -> value
  - These produce different values

The fundamental issue is that PUBLIC APIs don't expose direct control
over the COUNTER values used in the hash function.

BEST YOU CAN DO with public APIs:
  Generate array up to max(indices) and index into it:

  ```python
  indices = [7, 9]
  array = random.uniform(key, (max(indices) + 1,))
  values = array[jnp.array(indices)]
  ```

This gives exact values with minimal unnecessary generation.

For TRUE sparse selective indexing (e.g., 2 values from 1M array),
you would need either:
  1. Internal APIs (threefry2x32_p.bind with custom counters)
  2. A new public API in JAX for counter-based generation
  3. Accept generating up to max(indices)
""")

print("\n" + "=" * 80)
print("Verification of Best Public API Approach")
print("=" * 80)

# Best approach with public APIs
indices = jnp.array([7, 9])
efficient_array = random.uniform(key, (jnp.max(indices) + 1,))
efficient_values = efficient_array[indices]

print(f"\nGenerate array of shape ({jnp.max(indices) + 1},) instead of (128,)")
print(f"Extract indices {indices.tolist()}")
print(f"\nResults:")
print(f"  Value at 7: {efficient_values[0]:.10f} (expected: {full_array[7]:.10f})")
print(f"  Value at 9: {efficient_values[1]:.10f} (expected: {full_array[9]:.10f})")
print(f"  Match: {jnp.allclose(efficient_values, jnp.array([full_array[7], full_array[9]]))}")

savings = (128 - (jnp.max(indices) + 1)) / 128 * 100
print(f"\nMemory savings: {savings:.1f}% ({128} -> {jnp.max(indices) + 1} elements)")
