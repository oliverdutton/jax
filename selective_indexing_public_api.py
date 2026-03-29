#!/usr/bin/env python3
"""
Selective indexing for JAX random arrays using ONLY public APIs.

This demonstrates how to compute specific random values without generating
the full array, using only jax.random, jax.lax, and jax.numpy APIs.
"""

import jax
import jax.numpy as jnp
import jax.random as random


def selective_uniform_v1(key, indices):
    """
    Compute uniform random values for specific indices using fold_in.

    This works by folding each index into the key to create a unique
    per-index key, then generating a single value.

    Args:
        key: PRNG key
        indices: List of indices to compute

    Returns:
        Array of uniform random values for the specified indices
    """
    def get_value_at_index(idx):
        # Fold index into key to get a unique key for this position
        idx_key = random.fold_in(key, idx)
        # Generate a single value (shape ())
        return random.uniform(idx_key, shape=())

    # Vectorize over indices
    return jax.vmap(get_value_at_index)(jnp.array(indices))


def selective_uniform_v2(key, indices):
    """
    Alternative approach using split to create per-index keys.

    Args:
        key: PRNG key
        indices: List of indices to compute

    Returns:
        Array of uniform random values for the specified indices
    """
    n_indices = len(indices)
    # Split key into enough subkeys
    subkeys = random.split(key, n_indices)

    # For each index, fold it in and generate
    def get_value(subkey, idx):
        idx_key = random.fold_in(subkey, idx)
        return random.uniform(idx_key, shape=())

    return jax.vmap(get_value)(subkeys, jnp.array(indices))


print("=" * 80)
print("SELECTIVE INDEXING WITH PUBLIC JAX APIs")
print("=" * 80)

print("""
Problem with previous approach:
  ✗ Used internal primitive: threefry2x32_p.bind()
  ✗ Used jax._src.dtypes (internal)
  ✗ Mixed numpy and JAX arrays

New approach using ONLY public APIs:
  ✓ jax.random.fold_in() to create per-index keys
  ✓ jax.random.uniform() for generation
  ✓ jax.vmap() for vectorization
  ✓ jax.numpy for all array operations
""")

# Test both approaches
key = random.key(42)

# Generate full array for comparison
full_array = random.uniform(key, (128,))
target_indices = [7, 9]

print("\n" + "=" * 80)
print("Testing Approach 1: fold_in")
print("=" * 80)

values_v1 = selective_uniform_v1(key, target_indices)
print(f"\nTarget indices: {target_indices}")
print(f"Selective values: {values_v1}")
print(f"Full array values: {jnp.array([full_array[i] for i in target_indices])}")

# Check if they match
match_v1 = jnp.allclose(values_v1, jnp.array([full_array[i] for i in target_indices]))
print(f"Match: {match_v1}")

print("\n" + "=" * 80)
print("Testing Approach 2: split + fold_in")
print("=" * 80)

values_v2 = selective_uniform_v2(key, target_indices)
print(f"\nTarget indices: {target_indices}")
print(f"Selective values: {values_v2}")
print(f"Full array values: {jnp.array([full_array[i] for i in target_indices])}")

match_v2 = jnp.allclose(values_v2, jnp.array([full_array[i] for i in target_indices]))
print(f"Match: {match_v2}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print(f"""
Unfortunately, neither approach produces the EXACT same values as the full array:
  - Approach 1 match: {match_v1}
  - Approach 2 match: {match_v2}

Why? Because fold_in creates a DIFFERENT key stream than sequential indexing.

The counter-based property of threefry means:
  - full_array[i] = hash(key, counter=i)

But fold_in creates:
  - fold_in(key, i) = hash(key, data=i)

These use different internal pathways in the PRNG!

To get EXACT matching values, we need to understand how JAX actually generates
the sequential array. Let's investigate...
""")

print("\n" + "=" * 80)
print("INVESTIGATION: How does uniform() actually work?")
print("=" * 80)

print("""
Looking at jax/_src/random.py, uniform(key, shape) calls:
  1. _random_bits(key, bit_width, shape)
  2. Which calls random_bits_p.bind(keys, bit_width=bit_width, shape=shape)
  3. This uses the PRNG's random_bits() implementation

For threefry with config.threefry_partitionable=True:
  - Creates counters using iota_2x32_shape(shape)
  - For shape (128,), this creates counters [0, 1, 2, ..., 127]
  - Each position i gets counter (0, i)
  - Hash: bits = threefry2x32(k1, k2, 0, i)

The issue is that there's NO public API that lets us pass specific counters!
The only way to get exact matching is to use the internal primitive.
""")

print("\n" + "=" * 80)
print("SOLUTION OPTIONS")
print("=" * 80)

print("""
Option 1: Use internal APIs (what we did before)
  ✓ Exact matching values
  ✗ Uses internal _src APIs
  ✗ Not guaranteed stable across JAX versions

Option 2: Use fold_in approach (public API)
  ✓ Uses only public APIs
  ✓ Stable across versions
  ✗ Different values than full array
  ✗ Defeats purpose of "selective indexing"

Option 3: Generate full array and index (current JAX approach)
  ✓ Simple and uses public APIs
  ✓ Guaranteed correct values
  ✗ Memory inefficient for sparse access

Option 4: Request feature addition to JAX
  ✓ Would provide public API for selective indexing
  ✗ Requires JAX team approval and implementation

RECOMMENDATION:
For production use, stick with Option 3 (generate full array).
For research/exploration of the PRNG internals, Option 1 is acceptable.
If this is a common use case, propose Option 4 to JAX developers.
""")

print("\n" + "=" * 80)
print("USING ONLY PUBLIC APIS: Best Alternative")
print("=" * 80)

def selective_uniform_public(key, indices, max_index=None):
    """
    Most efficient selective indexing using ONLY public APIs.

    If indices are sparse relative to max_index, generates full array
    and indexes. If no max_index provided, generates minimal array.

    Args:
        key: PRNG key
        indices: List of indices to compute
        max_index: Optional max index (if known, can optimize)

    Returns:
        Array of uniform random values for the specified indices
    """
    indices_arr = jnp.array(indices)

    if max_index is None:
        max_index = jnp.max(indices_arr)

    # Generate array up to max needed index
    full_array = random.uniform(key, (max_index + 1,))

    # Index into it
    return full_array[indices_arr]


print("""
Using public APIs, the most efficient approach is:
  1. Generate array up to max(indices)
  2. Index into it

This is still more efficient than generating unnecessarily large arrays:
""")

# Example with sparse indices
sparse_indices = [7, 9, 100, 500]
print(f"\nExample: Need indices {sparse_indices} from potential array of (1000,)")

# Bad: generate full 1000 elements
full = random.uniform(key, (1000,))
values_bad = full[jnp.array(sparse_indices)]
print(f"  Inefficient: Generate 1000 elements, use 4")

# Better: generate only up to max needed
values_good = selective_uniform_public(key, sparse_indices)
print(f"  Efficient: Generate {max(sparse_indices) + 1} elements, use 4")

# Verify they match
print(f"  Values match: {jnp.allclose(values_bad, values_good)}")

print("\n" + "=" * 80)
print("FINAL RECOMMENDATION")
print("=" * 80)

print("""
For your use case of computing indices (7, 9) from array of shape (128,):

BEST PRACTICE with public APIs:
```python
key = random.key(42)
# Generate only up to max needed index
needed_array = random.uniform(key, (10,))  # max(7, 9) + 1
value_7 = needed_array[7]
value_9 = needed_array[9]
```

This gives you:
  ✓ Exact correct values
  ✓ Only public APIs
  ✓ Memory efficient (10 vs 128 elements)
  ✓ Future-proof

The truly sparse case (needing 2 indices from 1,000,000) would still need
to generate 1,000,000 elements with public APIs. For that, you'd need to:
  1. Use internal APIs (not recommended for production)
  2. Request feature from JAX team
  3. Accept the memory cost
""")
