# JAX Threefry 2x32 Algorithm and Selective Indexing

## Overview

This document explains how JAX's threefry 2x32 PRNG algorithm works and demonstrates how to efficiently compute random values for specific indices without generating the entire array.

## How Threefry 2x32 Works

### Algorithm Structure

Threefry is a **counter-based PRNG**, which is fundamentally different from traditional state-based PRNGs:

```
Traditional PRNG:        Counter-based PRNG (Threefry):
state₀ → generate()      (key, counter₀) → hash() → random₀
  ↓                      (key, counter₁) → hash() → random₁
state₁ → generate()      (key, counter₂) → hash() → random₂
  ↓                      ...
state₂ → generate()
```

### Key Components

1. **Key**: A pair of uint32 values `[k1, k2]`
   - Created from a seed using `jax.random.key(seed)`
   - For seed=42: key = `[0, 42]`

2. **Counter**: Position-dependent uint32 pair `[counter_hi, counter_lo]`
   - For small arrays (< 2³² elements): `counter_hi = 0, counter_lo = index`
   - For large arrays: Uses 64-bit counter split into hi/lo parts

3. **Hash Function**: Threefry2x32
   - Input: (k1, k2, counter_hi, counter_lo)
   - Output: (bits1, bits2)
   - Process: 5 rounds of mixing with rotations, XORs, and additions

### Algorithm for `jax.random.uniform(key, (N,))`

```python
# Step 1: Extract key
k1, k2 = key[0], key[1]

# Step 2: For each index i in 0..N-1:
for i in range(N):
    counter_hi = 0
    counter_lo = i

    # Step 3: Apply threefry hash
    bits1, bits2 = threefry2x32(k1, k2, counter_hi, counter_lo)

    # Step 4: Combine outputs
    bits = bits1 XOR bits2  # 32-bit random integer

    # Step 5: Convert to float in [0, 1)
    # a) Keep only mantissa bits (23 for float32)
    float_bits = bits >> 9  # Shift right by (32 - 23)

    # b) Set exponent to represent 1.0
    float_bits |= 0x3F800000  # IEEE 754 exponent for 1.0

    # c) Reinterpret as float and subtract 1.0
    result[i] = (reinterpret_as_float(float_bits)) - 1.0
```

### Example: Computing `random.uniform(key, (128,))[7]`

```python
key = jax.random.key(42)  # key = [0, 42]

# For index 7:
k1, k2 = 0, 42
counter_hi, counter_lo = 0, 7

# Apply threefry hash (simplified):
bits1, bits2 = threefry2x32(k1=0, k2=42, c_hi=0, c_lo=7)
# Result: bits1 and bits2 are two pseudo-random uint32 values

# Combine
bits = bits1 ^ bits2  # e.g., 0xC6433333

# Convert to float
float_bits = (bits >> 9) | 0x3F800000
result = reinterpret_as_float(float_bits) - 1.0
# Result: 0.7751333714
```

## Why Selective Indexing Works

**Key Insight**: Each output depends ONLY on `(key, counter)`, not on previous outputs.

This means:
- Index 7 depends only on `threefry2x32(key, counter=7)`
- Index 9 depends only on `threefry2x32(key, counter=9)`
- We can compute them independently!

```python
# Traditional approach - must compute ALL values
full_array = random.uniform(key, (128,))  # Compute 128 values
value_7 = full_array[7]  # Extract index 7
value_9 = full_array[9]  # Extract index 9

# Selective approach - compute ONLY what we need
value_7 = selective_uniform(key, [7])[0]
value_9 = selective_uniform(key, [9])[0]
```

## Implementation

### Basic Selective Indexing (1D)

```python
from jax._src.prng import threefry2x32_p
from jax._src import dtypes as jax_dtypes
import jax.numpy as jnp
import jax.random as random

def selective_uniform(key, indices):
    """Compute uniform random values for specific indices only."""

    # Extract key components
    key_data = random.key_data(key)
    k1, k2 = key_data[0], key_data[1]

    # Create counters for target indices
    indices_arr = jnp.array(indices, dtype=jnp.uint32)
    counter_hi = jnp.zeros_like(indices_arr)  # 0 for small arrays
    counter_lo = indices_arr                   # index value

    # Apply threefry hash
    bits1, bits2 = threefry2x32_p.bind(k1, k2, counter_hi, counter_lo)
    bits = bits1 ^ bits2

    # Convert to float32 in [0, 1)
    dtype = jnp.float32
    finfo = jax_dtypes.finfo(dtype)
    nbits, nmant = finfo.bits, finfo.nmant  # 32, 23

    # Right-shift to keep mantissa bits
    float_bits = jax.lax.shift_right_logical(
        bits, jnp.array(nbits - nmant, jnp.uint32))

    # OR with exponent for 1.0
    float_bits = jax.lax.bitwise_or(
        float_bits,
        jnp.asarray(np.array(1.0, dtype).view(jnp.uint32), dtype=jnp.uint32)
    )

    # Bitcast to float and subtract 1.0
    floats = jax.lax.bitcast_convert_type(float_bits, dtype) - 1.0

    return floats
```

### Usage Example

```python
# Create key
key = random.key(42)

# Full array approach
full = random.uniform(key, (128,))
print(f"Index 7: {full[7]}")  # 0.7751333714
print(f"Index 9: {full[9]}")  # 0.8186336756

# Selective approach - identical results!
selective = selective_uniform(key, [7, 9])
print(f"Index 7: {selective[0]}")  # 0.7751333714
print(f"Index 9: {selective[1]}")  # 0.8186336756
```

### Multi-dimensional Arrays

For multi-dimensional arrays, convert to flat index:

```python
# For shape (100, 50), index (7, 9):
flat_index = 7 * 50 + 9 = 359

# Then use flat index as counter
counter_lo = 359
```

Complete implementation in `threefry_advanced_selective.py`.

## Performance Analysis

### Memory Efficiency

| Scenario | Full Array | Selective | Reduction |
|----------|-----------|-----------|-----------|
| 10 from 1M | 4 MB | 40 bytes | 100,000x |
| 100 from 10M | 40 MB | 400 bytes | 100,000x |
| 1K from 1B | 4 GB | 4 KB | 1,000,000x |

### Computational Efficiency

Benchmark results (from `threefry_advanced_selective.py`):

| Configuration | Full (ms) | Selective (ms) | Speedup |
|---------------|-----------|----------------|---------|
| 1K array, 10 indices | 0.23 | 0.88 | 0.26x |
| 100K array, 100 indices | 1.87 | 1.09 | 1.72x |
| 1M array, 100 indices | 6.36 | 1.12 | 5.68x |

**Note**: For small arrays, kernel launch overhead dominates. Speedup increases with array size and sparsity.

### When to Use Selective Indexing

Use selective indexing when:
1. **Sparse selection**: Need few values from large array
   - Example: 100 values from 1,000,000 element array
2. **Memory constrained**: Cannot fit full array in memory
   - Example: 4 GB array but only need a few values
3. **Large-scale distributed**: Each worker needs different indices
   - Example: Parallel Monte Carlo with different sample points per worker

Don't use for:
1. Dense selection (> 10% of array)
2. Very small arrays (overhead dominates)
3. When you need most/all values anyway

## Verification

All three demonstration scripts verify exact matching:

```bash
# Basic demonstration
python threefry_selective_indexing_demo.py

# Advanced (multi-dim, benchmarks)
python threefry_advanced_selective.py
```

Output confirms:
```
✓ All values match: True
Maximum absolute difference: 0.00e+00
```

## Use Cases

### 1. Sparse Neural Network Initialization
```python
# Initialize only active weights in sparse layer
active_indices = [7, 42, 1337, ...]  # Sparse pattern
weights[active_indices] = selective_uniform(key, active_indices)
```

### 2. Monte Carlo Sampling
```python
# Sample specific points in high-dim space
sample_points = [(i, j, k) for ...]  # Selected points
values = selective_uniform(key, sample_points, shape=(1000, 1000, 1000))
```

### 3. Distributed Random Generation
```python
# Each worker generates only its assigned indices
worker_indices = get_worker_indices(worker_id, total_workers)
worker_values = selective_uniform(global_key, worker_indices)
```

### 4. Debugging and Reproducibility
```python
# Reproduce specific random values without full generation
suspicious_index = 12345
debug_value = selective_uniform(key, [suspicious_index])
```

## Technical Details

### Counter Mapping

For array of shape `(D1, D2, D3)` and index `(i, j, k)`:
```python
flat_index = i * (D2 * D3) + j * D3 + k
counter_hi = flat_index >> 32  # High 32 bits (0 for small arrays)
counter_lo = flat_index & 0xFFFFFFFF  # Low 32 bits
```

### Threefry2x32 Hash Details

The hash function (in `jax/_src/prng.py` lines 883-933):
1. Initialize with key and counter
2. Apply 5 rounds of:
   - 4 rotation operations per round
   - XOR and addition mixing
3. Rotations: `[13, 15, 26, 6, 17, 29, 16, 24]` bits
4. Key schedule with constant `0x1BD11BDA`

### Float Conversion Details

For float32 (IEEE 754):
- 1 sign bit, 8 exponent bits, 23 mantissa bits
- Exponent for 1.0: `0x7F` (binary: 01111111)
- Full 1.0 representation: `0x3F800000`

Process:
```python
# Get 23 random mantissa bits
mantissa = bits >> 9  # Keep bits [31:9], discard [8:0]

# Create float with value in [1.0, 2.0)
float_bits = mantissa | 0x3F800000  # Set exponent for 1.0

# Subtract 1.0 to get [0.0, 1.0)
result = reinterpret_cast<float>(float_bits) - 1.0
```

## Limitations and Future Work

### Current Limitations
1. Only float32 supported
2. Only uniform distribution implemented
3. Arrays must have < 2³² elements
4. Requires threefry partitionable mode

### Potential Extensions
1. **More dtypes**: float16, float64, int32, etc.
2. **More distributions**: normal, exponential, categorical
3. **Batched selection**: Select multiple sets of indices efficiently
4. **Sharding integration**: Combine with JAX sharding for distributed generation
5. **Advanced indexing**: Support boolean masks, slices

## References

- JAX PRNG implementation: `jax/_src/prng.py`
- Threefry algorithm: https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
- JAX random module: https://jax.readthedocs.io/en/latest/jax.random.html

## Files

- `threefry_selective_indexing_demo.py` - Basic demonstration and verification
- `threefry_advanced_selective.py` - Multi-dimensional, benchmarks, use cases
- `test_threefry_selective.py` - Initial exploration (debugging)
- `THREEFRY_SELECTIVE_INDEXING_EXPLAINED.md` - This document

## Summary

**JAX's threefry 2x32 is a counter-based PRNG that enables efficient selective indexing:**

✓ Each value depends only on `(key, index)`, not previous values
✓ Can compute any index independently
✓ Produces **exact same values** as full array generation
✓ Massive memory savings for sparse selection (up to 1,000,000x)
✓ Computational speedup for large sparse selections (up to 5.68x measured)
✓ Enables scalable distributed random generation
✓ Perfect for sparse sampling, debugging, and memory-constrained scenarios

The counter-based design makes threefry ideal for parallel and selective computation!
