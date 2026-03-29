#!/usr/bin/env python3
"""
Advanced Selective Indexing for JAX Threefry PRNG

This demonstrates selective indexing for:
1. 1D arrays (indices)
2. Multi-dimensional arrays (index tuples)
3. Performance benchmarks
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax._src.prng import threefry2x32_p
from jax._src import dtypes as jax_dtypes
import time

def selective_uniform(key, indices, shape=None, dtype=jnp.float32):
    """
    Compute uniform random values for specific indices without generating full array.

    For 1D arrays:
        indices: list of integer indices
        shape: optional, used to validate indices

    For multi-dimensional arrays:
        indices: list of tuples (i, j, ...) for each position
        shape: required, the full array shape

    Args:
        key: PRNG key
        indices: Integer indices (for 1D) or list of tuples (for multi-dim)
        shape: Optional for 1D, required for multi-dim
        dtype: Output dtype (default float32)

    Returns:
        Array of uniform random values for the specified indices
    """
    if dtype != jnp.float32:
        raise NotImplementedError("Currently only float32 is supported")

    # Extract key components
    key_data = random.key_data(key)
    k1, k2 = key_data[0], key_data[1]

    # Convert multi-dimensional indices to flat indices if needed
    if shape is not None and len(shape) > 1:
        # Multi-dimensional indexing
        # Convert tuple indices to flat indices
        if not all(isinstance(idx, (tuple, list)) for idx in indices):
            raise ValueError("For multi-dim arrays, indices must be tuples")

        flat_indices = []
        for idx_tuple in indices:
            if len(idx_tuple) != len(shape):
                raise ValueError(f"Index {idx_tuple} doesn't match shape {shape}")
            # Convert to flat index using row-major ordering
            flat_idx = 0
            multiplier = 1
            for i in range(len(shape) - 1, -1, -1):
                flat_idx += idx_tuple[i] * multiplier
                multiplier *= shape[i]
            flat_indices.append(flat_idx)
        indices_arr = jnp.array(flat_indices, dtype=jnp.uint32)
    else:
        # 1D indexing
        indices_arr = jnp.array(indices, dtype=jnp.uint32)

    # Create counter values (hi=0, lo=index for small arrays)
    counter_hi = jnp.zeros_like(indices_arr)
    counter_lo = indices_arr

    # Apply threefry hash
    bits1, bits2 = threefry2x32_p.bind(k1, k2, counter_hi, counter_lo)
    bits = bits1 ^ bits2

    # Convert to float32 in [0, 1)
    finfo = jax_dtypes.finfo(dtype)
    nbits, nmant = finfo.bits, finfo.nmant
    uint_dtype = np.dtype('uint32')

    float_bits = jax.lax.shift_right_logical(bits, jnp.array(nbits - nmant, uint_dtype))
    float_bits = jax.lax.bitwise_or(
        float_bits,
        jnp.asarray(np.array(1.0, dtype).view(uint_dtype), dtype=uint_dtype)
    )
    floats = jax.lax.bitcast_convert_type(float_bits, dtype) - jnp.array(1., dtype)

    return floats


print("=" * 80)
print("ADVANCED SELECTIVE INDEXING EXAMPLES")
print("=" * 80)

# Example 1: 1D Array
print("\n" + "=" * 80)
print("Example 1: 1D Array")
print("=" * 80)

key = random.key(42)
array_1d = random.uniform(key, (1000,))

# Select specific indices
indices_1d = [7, 42, 100, 500, 999]
selective_1d = selective_uniform(key, indices_1d)

print(f"\nArray shape: (1000,)")
print(f"Selected indices: {indices_1d}")
print(f"\nResults:")
print(f"{'Index':<8} {'Selective':<15} {'Full':<15} {'Match':<10}")
print("-" * 60)
for idx, sel_val in zip(indices_1d, selective_1d):
    full_val = array_1d[idx]
    match = jnp.allclose(sel_val, full_val, rtol=1e-7)
    print(f"{idx:<8} {sel_val:<15.10f} {full_val:<15.10f} {str(match):<10}")

# Example 2: 2D Array
print("\n" + "=" * 80)
print("Example 2: 2D Array (Matrix)")
print("=" * 80)

shape_2d = (100, 50)
array_2d = random.uniform(key, shape_2d)

# Select specific (row, col) positions
indices_2d = [(0, 0), (7, 9), (42, 13), (99, 49)]
selective_2d = selective_uniform(key, indices_2d, shape=shape_2d)

print(f"\nArray shape: {shape_2d}")
print(f"Selected positions (row, col): {indices_2d}")
print(f"\nResults:")
print(f"{'Index':<12} {'Selective':<15} {'Full':<15} {'Match':<10}")
print("-" * 65)
for idx_tuple, sel_val in zip(indices_2d, selective_2d):
    full_val = array_2d[idx_tuple]
    match = jnp.allclose(sel_val, full_val, rtol=1e-7)
    print(f"{str(idx_tuple):<12} {sel_val:<15.10f} {full_val:<15.10f} {str(match):<10}")

# Example 3: 3D Array
print("\n" + "=" * 80)
print("Example 3: 3D Array (Tensor)")
print("=" * 80)

shape_3d = (10, 20, 30)
array_3d = random.uniform(key, shape_3d)

# Select specific (i, j, k) positions
indices_3d = [(0, 0, 0), (5, 10, 15), (9, 19, 29)]
selective_3d = selective_uniform(key, indices_3d, shape=shape_3d)

print(f"\nArray shape: {shape_3d}")
print(f"Selected positions (i, j, k): {indices_3d}")
print(f"\nResults:")
print(f"{'Index':<15} {'Selective':<15} {'Full':<15} {'Match':<10}")
print("-" * 70)
for idx_tuple, sel_val in zip(indices_3d, selective_3d):
    full_val = array_3d[idx_tuple]
    match = jnp.allclose(sel_val, full_val, rtol=1e-7)
    print(f"{str(idx_tuple):<15} {sel_val:<15.10f} {full_val:<15.10f} {str(match):<10}")

# Performance Benchmark
print("\n" + "=" * 80)
print("PERFORMANCE BENCHMARK")
print("=" * 80)

# Test different array sizes and selection ratios
test_configs = [
    (1_000, 10, "Small array, few indices"),
    (100_000, 100, "Medium array, moderate indices"),
    (1_000_000, 100, "Large array, sparse selection"),
]

print(f"\nBenchmarking (averaged over 10 runs):\n")
print(f"{'Config':<35} {'Full (ms)':<12} {'Selective (ms)':<15} {'Speedup':<10}")
print("-" * 80)

for size, n_indices, description in test_configs:
    indices = list(np.random.choice(size, n_indices, replace=False))

    # Benchmark full array generation
    key = random.key(12345)

    # Warmup
    _ = random.uniform(key, (size,))
    _ = selective_uniform(key, indices)

    # Time full array
    times_full = []
    for _ in range(10):
        start = time.time()
        _ = random.uniform(key, (size,)).block_until_ready()
        times_full.append((time.time() - start) * 1000)
    avg_full = np.mean(times_full)

    # Time selective
    times_selective = []
    for _ in range(10):
        start = time.time()
        _ = selective_uniform(key, indices).block_until_ready()
        times_selective.append((time.time() - start) * 1000)
    avg_selective = np.mean(times_selective)

    speedup = avg_full / avg_selective if avg_selective > 0 else float('inf')

    print(f"{description:<35} {avg_full:<12.3f} {avg_selective:<15.3f} {speedup:<10.2f}x")

print("\n" + "=" * 80)
print("MEMORY EFFICIENCY")
print("=" * 80)

print("""
Memory comparison for different scenarios:

Scenario 1: Select 10 values from 1,000,000 element array
  Full approach:   4,000,000 bytes (4 MB)
  Selective:       40 bytes
  Reduction:       100,000x less memory

Scenario 2: Select 100 values from 10,000,000 element array
  Full approach:   40,000,000 bytes (40 MB)
  Selective:       400 bytes
  Reduction:       100,000x less memory

Scenario 3: Select 1,000 values from 1,000,000,000 element array
  Full approach:   4,000,000,000 bytes (4 GB)
  Selective:       4,000 bytes (4 KB)
  Reduction:       1,000,000x less memory
""")

print("\n" + "=" * 80)
print("PRACTICAL USE CASES")
print("=" * 80)

print("""
1. SPARSE NEURAL NETWORK INITIALIZATION
   Initialize only specific weights in a huge parameter space
   Example: Initialize 1000 random weights in a 10M parameter model

2. MONTE CARLO SAMPLING
   Sample specific points in a high-dimensional space
   Example: Evaluate 100 points in a 1000×1000×1000 grid

3. REINFORCEMENT LEARNING
   Sample random actions for specific states
   Example: Generate exploration noise for specific state-action pairs

4. SCIENTIFIC SIMULATION
   Initialize random values at specific spatial/temporal coordinates
   Example: Random perturbations at specific grid points in climate model

5. DISTRIBUTED TRAINING
   Each worker generates random values for its assigned indices
   Example: Data parallel training where each worker owns specific parameters

6. DEBUGGING AND TESTING
   Reproduce specific random values without regenerating entire array
   Example: Test behavior at specific random initializations
""")

print("\n" + "=" * 80)
print("IMPLEMENTATION NOTES")
print("=" * 80)

print("""
Counter Mapping for Multi-dimensional Arrays:
  - JAX uses row-major (C-style) flattening
  - For shape (D1, D2, D3), index (i, j, k) maps to:
    flat_index = i * (D2 * D3) + j * D3 + k
  - This flat index becomes the counter_lo value
  - counter_hi = 0 for arrays with < 2^32 total elements

Limitations:
  - Currently only supports float32 output
  - Arrays must have < 2^32 total elements
  - Uses threefry partitionable mode (config flag must be set)

Extensions:
  - Support for float16, float64
  - Support for other distributions (normal, exponential, etc.)
  - Batched selective indexing
  - Integration with JAX's sharding for distributed generation
""")
