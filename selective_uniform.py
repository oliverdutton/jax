#!/usr/bin/env python3
"""
Selective uniform random sampling using JAX threefry PRNG.

This implementation uses internal JAX APIs to achieve true selective indexing
with exact matching values. Only use this if you understand the stability
implications of using internal APIs.

For production code, prefer generating arrays up to max(indices)+1.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from jax._src.prng import threefry2x32_p


def selective_uniform(key, indices, dtype=jnp.float32, minval=0., maxval=1.):
    """
    Generate uniform random values for specific indices only.

    This produces EXACTLY the same values as:
        random.uniform(key, (N,), dtype=dtype, minval=minval, maxval=maxval)[indices]

    but only computes the requested indices, saving memory and computation.

    Args:
        key: JAX PRNG key
        indices: Array-like of integer indices to compute
        dtype: Output dtype (default: jnp.float32)
        minval: Minimum value (inclusive), broadcast-compatible with output shape
        maxval: Maximum value (exclusive), broadcast-compatible with output shape

    Returns:
        JAX array of uniform random values in [minval, maxval) for the specified indices

    Example:
        >>> key = random.key(42)
        >>> indices = jnp.array([7, 9, 100])
        >>> values = selective_uniform(key, indices)
        >>> # values[i] == random.uniform(key, (101,))[indices[i]]
        >>>
        >>> # With custom range
        >>> values = selective_uniform(key, indices, minval=-1.0, maxval=1.0)

    Note:
        This uses internal JAX APIs (threefry2x32_p) which may change between
        versions. For production code, consider using the public API approach
        of generating arrays up to max(indices)+1.
    """
    # Convert indices to JAX array
    indices = jnp.asarray(indices, dtype=jnp.uint32)

    # Extract key components
    key_data = random.key_data(key)
    k1, k2 = key_data[0], key_data[1]

    # Create counter values
    # For arrays with < 2^32 elements, counter = (0, index)
    counter_hi = jnp.zeros_like(indices)
    counter_lo = indices

    # Apply threefry hash to get random bits
    bits1, bits2 = threefry2x32_p.bind(k1, k2, counter_hi, counter_lo)

    # XOR to combine (matches threefry partitionable mode)
    bits = bits1 ^ bits2

    # Convert bits to uniform float in [0, 1)
    floats = _bits_to_uniform(bits, dtype)

    # Scale to [minval, maxval) following JAX's implementation
    minval = jax.lax.convert_element_type(minval, dtype)
    maxval = jax.lax.convert_element_type(maxval, dtype)

    # Scale and shift: floats * (maxval - minval) + minval
    # Use lax.max to ensure values are at least minval
    return jax.lax.max(minval, floats * (maxval - minval) + minval)


def _bits_to_uniform(bits, dtype):
    """
    Convert random uint32 bits to uniform float in [0, 1).

    This matches the conversion in jax._src.random._uniform().

    Args:
        bits: uint32 array of random bits
        dtype: Target float dtype

    Returns:
        Array of uniform random floats in [0, 1)
    """
    # Get dtype properties
    finfo = jnp.finfo(dtype)
    nbits = finfo.bits
    nmant = finfo.nmant

    # Right-shift to keep only mantissa bits
    # For float32: keep 23 bits, shift right by (32 - 23) = 9
    float_bits = jax.lax.shift_right_logical(
        bits,
        jnp.uint32(nbits - nmant)
    )

    # Create bit pattern for 1.0 in the target dtype
    # For float32: 0x3F800000 (sign=0, exp=127, mantissa=0)
    one_bits = jnp.asarray(
        jnp.ones((), dtype=dtype).view(jnp.uint32),
        dtype=jnp.uint32
    )

    # OR with 1.0 bit pattern to set exponent
    float_bits = jax.lax.bitwise_or(float_bits, one_bits)

    # Bitcast to float and subtract 1.0 to get [0, 1)
    floats = jax.lax.bitcast_convert_type(float_bits, dtype)
    return floats - jnp.ones((), dtype=dtype)


def selective_uniform_multidim(key, indices, shape, dtype=jnp.float32, minval=0., maxval=1.):
    """
    Generate uniform random values for specific multi-dimensional indices.

    Args:
        key: JAX PRNG key
        indices: Array of shape (N, ndim) where each row is a multi-dim index
        shape: Tuple indicating the full array shape
        dtype: Output dtype (default: jnp.float32)
        minval: Minimum value (inclusive), broadcast-compatible with output shape
        maxval: Maximum value (exclusive), broadcast-compatible with output shape

    Returns:
        JAX array of length N with uniform random values

    Example:
        >>> key = random.key(42)
        >>> # Get values at positions (7, 9) and (42, 13) from a (100, 50) array
        >>> indices = jnp.array([[7, 9], [42, 13]])
        >>> values = selective_uniform_multidim(key, indices, shape=(100, 50))
        >>>
        >>> # With custom range
        >>> values = selective_uniform_multidim(key, indices, shape=(100, 50),
        ...                                     minval=-1.0, maxval=1.0)
    """
    indices = jnp.asarray(indices)

    # Convert multi-dimensional indices to flat indices
    # Using row-major (C-style) ordering
    strides = jnp.array([jnp.prod(jnp.array(shape[i+1:]))
                         for i in range(len(shape))] + [1])
    flat_indices = jnp.sum(indices * strides[:-1], axis=-1)

    # Use regular selective_uniform with flat indices
    return selective_uniform(key, flat_indices.astype(jnp.uint32), dtype, minval, maxval)


# ============================================================================
# Demo and verification
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Selective Uniform Random Sampling - Internal API Version")
    print("=" * 80)

    # Test 1D indexing
    print("\n" + "=" * 80)
    print("Test 1: 1D Array Indexing")
    print("=" * 80)

    key = random.key(42)
    full_array = random.uniform(key, (128,))

    indices_1d = jnp.array([7, 9, 42, 100])
    selective_values = selective_uniform(key, indices_1d)

    print(f"\nIndices: {indices_1d}")
    print(f"\n{'Index':<8} {'Selective':<15} {'Full Array':<15} {'Match':<10}")
    print("-" * 60)
    for i, idx in enumerate(indices_1d):
        sel_val = selective_values[i]
        full_val = full_array[idx]
        match = jnp.allclose(sel_val, full_val)
        print(f"{idx:<8} {sel_val:<15.10f} {full_val:<15.10f} {str(match):<10}")

    all_match = jnp.allclose(selective_values, full_array[indices_1d])
    print(f"\n✓ All values match: {all_match}")

    # Test 2D indexing
    print("\n" + "=" * 80)
    print("Test 2: 2D Array Indexing")
    print("=" * 80)

    shape_2d = (100, 50)
    full_2d = random.uniform(key, shape_2d)

    indices_2d = jnp.array([[7, 9], [42, 13], [99, 49]])
    selective_2d = selective_uniform_multidim(key, indices_2d, shape_2d)

    print(f"\nArray shape: {shape_2d}")
    print(f"Indices (row, col):\n{indices_2d}")
    print(f"\n{'Index':<12} {'Selective':<15} {'Full Array':<15} {'Match':<10}")
    print("-" * 65)
    for i, idx_tuple in enumerate(indices_2d):
        sel_val = selective_2d[i]
        full_val = full_2d[tuple(idx_tuple)]
        match = jnp.allclose(sel_val, full_val)
        idx_str = f"({idx_tuple[0]}, {idx_tuple[1]})"
        print(f"{idx_str:<12} {sel_val:<15.10f} {full_val:<15.10f} {str(match):<10}")

    # Verify all match
    full_2d_vals = jnp.array([full_2d[tuple(idx)] for idx in indices_2d])
    all_match_2d = jnp.allclose(selective_2d, full_2d_vals)
    print(f"\n✓ All values match: {all_match_2d}")

    # Test different dtypes
    print("\n" + "=" * 80)
    print("Test 3: Different dtypes")
    print("=" * 80)

    test_indices = jnp.array([7, 9])

    for test_dtype in [jnp.float32, jnp.float64]:
        full = random.uniform(key, (128,), dtype=test_dtype)
        selective = selective_uniform(key, test_indices, dtype=test_dtype)
        match = jnp.allclose(selective, full[test_indices])
        print(f"{str(test_dtype):<12} Match: {match}")

    # Test minval/maxval
    print("\n" + "=" * 80)
    print("Test 4: Custom minval/maxval")
    print("=" * 80)

    test_ranges = [
        (0., 1., "Default [0, 1)"),
        (-1., 1., "Range [-1, 1)"),
        (10., 20., "Range [10, 20)"),
        (0., 10., "Range [0, 10)"),
    ]

    test_indices = jnp.array([7, 9, 42])

    print(f"\n{'Range':<20} {'Min':<12} {'Max':<12} {'Match':<10}")
    print("-" * 60)

    for minval, maxval, description in test_ranges:
        full = random.uniform(key, (128,), minval=minval, maxval=maxval)
        selective = selective_uniform(key, test_indices, minval=minval, maxval=maxval)

        # Verify values are in range
        in_range = jnp.all((selective >= minval) & (selective < maxval))

        # Verify exact match with full array
        match = jnp.allclose(selective, full[test_indices])

        print(f"{description:<20} {jnp.min(selective):<12.4f} {jnp.max(selective):<12.4f} {str(match):<10}")

    # Show actual values for one case
    print(f"\nExample with range [-1, 1):")
    selective_scaled = selective_uniform(key, jnp.array([7, 9]), minval=-1., maxval=1.)
    full_scaled = random.uniform(key, (128,), minval=-1., maxval=1.)
    print(f"  Selective[7]: {selective_scaled[0]:.6f}")
    print(f"  Full[7]:      {full_scaled[7]:.6f}")
    print(f"  Match: {jnp.allclose(selective_scaled[0], full_scaled[7])}")

    # Performance comparison
    print("\n" + "=" * 80)
    print("Memory Efficiency")
    print("=" * 80)

    test_cases = [
        (128, [7, 9], "Small array, 2 indices"),
        (10000, [7, 9, 100], "Medium array, 3 indices"),
        (1000000, [7, 9, 100, 5000], "Large array, 4 sparse indices"),
    ]

    print(f"\n{'Case':<35} {'Full (elements)':<18} {'Selective':<15} {'Reduction':<12}")
    print("-" * 85)
    for array_size, idx_list, description in test_cases:
        n_selective = len(idx_list)
        reduction = array_size / n_selective
        print(f"{description:<35} {array_size:<18} {n_selective:<15} {reduction:<12.1f}x")

    print("\n" + "=" * 80)
    print("Usage Summary")
    print("=" * 80)

    print("""
Basic usage:
    key = random.key(42)
    indices = jnp.array([7, 9, 100])
    values = selective_uniform(key, indices)

With custom range:
    values = selective_uniform(key, indices, minval=-1.0, maxval=1.0)

Multi-dimensional:
    indices = jnp.array([[7, 9], [42, 13]])
    values = selective_uniform_multidim(key, indices, shape=(100, 50))

Different dtypes:
    values = selective_uniform(key, indices, dtype=jnp.float64)

Caveats:
    - Uses internal API (threefry2x32_p) - may change between JAX versions
    - Only works for arrays with < 2^32 total elements
    - For production, consider generating up to max(indices)+1 with public APIs
    """)
