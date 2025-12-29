"""Test script to reproduce broadcast layout bug and dump IR."""

import functools
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import os

# Set environment variables to dump IR
os.environ['JAX_DUMP_IR_TO'] = '/tmp/jax_ir_dump'
os.environ['JAX_ENABLE_X64'] = 'false'

NUM_SUBLANES = 8
NUM_LANES = 128


def pad(arr, tile_shape, val=0):
    """Pad array to be multiple of tile_shape dimensions."""
    if len(tile_shape) != arr.ndim:
        raise ValueError(
            f"tile_shape length {len(tile_shape)} must match array ndim {arr.ndim}"
        )

    pad_widths = []
    for dim_size, block_size in zip(arr.shape, tile_shape):
        target_size = ((dim_size + block_size - 1) // block_size) * block_size
        pad_size = target_size - dim_size
        pad_widths.append((0, pad_size))

    if all(w == (0, 0) for w in pad_widths):
        return arr

    return jnp.pad(arr, pad_widths, mode='constant', constant_values=val)


def take_along_axis_arrays(val, idx, axis):
    """Gather values from val array using indices in idx array."""
    shape = idx.shape
    tile_shape = (NUM_SUBLANES, NUM_LANES)
    val, idx = (pad(x, tile_shape, val=0) for x in (val, idx))

    def _gather_arrays(val, idx):
        accumulators = [
            jnp.zeros(tile_shape, dtype=val.dtype)
            for _ in range(idx.shape[axis] // tile_shape[axis])
        ]
        for val_offset in range(0, val.shape[axis], tile_shape[axis]):
            val_tile = lax.slice_in_dim(val, val_offset, val_offset+tile_shape[axis], axis=axis)

            for idx_offset in range(0, idx.shape[axis], tile_shape[axis]):
                idx_tile = lax.slice_in_dim(idx, idx_offset, idx_offset+tile_shape[axis], axis=axis)
                mask = (idx_tile >= val_offset) & (idx_tile < val_offset + tile_shape[axis])
                gather_tile = jnp.take_along_axis(
                    val_tile,
                    (idx_tile - val_offset) % tile_shape[axis],
                    axis=axis
                )
                i = idx_offset // tile_shape[axis]
                accumulators[i] = jnp.where(mask, gather_tile, accumulators[i])
        return jnp.concatenate(accumulators, axis=axis)

    batch_axis = 1 - axis
    assert val.shape[batch_axis]==idx.shape[batch_axis]
    return jnp.concatenate(
        [_gather_arrays(v, i)
         for v, i in zip(*map(lambda arr: jnp.split(
             arr, arr.shape[batch_axis] // tile_shape[batch_axis], axis=batch_axis), (val, idx)))
        ],
        axis=batch_axis
    )[:shape[0], :shape[1]]


def make_reproducer_kernel(method: int):
    """Create reproducer kernel with specified broadcast method.

    Args:
        method: 0 = original broadcast (fails)
                1 = reshape workaround
                2 = pltpu.repeat workaround
                3 = jnp.tile workaround
    """
    def reproducer_kernel(topk_logits_ref, p_ref, out_ref):
        topk_logits = topk_logits_ref[...]
        p = p_ref[...]
        shape = topk_logits.shape
        n, b = shape

        cumsum_probs = topk_logits  # skip for reproducer

        threshold_idx = (cumsum_probs < p[None, :]).sum(0, keepdims=True)
        threshold_idx = jnp.where(p[None, :] == 1., shape[0] - 1, threshold_idx)

        # Broadcast using specified method
        if method == 0:
            # Original: direct broadcast (fails for some shapes)
            broadcasted_idx = jnp.broadcast_to(threshold_idx, shape)

        elif method == 1:
            # Workaround 1: reshape pattern
            broadcasted_idx = jnp.broadcast_to(threshold_idx, shape)
            if n % 8 == 0 and b % 128 == 0 and b > 128:
                # Retile to avoid bug
                reshaped = broadcasted_idx.reshape(8, n // 8, b)
                broadcasted_idx = reshaped.reshape(n, b)

        elif method == 2:
            # Workaround 2: pltpu.repeat
            broadcasted_idx = pltpu.repeat(threshold_idx, n, axis=0)

        elif method == 3:
            # Workaround 3: jnp.tile
            broadcasted_idx = jnp.tile(threshold_idx, (n, 1))

        else:
            raise ValueError(f"Invalid method: {method}")

        thresholds = take_along_axis_arrays(
            topk_logits, broadcasted_idx, 0)

        topp_logits = jnp.where(
            topk_logits >= thresholds,
            topk_logits, -1e12)
        out_ref[...] = topp_logits

    return reproducer_kernel


def call_reproducer_kernel(topk_logits, p, method=0):
    """Pallas call wrapper for the reproducer kernel.

    Args:
        method: 0 = original broadcast (fails)
                1 = reshape workaround
                2 = pltpu.repeat workaround
                3 = jnp.tile workaround
    """
    n, b = topk_logits.shape

    out_shape = jax.ShapeDtypeStruct(
        shape=(n, b),
        dtype=topk_logits.dtype
    )

    kernel = make_reproducer_kernel(method)
    result = pl.pallas_call(
        kernel,
        out_shape=out_shape
    )(topk_logits, p)

    return result


def test_case(n, b, label="test", method=None):
    """Test a specific case and dump IR.

    Args:
        method: If None, test all methods. Otherwise test specific method 0-3.
    """
    method_names = {
        0: "Original broadcast (baseline)",
        1: "Reshape workaround",
        2: "pltpu.repeat workaround",
        3: "jnp.tile workaround",
    }

    print(f"\n{'='*60}")
    print(f"Testing {label}: (n, b) = ({n}, {b})")
    print(f"{'='*60}")

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Create inputs ONCE for all methods
    topk_logits = jax.random.normal(key1, (n, b))
    p = jax.random.uniform(key2, (b,), minval=0.0, maxval=1.0)

    # Test specified method(s)
    methods_to_test = [method] if method is not None else [0, 1, 2, 3]

    for m in methods_to_test:
        method_name = method_names[m]
        print(f"\n{m}. Testing {method_name}...")

        try:
            # Compile and dump IR
            compiled = jit(functools.partial(call_reproducer_kernel, method=m)).lower(topk_logits, p)
            print(f"   ✓ Lowered successfully!")

            # Try to compile
            compiled_fn = compiled.compile()
            print(f"   ✓ Compiled successfully!")

        except Exception as e:
            error_msg = str(e)
            if "Invalid input layout" in error_msg:
                if m == 0:
                    print(f"   ✗ Failed with layout bug (expected for some shapes)")
                else:
                    print(f"   ✗ Also failed with layout bug")
            else:
                print(f"   ✗ Failed: {type(e).__name__}")
                if len(error_msg) > 0:
                    print(f"      {error_msg[:150]}")

    return True


if __name__ == "__main__":
    # Test good case (should pass with all methods)
    print("\n" + "="*60)
    print("GOOD CASE - All methods should succeed")
    print("="*60)
    test_case(128, 128, "GOOD: 128x128")

    # Test bad cases (method 0 should fail, workarounds might succeed)
    print("\n" + "="*60)
    print("BAD CASES - Method 0 should fail, testing workarounds")
    print("="*60)
    test_case(128, 256, "BAD: 128x256")
    test_case(128, 384, "BAD: 128x384")
    test_case(256, 256, "BAD: 256x256")
