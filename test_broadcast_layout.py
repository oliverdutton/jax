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


def reproducer_kernel(topk_logits_ref, p_ref, out_ref):
    topk_logits = topk_logits_ref[...]
    p = p_ref[...]
    shape = topk_logits.shape

    cumsum_probs = topk_logits  # skip for reproducer

    threshold_idx = (cumsum_probs < p[None, :]).sum(0, keepdims=True)
    threshold_idx = jnp.where(p[None, :] == 1., shape[0] - 1, threshold_idx)

    # This broadcast is the problematic one
    thresholds = take_along_axis_arrays(
        topk_logits, jnp.broadcast_to(threshold_idx, shape), 0)

    topp_logits = jnp.where(
        topk_logits >= thresholds,
        topk_logits, -1e12)
    out_ref[...] = topp_logits


def call_reproducer_kernel(topk_logits, p):
    """Pallas call wrapper for the reproducer kernel."""
    n, b = topk_logits.shape

    out_shape = jax.ShapeDtypeStruct(
        shape=(n, b),
        dtype=topk_logits.dtype
    )

    result = pl.pallas_call(
        reproducer_kernel,
        out_shape=out_shape
    )(topk_logits, p)

    return result


def test_case(n, b, label="test"):
    """Test a specific case and dump IR."""
    print(f"\n{'='*60}")
    print(f"Testing {label}: (n, b) = ({n}, {b})")
    print(f"{'='*60}")

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    topk_logits = jax.random.normal(key1, (n, b))
    p = jax.random.uniform(key2, (b,), minval=0.0, maxval=1.0)

    try:
        # Compile and dump IR
        compiled = jit(call_reproducer_kernel).lower(topk_logits, p)
        print(f"✓ Successfully lowered!")

        # Try to compile
        compiled_fn = compiled.compile()
        print(f"✓ Successfully compiled!")
        return True

    except Exception as e:
        print(f"✗ Failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        return False


if __name__ == "__main__":
    # Test good cases
    print("\n" + "="*60)
    print("GOOD CASES")
    print("="*60)
    test_case(128, 128, "GOOD: 128x128")
    test_case(128, 193, "GOOD: 128x193")
    test_case(137, 256, "GOOD: 137x256 (padded)")

    # Test bad cases
    print("\n" + "="*60)
    print("BAD CASES")
    print("="*60)
    test_case(128, 256, "BAD: 128x256")
    test_case(128, 384, "BAD: 128x384")
    test_case(256, 256, "BAD: 256x256")
