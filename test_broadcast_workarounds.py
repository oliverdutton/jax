"""Test workarounds for the broadcast layout bug using repeat and reshape."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_SUBLANES = 8
NUM_LANES = 128


def workaround_with_reshape(topk_logits, threshold_idx):
    """Workaround using reshape to change tiling pattern.

    Based on test_retiling_with_replicated_lane pattern.
    Instead of directly broadcasting (1, b) to (n, b), we:
    1. Broadcast (1, b) to (n, b)
    2. Reshape to change the tile structure
    3. Reshape back
    """
    n, b = topk_logits.shape

    # Broadcast from (1, b) to (n, b)
    broadcasted = jnp.broadcast_to(threshold_idx, (n, b))

    # Reshape to (8, n//8, b) to change tiling - this might avoid the bug
    # by making the layout inference see a different shape
    if n % 8 == 0:
        reshaped = broadcasted.reshape(8, n // 8, b)
        # Reshape back to (n, b)
        result = reshaped.reshape(n, b)
    else:
        result = broadcasted

    return result


def workaround_with_repeat(threshold_idx, n):
    """Workaround using pltpu.repeat to tile the data.

    Instead of broadcast_to, use pltpu.repeat which might have better
    layout handling.
    """
    # Use pltpu.repeat to replicate along axis 0
    # This is semantically the same as broadcast but might have different
    # layout inference
    return pltpu.repeat(threshold_idx, n, axis=0)


def workaround_with_tile(threshold_idx, n):
    """Workaround using jnp.tile instead of broadcast_to.

    Sometimes tile has different lowering than broadcast.
    """
    return jnp.tile(threshold_idx, (n, 1))


def test_workarounds():
    """Test different workarounds for the broadcast bug."""

    def kernel_with_reshape(topk_logits_ref, threshold_idx_ref, out_ref):
        topk_logits = topk_logits_ref[...]
        threshold_idx = threshold_idx_ref[...]

        # Workaround: use reshape pattern
        thresholds = workaround_with_reshape(topk_logits, threshold_idx)
        out_ref[...] = thresholds

    def kernel_with_repeat(topk_logits_ref, threshold_idx_ref, out_ref):
        topk_logits = topk_logits_ref[...]
        threshold_idx = threshold_idx_ref[...]
        n = topk_logits.shape[0]

        # Workaround: use pltpu.repeat
        thresholds = workaround_with_repeat(threshold_idx, n)
        out_ref[...] = thresholds

    def kernel_with_tile(topk_logits_ref, threshold_idx_ref, out_ref):
        topk_logits = topk_logits_ref[...]
        threshold_idx = threshold_idx_ref[...]
        n = topk_logits.shape[0]

        # Workaround: use jnp.tile
        thresholds = workaround_with_tile(threshold_idx, n)
        out_ref[...] = thresholds

    # Test with problematic shapes
    test_shapes = [
        (128, 256),  # Known to fail
        (128, 384),  # Known to fail
        (256, 256),  # Known to fail
    ]

    for n, b in test_shapes:
        print(f"\n{'='*60}")
        print(f"Testing workarounds for shape ({n}, {b})")
        print(f"{'='*60}")

        topk_logits = jnp.ones((n, b), dtype=jnp.float32)
        threshold_idx = jnp.zeros((1, b), dtype=jnp.int32)

        # Test reshape workaround
        print("\n1. Testing reshape workaround...")
        try:
            out_shape = jax.ShapeDtypeStruct((n, b), jnp.float32)
            result = pl.pallas_call(
                kernel_with_reshape,
                out_shape=out_shape
            )(topk_logits, threshold_idx)
            print(f"   ✓ Reshape workaround succeeded!")
        except Exception as e:
            print(f"   ✗ Reshape workaround failed: {type(e).__name__}")

        # Test repeat workaround
        print("\n2. Testing pltpu.repeat workaround...")
        try:
            out_shape = jax.ShapeDtypeStruct((n, b), jnp.float32)
            result = pl.pallas_call(
                kernel_with_repeat,
                out_shape=out_shape
            )(topk_logits, threshold_idx)
            print(f"   ✓ pltpu.repeat workaround succeeded!")
        except Exception as e:
            print(f"   ✗ pltpu.repeat workaround failed: {type(e).__name__}")

        # Test tile workaround
        print("\n3. Testing jnp.tile workaround...")
        try:
            out_shape = jax.ShapeDtypeStruct((n, b), jnp.float32)
            result = pl.pallas_call(
                kernel_with_tile,
                out_shape=out_shape
            )(topk_logits, threshold_idx)
            print(f"   ✓ jnp.tile workaround succeeded!")
        except Exception as e:
            print(f"   ✗ jnp.tile workaround failed: {type(e).__name__}")


if __name__ == "__main__":
    print("Testing workarounds for broadcast layout bug")
    test_workarounds()
