"""Test workarounds for the broadcast layout bug using repeat and reshape."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def test_workarounds():
    """Test different workarounds for the broadcast bug."""

    def kernel_with_reshape(topk_logits_ref, threshold_idx_ref, out_ref):
        topk_logits = topk_logits_ref[...]
        threshold_idx = threshold_idx_ref[...]
        n, b = topk_logits.shape

        # Workaround: use reshape pattern
        # Broadcast from (1, b) to (n, b)
        broadcasted = jnp.broadcast_to(threshold_idx, (n, b))

        # Reshape to change tiling - this might avoid the bug
        # by making the layout inference see a different shape
        if n % 8 == 0 and b % 128 == 0 and b > 128:
            reshaped = broadcasted.reshape(8, n // 8, b)
            # Reshape back to (n, b)
            thresholds = reshaped.reshape(n, b)
        else:
            thresholds = broadcasted

        out_ref[...] = thresholds

    def kernel_with_repeat(topk_logits_ref, threshold_idx_ref, out_ref):
        topk_logits = topk_logits_ref[...]
        threshold_idx = threshold_idx_ref[...]
        n = topk_logits.shape[0]

        # Workaround: use pltpu.repeat
        # This is semantically the same as broadcast but might have different
        # layout inference
        thresholds = pltpu.repeat(threshold_idx, n, axis=0)
        out_ref[...] = thresholds

    def kernel_with_tile(topk_logits_ref, threshold_idx_ref, out_ref):
        topk_logits = topk_logits_ref[...]
        threshold_idx = threshold_idx_ref[...]
        n = topk_logits.shape[0]

        # Workaround: use jnp.tile instead of broadcast_to
        # Sometimes tile has different lowering than broadcast
        thresholds = jnp.tile(threshold_idx, (n, 1))
        out_ref[...] = thresholds

    def kernel_original(topk_logits_ref, threshold_idx_ref, out_ref):
        """Original version that fails."""
        topk_logits = topk_logits_ref[...]
        threshold_idx = threshold_idx_ref[...]
        n, b = topk_logits.shape

        # This is what fails
        thresholds = jnp.broadcast_to(threshold_idx, (n, b))
        out_ref[...] = thresholds

    # Test with problematic shapes
    test_shapes = [
        (128, 128),  # Should pass (not problematic)
        (128, 256),  # Known to fail
        (128, 384),  # Known to fail
        (256, 256),  # Known to fail
    ]

    for n, b in test_shapes:
        print(f"\n{'='*60}")
        print(f"Testing workarounds for shape ({n}, {b})")
        print(f"{'='*60}")

        topk_logits = jnp.ones((n, b), dtype=jnp.float32)
        threshold_idx = jnp.zeros((1, b), dtype=jnp.float32)  # Changed to float32

        # Test original (should fail for some shapes)
        print("\n0. Testing original broadcast (baseline)...")
        try:
            out_shape = jax.ShapeDtypeStruct((n, b), jnp.float32)
            result = pl.pallas_call(
                kernel_original,
                out_shape=out_shape
            )(topk_logits, threshold_idx)
            print(f"   ✓ Original succeeded!")
        except Exception as e:
            error_msg = str(e)
            if "Invalid input layout" in error_msg:
                print(f"   ✗ Original failed with layout bug (expected for some shapes)")
            else:
                print(f"   ✗ Original failed: {type(e).__name__}: {error_msg[:100]}")

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
            error_msg = str(e)
            if "Invalid input layout" in error_msg:
                print(f"   ✗ Reshape workaround also failed with layout bug")
            else:
                print(f"   ✗ Reshape workaround failed: {type(e).__name__}: {error_msg[:100]}")

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
            error_msg = str(e)
            if "Invalid input layout" in error_msg:
                print(f"   ✗ pltpu.repeat workaround also failed with layout bug")
            else:
                print(f"   ✗ pltpu.repeat workaround failed: {type(e).__name__}: {error_msg[:100]}")

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
            error_msg = str(e)
            if "Invalid input layout" in error_msg:
                print(f"   ✗ jnp.tile workaround also failed with layout bug")
            else:
                print(f"   ✗ jnp.tile workaround failed: {type(e).__name__}: {error_msg[:100]}")


if __name__ == "__main__":
    print("Testing workarounds for broadcast layout bug")
    test_workarounds()
