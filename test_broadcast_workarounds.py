"""Test workarounds for the broadcast layout bug using repeat and reshape."""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def make_kernel(method: int):
    """Create a kernel with the specified broadcast method.

    Args:
        method: 0 = original broadcast (fails)
                1 = reshape workaround
                2 = pltpu.repeat workaround
                3 = jnp.tile workaround
    """
    def kernel(topk_logits_ref, threshold_idx_ref, out_ref):
        topk_logits = topk_logits_ref[...]
        threshold_idx = threshold_idx_ref[...]
        n, b = topk_logits.shape

        if method == 0:
            # Original: direct broadcast (fails for some shapes)
            thresholds = jnp.broadcast_to(threshold_idx, (n, b))

        elif method == 1:
            # Workaround 1: reshape pattern
            broadcasted = jnp.broadcast_to(threshold_idx, (n, b))
            if n % 8 == 0 and b % 128 == 0 and b > 128:
                # Retile to avoid bug
                reshaped = broadcasted.reshape(8, n // 8, b)
                thresholds = reshaped.reshape(n, b)
            else:
                thresholds = broadcasted

        elif method == 2:
            # Workaround 2: pltpu.repeat
            thresholds = pltpu.repeat(threshold_idx, n, axis=0)

        elif method == 3:
            # Workaround 3: jnp.tile
            thresholds = jnp.tile(threshold_idx, (n, 1))

        else:
            raise ValueError(f"Invalid method: {method}")

        out_ref[...] = thresholds

    return kernel


def test_workarounds():
    """Test different workarounds for the broadcast bug."""

    method_names = {
        0: "Original broadcast (baseline)",
        1: "Reshape workaround",
        2: "pltpu.repeat workaround",
        3: "jnp.tile workaround",
    }

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

        # Create inputs ONCE for all methods (exact same conditions)
        topk_logits = jnp.ones((n, b), dtype=jnp.float32)
        threshold_idx = jnp.zeros((1, b), dtype=jnp.float32)
        out_shape = jax.ShapeDtypeStruct((n, b), jnp.float32)

        # Test all 4 methods sequentially with IDENTICAL inputs
        for method in [0, 1, 2, 3]:
            method_name = method_names[method]
            print(f"\n{method}. Testing {method_name}...")

            try:
                # Create kernel with this method
                kernel = make_kernel(method)

                # Call with EXACT SAME inputs
                result = pl.pallas_call(
                    kernel,
                    out_shape=out_shape
                )(topk_logits, threshold_idx)

                print(f"   ✓ {method_name} succeeded!")

            except Exception as e:
                error_msg = str(e)
                if "Invalid input layout" in error_msg:
                    if method == 0:
                        print(f"   ✗ {method_name} failed with layout bug (expected)")
                    else:
                        print(f"   ✗ {method_name} also failed with layout bug")
                else:
                    print(f"   ✗ {method_name} failed: {type(e).__name__}")
                    if len(error_msg) > 0:
                        print(f"      {error_msg[:150]}")


if __name__ == "__main__":
    print("Testing workarounds for broadcast layout bug")
    test_workarounds()
