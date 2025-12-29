"""Minimal test to isolate broadcast layout bug."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax._src.pallas.mosaic import lowering
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager
import sys

NUM_SUBLANES = 8
NUM_LANES = 128


def simple_broadcast_kernel(x_ref, out_ref):
    """Minimal kernel that just does a broadcast."""
    x = x_ref[...]  # Load: shape (1, 128)
    # Broadcast from (1, 128) to (8, 128)
    broadcasted = jnp.broadcast_to(x, (8, 128))
    out_ref[...] = broadcasted


def test_simple_broadcast(b):
    """Test simple broadcast case."""
    print(f"\n{'='*60}")
    print(f"Testing broadcast with b={b}")
    print(f"{'='*60}")

    # Input: (1, b)
    # Output: (8, b)
    x = jnp.ones((1, b), dtype=jnp.float32)

    def kernel_wrapper(x):
        # Pad to tile boundaries
        pad_b = ((b + NUM_LANES - 1) // NUM_LANES) * NUM_LANES
        if b != pad_b:
            x = jnp.pad(x, ((0, 0), (0, pad_b - b)))

        out_shape = jax.ShapeDtypeStruct(
            shape=(NUM_SUBLANES, pad_b),
            dtype=x.dtype
        )

        def kernel_fn(x_ref, out_ref):
            x_val = x_ref[...]
            # This is the problematic broadcast
            broadcasted = jnp.broadcast_to(x_val, (NUM_SUBLANES, pad_b))
            out_ref[...] = broadcasted

        result = pl.pallas_call(
            kernel_fn,
            out_shape=out_shape
        )(x)

        return result[:, :b]

    try:
        # Lower to MLIR
        print(f"Lowering...")
        lowered = jax.jit(kernel_wrapper).lower(x)
        print(f"✓ Successfully lowered!")

        # Get the module
        module = lowered._lowering.mhlo()
        print(f"\nMHLO Module:")
        print(str(module)[:2000])  # Print first 2000 chars

        # Try to compile
        print(f"\nCompiling...")
        compiled = lowered.compile()
        print(f"✓ Successfully compiled!")
        return True

    except Exception as e:
        print(f"✗ Failed:")
        print(f"  {type(e).__name__}")
        # Print more of the error
        error_str = str(e)
        print(f"  {error_str[:500]}")
        if "Invalid input layout" in error_str:
            # Try to extract the MLIR operation
            import re
            match = re.search(r'(%\d+ = "vector\.broadcast"[^}]+})', error_str)
            if match:
                print(f"\n  Problematic operation:")
                print(f"  {match.group(1)}")
        return False


if __name__ == "__main__":
    print("Testing GOOD cases:")
    test_simple_broadcast(128)  # Should work
    test_simple_broadcast(193)  # Should work (not aligned)

    print("\n\nTesting BAD cases:")
    test_simple_broadcast(256)  # Should fail
    test_simple_broadcast(384)  # Should fail
