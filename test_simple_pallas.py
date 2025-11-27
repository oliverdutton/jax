"""Simple test of Pallas with interpret mode."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import time

def simple_kernel(x_ref, o_ref):
    """Simple kernel that copies input to output."""
    o_ref[...] = x_ref[...]

def test_pallas_interpret():
    """Test that Pallas interpret mode works."""
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 128), dtype=jnp.float32)

    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

    print("Testing Pallas with interpret=True...")
    start = time.time()
    result = pl.pallas_call(
        simple_kernel,
        out_shape=out_shape,
        interpret=True,
    )(x)
    result = jax.block_until_ready(result)
    end = time.time()

    print(f"Time: {end - start:.4f}s")
    print(f"Result shape: {result.shape}")
    print(f"Results match: {jnp.allclose(x, result)}")

if __name__ == "__main__":
    test_pallas_interpret()
