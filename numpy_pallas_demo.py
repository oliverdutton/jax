"""
Demonstration of Pallas call interpretation using pure NumPy via io_callback.

This shows how to progressively convert Pallas operations to use NumPy arrays,
leveraging io_callback for the conversion boundary.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental import io_callback
import time


def simple_add_kernel_pallas(x_ref, y_ref, o_ref):
    """Pallas kernel that adds two arrays."""
    o_ref[...] = x_ref[...] + y_ref[...]


def simple_add_kernel_numpy(x_np, y_np, o_np):
    """Pure NumPy version of the kernel."""
    o_np[:] = x_np + y_np


def pallas_with_numpy_callback(x, y):
    """
    Version that uses io_callback to execute the kernel in pure NumPy.

    This demonstrates the concept of using io_callback to convert JAX arrays
    to NumPy, execute in pure NumPy, then convert back.
    """
    def numpy_kernel_wrapper(x_jax, y_jax):
        # Convert to numpy
        x_np = np.asarray(x_jax)
        y_np = np.asarray(y_jax)
        o_np = np.zeros_like(x_np)

        # Execute in pure NumPy (stateful)
        simple_add_kernel_numpy(x_np, y_np, o_np)

        # Return as JAX array
        return o_np

    # Use io_callback to execute NumPy code
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return io_callback(
        numpy_kernel_wrapper,
        result_shape,
        x, y
    )


def compare_approaches(x, y):
    """Compare standard Pallas interpret mode vs NumPy callback."""

    print("=" * 60)
    print("Comparing Pallas interpret vs NumPy callback")
    print("=" * 60)

    # 1. Standard Pallas with interpret=True
    print("\n1. Standard Pallas (interpret=True):")
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

    start = time.time()
    result_pallas = pl.pallas_call(
        simple_add_kernel_pallas,
        out_shape=out_shape,
        interpret=True,
    )(x, y)
    result_pallas = jax.block_until_ready(result_pallas)
    end = time.time()
    print(f"   Time: {(end - start)*1000:.2f}ms")
    print(f"   Result shape: {result_pallas.shape}")

    # 2. Pure NumPy via io_callback
    print("\n2. Pure NumPy via io_callback:")
    start = time.time()
    result_numpy = pallas_with_numpy_callback(x, y)
    result_numpy = jax.block_until_ready(result_numpy)
    end = time.time()
    print(f"   Time: {(end - start)*1000:.2f}ms")
    print(f"   Result shape: {result_numpy.shape}")

    # 3. Compare results
    print("\n3. Results match:")
    print(f"   Pallas vs NumPy: {jnp.allclose(result_pallas, result_numpy)}")
    print(f"   Expected (x+y): {jnp.allclose(result_pallas, x + y)}")


if __name__ == "__main__":
    # Create test data
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 128), dtype=jnp.float32)
    y = jax.random.normal(jax.random.PRNGKey(1), (8, 128), dtype=jnp.float32)

    compare_approaches(x, y)

    print("\n" + "=" * 60)
    print("This demonstrates the concept of using io_callback to")
    print("convert Pallas operations to pure NumPy execution.")
    print("=" * 60)
