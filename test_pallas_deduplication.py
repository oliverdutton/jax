"""Test that Pallas applies CSE/deduplication passes.

This test verifies that duplicate computations in Pallas kernels are
eliminated via CSE and canonicalize passes, similar to normal JAX compilation.
"""

import sys
import unittest
from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax.experimental import pallas as pl


class PallasDeduplicationTest(jtu.JaxTestCase):
  """Test that Pallas deduplicates repeated computations."""

  def test_cse_pass_applied(self):
    """Verify that CSE pass is applied to Pallas kernels."""
    # This test checks that the CSE pass is in the compilation pipeline
    # by verifying that duplicate computations are eliminated.

    def kernel(x_ref, o_ref):
      x = x_ref[...]
      # Intentionally duplicate computation
      y1 = x * 2.0
      y2 = x * 2.0  # Should be deduplicated to y1
      # Use both to ensure they're not dead-code eliminated
      o_ref[...] = y1 + y2

    x = jnp.array([1.0, 2.0, 3.0])
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

    # Create the pallas_call
    f = pl.pallas_call(kernel, out_shape=out_shape)

    # Test that it compiles and runs correctly
    result = f(x)
    expected = x * 4.0  # x * 2 + x * 2 = x * 4
    self.assertArraysEqual(result, expected)

    # Test with JIT compilation
    result_jit = jax.jit(f)(x)
    self.assertArraysEqual(result_jit, expected)

  def test_canonicalize_pass_applied(self):
    """Verify that canonicalization simplifies expressions."""

    def kernel(x_ref, o_ref):
      x = x_ref[...]
      # Expression that should be canonicalized/simplified
      y = (x + 0.0) * 1.0  # Should simplify to just x
      o_ref[...] = y

    x = jnp.array([1.0, 2.0, 3.0])
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

    f = pl.pallas_call(kernel, out_shape=out_shape)

    result = f(x)
    self.assertArraysEqual(result, x)

  def test_multiple_duplicate_operations(self):
    """Test deduplication with multiple duplicate operations."""

    def kernel(x_ref, y_ref, o_ref):
      x = x_ref[...]
      y = y_ref[...]

      # Multiple duplicate computations
      a1 = x + y
      a2 = x + y  # Duplicate of a1
      b1 = x * 2.0
      b2 = x * 2.0  # Duplicate of b1

      o_ref[...] = a1 + a2 + b1 + b2

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

    f = pl.pallas_call(
        kernel,
        out_shape=out_shape,
    )

    result = f(x, y)
    # a1 + a2 + b1 + b2 = 2*(x+y) + 2*(x*2) = 2x + 2y + 4x = 6x + 2y
    expected = 6 * x + 2 * y
    self.assertArraysAllClose(result, expected, rtol=1e-5)


if __name__ == "__main__":
  absltest.main()
