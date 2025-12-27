"""Test Pallas kernel with shape (17,128), grid (3,), block (8,128)"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_one_kernel(x_ref, o_ref):
  """Kernel that adds 1 to input"""
  o_ref[...] = x_ref[...] + 1

def add_one(x):
  """Pallas call with grid (3,) iterating over dim0 with block (8,128)"""
  return pl.pallas_call(
      add_one_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      grid=(3,),
      in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
      out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
  )(x)

if __name__ == "__main__":
  # Create input array of shape (17, 128)
  x = jnp.arange(17 * 128, dtype=jnp.float32).reshape(17, 128)

  # Run the kernel
  result = add_one(x)

  # Verify
  expected = x + 1
  print(f"Input shape: {x.shape}")
  print(f"Output shape: {result.shape}")
  print(f"Match: {jnp.allclose(result, expected)}")
  print(f"Max diff: {jnp.max(jnp.abs(result - expected))}")
  print(f"\nFirst few elements of result[16, :5]: {result[16, :5]}")
  print(f"Expected result[16, :5]: {expected[16, :5]}")
