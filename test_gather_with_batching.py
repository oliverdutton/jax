#!/usr/bin/env python3
"""Test gather with batching dimensions."""

import jax
import jax.numpy as jnp
from jax import lax

def test_gather_with_batching():
    """Test gather operation that uses batching dimensions."""

    # From tallax-like pattern
    operand = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    indices = jnp.array([[[0], [2]], [[1], [0]]])

    print("Operand shape:", operand.shape)
    print("Indices shape:", indices.shape)

    # This uses batching dimensions
    def gather_with_batch(operand, indices):
        dimension_numbers = lax.GatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(1,),
            start_index_map=(1,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,)
        )
        slice_sizes = (1, 1)
        return lax.gather(operand, indices, dimension_numbers, slice_sizes)

    result = gather_with_batch(operand, indices)
    print("Result shape:", result.shape)
    print("Result:", result)

    # Lower to StableHLO
    lowered = jax.jit(gather_with_batch).lower(operand, indices)
    mlir = lowered.compiler_ir(dialect='stablehlo')

    # Show the gather operation
    for line in str(mlir).split('\n'):
        if 'stablehlo.gather' in line:
            print("\nGather operation:")
            print(line)

    # Show dimension numbers
    mlir_str = str(mlir)
    if 'operand_batching_dims' in mlir_str:
        print("\nHas operand_batching_dims!")
        for line in mlir_str.split('\n'):
            if 'operand_batching_dims' in line or 'start_indices_batching_dims' in line:
                print(line.strip())

if __name__ == '__main__':
    test_gather_with_batching()
