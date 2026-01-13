#!/usr/bin/env python3
"""
Explore gather and scatter operations in StableHLO to implement proper support.

This file analyzes how JAX's gather and scatter operations lower to StableHLO,
then implements proper decompilation back to jax.lax operations.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


def explore_gather():
    """Explore how JAX gather lowers to StableHLO."""
    print("=" * 80)
    print("EXPLORING JAX GATHER")
    print("=" * 80)

    # Simple 1D gather
    print("\n1. Simple 1D gather:")
    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = jnp.array([[0], [2], [4]])

    def gather_1d(operand, indices):
        dimension_numbers = lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(0,),
            start_index_map=(0,)
        )
        return lax.gather(operand, indices, dimension_numbers, slice_sizes=(1,))

    result = gather_1d(operand, indices)
    print(f"Operand: {operand}")
    print(f"Indices: {indices}")
    print(f"Result: {result}")

    # Lower to StableHLO
    lowered = jax.jit(gather_1d).lower(operand, indices)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')
    mlir_str = str(mlir_module)

    # Extract gather operation
    print("\nStableHLO gather operation:")
    for line in mlir_str.split('\n'):
        if 'gather' in line.lower() and 'stablehlo' in line:
            print(line.strip())

    # Show dimension numbers
    if 'offset_dims' in mlir_str:
        print("\nFound dimension numbers in IR")
        for line in mlir_str.split('\n'):
            if 'offset_dims' in line or 'collapsed' in line or 'start_index' in line:
                print(line.strip())

    # Simple 2D gather
    print("\n" + "=" * 80)
    print("2. 2D gather (like take_along_axis):")
    operand = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    indices = jnp.array([[0, 2], [1, 0]])

    def gather_2d(operand, indices):
        return jnp.take_along_axis(operand, indices, axis=1)

    result = gather_2d(operand, indices)
    print(f"Operand shape: {operand.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Result: {result}")

    lowered = jax.jit(gather_2d).lower(operand, indices)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')
    mlir_str = str(mlir_module)

    print("\nStableHLO operations:")
    for line in mlir_str.split('\n'):
        if 'stablehlo.gather' in line:
            print(line.strip())


def explore_scatter():
    """Explore how JAX scatter lowers to StableHLO."""
    print("\n" + "=" * 80)
    print("EXPLORING JAX SCATTER")
    print("=" * 80)

    # Simple 1D scatter
    print("\n1. Simple 1D scatter:")
    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = jnp.array([[1], [3]])
    updates = jnp.array([10.0, 20.0])

    def scatter_1d(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)
        )
        return lax.scatter(operand, indices, updates, dimension_numbers)

    result = scatter_1d(operand, indices, updates)
    print(f"Operand: {operand}")
    print(f"Indices: {indices}")
    print(f"Updates: {updates}")
    print(f"Result: {result}")

    # Lower to StableHLO
    lowered = jax.jit(scatter_1d).lower(operand, indices, updates)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')
    mlir_str = str(mlir_module)

    # Extract scatter operation
    print("\nStableHLO scatter operation:")
    for line in mlir_str.split('\n'):
        if 'scatter' in line.lower() and 'stablehlo' in line:
            print(line.strip())


def test_simple_gather():
    """Test a simple gather that we can decompile."""
    print("\n" + "=" * 80)
    print("TESTING SIMPLE GATHER FOR DECOMPILATION")
    print("=" * 80)

    def simple_gather(operand, indices):
        # Most basic gather - just indexing
        return jnp.take(operand, indices, axis=0)

    operand = jnp.array([10.0, 20.0, 30.0, 40.0])
    indices = jnp.array([0, 2, 1])

    result = simple_gather(operand, indices)
    print(f"Operand: {operand}")
    print(f"Indices: {indices}")
    print(f"Result: {result}")

    # Show full MLIR
    lowered = jax.jit(simple_gather).lower(operand, indices)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')
    print("\nFull StableHLO IR:")
    print(mlir_module)


def test_dynamic_slice_vs_gather():
    """Compare dynamic_slice (which works) with gather."""
    print("\n" + "=" * 80)
    print("COMPARING DYNAMIC_SLICE VS GATHER")
    print("=" * 80)

    # Dynamic slice (works)
    def use_dynamic_slice(operand, start):
        return lax.dynamic_slice(operand, (start,), (2,))

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    start = jnp.array(1)

    result = use_dynamic_slice(operand, start)
    print(f"Dynamic slice result: {result}")

    lowered = jax.jit(use_dynamic_slice).lower(operand, start)
    mlir = lowered.compiler_ir(dialect='stablehlo')
    print("Uses dynamic_slice:", 'dynamic_slice' in str(mlir))
    print("Uses gather:", 'gather' in str(mlir))


if __name__ == '__main__':
    explore_gather()
    explore_scatter()
    test_simple_gather()
    test_dynamic_slice_vs_gather()
