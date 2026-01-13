#!/usr/bin/env python3
"""Explore how to programmatically access HLO from compiled JAX objects."""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

# Test 1: Simple arithmetic
def simple_add(x, y):
    return x + y

# Test 2: Control flow - scan
def cumsum_scan(arr):
    def body(carry, x):
        new_carry = carry + x
        return new_carry, new_carry
    _, result = lax.scan(body, 0.0, arr)
    return result

# Test 3: Control flow - while
def while_sum(n):
    def cond_fun(val):
        i, total = val
        return i < n
    def body_fun(val):
        i, total = val
        return (i + 1, total + i)
    _, result = lax.while_loop(cond_fun, body_fun, (0, 0))
    return result

# Test 4: Control flow - cond
def abs_value(x):
    return lax.cond(x >= 0, lambda x: x, lambda x: -x, x)

# Test 5: More complex ops
def complex_ops(x):
    return jnp.tanh(jnp.dot(x, x.T))


print("=" * 80)
print("EXPLORING JAX COMPILED OBJECTS AND HLO")
print("=" * 80)

# Compile functions
print("\n1. Simple Add")
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
lowered = jax.jit(simple_add).lower(x, y)
compiled = lowered.compile()

print("\nLowered object type:", type(lowered))
print("Compiled object type:", type(compiled))

# Access the StableHLO module
print("\n" + "=" * 80)
print("ACCESSING STABLEHLO IR PROGRAMMATICALLY")
print("=" * 80)

# Get the MLIR module
mlir_module = lowered.compiler_ir(dialect='stablehlo')
print("\nMLIR Module type:", type(mlir_module))
print("Module attributes:", dir(mlir_module))

# Try to explore the structure
print("\n" + "=" * 80)
print("EXPLORING MODULE STRUCTURE")
print("=" * 80)

print("\nModule operations:")
for op in mlir_module.body.operations:
    op_name = str(op.name)
    print(f"  Operation: {op_name}")
    print(f"  Type: {type(op)}")
    print(f"  Attributes: {[attr for attr in dir(op) if not attr.startswith('_')]}")

    # Explore function operations
    if 'func' in str(type(op)) or hasattr(op, 'function_type'):
        print(f"\n  Function details:")
        print(f"    Function type: {op.type}")
        print(f"    Number of regions: {len(list(op.regions))}")

        for region in op.regions:
            print(f"\n    Region blocks: {len(list(region.blocks))}")
            for block in region.blocks:
                print(f"      Block arguments: {len(list(block.arguments))}")
                for arg in block.arguments:
                    print(f"        Arg type: {arg.type}")

                print(f"      Block operations: {len(list(block.operations))}")
                for block_op in block.operations:
                    print(f"        Op: {block_op.name}")
                    if hasattr(block_op, 'operands'):
                        print(f"          Operands: {len(list(block_op.operands))}")
                    if hasattr(block_op, 'results'):
                        print(f"          Results: {len(list(block_op.results))}")
                    if hasattr(block_op, 'attributes'):
                        print(f"          Attributes: {block_op.attributes}")

print("\n" + "=" * 80)
print("TESTING WITH CONTROL FLOW")
print("=" * 80)

# Test scan
arr = jnp.array([1.0, 2.0, 3.0, 4.0])
lowered_scan = jax.jit(cumsum_scan).lower(arr)
mlir_scan = lowered_scan.compiler_ir(dialect='stablehlo')

print("\nScan function module:")
for op in mlir_scan.body.operations:
    if hasattr(op, 'function_type'):
        for region in op.regions:
            for block in region.blocks:
                for block_op in block.operations:
                    print(f"  {block_op.operation.name}")

print("\n" + "=" * 80)
print("EXAMINING OPERATION DETAILS")
print("=" * 80)

# Get more details about operations
def explore_operation(op, indent=0):
    prefix = "  " * indent
    print(f"{prefix}Op: {op.operation.name}")

    # Get operands
    if hasattr(op, 'operands'):
        operands = list(op.operands)
        if operands:
            print(f"{prefix}  Operands ({len(operands)}):")
            for i, operand in enumerate(operands):
                print(f"{prefix}    [{i}] type: {operand.type}")

    # Get results
    if hasattr(op, 'results'):
        results = list(op.results)
        if results:
            print(f"{prefix}  Results ({len(results)}):")
            for i, result in enumerate(results):
                print(f"{prefix}    [{i}] type: {result.type}")

    # Get attributes
    if hasattr(op, 'attributes'):
        attrs = dict(op.attributes)
        if attrs:
            print(f"{prefix}  Attributes:")
            for name, value in attrs.items():
                print(f"{prefix}    {name}: {value}")

    # Explore nested regions
    if hasattr(op, 'regions'):
        regions = list(op.regions)
        if regions:
            print(f"{prefix}  Regions ({len(regions)}):")
            for i, region in enumerate(regions):
                print(f"{prefix}    Region {i}:")
                for block in region.blocks:
                    for nested_op in block.operations:
                        explore_operation(nested_op, indent + 2)

# Explore the simple add function in detail
print("\nDetailed exploration of simple_add:")
for op in mlir_module.body.operations:
    if hasattr(op, 'function_type'):
        for region in op.regions:
            for block in region.blocks:
                for block_op in block.operations:
                    explore_operation(block_op)

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
