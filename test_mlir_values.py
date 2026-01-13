#!/usr/bin/env python3
"""Test if MLIR values are unique objects."""

import jax
import jax.numpy as jnp

def simple_topk(x):
    return jax.lax.top_k(x, k=3)

x = jnp.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0])

# Get MLIR
lowered = jax.jit(simple_topk).lower(x)
mlir_module = lowered.compiler_ir(dialect='stablehlo')

# Parse to find the top_k operation
for module_op in mlir_module.body.operations:
    for region in module_op.regions:
        for block in region.blocks:
            print(f"Block: {block}")
            print(f"Operations: {list(block.operations)}")

            topk_op = None
            return_op = None
            for op in block.operations:
                if 'top_k' in str(op.operation.name):
                    topk_op = op
                if 'return' in str(op.operation.name):
                    return_op = op

            if topk_op and return_op:
                print(f"\nTop-K operation: {topk_op.operation.name}")
                print(f"Top-K results: {list(topk_op.results)}")
                print(f"Result 0: {topk_op.results[0]}")
                print(f"Result 1: {topk_op.results[1]}")
                print(f"Result 0 str: '{str(topk_op.results[0])}'")
                print(f"Result 1 str: '{str(topk_op.results[1])}'")
                print(f"Result 0 id: {id(topk_op.results[0])}")
                print(f"Result 1 id: {id(topk_op.results[1])}")
                print(f"Are they the same object? {topk_op.results[0] is topk_op.results[1]}")

                print(f"\nReturn operation: {return_op.operation.name}")
                print(f"Return operands: {list(return_op.operands)}")
                print(f"Operand 0: {return_op.operands[0]}")
                print(f"Operand 1: {return_op.operands[1]}")
                print(f"Operand 0 str: '{str(return_op.operands[0])}'")
                print(f"Operand 1 str: '{str(return_op.operands[1])}'")
                print(f"Operand 0 id: {id(return_op.operands[0])}")
                print(f"Operand 1 id: {id(return_op.operands[1])}")

                print(f"\nIs operand 0 the same as result 0? {return_op.operands[0] is topk_op.results[0]}")
                print(f"Is operand 1 the same as result 1? {return_op.operands[1] is topk_op.results[1]}")
