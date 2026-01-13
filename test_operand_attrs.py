#!/usr/bin/env python3
"""Test operand attributes."""

import jax
import jax.numpy as jnp

def simple_topk(x):
    return jax.lax.top_k(x, k=3)

x = jnp.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0])

# Get MLIR
lowered = jax.jit(simple_topk).lower(x)
mlir_module = lowered.compiler_ir(dialect='stablehlo')

# Parse to find the return operation
for module_op in mlir_module.body.operations:
    for region in module_op.regions:
        for block in region.blocks:
            for op in block.operations:
                if 'return' in str(op.operation.name):
                    print(f"Return operation: {op.operation.name}")
                    print(f"Operands: {list(op.operands)}")

                    for i, operand in enumerate(op.operands):
                        print(f"\nOperand {i}:")
                        print(f"  Type: {type(operand)}")
                        print(f"  Str: {str(operand)}")
                        print(f"  Has result_number: {hasattr(operand, 'result_number')}")
                        if hasattr(operand, 'result_number'):
                            print(f"  result_number: {operand.result_number}")
                        if hasattr(operand, 'owner'):
                            print(f"  owner: {operand.owner}")
