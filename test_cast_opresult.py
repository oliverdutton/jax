#!/usr/bin/env python3
"""Test casting Value to OpResult."""

import jax
import jax.numpy as jnp
from jaxlib.mlir import ir

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

                    for i, operand in enumerate(op.operands):
                        print(f"\nOperand {i}:")
                        print(f"  Type: {type(operand)}")

                        # Try to check if it's an OpResult
                        try:
                            if ir.OpResult.isinstance(operand):
                                print(f"  Is OpResult: True")
                                opresult = ir.OpResult(operand)
                                print(f"  result_number: {opresult.result_number}")
                            else:
                                print(f"  Is OpResult: False")
                        except Exception as e:
                            print(f"  Error checking OpResult: {e}")
