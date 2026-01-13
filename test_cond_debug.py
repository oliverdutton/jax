#!/usr/bin/env python3
"""Debug conditional operations."""

import jax
import jax.numpy as jnp
from jax import lax

def f(x):
    return lax.cond(x < 0, lambda x: -x, lambda x: x, x)

x = jnp.array(-5.0)

lowered = jax.jit(f).lower(x)
mlir_module = lowered.compiler_ir(dialect='stablehlo')

print("StableHLO IR:")
print(mlir_module)
print("\n" + "="*80 + "\n")

# Now let's see what the structure looks like
for op in mlir_module.body.operations:
    if hasattr(op, 'function_type'):
        print(f"Function: {op.name}")

        for region in op.regions:
            for block in region.blocks:
                print(f"\nBlock with {len(list(block.arguments))} arguments")

                for block_op in block.operations:
                    print(f"\nOp: {block_op.operation.name}")

                    if 'case' in str(block_op.operation.name):
                        print("  Found case operation!")
                        print(f"  Operands: {len(list(block_op.operands))}")
                        for i, operand in enumerate(block_op.operands):
                            print(f"    operand[{i}]: {operand}")

                        print(f"  Regions: {len(list(block_op.regions))}")
                        for i, region in enumerate(block_op.regions):
                            print(f"  Region {i}:")
                            for block in region.blocks:
                                print(f"    Block with {len(list(block.arguments))} arguments:")
                                for j, arg in enumerate(block.arguments):
                                    print(f"      arg[{j}]: {arg}")
