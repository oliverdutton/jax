#!/usr/bin/env python3
"""Simple test to debug the decompiler."""

import jax
import jax.numpy as jnp
from hlo_to_jaxpr import StableHLOToJaxpr

def simple_add(x, y):
    return x + y

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

lowered = jax.jit(simple_add).lower(x, y)
mlir_module = lowered.compiler_ir(dialect='stablehlo')

# Print the MLIR to understand structure
print("MLIR Module:")
print(mlir_module)
print("\n" + "=" * 80 + "\n")

# Now decompile
decompiler = StableHLOToJaxpr()

# Manually walk through to debug
for op in mlir_module.body.operations:
    if hasattr(op, 'function_type'):
        print(f"Function: {op.name}")
        print(f"Type: {op.function_type}")

        for region in op.regions:
            for block in region.blocks:
                print(f"\nBlock arguments ({len(list(block.arguments))}):")
                for i, arg in enumerate(block.arguments):
                    var = decompiler._get_var(arg)
                    print(f"  arg[{i}]: id={id(arg)}, type={arg.type}, var={var}")

                print(f"\nBlock operations:")
                for block_op in block.operations:
                    print(f"\n  Op: {block_op.operation.name}")

                    # Operands
                    if hasattr(block_op, 'operands'):
                        operands = list(block_op.operands)
                        print(f"  Operands: {len(operands)}")
                        for i, operand in enumerate(operands):
                            print(f"    operand[{i}]: id={id(operand)}, type={operand.type}")
                            if id(operand) in decompiler.value_map:
                                print(f"      -> mapped to var: {decompiler.value_map[id(operand)]}")
                            else:
                                print(f"      -> NOT IN VALUE MAP!")

                    # Results
                    if hasattr(block_op, 'results'):
                        results = list(block_op.results)
                        print(f"  Results: {len(results)}")
                        for i, result in enumerate(results):
                            var = decompiler._get_var(result)
                            print(f"    result[{i}]: id={id(result)}, type={result.type}, var={var}")
