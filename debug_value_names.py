#!/usr/bin/env python3
"""Debug what value names are being generated."""

import jax
import jax.numpy as jnp
from hlo_to_jaxpr import StableHLOToJaxpr

def simple_topk(x):
    return jax.lax.top_k(x, k=3)

x = jnp.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0])

# Get MLIR
lowered = jax.jit(simple_topk).lower(x)
mlir_module = lowered.compiler_ir(dialect='stablehlo')

print("MLIR:")
print(mlir_module)
print("\n")

# Parse module
decompiler = StableHLOToJaxpr()

print("Iterating over operations:")
for module_op in mlir_module.body.operations:
    print(f"Module op: {module_op.operation.name}")
    for region in module_op.regions:
        print(f"  Region: {region}")
        for block in region.blocks:
            print(f"    Block: {block}")
            for func_op in block.operations:
                print(f"      Function op: {func_op.operation.name}")
                if 'func.func' in str(func_op.operation.name):
                    # Get the block
                    for region2 in func_op.regions:
                        for block2 in region2.blocks:
                            # Find the top_k operation
                            for op in block2.operations:
                                op_name = str(op.operation.name)
                                print(f"        Operation: {op_name}")

                                if 'top_k' in op_name:
                                    print(f"          Operands: {list(op.operands)}")
                                    print(f"          Results: {list(op.results)}")
                                    print(f"          Number of results: {len(list(op.results))}")

                                    for i, res in enumerate(op.results):
                                        result_name = decompiler._get_value_name(res)
                                        print(f"          Result {i} name: {result_name}")
                                        print(f"          Result {i} value: {res}")
