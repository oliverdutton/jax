#!/usr/bin/env python3
"""Test if OpResult objects have a result number."""

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
            for op in block.operations:
                if 'top_k' in str(op.operation.name):
                    print(f"Top-K operation: {op.operation.name}")
                    print(f"Results: {list(op.results)}")

                    for i, res in enumerate(op.results):
                        print(f"\nResult {i}:")
                        print(f"  Type: {type(res)}")
                        print(f"  Dir: {[x for x in dir(res) if not x.startswith('_')]}")
                        print(f"  Str: {str(res)}")

                        # Check for result_number or similar attribute
                        if hasattr(res, 'result_number'):
                            print(f"  result_number: {res.result_number}")
                        if hasattr(res, 'index'):
                            print(f"  index: {res.index}")
                        if hasattr(res, 'number'):
                            print(f"  number: {res.number}")

                        # Try calling get_result_number if it exists
                        if hasattr(res, 'get_result_number'):
                            print(f"  get_result_number(): {res.get_result_number()}")
