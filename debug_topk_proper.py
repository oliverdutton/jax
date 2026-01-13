#!/usr/bin/env python3
"""Debug top_k decompilation properly."""

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

# Decompile with instrumentation
decompiler = StableHLOToJaxpr()

# Add some instrumentation to the decompiler
original_execute = decompiler._execute_operation

def instrumented_execute(op_name, op, value_dict):
    result = original_execute(op_name, op, value_dict)
    if 'top_k' in str(op_name):
        print(f"\nExecuting {op_name}")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        if isinstance(result, (list, tuple)):
            for i, r in enumerate(result):
                print(f"  Result[{i}]: {r}")
    return result

decompiler._execute_operation = instrumented_execute

functions = decompiler.decompile_module(mlir_module)
main_func = functions.get('"main"')

print("\n\nCalling decompiled function:")
result = main_func.callable_fn(x)
print(f"Final result type: {type(result)}")
print(f"Final result: {result}")

if isinstance(result, tuple):
    vals, idxs = result
    print(f"\nValues: {vals}")
    print(f"Indices: {idxs}")
