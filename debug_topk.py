#!/usr/bin/env python3
"""Debug top_k decompilation."""

import jax
import jax.numpy as jnp
from hlo_to_jaxpr import StableHLOToJaxpr

def simple_topk(x):
    return jax.lax.top_k(x, k=3)

x = jnp.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0])

# Get expected result
expected_vals, expected_idxs = simple_topk(x)
print(f"Expected values: {expected_vals}")
print(f"Expected indices: {expected_idxs}")

# Get MLIR
lowered = jax.jit(simple_topk).lower(x)
mlir_module = lowered.compiler_ir(dialect='stablehlo')
print("\nMLIR:")
print(mlir_module)

# Try to decompile
decompiler = StableHLOToJaxpr()
functions = decompiler.decompile_module(mlir_module)

main_func = functions.get('"main"')
result = main_func.callable_fn(x)
print(f"\nResult type: {type(result)}")
print(f"Result: {result}")

if isinstance(result, tuple):
    result_vals, result_idxs = result
    print(f"Decompiled values: {result_vals}")
    print(f"Decompiled indices: {result_idxs}")
