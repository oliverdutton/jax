#!/usr/bin/env python3
"""Debug cummin decompilation."""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr

x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

def f(x):
    return lax.cummin(x, axis=1)

# Get expected result
expected = f(x)
print(f"Expected:\n{expected}")

# Compile and decompile
jitted_fn = jax.jit(f)
lowered = jitted_fn.lower(x)
mlir_module = lowered.compiler_ir(dialect='stablehlo')

print("\nMLIR (key parts):")
for line in str(mlir_module).split('\n'):
    if 'reduce_window' in line or 'minimum' in line or 'padding' in line:
        print(line)

decompiler = StableHLOToJaxpr()
functions = decompiler.decompile_module(mlir_module)

main_func = functions.get('"main"')
if main_func:
    result = main_func.callable_fn(x)
    print(f"\nDecompiled:\n{result}")
    print(f"Match: {np.allclose(result, expected)}")
