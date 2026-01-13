#!/usr/bin/env python3
"""Debug argmax decompilation."""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr

x_int = jnp.array([[1, 5, 3], [4, 2, 6]], dtype=jnp.int32)

def f(x):
    return lax.argmax(x, axis=1, index_dtype=jnp.int32)

# Get expected result
expected = f(x_int)
print(f"Expected: {expected}")

# Compile and decompile
jitted_fn = jax.jit(f)
lowered = jitted_fn.lower(x_int)
mlir_module = lowered.compiler_ir(dialect='stablehlo')

print("\nMLIR:")
print(str(mlir_module))

decompiler = StableHLOToJaxpr()
functions = decompiler.decompile_module(mlir_module)

main_func = functions.get('"main"')
if main_func:
    result = main_func.callable_fn(x_int)
    print(f"\nDecompiled: {result}")
    print(f"Match: {np.allclose(result, expected)}")
