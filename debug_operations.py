#!/usr/bin/env python3
"""Debug script to investigate failing operations."""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

# Test argmax to see what MLIR it generates
print("="*80)
print("ARGMAX MLIR")
print("="*80)
x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
def f(x):
    return lax.argmax(x, axis=1, index_dtype=jnp.int32)

jitted = jax.jit(f)
lowered = jitted.lower(x)
mlir = lowered.compiler_ir(dialect='stablehlo')
print(str(mlir)[:2000])

# Test scatter to see what's happening
print("\n" + "="*80)
print("SCATTER MLIR")
print("="*80)
x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
indices = jnp.array([[0, 1], [1, 2]])
updates = jnp.array([10.0, 20.0])

scatter_dnums = lax.ScatterDimensionNumbers(
    update_window_dims=(),
    inserted_window_dims=(0, 1),
    scatter_dims_to_operand_dims=(0, 1)
)

def f_scatter(x):
    return lax.scatter(x, indices, updates, scatter_dnums)

jitted = jax.jit(f_scatter)
lowered = jitted.lower(x)
mlir = lowered.compiler_ir(dialect='stablehlo')
print(str(mlir)[:2000])

# Test select
print("\n" + "="*80)
print("SELECT MLIR")
print("="*80)
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

def f_select(x, y):
    return lax.select(jnp.array([True, False, True]), x, y)

jitted = jax.jit(f_select)
lowered = jitted.lower(x, y)
mlir = lowered.compiler_ir(dialect='stablehlo')
print(str(mlir)[:1500])
