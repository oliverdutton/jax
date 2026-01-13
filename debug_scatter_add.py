#!/usr/bin/env python3
"""Debug scatter add to see what the MLIR looks like."""

import jax
import jax.numpy as jnp

def scatter_add_test(operand, indices, updates):
    """Scatter with addition."""
    return operand.at[indices].add(updates)

# Test inputs
operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
indices = jnp.array([1, 3])
updates = jnp.array([10.0, 20.0])

# Get MLIR
lowered = jax.jit(scatter_add_test).lower(operand, indices, updates)
mlir_module = lowered.compiler_ir(dialect='stablehlo')
print(mlir_module)
