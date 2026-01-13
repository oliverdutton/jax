#!/usr/bin/env python3
"""Explore how to access scatter computation regions in MLIR."""

import jax
import jax.numpy as jnp
from hlo_to_jaxpr import StableHLOToJaxpr

def scatter_add_test(operand, indices, updates):
    """Scatter with addition."""
    return operand.at[indices].add(updates)

def scatter_mul_test(operand, indices, updates):
    """Scatter with multiplication."""
    return operand.at[indices].multiply(updates)

def scatter_min_test(operand, indices, updates):
    """Scatter with minimum."""
    return operand.at[indices].min(updates)

def scatter_max_test(operand, indices, updates):
    """Scatter with maximum."""
    return operand.at[indices].max(updates)

# Test inputs
operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
indices = jnp.array([1, 3])
updates = jnp.array([10.0, 20.0])

# Get MLIR for different scatter types
for name, func in [('add', scatter_add_test), ('mul', scatter_mul_test),
                    ('min', scatter_min_test), ('max', scatter_max_test)]:
    print(f"\n{'='*60}")
    print(f"Scatter {name}:")
    print('='*60)
    lowered = jax.jit(func).lower(operand, indices, updates)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    # Extract just the scatter operation
    mlir_str = str(mlir_module)
    lines = mlir_str.split('\n')
    in_scatter = False
    for line in lines:
        if 'stablehlo.scatter' in line:
            in_scatter = True
        if in_scatter:
            print(line)
            if 'stablehlo.return' in line:
                break
