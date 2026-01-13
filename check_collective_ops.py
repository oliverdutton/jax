#!/usr/bin/env python3
"""
Check if there are any collective operations in StableHLO that we need to support.
"""

import jax
import jax.numpy as jnp
from jax import lax

# Test various operations that might generate collective ops in distributed settings

def test_sum_reduction():
    """Test if sum reduction has special ops."""
    def f(x):
        return jnp.sum(x, axis=0, keepdims=True)

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    lowered = jax.jit(f).lower(x)
    mlir = str(lowered.compiler_ir(dialect='stablehlo'))

    print("="*80)
    print("Sum Reduction")
    print("="*80)
    for line in mlir.split('\n'):
        if 'stablehlo' in line and not line.strip().startswith('//'):
            print(line)


def test_broadcast():
    """Test broadcast operations."""
    def f(x):
        return jnp.broadcast_to(x[0], (10,))

    x = jnp.array([1.0, 2.0, 3.0])
    lowered = jax.jit(f).lower(x)
    mlir = str(lowered.compiler_ir(dialect='stablehlo'))

    print("\n" + "="*80)
    print("Broadcast")
    print("="*80)
    for line in mlir.split('\n'):
        if 'stablehlo' in line and not line.strip().startswith('//'):
            print(line)


# Run tests
test_sum_reduction()
test_broadcast()

print("\n" + "="*80)
print("Conclusion")
print("="*80)
print("In single-device mode, JAX doesn't generate explicit collective ops.")
print("Collective ops (all_reduce, all_gather, etc.) only appear in multi-device")
print("compilation with pmap or shard_map.")
print("\nOur decompiler should handle standard StableHLO ops, which is sufficient")
print("for single-device decompilation. Multi-device collectives would need")
print("special handling if they appear in the IR.")
