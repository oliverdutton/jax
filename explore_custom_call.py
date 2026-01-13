#!/usr/bin/env python3
"""
Explore custom_call operations in StableHLO.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

# Test operations that typically use custom_call

print("="*80)
print("TEST 1: Random number generation (uses custom_call)")
print("="*80)
def f_rng(key):
    return jax.random.uniform(key, (4,))

key = jax.random.PRNGKey(0)
jitted = jax.jit(f_rng)
lowered = jitted.lower(key)
mlir = lowered.compiler_ir(dialect='stablehlo')
mlir_str = str(mlir)

# Look for custom_call
if 'custom_call' in mlir_str.lower():
    print("✓ Found custom_call in RNG")
    for line in mlir_str.split('\n'):
        if 'custom_call' in line.lower():
            print(f"  {line}")
else:
    print("✗ No custom_call found")

print("\n" + "="*80)
print("TEST 2: LU decomposition (linear algebra custom_call)")
print("="*80)
def f_lu(x):
    return jnp.linalg.lu(x)

x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
try:
    jitted = jax.jit(f_lu)
    lowered = jitted.lower(x)
    mlir = lowered.compiler_ir(dialect='stablehlo')
    mlir_str = str(mlir)

    if 'custom_call' in mlir_str.lower():
        print("✓ Found custom_call in LU")
        for line in mlir_str.split('\n'):
            if 'custom_call' in line.lower() or 'call_target' in line.lower():
                print(f"  {line[:150]}")
    else:
        print("✗ No custom_call found")
        print("First 500 chars of MLIR:")
        print(mlir_str[:500])
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*80)
print("TEST 3: FFT (might use custom_call)")
print("="*80)
def f_fft(x):
    return jnp.fft.fft(x)

x = jnp.array([1.0, 2.0, 3.0, 4.0])
jitted = jax.jit(f_fft)
lowered = jitted.lower(x)
mlir = lowered.compiler_ir(dialect='stablehlo')
mlir_str = str(mlir)

if 'custom_call' in mlir_str.lower():
    print("✓ Found custom_call in FFT")
    for line in mlir_str.split('\n'):
        if 'custom_call' in line.lower():
            print(f"  {line[:150]}")
else:
    print("✗ No custom_call found")
    # FFT is usually compiled to stablehlo.fft, not custom_call
    for line in mlir_str.split('\n'):
        if 'fft' in line.lower():
            print(f"  {line[:150]}")

print("\n" + "="*80)
print("TEST 4: Top-K (might use custom_call)")
print("="*80)
def f_topk(x):
    return lax.top_k(x, 2)

x = jnp.array([5.0, 2.0, 8.0, 1.0, 9.0])
jitted = jax.jit(f_topk)
lowered = jitted.lower(x)
mlir = lowered.compiler_ir(dialect='stablehlo')
mlir_str = str(mlir)

if 'custom_call' in mlir_str.lower():
    print("✓ Found custom_call in top_k")
    for line in mlir_str.split('\n'):
        if 'custom_call' in line.lower() or 'call_target' in line.lower():
            print(f"  {line[:150]}")
else:
    print("✗ No custom_call found in top_k")
    # top_k uses chlo.top_k
    for line in mlir_str.split('\n'):
        if 'top_k' in line.lower():
            print(f"  {line[:150]}")
