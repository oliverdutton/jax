#!/usr/bin/env python3
"""
Verify that INTEGER * 0 gets optimized in HLO.
"""

import jax
import jax.numpy as jnp

print("="*70)
print("Verifying INTEGER * 0 optimization in HLO")
print("="*70)

# Test 1: Integer array * 0 (constant zero)
def f_int(x):
    return x * 0

x_int = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)

print("\n--- Test 1: int32 array * 0 ---")
print(f"Input: {x_int}")
result = jax.jit(f_int)(x_int)
print(f"Result: {result}")

print("\nHLO IR:")
lowered = jax.jit(f_int).lower(x_int)
hlo_text = lowered.as_text()
print(hlo_text)

# Check if optimization happened
if "multiply" in hlo_text.lower():
    print("\n❌ MULTIPLY OPERATION FOUND - Not fully optimized")
else:
    print("\n✓ NO MULTIPLY OPERATION - Optimized away!")

if "constant" in hlo_text.lower() and "0" in hlo_text:
    print("✓ CONSTANT ZERO FOUND - Likely optimized to constant")

print("\n" + "="*70)

# Test 2: Floating point for comparison
def f_float(x):
    return x * 0.0

x_float = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)

print("\n--- Test 2: float32 array * 0 (for comparison) ---")
print(f"Input: {x_float}")
result_float = jax.jit(f_float)(x_float)
print(f"Result: {result_float}")

print("\nHLO IR:")
lowered_float = jax.jit(f_float).lower(x_float)
hlo_text_float = lowered_float.as_text()
print(hlo_text_float)

if "multiply" in hlo_text_float.lower():
    print("\n⚠ MULTIPLY OPERATION FOUND - Float may not optimize due to NaN/Inf")
else:
    print("\n✓ NO MULTIPLY OPERATION - Optimized away!")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Integer result: {result}")
print(f"Float result: {result_float}")
print("\nKey observation:")
print("Check if the HLO for integers shows optimization while floats may not.")
