"""Directly compare HLO and NumPy outputs from the actual sort operation."""

import jax
import jax.numpy as jnp
import numpy as np
from sort import sort
import pallas_numpy_interpreter

print("="*80)
print("DIRECT HLO vs NumPy COMPARISON ON ACTUAL SORT")
print("="*80)

test_data = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)

print(f"\nInput: {test_data}")

# Test with HLO
print("\n" + "="*80)
print("RUNNING WITH HLO INTERPRETER")
print("="*80)
result_hlo = sort(test_data, num_keys=1, interpret=True)
result_hlo = np.array(result_hlo)
print(f"Output: {result_hlo[0,0,:3]}")

# Test with NumPy - but first disable all debug flags
print("\n" + "="*80)
print("RUNNING WITH NUMPY INTERPRETER")
print("="*80)

# Disable all debug output
import pallas_numpy_interpreter as pni
# Reinstall to reset state
pni.install_numpy_interpreter()

# Reload sort module to pick up NumPy interpreter
import importlib
import sort as sort_module
importlib.reload(sort_module)

result_numpy = sort_module.sort(test_data, num_keys=1, interpret=True)
result_numpy = np.array(result_numpy)
print(f"Output: {result_numpy[0,0,:3]}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"HLO:   {result_hlo[0,0,:5]}")
print(f"NumPy: {result_numpy[0,0,:5]}")
print(f"Match: {np.allclose(result_hlo, result_numpy)}")

if not np.allclose(result_hlo, result_numpy):
    print("\n❌ OUTPUTS DIFFER!")
    print(f"HLO unique values: {np.unique(result_hlo[0,0,~np.isnan(result_hlo[0,0,:])])}")
    print(f"NumPy unique values: {np.unique(result_numpy[0,0,~np.isnan(result_numpy[0,0,:])])}")
else:
    print("\n✅ OUTPUTS MATCH!")

print("\n" + "="*80)
print("NOW TESTING INDIVIDUAL PRIMITIVE IMPLEMENTATIONS")
print("="*80)

# Test the critical primitives directly using NumPy
print("\n--- Testing BITWISE AND ---")
a = np.arange(10, dtype=np.int32)
print(f"Input: {a}")
print(f"a & 1 = {a & 1}")
print(f"Expected: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]")
print(f"Match: {np.array_equal(a & 1, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])}")

print("\n--- Testing BITWISE XOR ---")
a = np.array([0, 1, 0, 1], dtype=np.int32)
b = np.array([0, 0, 1, 1], dtype=np.int32)
print(f"a: {a}")
print(f"b: {b}")
print(f"a ^ b = {a ^ b}")
print(f"Expected: [0, 1, 1, 0]")
print(f"Match: {np.array_equal(a ^ b, [0, 1, 1, 0])}")

print("\n--- Testing SELECT_N (np.choose) ---")
which = np.array([0, 1, 0, 1], dtype=np.int32)
cases = [
    np.array([10, 20, 30, 40], dtype=np.int32),
    np.array([50, 60, 70, 80], dtype=np.int32)
]
result = np.choose(which, cases)
print(f"which: {which}")
print(f"case0: {cases[0]}")
print(f"case1: {cases[1]}")
print(f"result: {result}")
print(f"Expected: [10, 60, 30, 80]")
print(f"Match: {np.array_equal(result, [10, 60, 30, 80])}")

print("\n--- Testing ADVANCED INDEXING (gather-like) ---")
operand = np.array([
    [3.0, np.nan, np.nan],
    [1.0, np.nan, np.nan],
    [2.0, np.nan, np.nan],
], dtype=np.float32)
indices = np.array([
    [1, 0, 0],  # Row 0: select from row 1
    [0, 0, 0],  # Row 1: select from row 0
    [0, 0, 0],  # Row 2: select from row 0
], dtype=np.int32)
batch_idx = np.arange(3)[None, :]
result = operand[indices, batch_idx]
print(f"operand[:,0]: {operand[:,0]}")
print(f"indices[:,0]: {indices[:,0]}")
print(f"result[:,0]: {result[:,0]}")
print(f"Expected result[:,0]: [1.0, 3.0, 3.0]")
print(f"Match: {np.allclose(result[:,0], [1.0, 3.0, 3.0], equal_nan=True)}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If all primitive tests above pass, the bug is NOT in the primitive")
print("implementations themselves, but in how they interact in the jaxpr.")
print("This suggests the bug might be in:")
print("  1. Ref mutation/aliasing causing unexpected data sharing")
print("  2. Dtype conversion issues (float32 <-> int32 bitcast)")
print("  3. Array view vs copy semantics")
