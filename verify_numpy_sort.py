"""Verify NumPy interpreter produces correct sorted results."""

import pallas_numpy_interpreter
pallas_numpy_interpreter.install_numpy_interpreter()

import jax
import jax.numpy as jnp
import numpy as np

# Import the sort function
from sort import sort

# Create test data
np.random.seed(42)
test_data = np.random.randn(8, 128).astype(np.float32)
print(f"Input data shape: {test_data.shape}")
print(f"Input sample (first 10 of row 0): {test_data[0, :10]}")

# Sort using NumPy interpreter
result = sort(test_data, num_keys=1, interpret=True)
result = np.array(result)

print(f"\nOutput data shape: {result.shape}")
# Handle extra dimension if present
if result.ndim == 3:
    result = result[0]
    print(f"Squeezed to shape: {result.shape}")
print(f"Output sample (first 10 of row 0): {result[0, :10]}")

# Verify it's actually sorted
for i in range(result.shape[0]):
    row = result[i]
    is_sorted = np.all(row[:-1] <= row[1:])
    print(f"Row {i} sorted: {is_sorted}")
    if not is_sorted:
        # Find first unsorted position
        for j in range(len(row)-1):
            if row[j] > row[j+1]:
                print(f"  First violation at position {j}: {row[j]} > {row[j+1]}")
                break

# Overall verification
all_sorted = all(np.all(result[i, :-1] <= result[i, 1:]) for i in range(result.shape[0]))
print(f"\n{'✓' if all_sorted else '✗'} All rows correctly sorted: {all_sorted}")

# Compare with numpy sort
expected = np.sort(test_data, axis=1)
matches = np.allclose(result, expected)
print(f"{'✓' if matches else '✗'} Matches np.sort output: {matches}")

if not matches:
    max_diff = np.max(np.abs(result - expected))
    print(f"  Max difference: {max_diff}")
