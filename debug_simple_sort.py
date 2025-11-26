"""Debug simple sort with detailed tracing."""

import numpy as np

# First test with HLO interpreter
print("="*60)
print("TESTING WITH HLO INTERPRETER")
print("="*60)

import jax
import jax.numpy as jnp
from sort import sort

# Very small test
test_data = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)
print(f"Input:  {test_data}")

result_hlo = sort(test_data, num_keys=1, interpret=True)
result_hlo = np.array(result_hlo)
print(f"Output: {result_hlo}")
print(f"Correct: {np.allclose(result_hlo, np.sort(test_data, axis=1))}")
print()

# Now test with NumPy interpreter
print("="*60)
print("TESTING WITH NUMPY INTERPRETER")
print("="*60)

import pallas_numpy_interpreter
pallas_numpy_interpreter.install_numpy_interpreter()

# Need to reimport after installing interpreter
import importlib
import sort as sort_module
importlib.reload(sort_module)
from sort import sort as sort_numpy

print(f"Input:  {test_data}")

result_numpy = sort_numpy(test_data, num_keys=1, interpret=True)
result_numpy = np.array(result_numpy)
print(f"Output: {result_numpy}")
print(f"Correct: {np.allclose(result_numpy, np.sort(test_data, axis=1))}")
print()

# Compare
print("="*60)
print("COMPARISON")
print("="*60)
print(f"Input:    {test_data}")
print(f"Expected: {np.sort(test_data, axis=1)}")
print(f"HLO:      {result_hlo}")
print(f"NumPy:    {result_numpy}")
print(f"HLO correct:   {np.allclose(result_hlo, np.sort(test_data, axis=1))}")
print(f"NumPy correct: {np.allclose(result_numpy, np.sort(test_data, axis=1))}")
