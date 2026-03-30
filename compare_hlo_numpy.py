"""Compare HLO and NumPy interpreters by running each primitive and comparing results."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax._src import core as jax_core

# Simple test kernel
def simple_sort_kernel(x_ref, y_ref):
    """Very simple kernel that does a basic operation."""
    i = pl.program_id(0)
    # Just load, compute, and store
    val = x_ref[i, 0]
    y_ref[i, 0] = val

# Run with HLO to get jaxpr
print("="*60)
print("Extracting jaxpr from HLO execution...")
print("="*60)

test_data = np.array([[3.0, 1.0]], dtype=np.float32)

@jax.jit
def run_kernel_hlo(x):
    return pl.pallas_call(
        simple_sort_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(1,),
        interpret=True,
    )(x)

# Execute to get the lowered form
result_hlo = run_kernel_hlo(test_data)
print(f"HLO result: {result_hlo}")

# Now let's compare HLO and NumPy for the actual sort kernel
print("\n" + "="*60)
print("Comparing HLO vs NumPy for actual sort...")
print("="*60)

from sort import sort

# Small test case
test_data = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)
print(f"Input: {test_data}")

# HLO result
result_hlo = sort(test_data, num_keys=1, interpret=True)
result_hlo = np.array(result_hlo)
print(f"HLO result: {result_hlo}")

# NumPy result
import pallas_numpy_interpreter
pallas_numpy_interpreter.install_numpy_interpreter()

# Reimport to get NumPy version
import importlib
import sort as sort_module
importlib.reload(sort_module)
from sort import sort as sort_numpy

result_numpy = sort_numpy(test_data, num_keys=1, interpret=True)
result_numpy = np.array(result_numpy)
print(f"NumPy result: {result_numpy}")

# Compare
print(f"\nMatch: {np.allclose(result_hlo, result_numpy)}")
if not np.allclose(result_hlo, result_numpy):
    print(f"HLO:   {result_hlo[0, 0, :5]}")
    print(f"NumPy: {result_numpy[0, 0, :5]}")
