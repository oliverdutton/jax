"""Compare HLO and NumPy interpreters equation by equation.

This runs both interpreters and logs the output of each equation,
then compares them to find where they first diverge.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
import pickle

# Test kernel - simple sort
def sort_kernel(x_ref, y_ref, scratch_ref1, scratch_ref2):
    """Simplified sort kernel."""
    # This will execute the problematic operations
    from sort import bitonic_sort_kernel_inner
    # Just use program_id to get position and do basic ops
    i = pl.program_id(0)
    j = pl.program_id(1)

    # Simple operations that should work
    val = x_ref[i, j]
    y_ref[i, j] = val

# First, run with HLO and capture equation results
print("="*60)
print("STEP 1: Run with HLO interpreter and log equation outputs")
print("="*60)

# We'll instrument the HLO interpreter to log outputs
# Since we can't easily do that, let's instead run NumPy with logging

# Better approach: Patch the NumPy interpreter to log each equation's result
import pallas_numpy_interpreter

equation_logs = []

original_eval = pallas_numpy_interpreter._eval_jaxpr_numpy_impl

def logging_eval(jaxpr, consts, *args, grid_env=None):
    """Wrapper that logs each equation's result."""
    if grid_env is None:
        grid_env = {}

    # Use the original implementation but intercept at equation level
    # Actually, let me just patch the equation execution directly
    from jax._src import core as jax_core

    # Read/write functions
    def read(v):
        return env[v]

    def write(v, val):
        if not isinstance(v, jax_core.Literal):
            if not val.flags.writeable:
                val = val.copy()
        env[v] = val

    env = {}

    # Initialize environment
    for v, c in zip(jaxpr.constvars, consts):
        write(v, np.asarray(c))
    for v, a in zip(jaxpr.invars, args):
        write(v, np.asarray(a))

    # Track depth
    depth = pallas_numpy_interpreter.eval_jaxpr_numpy._depth

    # Execute equations with logging
    for eqn_idx, eqn in enumerate(jaxpr.eqns):
        in_vals = [read(v) for v in eqn.invars]
        prim = eqn.primitive

        # Execute the equation using the original interpreter's logic
        # This is complex, so let me just call the original function
        # and capture its result

        # Skip logging for now, just call original
        pass

    # Actually, this is getting too complex. Let me use a simpler approach.
    return original_eval(jaxpr, consts, *args, grid_env=grid_env)

# Simpler approach: Just add detailed logging at key operations
# and run with small input to manually compare

print("\nRunning simplified test with detailed logging...")

# Enable all debug flags
import importlib
importlib.reload(pallas_numpy_interpreter)

# Patch to enable debugging
with open('/home/user/jax/pallas_numpy_interpreter.py', 'r') as f:
    code = f.read()

# Check which debug flags exist
print("\nAvailable debug flags:")
for line in code.split('\n'):
    if 'DEBUG' in line and '=' in line and 'False' in line:
        print(f"  {line.strip()}")

print("\n" + "="*60)
print("To do equation-by-equation comparison:")
print("="*60)
print("1. Enable DEBUG flags in pallas_numpy_interpreter.py")
print("2. Run with HLO: verify_hlo_sort.py > hlo_output.txt")
print("3. Run with NumPy: verify_numpy_sort.py > numpy_output.txt")
print("4. Compare the logs to find first divergence")
print()
print("Or use this script with logging enabled...")

# For now, let's at least run both and show the difference
from sort import sort

test_data = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)

print("\nHLO Result:")
result_hlo = sort(test_data, num_keys=1, interpret=True)
print(f"  {np.array(result_hlo)[0,0,:5]}")

print("\nNumPy Result:")
pallas_numpy_interpreter.install_numpy_interpreter()
importlib.reload(pallas_numpy_interpreter)

# Re-import sort after installing interpreter
import sort as sort_module
importlib.reload(sort_module)
result_numpy = sort_module.sort(test_data, num_keys=1, interpret=True)
print(f"  {np.array(result_numpy)[0,0,:5]}")

print("\nMatch:", np.allclose(result_hlo, result_numpy))
