"""Compare HLO and NumPy interpreter outputs step by step."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax._src import core
from jax.interpreters import partial_eval as pe

# Simple sort kernel to debug
def bitonic_sort_kernel(x_ref, _):
    """Minimal kernel for debugging - just one compare-and-swap."""
    i = pl.program_id(0)

    # Load two adjacent elements
    a = x_ref[i, 0]
    b = x_ref[i, 1]

    # Compare and swap
    a_int = a.view(jnp.int32)
    b_int = b.view(jnp.int32)

    should_swap = a_int > b_int

    new_a = jnp.where(should_swap, b, a)
    new_b = jnp.where(should_swap, a, b)

    # Store back
    x_ref[i, 0] = new_a
    x_ref[i, 1] = new_b

def trace_interpreter_execution(jaxpr, *args, interpreter_name="Unknown"):
    """Execute jaxpr and trace all intermediate values."""
    print(f"\n{'='*60}")
    print(f"{interpreter_name} Interpreter Execution")
    print(f"{'='*60}")

    env = {}

    # Map input vars to args
    for var, arg in zip(jaxpr.invars, args):
        env[var] = arg
        print(f"Input {var}: {np.array(arg).flatten()[:5]}... (shape={np.array(arg).shape}, dtype={np.array(arg).dtype})")

    print()

    # Execute each equation
    for eqn_idx, eqn in enumerate(jaxpr.eqns):
        print(f"[{eqn_idx}] {eqn.primitive.name}")

        # Get input values
        invals = [env[v] for v in eqn.invars]
        for i, (var, val) in enumerate(zip(eqn.invars, invals)):
            val_arr = np.array(val)
            if val_arr.size <= 10:
                print(f"  in[{i}] {var}: {val_arr.flatten()}")
            else:
                print(f"  in[{i}] {var}: {val_arr.flatten()[:5]}... (shape={val_arr.shape}, dtype={val_arr.dtype})")

        if eqn.params:
            print(f"  params: {eqn.params}")

        # For debugging, we just print the operation
        # The actual execution happens in the interpreter
        print(f"  → outputs to: {eqn.outvars}")

    print(f"{'='*60}\n")

# Create test data - just 2 elements to keep it simple
test_data = np.array([[3.0, 1.0]], dtype=np.float32)
print(f"Input: {test_data}")

# Run with HLO interpreter
print("\n" + "="*60)
print("RUNNING WITH HLO INTERPRETER")
print("="*60)

@jax.jit
def sort_hlo(x):
    return pl.pallas_call(
        bitonic_sort_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(1,),
        interpret=True,
    )(x)

result_hlo = sort_hlo(test_data)
result_hlo = np.array(result_hlo)
print(f"\nHLO Result: {result_hlo}")
print(f"Correct: {result_hlo[0, 0] < result_hlo[0, 1]}")

# Run with NumPy interpreter
print("\n" + "="*60)
print("RUNNING WITH NUMPY INTERPRETER")
print("="*60)

import pallas_numpy_interpreter
pallas_numpy_interpreter.install_numpy_interpreter()

# Need to re-jit after installing interpreter
@jax.jit
def sort_numpy(x):
    return pl.pallas_call(
        bitonic_sort_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(1,),
        interpret=True,
    )(x)

result_numpy = sort_numpy(test_data)
result_numpy = np.array(result_numpy)
print(f"\nNumPy Result: {result_numpy}")
print(f"Correct: {result_numpy[0, 0] < result_numpy[0, 1]}")

# Compare
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Input:  {test_data}")
print(f"HLO:    {result_hlo}")
print(f"NumPy:  {result_numpy}")
print(f"Match:  {np.array_equal(result_hlo, result_numpy)}")
