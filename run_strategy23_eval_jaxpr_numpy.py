import time
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import core
from jax._src import dispatch
from jax.interpreters import mlir

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("Strategy: Hook core.eval_jaxpr to use NumPy + Python control flow")
print("=" * 60)

# Store original eval_jaxpr
_original_eval_jaxpr = core.eval_jaxpr

# Track how many times we've intercepted
_interception_count = 0
_primitive_counts = {}

def numpy_eval_jaxpr(jaxpr, consts, *args):
    """Custom JAXpr evaluator using NumPy and Python control flow."""
    global _interception_count, _primitive_counts

    _interception_count += 1

    # Build environment
    env = {}

    # Add constants
    def read(var):
        if isinstance(var, core.Literal):
            return var.val
        return env[var]

    def write(var, val):
        # Convert to NumPy if it's a JAX array
        if hasattr(val, '__array__'):
            val = np.asarray(val)
        env[var] = val

    # Map constants
    for c_var, c_val in zip(jaxpr.constvars, consts):
        write(c_var, c_val)

    # Map arguments
    for a_var, a_val in zip(jaxpr.invars, args):
        write(a_var, a_val)

    # Execute equations
    for eqn in jaxpr.eqns:
        prim = eqn.primitive

        # Track primitive usage
        _primitive_counts[prim.name] = _primitive_counts.get(prim.name, 0) + 1

        # Read inputs
        invals = [read(v) for v in eqn.invars]

        # Special handling for control flow primitives
        if prim.name in ('scan', 'while', 'cond'):
            # Use Python control flow
            outvals = _eval_control_flow_numpy(eqn, invals)
        elif prim.name == 'pjit':
            # Recursively evaluate pjit
            jaxpr_param = eqn.params.get('jaxpr', None)
            if jaxpr_param:
                outvals = numpy_eval_jaxpr(jaxpr_param.jaxpr, [], *invals)
            else:
                # Fallback to original
                outvals = prim.bind(*invals, **eqn.params)
        else:
            # Try to execute with NumPy
            try:
                # Convert NumPy arrays to JAX for primitive execution
                jax_invals = [jnp.asarray(v) if isinstance(v, np.ndarray) else v
                             for v in invals]

                # Execute primitive
                result = prim.bind(*jax_invals, **eqn.params)

                # Convert back to NumPy
                if hasattr(result, '__array__'):
                    outvals = np.asarray(result)
                else:
                    outvals = result
            except Exception as e:
                # Fallback to original evaluation
                outvals = prim.bind(*invals, **eqn.params)

        # Handle multiple results
        if not prim.multiple_results:
            outvals = [outvals]

        # Write outputs
        for v, val in zip(eqn.outvars, outvals):
            write(v, val)

    # Return outputs
    return [read(v) for v in jaxpr.outvars]

def _eval_control_flow_numpy(eqn, invals):
    """Execute control flow using Python."""
    prim_name = eqn.primitive.name

    if prim_name == 'scan':
        # Get parameters
        jaxpr = eqn.params['jaxpr'].jaxpr
        num_consts = eqn.params.get('num_consts', 0)
        num_carry = eqn.params.get('num_carry', 0)

        consts = invals[:num_consts]
        carry = invals[num_consts:num_consts + num_carry]
        xs = invals[num_consts + num_carry:]

        # Python for loop
        length = xs[0].shape[0] if xs else 0
        ys = []

        for i in range(length):
            x_i = [x[i] for x in xs]
            body_invals = consts + carry + x_i

            # Recursively evaluate
            results = numpy_eval_jaxpr(jaxpr, [], *body_invals)

            carry = results[:num_carry]
            y_i = results[num_carry:]

            if i == 0:
                ys = [[] for _ in y_i]
            for j, y_val in enumerate(y_i):
                ys[j].append(y_val)

        # Stack outputs
        ys = [np.stack(y) if y else np.array([]) for y in ys]
        return carry + ys

    elif prim_name == 'while':
        # Get jaxprs
        cond_jaxpr = eqn.params['cond_jaxpr'].jaxpr
        body_jaxpr = eqn.params['body_jaxpr'].jaxpr

        # Python while loop
        carry = list(invals)
        while True:
            cond_result = numpy_eval_jaxpr(cond_jaxpr, [], *carry)
            if not cond_result[0]:
                break
            carry = numpy_eval_jaxpr(body_jaxpr, [], *carry)

        return carry

    elif prim_name == 'cond':
        pred = invals[0]
        branches = eqn.params['branches']

        branch_idx = 0 if pred else 1
        jaxpr = branches[branch_idx].jaxpr

        return numpy_eval_jaxpr(jaxpr, [], *invals[1:])

    else:
        # Fallback
        return eqn.primitive.bind(*invals, **eqn.params)

# Monkey-patch core.eval_jaxpr
core.eval_jaxpr = numpy_eval_jaxpr

print("\n🔥 STRATEGY 23: Hook core.eval_jaxpr for NumPy Execution\n")
print("Intercepting JAXpr evaluation to use Python control flow + NumPy")

try:
    from benchmark_sort import run_benchmarks

    start_time = time.time()
    run_benchmarks()
    end_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print(f"JAXpr evaluations intercepted: {_interception_count}")
    print(f"\nTop 10 primitives executed:")
    for prim, count in sorted(_primitive_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {prim}: {count}")
    print(f"{'=' * 60}")

finally:
    # Restore original
    core.eval_jaxpr = _original_eval_jaxpr
