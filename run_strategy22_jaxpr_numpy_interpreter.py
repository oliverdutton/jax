import time
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import partial_eval as pe

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("Strategy: Interpret JAXpr with NumPy backend + Python control flow")
print("=" * 60)

# Custom JAXpr interpreter using NumPy
class NumpyInterpreter:
    """Interpret JAXpr using NumPy operations and Python control flow."""

    def __init__(self):
        self.env = {}

    def interpret_jaxpr(self, jaxpr, *args):
        """Main interpreter entry point."""
        # Map input variables to arguments
        self.env = {}
        for var, val in zip(jaxpr.invars, args):
            self.env[var] = self._to_numpy(val)

        # Process each equation
        for eqn in jaxpr.eqns:
            self._eval_eqn(eqn)

        # Return outputs
        return [self.env[var] for var in jaxpr.outvars]

    def _to_numpy(self, val):
        """Convert JAX array to NumPy."""
        if isinstance(val, jax.Array):
            return np.asarray(val)
        return val

    def _eval_eqn(self, eqn):
        """Evaluate a single JAXpr equation."""
        invals = [self.env[v] for v in eqn.invars]

        # Check if this is a control flow primitive
        if eqn.primitive.name in ('scan', 'while', 'cond', 'fori_loop'):
            # Use Python control flow
            outvals = self._eval_control_flow(eqn, invals)
        else:
            # Execute primitive with NumPy
            outvals = self._eval_primitive(eqn, invals)

        # Store outputs
        if not eqn.primitive.multiple_results:
            outvals = [outvals]

        for var, val in zip(eqn.outvars, outvals):
            self.env[var] = val

    def _eval_control_flow(self, eqn, invals):
        """Execute control flow primitives using Python."""
        prim_name = eqn.primitive.name

        if prim_name == 'scan':
            return self._eval_scan(eqn, invals)
        elif prim_name == 'while':
            return self._eval_while(eqn, invals)
        elif prim_name == 'cond':
            return self._eval_cond(eqn, invals)
        elif prim_name == 'fori_loop':
            return self._eval_fori_loop(eqn, invals)
        else:
            raise NotImplementedError(f"Control flow {prim_name} not implemented")

    def _eval_scan(self, eqn, invals):
        """Execute scan using Python for loop."""
        # Get the body function
        jaxpr = eqn.params['jaxpr']
        num_consts = eqn.params['num_consts']
        num_carry = eqn.params['num_carry']

        consts = invals[:num_consts]
        carry = invals[num_consts:num_consts + num_carry]
        xs = invals[num_consts + num_carry:]

        # Get length from first input array
        length = xs[0].shape[0] if len(xs) > 0 else eqn.params.get('length', 0)

        ys = []
        for i in range(length):
            # Get slice of inputs for this iteration
            x_slice = [x[i] for x in xs]

            # Call body function
            body_invals = consts + carry + x_slice
            body_outvals = self.interpret_jaxpr(jaxpr.jaxpr, *body_invals)

            # Split carry and output
            carry = body_outvals[:num_carry]
            y = body_outvals[num_carry:]

            if i == 0:
                ys = [[] for _ in y]
            for j, y_val in enumerate(y):
                ys[j].append(y_val)

        # Stack outputs
        ys = [np.stack(y_list) for y_list in ys]

        return carry + ys

    def _eval_fori_loop(self, eqn, invals):
        """Execute fori_loop using Python for loop."""
        jaxpr = eqn.params['jaxpr']
        lower, upper = invals[:2]
        body_args = invals[2:]

        result = list(body_args)
        for i in range(int(lower), int(upper)):
            # Call body with (i, *carry)
            body_invals = [i] + result
            result = self.interpret_jaxpr(jaxpr.jaxpr, *body_invals)

        return result

    def _eval_while(self, eqn, invals):
        """Execute while using Python while loop."""
        cond_jaxpr = eqn.params['cond_jaxpr']
        body_jaxpr = eqn.params['body_jaxpr']

        carry = list(invals)
        while True:
            # Evaluate condition
            cond_result = self.interpret_jaxpr(cond_jaxpr.jaxpr, *carry)
            if not cond_result[0]:
                break

            # Evaluate body
            carry = self.interpret_jaxpr(body_jaxpr.jaxpr, *carry)

        return carry

    def _eval_cond(self, eqn, invals):
        """Execute cond using Python if/else."""
        pred = invals[0]
        branches = eqn.params['branches']

        branch_idx = 0 if pred else 1
        jaxpr = branches[branch_idx]

        return self.interpret_jaxpr(jaxpr.jaxpr, *invals[1:])

    def _eval_primitive(self, eqn, invals):
        """Execute primitive operation with NumPy."""
        prim = eqn.primitive

        # Try to map JAX primitive to NumPy operation
        try:
            # Get the abstract evaluation rule to understand the operation
            if hasattr(prim, 'impl'):
                # Use JAX's implementation but on NumPy arrays
                return prim.impl(*invals, **eqn.params)
            else:
                # Try direct NumPy mapping
                return self._numpy_primitive(prim.name, invals, eqn.params)
        except Exception as e:
            # Fallback: let JAX handle it but with NumPy arrays
            print(f"Warning: falling back to JAX for {prim.name}: {e}")
            jax_invals = [jnp.asarray(v) for v in invals]
            result = prim.bind(*jax_invals, **eqn.params)
            return np.asarray(result)

    def _numpy_primitive(self, name, invals, params):
        """Map common JAX primitives to NumPy."""
        # Arithmetic
        if name == 'add': return np.add(invals[0], invals[1])
        elif name == 'sub': return np.subtract(invals[0], invals[1])
        elif name == 'mul': return np.multiply(invals[0], invals[1])
        elif name == 'div': return np.divide(invals[0], invals[1])

        # Comparisons
        elif name == 'gt': return np.greater(invals[0], invals[1])
        elif name == 'ge': return np.greater_equal(invals[0], invals[1])
        elif name == 'lt': return np.less(invals[0], invals[1])
        elif name == 'le': return np.less_equal(invals[0], invals[1])
        elif name == 'eq': return np.equal(invals[0], invals[1])

        # Array ops
        elif name == 'slice':
            start_indices = params.get('start_indices', ())
            limit_indices = params.get('limit_indices', ())
            slices = tuple(slice(s, l) for s, l in zip(start_indices, limit_indices))
            return invals[0][slices]

        elif name == 'concatenate':
            axis = params.get('dimension', 0)
            return np.concatenate(invals, axis=axis)

        elif name == 'reshape':
            new_sizes = params.get('new_sizes', params.get('dimensions', ()))
            return np.reshape(invals[0], new_sizes)

        elif name == 'broadcast_in_dim':
            shape = params['shape']
            broadcast_dimensions = params.get('broadcast_dimensions', ())
            # Implement broadcasting
            result = np.broadcast_to(invals[0], shape)
            return result

        elif name == 'transpose':
            permutation = params.get('permutation', None)
            return np.transpose(invals[0], permutation)

        elif name == 'select':
            return np.where(invals[0], invals[1], invals[2])

        # Fallback
        else:
            raise NotImplementedError(f"Primitive {name} not implemented")

# Monkey-patch JAX's compilation to use our interpreter
_original_jit = jax.jit
_interpreter = NumpyInterpreter()

def numpy_jit(fun, **kwargs):
    """Replace jit with interpretation."""
    def wrapped(*args, **fn_kwargs):
        # Try to get the JAXpr
        try:
            # Trace the function to get JAXpr
            closed_jaxpr = jax.make_jaxpr(fun)(*args, **fn_kwargs)

            # Interpret using NumPy
            result = _interpreter.interpret_jaxpr(closed_jaxpr.jaxpr, *args)

            if len(result) == 1:
                return result[0]
            return tuple(result)
        except Exception as e:
            print(f"Warning: NumPy interpretation failed, using JAX: {e}")
            # Fallback to original JAX
            return _original_jit(fun, **kwargs)(*args, **fn_kwargs)

    return wrapped

# Install the monkey-patch
jax.jit = numpy_jit

print("\n🔥 STRATEGY 22: JAXpr NumPy Interpreter\n")
print("Intercepting JAX compilation and interpreting with NumPy")

from benchmark_sort import run_benchmarks

start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")

# Restore original
jax.jit = _original_jit
