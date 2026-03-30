"""Find where HLO and NumPy diverge by adding result checksums."""

import numpy as np
import jax
import jax.numpy as jnp

# Modify the interpreter to add result logging
import pallas_numpy_interpreter

# Add logging to primitives
original_eval = pallas_numpy_interpreter._eval_jaxpr_numpy_impl

result_log = []

def logged_eval(*args, **kwargs):
    """Wrapper that logs results."""
    result = original_eval(*args, **kwargs)
    # Log a checksum of the result
    if isinstance(result, list):
        for r in result:
            if isinstance(r, np.ndarray):
                result_log.append({
                    'shape': r.shape,
                    'dtype': r.dtype,
                    'sum': float(np.sum(r)),
                    'mean': float(np.mean(r)),
                    'values': r.flatten()[:5].tolist()
                })
    return result

# Monkey patch
pallas_numpy_interpreter._eval_jaxpr_numpy_impl = logged_eval

# Now run the sort
from sort import sort

test_data = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)
print(f"Input: {test_data}")

pallas_numpy_interpreter.install_numpy_interpreter()

result = sort(test_data, num_keys=1, interpret=True)
result = np.array(result)

print(f"Output: {result}")
print(f"\nLogged {len(result_log)} operations")

# Show operations with suspicious results (all same value)
print("\nSuspicious operations (constant results):")
for i, log in enumerate(result_log):
    if log['shape'] == (8, 128):
        vals = log['values']
        if len(set(vals)) == 1 and vals[0] == 1065353216:
            print(f"  Op {i}: shape={log['shape']}, values all 1.0 (1065353216)")
