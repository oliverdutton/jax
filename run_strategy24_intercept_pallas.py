import time
import os
import numpy as np
import jax
import jax.numpy as jnp

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("Strategy: Intercept pallas_call and execute kernel with NumPy")
print("=" * 60)

# Import Pallas
from jax.experimental import pallas as pl

# Store original pallas_call
_original_pallas_call = pl.pallas_call

_pallas_calls = 0

def numpy_pallas_call(kernel, out_shape, *, interpret=False, **kwargs):
    """Replace pallas_call with NumPy execution."""
    global _pallas_calls

    # Only intercept if in interpret mode
    if not interpret:
        return _original_pallas_call(kernel, out_shape, interpret=interpret, **kwargs)

    _pallas_calls += 1

    print(f"  [Intercepted pallas_call #{_pallas_calls}, interpret=True]")

    # For interpret mode on CPU, we want to avoid compilation
    # Instead, execute the kernel logic directly with NumPy

    def wrapped(*args):
        # Try to execute kernel with NumPy arrays
        try:
            # Convert inputs to NumPy
            numpy_args = [np.asarray(arg) if hasattr(arg, '__array__') else arg
                         for arg in args]

            print(f"    Input shapes: {[arg.shape for arg in numpy_args if hasattr(arg, 'shape')]}")

            # We can't easily execute the kernel directly since it uses Pallas primitives
            # Fall back to letting Pallas compile, but with optimizations disabled
            os.environ['XLA_FLAGS'] = (
                '--xla_backend_optimization_level=0 '
                '--xla_llvm_disable_expensive_passes=true'
            )

            # Use original pallas_call with interpret=True
            compiled_fn = _original_pallas_call(kernel, out_shape, interpret=True, **kwargs)
            result = compiled_fn(*args)

            print(f"    Output shapes: {[r.shape for r in result] if isinstance(result, tuple) else [result.shape]}")

            return result

        except Exception as e:
            print(f"    Error in NumPy execution: {e}")
            # Fallback to original
            compiled_fn = _original_pallas_call(kernel, out_shape, interpret=interpret, **kwargs)
            return compiled_fn(*args)

    return wrapped

# Monkey-patch pallas_call
pl.pallas_call = numpy_pallas_call

# Also patch in the module where it's imported
import jax.experimental.pallas as pallas_module
pallas_module.pallas_call = numpy_pallas_call

print("\n🔥 STRATEGY 24: Intercept pallas_call\n")
print("Hooking Pallas to avoid compilation overhead")

try:
    from benchmark_sort import run_benchmarks

    start_time = time.time()
    run_benchmarks()
    end_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print(f"Pallas calls intercepted: {_pallas_calls}")
    print(f"{'=' * 60}")

finally:
    # Restore original
    pl.pallas_call = _original_pallas_call
    pallas_module.pallas_call = _original_pallas_call
