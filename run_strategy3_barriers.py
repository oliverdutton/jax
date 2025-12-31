import time
import jax
import jax.numpy as jnp

# Verify JAX version
print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("=" * 60)

# Monkey-patch to add optimization barriers
# We'll wrap the tax.sort function to add barriers around operands
import tallax.tax as tax
original_sort = tax.sort

def sort_with_barriers(operands, **kwargs):
    # Add optimization barriers to prevent aggressive optimization
    # This forces the compiler to materialize intermediate results
    if isinstance(operands, list):
        operands = [jax.lax.optimization_barrier(op) for op in operands]
    else:
        operands = jax.lax.optimization_barrier(operands)

    result = original_sort(operands, **kwargs)

    # Add barrier on output as well
    if isinstance(result, tuple):
        result = tuple(jax.lax.optimization_barrier(r) for r in result)
    else:
        result = jax.lax.optimization_barrier(result)

    return result

# Replace the function
tax.sort = sort_with_barriers

# Import and run benchmark
from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 3: Optimization Barriers Between Everything\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time (including cold start): {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
