import time
import os
import jax
import jax.numpy as jnp

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("Strategy: Disable shape polymorphism and dynamic shapes")
print("=" * 60)

# Disable dynamic shapes and shape polymorphism
jax.config.update('jax_dynamic_shapes', False)
jax.config.update('jax_enable_x64', False)  # Ensure using 32-bit

# Best XLA flags
os.environ['XLA_FLAGS'] = (
    '--xla_backend_optimization_level=0 '
    '--xla_llvm_disable_expensive_passes=true'
)

# Disable Python overhead
jax.config.update('jax_traceback_filtering', 'off')
jax.config.update('jax_enable_checks', False)

# Try to use static argument nums more aggressively
# This requires modifying how functions are called

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 16: Disable Shape Polymorphism\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
