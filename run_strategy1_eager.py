import time
import jax

# Configure JAX for eager mode (disable JIT)
jax.config.update('jax_disable_jit', True)

# Verify JAX version and config
print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"JAX JIT disabled: {jax.config.jax_disable_jit}")
print("=" * 60)

# Import and run benchmark
from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 1: Eager Mode (No JIT) - Cold Start Compilation Time\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time (including cold start): {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
