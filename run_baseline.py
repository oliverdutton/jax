import time
import jax

# Verify JAX version
print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"JAX file location: {jax.__file__}")
print("=" * 60)

# Import and run benchmark
from benchmark_sort import run_benchmarks

print("\n🔥 BASELINE TEST - Cold Start Compilation Time\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time (including cold start): {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
