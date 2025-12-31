import time
import os
import jax

# Try setting compiler optimization level to reduce compile time
xla_flags = [
    '--xla_backend_optimization_level=0',  # Minimal optimization
    '--xla_cpu_use_thunk_runtime=false',  # Disable thunk runtime overhead
    '--xla_llvm_disable_expensive_passes=true',  # Disable expensive LLVM passes
]
os.environ['XLA_FLAGS'] = ' '.join(xla_flags)

# Verify JAX version
print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'not set')}")
print("=" * 60)

# Import and run benchmark
from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 2c: Minimal Optimization Level\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time (including cold start): {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
