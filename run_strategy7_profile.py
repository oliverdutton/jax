import time
import os
import jax
import jax.profiler

# Best flags so far
xla_flags = [
    '--xla_backend_optimization_level=0',
    '--xla_llvm_disable_expensive_passes=true',
]
os.environ['XLA_FLAGS'] = ' '.join(xla_flags)

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
print("=" * 60)

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 7: Profile Compilation\n")

# Start profiling
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False):
    start_time = time.time()
    run_benchmarks()
    end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"Profile saved to /tmp/jax-trace")
print(f"{'=' * 60}")
