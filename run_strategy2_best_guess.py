import time
import os
import jax

# Best guess: Disable expensive optimization passes but keep essential ones
# Try to minimize compilation time while keeping correctness
xla_flags = [
    '--xla_cpu_enable_fast_math=false',  # Disable expensive math optimizations
    '--xla_cpu_enable_fast_min_max=false',  # Disable fast min/max
    '--xla_cpu_enable_xprof_traceme=false',  # Disable profiling overhead
    '--xla_disable_hlo_passes=algebraic-simplifier,all-reduce-combiner,all-to-all-decomposer',  # Disable expensive passes
]
os.environ['XLA_FLAGS'] = ' '.join(xla_flags)

# Also try to reduce compilation parallelism to avoid overhead
jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')
jax.config.update('jax_threefry_partitionable', False)

# Verify JAX version
print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'not set')}")
print("=" * 60)

# Import and run benchmark
from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 2b: JIT with Selective HLO Pass Optimization (Best Guess)\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time (including cold start): {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
