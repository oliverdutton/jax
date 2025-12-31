import time
import os
import jax
import cProfile
import pstats
from io import StringIO

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("Strategy: Profile to find bottlenecks")
print("=" * 60)

# Best XLA flags
os.environ['XLA_FLAGS'] = (
    '--xla_backend_optimization_level=0 '
    '--xla_llvm_disable_expensive_passes=true'
)

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 17: Profile Compilation Bottlenecks\n")

# Profile the run
profiler = cProfile.Profile()
profiler.enable()

start_time = time.time()
run_benchmarks()
end_time = time.time()

profiler.disable()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")

# Print top time consumers
s = StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)  # Top 30 functions
print("\nTop 30 time-consuming functions:")
print(s.getvalue())

# Focus on compilation-related functions
print("\n\nCompilation-related functions (containing 'compile', 'lower', 'xla'):")
ps.print_stats('compile|lower|xla')
