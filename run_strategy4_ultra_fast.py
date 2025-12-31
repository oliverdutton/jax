import time
import os
import jax

# Ultra-aggressive: Disable everything possible
xla_flags = [
    '--xla_backend_optimization_level=0',  # No optimization
    '--xla_llvm_disable_expensive_passes=true',  # No expensive LLVM passes
    '--xla_disable_hlo_passes=*',  # Try to disable all HLO passes with wildcard
    '--xla_cpu_enable_fast_math=false',
    '--xla_cpu_enable_xprof_traceme=false',
]
os.environ['XLA_FLAGS'] = ' '.join(xla_flags)

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
print("=" * 60)

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 4a: Ultra-Fast - Disable All Passes with Wildcard\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
