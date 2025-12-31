import time
import os
import jax

# Focus on reducing LLVM compilation time
os.environ['LLVM_PROFILE_FILE'] = ''  # Disable profiling
os.environ['XLA_FLAGS'] = '--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true --xla_cpu_enable_xprof_traceme=false'

# Combine with eager mode
jax.config.update('jax_disable_jit', True)

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"Eager mode + Minimal LLVM")
print("=" * 60)

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 11: Eager + Minimal LLVM\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
