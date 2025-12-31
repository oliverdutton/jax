import time
import os
import jax

# Enable compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Best XLA flags
xla_flags = [
    '--xla_backend_optimization_level=0',
    '--xla_llvm_disable_expensive_passes=true',
]
os.environ['XLA_FLAGS'] = ' '.join(xla_flags)

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"Compilation cache: /tmp/jax_cache")
print("=" * 60)

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 9: Persistent Compilation Cache (Run 1 - Cold)\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Run 1 (cold) time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")

print("\n🔥 STRATEGY 9: Persistent Compilation Cache (Run 2 - Warm)\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Run 2 (warm) time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
