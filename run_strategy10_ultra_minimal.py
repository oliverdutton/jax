import time
import os
import jax

# Ultra-minimal: skip as much as possible
xla_flags = [
    '--xla_backend_optimization_level=0',
    '--xla_llvm_disable_expensive_passes=true',
    '--xla_cpu_enable_concurrency_optimized_scheduler=false',  # Disable scheduler overhead
    '--xla_cpu_disable_fixed_constant_allocation=true',  # Disable allocation optimization
    '--xla_cpu_enable_experimental_deallocation=false',  # Disable deallocation pass
]
os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
os.environ['XLA_FLAGS'] += ' --xla_cpu_llvm_opt_level=0'  # LLVM optimization level 0

jax.config.update('jax_traceback_filtering', 'off')  # Disable traceback processing
jax.config.update('jax_check_tracer_leaks', False)  # Disable tracer leak checking

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
print("=" * 60)

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 10: Ultra-Minimal Compilation\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
