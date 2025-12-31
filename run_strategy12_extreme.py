import time
import os
import jax

# Extreme: Set all optimization levels to 0
os.environ['XLA_FLAGS'] = (
    '--xla_backend_optimization_level=0 '
    '--xla_llvm_disable_expensive_passes=true '
    '--xla_cpu_enable_xprof_traceme=false '
)

# Try to speed up Python overhead
import sys
sys.set_coroutine_origin_tracking_depth(0)

# Disable all JAX safety checks
jax.config.update('jax_enable_checks', False)
jax.config.update('jax_check_tracer_leaks', False)
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_debug_infs', False)
jax.config.update('jax_traceback_filtering', 'off')

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"All optimizations disabled")
print("=" * 60)

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 12: Extreme - All Optimizations Off\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
