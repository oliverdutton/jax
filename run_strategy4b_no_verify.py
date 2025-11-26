import time
import os
import jax

# Disable verification and checking overhead
xla_flags = [
    '--xla_backend_optimization_level=0',
    '--xla_llvm_disable_expensive_passes=true',
    '--xla_cpu_enable_xprof_traceme=false',
    '--xla_hlo_graph_addresses=false',  # Disable address tracking
    '--xla_dump_to=',  # Disable dumping
    '--xla_force_host_platform_device_count=1',  # Single device
]
os.environ['XLA_FLAGS'] = ' '.join(xla_flags)

# Disable JAX verification
jax.config.update('jax_enable_checks', False)
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_debug_infs', False)

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
print(f"jax_enable_checks: {jax.config.jax_enable_checks}")
print("=" * 60)

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 4b: Disable Verification & Checks\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
