import time
import os
import jax

# Skip HLO verification which can be expensive
xla_flags = [
    '--xla_backend_optimization_level=0',
    '--xla_llvm_disable_expensive_passes=true',
    '--xla_cpu_enable_xprof_traceme=false',
    '--xla_hlo_evaluator_use_fast_path=true',  # Use fast evaluation
    '--xla_cpu_multi_thread_eigen=false',  # Disable multi-threading overhead in compilation
]
os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce logging overhead

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
print("=" * 60)

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 4c: Skip HLO Verification + Fast Eval\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
