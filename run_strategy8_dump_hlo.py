import time
import os
import jax

# Enable HLO dumping with minimal optimization
xla_flags = [
    '--xla_backend_optimization_level=0',
    '--xla_llvm_disable_expensive_passes=true',
    '--xla_dump_to=/tmp/xla_dump',
    '--xla_dump_hlo_as_text',
    '--xla_dump_hlo_pass_re=.*',  # Dump all passes to see which ones run
]
os.environ['XLA_FLAGS'] = ' '.join(xla_flags)

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("=" * 60)

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 8: Dump HLO to Analyze Passes\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"HLO dumps in /tmp/xla_dump")
print(f"{'=' * 60}")
