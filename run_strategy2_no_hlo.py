import time
import os
import jax

# Set XLA flags to disable all HLO optimization passes
os.environ['XLA_FLAGS'] = '--xla_disable_all_hlo_passes=true'

# Verify JAX version
print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'not set')}")
print("=" * 60)

# Import and run benchmark
from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 2a: JIT with ALL HLO Passes Disabled\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time (including cold start): {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
