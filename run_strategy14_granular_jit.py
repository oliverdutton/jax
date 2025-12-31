import time
import os
import jax
import jax.numpy as jnp
from functools import partial

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("Strategy: Granular JIT with Control Flow Unrolling")
print("=" * 60)

# Enable aggressive compilation caching
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_granular_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Minimal XLA optimization
os.environ['XLA_FLAGS'] = (
    '--xla_backend_optimization_level=0 '
    '--xla_llvm_disable_expensive_passes=true '
    '--xla_cpu_enable_xprof_traceme=false'
)

# Try to monkey-patch control flow to be more granular
import jax.lax as lax

# Store originals
_original_fori_loop = lax.fori_loop
_original_scan = lax.scan

def granular_fori_loop(lower, upper, body_fun, init_val):
    """Replace fori_loop with Python for loop for better compilation granularity."""
    # For small iterations, unroll to Python loop
    if isinstance(upper, int) and isinstance(lower, int):
        iterations = upper - lower
        if iterations <= 10:  # Unroll small loops
            val = init_val
            for i in range(lower, upper):
                val = jax.jit(body_fun, backend='cpu')(i, val)
            return val

    # For larger loops, use original
    return _original_fori_loop(lower, upper, body_fun, init_val)

def granular_scan(f, init, xs, length=None, reverse=False, unroll=1):
    """Replace scan with more granular compilation."""
    # Try to use original but with higher unroll factor
    if unroll == 1:
        unroll = min(4, len(xs) if hasattr(xs, '__len__') else 1)

    return _original_scan(f, init, xs, length, reverse, unroll)

# Monkey-patch
lax.fori_loop = granular_fori_loop
lax.scan = granular_scan

try:
    from benchmark_sort import run_benchmarks

    print("\n🔥 STRATEGY 14: Granular JIT with Unrolled Control Flow\n")
    start_time = time.time()
    run_benchmarks()
    end_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print(f"Primitives in cache: {len(os.listdir('/tmp/jax_granular_cache')) if os.path.exists('/tmp/jax_granular_cache') else 0}")
    print(f"{'=' * 60}")

finally:
    # Restore
    lax.fori_loop = _original_fori_loop
    lax.scan = _original_scan
