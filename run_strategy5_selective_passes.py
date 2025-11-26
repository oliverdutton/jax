import time
import os
import jax

# Disable specific known-expensive HLO passes
expensive_passes = [
    'algebraic-simplifier',
    'batchnorm-expander',
    'call-inliner',
    'conditional-simplifier',
    'convolution-4d-expander',
    'dot-decomposer',
    'flatten-call-graph',
    'hlo-constant-folding',
    'hlo-cse',
    'hlo-dce',
    'loop-invariant-code-motion',
    'reshape-mover',
    'slice-sinker',
    'transpose-folding',
    'tuple-simplifier',
    'while-loop-constant-sinking',
    'while-loop-invariant-code-motion',
    'zero-sized-hlo-elimination',
]

xla_flags = [
    '--xla_backend_optimization_level=0',
    '--xla_llvm_disable_expensive_passes=true',
    f'--xla_disable_hlo_passes={",".join(expensive_passes)}',
]
os.environ['XLA_FLAGS'] = ' '.join(xla_flags)

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"Disabling {len(expensive_passes)} HLO passes")
print("=" * 60)

from benchmark_sort import run_benchmarks

print("\n🔥 STRATEGY 5: Disable Specific Expensive HLO Passes\n")
start_time = time.time()
run_benchmarks()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
