#!/usr/bin/env python3
"""
Benchmark topk operation with vocab_size=2048, 128 blocks, top 32.

Tests different implementations:
1. JAX native jax.lax.top_k
2. JAX native jnp.argsort + slice
3. Pallas TPU interpret mode (if available)
"""

import time
import jax
import jax.numpy as jnp
from jax import lax

def benchmark_jax_topk(logits, k, num_runs=10):
    """Benchmark JAX native top_k."""
    @jax.jit
    def topk_fn(logits):
        return lax.top_k(logits, k)

    # Warmup
    values, indices = topk_fn(logits)
    _ = values.block_until_ready()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        values, indices = topk_fn(logits)
        _ = values.block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)

    return sum(times) / len(times), min(times), (values, indices)

def benchmark_argsort_topk(logits, k, num_runs=10):
    """Benchmark using argsort + slice."""
    @jax.jit
    def topk_fn(logits):
        # Sort descending and take top k
        indices = jnp.argsort(-logits, axis=-1)[..., :k]
        values = jnp.take_along_axis(logits, indices, axis=-1)
        return values, indices

    # Warmup
    values, indices = topk_fn(logits)
    _ = values.block_until_ready()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        values, indices = topk_fn(logits)
        _ = values.block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)

    return sum(times) / len(times), min(times), (values, indices)

def benchmark_sort_only(logits, num_runs=10):
    """Benchmark just the sort operation."""
    @jax.jit
    def sort_fn(logits):
        return jnp.sort(logits, axis=-1)

    # Warmup
    _ = sort_fn(logits).block_until_ready()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        result = sort_fn(logits).block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)

    return sum(times) / len(times), min(times), result

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*25 + "TOPK BENCHMARK")
    print(" "*20 + "vocab_size=2048, 128 blocks, top 32")
    print("="*80)

    # Configuration as requested
    batch_size = 128  # "128 blocks"
    vocab_size = 2048
    k = 32  # "top 32"

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size} blocks")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Top k: {k}")
    print(f"  Input shape: ({batch_size}, {vocab_size})")
    print(f"  Output shape: ({batch_size}, {k})")

    # Create random logits
    key = jax.random.key(42)
    logits = jax.random.normal(key, (batch_size, vocab_size), jnp.float32)

    print("\n" + "="*80)
    print("BENCHMARKS (execution time with block_until_ready)")
    print("="*80)

    # Benchmark 1: JAX native top_k
    print("\n[1] JAX NATIVE lax.top_k")
    print("-" * 80)
    avg_topk, min_topk, (values_topk, indices_topk) = benchmark_jax_topk(logits, k)
    print(f"  Average: {avg_topk*1000:.3f}ms ({avg_topk:.6f}s)")
    print(f"  Min:     {min_topk*1000:.3f}ms ({min_topk:.6f}s)")

    # Benchmark 2: Argsort + slice
    print("\n[2] ARGSORT + SLICE")
    print("-" * 80)
    avg_argsort, min_argsort, (values_argsort, indices_argsort) = benchmark_argsort_topk(logits, k)
    print(f"  Average: {avg_argsort*1000:.3f}ms ({avg_argsort:.6f}s)")
    print(f"  Min:     {min_argsort*1000:.3f}ms ({min_argsort:.6f}s)")

    # Verify results match
    assert jnp.allclose(values_topk, values_argsort, atol=1e-5), "Values don't match!"
    # Note: indices might differ for equal values, so we check values only
    print("  ✓ Results match")

    # Benchmark 3: Full sort (for comparison)
    print("\n[3] FULL SORT (for comparison)")
    print("-" * 80)
    avg_sort, min_sort, _ = benchmark_sort_only(logits)
    print(f"  Average: {avg_sort*1000:.3f}ms ({avg_sort:.6f}s)")
    print(f"  Min:     {min_sort*1000:.3f}ms ({min_sort:.6f}s)")

    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\n{'Method':<30} {'Avg Time':<15} {'Min Time':<15} {'Relative'}")
    print("-" * 80)

    baseline = avg_topk
    print(f"{'lax.top_k':<30} {avg_topk*1000:>10.3f}ms {min_topk*1000:>10.3f}ms {avg_topk/baseline:>9.2f}x")
    print(f"{'argsort + slice':<30} {avg_argsort*1000:>10.3f}ms {min_argsort*1000:>10.3f}ms {avg_argsort/baseline:>9.2f}x")
    print(f"{'full sort':<30} {avg_sort*1000:>10.3f}ms {min_sort*1000:>10.3f}ms {avg_sort/baseline:>9.2f}x")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    print(f"\nFor {batch_size} blocks × {vocab_size} vocab × top {k}:")
    print(f"  Best method: {'lax.top_k' if avg_topk <= avg_argsort else 'argsort'}")
    print(f"  Best time: {min(min_topk, min_argsort)*1000:.3f}ms")

    print(f"\nObservations:")
    print(f"  - lax.top_k is optimized for partial sorting")
    print(f"  - argsort must sort all {vocab_size} elements")
    print(f"  - For k={k} << vocab_size={vocab_size}, top_k should be faster")

    if avg_topk < avg_argsort:
        speedup = avg_argsort / avg_topk
        print(f"  - lax.top_k is {speedup:.2f}x faster than argsort")
    else:
        speedup = avg_topk / avg_argsort
        print(f"  - argsort is {speedup:.2f}x faster than lax.top_k")

    print(f"\nThroughput:")
    ops_per_sec = batch_size / min_topk
    elements_per_sec = (batch_size * vocab_size) / min_topk
    print(f"  - {ops_per_sec:,.0f} top-k operations/second")
    print(f"  - {elements_per_sec:,.0f} elements processed/second")
    print(f"  - {elements_per_sec/1e6:.2f}M elements/second")

    print("\n" + "="*80)
    print("✓ Topk benchmark complete")
    print("="*80 + "\n")
