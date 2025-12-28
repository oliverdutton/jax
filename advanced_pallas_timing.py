"""Advanced timing and profiling utilities for Pallas TPU compilation.

This module extends the basic timing instrumentation with additional features:
- HLO inspection and analysis
- Memory usage tracking
- Pass-by-pass timing for MLIR pipelines
- Export format analysis
"""

import contextlib
import time
import tracemalloc
from typing import Any, Dict, Optional
import json

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax._src.interpreters import mlir
from jaxlib.mlir.passmanager import PassManager
from jaxlib.mlir import ir

from pallas_tpu_timing_instrumentation import (
    TIMING_STATS,
    time_pallas_compilation,
)


class MemoryTracker:
    """Track memory usage during compilation."""

    def __init__(self):
        self.snapshots = []
        self.enabled = False

    def start(self):
        """Start memory tracking."""
        tracemalloc.start()
        self.enabled = True
        self.snapshots = []

    def snapshot(self, label: str):
        """Take a memory snapshot."""
        if not self.enabled:
            return

        current, peak = tracemalloc.get_traced_memory()
        self.snapshots.append({
            'label': label,
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'timestamp': time.perf_counter(),
        })

    def stop(self):
        """Stop memory tracking."""
        if self.enabled:
            tracemalloc.stop()
            self.enabled = False

    def print_summary(self):
        """Print memory usage summary."""
        if not self.snapshots:
            print("No memory snapshots collected")
            return

        print("\n" + "=" * 80)
        print("MEMORY USAGE SUMMARY")
        print("=" * 80)

        for snap in self.snapshots:
            print(f"{snap['label']:40s} Current: {snap['current_mb']:8.2f} MB  "
                  f"Peak: {snap['peak_mb']:8.2f} MB")

        print("=" * 80 + "\n")


MEMORY_TRACKER = MemoryTracker()


def analyze_hlo(lowered):
    """Analyze the HLO output from a lowered function."""
    print("\n" + "=" * 80)
    print("HLO ANALYSIS")
    print("=" * 80)

    with TIMING_STATS.measure("hlo_analysis"):
        # Get HLO text
        with TIMING_STATS.measure("hlo_get_text"):
            hlo_text = lowered.as_text()

        # Parse HLO for statistics
        with TIMING_STATS.measure("hlo_parse_stats"):
            lines = hlo_text.split('\n')
            stats = {
                'total_lines': len(lines),
                'custom_calls': sum(1 for l in lines if 'custom-call' in l or 'custom_call' in l),
                'tpu_custom_calls': sum(1 for l in lines if 'tpu_custom_call' in l),
                'total_ops': sum(1 for l in lines if '=' in l and not l.strip().startswith('//')),
            }

        # Look for backend config
        with TIMING_STATS.measure("hlo_find_backend_config"):
            backend_configs = []
            for line in lines:
                if 'backend_config' in line:
                    backend_configs.append(line.strip())

    print(f"Total HLO lines: {stats['total_lines']}")
    print(f"Total operations: {stats['total_ops']}")
    print(f"Custom calls: {stats['custom_calls']}")
    print(f"TPU custom calls: {stats['tpu_custom_calls']}")
    print(f"Backend configs found: {len(backend_configs)}")

    if backend_configs:
        print("\nFirst backend_config (truncated):")
        config_str = backend_configs[0]
        print(config_str[:200] + "..." if len(config_str) > 200 else config_str)

    print("=" * 80)

    return stats


def instrument_mlir_passes(pass_pipeline: str = "builtin.module(canonicalize)"):
    """Create an instrumented MLIR pass manager.

    This allows timing individual MLIR passes during lowering.
    """
    class InstrumentedPassManager:
        def __init__(self, pm: PassManager, name: str):
            self.pm = pm
            self.name = name

        def run(self, module_op):
            with TIMING_STATS.measure(f"mlir_pass_{self.name}"):
                return self.pm.run(module_op)

        def __getattr__(self, name):
            return getattr(self.pm, name)

    # Parse the pass manager
    pm = PassManager.parse(pass_pipeline)
    return InstrumentedPassManager(pm, pass_pipeline.split('(')[0])


def detailed_compilation_analysis(kernel_fn, *args, kernel_name: str = "kernel"):
    """Perform detailed compilation analysis with all instrumentation enabled.

    Args:
        kernel_fn: The Pallas kernel function (wrapped in jit)
        *args: Input arguments for the kernel
        kernel_name: Name for this kernel (for reporting)

    Returns:
        Dictionary with compilation statistics
    """
    print(f"\n{'='*80}")
    print(f"DETAILED COMPILATION ANALYSIS: {kernel_name}")
    print('='*80)

    results = {
        'kernel_name': kernel_name,
        'timings': {},
        'memory': {},
        'hlo_stats': {},
    }

    # Start memory tracking
    MEMORY_TRACKER.start()
    MEMORY_TRACKER.snapshot("start")

    with time_pallas_compilation():
        # Lowering phase
        with TIMING_STATS.measure("phase_1_lowering"):
            MEMORY_TRACKER.snapshot("before_lowering")
            lowered = kernel_fn.lower(*args)
            MEMORY_TRACKER.snapshot("after_lowering")

        # HLO analysis
        with TIMING_STATS.measure("phase_2_hlo_analysis"):
            hlo_stats = analyze_hlo(lowered)
            results['hlo_stats'] = hlo_stats

        # Compilation phase
        with TIMING_STATS.measure("phase_3_compilation"):
            MEMORY_TRACKER.snapshot("before_compilation")
            compiled = lowered.compile()
            MEMORY_TRACKER.snapshot("after_compilation")

        # First execution
        with TIMING_STATS.measure("phase_4_first_execution"):
            MEMORY_TRACKER.snapshot("before_execution")
            result = compiled(*args)
            result.block_until_ready()
            MEMORY_TRACKER.snapshot("after_execution")

    # Stop memory tracking
    MEMORY_TRACKER.stop()

    # Collect results
    results['timings'] = TIMING_STATS.get_summary()
    results['memory'] = MEMORY_TRACKER.snapshots

    # Print summaries
    TIMING_STATS.print_summary()
    MEMORY_TRACKER.print_summary()

    return results


def compare_kernel_representations(kernel_fn, *args):
    """Compare different representations of a Pallas kernel.

    Shows the kernel at different stages: Jaxpr, MLIR, HLO, etc.
    """
    print("\n" + "=" * 80)
    print("KERNEL REPRESENTATION COMPARISON")
    print("=" * 80)

    representations = {}

    with time_pallas_compilation():
        # Get lowered representation
        with TIMING_STATS.measure("get_lowered"):
            lowered = kernel_fn.lower(*args)

        # Get HLO
        with TIMING_STATS.measure("get_hlo_text"):
            hlo_text = lowered.as_text()
            representations['hlo'] = {
                'size_bytes': len(hlo_text),
                'lines': len(hlo_text.split('\n')),
                'sample': hlo_text[:500],
            }

        # Get compiler IR (if available)
        with TIMING_STATS.measure("get_compiler_ir"):
            try:
                compiler_ir = lowered.compiler_ir()
                representations['mlir'] = {
                    'available': True,
                    'sample': str(compiler_ir)[:500],
                }
            except Exception as e:
                representations['mlir'] = {
                    'available': False,
                    'error': str(e),
                }

        # Get cost analysis (if available)
        with TIMING_STATS.measure("get_cost_analysis"):
            try:
                cost = lowered.cost_analysis()
                representations['cost_analysis'] = cost
            except Exception as e:
                representations['cost_analysis'] = {
                    'available': False,
                    'error': str(e),
                }

    # Print comparison
    print("\nRepresentation Sizes:")
    for name, info in representations.items():
        if isinstance(info, dict) and 'size_bytes' in info:
            print(f"  {name:20s}: {info['size_bytes']:10d} bytes, "
                  f"{info['lines']:6d} lines")
        elif isinstance(info, dict) and 'available' in info:
            print(f"  {name:20s}: {'Available' if info['available'] else 'Not available'}")

    print("\n" + "=" * 80)

    return representations


# Example usage
if __name__ == "__main__":
    print("Advanced Pallas TPU Timing and Analysis")
    print("=" * 80)

    # Example 1: Detailed compilation analysis
    def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    size = 2048
    x = jnp.ones(size, dtype=jnp.float32)
    y = jnp.ones(size, dtype=jnp.float32)

    def add_pallas(x, y):
        return pl.pallas_call(
            add_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        )(x, y)

    add_jit = jax.jit(add_pallas)

    # Run detailed analysis
    results = detailed_compilation_analysis(
        add_jit, x, y,
        kernel_name="element_wise_add"
    )

    # Export results
    output_file = '/tmp/pallas_detailed_analysis.json'
    with open(output_file, 'w') as f:
        # Convert complex objects to serializable format
        serializable_results = {
            'kernel_name': results['kernel_name'],
            'hlo_stats': results['hlo_stats'],
            'memory': results['memory'],
            'timings': {
                k: {
                    'count': v['count'],
                    'total': v['total'],
                    'mean': v['mean'],
                    'min': v['min'],
                    'max': v['max'],
                }
                for k, v in results['timings'].items()
            }
        }
        json.dump(serializable_results, f, indent=2)

    print(f"\nDetailed analysis exported to {output_file}")

    # Reset for next example
    TIMING_STATS.reset()

    # Example 2: Compare representations
    print("\n\n")
    representations = compare_kernel_representations(add_jit, x, y)

    print("\n" + "=" * 80)
    print("FINAL TIMING SUMMARY")
    print("=" * 80)
    TIMING_STATS.print_summary()
