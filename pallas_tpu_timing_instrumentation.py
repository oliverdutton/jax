"""Detailed timing instrumentation for Pallas TPU lowering and compilation pipeline.

This module provides comprehensive timing for every stage of the Pallas TPU
compilation process, from the initial pallas_call through to the final HLO.
"""

import contextlib
import functools
import time
from typing import Any, Callable, Dict, List
import json

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax._src.pallas.mosaic import pallas_call_registration
from jax._src import tpu_custom_call
from jax._src.interpreters import mlir
from jaxlib.mlir.passmanager import PassManager


class TimingStats:
    """Collects timing statistics for the compilation pipeline."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.stack: List[tuple[str, float]] = []
        self.enabled = True

    @contextlib.contextmanager
    def measure(self, name: str):
        """Context manager to measure a block of code."""
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        self.stack.append((name, start))
        try:
            yield
        finally:
            actual_name, actual_start = self.stack.pop()
            assert actual_name == name
            elapsed = time.perf_counter() - actual_start

            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all measured stages."""
        summary = {}
        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    'count': len(times),
                    'total': sum(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'times': times,
                }
        return summary

    def print_summary(self):
        """Print a formatted summary of all timings."""
        summary = self.get_summary()
        print("\n" + "="*80)
        print("PALLAS TPU COMPILATION PIPELINE TIMING SUMMARY")
        print("="*80)

        # Sort by total time descending
        sorted_stages = sorted(summary.items(),
                              key=lambda x: x[1]['total'],
                              reverse=True)

        for name, stats in sorted_stages:
            print(f"\n{name}:")
            print(f"  Total:  {stats['total']*1000:.3f} ms")
            print(f"  Mean:   {stats['mean']*1000:.3f} ms")
            print(f"  Min:    {stats['min']*1000:.3f} ms")
            print(f"  Max:    {stats['max']*1000:.3f} ms")
            print(f"  Count:  {stats['count']}")
            if len(stats['times']) > 1:
                print(f"  Times:  {[f'{t*1000:.3f}' for t in stats['times']]}")

        print("\n" + "="*80)
        total_measured = sum(s['total'] for s in summary.values())
        print(f"Total measured time: {total_measured*1000:.3f} ms")
        print("="*80 + "\n")

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.stack.clear()


# Global timing stats instance
TIMING_STATS = TimingStats()


def timed(name: str):
    """Decorator to time a function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with TIMING_STATS.measure(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def instrument_pallas_tpu_lowering():
    """Monkey-patch the Pallas TPU lowering pipeline to add timing."""

    # Save original functions
    original_lowering_rule = pallas_call_registration.pallas_call_tpu_lowering_rule
    original_lower_mosaic_module_to_asm = tpu_custom_call._lower_mosaic_module_to_asm
    original_tpu_custom_call_lowering = tpu_custom_call._tpu_custom_call_lowering
    original_lower_to_custom_call_config = tpu_custom_call._lower_to_custom_call_config

    # Import the actual lowering functions
    from jax._src.pallas.mosaic import lowering as tc_lowering
    from jax._src.pallas.mosaic import sc_lowering

    original_tc_lower_jaxpr = tc_lowering.lower_jaxpr_to_module
    original_sc_lower_jaxpr = sc_lowering.lower_jaxpr_to_module

    @functools.wraps(original_lowering_rule)
    def timed_lowering_rule(ctx, *in_nodes, **kwargs):
        """Timed wrapper for pallas_call_tpu_lowering_rule."""
        with TIMING_STATS.measure("1_total_pallas_call_tpu_lowering"):

            # Time the jaxpr to module lowering
            jaxpr = kwargs.get('jaxpr')
            compiler_params = kwargs.get('compiler_params', {})
            mosaic_params = compiler_params.get('mosaic_tpu')

            # Determine kernel type
            kernel_type = None
            if mosaic_params:
                kernel_type = mosaic_params.kernel_type

            # Wrap the appropriate lower_jaxpr_to_module function
            if kernel_type and hasattr(kernel_type, 'name'):
                kernel_type_name = kernel_type.name
            else:
                kernel_type_name = str(kernel_type)

            # Save and wrap the lowering functions
            if 'SC' in kernel_type_name:
                original_func = sc_lowering.lower_jaxpr_to_module
                def timed_sc_lower(*args, **kw):
                    with TIMING_STATS.measure("2_sparsecore_lower_jaxpr_to_module"):
                        return original_func(*args, **kw)
                sc_lowering.lower_jaxpr_to_module = timed_sc_lower
            else:
                original_func = tc_lowering.lower_jaxpr_to_module
                def timed_tc_lower(*args, **kw):
                    with TIMING_STATS.measure("2_tensorcore_lower_jaxpr_to_module"):
                        return original_func(*args, **kw)
                tc_lowering.lower_jaxpr_to_module = timed_tc_lower

            try:
                result = original_lowering_rule(ctx, *in_nodes, **kwargs)
            finally:
                # Restore original functions
                if 'SC' in kernel_type_name:
                    sc_lowering.lower_jaxpr_to_module = original_func
                else:
                    tc_lowering.lower_jaxpr_to_module = original_func

            return result

    @functools.wraps(original_lower_mosaic_module_to_asm)
    def timed_lower_mosaic_module_to_asm(module, **kwargs):
        """Timed wrapper for _lower_mosaic_module_to_asm."""
        with TIMING_STATS.measure("3_lower_mosaic_module_to_asm"):
            # Time the communication check
            with TIMING_STATS.measure("3a_check_has_communication"):
                from jax._src.lib import tpu
                has_communication, has_custom_barrier = tpu.private_has_communication(
                    module.operation
                )

            # Time the module cloning
            with TIMING_STATS.measure("3b_clone_module"):
                with module.context as ctx, module.operation.location as _:
                    module_op = module.operation.clone()

            # Time the mosaic-serde pass
            ir_version = kwargs.get('ir_version')
            target_version = (
                f"target-version={ir_version}" if ir_version is not None else ""
            )

            with TIMING_STATS.measure("3c_mosaic_serde_pass"):
                with module.context as ctx:
                    prev_allow_unregistered_dialects = ctx.allow_unregistered_dialects
                    ctx.allow_unregistered_dialects = True
                    try:
                        pipeline = PassManager.parse(
                            "builtin.module(mosaic-serde{serialize=true " + target_version + "})"
                        )
                        pipeline.run(module_op)
                    finally:
                        ctx.allow_unregistered_dialects = prev_allow_unregistered_dialects

            # Time the bytecode writing
            with TIMING_STATS.measure("3d_write_bytecode"):
                import io
                bytecode_buffer = io.BytesIO()
                module_op.write_bytecode(bytecode_buffer, desired_version=0)
                asm = bytecode_buffer.getvalue()

            return asm, (has_communication, has_custom_barrier)

    @functools.wraps(original_tpu_custom_call_lowering)
    def timed_tpu_custom_call_lowering(ctx, *in_nodes, **kwargs):
        """Timed wrapper for _tpu_custom_call_lowering."""
        with TIMING_STATS.measure("4_tpu_custom_call_lowering"):

            # Time the mlir.custom_call creation
            config = kwargs.get('config')

            # Time backend config JSON serialization
            if config:
                with TIMING_STATS.measure("4a_backend_config_to_json"):
                    _ = config.to_json()

            with TIMING_STATS.measure("4b_create_custom_call_op"):
                result = original_tpu_custom_call_lowering(ctx, *in_nodes, **kwargs)

            return result

    @functools.wraps(original_lower_to_custom_call_config)
    def timed_lower_to_custom_call_config(module, **kwargs):
        """Timed wrapper for _lower_to_custom_call_config."""
        with TIMING_STATS.measure("3x_lower_to_custom_call_config"):

            # Time device type detection
            with TIMING_STATS.measure("3x1_get_device_type"):
                device_type = tpu_custom_call._get_device_type(module)

            # Time active core count detection
            with TIMING_STATS.measure("3x2_get_active_core_count"):
                active_core_count = tpu_custom_call._get_active_core_count(module)

            # The rest is handled by _lower_mosaic_module_to_asm
            with TIMING_STATS.measure("3x3_call_lower_mosaic_module_to_asm"):
                result = original_lower_to_custom_call_config(module, **kwargs)

            return result

    # Apply monkey patches
    pallas_call_registration.pallas_call_tpu_lowering_rule = timed_lowering_rule
    tpu_custom_call._lower_mosaic_module_to_asm = timed_lower_mosaic_module_to_asm
    tpu_custom_call._tpu_custom_call_lowering = timed_tpu_custom_call_lowering
    tpu_custom_call._lower_to_custom_call_config = timed_lower_to_custom_call_config

    # Store originals for restoration
    return {
        'pallas_call_tpu_lowering_rule': original_lowering_rule,
        '_lower_mosaic_module_to_asm': original_lower_mosaic_module_to_asm,
        '_tpu_custom_call_lowering': original_tpu_custom_call_lowering,
        '_lower_to_custom_call_config': original_lower_to_custom_call_config,
    }


def restore_pallas_tpu_lowering(originals: Dict[str, Callable]):
    """Restore original functions after instrumentation."""
    pallas_call_registration.pallas_call_tpu_lowering_rule = originals['pallas_call_tpu_lowering_rule']
    tpu_custom_call._lower_mosaic_module_to_asm = originals['_lower_mosaic_module_to_asm']
    tpu_custom_call._tpu_custom_call_lowering = originals['_tpu_custom_call_lowering']
    tpu_custom_call._lower_to_custom_call_config = originals['_lower_to_custom_call_config']


@contextlib.contextmanager
def time_pallas_compilation(reset_after: bool = False):
    """Context manager for timing Pallas compilation.

    Args:
        reset_after: If True, reset timing stats after exiting the context.

    Example:
        with time_pallas_compilation():
            # Your Pallas code here
            result = pallas_function(x)

        TIMING_STATS.print_summary()
    """
    originals = instrument_pallas_tpu_lowering()
    try:
        yield TIMING_STATS
    finally:
        restore_pallas_tpu_lowering(originals)
        if reset_after:
            TIMING_STATS.reset()


# Example usage
if __name__ == "__main__":
    print("Pallas TPU Timing Instrumentation Example")
    print("=" * 80)

    # Define a simple matrix multiplication kernel
    def matmul_kernel(x_ref, y_ref, o_ref):
        """Simple matrix multiply kernel."""
        @pl.when(pl.program_id(0) == 0)
        def _():
            o_ref[...] = x_ref[...] @ y_ref[...]

    def create_matmul(m: int, k: int, n: int):
        """Create a Pallas matmul function."""
        def matmul(x: jax.Array, y: jax.Array) -> jax.Array:
            return pl.pallas_call(
                matmul_kernel,
                out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
                grid=(1,),
            )(x, y)
        return matmul

    # Create test inputs
    m, k, n = 128, 128, 128
    x = jnp.ones((m, k), dtype=jnp.float32)
    y = jnp.ones((k, n), dtype=jnp.float32)

    print(f"\nTiming Pallas matmul compilation: ({m}, {k}) @ ({k}, {n})")
    print("-" * 80)

    # Time the compilation
    with time_pallas_compilation():
        matmul = create_matmul(m, k, n)

        # Time JIT compilation (includes lowering)
        with TIMING_STATS.measure("0_total_jit_compilation"):
            with TIMING_STATS.measure("0a_jit_lower"):
                lowered = jax.jit(matmul).lower(x, y)

            with TIMING_STATS.measure("0b_compile_from_lowered"):
                compiled = lowered.compile()

        # Time the first execution (may include some lazy initialization)
        with TIMING_STATS.measure("5_first_execution"):
            result = compiled(x, y)
            result.block_until_ready()

    # Print detailed timing summary
    TIMING_STATS.print_summary()

    # Export timing data as JSON
    summary = TIMING_STATS.get_summary()
    with open('/tmp/pallas_tpu_timing.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nTiming data exported to /tmp/pallas_tpu_timing.json")
