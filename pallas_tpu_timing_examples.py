"""Examples demonstrating Pallas TPU compilation timing instrumentation.

This module provides several examples showing how to use the timing
instrumentation for different Pallas kernels and compilation scenarios.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from pallas_tpu_timing_instrumentation import (
    time_pallas_compilation,
    TIMING_STATS,
)


def example_1_simple_add():
    """Example 1: Simple element-wise addition kernel."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Element-wise Addition")
    print("=" * 80)

    def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    size = 1024
    x = jnp.ones(size, dtype=jnp.float32)
    y = jnp.ones(size, dtype=jnp.float32)

    with time_pallas_compilation():
        def add(x, y):
            return pl.pallas_call(
                add_kernel,
                out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            )(x, y)

        with TIMING_STATS.measure("total_compilation"):
            compiled = jax.jit(add).lower(x, y).compile()

        result = compiled(x, y)
        result.block_until_ready()

    TIMING_STATS.print_summary()
    TIMING_STATS.reset()


def example_2_matmul_with_grid():
    """Example 2: Matrix multiplication with grid mapping."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Matrix Multiplication with Grid")
    print("=" * 80)

    def matmul_kernel(x_ref, y_ref, o_ref):
        i = pl.program_id(0)
        j = pl.program_id(1)

        @pl.when(i * 8 < x_ref.shape[0] and j * 8 < y_ref.shape[1])
        def _():
            o_ref[i, j] = jnp.dot(x_ref[i, :], y_ref[:, j])

    m, k, n = 256, 256, 256
    block_m, block_n = 8, 8

    x = jnp.ones((m, k), dtype=jnp.float32)
    y = jnp.ones((k, n), dtype=jnp.float32)

    with time_pallas_compilation():
        def matmul(x, y):
            grid = (m // block_m, n // block_n)
            return pl.pallas_call(
                matmul_kernel,
                out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
                grid=grid,
            )(x, y)

        with TIMING_STATS.measure("total_compilation"):
            lowered = jax.jit(matmul).lower(x, y)
            compiled = lowered.compile()

        result = compiled(x, y)
        result.block_until_ready()

    TIMING_STATS.print_summary()
    TIMING_STATS.reset()


def example_3_blocked_matmul():
    """Example 3: Blocked matrix multiplication using BlockSpec."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Blocked Matrix Multiplication")
    print("=" * 80)

    def matmul_kernel(x_ref, y_ref, o_ref):
        # Accumulate block-wise
        acc = jnp.zeros(o_ref.shape, dtype=jnp.float32)
        acc += x_ref[...] @ y_ref[...]
        o_ref[...] = acc

    m, k, n = 512, 512, 512
    bm, bk, bn = 128, 128, 128

    x = jnp.ones((m, k), dtype=jnp.float32)
    y = jnp.ones((k, n), dtype=jnp.float32)

    with time_pallas_compilation():
        def blocked_matmul(x, y):
            grid = (m // bm, n // bn)
            return pl.pallas_call(
                matmul_kernel,
                out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, k), lambda i, j: (i, 0)),
                    pl.BlockSpec((k, bn), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            )(x, y)

        with TIMING_STATS.measure("total_compilation"):
            compiled = jax.jit(blocked_matmul).lower(x, y).compile()

        result = compiled(x, y)
        result.block_until_ready()

    TIMING_STATS.print_summary()
    TIMING_STATS.reset()


def example_4_scan_operation():
    """Example 4: Cumulative sum using scan."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Cumulative Sum (Scan)")
    print("=" * 80)

    def cumsum_kernel(x_ref, o_ref):
        # Simple sequential cumsum
        o_ref[0] = x_ref[0]
        for i in range(1, x_ref.shape[0]):
            o_ref[i] = o_ref[i-1] + x_ref[i]

    size = 1024
    x = jnp.arange(size, dtype=jnp.float32)

    with time_pallas_compilation():
        def cumsum(x):
            return pl.pallas_call(
                cumsum_kernel,
                out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            )(x)

        with TIMING_STATS.measure("total_compilation"):
            compiled = jax.jit(cumsum).lower(x).compile()

        result = compiled(x)
        result.block_until_ready()

    TIMING_STATS.print_summary()
    TIMING_STATS.reset()


def example_5_multiple_kernels():
    """Example 5: Time compilation of multiple different kernels."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Multiple Kernel Compilation Timing")
    print("=" * 80)

    # Define several kernels
    def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    def mul_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] * y_ref[...]

    def relu_kernel(x_ref, o_ref):
        o_ref[...] = jnp.maximum(x_ref[...], 0.0)

    size = 2048
    x = jnp.ones(size, dtype=jnp.float32)
    y = jnp.ones(size, dtype=jnp.float32)

    with time_pallas_compilation():
        # Compile each kernel separately and time them
        kernels_to_compile = [
            ("add", add_kernel, (x, y)),
            ("mul", mul_kernel, (x, y)),
            ("relu", relu_kernel, (x,)),
        ]

        for name, kernel, args in kernels_to_compile:
            with TIMING_STATS.measure(f"kernel_{name}_compilation"):
                if len(args) == 2:
                    def fn(x, y):
                        return pl.pallas_call(
                            kernel,
                            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                        )(x, y)
                    compiled = jax.jit(fn).lower(*args).compile()
                else:
                    def fn(x):
                        return pl.pallas_call(
                            kernel,
                            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                        )(x)
                    compiled = jax.jit(fn).lower(*args).compile()

    TIMING_STATS.print_summary()
    TIMING_STATS.reset()


def example_6_compare_compilation_methods():
    """Example 6: Compare different compilation paths."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Compare Compilation Methods")
    print("=" * 80)

    def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    size = 1024
    x = jnp.ones(size, dtype=jnp.float32)
    y = jnp.ones(size, dtype=jnp.float32)

    def add(x, y):
        return pl.pallas_call(
            add_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        )(x, y)

    with time_pallas_compilation():
        # Method 1: Direct jit
        print("\nMethod 1: Direct jit()")
        with TIMING_STATS.measure("method1_direct_jit"):
            compiled1 = jax.jit(add)
            result1 = compiled1(x, y)
            result1.block_until_ready()

        # Method 2: lower then compile
        print("Method 2: lower().compile()")
        with TIMING_STATS.measure("method2_lower_compile"):
            lowered2 = jax.jit(add).lower(x, y)
            compiled2 = lowered2.compile()
            result2 = compiled2(x, y)
            result2.block_until_ready()

        # Method 3: aot_compile
        print("Method 3: Using as_text() for HLO inspection")
        with TIMING_STATS.measure("method3_get_hlo"):
            lowered3 = jax.jit(add).lower(x, y)
            hlo_text = lowered3.as_text()
            compiled3 = lowered3.compile()

    TIMING_STATS.print_summary()
    TIMING_STATS.reset()


def run_all_examples():
    """Run all examples."""
    examples = [
        example_1_simple_add,
        example_2_matmul_with_grid,
        example_3_blocked_matmul,
        example_4_scan_operation,
        example_5_multiple_kernels,
        example_6_compare_compilation_methods,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("Pallas TPU Compilation Timing Examples")
    print("=" * 80)
    print("This script demonstrates timing instrumentation for various")
    print("Pallas TPU kernels and compilation scenarios.")
    print("=" * 80)

    # Run a specific example or all
    import sys
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_simple_add,
            example_2_matmul_with_grid,
            example_3_blocked_matmul,
            example_4_scan_operation,
            example_5_multiple_kernels,
            example_6_compare_compilation_methods,
        ]
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        run_all_examples()
