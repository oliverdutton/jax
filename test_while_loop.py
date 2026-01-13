#!/usr/bin/env python3
"""Test while loop decompilation."""

import jax
import jax.numpy as jnp
from jax import lax
from hlo_to_jaxpr import StableHLOToJaxpr
import numpy as np


def test_simple_while():
    """Test a simple while loop."""
    print("=" * 80)
    print("TESTING WHILE LOOP DECOMPILATION")
    print("=" * 80)

    # Countdown from n to 0
    def countdown(n):
        def cond_fn(i):
            return i > 0

        def body_fn(i):
            return i - 1

        return lax.while_loop(cond_fn, body_fn, n)

    # Test with different values
    test_values = [jnp.array(5), jnp.array(10), jnp.array(1)]

    for val in test_values:
        print(f"\n--- Testing countdown from {val} ---")
        expected = countdown(val)
        print(f"Expected result: {expected}")

        # Lower to StableHLO
        lowered = jax.jit(countdown).lower(val)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        print("\nStableHLO IR snippet:")
        ir_str = str(mlir_module)
        # Show just the while operation
        if 'stablehlo.while' in ir_str:
            print("Found stablehlo.while operation")
            lines = ir_str.split('\n')
            for i, line in enumerate(lines):
                if 'while' in line.lower():
                    print(line)

        # Decompile
        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions['"main"']

        # Execute decompiled function
        try:
            result = main_func.callable_fn(val)
            print(f"Decompiled result: {result}")
            print(f"Match: {np.allclose(result, expected)}")
        except Exception as e:
            print(f"Error executing decompiled function: {e}")
            import traceback
            traceback.print_exc()


def test_accumulator_while():
    """Test while loop with accumulator."""
    print("\n" + "=" * 80)
    print("TESTING ACCUMULATOR WHILE LOOP")
    print("=" * 80)

    # Sum from 0 to n
    def sum_to_n(n):
        def cond_fn(state):
            i, acc = state
            return i <= n

        def body_fn(state):
            i, acc = state
            return i + 1, acc + i

        init_state = (jnp.array(0), jnp.array(0))
        _, result = lax.while_loop(cond_fn, body_fn, init_state)
        return result

    test_vals = [jnp.array(5), jnp.array(10)]

    for val in test_vals:
        print(f"\n--- Sum from 0 to {val} ---")
        expected = sum_to_n(val)
        print(f"Expected: {expected}")

        # Lower and decompile
        lowered = jax.jit(sum_to_n).lower(val)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)
        main_func = functions['"main"']

        try:
            result = main_func.callable_fn(val)
            print(f"Decompiled: {result}")
            print(f"Match: {np.allclose(result, expected)}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    test_simple_while()
    test_accumulator_while()
    print("\n" + "=" * 80)
    print("WHILE LOOP TESTS COMPLETE")
    print("=" * 80)
