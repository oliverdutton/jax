#!/usr/bin/env python3
"""Test decompiler with control flow operations."""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr
from jax._src import core


def test_while_loop():
    """Test while loop decompilation."""
    print("=" * 80)
    print("TEST: WHILE LOOP")
    print("=" * 80)

    def while_sum(n):
        def cond_fun(val):
            i, total = val
            return i < n
        def body_fun(val):
            i, total = val
            return (i + 1, total + i)
        _, result = lax.while_loop(cond_fun, body_fun, (0, 0))
        return result

    n = 5
    lowered = jax.jit(while_sum).lower(n)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    decompiler = StableHLOToJaxpr()
    functions = decompiler.decompile_module(mlir_module)

    for name, func in functions.items():
        print(f"\nFunction: {name}")
        print(f"Jaxpr:\n{func.jaxpr}")

        print("\nExecuting decompiled jaxpr:")
        try:
            constvals = decompiler.constvals
            result = core.eval_jaxpr(func.jaxpr, constvals, n)
            expected = while_sum(n)
            print(f"Result: {result}")
            print(f"Expected: {expected}")
            print(f"Match: {result[0] == expected}")
        except Exception as e:
            print(f"Execution failed: {e}")
            import traceback
            traceback.print_exc()


def test_cond():
    """Test conditional decompilation."""
    print("\n" + "=" * 80)
    print("TEST: COND")
    print("=" * 80)

    def cond_abs(x):
        return lax.cond(x >= 0, lambda x: x, lambda x: -x, x)

    x = jnp.array(5.0)
    lowered = jax.jit(cond_abs).lower(x)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    decompiler = StableHLOToJaxpr()
    functions = decompiler.decompile_module(mlir_module)

    for name, func in functions.items():
        print(f"\nFunction: {name}")
        print(f"Jaxpr:\n{func.jaxpr}")

        print("\nExecuting decompiled jaxpr (x=5.0):")
        try:
            constvals = decompiler.constvals
            result = core.eval_jaxpr(func.jaxpr, constvals, x)
            expected = cond_abs(x)
            print(f"Result: {result}")
            print(f"Expected: {expected}")
            print(f"Match: {np.allclose(result, expected)}")

            # Test with negative value
            x_neg = jnp.array(-3.0)
            result_neg = core.eval_jaxpr(func.jaxpr, constvals, x_neg)
            expected_neg = cond_abs(x_neg)
            print(f"\nWith x=-3.0:")
            print(f"Result: {result_neg}")
            print(f"Expected: {expected_neg}")
            print(f"Match: {np.allclose(result_neg, expected_neg)}")
        except Exception as e:
            print(f"Execution failed: {e}")
            import traceback
            traceback.print_exc()


def test_scan():
    """Test scan decompilation."""
    print("\n" + "=" * 80)
    print("TEST: SCAN")
    print("=" * 80)

    def cumsum_scan(arr):
        def body(carry, x):
            new_carry = carry + x
            return new_carry, new_carry
        _, result = lax.scan(body, 0.0, arr)
        return result

    arr = jnp.array([1.0, 2.0, 3.0, 4.0])
    lowered = jax.jit(cumsum_scan).lower(arr)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    decompiler = StableHLOToJaxpr()
    functions = decompiler.decompile_module(mlir_module)

    for name, func in functions.items():
        print(f"\nFunction: {name}")
        print(f"Jaxpr (first 500 chars):\n{str(func.jaxpr)[:500]}...")

        if name == '"main"':
            print("\nExecuting decompiled jaxpr:")
            try:
                constvals = decompiler.constvals
                result = core.eval_jaxpr(func.jaxpr, constvals, arr)
                expected = cumsum_scan(arr)
                print(f"Result: {result}")
                print(f"Expected: {expected}")
                print(f"Match: {np.allclose(result, expected)}")
            except Exception as e:
                print(f"Execution failed: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    test_while_loop()
    test_cond()
    test_scan()
