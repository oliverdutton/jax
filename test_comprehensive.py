#!/usr/bin/env python3
"""
Comprehensive test of the StableHLO to Jaxpr decompiler.

Tests that decompiled jaxpr produces the same results as the original compiled function.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr
from jax._src import core


def decompile_and_test(func, *args, name="test", atol=1e-5):
    """
    Decompile a compiled function and verify it produces the same results.

    Args:
        func: Function to test
        *args: Arguments to pass to the function
        name: Test name for display
        atol: Absolute tolerance for comparing results

    Returns:
        bool: True if test passed
    """
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

    # Get the lowered and compiled versions
    lowered = jax.jit(func).lower(*args)
    compiled = lowered.compile()

    # Run the original compiled version
    expected = compiled(*args)
    print(f"Original result: {expected}")

    # Decompile from StableHLO
    mlir_module = lowered.compiler_ir(dialect='stablehlo')
    decompiler = StableHLOToJaxpr()
    functions = decompiler.decompile_module(mlir_module)

    # Find the main function
    main_func = None
    for fname, func_obj in functions.items():
        if 'main' in fname:
            main_func = func_obj
            break

    if main_func is None:
        print("ERROR: Could not find main function")
        return False

    print(f"\nDecompiled jaxpr (first 500 chars):")
    jaxpr_str = str(main_func.jaxpr)
    print(jaxpr_str[:500])
    if len(jaxpr_str) > 500:
        print("...")

    # Execute the decompiled jaxpr
    try:
        constvals = decompiler.constvals
        result = core.eval_jaxpr(main_func.jaxpr, constvals, *args)

        # Compare results
        if isinstance(result, (list, tuple)):
            result = result[0] if len(result) == 1 else result

        print(f"\nDecompiled result: {result}")

        match = np.allclose(result, expected, atol=atol)
        print(f"Match: {match}")

        if not match:
            print(f"ERROR: Results don't match!")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
            print(f"  Max diff: {np.max(np.abs(np.asarray(result) - np.asarray(expected)))}")

        return match

    except Exception as e:
        print(f"\nERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arithmetic():
    """Test basic arithmetic operations."""
    def f(x, y):
        return x + y * 2.0 - x / y

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    return decompile_and_test(f, x, y, name="Arithmetic Operations")


def test_unary_ops():
    """Test unary operations."""
    def f(x):
        return jnp.tanh(jnp.exp(jnp.sqrt(jnp.abs(x))))

    x = jnp.array([1.0, 4.0, 9.0])
    return decompile_and_test(f, x, name="Unary Operations")


def test_matrix_multiply():
    """Test matrix multiplication."""
    def f(x, y):
        return jnp.dot(x, y)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    return decompile_and_test(f, x, y, name="Matrix Multiplication")


def test_while_loop():
    """Test while loop."""
    def f(n):
        def cond_fun(val):
            i, total = val
            return i < n
        def body_fun(val):
            i, total = val
            return (i + 1, total + i)
        _, result = lax.while_loop(cond_fun, body_fun, (0, 0))
        return result

    n = 10
    return decompile_and_test(f, n, name="While Loop")


def test_cond():
    """Test conditional."""
    def f(x):
        return lax.cond(x >= 0, lambda x: x * 2, lambda x: -x, x)

    x = jnp.array(5.0)
    passed = decompile_and_test(f, x, name="Cond (positive)")

    x_neg = jnp.array(-3.0)
    passed &= decompile_and_test(f, x_neg, name="Cond (negative)")

    return passed


def test_nested_control_flow():
    """Test nested control flow."""
    def f(x):
        def body(i, acc):
            return lax.cond(i % 2 == 0,
                           lambda a: a + i,
                           lambda a: a - i,
                           acc)
        return lax.fori_loop(0, x, body, 0)

    x = 5
    return decompile_and_test(f, x, name="Nested Control Flow (fori_loop with cond)")


def test_broadcasting():
    """Test broadcasting operations."""
    def f(x, y):
        return x + y

    x = jnp.array([[1.0, 2.0, 3.0]])
    y = jnp.array([[4.0], [5.0], [6.0]])
    return decompile_and_test(f, x, y, name="Broadcasting")


def test_reshape_transpose():
    """Test reshape and transpose."""
    def f(x):
        return jnp.transpose(jnp.reshape(x, (2, 3, 4)))

    x = jnp.arange(24.0)
    return decompile_and_test(f, x, name="Reshape and Transpose")


def test_reduction():
    """Test reduction operations."""
    def f(x):
        return jnp.sum(x, axis=0)

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return decompile_and_test(f, x, name="Reduction (sum)")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE STABLEHLO TO JAXPR DECOMPILER TEST SUITE")
    print("="*80)

    tests = [
        test_arithmetic,
        test_unary_ops,
        test_matrix_multiply,
        test_while_loop,
        test_cond,
        test_broadcasting,
        test_reshape_transpose,
    ]

    results = []
    for test in tests:
        try:
            passed = test()
            results.append((test.__name__, passed))
        except Exception as e:
            print(f"\nFATAL ERROR in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed")

    return passed_count == total


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
