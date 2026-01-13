#!/usr/bin/env python3
"""
Comprehensive test suite for the refactored StableHLO decompiler.
Tests all operations including arithmetic, reductions, control flow, and more.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr


def decompile_and_test(func, *args, name="test", atol=1e-5, rtol=1e-5):
    """
    Decompile a function and verify it produces the same results.

    Args:
        func: Function to test
        *args: Arguments to pass to the function
        name: Test name for display
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        bool: True if test passed
    """
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

    try:
        # Get expected result
        expected = func(*args)
        print(f"Expected result: {expected}")

        # Lower to StableHLO and decompile
        lowered = jax.jit(func).lower(*args)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        # Find main function
        main_func = functions.get('"main"')
        if main_func is None:
            print("ERROR: Could not find main function")
            return False

        # Execute decompiled function
        result = main_func.callable_fn(*args)
        print(f"Decompiled result: {result}")

        # Compare results
        if isinstance(result, tuple) and isinstance(expected, tuple):
            match = all(np.allclose(r, e, atol=atol, rtol=rtol) for r, e in zip(result, expected))
        else:
            match = np.allclose(result, expected, atol=atol, rtol=rtol)

        print(f"✓ PASS" if match else "✗ FAIL")

        if not match:
            print(f"ERROR: Results don't match!")
            if not isinstance(result, tuple):
                print(f"  Max diff: {np.max(np.abs(np.asarray(result) - np.asarray(expected)))}")

        return match

    except Exception as e:
        print(f"✗ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# ARITHMETIC TESTS
# ============================================================================

def test_arithmetic_basic():
    """Test basic arithmetic operations."""
    def f(x, y):
        return x + y * 2.0 - x / y

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    return decompile_and_test(f, x, y, name="Arithmetic: add, mul, sub, div")


def test_arithmetic_advanced():
    """Test advanced arithmetic."""
    def f(x, y):
        return lax.max(x, y) + lax.min(x, y) + lax.rem(x, y)

    x = jnp.array([1.0, 5.0, 3.0])
    y = jnp.array([4.0, 2.0, 6.0])
    return decompile_and_test(f, x, y, name="Arithmetic: max, min, rem")


# ============================================================================
# UNARY OPERATION TESTS
# ============================================================================

def test_unary_basic():
    """Test basic unary operations."""
    def f(x):
        return jnp.abs(jnp.sin(x)) + jnp.cos(x)

    x = jnp.array([0.0, jnp.pi/4, jnp.pi/2])
    return decompile_and_test(f, x, name="Unary: abs, sin, cos")


def test_unary_advanced():
    """Test advanced unary operations."""
    def f(x):
        return jnp.exp(jnp.log(jnp.abs(x) + 1.0)) + jnp.sqrt(jnp.abs(x))

    x = jnp.array([1.0, 4.0, 9.0])
    return decompile_and_test(f, x, name="Unary: exp, log, sqrt")


def test_unary_tanh():
    """Test tanh."""
    def f(x):
        return jnp.tanh(x * 2.0)

    x = jnp.array([-1.0, 0.0, 1.0])
    return decompile_and_test(f, x, name="Unary: tanh")


# ============================================================================
# COMPARISON TESTS
# ============================================================================

def test_comparison():
    """Test comparison operations."""
    def f(x, y):
        return (x < y).astype(jnp.float32) + (x > y).astype(jnp.float32) * 2.0

    x = jnp.array([1.0, 5.0, 3.0])
    y = jnp.array([4.0, 2.0, 3.0])
    return decompile_and_test(f, x, y, name="Comparison: lt, gt")


def test_comparison_equality():
    """Test equality comparisons."""
    def f(x, y):
        return (x == y).astype(jnp.float32) + (x != y).astype(jnp.float32) * 2.0

    x = jnp.array([1.0, 5.0, 3.0])
    y = jnp.array([1.0, 2.0, 3.0])
    return decompile_and_test(f, x, y, name="Comparison: eq, ne")


# ============================================================================
# REDUCTION TESTS
# ============================================================================

def test_reduce_sum():
    """Test sum reduction."""
    def f(x):
        return jnp.sum(x)

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    return decompile_and_test(f, x, name="Reduction: sum (all axes)")


def test_reduce_sum_axis():
    """Test sum reduction along axis."""
    def f(x):
        return jnp.sum(x, axis=0)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    return decompile_and_test(f, x, name="Reduction: sum (axis=0)")


def test_reduce_sum_keepdims():
    """Test sum with keepdims."""
    def f(x):
        return jnp.sum(x, axis=1, keepdims=True)

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return decompile_and_test(f, x, name="Reduction: sum (axis=1, keepdims)")


# ============================================================================
# SHAPE MANIPULATION TESTS
# ============================================================================

def test_reshape():
    """Test reshape."""
    def f(x):
        return jnp.reshape(x, (2, 3))

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return decompile_and_test(f, x, name="Shape: reshape")


def test_transpose():
    """Test transpose."""
    def f(x):
        return jnp.transpose(x)

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return decompile_and_test(f, x, name="Shape: transpose")


def test_transpose_axes():
    """Test transpose with axes."""
    def f(x):
        return jnp.transpose(x, (2, 0, 1))

    x = jnp.ones((2, 3, 4))
    return decompile_and_test(f, x, name="Shape: transpose with axes")


def test_broadcast():
    """Test broadcasting."""
    def f(x, y):
        return x + y  # y will be broadcasted

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([10.0, 20.0])
    return decompile_and_test(f, x, y, name="Shape: broadcast addition")


# ============================================================================
# MATRIX OPERATION TESTS
# ============================================================================

def test_matmul():
    """Test matrix multiplication."""
    def f(x, y):
        return jnp.dot(x, y)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    return decompile_and_test(f, x, y, name="Matrix: dot product")


def test_matmul_vector():
    """Test matrix-vector multiplication."""
    def f(x, y):
        return jnp.dot(x, y)

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = jnp.array([1.0, 2.0, 3.0])
    return decompile_and_test(f, x, y, name="Matrix: matrix-vector product")


# ============================================================================
# CONTROL FLOW TESTS
# ============================================================================

def test_while_simple():
    """Test simple while loop."""
    def f(n):
        def cond(i):
            return i > 0
        def body(i):
            return i - 1
        return lax.while_loop(cond, body, n)

    n = jnp.array(10)
    return decompile_and_test(f, n, name="Control flow: simple while loop")


def test_while_accumulator():
    """Test while loop with accumulator."""
    def f(n):
        def cond(state):
            i, acc = state
            return i <= n
        def body(state):
            i, acc = state
            return i + 1, acc + i
        init = (jnp.array(0), jnp.array(0))
        _, result = lax.while_loop(cond, body, init)
        return result

    n = jnp.array(10)
    return decompile_and_test(f, n, name="Control flow: while with accumulator")


def test_cond_simple():
    """Test simple conditional."""
    def f(x):
        return lax.cond(x < 0, lambda x: -x, lambda x: x, x)

    return (decompile_and_test(f, jnp.array(-5.0), name="Control flow: cond (negative)") and
            decompile_and_test(f, jnp.array(5.0), name="Control flow: cond (positive)"))


# ============================================================================
# SELECT AND CLAMP TESTS
# ============================================================================

def test_select():
    """Test select operation."""
    def f(pred, x, y):
        return lax.select(pred, x, y)

    pred = jnp.array([True, False, True])
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([10.0, 20.0, 30.0])
    return decompile_and_test(f, pred, x, y, name="Select operation")


def test_clamp():
    """Test clamp operation."""
    def f(x):
        return lax.clamp(0.0, x, 10.0)

    x = jnp.array([-5.0, 5.0, 15.0])
    return decompile_and_test(f, x, name="Clamp operation")


# ============================================================================
# TYPE CONVERSION TESTS
# ============================================================================

def test_type_conversion():
    """Test type conversion."""
    def f(x):
        return x.astype(jnp.int32).astype(jnp.float32)

    x = jnp.array([1.5, 2.7, 3.2])
    return decompile_and_test(f, x, name="Type conversion")


# ============================================================================
# COMPLEX TESTS
# ============================================================================

def test_neural_network_layer():
    """Test a simple neural network layer."""
    def f(x, W, b):
        return jnp.tanh(jnp.dot(W, x) + b)

    x = jnp.array([1.0, 2.0, 3.0])
    W = jnp.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
    b = jnp.array([0.1, -0.2])
    return decompile_and_test(f, x, W, b, name="Complex: neural network layer")


def test_polynomial():
    """Test polynomial evaluation."""
    def f(x):
        return 2.0 * x**3 - 3.0 * x**2 + 5.0 * x - 1.0

    x = jnp.array([1.0, 2.0, 3.0])
    return decompile_and_test(f, x, name="Complex: polynomial")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        # Arithmetic
        test_arithmetic_basic,
        test_arithmetic_advanced,

        # Unary
        test_unary_basic,
        test_unary_advanced,
        test_unary_tanh,

        # Comparison
        test_comparison,
        test_comparison_equality,

        # Reductions
        test_reduce_sum,
        test_reduce_sum_axis,
        test_reduce_sum_keepdims,

        # Shape manipulation
        test_reshape,
        test_transpose,
        test_transpose_axes,
        test_broadcast,

        # Matrix operations
        test_matmul,
        test_matmul_vector,

        # Control flow
        test_while_simple,
        test_while_accumulator,
        test_cond_simple,

        # Select and clamp
        test_select,
        test_clamp,

        # Type conversion
        test_type_conversion,

        # Complex
        test_neural_network_layer,
        test_polynomial,
    ]

    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*80)

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Test {test.__name__} raised exception: {e}")
            failed += 1

    print("\n" + "="*80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*80)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
