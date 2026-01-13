#!/usr/bin/env python3
"""
Comprehensive test suite for all lax operations - Part 1
Tests: Arithmetic, Comparison, Logical, Trigonometric, Exponential operations
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr


def test_operation(name, fn, inputs, test_num):
    """Test a single operation for decompilation."""
    try:
        # Get expected result
        expected = fn(*inputs)

        # Compile and decompile
        jitted_fn = jax.jit(fn)
        lowered = jitted_fn.lower(*inputs)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print(f"  {test_num}. {name}: ❌ FAIL - No main function")
            return False

        # Call decompiled function
        result = main_func.callable_fn(*inputs)

        # Compare results
        match = np.allclose(result, expected, atol=1e-5, rtol=1e-5, equal_nan=True)
        if match:
            print(f"  {test_num}. {name}: ✅ PASS")
        else:
            print(f"  {test_num}. {name}: ❌ FAIL - Result mismatch")
            print(f"      Expected: {expected}")
            print(f"      Got: {result}")
        return match

    except Exception as e:
        print(f"  {test_num}. {name}: ❌ FAIL - Exception: {e}")
        return False


def test_arithmetic_operations():
    """Test all 15 arithmetic operations."""
    print("\n" + "="*80)
    print("ARITHMETIC OPERATIONS (15 tests)")
    print("="*80)

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.array([2.0, 3.0, 4.0, 5.0])

    tests = [
        ("lax.abs", lambda x: lax.abs(x), [x]),
        ("lax.add", lambda x, y: lax.add(x, y), [x, y]),
        ("lax.ceil", lambda x: lax.ceil(x), [x]),
        ("lax.div", lambda x, y: lax.div(x, y), [x, y]),
        ("lax.floor", lambda x: lax.floor(x), [x]),
        ("lax.integer_pow", lambda x: lax.integer_pow(x, 2), [x]),
        ("lax.mul", lambda x, y: lax.mul(x, y), [x, y]),
        ("lax.neg", lambda x: lax.neg(x), [x]),
        ("lax.pow", lambda x, y: lax.pow(x, y), [x, jnp.array([2.0, 2.0, 2.0, 2.0])]),
        ("lax.rem", lambda x, y: lax.rem(x, y), [x, y]),
        ("lax.round", lambda x: lax.round(x), [x]),
        ("lax.rsqrt", lambda x: lax.rsqrt(x), [x]),
        ("lax.sign", lambda x: lax.sign(x), [x]),
        ("lax.sqrt", lambda x: lax.sqrt(x), [x]),
        ("lax.sub", lambda x, y: lax.sub(x, y), [x, y]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


def test_comparison_operations():
    """Test all 9 comparison operations."""
    print("\n" + "="*80)
    print("COMPARISON OPERATIONS (9 tests)")
    print("="*80)

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.array([2.0, 2.0, 2.0, 2.0])

    tests = [
        ("lax.clamp", lambda mn, x, mx: lax.clamp(mn, x, mx), [jnp.array(1.0), y, jnp.array(3.0)]),
        ("lax.eq", lambda x, y: lax.eq(x, y), [x, y]),
        ("lax.ge", lambda x, y: lax.ge(x, y), [x, y]),
        ("lax.gt", lambda x, y: lax.gt(x, y), [x, y]),
        ("lax.le", lambda x, y: lax.le(x, y), [x, y]),
        ("lax.lt", lambda x, y: lax.lt(x, y), [x, y]),
        ("lax.max", lambda x, y: lax.max(x, y), [x, y]),
        ("lax.min", lambda x, y: lax.min(x, y), [x, y]),
        ("lax.ne", lambda x, y: lax.ne(x, y), [x, y]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


def test_logical_operations():
    """Test all 7 logical/bitwise operations."""
    print("\n" + "="*80)
    print("LOGICAL OPERATIONS (7 tests)")
    print("="*80)

    x = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    y = jnp.array([2, 2, 2, 2], dtype=jnp.int32)
    shift = jnp.array([1, 2, 1, 2], dtype=jnp.int32)

    tests = [
        ("lax.bitwise_and", lambda x, y: lax.bitwise_and(x, y), [x, y]),
        ("lax.bitwise_not", lambda x: lax.bitwise_not(x), [x]),
        ("lax.bitwise_or", lambda x, y: lax.bitwise_or(x, y), [x, y]),
        ("lax.bitwise_xor", lambda x, y: lax.bitwise_xor(x, y), [x, y]),
        ("lax.shift_left", lambda x, s: lax.shift_left(x, s), [x, shift]),
        ("lax.shift_right_arithmetic", lambda x, s: lax.shift_right_arithmetic(x, s), [x, shift]),
        ("lax.shift_right_logical", lambda x, s: lax.shift_right_logical(x, s), [x, shift]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


def test_trigonometric_operations():
    """Test all 13 trigonometric operations."""
    print("\n" + "="*80)
    print("TRIGONOMETRIC OPERATIONS (13 tests)")
    print("="*80)

    x = jnp.array([0.1, 0.5, 1.0, 1.5])
    y = jnp.array([0.2, 0.6, 1.1, 1.6])

    tests = [
        ("lax.acos", lambda x: lax.acos(x), [jnp.array([0.1, 0.5, 0.9, 0.99])]),
        ("lax.acosh", lambda x: lax.acosh(x), [jnp.array([1.1, 1.5, 2.0, 3.0])]),
        ("lax.asin", lambda x: lax.asin(x), [jnp.array([0.1, 0.5, 0.9, 0.99])]),
        ("lax.asinh", lambda x: lax.asinh(x), [x]),
        ("lax.atan", lambda x: lax.atan(x), [x]),
        ("lax.atan2", lambda x, y: lax.atan2(x, y), [x, y]),
        ("lax.atanh", lambda x: lax.atanh(x), [jnp.array([0.1, 0.5, 0.9, 0.99])]),
        ("lax.cos", lambda x: lax.cos(x), [x]),
        ("lax.cosh", lambda x: lax.cosh(x), [x]),
        ("lax.sin", lambda x: lax.sin(x), [x]),
        ("lax.sinh", lambda x: lax.sinh(x), [x]),
        ("lax.tan", lambda x: lax.tan(x), [x]),
        ("lax.tanh", lambda x: lax.tanh(x), [x]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


def test_exponential_operations():
    """Test all 8 exponential operations."""
    print("\n" + "="*80)
    print("EXPONENTIAL OPERATIONS (8 tests)")
    print("="*80)

    x = jnp.array([0.1, 0.5, 1.0, 2.0])

    tests = [
        ("lax.erf", lambda x: lax.erf(x), [x]),
        ("lax.erf_inv", lambda x: lax.erf_inv(x), [jnp.array([0.1, 0.3, 0.5, 0.7])]),
        ("lax.erfc", lambda x: lax.erfc(x), [x]),
        ("lax.exp", lambda x: lax.exp(x), [x]),
        ("lax.expm1", lambda x: lax.expm1(x), [x]),
        ("lax.log", lambda x: lax.log(x), [x]),
        ("lax.log1p", lambda x: lax.log1p(x), [x]),
        ("lax.logistic", lambda x: lax.logistic(x), [x]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE LAX OPERATIONS TEST - PART 1")
    print("Testing: Arithmetic, Comparison, Logical, Trigonometric, Exponential")
    print("="*80)

    all_passed = []

    all_passed.extend(test_arithmetic_operations())
    all_passed.extend(test_comparison_operations())
    all_passed.extend(test_logical_operations())
    all_passed.extend(test_trigonometric_operations())
    all_passed.extend(test_exponential_operations())

    print("\n" + "="*80)
    print(f"PART 1 SUMMARY: {sum(all_passed)}/{len(all_passed)} tests passed")
    print("="*80)

    exit(0 if all(all_passed) else 1)
