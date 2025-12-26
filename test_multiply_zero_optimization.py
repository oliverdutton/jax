#!/usr/bin/env python3
"""
Test script to verify XLA's handling of multiply-by-zero optimization.

This demonstrates whether JAX/XLA will optimize away `x * 0` to just `0`,
and how it handles special cases like NaN and infinity.
"""

import jax
import jax.numpy as jnp
import numpy as np


def inspect_hlo(func, *args, **kwargs):
    """Helper to print the HLO for a function."""
    lowered = jax.jit(func).lower(*args, **kwargs)
    print(lowered.as_text())


def test_basic_multiply_zero():
    """Test basic x * 0 optimization."""
    print("="*70)
    print("Test 1: Basic multiply by zero (x * 0)")
    print("="*70)

    def f(x):
        return x * 0.0

    x = jnp.array(5.0)
    result = jax.jit(f)(x)

    print(f"f(5.0) = {result}")
    print(f"Expected: 0.0")
    print("\nHLO IR:")
    inspect_hlo(f, x)
    print()


def test_nan_times_zero():
    """Test NaN * 0 (should preserve NaN, not optimize to 0)."""
    print("="*70)
    print("Test 2: NaN * 0 (should be NaN, not 0)")
    print("="*70)

    def f(x):
        return x * 0.0

    x = jnp.array(jnp.nan)
    result_jit = jax.jit(f)(x)
    result_no_jit = f(x)

    print(f"No JIT: f(NaN) = {result_no_jit}")
    print(f"   JIT: f(NaN) = {result_jit}")
    print(f"Expected: NaN")
    print(f"✓ Correct!" if jnp.isnan(result_jit) else "✗ WRONG!")
    print("\nHLO IR:")
    inspect_hlo(f, x)
    print()


def test_inf_times_zero():
    """Test infinity * 0 (should be NaN according to IEEE 754)."""
    print("="*70)
    print("Test 3: Infinity * 0 (should be NaN per IEEE 754)")
    print("="*70)

    def f(x):
        return x * 0.0

    x = jnp.array(jnp.inf)
    result_jit = jax.jit(f)(x)
    result_no_jit = f(x)

    print(f"No JIT: f(inf) = {result_no_jit}")
    print(f"   JIT: f(inf) = {result_jit}")
    print(f"Expected: NaN")
    print(f"✓ Correct!" if jnp.isnan(result_jit) else "✗ WRONG!")
    print("\nHLO IR:")
    inspect_hlo(f, x)
    print()


def test_expression_with_multiply_zero():
    """Test more complex expression like 1 + x * 0."""
    print("="*70)
    print("Test 4: Complex expression (1 + x * 0)")
    print("="*70)

    def f(x):
        return 1.0 + x * 0.0

    # Test with NaN (from JAX issue #4780)
    x_nan = jnp.array(jnp.nan)
    result_nan = jax.jit(f)(x_nan)

    # Test with regular value
    x_reg = jnp.array(5.0)
    result_reg = jax.jit(f)(x_reg)

    print(f"f(5.0) = {result_reg}, Expected: 1.0")
    print(f"f(NaN) = {result_nan}, Expected: NaN")
    print(f"NaN case: {'✓ Correct!' if jnp.isnan(result_nan) else '✗ WRONG!'}")

    print("\nHLO IR for f(5.0):")
    inspect_hlo(f, x_reg)
    print()


def test_constant_zero():
    """Test when zero is a compile-time constant."""
    print("="*70)
    print("Test 5: Compile-time constant zero")
    print("="*70)

    def f(x):
        # This is a constant zero known at compile time
        zero = 0.0
        return x * zero

    x = jnp.array(5.0)
    result = jax.jit(f)(x)

    print(f"f(5.0) = {result}")
    print(f"Expected: 0.0")
    print("\nHLO IR (should XLA optimize this?):")
    inspect_hlo(f, x)
    print()


def test_dynamic_zero():
    """Test when zero is a runtime value (not compile-time constant)."""
    print("="*70)
    print("Test 6: Runtime (dynamic) zero")
    print("="*70)

    def f(x, y):
        # y is a runtime value that happens to be zero
        return x * y

    x = jnp.array(5.0)
    y = jnp.array(0.0)
    result = jax.jit(f)(x, y)

    print(f"f(5.0, 0.0) = {result}")
    print(f"Expected: 0.0")
    print("\nHLO IR (cannot optimize since y is dynamic):")
    inspect_hlo(f, x, y)
    print()


def test_array_multiply_zero():
    """Test array * 0 to see if optimization applies element-wise."""
    print("="*70)
    print("Test 7: Array multiply by zero")
    print("="*70)

    def f(x):
        return x * 0.0

    x = jnp.array([1.0, 2.0, jnp.nan, jnp.inf, -jnp.inf])
    result = jax.jit(f)(x)

    print(f"Input:  {x}")
    print(f"Result: {result}")
    print(f"Expected: [0., 0., nan, nan, nan]")
    print("\nHLO IR:")
    inspect_hlo(f, x)
    print()


def main():
    print("="*70)
    print("JAX/XLA Multiply-by-Zero Optimization Test")
    print("="*70)
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print()

    test_basic_multiply_zero()
    test_nan_times_zero()
    test_inf_times_zero()
    test_expression_with_multiply_zero()
    test_constant_zero()
    test_dynamic_zero()
    test_array_multiply_zero()

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key findings:

1. XLA's algebraic simplifier CAN optimize `x * 0` to `0` in some cases,
   BUT it must preserve IEEE 754 floating-point semantics.

2. Special cases that CANNOT be optimized to 0:
   - NaN * 0 = NaN (not 0)
   - Inf * 0 = NaN (not 0)
   - -Inf * 0 = NaN (not 0)

3. XLA is conservative with these optimizations to avoid breaking
   floating-point semantics. Past bugs (JAX issue #4780) were fixed.

4. Whether the optimization happens depends on:
   - Whether zero is a compile-time constant
   - The XLA optimization level
   - Whether the value could be NaN/Inf

5. For guaranteed optimization, use explicit constants in your code.
   XLA may NOT optimize dynamic zeros (runtime values).

References:
- JAX issue #4780: https://github.com/jax-ml/jax/issues/4780
- XLA algebraic_simplifier.cc source code
- JAX test: tests/api_test.py::test_jit_nan_times_zero
""")


if __name__ == "__main__":
    main()
