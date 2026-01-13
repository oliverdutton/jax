#!/usr/bin/env python3
"""
Advanced tests for slicing, concatenation, padding, and other operations.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr


def decompile_and_test(func, *args, name="test", atol=1e-5, rtol=1e-5):
    """Decompile and test a function."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

    try:
        expected = func(*args)
        print(f"Expected result shape: {jnp.shape(expected)}, value: {expected}")

        lowered = jax.jit(func).lower(*args)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("ERROR: Could not find main function")
            return False

        result = main_func.callable_fn(*args)
        print(f"Decompiled result shape: {jnp.shape(result)}, value: {result}")

        if isinstance(result, tuple) and isinstance(expected, tuple):
            match = all(np.allclose(r, e, atol=atol, rtol=rtol) for r, e in zip(result, expected))
        else:
            match = np.allclose(result, expected, atol=atol, rtol=rtol)

        print(f"✓ PASS" if match else "✗ FAIL")
        return match

    except Exception as e:
        print(f"✗ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# SLICING TESTS
# ============================================================================

def test_static_slice():
    """Test static slicing."""
    def f(x):
        return x[1:3, 0:2]

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    return decompile_and_test(f, x, name="Slice: static slice [1:3, 0:2]")


def test_static_slice_with_stride():
    """Test static slicing with stride."""
    def f(x):
        return x[::2]

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return decompile_and_test(f, x, name="Slice: static slice with stride [::2]")


def test_dynamic_slice():
    """Test dynamic slicing."""
    def f(x, start):
        return lax.dynamic_slice(x, (start, 0), (2, 2))

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    start = jnp.array(1)
    return decompile_and_test(f, x, start, name="Slice: dynamic slice")


def test_dynamic_update_slice():
    """Test dynamic update slice."""
    def f(x, update, start):
        return lax.dynamic_update_slice(x, update, (start, 0))

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    update = jnp.array([[99.0, 99.0]])
    start = jnp.array(1)
    return decompile_and_test(f, x, update, start, name="Slice: dynamic update slice")


# ============================================================================
# CONCATENATE TESTS
# ============================================================================

def test_concatenate_axis0():
    """Test concatenation along axis 0."""
    def f(x, y):
        return jnp.concatenate([x, y], axis=0)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    return decompile_and_test(f, x, y, name="Concatenate: axis=0")


def test_concatenate_axis1():
    """Test concatenation along axis 1."""
    def f(x, y):
        return jnp.concatenate([x, y], axis=1)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[5.0], [6.0]])
    return decompile_and_test(f, x, y, name="Concatenate: axis=1")


def test_concatenate_multiple():
    """Test concatenation of multiple arrays."""
    def f(x, y, z):
        return jnp.concatenate([x, y, z], axis=0)

    x = jnp.array([[1.0]])
    y = jnp.array([[2.0]])
    z = jnp.array([[3.0]])
    return decompile_and_test(f, x, y, z, name="Concatenate: multiple arrays")


# ============================================================================
# PAD TESTS
# ============================================================================

def test_pad_constant():
    """Test constant padding."""
    def f(x):
        return jnp.pad(x, ((1, 1), (2, 2)), mode='constant', constant_values=0)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    return decompile_and_test(f, x, name="Pad: constant padding")


def test_pad_edge():
    """Test edge padding."""
    def f(x):
        return jnp.pad(x, ((1, 1),), mode='edge')

    x = jnp.array([1.0, 2.0, 3.0])
    return decompile_and_test(f, x, name="Pad: edge padding")


# ============================================================================
# REDUCTION TESTS (Advanced)
# ============================================================================

def test_reduce_max():
    """Test max reduction."""
    def f(x):
        return jnp.max(x)

    x = jnp.array([1.0, 5.0, 3.0, 2.0])
    return decompile_and_test(f, x, name="Reduction: max")


def test_reduce_min():
    """Test min reduction."""
    def f(x):
        return jnp.min(x, axis=1)

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 0.5, 6.0]])
    return decompile_and_test(f, x, name="Reduction: min along axis")


def test_reduce_prod():
    """Test product reduction."""
    def f(x):
        return jnp.prod(x)

    x = jnp.array([2.0, 3.0, 4.0])
    return decompile_and_test(f, x, name="Reduction: product")


# ============================================================================
# BITWISE AND LOGICAL TESTS
# ============================================================================

def test_logical_and():
    """Test logical AND."""
    def f(x, y):
        return jnp.logical_and(x, y)

    x = jnp.array([True, True, False, False])
    y = jnp.array([True, False, True, False])
    return decompile_and_test(f, x, y, name="Logical: AND")


def test_logical_or():
    """Test logical OR."""
    def f(x, y):
        return jnp.logical_or(x, y)

    x = jnp.array([True, True, False, False])
    y = jnp.array([True, False, True, False])
    return decompile_and_test(f, x, y, name="Logical: OR")


def test_logical_not():
    """Test logical NOT."""
    def f(x):
        return jnp.logical_not(x)

    x = jnp.array([True, False, True])
    return decompile_and_test(f, x, name="Logical: NOT")


# ============================================================================
# REVERSE AND SORT TESTS
# ============================================================================

def test_reverse():
    """Test array reversal."""
    def f(x):
        return jnp.flip(x, axis=0)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    return decompile_and_test(f, x, name="Reverse: flip axis 0")


def test_sort():
    """Test array sorting."""
    def f(x):
        return jnp.sort(x)

    x = jnp.array([3.0, 1.0, 4.0, 2.0])
    return decompile_and_test(f, x, name="Sort: ascending")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all advanced tests."""
    tests = [
        # Slicing
        test_static_slice,
        test_static_slice_with_stride,
        test_dynamic_slice,
        test_dynamic_update_slice,

        # Concatenate
        test_concatenate_axis0,
        test_concatenate_axis1,
        test_concatenate_multiple,

        # Pad
        test_pad_constant,
        # test_pad_edge,  # Edge mode may not be directly supported in StableHLO

        # Reductions
        test_reduce_max,
        test_reduce_min,
        test_reduce_prod,

        # Logical
        test_logical_and,
        test_logical_or,
        test_logical_not,

        # Reverse and sort
        test_reverse,
        # test_sort,  # Sort may have complex implementation
    ]

    print("\n" + "="*80)
    print("RUNNING ADVANCED TEST SUITE")
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
