#!/usr/bin/env python3
"""
Comprehensive scatter tests extracted from JAX's test suite.
Tests 20 different scatter configurations to ensure our decompiler handles all cases.
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
        print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")

        lowered = jax.jit(func).lower(*args)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("ERROR: Could not find main function")
            return False

        result = main_func.callable_fn(*args)
        print(f"Decompiled shape: {result.shape}, dtype: {result.dtype}")

        match = np.allclose(result, expected, atol=atol, rtol=rtol)
        print(f"✓ PASS" if match else f"✗ FAIL")
        if not match:
            print(f"Expected: {expected}")
            print(f"Got: {result}")
        return match

    except Exception as e:
        print(f"✗ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 1-5: Basic Scatter Tests
# ============================================================================

def test_scatter_1d_basic():
    """Test 1: Simple 1D scatter."""
    def f(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)
        )
        return lax.scatter(operand, indices, updates, dimension_numbers)

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = jnp.array([[0], [2]])
    updates = jnp.array([10.0, 20.0])

    return decompile_and_test(f, operand, indices, updates, name="Scatter 1D Basic")


def test_scatter_1d_window():
    """Test 2: 1D scatter with update window dimensions."""
    def f(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,)
        )
        return lax.scatter(operand, indices, updates, dimension_numbers)

    operand = jnp.arange(10.0)
    indices = jnp.array([[0], [0], [0]])
    updates = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    return decompile_and_test(f, operand, indices, updates, name="Scatter 1D Window")


def test_scatter_2d_mixed():
    """Test 3: 2D scatter with mixed dimensions."""
    def f(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)
        )
        return lax.scatter(operand, indices, updates, dimension_numbers)

    operand = jnp.arange(50.0).reshape(10, 5)
    indices = jnp.array([[0], [2], [1]])
    updates = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    return decompile_and_test(f, operand, indices, updates, name="Scatter 2D Mixed")


def test_scatter_2d_full_window():
    """Test 4: 2D scatter with full window update."""
    def f(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(1, 2),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0, 1)
        )
        return lax.scatter(operand, indices, updates, dimension_numbers)

    operand = jnp.arange(24.0).reshape(4, 6)
    indices = jnp.array([[0, 0]])
    updates = jnp.array([[[10.0, 11.0], [12.0, 13.0]]])

    return decompile_and_test(f, operand, indices, updates, name="Scatter 2D Full Window")


def test_scatter_3d():
    """Test 5: 3D scatter."""
    def f(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1)
        )
        return lax.scatter(operand, indices, updates, dimension_numbers)

    operand = jnp.arange(60.0).reshape(3, 4, 5)
    indices = jnp.array([[0, 0], [1, 2]])
    updates = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])

    return decompile_and_test(f, operand, indices, updates, name="Scatter 3D")


# ============================================================================
# TEST 6-10: Scatter Add Tests
# ============================================================================

def test_scatter_add_1d():
    """Test 6: Scatter add 1D."""
    def f(operand, indices, updates):
        return operand.at[indices].add(updates)

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = jnp.array([0, 2, 4])
    updates = jnp.array([10.0, 20.0, 30.0])

    return decompile_and_test(f, operand, indices, updates, name="Scatter Add 1D")


def test_scatter_add_2d():
    """Test 7: Scatter add 2D."""
    def f(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)
        )
        return lax.scatter_add(operand, indices, updates, dimension_numbers)

    operand = jnp.arange(20.0).reshape(4, 5)
    indices = jnp.array([[0], [2]])
    updates = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])

    return decompile_and_test(f, operand, indices, updates, name="Scatter Add 2D")


def test_scatter_add_duplicate_indices():
    """Test 8: Scatter add with duplicate indices."""
    def f(operand, indices, updates):
        return operand.at[indices].add(updates)

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = jnp.array([1, 1, 3])  # duplicate index 1
    updates = jnp.array([10.0, 20.0, 30.0])

    return decompile_and_test(f, operand, indices, updates, name="Scatter Add Duplicate Indices")


def test_scatter_add_slice():
    """Test 9: Scatter add with slices."""
    def f(operand):
        return operand.at[1:3].add(10.0)

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    return decompile_and_test(f, operand, name="Scatter Add Slice")


def test_scatter_add_multidim():
    """Test 10: Scatter add with multi-dimensional updates."""
    def f(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(1, 2),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0, 1)
        )
        return lax.scatter_add(operand, indices, updates, dimension_numbers)

    operand = jnp.arange(24.0).reshape(4, 6)
    indices = jnp.array([[0, 0], [1, 2]])
    updates = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    return decompile_and_test(f, operand, indices, updates, name="Scatter Add Multidim")


# ============================================================================
# TEST 11-15: Scatter Min/Max Tests
# ============================================================================

def test_scatter_min_1d():
    """Test 11: Scatter min 1D."""
    def f(operand, indices, updates):
        return operand.at[indices].min(updates)

    operand = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    indices = jnp.array([1, 3])
    updates = jnp.array([5.0, 35.0])

    return decompile_and_test(f, operand, indices, updates, name="Scatter Min 1D")


def test_scatter_max_1d():
    """Test 12: Scatter max 1D."""
    def f(operand, indices, updates):
        return operand.at[indices].max(updates)

    operand = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    indices = jnp.array([1, 3])
    updates = jnp.array([25.0, 35.0])

    return decompile_and_test(f, operand, indices, updates, name="Scatter Max 1D")


def test_scatter_min_2d():
    """Test 13: Scatter min 2D."""
    def f(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)
        )
        return lax.scatter_min(operand, indices, updates, dimension_numbers)

    operand = jnp.arange(20.0).reshape(4, 5) + 10.0
    indices = jnp.array([[0], [2]])
    updates = jnp.array([[5.0, 6.0, 7.0, 8.0, 9.0], [1.0, 2.0, 3.0, 4.0, 5.0]])

    return decompile_and_test(f, operand, indices, updates, name="Scatter Min 2D")


def test_scatter_max_2d():
    """Test 14: Scatter max 2D."""
    def f(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)
        )
        return lax.scatter_max(operand, indices, updates, dimension_numbers)

    operand = jnp.arange(20.0).reshape(4, 5)
    indices = jnp.array([[1], [3]])
    updates = jnp.array([[20.0, 21.0, 22.0, 23.0, 24.0], [25.0, 26.0, 27.0, 28.0, 29.0]])

    return decompile_and_test(f, operand, indices, updates, name="Scatter Max 2D")


def test_scatter_mul_1d():
    """Test 15: Scatter multiply 1D."""
    def f(operand, indices, updates):
        return operand.at[indices].multiply(updates)

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = jnp.array([1, 3])
    updates = jnp.array([10.0, 20.0])

    return decompile_and_test(f, operand, indices, updates, name="Scatter Multiply 1D")


# ============================================================================
# TEST 16-20: Complex Scatter Patterns
# ============================================================================

def test_scatter_negative_indices():
    """Test 16: Scatter with negative indices."""
    def f(operand, indices, updates):
        return operand.at[indices].set(updates)

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = jnp.array([-1, -2])  # -1 -> 4, -2 -> 3
    updates = jnp.array([10.0, 20.0])

    return decompile_and_test(f, operand, indices, updates, name="Scatter Negative Indices")


def test_scatter_multiple_dims():
    """Test 17: Scatter with multiple dimension mapping."""
    def f(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1)
        )
        return lax.scatter(operand, indices, updates, dimension_numbers)

    operand = jnp.arange(30.0).reshape(5, 6)
    indices = jnp.array([[0, 1], [2, 3], [4, 5]])
    updates = jnp.array([100.0, 200.0, 300.0])

    return decompile_and_test(f, operand, indices, updates, name="Scatter Multiple Dims")


def test_scatter_strided():
    """Test 18: Scatter with strided slicing."""
    def f(operand):
        return operand.at[::2].set(100.0)

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    return decompile_and_test(f, operand, name="Scatter Strided")


def test_scatter_2d_indices():
    """Test 19: Scatter with 2D index arrays."""
    def f(operand, row_indices, col_indices, updates):
        return operand.at[row_indices, col_indices].set(updates)

    operand = jnp.arange(20.0).reshape(4, 5)
    row_indices = jnp.array([0, 2, 1])
    col_indices = jnp.array([1, 3, 4])
    updates = jnp.array([100.0, 200.0, 300.0])

    return decompile_and_test(f, operand, row_indices, col_indices, updates,
                            name="Scatter 2D Indices")


def test_scatter_chain():
    """Test 20: Chained scatter operations."""
    def f(operand):
        x = operand.at[0].add(10.0)
        x = x.at[1].multiply(2.0)
        x = x.at[2].set(50.0)
        return x

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    return decompile_and_test(f, operand, name="Scatter Chain")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all scatter tests."""
    tests = [
        # Basic scatter tests
        test_scatter_1d_basic,
        test_scatter_1d_window,
        test_scatter_2d_mixed,
        test_scatter_2d_full_window,
        test_scatter_3d,
        # Scatter add tests
        test_scatter_add_1d,
        test_scatter_add_2d,
        test_scatter_add_duplicate_indices,
        test_scatter_add_slice,
        test_scatter_add_multidim,
        # Scatter min/max/mul tests
        test_scatter_min_1d,
        test_scatter_max_1d,
        test_scatter_min_2d,
        test_scatter_max_2d,
        test_scatter_mul_1d,
        # Complex patterns
        test_scatter_negative_indices,
        test_scatter_multiple_dims,
        test_scatter_strided,
        test_scatter_2d_indices,
        test_scatter_chain,
    ]

    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE SCATTER TEST SUITE (20 Tests)")
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
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*80)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
