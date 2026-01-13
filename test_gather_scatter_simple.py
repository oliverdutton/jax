#!/usr/bin/env python3
"""
Test simple gather and scatter operations that should now work.
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr


def decompile_and_test(func, *args, name="test"):
    """Decompile and test a function."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

    try:
        expected = func(*args)
        print(f"Expected result: {expected}")

        lowered = jax.jit(func).lower(*args)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("ERROR: Could not find main function")
            return False

        result = main_func.callable_fn(*args)
        print(f"Decompiled result: {result}")

        match = np.allclose(result, expected)
        print(f"✓ PASS" if match else "✗ FAIL")
        return match

    except Exception as e:
        print(f"✗ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# GATHER TESTS
# ============================================================================

def test_gather_1d():
    """Test simple 1D gather."""
    def f(operand, indices):
        dimension_numbers = lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(0,),
            start_index_map=(0,)
        )
        return lax.gather(operand, indices, dimension_numbers, slice_sizes=(1,))

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = jnp.array([[0], [2], [4]])

    return decompile_and_test(f, operand, indices, name="Gather: Simple 1D")


def test_gather_2d_no_collapse():
    """Test 2D gather without collapsing."""
    def f(operand, indices):
        dimension_numbers = lax.GatherDimensionNumbers(
            offset_dims=(1, 2),
            collapsed_slice_dims=(),
            start_index_map=(0, 1)
        )
        return lax.gather(operand, indices, dimension_numbers, slice_sizes=(1, 1))

    operand = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    indices = jnp.array([[[0, 0]], [[1, 1]]])

    return decompile_and_test(f, operand, indices, name="Gather: 2D no collapse")


def test_slice_with_gather():
    """Test array slicing that uses gather."""
    def f(operand):
        return operand[::2]  # Uses gather internally

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    return decompile_and_test(f, operand, name="Gather: Array slicing [::2]")


# ============================================================================
# SCATTER TESTS
# ============================================================================

def test_scatter_1d():
    """Test simple 1D scatter."""
    def f(operand, indices, updates):
        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)
        )
        return lax.scatter(operand, indices, updates, dimension_numbers)

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = jnp.array([[1], [3]])
    updates = jnp.array([10.0, 20.0])

    return decompile_and_test(f, operand, indices, updates, name="Scatter: Simple 1D")


def test_scatter_add():
    """Test scatter with addition."""
    def f(operand, indices, updates):
        return operand.at[indices].add(updates)

    operand = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = jnp.array([1, 3])
    updates = jnp.array([10.0, 20.0])

    return decompile_and_test(f, operand, indices, updates, name="Scatter: Array .at[] add")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all gather/scatter tests."""
    tests = [
        test_gather_1d,
        test_gather_2d_no_collapse,
        test_slice_with_gather,
        test_scatter_1d,
        test_scatter_add,
    ]

    print("\n" + "="*80)
    print("RUNNING GATHER/SCATTER TEST SUITE")
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
