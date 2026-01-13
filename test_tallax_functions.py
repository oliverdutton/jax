#!/usr/bin/env python3
"""
Test tallax functions with the decompiler.

Tests take_along_axis_arrays and bitonic_topk_arrays.
"""

import sys
sys.path.insert(0, '/home/user/tallax')

import jax
import jax.numpy as jnp
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr

# Import tallax functions
try:
    from tallax.tax.gather import take_along_axis_arrays
    from tallax.tax.bitonic.topk import bitonic_topk_arrays
    TALLAX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import tallax: {e}")
    TALLAX_AVAILABLE = False


def decompile_and_test(func, *args, name="test", atol=1e-5, rtol=1e-5):
    """Decompile and test a function."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

    try:
        # Get expected result
        expected = func(*args)
        print(f"Expected result shape: {jnp.shape(expected)}")
        if isinstance(expected, list):
            print(f"Expected result (list of {len(expected)} arrays)")
            for i, e in enumerate(expected[:2]):  # Show first 2
                print(f"  [{i}] shape: {e.shape}, first elements: {e.flatten()[:5]}")
        else:
            print(f"Expected result: {expected.flatten()[:10]}")

        # Lower to StableHLO and decompile
        lowered = jax.jit(func).lower(*args)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        # Show IR size
        ir_str = str(mlir_module)
        print(f"\nMLIR size: {len(ir_str)} chars, {ir_str.count('stablehlo')} stablehlo ops")

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        # Find main function
        main_func = functions.get('"main"')
        if main_func is None:
            print("ERROR: Could not find main function")
            return False

        # Execute decompiled function
        result = main_func.callable_fn(*args)
        print(f"Decompiled result shape: {jnp.shape(result)}")
        if isinstance(result, list):
            print(f"Decompiled result (list of {len(result)} arrays)")
            for i, r in enumerate(result[:2]):
                print(f"  [{i}] shape: {r.shape}, first elements: {r.flatten()[:5]}")
        else:
            print(f"Decompiled result: {result.flatten()[:10]}")

        # Compare results
        if isinstance(result, list) and isinstance(expected, list):
            if len(result) != len(expected):
                print(f"ERROR: Length mismatch: {len(result)} vs {len(expected)}")
                return False
            match = all(np.allclose(r, e, atol=atol, rtol=rtol) for r, e in zip(result, expected))
        elif isinstance(result, tuple) and isinstance(expected, tuple):
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
# TAKE_ALONG_AXIS_ARRAYS TESTS
# ============================================================================

def test_take_along_axis_simple():
    """Test simple take_along_axis_arrays."""
    def f(values, indices):
        return take_along_axis_arrays(values, indices, axis=1)

    # Small test case
    values = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    indices = jnp.array([[0, 2], [1, 3]], dtype=jnp.int32)

    return decompile_and_test(f, values, indices, name="tallax: take_along_axis_arrays (simple)")


def test_take_along_axis_larger():
    """Test larger take_along_axis_arrays."""
    def f(values, indices):
        return take_along_axis_arrays(values, indices, axis=1)

    # Larger test case (but still reasonable)
    values = jnp.arange(32).reshape(4, 8).astype(jnp.float32)
    indices = jnp.array([[0, 1, 2], [3, 4, 5], [1, 2, 3], [0, 7, 6]], dtype=jnp.int32)

    return decompile_and_test(f, values, indices, name="tallax: take_along_axis_arrays (larger)")


# ============================================================================
# BITONIC_TOPK_ARRAYS TESTS
# ============================================================================

def test_bitonic_topk_simple():
    """Test simple bitonic_topk_arrays."""
    def f(arr):
        # Return top-2 along axis 1
        return bitonic_topk_arrays([arr], k=2, axis=1)

    # Small test case
    arr = jnp.array([[3.0, 1.0, 4.0, 2.0], [8.0, 5.0, 7.0, 6.0]])

    return decompile_and_test(f, arr, name="tallax: bitonic_topk_arrays (simple)", rtol=1e-4)


def test_bitonic_topk_larger_k():
    """Test bitonic_topk with k=4."""
    def f(arr):
        return bitonic_topk_arrays([arr], k=4, axis=1)

    # Larger k
    arr = jnp.arange(16).reshape(2, 8).astype(jnp.float32)

    return decompile_and_test(f, arr, name="tallax: bitonic_topk_arrays (k=4)", rtol=1e-4)


def test_bitonic_topk_multiple_arrays():
    """Test bitonic_topk with multiple arrays (values + indices)."""
    def f(values, indices):
        return bitonic_topk_arrays([values, indices], k=2, num_keys=1, axis=1)

    # Two arrays: values and their indices
    values = jnp.array([[3.0, 1.0, 4.0, 2.0], [8.0, 5.0, 7.0, 6.0]])
    indices = jnp.arange(8).reshape(2, 4).astype(jnp.float32)

    return decompile_and_test(f, values, indices, name="tallax: bitonic_topk_arrays (multi-array)", rtol=1e-4)


# ============================================================================
# SIMPLIFIED VERSIONS (For debugging)
# ============================================================================

def test_simple_take_along_axis():
    """Test JAX built-in take_along_axis."""
    def f(values, indices):
        return jnp.take_along_axis(values, indices, axis=1)

    values = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    indices = jnp.array([[0, 2], [1, 3]], dtype=jnp.int32)

    return decompile_and_test(f, values, indices, name="JAX: take_along_axis (baseline)")


def test_simple_topk():
    """Test simplified top-k pattern."""
    def f(arr):
        # Simple top-k using sort
        sorted_arr = jnp.sort(arr, axis=1)
        return sorted_arr[:, -2:]  # Top 2

    arr = jnp.array([[3.0, 1.0, 4.0, 2.0], [8.0, 5.0, 7.0, 6.0]])

    return decompile_and_test(f, arr, name="JAX: simple top-k pattern")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tallax tests."""

    if not TALLAX_AVAILABLE:
        print("\n" + "="*80)
        print("TALLAX NOT AVAILABLE - SKIPPING TESTS")
        print("="*80)
        return True

    tests = [
        # Baseline JAX tests
        test_simple_take_along_axis,
        test_simple_topk,

        # Tallax tests
        test_take_along_axis_simple,
        test_take_along_axis_larger,
        test_bitonic_topk_simple,
        test_bitonic_topk_larger_k,
        test_bitonic_topk_multiple_arrays,
    ]

    print("\n" + "="*80)
    print("RUNNING TALLAX FUNCTION TESTS")
    print("="*80)
    print("\nTesting take_along_axis_arrays and bitonic_topk_arrays")
    print("These are complex TPU-optimized functions using Pallas.")

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nTest {test.__name__} raised exception: {e}")
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
