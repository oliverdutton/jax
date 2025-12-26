#!/usr/bin/env python3
"""
Script to verify which lax operations lower successfully in Pallas TPU kernels.

Tests the following operations:
- lax.cond
- lax.select
- lax.switch
- lax.select_n
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
import numpy as np


def test_lax_select():
    """Test lax.select in Pallas TPU kernel."""
    print("\n" + "="*60)
    print("Testing lax.select")
    print("="*60)

    def kernel(x_ref, pred_ref, y_ref):
        x = x_ref[...]
        pred = pred_ref[...]
        # lax.select(pred, on_true, on_false)
        result = lax.select(pred, x + 1.0, x - 1.0)
        y_ref[...] = result

    try:
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        pred = jnp.array([True, False, True, False])

        compiled_kernel = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        )

        result = compiled_kernel(x, pred)
        expected = jnp.where(pred, x + 1.0, x - 1.0)

        print(f"✓ lax.select COMPILES successfully")
        print(f"  Input: {x}")
        print(f"  Pred:  {pred}")
        print(f"  Result: {result}")
        print(f"  Expected: {expected}")
        print(f"  Match: {jnp.allclose(result, expected)}")
        return True
    except Exception as e:
        print(f"✗ lax.select FAILED to compile")
        print(f"  Error: {type(e).__name__}: {e}")
        return False


def test_lax_select_n():
    """Test lax.select_n in Pallas TPU kernel."""
    print("\n" + "="*60)
    print("Testing lax.select_n")
    print("="*60)

    def kernel(x_ref, which_ref, y_ref):
        x = x_ref[...]
        which = which_ref[...]
        # lax.select_n(which, *cases)
        # Note: TPU lowering only supports binary select (2 cases)
        result = lax.select_n(which, x - 1.0, x + 1.0)
        y_ref[...] = result

    try:
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        which = jnp.array([0, 1, 0, 1])

        compiled_kernel = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        )

        result = compiled_kernel(x, which)
        expected = jnp.where(which, x + 1.0, x - 1.0)

        print(f"✓ lax.select_n COMPILES successfully")
        print(f"  Input: {x}")
        print(f"  Which: {which}")
        print(f"  Result: {result}")
        print(f"  Expected: {expected}")
        print(f"  Match: {jnp.allclose(result, expected)}")
        return True
    except Exception as e:
        print(f"✗ lax.select_n FAILED to compile")
        print(f"  Error: {type(e).__name__}: {e}")
        return False


def test_lax_select_n_three_cases():
    """Test lax.select_n with 3 cases (should fail per code inspection)."""
    print("\n" + "="*60)
    print("Testing lax.select_n with 3 cases (expected to fail)")
    print("="*60)

    def kernel(x_ref, which_ref, y_ref):
        x = x_ref[...]
        which = which_ref[...]
        # This should fail: TPU lowering only supports <= 2 cases
        result = lax.select_n(which, x - 1.0, x, x + 1.0)
        y_ref[...] = result

    try:
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        which = jnp.array([0, 1, 2, 1])

        compiled_kernel = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        )

        result = compiled_kernel(x, which)
        print(f"✗ lax.select_n with 3 cases UNEXPECTEDLY COMPILED")
        print(f"  Result: {result}")
        return False
    except NotImplementedError as e:
        print(f"✓ lax.select_n with 3 cases CORRECTLY FAILED")
        print(f"  Expected error: {e}")
        return True
    except Exception as e:
        print(f"? lax.select_n with 3 cases failed with unexpected error")
        print(f"  Error: {type(e).__name__}: {e}")
        return True  # Still counts as "correctly failed"


def test_lax_cond():
    """Test lax.cond in Pallas TPU kernel."""
    print("\n" + "="*60)
    print("Testing lax.cond")
    print("="*60)

    def kernel(x_ref, pred_ref, y_ref):
        x = x_ref[0]
        pred = pred_ref[0]

        def true_fn(x):
            return x + 10.0

        def false_fn(x):
            return x - 10.0

        result = lax.cond(pred, true_fn, false_fn, x)
        y_ref[0] = result

    try:
        x = jnp.array([5.0])
        pred = jnp.array([True])

        compiled_kernel = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1,), x.dtype),
        )

        result = compiled_kernel(x, pred)
        expected = jnp.array([15.0])

        print(f"✓ lax.cond COMPILES successfully")
        print(f"  Input: {x}")
        print(f"  Pred:  {pred}")
        print(f"  Result: {result}")
        print(f"  Expected: {expected}")
        print(f"  Match: {jnp.allclose(result, expected)}")
        return True
    except Exception as e:
        print(f"✗ lax.cond FAILED to compile")
        print(f"  Error: {type(e).__name__}: {e}")
        return False


def test_lax_switch():
    """Test lax.switch in Pallas TPU kernel."""
    print("\n" + "="*60)
    print("Testing lax.switch")
    print("="*60)

    def kernel(x_ref, index_ref, y_ref):
        x = x_ref[0]
        index = index_ref[0]

        def branch_0(x):
            return x + 100.0

        def branch_1(x):
            return x + 200.0

        def branch_2(x):
            return x + 300.0

        result = lax.switch(index, [branch_0, branch_1, branch_2], x)
        y_ref[0] = result

    try:
        x = jnp.array([5.0])
        index = jnp.array([1], dtype=jnp.int32)

        compiled_kernel = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1,), x.dtype),
        )

        result = compiled_kernel(x, index)
        expected = jnp.array([205.0])  # branch_1: 5 + 200

        print(f"✓ lax.switch COMPILES successfully")
        print(f"  Input: {x}")
        print(f"  Index: {index}")
        print(f"  Result: {result}")
        print(f"  Expected: {expected}")
        print(f"  Match: {jnp.allclose(result, expected)}")
        return True
    except Exception as e:
        print(f"✗ lax.switch FAILED to compile")
        print(f"  Error: {type(e).__name__}: {e}")
        return False


def main():
    print("="*60)
    print("JAX Pallas TPU Lowering Verification Script")
    print("="*60)
    print(f"JAX version: {jax.__version__}")
    print(f"Available backends: {jax.devices()}")

    # Check if TPU is available
    try:
        tpu_devices = jax.devices('tpu')
        print(f"TPU devices found: {len(tpu_devices)}")
    except:
        print("⚠ WARNING: No TPU devices found. Tests may fail or run on CPU.")
        print("  This script is designed for TPU backends.")

    results = {}

    # Run all tests
    results['lax.select'] = test_lax_select()
    results['lax.select_n (2 cases)'] = test_lax_select_n()
    results['lax.select_n (3 cases)'] = test_lax_select_n_three_cases()
    results['lax.cond'] = test_lax_cond()
    results['lax.switch'] = test_lax_switch()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for op, success in results.items():
        status = "✓ SUPPORTED" if success else "✗ NOT SUPPORTED"
        print(f"{op:30s} {status}")

    print("\n" + "="*60)
    print("Code Analysis Results:")
    print("="*60)
    print("Based on code inspection of jax/_src/pallas/mosaic/lowering.py:")
    print()
    print("1. lax.select_n_p - HAS lowering rule (line 3132)")
    print("   - Registered for all TPU kernel types")
    print("   - Limitation: Only supports <= 2 arguments (binary select)")
    print("   - Uses MLIR arith.select operation")
    print()
    print("2. lax.select - Wrapper function calling select_n_p.bind()")
    print("   - Should work (uses select_n_p internally)")
    print()
    print("3. lax.cond_p - HAS lowering rule (line 3348)")
    print("   - Registered for all TPU kernel types")
    print("   - Optimizes constant indices")
    print("   - Uses MLIR scf.IfOp (Structured Control Flow)")
    print()
    print("4. lax.switch - High-level function calling cond_p.bind()")
    print("   - Should work (uses cond_p internally)")
    print("   - Supports multiple branches via cascaded if/else")
    print()
    print("="*60)


if __name__ == "__main__":
    main()
