#!/usr/bin/env python3
"""
Test decompiling collective operations.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr


def test_psum_decompilation():
    """Test decompiling psum operation."""
    print("\n" + "="*80)
    print("TEST: Decompile psum")
    print("="*80)

    def f_psum(x):
        return lax.psum(x, axis_name='i')

    x = jnp.ones((8, 4))

    # Expected result with pmap
    f_pmapped = jax.pmap(f_psum, axis_name='i')
    expected = f_pmapped(x)
    print(f"Expected shape: {expected.shape}, value sample: {expected[0]}")

    # Decompile
    try:
        lowered = f_pmapped.lower(x)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        print("\nMLIR (first 100 lines):")
        mlir_lines = str(mlir_module).split('\n')
        for line in mlir_lines[:100]:
            print(line)

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("❌ Could not find main function")
            return False

        # Call the decompiled function
        result = main_func.callable_fn(x)
        print(f"\nDecompiled shape: {result.shape}, value sample: {result[0]}")

        match = np.allclose(result, expected, atol=1e-5, rtol=1e-5)
        print(f"{'✅ PASS' if match else '❌ FAIL'}")
        return match

    except Exception as e:
        print(f"❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_axis_index_decompilation():
    """Test decompiling axis_index operation."""
    print("\n" + "="*80)
    print("TEST: Decompile axis_index")
    print("="*80)

    def f_axis_index(x):
        idx = lax.axis_index(axis_name='i')
        return x + idx

    x = jnp.ones((8, 4))

    # Expected result with pmap
    f_pmapped = jax.pmap(f_axis_index, axis_name='i')
    expected = f_pmapped(x)
    print(f"Expected shape: {expected.shape}, value sample: {expected[0]}, {expected[7]}")

    # Decompile
    try:
        lowered = f_pmapped.lower(x)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("❌ Could not find main function")
            return False

        # Call the decompiled function
        result = main_func.callable_fn(x)
        print(f"\nDecompiled shape: {result.shape}, value sample: {result[0]}, {result[7]}")

        match = np.allclose(result, expected, atol=1e-5, rtol=1e-5)
        print(f"{'✅ PASS' if match else '❌ FAIL'}")
        return match

    except Exception as e:
        print(f"❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_gather_decompilation():
    """Test decompiling all_gather operation."""
    print("\n" + "="*80)
    print("TEST: Decompile all_gather")
    print("="*80)

    def f_all_gather(x):
        return lax.all_gather(x, axis_name='i')

    x = jnp.ones((8, 4))

    # Expected result with pmap
    f_pmapped = jax.pmap(f_all_gather, axis_name='i')
    expected = f_pmapped(x)
    print(f"Expected shape: {expected.shape}")

    # Decompile
    try:
        lowered = f_pmapped.lower(x)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("❌ Could not find main function")
            return False

        # Call the decompiled function
        result = main_func.callable_fn(x)
        print(f"\nDecompiled shape: {result.shape}")

        match = np.allclose(result, expected, atol=1e-5, rtol=1e-5)
        print(f"{'✅ PASS' if match else '❌ FAIL'}")
        return match

    except Exception as e:
        print(f"❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*80)
    print("COLLECTIVE OPERATIONS DECOMPILATION TESTS")
    print("="*80)

    passed = []
    passed.append(test_psum_decompilation())
    passed.append(test_axis_index_decompilation())
    passed.append(test_all_gather_decompilation())

    print("\n" + "="*80)
    print(f"SUMMARY: {sum(passed)}/{len(passed)} tests passed")
    print("="*80)

    exit(0 if all(passed) else 1)
