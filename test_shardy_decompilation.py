#!/usr/bin/env python3
"""
Test decompiling sharded code using modern shardy approach.
Uses jax.jit with in_shardings/out_shardings.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import numpy as np
from functools import partial
from hlo_to_jaxpr import StableHLOToJaxpr


def test_sharded_psum():
    """Test decompiling sharded psum operation."""
    print("\n" + "="*80)
    print("TEST: Decompile Sharded psum")
    print("="*80)

    # Create mesh
    devices = mesh_utils.create_device_mesh((8,))
    mesh = Mesh(devices, axis_names=('i',))

    # Define function that uses psum
    def f(x):
        # This will be sharded across the 'i' axis
        # psum will sum across all shards
        return lax.psum(x, axis_name='i')

    # Input data
    x = jnp.ones((8, 4))

    # Shard across first dimension
    sharding = NamedSharding(mesh, P('i', None))

    # JIT with sharding
    @partial(jax.jit, in_shardings=(sharding,), out_shardings=sharding)
    def f_sharded(x):
        return f(x)

    # Get expected result
    expected = f_sharded(x)
    print(f"Expected shape: {expected.shape}, value sample: {expected[0]}")
    print(f"Expected values: {expected}")

    # Get MLIR
    lowered = f_sharded.lower(x)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    print("\nMLIR (key parts):")
    mlir_str = str(mlir_module)
    for line in mlir_str.split('\n'):
        if any(op in line for op in ['all_reduce', 'manual_computation', 'psum']):
            print(line)

    # Decompile
    try:
        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("❌ Could not find main function")
            return False

        # The decompiled function contains collective operations
        # We need to wrap it in a jit with the same sharding
        decompiled_fn = main_func.callable_fn

        # JIT the decompiled function with sharding
        @partial(jax.jit, in_shardings=(sharding,), out_shardings=sharding)
        def decompiled_sharded(x):
            return decompiled_fn(x)

        # Call it
        result = decompiled_sharded(x)
        print(f"\nDecompiled shape: {result.shape}, value sample: {result[0]}")
        print(f"Decompiled values: {result}")

        match = np.allclose(result, expected, atol=1e-5, rtol=1e-5)
        print(f"{'✅ PASS' if match else '❌ FAIL'}")

        if not match:
            print(f"Difference: {result - expected}")

        return match

    except Exception as e:
        print(f"❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sharded_axis_index():
    """Test decompiling sharded axis_index operation."""
    print("\n" + "="*80)
    print("TEST: Decompile Sharded axis_index")
    print("="*80)

    # Create mesh
    devices = mesh_utils.create_device_mesh((8,))
    mesh = Mesh(devices, axis_names=('i',))

    # Define function that uses axis_index
    def f(x):
        idx = lax.axis_index(axis_name='i')
        return x + idx

    # Input data
    x = jnp.ones((8, 4))

    # Shard across first dimension
    sharding = NamedSharding(mesh, P('i', None))

    # JIT with sharding
    @partial(jax.jit, in_shardings=(sharding,), out_shardings=sharding)
    def f_sharded(x):
        return f(x)

    # Get expected result
    expected = f_sharded(x)
    print(f"Expected shape: {expected.shape}")
    print(f"Expected first row: {expected[0]}, last row: {expected[7]}")

    # Decompile
    try:
        lowered = f_sharded.lower(x)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("❌ Could not find main function")
            return False

        decompiled_fn = main_func.callable_fn

        # JIT the decompiled function with sharding
        @partial(jax.jit, in_shardings=(sharding,), out_shardings=sharding)
        def decompiled_sharded(x):
            return decompiled_fn(x)

        result = decompiled_sharded(x)
        print(f"\nDecompiled shape: {result.shape}")
        print(f"Decompiled first row: {result[0]}, last row: {result[7]}")

        match = np.allclose(result, expected, atol=1e-5, rtol=1e-5)
        print(f"{'✅ PASS' if match else '❌ FAIL'}")

        return match

    except Exception as e:
        print(f"❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sharded_all_gather():
    """Test decompiling sharded all_gather operation."""
    print("\n" + "="*80)
    print("TEST: Decompile Sharded all_gather")
    print("="*80)

    # Create mesh
    devices = mesh_utils.create_device_mesh((8,))
    mesh = Mesh(devices, axis_names=('i',))

    # Define function that uses all_gather
    def f(x):
        return lax.all_gather(x, axis_name='i')

    # Input data
    x = jnp.ones((8, 4))

    # Shard input across first dimension
    in_sharding = NamedSharding(mesh, P('i', None))
    # Output is gathered, so has extra dimension
    out_sharding = NamedSharding(mesh, P('i', None, None))

    # JIT with sharding
    @partial(jax.jit, in_shardings=(in_sharding,), out_shardings=out_sharding)
    def f_sharded(x):
        return f(x)

    # Get expected result
    expected = f_sharded(x)
    print(f"Expected shape: {expected.shape}")

    # Decompile
    try:
        lowered = f_sharded.lower(x)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("❌ Could not find main function")
            return False

        decompiled_fn = main_func.callable_fn

        # JIT the decompiled function with sharding
        @partial(jax.jit, in_shardings=(in_sharding,), out_shardings=out_sharding)
        def decompiled_sharded(x):
            return decompiled_fn(x)

        result = decompiled_sharded(x)
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
    print("SHARDY DECOMPILATION TESTS (Modern Sharding API)")
    print("="*80)
    print(f"JAX Devices: {jax.devices()}")

    passed = []
    passed.append(test_sharded_psum())
    passed.append(test_sharded_axis_index())
    passed.append(test_sharded_all_gather())

    print("\n" + "="*80)
    print(f"SUMMARY: {sum(passed)}/{len(passed)} tests passed")
    print("="*80)

    exit(0 if all(passed) else 1)
