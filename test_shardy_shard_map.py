#!/usr/bin/env python3
"""
Test decompiling sharded code using shard_map.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import numpy as np
from functools import partial
from hlo_to_jaxpr import StableHLOToJaxpr


def test_shard_map_psum():
    """Test decompiling shard_map with psum."""
    print("\n" + "="*80)
    print("TEST: Decompile shard_map with psum")
    print("="*80)

    # Create mesh
    devices = mesh_utils.create_device_mesh((8,))
    mesh = Mesh(devices, axis_names=('i',))

    # Define function that uses psum inside shard_map
    @partial(shard_map, mesh=mesh, in_specs=P('i', None), out_specs=P('i', None))
    def f(x):
        # Inside shard_map, we can use collective operations
        return lax.psum(x, axis_name='i')

    # Input data
    x = jnp.ones((8, 4))

    # Get expected result
    expected = jax.jit(f)(x)
    print(f"Expected shape: {expected.shape}, value sample: {expected[0]}")
    print(f"Expected values: {expected}")

    # Get MLIR
    lowered = jax.jit(f).lower(x)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    print("\nMLIR (key parts):")
    mlir_str = str(mlir_module)
    for line in mlir_str.split('\n')[:50]:
        print(line)

    # Decompile
    try:
        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("❌ Could not find main function")
            return False

        # The decompiled function contains collectives
        # We need to wrap it in shard_map
        decompiled_fn = main_func.callable_fn

        if main_func.uses_collectives and main_func.mesh_info:
            print(f"\nFunction uses collectives with mesh axes: {main_func.mesh_info['axis_names']}")
            # Wrap in shard_map
            @partial(shard_map, mesh=mesh, in_specs=P('i', None), out_specs=P('i', None))
            def decompiled_sharded(x):
                return decompiled_fn(x)

            result = jax.jit(decompiled_sharded)(x)
        else:
            # JIT and run
            result = jax.jit(decompiled_fn)(x)
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


def test_shard_map_axis_index():
    """Test decompiling shard_map with axis_index."""
    print("\n" + "="*80)
    print("TEST: Decompile shard_map with axis_index")
    print("="*80)

    # Create mesh
    devices = mesh_utils.create_device_mesh((8,))
    mesh = Mesh(devices, axis_names=('i',))

    # Define function that uses axis_index inside shard_map
    @partial(shard_map, mesh=mesh, in_specs=P('i', None), out_specs=P('i', None))
    def f(x):
        idx = lax.axis_index(axis_name='i')
        return x + idx

    # Input data
    x = jnp.ones((8, 4))

    # Get expected result
    expected = jax.jit(f)(x)
    print(f"Expected shape: {expected.shape}")
    print(f"Expected first row: {expected[0]}, last row: {expected[7]}")

    # Decompile
    try:
        lowered = jax.jit(f).lower(x)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("❌ Could not find main function")
            return False

        decompiled_fn = main_func.callable_fn

        if main_func.uses_collectives and main_func.mesh_info:
            print(f"\nFunction uses collectives with mesh axes: {main_func.mesh_info['axis_names']}")
            # Wrap in shard_map
            @partial(shard_map, mesh=mesh, in_specs=P('i', None), out_specs=P('i', None))
            def decompiled_sharded(x):
                return decompiled_fn(x)

            result = jax.jit(decompiled_sharded)(x)
        else:
            result = jax.jit(decompiled_fn)(x)

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


def test_shard_map_all_gather():
    """Test decompiling shard_map with all_gather."""
    print("\n" + "="*80)
    print("TEST: Decompile shard_map with all_gather")
    print("="*80)

    # Create mesh
    devices = mesh_utils.create_device_mesh((8,))
    mesh = Mesh(devices, axis_names=('i',))

    # Define function that uses all_gather inside shard_map
    @partial(shard_map, mesh=mesh, in_specs=P('i', None), out_specs=P('i', None, None))
    def f(x):
        return lax.all_gather(x, axis_name='i')

    # Input data
    x = jnp.ones((8, 4))

    # Get expected result
    expected = jax.jit(f)(x)
    print(f"Expected shape: {expected.shape}")

    # Decompile
    try:
        lowered = jax.jit(f).lower(x)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("❌ Could not find main function")
            return False

        decompiled_fn = main_func.callable_fn

        if main_func.uses_collectives and main_func.mesh_info:
            print(f"\nFunction uses collectives with mesh axes: {main_func.mesh_info['axis_names']}")
            # Wrap in shard_map
            @partial(shard_map, mesh=mesh, in_specs=P('i', None), out_specs=P('i', None, None))
            def decompiled_sharded(x):
                return decompiled_fn(x)

            result = jax.jit(decompiled_sharded)(x)
        else:
            result = jax.jit(decompiled_fn)(x)

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
    print("SHARD_MAP DECOMPILATION TESTS")
    print("="*80)
    print(f"JAX Devices: {jax.devices()}")

    passed = []
    passed.append(test_shard_map_psum())
    passed.append(test_shard_map_axis_index())
    passed.append(test_shard_map_all_gather())

    print("\n" + "="*80)
    print(f"SUMMARY: {sum(passed)}/{len(passed)} tests passed")
    print("="*80)

    exit(0 if all(passed) else 1)
