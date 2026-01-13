#!/usr/bin/env python3
"""
Explore what collective operations are generated in sharded code.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import numpy as np

print(f"JAX Devices: {jax.devices()}")

# Test 1: pmap with psum
print("\n" + "="*80)
print("TEST 1: pmap with lax.psum")
print("="*80)

def f_psum(x):
    return lax.psum(x, axis_name='i')

x = jnp.ones((8, 4))
f_pmapped = jax.pmap(f_psum, axis_name='i')

try:
    lowered = f_pmapped.lower(x)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')
    print(mlir_module)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()


# Test 2: pmap with all_gather
print("\n" + "="*80)
print("TEST 2: pmap with lax.all_gather")
print("="*80)

def f_all_gather(x):
    return lax.all_gather(x, axis_name='i')

x = jnp.ones((8, 4))
f_pmapped = jax.pmap(f_all_gather, axis_name='i')

try:
    lowered = f_pmapped.lower(x)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')
    print(mlir_module)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()


# Test 3: pmap with pmean
print("\n" + "="*80)
print("TEST 3: pmap with lax.pmean")
print("="*80)

def f_pmean(x):
    return lax.pmean(x, axis_name='i')

x = jnp.ones((8, 4))
f_pmapped = jax.pmap(f_pmean, axis_name='i')

try:
    lowered = f_pmapped.lower(x)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')
    print(mlir_module)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()


# Test 4: pmap with pmax
print("\n" + "="*80)
print("TEST 4: pmap with lax.pmax")
print("="*80)

def f_pmax(x):
    return lax.pmax(x, axis_name='i')

x = jnp.ones((8, 4))
f_pmapped = jax.pmap(f_pmax, axis_name='i')

try:
    lowered = f_pmapped.lower(x)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')
    print(mlir_module)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()


# Test 5: pmap with axis_index
print("\n" + "="*80)
print("TEST 5: pmap with lax.axis_index")
print("="*80)

def f_axis_index(x):
    idx = lax.axis_index(axis_name='i')
    return x + idx

x = jnp.ones((8, 4))
f_pmapped = jax.pmap(f_axis_index, axis_name='i')

try:
    lowered = f_pmapped.lower(x)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')
    print(mlir_module)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
