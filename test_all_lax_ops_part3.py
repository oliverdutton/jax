#!/usr/bin/env python3
"""
Comprehensive test suite for all lax operations - Part 3
Tests: Other commonly used operations from the "OTHER" category
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr


def test_operation(name, fn, inputs, test_num):
    """Test a single operation for decompilation."""
    try:
        # Get expected result
        expected = fn(*inputs)

        # Compile and decompile
        jitted_fn = jax.jit(fn)
        lowered = jitted_fn.lower(*inputs)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print(f"  {test_num}. {name}: ❌ FAIL - No main function")
            return False

        # Call decompiled function
        result = main_func.callable_fn(*inputs)

        # Compare results
        match = np.allclose(result, expected, atol=1e-5, rtol=1e-5, equal_nan=True)
        if match:
            print(f"  {test_num}. {name}: ✅ PASS")
        else:
            print(f"  {test_num}. {name}: ❌ FAIL - Result mismatch")
        return match

    except Exception as e:
        print(f"  {test_num}. {name}: ❌ FAIL - Exception: {str(e)[:100]}")
        return False


def test_other_operations():
    """Test other commonly used operations."""
    print("\n" + "="*80)
    print("OTHER OPERATIONS (commonly used)")
    print("="*80)

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.array([2.0, 3.0, 4.0, 5.0])
    x_complex = jnp.array([1.0+2.0j, 3.0+4.0j])

    tests = [
        ("lax.bessel_i0e", lambda x: lax.bessel_i0e(x), [x]),
        ("lax.bessel_i1e", lambda x: lax.bessel_i1e(x), [x]),
        ("lax.betainc", lambda a, b, x: lax.betainc(a, b, x), [x, y, jnp.array([0.1, 0.2, 0.3, 0.4])]),
        ("lax.bitcast_convert_type", lambda x: lax.bitcast_convert_type(x.astype(jnp.int32), jnp.float32), [x]),
        ("lax.cbrt", lambda x: lax.cbrt(x), [x]),
        ("lax.clz", lambda x: lax.clz(x.astype(jnp.int32)), [x]),
        ("lax.complex", lambda x, y: lax.complex(x, y), [x, y]),
        ("lax.conj", lambda x: lax.conj(x), [x_complex]),
        ("lax.convert_element_type", lambda x: lax.convert_element_type(x, jnp.int32), [x]),
        ("lax.digamma", lambda x: lax.digamma(x), [x]),
        ("lax.exp2", lambda x: lax.exp2(x), [jnp.array([1.0, 2.0, 3.0])]),
        ("lax.igamma", lambda a, x: lax.igamma(a, x), [x, y]),
        ("lax.igammac", lambda a, x: lax.igammac(a, x), [x, y]),
        ("lax.imag", lambda x: lax.imag(x), [x_complex]),
        ("lax.is_finite", lambda x: lax.is_finite(x), [x]),
        ("lax.lgamma", lambda x: lax.lgamma(x), [x]),
        ("lax.log10", lambda x: lax.log10(x), [x]),
        ("lax.log2", lambda x: lax.log2(x), [x]),
        ("lax.nextafter", lambda x, y: lax.nextafter(x, y), [x, y]),
        ("lax.polygamma", lambda m, x: lax.polygamma(2, x), [x]),
        ("lax.population_count", lambda x: lax.population_count(x.astype(jnp.int32)), [x]),
        ("lax.real", lambda x: lax.real(x), [x_complex]),
        ("lax.reciprocal", lambda x: lax.reciprocal(x), [x]),
        ("lax.reduce_and", lambda x: lax.reduce_and(x > 2, (0,)), [x]),
        ("lax.reduce_max", lambda x: lax.reduce_max(x, (0,)), [jnp.array([[1.0, 2.0], [3.0, 4.0]])]),
        ("lax.reduce_min", lambda x: lax.reduce_min(x, (0,)), [jnp.array([[1.0, 2.0], [3.0, 4.0]])]),
        ("lax.reduce_or", lambda x: lax.reduce_or(x > 2, (0,)), [x]),
        ("lax.reduce_prod", lambda x: lax.reduce_prod(x, (0,)), [jnp.array([[1.0, 2.0], [3.0, 4.0]])]),
        ("lax.reduce_sum", lambda x: lax.reduce_sum(x, (0,)), [jnp.array([[1.0, 2.0], [3.0, 4.0]])]),
        ("lax.reduce_xor", lambda x: lax.reduce_xor(x > 2, (0,)), [x]),
        ("lax.sort", lambda x: lax.sort(x), [jnp.array([3.0, 1.0, 4.0, 2.0])]),
        ("lax.square", lambda x: lax.square(x), [x]),
        ("lax.stop_gradient", lambda x: lax.stop_gradient(x), [x]),
        ("lax.top_k", lambda x: lax.top_k(x, 2)[0], [x]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE LAX OPERATIONS TEST - PART 3")
    print("Testing: Other commonly used operations")
    print("="*80)

    all_passed = []
    all_passed.extend(test_other_operations())

    print("\n" + "="*80)
    print(f"PART 3 SUMMARY: {sum(all_passed)}/{len(all_passed)} tests passed")
    print("="*80)

    exit(0 if all(all_passed) else 1)
