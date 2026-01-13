#!/usr/bin/env python3
"""
Comprehensive test suite for all lax operations - Part 2
Tests: Shape, Indexing, Reduction, Linear Algebra, FFT, Control Flow operations
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
            print(f"      Expected: {expected}")
            print(f"      Got: {result}")
        return match

    except Exception as e:
        print(f"  {test_num}. {name}: ❌ FAIL - Exception: {e}")
        return False


def test_shape_operations():
    """Test shape manipulation operations."""
    print("\n" + "="*80)
    print("SHAPE OPERATIONS (13 tests)")
    print("="*80)

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    tests = [
        ("lax.broadcast", lambda x: lax.broadcast(x[0], (3, 3)), [x]),
        ("lax.broadcast_in_dim", lambda x: lax.broadcast_in_dim(x[0], (3, 3), (1,)), [x]),
        ("lax.collapse", lambda x: lax.collapse(x, 0, 2), [x]),
        ("lax.concatenate", lambda x, y: lax.concatenate([x, y], 0), [x, x]),
        ("lax.dynamic_slice", lambda x: lax.dynamic_slice(x, (0, 1), (2, 2)), [x]),
        ("lax.dynamic_update_slice", lambda x, y: lax.dynamic_update_slice(x, y, (0, 1)), [x, jnp.array([[10.0, 11.0]])]),
        ("lax.expand_dims", lambda x: lax.expand_dims(x, (1,)), [x]),
        ("lax.pad", lambda x: lax.pad(x, jnp.float32(0), ((1, 1, 0), (1, 1, 0))), [x]),
        ("lax.reshape", lambda x: lax.reshape(x, (6,)), [x]),
        ("lax.rev", lambda x: lax.rev(x, (1,)), [x]),
        ("lax.slice", lambda x: lax.slice(x, (0, 1), (2, 3)), [x]),
        ("lax.squeeze", lambda x: lax.squeeze(jnp.expand_dims(x, 0), (0,)), [x]),
        ("lax.transpose", lambda x: lax.transpose(x, (1, 0)), [x]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


def test_indexing_operations():
    """Test indexing operations (gather, scatter, etc.)."""
    print("\n" + "="*80)
    print("INDEXING OPERATIONS (8 tests)")
    print("="*80)

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    indices = jnp.array([[0, 1], [1, 2]])
    updates = jnp.array([10.0, 20.0])

    # GatherDimensionNumbers for gather test
    gather_dnums = lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1)
    )

    # ScatterDimensionNumbers for scatter tests
    scatter_dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1)
    )

    tests = [
        ("lax.gather", lambda x, idx: lax.gather(x, idx, gather_dnums, (1, 1)), [x, jnp.array([[0, 1]])]),
        ("lax.index_in_dim", lambda x: lax.index_in_dim(x, 0, axis=0), [x]),
        ("lax.scatter", lambda x, idx, upd: lax.scatter(x, idx, upd, scatter_dnums), [x, indices, updates]),
        ("lax.scatter_add", lambda x, idx, upd: lax.scatter_add(x, idx, upd, scatter_dnums), [x, indices, updates]),
        ("lax.scatter_max", lambda x, idx, upd: lax.scatter_max(x, idx, upd, scatter_dnums), [x, indices, updates]),
        ("lax.scatter_min", lambda x, idx, upd: lax.scatter_min(x, idx, upd, scatter_dnums), [x, indices, updates]),
        ("lax.scatter_mul", lambda x, idx, upd: lax.scatter_mul(x, idx, upd, scatter_dnums), [x, indices, updates]),
        ("lax.slice_in_dim", lambda x: lax.slice_in_dim(x, 0, 2, axis=0), [x]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


def test_reduction_operations():
    """Test reduction operations."""
    print("\n" + "="*80)
    print("REDUCTION OPERATIONS (8 tests)")
    print("="*80)

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_int = jnp.array([[1, 5, 3], [4, 2, 6]], dtype=jnp.int32)

    tests = [
        ("lax.argmax", lambda x: lax.argmax(x, axis=1, index_dtype=jnp.int32), [x_int]),
        ("lax.argmin", lambda x: lax.argmin(x, axis=1, index_dtype=jnp.int32), [x_int]),
        ("lax.cummax", lambda x: lax.cummax(x, axis=1), [x]),
        ("lax.cummin", lambda x: lax.cummin(x, axis=1), [x]),
        ("lax.cumprod", lambda x: lax.cumprod(x, axis=1), [x]),
        ("lax.cumsum", lambda x: lax.cumsum(x, axis=1), [x]),
        ("lax.reduce", lambda x: lax.reduce(x, jnp.float32(0), lax.add, (1,)), [x]),
        ("lax.reduce_window", lambda x: lax.reduce_window(x, jnp.float32(0), lax.add, (1, 2), (1, 1), 'VALID'), [x]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


def test_linear_algebra_operations():
    """Test linear algebra operations."""
    print("\n" + "="*80)
    print("LINEAR ALGEBRA OPERATIONS (6 tests)")
    print("="*80)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[2.0, 0.0], [1.0, 2.0]])
    v = jnp.array([1.0, 2.0])

    # For batch_matmul
    batch_x = jnp.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    batch_y = jnp.array([[[2.0], [1.0]], [[1.0], [2.0]]])

    # For conv
    lhs = jnp.ones((1, 4, 4, 1))  # NHWC format
    rhs = jnp.ones((3, 3, 1, 1))  # HWIO format

    dn = lax.ConvDimensionNumbers(
        lhs_spec=(0, 3, 1, 2),  # NCHW
        rhs_spec=(3, 2, 0, 1),  # OIHW
        out_spec=(0, 3, 1, 2)   # NCHW
    )

    tests = [
        ("lax.batch_matmul", lambda x, y: lax.batch_matmul(x, y), [batch_x, batch_y]),
        ("lax.conv", lambda x, y: lax.conv(x, y, (1, 1), 'SAME'), [lhs, rhs]),
        ("lax.conv_general_dilated", lambda x, y: lax.conv_general_dilated(x, y, (1, 1), 'SAME', (1, 1), (1, 1), dn), [lhs, rhs]),
        ("lax.conv_transpose", lambda x, y: lax.conv_transpose(x, y, (1, 1), 'SAME'), [lhs, rhs]),
        ("lax.dot", lambda x, y: lax.dot(x, y), [v, v]),
        ("lax.dot_general", lambda x, y: lax.dot_general(x, y, (((1,), (0,)), ((), ()))), [x, y]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


def test_fft_operations():
    """Test FFT operations."""
    print("\n" + "="*80)
    print("FFT OPERATIONS (1 test)")
    print("="*80)

    x = jnp.array([[1.0+0j, 2.0+0j, 3.0+0j, 4.0+0j], [5.0+0j, 6.0+0j, 7.0+0j, 8.0+0j]])

    tests = [
        ("lax.fft", lambda x: lax.fft(x.astype(jnp.complex64), 'FFT', (1,)), [x]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


def test_control_flow_operations():
    """Test control flow operations."""
    print("\n" + "="*80)
    print("CONTROL FLOW OPERATIONS (7 tests)")
    print("="*80)

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])

    tests = [
        ("lax.cond", lambda pred, x: lax.cond(pred, lambda x: x + 1, lambda x: x - 1, x), [True, x]),
        ("lax.fori_loop", lambda x: lax.fori_loop(0, 3, lambda i, x: x + 1, x), [x]),
        ("lax.scan", lambda x: lax.scan(lambda c, x: (c + x, c + x), jnp.float32(0), x)[1], [x]),
        ("lax.select", lambda pred, x, y: lax.select(pred, x, y), [jnp.array([True, False, True]), x, y]),
        ("lax.select_n", lambda pred, x, y: lax.select_n(pred, x, y), [jnp.array([0, 1, 0], dtype=jnp.int32), x, y]),
        ("lax.switch", lambda idx, x: lax.switch(idx, [lambda x: x + 1, lambda x: x + 2], x), [0, x]),
        ("lax.while_loop", lambda init: lax.while_loop(lambda x: x[0] < 5, lambda x: jnp.array([x[0] + 1, x[1]]), init), [jnp.array([0.0, 1.0])]),
    ]

    passed = []
    for i, (name, fn, inputs) in enumerate(tests, 1):
        passed.append(test_operation(name, fn, inputs, i))

    return passed


if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE LAX OPERATIONS TEST - PART 2")
    print("Testing: Shape, Indexing, Reduction, Linear Algebra, FFT, Control Flow")
    print("="*80)

    all_passed = []

    all_passed.extend(test_shape_operations())
    all_passed.extend(test_indexing_operations())
    all_passed.extend(test_reduction_operations())
    all_passed.extend(test_linear_algebra_operations())
    all_passed.extend(test_fft_operations())
    all_passed.extend(test_control_flow_operations())

    print("\n" + "="*80)
    print(f"PART 2 SUMMARY: {sum(all_passed)}/{len(all_passed)} tests passed")
    print("="*80)

    exit(0 if all(all_passed) else 1)
