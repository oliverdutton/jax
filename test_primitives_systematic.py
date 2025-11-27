"""Systematically test each primitive to find HLO vs NumPy differences."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
import pallas_numpy_interpreter

def test_primitive_comparison():
    """Compare primitive operations between HLO and NumPy interpreters."""

    print("="*80)
    print("SYSTEMATIC PRIMITIVE TESTING")
    print("="*80)

    # Test 1: IOTA operation
    print("\n" + "="*80)
    print("TEST 1: IOTA")
    print("="*80)

    def iota_kernel(x_ref, y_ref):
        # Create iota along dimension 0
        idx = jnp.arange(8, dtype=jnp.int32)[:, None]
        y_ref[:, :] = jnp.broadcast_to(idx, (8, 128))

    test_iota = pl.pallas_call(
        iota_kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
        grid=(1,)
    )

    dummy_input = np.zeros((8, 128), dtype=np.int32)

    # Run with HLO
    result_hlo = jax.jit(test_iota, backend='cpu')(dummy_input)
    result_hlo = np.array(result_hlo)

    # Run with NumPy
    pallas_numpy_interpreter.install_numpy_interpreter()
    result_numpy = jax.jit(test_iota, backend='cpu')(dummy_input)
    result_numpy = np.array(result_numpy)

    print(f"HLO result[:,0]: {result_hlo[:,0]}")
    print(f"NumPy result[:,0]: {result_numpy[:,0]}")
    print(f"Match: {np.allclose(result_hlo, result_numpy)}")

    if not np.allclose(result_hlo, result_numpy):
        print("❌ IOTA differs!")
        diff_mask = result_hlo != result_numpy
        print(f"  Differences at {np.sum(diff_mask)} positions")
        print(f"  First difference at {np.argwhere(diff_mask)[0] if np.any(diff_mask) else 'none'}")
    else:
        print("✅ IOTA matches")

    # Test 2: AND operation (bitwise)
    print("\n" + "="*80)
    print("TEST 2: BITWISE AND")
    print("="*80)

    def and_kernel(x_ref, y_ref):
        # Create array [0, 1, 2, ..., 127] for each row
        idx = jnp.arange(128, dtype=jnp.int32)[None, :]
        vals = jnp.broadcast_to(idx, (8, 128))
        # AND with 1 to get only LSB
        y_ref[:, :] = vals & 1

    test_and = pl.pallas_call(
        and_kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
        grid=(1,)
    )

    # Run with HLO
    result_hlo = test_and(dummy_input, interpret=True)
    result_hlo = np.array(result_hlo)

    # Run with NumPy
    result_numpy = test_and(dummy_input, interpret=True)
    result_numpy = np.array(result_numpy)

    print(f"HLO result[0,:10]: {result_hlo[0,:10]}")
    print(f"NumPy result[0,:10]: {result_numpy[0,:10]}")
    print(f"HLO unique values: {np.unique(result_hlo)}")
    print(f"NumPy unique values: {np.unique(result_numpy)}")
    print(f"Match: {np.allclose(result_hlo, result_numpy)}")

    if not np.allclose(result_hlo, result_numpy):
        print("❌ AND differs!")
    else:
        print("✅ AND matches")

    # Test 3: XOR operation
    print("\n" + "="*80)
    print("TEST 3: BITWISE XOR")
    print("="*80)

    def xor_kernel(x_ref, y_ref):
        # Create two arrays and XOR them
        a = jnp.arange(128, dtype=jnp.int32)[None, :] & 1
        b = jnp.zeros((8, 128), dtype=jnp.int32)
        y_ref[:, :] = a ^ b

    test_xor = pl.pallas_call(
        xor_kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
        grid=(1,)
    )

    # Run with HLO
    result_hlo = test_xor(dummy_input, interpret=True)
    result_hlo = np.array(result_hlo)

    # Run with NumPy
    result_numpy = test_xor(dummy_input, interpret=True)
    result_numpy = np.array(result_numpy)

    print(f"HLO result[0,:10]: {result_hlo[0,:10]}")
    print(f"NumPy result[0,:10]: {result_numpy[0,:10]}")
    print(f"Match: {np.allclose(result_hlo, result_numpy)}")

    if not np.allclose(result_hlo, result_numpy):
        print("❌ XOR differs!")
    else:
        print("✅ XOR matches")

    # Test 4: SELECT_N operation
    print("\n" + "="*80)
    print("TEST 4: SELECT_N")
    print("="*80)

    def select_n_kernel(x_ref, y_ref):
        # Test select_n with simple case
        which = jnp.zeros((8, 128), dtype=jnp.int32)  # All zeros, select first case
        case0 = jnp.zeros((8, 128), dtype=jnp.int32)
        case1 = jnp.ones((8, 128), dtype=jnp.int32)
        # select_n chooses from cases based on which
        result = jnp.where(which == 0, case0, case1)
        y_ref[:, :] = result

    test_select_n = pl.pallas_call(
        select_n_kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
        grid=(1,)
    )

    # Run with HLO
    result_hlo = test_select_n(dummy_input, interpret=True)
    result_hlo = np.array(result_hlo)

    # Run with NumPy
    result_numpy = test_select_n(dummy_input, interpret=True)
    result_numpy = np.array(result_numpy)

    print(f"HLO result[0,:10]: {result_hlo[0,:10]}")
    print(f"NumPy result[0,:10]: {result_numpy[0,:10]}")
    print(f"Match: {np.allclose(result_hlo, result_numpy)}")

    if not np.allclose(result_hlo, result_numpy):
        print("❌ SELECT_N differs!")
    else:
        print("✅ SELECT_N matches")

    # Test 5: GATHER operation (the critical one)
    print("\n" + "="*80)
    print("TEST 5: GATHER (Batched)")
    print("="*80)

    def gather_kernel(x_ref, y_ref):
        # Create operand with values at different rows
        operand = jnp.zeros((8, 128), dtype=jnp.float32)
        operand = operand.at[0, 0].set(3.0)
        operand = operand.at[1, 0].set(1.0)
        operand = operand.at[2, 0].set(2.0)

        # Create indices [1, 0, 0, 0, 0, 0, 0, 0] for column 0
        indices = jnp.zeros((8, 128), dtype=jnp.int32)
        indices = indices.at[0, :].set(1)  # Row 0 gets index 1
        indices = indices.at[1:, :].set(0)  # Rows 1-7 get index 0

        # Gather: result[i,j] = operand[indices[i,j], j]
        result = operand[indices, jnp.arange(128)[None, :]]
        y_ref[:, :] = result

    test_gather = pl.pallas_call(
        gather_kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        grid=(1,)
    )

    dummy_input_f32 = np.zeros((8, 128), dtype=np.float32)

    # Run with HLO
    result_hlo = test_gather(dummy_input_f32, interpret=True)
    result_hlo = np.array(result_hlo)

    # Run with NumPy
    result_numpy = test_gather(dummy_input_f32, interpret=True)
    result_numpy = np.array(result_numpy)

    print(f"HLO result[:,0]: {result_hlo[:,0]}")
    print(f"NumPy result[:,0]: {result_numpy[:,0]}")
    print(f"HLO unique values: {np.unique(result_hlo[~np.isnan(result_hlo)])}")
    print(f"NumPy unique values: {np.unique(result_numpy[~np.isnan(result_numpy)])}")
    print(f"Match: {np.allclose(result_hlo, result_numpy, equal_nan=True)}")

    if not np.allclose(result_hlo, result_numpy, equal_nan=True):
        print("❌ GATHER differs!")
    else:
        print("✅ GATHER matches")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Review the results above to identify which primitive differs.")

if __name__ == "__main__":
    test_primitive_comparison()
