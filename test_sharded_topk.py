#!/usr/bin/env python3
"""
Test sharded top-k with decompilation.
This tests custom partitioning, shard_map, and collective operations.
"""

import os

# 1. Setup 8 CPU Simulation (Must run before importing JAX)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.custom_partitioning import custom_partitioning
import numpy as np
from functools import partial
from hlo_to_jaxpr import StableHLOToJaxpr

# --- Helper Functions & Constants (filling in missing parts) ---

# Usually -inf for float32 logits to ensure masked values aren't selected
REPLACE_VAL = -1e9

def _reduction_topk(operands, k):
    """
    Replacement for the missing _bitonic_topk_arrays.
    Performs a standard top_k on the gathered results.
    """
    all_logits, all_indices = operands
    # all_logits is [batch, num_shards * local_k]
    # all_indices is [batch, num_shards * local_k] (global indices)

    # Select top-k from the aggregated set
    top_vals, top_k_local_indices = jax.lax.top_k(all_logits, k=k)

    # Map back to the global indices using the local selection
    top_indices = jnp.take_along_axis(all_indices, top_k_local_indices, axis=1)

    return top_vals, top_indices

# --- Main Function Implementation ---

def top_bounded_k(
    logits: jax.Array,
    k: jax.Array, # passed as array for tracing flexibility
    max_k: int,
):
    """
    Computes top_k with custom sharding logic to handle distributed vocabularies.
    """

    # Helper for the local/closed top-k operation
    def _closed_topk(logits: jax.Array, k_curr: jax.Array):
        # We ensure k is concrete or bounded by max_k
        return jax.lax.top_k(logits, k=max_k)

    # 1. Define the partitioning logic
    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
        # Keeps batch sharding if present, forces replication on vocab dim
        logits_spec = arg_shapes[0].sharding.spec
        return (NamedSharding(mesh, P(logits_spec[0], None)),) * 2

    def partition(mesh, arg_shapes, out_shapes):
        # Fill in missing guarantee check (assuming True for this sim)
        guarantee_convergence = True
        if not guarantee_convergence:
            raise NotImplementedError

        arg_shardings, out_shardings = jax.tree.map(
            lambda s: s.sharding, (arg_shapes, out_shapes)
        )

        # Determine which axis the vocab is sharded on (axis 1)
        # arg_shardings[0].spec looks like P('data', 'model') or similar
        axis_name = arg_shardings[0].spec[1]

        def shmap_fn(logits, k_curr):
            # 1. Local TopK
            topk_logits, topk_idxs = _closed_topk(logits, k_curr)

            # If not sharded on vocab, we are done
            if axis_name is None:
                return topk_logits, topk_idxs

            # 2. Global Index Conversion
            # We are inside a shard. We need to offset the local indices
            # based on which shard we are to get global indices.
            i = jax.lax.axis_index(axis_name)

            # Total size of the vocab dimension for THIS shard
            shard_vocab_size = logits.shape[1]
            topk_idxs += i * shard_vocab_size

            # 3. All-Gather (collect results from all shards)
            operands = [
                jax.lax.collapse(
                    jax.lax.all_gather(x, axis_name, axis=1),
                    start_dimension=1 # flatten gathered chunks into one dim
                )
                for x in (topk_logits, topk_idxs)
            ]

            # 4. Reduction TopK (TopK of TopKs)
            # Replaces _bitonic_topk_arrays
            final_logits, final_idxs = _reduction_topk(operands, k=max_k)

            # 5. Masking (Optional based on dynamic k, but kept for fidelity to prompt)
            # Ensure we mask out values if dynamic k < max_k
            mask = jax.lax.broadcasted_iota(jnp.int32, final_logits.shape, 1) < k_curr[:, None]
            final_logits = jnp.where(
                mask,
                final_logits,
                REPLACE_VAL,
            )

            return final_logits, final_idxs

        return mesh, shmap_fn, out_shardings, arg_shardings

    # 2. Decorate the function
    @custom_partitioning
    def _sharded_topk(logits, k):
        return _closed_topk(logits, k)

    # 3. Register the partition implementation
    _sharded_topk.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule="b v, b -> b k, b k",
        #need_replication_factors=("k",),
    )

    return _sharded_topk(logits, k)


# --- Test with Decompilation ---

def test_decompile_simple_function():
    """Test decompiling a simple sharded function."""
    print("\n" + "="*80)
    print("TEST: Decompile Simple Sharded Function")
    print("="*80)

    def simple_sharded(x):
        # Simple operation that will work in single device
        return x * 2.0 + 1.0

    x = jnp.array([1.0, 2.0, 3.0])

    # Get expected result
    expected = simple_sharded(x)
    print(f"Expected: {expected}")

    # Try to decompile
    try:
        lowered = jax.jit(simple_sharded).lower(x)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("❌ Could not find main function")
            return False

        result = main_func.callable_fn(x)
        print(f"Decompiled: {result}")

        match = np.allclose(result, expected)
        print(f"{'✅ PASS' if match else '❌ FAIL'}")
        return match
    except Exception as e:
        print(f"❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_decompile_topk():
    """Test decompiling top_k operation."""
    print("\n" + "="*80)
    print("TEST: Decompile Top-K Operation")
    print("="*80)

    def simple_topk(x):
        return jax.lax.top_k(x, k=3)

    x = jnp.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0])

    # Get expected result
    expected_vals, expected_idxs = simple_topk(x)
    print(f"Expected values: {expected_vals}")
    print(f"Expected indices: {expected_idxs}")

    # Try to decompile
    try:
        lowered = jax.jit(simple_topk).lower(x)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions.get('"main"')
        if main_func is None:
            print("❌ Could not find main function")
            return False

        result = main_func.callable_fn(x)
        result_vals, result_idxs = result
        print(f"Decompiled values: {result_vals}")
        print(f"Decompiled indices: {result_idxs}")

        match_vals = np.allclose(result_vals, expected_vals)
        match_idxs = np.array_equal(result_idxs, expected_idxs)
        match = match_vals and match_idxs
        print(f"{'✅ PASS' if match else '❌ FAIL'}")
        return match
    except Exception as e:
        print(f"❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# --- Simulation and Test Cases ---

def run_simulation():
    print(f"JAX Devices: {jax.devices()}")
    assert len(jax.devices()) == 8, "Simulation requires 8 CPU devices."

    # Constants
    B, V = 16, 16384
    K = 64
    MAX_K = 64

    # Data Generation
    key = jax.random.PRNGKey(42)
    input_logits = jax.random.normal(key, (B, V))
    # We pass k as an array to simulate dynamic k (common in sampling)
    k_array = jnp.array([K] * B, dtype=jnp.int32)

    # Reference Result (Single Device)
    print("\n--- Computing Reference (Single Device) ---")
    ref_vals, ref_idxs = jax.lax.top_k(input_logits, K)
    print("Reference computed.")

    def verify(result_vals, result_idxs, name):
        # We sort results because parallel reductions might slightly alter order
        # for identical values, though unlikely with floats.
        # Strict equality check:
        try:
            np.testing.assert_allclose(result_vals, ref_vals, rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(result_idxs, ref_idxs)
            print(f"✅ {name} Passed")
            return True
        except AssertionError as e:
            print(f"❌ {name} Failed")
            print(e)
            return False

    # --- Test Case 1: Batch Sharding ---
    # Shard inputs along Batch (axis 0), replicate Vocab (axis 1)
    # Expected behavior: No communication needed (axis_name will be None)
    print("\n--- Test Case 1: Batch Sharding (Data Parallel) ---")
    mesh_1d = Mesh(jax.devices(), axis_names=('data'))
    sharding_batch = NamedSharding(mesh_1d, P('data', None))

    # Jit with input constraints
    @partial(jax.jit, in_shardings=(sharding_batch, None))
    def test_batch(x, k):
        return top_bounded_k(x, k, MAX_K)

    passed = []
    try:
        out_vals, out_idxs = test_batch(input_logits, k_array)
        passed.append(verify(out_vals, out_idxs, "Batch Sharding"))
    except Exception as e:
        print(f"❌ Batch Sharding Failed with exception: {e}")
        passed.append(False)

    # --- Test Case 2: Vocab Sharding ---
    # Replicate Batch, Shard Vocab (axis 1)
    # Expected behavior: Local TopK -> AllGather -> Reduction TopK
    print("\n--- Test Case 2: Vocab Sharding (Tensor Parallel) ---")
    mesh_1d = Mesh(jax.devices(), axis_names=('model'))
    sharding_vocab = NamedSharding(mesh_1d, P(None, 'model'))

    @partial(jax.jit, in_shardings=(sharding_vocab, None))
    def test_vocab(x, k):
        return top_bounded_k(x, k, MAX_K)

    try:
        out_vals, out_idxs = test_vocab(input_logits, k_array)
        passed.append(verify(out_vals, out_idxs, "Vocab Sharding"))
    except Exception as e:
        print(f"❌ Vocab Sharding Failed with exception: {e}")
        passed.append(False)

    # --- Test Case 3: Both Sharding (2D Mesh) ---
    # Split Batch by 2, Vocab by 4
    # Expected behavior: Sharded on both.
    # The partition logic uses arg_shardings[0].spec[1] to determine reduction axis.
    print("\n--- Test Case 3: Both Sharding (2x4 Mesh) ---")
    mesh_2d = Mesh(np.array(jax.devices()).reshape(2, 4), axis_names=('data', 'model'))
    sharding_both = NamedSharding(mesh_2d, P('data', 'model'))

    @partial(jax.jit, in_shardings=(sharding_both, None))
    def test_both(x, k):
        return top_bounded_k(x, k, MAX_K)

    try:
        out_vals, out_idxs = test_both(input_logits, k_array)
        passed.append(verify(out_vals, out_idxs, "Both Sharding"))
    except Exception as e:
        print(f"❌ Both Sharding Failed with exception: {e}")
        passed.append(False)

    # Print summary
    print("\n" + "="*80)
    print(f"Sharded TopK Tests: {sum(passed)}/{len(passed)} passed")
    print("="*80)

    return all(passed)


if __name__ == "__main__":
    print("="*80)
    print("SHARDED TOP-K TEST WITH DECOMPILATION")
    print("="*80)

    # First run decompilation tests on simpler functions
    decompile_tests_passed = []
    decompile_tests_passed.append(test_decompile_simple_function())
    decompile_tests_passed.append(test_decompile_topk())

    print("\n" + "="*80)
    print(f"Decompilation Tests: {sum(decompile_tests_passed)}/{len(decompile_tests_passed)} passed")
    print("="*80)

    # Then run the full sharded simulation
    sharded_tests_passed = run_simulation()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Decompilation Tests: {'✅ PASS' if all(decompile_tests_passed) else '❌ FAIL'}")
    print(f"Sharded Tests: {'✅ PASS' if sharded_tests_passed else '❌ FAIL'}")

    overall_pass = all(decompile_tests_passed) and sharded_tests_passed
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}")

    exit(0 if overall_pass else 1)
