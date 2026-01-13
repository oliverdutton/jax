#!/usr/bin/env python3
"""
Advanced test cases for the StableHLO to Jaxpr decompiler.

Tests:
- Multi-device sharding operations (psum, all-reduce)
- Attention mechanism
- Conditional SwiGLU activation
- Fixed-point iteration
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import numpy as np
from hlo_to_jaxpr import StableHLOToJaxpr
from jax._src import core

# Configure simulated multi-CPU environment for testing
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", False)

# Simulate 8 CPU devices for testing distributed operations
try:
    jax.config.update('jax_num_cpu_devices', 8)
    print(f"Configured {len(jax.devices())} simulated CPU devices")
except:
    print(f"Using default {len(jax.devices())} devices")


def decompile_and_verify(func, *args, name="test", test_equivalence=True):
    """
    Decompile a function and optionally verify results match.

    Args:
        func: Function to decompile
        *args: Arguments to pass to the function
        name: Test name
        test_equivalence: Whether to test that results match (may fail for distributed ops)
    """
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")

    # Lower and compile
    lowered = jax.jit(func).lower(*args)
    compiled = lowered.compile()

    # Get StableHLO MLIR from lowered object
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    print(f"\nOriginal function:")
    if test_equivalence:
        original_result = compiled(*args)
        print(f"Result shape: {getattr(original_result, 'shape', 'scalar')}")
        print(f"Result dtype: {getattr(original_result, 'dtype', type(original_result))}")

    # Decompile
    decompiler = StableHLOToJaxpr()
    try:
        functions = decompiler.decompile_module(mlir_module)

        # Find main function
        main_func = None
        for fname, func_obj in functions.items():
            if 'main' in fname:
                main_func = func_obj
                break

        if main_func is None:
            print("ERROR: Could not find main function")
            return False

        print(f"\nDecompiled jaxpr (first 600 chars):")
        jaxpr_str = str(main_func.jaxpr)
        print(jaxpr_str[:600])
        if len(jaxpr_str) > 600:
            print("...")

        # Try to execute if testing equivalence
        if test_equivalence:
            print("\nExecuting decompiled jaxpr:")
            try:
                result = core.eval_jaxpr(main_func.jaxpr, decompiler.constvals, *args)
                if isinstance(result, (list, tuple)) and len(result) == 1:
                    result = result[0]

                print(f"Decompiled result shape: {getattr(result, 'shape', 'scalar')}")
                print(f"Decompiled result dtype: {getattr(result, 'dtype', type(result))}")

                match = np.allclose(original_result, result, rtol=1e-5, atol=1e-5)
                print(f"Results match: {match}")
                return match
            except Exception as e:
                print(f"Execution failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("\nSkipping equivalence test")
            return True

    except Exception as e:
        print(f"\nDecompilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attention():
    """Test scaled dot-product attention."""
    print("\n" + "="*80)
    print("COMPLEX TEST 1: Scaled Dot-Product Attention")
    print("="*80)

    def attention(query, key, value):
        """Scaled dot-product attention."""
        # query: [batch, seq, d_k]
        # key: [batch, seq, d_k]
        # value: [batch, seq, d_v]

        d_k = query.shape[-1]
        scores = jnp.matmul(query, key.transpose(0, 2, 1)) / jnp.sqrt(d_k)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.matmul(attn_weights, value)
        return output

    batch, seq_len, d_k, d_v = 2, 4, 8, 8
    query = jnp.ones((batch, seq_len, d_k))
    key = jnp.ones((batch, seq_len, d_k))
    value = jnp.ones((batch, seq_len, d_v))

    return decompile_and_verify(attention, query, key, value,
                                name="Attention", test_equivalence=True)


def test_conditional_swiglu():
    """Test conditional SwiGLU activation."""
    print("\n" + "="*80)
    print("COMPLEX TEST 2: Conditional SwiGLU")
    print("="*80)

    def conditional_swiglu(x, gate_threshold=0.0):
        """
        SwiGLU with conditional gating.

        SwiGLU(x) = swish(x_1) * x_2
        But only apply if mean(x) > threshold, otherwise scale by 0.5.
        """
        x_mean = jnp.mean(x)

        def apply_swiglu(x):
            # Split input
            x1, x2 = jnp.split(x, 2, axis=-1)
            # Swish activation (SiLU)
            swish = x1 / (1 + jnp.exp(-x1))
            # Element-wise multiply
            return swish * x2

        def apply_scale(x):
            # Also split to match output shape
            x1, x2 = jnp.split(x, 2, axis=-1)
            return x1 * 0.5

        # Conditional: apply SwiGLU if mean > threshold, else scale
        return lax.cond(x_mean > gate_threshold, apply_swiglu, apply_scale, x)

    x = jnp.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])

    return decompile_and_verify(conditional_swiglu, x,
                                name="Conditional SwiGLU", test_equivalence=True)


def test_fixed_point_iteration():
    """Test fixed-point iteration."""
    print("\n" + "="*80)
    print("COMPLEX TEST 3: Fixed-Point Iteration")
    print("="*80)

    def fixed_point_sqrt(x, n_iters=5):
        """
        Compute sqrt using Newton's method (fixed-point iteration).

        x_{n+1} = 0.5 * (x_n + a / x_n)
        """
        def body(i, estimate):
            return 0.5 * (estimate + x / estimate)

        # Start with initial guess
        initial = x / 2.0
        result = lax.fori_loop(0, n_iters, body, initial)
        return result

    x = jnp.array([4.0, 9.0, 16.0])

    return decompile_and_verify(fixed_point_sqrt, x,
                                name="Fixed-Point Iteration", test_equivalence=True)


def test_distributed_psum():
    """Test distributed psum (all-reduce) using shard_map."""
    print("\n" + "="*80)
    print("COMPLEX TEST 4: Distributed psum (all-reduce)")
    print("="*80)

    from jax.experimental.shard_map import shard_map

    devices = jax.devices()[:4]  # Use 4 devices
    mesh = Mesh(devices, axis_names=('i',))

    def distributed_mean_body(x):
        """Compute mean across devices using psum."""
        # Sum across devices
        total = lax.psum(x, axis_name='i')
        # Divide by number of devices
        return total / len(devices)

    # Create sharded input
    x = jnp.arange(16.0).reshape(4, 4)

    with mesh:
        # Shard along first dimension
        sharding = NamedSharding(mesh, P('i', None))
        x_sharded = jax.device_put(x, sharding)

        # Use shard_map to properly bind axis name
        distributed_mean = shard_map(
            distributed_mean_body,
            mesh=mesh,
            in_specs=P('i', None),
            out_specs=P('i', None)
        )

        # Lower and get StableHLO
        try:
            lowered = jax.jit(distributed_mean).lower(x_sharded)
            mlir_module = lowered.compiler_ir(dialect='stablehlo')

            print("\nStableHLO for distributed operation (first 800 chars):")
            mlir_str = str(mlir_module)
            print(mlir_str[:800])
            if len(mlir_str) > 800:
                print("...")

            # Look for all-reduce operations
            if 'all-reduce' in mlir_str.lower() or 'all_reduce' in mlir_str:
                print("\n✓ Found all-reduce operation in StableHLO!")
            else:
                print("\n✗ No all-reduce found (might be optimized or use different op)")

            # Try to decompile
            print("\nAttempting decompilation...")
            decompiler = StableHLOToJaxpr()
            functions = decompiler.decompile_module(mlir_module)
            main_func = functions.get('"main"')
            if main_func:
                print("Decompiled jaxpr (first 600 chars):")
                print(str(main_func.jaxpr)[:600])
                print("\n✓ Successfully decompiled distributed operation!")
                return True
        except Exception as e:
            print(f"Decompilation note: {e}")
            print("(Distributed ops may need additional primitive mappings)")
            import traceback
            traceback.print_exc()
            return False


def test_layer_norm():
    """Test layer normalization."""
    print("\n" + "="*80)
    print("COMPLEX TEST 5: Layer Normalization")
    print("="*80)

    def layer_norm(x, eps=1e-5):
        """Layer normalization."""
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(var + eps)
        return normalized

    x = jnp.array([[1.0, 2.0, 3.0, 4.0],
                   [2.0, 4.0, 6.0, 8.0],
                   [1.0, 1.0, 1.0, 1.0]])

    return decompile_and_verify(layer_norm, x,
                                name="Layer Normalization", test_equivalence=True)


def main():
    """Run all advanced tests."""
    print("\n" + "="*80)
    print("ADVANCED STABLEHLO TO JAXPR DECOMPILER TESTS")
    print("="*80)
    print("\nTesting complex operations including:")
    print("  • Attention mechanism")
    print("  • Conditional SwiGLU activation")
    print("  • Fixed-point iteration")
    print("  • Distributed operations (psum/all-reduce)")
    print("  • Layer normalization")

    results = []

    # Run tests
    results.append(("Attention", test_attention()))
    results.append(("Conditional SwiGLU", test_conditional_swiglu()))
    results.append(("Fixed-Point Iteration", test_fixed_point_iteration()))
    results.append(("Distributed psum", test_distributed_psum()))
    results.append(("Layer Normalization", test_layer_norm()))

    # Print summary
    print("\n" + "="*80)
    print("ADVANCED TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n🎉 All advanced tests passed!")
    else:
        print(f"\n⚠️  {total - passed_count} test(s) need attention")

    return passed_count == total


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
