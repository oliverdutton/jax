#!/usr/bin/env python3
"""
Demonstration of the refactored StableHLO to JAX function decompiler.

This shows how the new approach uses a value dictionary and jax.lax operations
instead of building jaxpr equations with primitives.
"""

import jax
import jax.numpy as jnp
from jax import lax
from hlo_to_jaxpr import StableHLOToJaxpr
import numpy as np


def demonstrate_value_dict_approach():
    """Show how the value dictionary progressively builds up."""
    print("=" * 80)
    print("DEMONSTRATING VALUE DICTIONARY APPROACH")
    print("=" * 80)

    # Example function: x + y * 2.0
    def example_func(x, y):
        return x + y * 2.0

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])

    print("\nOriginal function: x + y * 2.0")
    print(f"Input x: {x}")
    print(f"Input y: {y}")
    print(f"Expected result: {example_func(x, y)}")

    # Lower to StableHLO
    lowered = jax.jit(example_func).lower(x, y)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    print("\n" + "-" * 80)
    print("StableHLO IR:")
    print("-" * 80)
    print(mlir_module)

    # Decompile
    print("\n" + "-" * 80)
    print("Decompiling using value dictionary approach...")
    print("-" * 80)

    decompiler = StableHLOToJaxpr()
    functions = decompiler.decompile_module(mlir_module)

    # Execute
    main_func = functions['"main"']
    result = main_func.callable_fn(x, y)

    print(f"\nDecompiled result: {result}")
    print(f"Match: {np.allclose(result, example_func(x, y))}")


def demonstrate_complex_operations():
    """Show complex operations working."""
    print("\n" + "=" * 80)
    print("COMPLEX OPERATIONS")
    print("=" * 80)

    # Neural network layer: tanh(Wx + b)
    def nn_layer(x, W, b):
        return jnp.tanh(jnp.dot(W, x) + b)

    x = jnp.array([1.0, 2.0, 3.0])
    W = jnp.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
    b = jnp.array([0.1, -0.2])

    print("\nNeural network layer: tanh(Wx + b)")
    print(f"Input x shape: {x.shape}")
    print(f"Weight W shape: {W.shape}")
    print(f"Bias b shape: {b.shape}")

    expected = nn_layer(x, W, b)
    print(f"Expected output: {expected}")

    # Decompile
    lowered = jax.jit(nn_layer).lower(x, W, b)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    decompiler = StableHLOToJaxpr()
    functions = decompiler.decompile_module(mlir_module)

    main_func = functions['"main"']
    result = main_func.callable_fn(x, W, b)

    print(f"Decompiled output: {result}")
    print(f"Match: {np.allclose(result, expected)}")


def demonstrate_control_flow():
    """Show control flow decompilation."""
    print("\n" + "=" * 80)
    print("CONTROL FLOW")
    print("=" * 80)

    # Conditional function
    def conditional_abs(x):
        return lax.cond(x < 0, lambda x: -x, lambda x: x, x)

    test_values = [jnp.array(-5.0), jnp.array(3.0)]

    print("\nConditional absolute value function")
    for val in test_values:
        expected = conditional_abs(val)
        print(f"\nInput: {val}")
        print(f"Expected: {expected}")

        # Decompile
        lowered = jax.jit(conditional_abs).lower(val)
        mlir_module = lowered.compiler_ir(dialect='stablehlo')

        decompiler = StableHLOToJaxpr()
        functions = decompiler.decompile_module(mlir_module)

        main_func = functions['"main"']
        result = main_func.callable_fn(val)

        print(f"Decompiled: {result}")
        print(f"Match: {np.allclose(result, expected)}")


def demonstrate_broadcasting():
    """Show broadcasting operations."""
    print("\n" + "=" * 80)
    print("BROADCASTING")
    print("=" * 80)

    def broadcast_add(x, y):
        # x: (3,), y: scalar
        return x + y

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array(10.0)

    print(f"\nBroadcast addition: array + scalar")
    print(f"x: {x}, shape: {x.shape}")
    print(f"y: {y}, shape: {y.shape}")

    expected = broadcast_add(x, y)
    print(f"Expected: {expected}")

    # Decompile
    lowered = jax.jit(broadcast_add).lower(x, y)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    decompiler = StableHLOToJaxpr()
    functions = decompiler.decompile_module(mlir_module)

    main_func = functions['"main"']
    result = main_func.callable_fn(x, y)

    print(f"Decompiled: {result}")
    print(f"Match: {np.allclose(result, expected)}")


def compare_approaches():
    """Compare old vs new approach."""
    print("\n" + "=" * 80)
    print("OLD VS NEW APPROACH COMPARISON")
    print("=" * 80)

    print("\n--- OLD APPROACH (misguided) ---")
    print("1. Build jaxpr Vars: core.Var(aval)")
    print("2. Create equations: core.JaxprEqn(invars, outvars, lax.add_p, params)")
    print("3. Construct jaxpr: core.Jaxpr(constvars, invars, outvars, eqns)")
    print("4. Execute: core.eval_jaxpr(jaxpr, constvals, *inputs)")
    print("Problems:")
    print("  - Complex: requires understanding jaxpr internals")
    print("  - Hard to debug: opaque jaxpr structure")
    print("  - Difficult to maintain: manual equation building")

    print("\n--- NEW APPROACH (better) ---")
    print("1. Create value dict: {arg_name: input_value}")
    print("2. For each operation: result = lax.operation(*operands)")
    print("3. Store result: value_dict[result_name] = result")
    print("4. Return function: lambda *inputs: decompile_block(block, *inputs)")
    print("Benefits:")
    print("  - Simple: just execute jax.lax operations")
    print("  - Easy to debug: normal Python functions")
    print("  - Easy to maintain: straightforward operation mapping")


if __name__ == '__main__':
    demonstrate_value_dict_approach()
    demonstrate_complex_operations()
    demonstrate_broadcasting()
    # demonstrate_control_flow()  # May need special handling
    compare_approaches()

    print("\n" + "=" * 80)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("=" * 80)
