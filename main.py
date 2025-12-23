
import jax
import jax.numpy as jnp
from jax import lax
from collections import Counter
import re
from jax._src.core import ClosedJaxpr

from cse_pass import cse_jaxpr

def iota_tile(dim):
    """Create iota array with tile shape - matching tallax implementation."""
    return lax.broadcasted_iota(jnp.int32, (128, 128), dim)

def create_bit_indicator(bit_position, index):
    """Create mask indicating which elements have specific bit set."""
    if type(bit_position) == int:
        bit = (index & (1 << bit_position))
        return bit > 0
    return (index >> bit_position) & 1

def body_fn(idx, val):
    i = idx
    for _ in range(200):
        # Pattern 1: Call iota_tile INSIDE loop (not hoisted!)
        # CSE should eliminate duplicate calls
        iota_0_local = iota_tile(0)
        iota_1_local = iota_tile(1)
        tile_local_offset = iota_0_local + (iota_1_local // 128) * 16

        # Pattern 2: is_right_half
        # Since i values repeat (0,1,0,1,2,0,2), CSE should recognize duplicates
        intra_tile_separation = 1 << i
        is_right_half = create_bit_indicator(i, iota_0_local)

        # Pattern 3: permutation
        # With repeated i values, these should also be CSE'd
        permutation = jnp.bitwise_xor(iota_0_local, intra_tile_separation)

        # Add together to create dependency
        val += tile_local_offset + is_right_half.astype(jnp.int32) + permutation
    return val

def test_rematerialization_patterns():
    num_iterations = 7
    # --- Jaxpr analysis ---
    # 1. Get the jaxpr of the body function
    body_jaxpr = jax.make_jaxpr(body_fn)(0, jnp.zeros((128, 128), dtype=jnp.int32))

    # 2. Count operations before CSE
    print("--- Before CSE ---")
    op_counts_before = count_operations_in_jaxpr(body_jaxpr)
    for op, count in op_counts_before.most_common():
        print(f"{op}: {count}")

    # 3. Apply CSE pass
    cse_body_jaxpr = cse_jaxpr(body_jaxpr.jaxpr)

    # 4. Count operations after CSE
    print("\n--- After CSE ---")
    op_counts_after = count_operations_in_jaxpr(cse_body_jaxpr)
    for op, count in op_counts_after.most_common():
        print(f"{op}: {count}")

    # --- Original execution ---
    final_result = lax.fori_loop(0, num_iterations, lambda i, val: body_fn(i, val), jnp.zeros((128, 128), dtype=jnp.int32))
    return final_result.sum()

def count_operations_in_jaxpr(jaxpr_obj):
    """Count operations in jaxpr including nested ones."""
    jaxpr = jaxpr_obj.jaxpr if hasattr(jaxpr_obj, 'jaxpr') else jaxpr_obj

    ops = []
    def collect_ops(j):
        for eqn in j.eqns:
            ops.append(eqn.primitive.name)
            # Recursively check nested jaxprs
            for param_name in ['jaxpr', 'body_jaxpr', 'cond_jaxpr']:
                if param_name in eqn.params:
                    nested = eqn.params[param_name]
                    if hasattr(nested, 'jaxpr'):
                        collect_ops(nested.jaxpr)
                    elif hasattr(nested, 'eqns'):
                        collect_ops(nested)

    collect_ops(jaxpr)
    return Counter(ops)

if __name__ == "__main__":
    test_rematerialization_patterns()
