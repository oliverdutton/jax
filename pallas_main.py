
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from cse_pass import cse_jaxpr
from main import iota_tile, create_bit_indicator, count_operations_in_jaxpr, body_fn

def pallas_kernel(x_ref, y_ref):
    # This kernel just calls the original body_fn
    x = x_ref[()]
    y = body_fn(0, x)
    y_ref[()] = y

def test_pallas_cse():
    # 1. Get the jaxpr of the pallas_call
    pallas_call = pl.pallas_call(
        pallas_kernel,
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.int32),
        interpret=True
    )
    pallas_jaxpr = jax.make_jaxpr(pallas_call)(jnp.zeros((128, 128), dtype=jnp.int32))

    # 2. Count operations before CSE
    print("--- Before CSE (Pallas) ---")
    op_counts_before = count_operations_in_jaxpr(pallas_jaxpr)
    for op, count in op_counts_before.most_common():
        print(f"{op}: {count}")

    # 3. Apply CSE pass
    cse_pallas_jaxpr = cse_jaxpr(pallas_jaxpr.jaxpr)

    # 4. Count operations after CSE
    print("\n--- After CSE (Pallas) ---")
    op_counts_after = count_operations_in_jaxpr(cse_pallas_jaxpr)
    for op, count in op_counts_after.most_common():
        print(f"{op}: {count}")

if __name__ == "__main__":
    test_pallas_cse()
