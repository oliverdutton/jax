import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

# Create a "fake" 2x2 mesh using the same device 4 times
devices = jax.devices() * 4  # Repeat the device list 4 times
mesh_devices = np.array(devices).reshape(2, 2)
mesh = Mesh(mesh_devices, ('m', 'k'))

print(f"Mesh: {mesh}")

# Create sharding specs
# First input (128, 128): shape is (m, k), shard on both axes
sharding_a = NamedSharding(mesh, P('m', 'k'))
# Second input (128, 128): shape is (k, n), shard only on k axis
sharding_b = NamedSharding(mesh, P('k', None))

# Create the matmul function
def matmul(a, b):
    return jnp.matmul(a, b)

# Use jax.jit with in_shardings and lower it
jitted_matmul = jax.jit(matmul, in_shardings=(sharding_a, sharding_b))

# Lower the jitted function
lowered = jitted_matmul.lower(jnp.ones((128, 128)), jnp.ones((128, 128)))

print("Lowered representation:")
print(lowered.as_text())
print("\n" + "="*80 + "\n")

# Compile the lowered function
compiled = lowered.compile()

print("Compiled successfully!")
print(f"Compiled type: {type(compiled)}")
print("\n" + "="*80 + "\n")

# Print the compiled HLO
print("Compiled HLO:")
print(compiled.as_text())
print("\n" + "="*80 + "\n")

# Try to run it (this will attempt to shard across the fake mesh)
print("Attempting to run the compiled function...")
try:
    a = jnp.ones((128, 128))
    b = jnp.ones((128, 128))
    result = compiled(a, b)
    print(f"Result shape: {result.shape}")
    print(f"Result sample values: {result[0, :5]}")
except Exception as e:
    print(f"Execution failed (expected with fake mesh): {type(e).__name__}")
    print(f"Error: {str(e)}")
    print("\nThis is expected because the fake mesh has duplicate devices.")

print("\n" + "="*80 + "\n")

# Reconstruct the computation using shard_map based on the compiled HLO
print("Reconstructing with shard_map (pure JAX):")
print()

from jax import shard_map

def distributed_matmul_kernel(a_shard, b_shard):
    """
    Per-device computation kernel.

    Based on the compiled HLO:
    - a_shard: f32[64,64] (sharded on both dimensions)
    - b_shard: f32[64,128] (sharded on k dimension only)
    - Local dot: produces f32[64,128]
    - All-reduce: sums across the k axis to get final result
    """
    # Local matrix multiplication on this device's shard
    local_result = jnp.matmul(a_shard, b_shard)

    # All-reduce across the 'k' axis to sum partial results
    # This corresponds to the all-reduce in the HLO
    result = jax.lax.psum(local_result, 'k')

    return result

# Create the sharded computation
sharded_matmul = shard_map(
    distributed_matmul_kernel,
    mesh=mesh,
    in_specs=(P('m', 'k'), P('k', None)),
    out_specs=P('m', None)
)

print("Shard_map function created with:")
print(f"  Mesh: {mesh}")
print(f"  Input specs: (P('m', 'k'), P('k', None))")
print(f"  Output spec: P('m', None)")
print(f"  Kernel: local matmul + psum over 'k' axis")
print()

# Show the lowered representation of the shard_map version
print("Lowered shard_map version:")
sharded_lowered = jax.jit(sharded_matmul).lower(jnp.ones((128, 128)), jnp.ones((128, 128)))
print(sharded_lowered.as_text())
