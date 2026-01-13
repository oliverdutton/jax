import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

# Check available devices
devices = jax.devices()
print(f"Available devices: {devices}")
print(f"Number of devices: {len(devices)}")

# If we have fewer than 4 devices, we'll create a simpler mesh
if len(devices) >= 4:
    mesh_devices = np.array(devices[:4]).reshape(2, 2)
    mesh = Mesh(mesh_devices, ('m', 'k'))
else:
    # Use a 1D mesh with available devices
    mesh_devices = np.array(devices)
    mesh = Mesh(mesh_devices, ('m',))

print(f"Mesh: {mesh}")

# Create sharding specs based on mesh
if len(devices) >= 4:
    # First input (128, 128): shape is (m, k), shard on both axes
    sharding_a = NamedSharding(mesh, P('m', 'k'))
    # Second input (128, 128): shape is (k, n), shard only on k axis
    sharding_b = NamedSharding(mesh, P('k', None))
else:
    # With 1D mesh, only shard on one axis
    sharding_a = NamedSharding(mesh, P('m', None))
    sharding_b = NamedSharding(mesh, P('m', None))

# Create the matmul function
def matmul(a, b):
    return jnp.matmul(a, b)

# Make jaxpr with sharded inputs
jaxpr = jax.make_jaxpr(matmul)(
    jax.lax.with_sharding_constraint(jnp.ones((128, 128)), sharding_a),
    jax.lax.with_sharding_constraint(jnp.ones((128, 128)), sharding_b)
)

print(jaxpr)
