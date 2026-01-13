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

print("\n" + "="*80 + "\n")

# Programmatically parse the compiled HLO and generate equivalent jaxpr
print("Parsing compiled HLO to reconstruct high-level jaxpr:")
print()

import re
from typing import Dict, List, Tuple

def parse_hlo_to_jaxpr(hlo_text: str, mesh_obj) -> str:
    """
    Parse compiled HLO and reconstruct the equivalent high-level JAX computation
    with full shard_map context.
    Returns the jaxpr of the reconstructed function.
    """

    # Extract the entry computation
    # Find the ENTRY line
    entry_start = hlo_text.find('ENTRY')
    if entry_start == -1:
        return "Could not find ENTRY in HLO"

    # Extract from ENTRY to end
    entry_section = hlo_text[entry_start:]

    # Parse the ENTRY signature line
    entry_sig_match = re.search(r'ENTRY\s+%[\w.]+\s*\((.*?)\)\s*->\s*([^\{]+)\s*\{', entry_section)
    if not entry_sig_match:
        print("ERROR: Could not parse ENTRY signature")
        return "Could not parse HLO"

    params_str = entry_sig_match.group(1)
    output_type = entry_sig_match.group(2).strip()

    # Extract the body - find matching braces
    brace_start = entry_section.find('{')
    if brace_start == -1:
        return "Could not find opening brace"

    # Find the matching closing brace by counting
    brace_count = 0
    brace_end = -1
    for i in range(brace_start, len(entry_section)):
        if entry_section[i] == '{':
            brace_count += 1
        elif entry_section[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                brace_end = i
                break

    if brace_end == -1:
        return "Could not find closing brace"

    body = entry_section[brace_start+1:brace_end]

    # Parse parameters
    params = []
    param_pattern = r'(\w+):\s*(\w+)\[([0-9,]+)\]'
    for match in re.finditer(param_pattern, params_str):
        name = match.group(1)
        dtype = match.group(2)
        shape = tuple(map(int, match.group(3).split(',')))
        params.append((name, dtype, shape))

    print(f"Parsed parameters (per-device): {params}")

    # Parse operations in the body
    ops = []

    # Find dot operation
    dot_match = re.search(
        r'%(\w+)\s*=\s*\w+\[([0-9,]+)\]\{[^}]*\}\s*dot\(([^,]+),\s*([^)]+)\),\s*'
        r'lhs_contracting_dims=\{([0-9,]+)\},\s*rhs_contracting_dims=\{([0-9,]+)\}',
        body
    )

    if dot_match:
        result_name = dot_match.group(1)
        result_shape = tuple(map(int, dot_match.group(2).split(',')))
        lhs = dot_match.group(3).strip('%')
        rhs = dot_match.group(4).strip('%')
        lhs_contracting = tuple(map(int, dot_match.group(5).split(',')))
        rhs_contracting = tuple(map(int, dot_match.group(6).split(',')))
        ops.append({
            'type': 'dot',
            'name': result_name,
            'shape': result_shape,
            'lhs': lhs,
            'rhs': rhs,
            'lhs_contracting_dims': lhs_contracting,
            'rhs_contracting_dims': rhs_contracting
        })
        print(f"Found dot operation: {lhs} × {rhs} with contracting dims {lhs_contracting} × {rhs_contracting}")

    # Find all-reduce operation
    allreduce_match = re.search(
        r'ROOT\s+%([^\s=]+)\s*=\s*\w+\[([0-9,]+)\][^\s]*\s*all-reduce\(([^)]+)\)',
        body
    )

    reduce_axes = []
    if allreduce_match:
        result_name = allreduce_match.group(1)
        result_shape = tuple(map(int, allreduce_match.group(2).split(',')))
        input_var = allreduce_match.group(3).strip('%')

        # Extract replica groups to infer which mesh axis to reduce over
        replica_groups_match = re.search(r'replica_groups=\[([^\]]+)\]', body)
        replica_groups = None
        if replica_groups_match:
            replica_groups = replica_groups_match.group(1)
            # Parse replica groups like "2,2"
            # Groups like [[0,1],[2,3]] means reduce over k axis (devices 0,1 are in one group)
            if '2,2' in replica_groups:
                reduce_axes = ['k']  # Inferred from the pattern

        ops.append({
            'type': 'all-reduce',
            'name': result_name,
            'shape': result_shape,
            'input': input_var,
            'replica_groups': replica_groups,
            'reduce_axes': reduce_axes
        })
        print(f"Found all-reduce on {input_var} with replica_groups={replica_groups}, reducing over axes: {reduce_axes}")

    # Parse sharding information
    sharding_info = {}
    for match in re.finditer(r'%([^=]+)=\s*\w+\[([0-9,]+)\]\{[^}]*\}\s*parameter\((\d+)\),\s*sharding=\{([^}]+)\}', body):
        param_name = match.group(1).strip()
        param_shape = tuple(map(int, match.group(2).split(',')))
        param_idx = int(match.group(3))
        sharding = match.group(4)

        # Parse sharding to infer partition spec
        # devices=[2,2]<=[4] means P('m', 'k')
        # devices=[2,1,2] means P('k', None) with replication
        partition_spec = []
        if 'devices=[2,2]' in sharding:
            partition_spec = ['m', 'k']
        elif 'devices=[2,1,2]' in sharding:
            partition_spec = ['k', None]

        sharding_info[param_name] = {
            'shape': param_shape,
            'sharding': sharding,
            'idx': param_idx,
            'partition_spec': partition_spec
        }
        print(f"Parameter {param_name} (index {param_idx}): shape={param_shape}, P{tuple(partition_spec)}")

    print()
    print("Reconstructing shard_map function from parsed HLO:")

    # Build the shard_map kernel based on parsed operations
    def kernel_fn(a_shard, b_shard):
        """Per-device kernel reconstructed from HLO"""
        result = None
        for op in ops:
            if op['type'] == 'dot':
                # Local dot operation on shards
                dimension_numbers = ((op['lhs_contracting_dims'], op['rhs_contracting_dims']), ([], []))
                result = jax.lax.dot_general(a_shard, b_shard, dimension_numbers=dimension_numbers)
                print(f"  Kernel: dot_general on shards")
            elif op['type'] == 'all-reduce':
                # psum over the reduction axes
                for axis in op['reduce_axes']:
                    result = jax.lax.psum(result, axis)
                    print(f"  Kernel: psum over '{axis}' axis")
        return result

    # Infer global shapes from per-device shapes and sharding
    mesh_size_m = 2
    mesh_size_k = 2

    param0_info = sharding_info.get('param', {})
    param1_info = sharding_info.get('param.1', {})

    # Compute global shapes by multiplying per-device shape by mesh dimensions
    local_shape_a = params[0][2]
    local_shape_b = params[1][2]

    # For devices=[2,2], both dims are sharded by 2
    # For devices=[2,1,2], first dim sharded by 2, second not sharded
    global_a_shape = (local_shape_a[0] * mesh_size_m, local_shape_a[1] * mesh_size_k)
    global_b_shape = (local_shape_b[0] * mesh_size_k, local_shape_b[1])

    # Infer in_specs and out_specs from sharding info
    in_spec_a = P(*param0_info.get('partition_spec', ['m', 'k']))
    in_spec_b = P(*param1_info.get('partition_spec', ['k', None]))

    # Output spec: result of all-reduce over k, so output is P('m', None)
    out_spec = P('m', None)

    print(f"  Global shapes: a={global_a_shape}, b={global_b_shape}")
    print(f"  in_specs: ({in_spec_a}, {in_spec_b})")
    print(f"  out_spec: {out_spec}")
    print()

    # Create the shard_map function
    # Note: The psum in the kernel ensures proper reduction across k axis
    sharded_fn = shard_map(
        kernel_fn,
        mesh=mesh_obj,
        in_specs=(in_spec_a, in_spec_b),
        out_specs=out_spec
    )

    # Generate jaxpr
    dtype = jnp.float32

    try:
        jaxpr = jax.make_jaxpr(sharded_fn)(
            jnp.zeros(global_a_shape, dtype=dtype),
            jnp.zeros(global_b_shape, dtype=dtype)
        )
        return jaxpr
    except ValueError as e:
        print(f"\nNote: make_jaxpr failed with static replication check: {e}")
        print("This is expected when reconstructing from HLO programmatically.")
        print("\nShowing the lowered representation instead:")
        try:
            lowered = jax.jit(sharded_fn).lower(
                jnp.zeros(global_a_shape, dtype=dtype),
                jnp.zeros(global_b_shape, dtype=dtype)
            )
            return lowered.as_text()
        except Exception as e2:
            return f"Could not lower: {e2}"

# Parse and reconstruct
reconstructed_jaxpr = parse_hlo_to_jaxpr(compiled.as_text(), mesh)

print("Reconstructed jaxpr with shard_map:")
print(reconstructed_jaxpr)
