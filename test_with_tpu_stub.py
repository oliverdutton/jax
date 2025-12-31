"""
Test TPU compilation on CPU with the custom call stub registered.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh
from jax._src.lib import xla_client

# Force CPU platform
jax.config.update('jax_platforms', 'cpu')

print("="*80)
print("TESTING TPU COMPILATION WITH CUSTOM CALL STUB")
print("="*80)

# Step 1: Load the stub extension and register it
print("\n1. Loading and registering tpu_custom_call stub...")
try:
    import tpu_stub_extension

    # Get the PyCapsule containing the function pointer
    capsule = tpu_stub_extension.tpu_custom_call_capsule

    # Register it with XLA for the CPU platform
    xla_client.register_custom_call_target(
        "tpu_custom_call",
        capsule,
        "cpu",
        api_version=0  # Untyped custom call API
    )

    print("   ✓ Stub registered successfully!")
except Exception as e:
    print(f"   ✗ Failed to register stub: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 2: Create a simple Pallas TPU kernel
print("\n2. Creating Pallas TPU kernel...")

def add_kernel(x_ref, y_ref, o_ref):
    """Simple addition kernel."""
    o_ref[...] = x_ref[...] + y_ref[...]

x = jnp.ones((8, 128), dtype=jnp.float32)
y = jnp.ones((8, 128), dtype=jnp.float32)

def add_matrices(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        grid=(1,),
        in_specs=[
            pl.BlockSpec((8, 128), lambda i: (0, 0)),
            pl.BlockSpec((8, 128), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        backend="mosaic_tpu",
        interpret=False,
    )(x, y)

print("   ✓ Kernel created")

# Step 3: Lower with mocked TPU v5e device
print("\n3. Lowering with mocked TPU v5e...")
tpu_v5e = AbstractDevice(device_kind="TPU v5e", num_cores=1)
mesh = AbstractMesh((), (), abstract_device=tpu_v5e)

try:
    with use_abstract_mesh(mesh):
        lowered = jax.jit(add_matrices).lower(x, y)
        print("   ✓ Lowering succeeded")

        # Step 4: ATTEMPT COMPILATION!
        print("\n4. Attempting compilation with stub...")
        print("   (Note: stub is registered, so this might work!)")

        compiled = lowered.compile()

        print("   ✓✓✓ COMPILATION SUCCEEDED! ✓✓✓")
        print(f"   Compiled object: {type(compiled)}")

        # Step 5: Try to run it (will probably fail or give wrong results)
        print("\n5. Attempting execution (will likely fail or be incorrect)...")
        try:
            result = compiled(x, y)
            print(f"   ⚠ Execution completed (but result may be wrong)")
            print(f"   Result shape: {result.shape}")
            print(f"   Result dtype: {result.dtype}")
            print(f"   Result sample: {result[0, :5]}")

            # Check if result is correct (should be all 2.0)
            expected = x + y
            if jnp.allclose(result, expected):
                print("   ✓✓✓ RESULT IS CORRECT!!! ✓✓✓")
            else:
                print("   ⚠ Result is incorrect (as expected, stub is no-op)")

        except Exception as e:
            print(f"   ✗ Execution failed: {e}")
            print("   (This is expected - the stub is just a placeholder)")

except Exception as e:
    print(f"   ✗ Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("SUCCESS! TPU code can now compile on CPU!")
print("="*80)
print("\nNotes:")
print("  • Compilation works because we registered a stub for tpu_custom_call")
print("  • Execution will fail or produce wrong results (stub is a no-op)")
print("  • This is useful for testing the compilation pipeline")
print("  • For actual execution, use interpret=True or real TPU hardware")
print("="*80)
