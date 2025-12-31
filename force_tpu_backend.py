"""
Attempt to force TPU backend initialization by patching JAX internals.

This tries various monkey-patching approaches to bypass hardware detection.
"""
import os
import sys
from unittest import mock

# Set all possible TPU environment variables BEFORE importing JAX
os.environ['TPU_ACCELERATOR_TYPE'] = 'v5e'
os.environ['TPU_WORKER_HOSTNAMES'] = '127.0.0.1:8470'
os.environ['TPU_WORKER_ID'] = '0'
os.environ['TPU_WORKER_NETWORK_ENDPOINTS'] = '127.0.0.1:8470'
os.environ['TPU_NUM_DEVICES'] = '1'
os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '1,1,1'
os.environ['TPU_HOST_BOUNDS'] = '1,1,1'
os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = '127.0.0.1:8476'
os.environ['TPU_MESH_CONTROLLER_PORT'] = '8476'


def attempt_1_mock_device_detection():
    """Try to mock the device detection in libtpu."""
    print("\n" + "="*80)
    print("Attempt 1: Mock TPU device detection")
    print("="*80)

    # Idea: Patch the device detection before importing JAX
    # This won't work because libtpu is a compiled .so file

    import jax
    from jax._src import xla_bridge

    try:
        backend = xla_bridge.get_backend('tpu')
        print(f"✓ SUCCESS! {backend}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def attempt_2_patch_backend_factory():
    """Try to patch the backend factory to return a fake TPU backend."""
    print("\n" + "="*80)
    print("Attempt 2: Patch backend factory")
    print("="*80)

    import jax
    from jax._src import xla_bridge

    # Try to access the internals
    print("Backend factories:", xla_bridge._backends)

    # Try to manually create a client
    try:
        from jax._src.lib import xla_client
        from jax._src.lib import xla_extension

        # This will probably fail, but worth a try
        print("Attempting to create TPU client directly...")

        # Check if we can access the TPU plugin
        print("XLA extension attributes:", dir(xla_extension))

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def attempt_3_fake_tpu_device():
    """Try to create a fake TPU device class."""
    print("\n" + "="*80)
    print("Attempt 3: Create fake TPU device")
    print("="*80)

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh

    # Create mocked TPU device (this we know works for lowering)
    tpu_v5e = AbstractDevice(device_kind="TPU v5e", num_cores=1)
    mesh = AbstractMesh((), (), abstract_device=tpu_v5e)

    print("Created abstract TPU device:", tpu_v5e)

    # Try to compile with the abstract device
    def add_kernel(x_ref, y_ref, o_ref):
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

    try:
        print("Lowering with abstract device...")
        with use_abstract_mesh(mesh):
            lowered = jax.jit(add_matrices).lower(x, y)
            print("✓ Lowering succeeded")

            print("Attempting compilation...")
            compiled = lowered.compile()
            print("✓✓✓ COMPILATION SUCCEEDED!")
            return True
    except Exception as e:
        print(f"✗ Compilation failed: {type(e).__name__}: {e}")
        return False


def attempt_4_intercept_custom_call():
    """Try to register a stub for tpu_custom_call on CPU."""
    print("\n" + "="*80)
    print("Attempt 4: Register tpu_custom_call stub on CPU")
    print("="*80)

    try:
        from jax._src.lib import xla_client

        # The problem is that we need a PyCapsule for the function pointer
        # Python-only functions won't work for XLA custom calls

        print("Checking if we can register a stub...")
        print("xla_client.register_custom_call_target available:",
              hasattr(xla_client, 'register_custom_call_target'))

        if hasattr(xla_client, 'register_custom_call_target'):
            # Try to register with None (will fail, but shows the signature)
            try:
                xla_client.register_custom_call_target(
                    "test_call",
                    None,
                    "cpu",
                    api_version=0
                )
            except TypeError as e:
                print(f"Registration signature error: {e}")
                print("\nThis confirms we need a C/C++ PyCapsule, not a Python function")

        return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def main():
    """Try all attempts."""
    print("="*80)
    print("FORCING TPU BACKEND COMPILATION ON CPU")
    print("="*80)

    results = []

    results.append(("Mock device detection", attempt_1_mock_device_detection()))
    results.append(("Patch backend factory", attempt_2_patch_backend_factory()))
    results.append(("Fake TPU device", attempt_3_fake_tpu_device()))
    results.append(("Register stub", attempt_4_intercept_custom_call()))

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name:30s}: {status}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if any(success for _, success in results):
        print("✓ At least one approach worked!")
    else:
        print("✗ All approaches failed.")
        print("\nThe fundamental issue is that:")
        print("  1. libtpu.so is compiled C++ code that checks for hardware")
        print("  2. Without hardware, the TPU backend won't initialize")
        print("  3. Without backend init, tpu_custom_call isn't registered")
        print("  4. tpu_custom_call requires a C/C++ implementation (not Python)")
        print("\nWorkarounds:")
        print("  • Use interpret=True for execution on CPU")
        print("  • Use lowering + IR inspection (already works)")
        print("  • Deploy to actual TPU hardware for compilation")


if __name__ == "__main__":
    main()
