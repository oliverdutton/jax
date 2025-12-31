"""
Try to use mosaic_tpu compiler directly to compile HLO to LLO.

This bypasses the device initialization and tries to invoke the TPU compiler
directly on MLIR/HLO.
"""
import os
import tempfile

os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh

print("="*80)
print("ATTEMPTING TO USE MOSAIC TPU COMPILER DIRECTLY")
print("="*80)

# Mini-attention kernel
def mini_attention(x, w1, w2):
    h = x @ w1
    rms = jnp.sqrt(jnp.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
    h = h / rms
    h_max = jnp.max(h, axis=-1, keepdims=True)
    exp_h = jnp.exp(h - h_max)
    h = exp_h / jnp.sum(exp_h, axis=-1, keepdims=True)
    out = h @ w2
    return out

# Create inputs
batch, d_in, d_mid, d_out = 16, 64, 64, 32
key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)
x = jax.random.normal(k1, (batch, d_in))
w1 = jax.random.normal(k2, (d_in, d_mid)) * 0.02
w2 = jax.random.normal(k3, (d_mid, d_out)) * 0.02

# Lower with mocked TPU
tpu_v5e = AbstractDevice(device_kind="TPU v5e", num_cores=1)
mesh = AbstractMesh((), (), abstract_device=tpu_v5e)

print("\n1. Lowering to get MLIR module...")
with use_abstract_mesh(mesh):
    lowered = jax.jit(mini_attention).lower(x, w1, w2)
    mlir_module = lowered.compiler_ir()
    print(f"✓ Got MLIR module: {len(str(mlir_module))} chars")

print("\n2. Checking for mosaic_tpu module...")
try:
    # Try importing mosaic_tpu backend
    from jax.experimental.mosaic import tpu as mosaic_tpu
    print("✓ mosaic_tpu imported successfully!")

    # See what's available
    print("\nAvailable functions in mosaic_tpu:")
    functions = [name for name in dir(mosaic_tpu) if not name.startswith('_')]
    for func in functions[:20]:
        print(f"  - {func}")

    # Check for compilation functions
    if hasattr(mosaic_tpu, 'compile'):
        print("\n✓ Found mosaic_tpu.compile!")
    if hasattr(mosaic_tpu, 'lower'):
        print("✓ Found mosaic_tpu.lower!")

except ImportError as e:
    print(f"✗ Cannot import mosaic_tpu: {e}")

print("\n3. Trying to access Pallas TPU backend...")
try:
    from jax._src.pallas.mosaic import pallas_call_registration
    print("✓ Got pallas_call_registration")

    # Try to access the TPU lowering
    from jax._src.pallas.mosaic import lowering as mosaic_lowering
    print("✓ Got mosaic_lowering")

    # See what's available
    print("\nAvailable in mosaic_lowering:")
    items = [name for name in dir(mosaic_lowering) if not name.startswith('_')]
    for item in items[:20]:
        print(f"  - {item}")

except ImportError as e:
    print(f"✗ Cannot import: {e}")

print("\n4. Trying to use jaxlib mosaic_tpu...")
try:
    # Check if jaxlib has mosaic tpu
    import jaxlib
    print(f"jaxlib version: {jaxlib.__version__}")

    # Try to import the C++ mosaic tpu module
    try:
        from jaxlib.mlir._mlir_libs import _tpu_ext
        print("✓ Found _tpu_ext (TPU MLIR extension)!")

        print("\nAvailable in _tpu_ext:")
        items = [name for name in dir(_tpu_ext) if not name.startswith('_')]
        for item in items[:30]:
            obj = getattr(_tpu_ext, item)
            print(f"  - {item}: {type(obj).__name__}")

        # Check for compilation-related functions
        if hasattr(_tpu_ext, 'compile'):
            print("\n✓✓✓ Found _tpu_ext.compile! ✓✓✓")
        if hasattr(_tpu_ext, 'lower'):
            print("✓✓✓ Found _tpu_ext.lower! ✓✓✓")

    except ImportError as e:
        print(f"✗ Cannot import _tpu_ext: {e}")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n5. Exploring libtpu loading...")
try:
    # See if we can force load libtpu
    import ctypes
    import glob

    libtpu_paths = glob.glob("/usr/local/lib/python*/dist-packages/libtpu/libtpu.so")
    if libtpu_paths:
        libtpu_path = libtpu_paths[0]
        print(f"Found libtpu at: {libtpu_path}")

        # Try to load it
        try:
            libtpu = ctypes.CDLL(libtpu_path)
            print("✓ Loaded libtpu.so!")

            # See if we can find compilation functions
            # This is a shot in the dark - looking for symbols
            print("\nNote: libtpu loaded but we need to know the C API")
            print("The TPU compiler is inside libtpu but requires proper initialization")

        except Exception as e:
            print(f"✗ Failed to load libtpu: {e}")
    else:
        print("✗ libtpu.so not found")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n6. Check XLA's TPU compilation path...")
try:
    from jax._src.lib import xla_client

    print("Available XLA platforms:")
    # Try to enumerate platforms
    try:
        # This might fail but let's try
        import jax.lib.xla_bridge as xla_bridge
        print(f"  {xla_bridge.get_backend().platform}")
    except Exception as e:
        print(f"  Error getting platforms: {e}")

    print("\nChecking XLA client capabilities...")
    print(f"  Has register_custom_call_target: {hasattr(xla_client, 'register_custom_call_target')}")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The issue is clear:

1. ✓ We can lower Pallas TPU code to MLIR/StableHLO on CPU
2. ✓ We can get HLO dumps
3. ✗ We CANNOT get LLO/VLIW dumps because:

   - LLO is generated by the TPU backend compiler (inside libtpu)
   - The TPU backend requires physical TPU hardware to initialize
   - Without initialization, we can't access the TPU compiler

The blog post works because they have:
   - Real TPU hardware → libtpu initializes → TPU backend available
   - LIBTPU_INIT_ARGS with --xla_jf_dump_llo_text=true
   - TPU compiler transforms: HLO → LLO → VLIW bundles

Our options:
   A. Get access to real TPU hardware (cloud TPU VM)
   B. Try to hack libtpu to skip hardware detection (difficult)
   C. Use our HLO-based estimation (what we already did)
   D. Explore if there's a TPU simulator/emulator mode in libtpu
""")

print("="*80)
