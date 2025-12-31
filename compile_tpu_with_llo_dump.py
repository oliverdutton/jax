"""
Attempt to compile TPU code and dump LLO/VLIW bundles using XLA compiler directly.

Strategy: Use XLA's compilation API instead of device-based execution.
This might allow us to compile for TPU target without needing physical hardware.
"""
import os
import tempfile

# Set XLA flags BEFORE importing JAX
dump_dir = tempfile.mkdtemp(prefix="tpu_llo_dump_")
os.environ['XLA_FLAGS'] = (
    f'--xla_dump_to={dump_dir} '
    '--xla_dump_hlo_as_text '
    '--xla_dump_hlo_as_proto '
    '--xla_dump_hlo_snapshots '
)

# Try to set LIBTPU flags (though these may not work without device)
os.environ['LIBTPU_INIT_ARGS'] = (
    '--xla_jf_dump_llo_text=true '
    '--xla_jf_dump_llo_proto=true '
    '--xla_jf_dump_llo_html=true '
    '--xla_jf_dump_hlo_text=true '
)

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TPU_LOAD_LIBRARY'] = '1'

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh
from jax._src.lib import xla_client

print("="*80)
print("ATTEMPTING TO COMPILE TPU CODE AND DUMP LLO/VLIW BUNDLES")
print("="*80)
print(f"\nDump directory: {dump_dir}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'Not set')}")
print(f"LIBTPU_INIT_ARGS: {os.environ.get('LIBTPU_INIT_ARGS', 'Not set')}")

# Mini-attention kernel (same as blog post)
def mini_attention(x, w1, w2):
    """Mini attention: matmul → rms_norm → softmax → matmul"""
    # matmul_1
    h = x @ w1

    # rms_norm
    rms = jnp.sqrt(jnp.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
    h = h / rms

    # softmax
    h_max = jnp.max(h, axis=-1, keepdims=True)
    exp_h = jnp.exp(h - h_max)
    h = exp_h / jnp.sum(exp_h, axis=-1, keepdims=True)

    # matmul_2
    out = h @ w2

    return out


# Create inputs (same shapes as blog post)
batch, d_in, d_mid, d_out = 16, 64, 64, 32
print(f"\nInput shapes: x=[{batch},{d_in}], w1=[{d_in},{d_mid}], w2=[{d_mid},{d_out}]")

key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)

x = jax.random.normal(k1, (batch, d_in))
w1 = jax.random.normal(k2, (d_in, d_mid)) * 0.02
w2 = jax.random.normal(k3, (d_mid, d_out)) * 0.02

print("\n" + "="*80)
print("APPROACH 1: Try lowering with TPU device mock and check for LLO dumps")
print("="*80)

# Mock TPU v5e device
tpu_v5e = AbstractDevice(device_kind="TPU v5e", num_cores=1)
mesh = AbstractMesh((), (), abstract_device=tpu_v5e)

try:
    with use_abstract_mesh(mesh):
        print("\nLowering with mocked TPU v5e...")
        jitted_fn = jax.jit(mini_attention)
        lowered = jitted_fn.lower(x, w1, w2)
        print("✓ Lowering succeeded")

        # Get compiler options
        print("\nAttempting to extract compiler backend info...")
        try:
            compile_options = lowered.compile_args
            print(f"Compile args: {compile_options}")
        except Exception as e:
            print(f"Cannot access compile args: {e}")

        # Try to compile
        print("\nAttempting compilation...")
        try:
            compiled = lowered.compile()
            print("✓ Compilation succeeded")

            # Check if LLO dumps were created
            import glob
            llo_files = glob.glob(os.path.join(dump_dir, "*.llo*"))
            hlo_files = glob.glob(os.path.join(dump_dir, "*.hlo*"))

            print(f"\nFiles in dump directory:")
            print(f"  LLO files: {len(llo_files)}")
            print(f"  HLO files: {len(hlo_files)}")

            if llo_files:
                print("\n✓✓✓ LLO FILES FOUND! ✓✓✓")
                for f in llo_files[:5]:
                    print(f"  - {os.path.basename(f)}")
            else:
                print("\n✗ No LLO files found (expected - no TPU backend)")

            if hlo_files:
                print("\nHLO files found:")
                for f in hlo_files[:5]:
                    print(f"  - {os.path.basename(f)}")

        except Exception as e:
            print(f"✗ Compilation failed: {e}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("APPROACH 2: Try to use XLA compilation API directly")
print("="*80)

try:
    # Get the HloModule from lowered representation
    print("\nExtracting HLO module...")

    with use_abstract_mesh(mesh):
        lowered = jax.jit(mini_attention).lower(x, w1, w2)

        # Try to access internal HLO representation
        try:
            # Get MLIR module
            mlir_module = lowered.compiler_ir()
            print(f"✓ Got MLIR module: {len(str(mlir_module))} chars")

            # Save MLIR for inspection
            mlir_path = os.path.join(dump_dir, "mini_attention.mlir")
            with open(mlir_path, 'w') as f:
                f.write(str(mlir_module))
            print(f"✓ Saved MLIR to: {mlir_path}")

        except Exception as e:
            print(f"Cannot access MLIR: {e}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("APPROACH 3: Check if we can invoke TPU compiler toolchain directly")
print("="*80)

try:
    # Check if we can access TPU compilation without device
    from jax._src.interpreters import mlir
    from jax._src.lib import mlir as mlir_lib

    print("\nChecking available XLA compilation backends...")

    # Try to see what compilation options are available
    print("Attempting to access XLA compiler API...")

    # Check if we can create a TPU compiler client
    try:
        # This is a hack - try to see if we can get the TPU compiler
        import jaxlib
        print(f"jaxlib version: {jaxlib.__version__}")

        # Check for TPU-specific modules
        try:
            from jaxlib import mosaic_tpu
            print("✓ mosaic_tpu module available")

            # See if we can access compiler functions
            if hasattr(mosaic_tpu, 'compile'):
                print("✓ mosaic_tpu.compile function found!")
        except ImportError:
            print("✗ mosaic_tpu not available")

    except Exception as e:
        print(f"Cannot access compiler: {e}")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nDump directory: {dump_dir}")
print("\nTo check for dumps:")
print(f"  ls -la {dump_dir}/")
print(f"  find {dump_dir} -type f")

# List all files in dump directory
import glob
all_files = glob.glob(os.path.join(dump_dir, "**/*"), recursive=True)
if all_files:
    print(f"\nFound {len(all_files)} files in dump directory:")
    for f in all_files[:20]:
        if os.path.isfile(f):
            size = os.path.getsize(f)
            print(f"  {os.path.basename(f)} ({size} bytes)")
else:
    print("\n✗ No files found in dump directory")

print("\n" + "="*80)
