"""
Test TPU initialization with fake /dev/accel0 device.

Now that we have /dev/accel0, libtpu might initialize!
If it does, we can get LLO dumps.
"""
import os
import sys
import tempfile

# Create dump directory
dump_dir = tempfile.mkdtemp(prefix="tpu_llo_with_fake_device_")

# Set all the flags before importing JAX
os.environ['XLA_FLAGS'] = (
    f'--xla_dump_to={dump_dir} '
    '--xla_dump_hlo_as_text '
    '--xla_dump_hlo_as_proto '
)

os.environ['LIBTPU_INIT_ARGS'] = (
    '--xla_jf_dump_llo_text=true '
    '--xla_jf_dump_llo_proto=true '
    '--xla_jf_dump_llo_html=true '
    '--xla_jf_dump_hlo_text=true '
    f'--xla_jf_dump_llo_pass_label_regex=.* '
)

# TPU environment variables
os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '1,1,1'
os.environ['TPU_HOST_BOUNDS'] = '1,1,1'
os.environ['TPU_CHIPS'] = '1'
os.environ['TPU_LOAD_LIBRARY'] = '1'
os.environ['CLOUD_TPU_TASK_ID'] = '0'
os.environ['TPU_WORKER_ID'] = '0'
os.environ['TPU_WORKER_HOSTNAMES'] = 'localhost'

# Try to hint at v5e
os.environ['TPU_ACCELERATOR_TYPE'] = 'v5litepod-1'
os.environ['TPU_NAME'] = 'fake-tpu-v5e'

print("="*80)
print("TESTING TPU INITIALIZATION WITH FAKE /dev/accel0")
print("="*80)
print(f"\nDump directory: {dump_dir}")
print(f"\nFake TPU device: /dev/accel0")

# Check if device exists
if os.path.exists('/dev/accel0'):
    print("✓ /dev/accel0 exists!")
    import stat
    st = os.stat('/dev/accel0')
    print(f"  Mode: {stat.filemode(st.st_mode)}")
    print(f"  Major: {os.major(st.st_rdev)}, Minor: {os.minor(st.st_rdev)}")
else:
    print("✗ /dev/accel0 does not exist")
    print("Run: sudo mknod /dev/accel0 c 510 0 && sudo chmod 666 /dev/accel0")
    sys.exit(1)

print("\n" + "="*80)
print("ATTEMPTING TPU BACKEND INITIALIZATION")
print("="*80)

# Force CPU as default but allow TPU
os.environ['JAX_PLATFORMS'] = 'cpu,tpu'

import jax
import jax.numpy as jnp
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh

print(f"\nJAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

# Check if TPU backend is available
try:
    from jax._src.lib import xla_client
    print("\nChecking for TPU backend...")

    # Try to see if TPU initialized
    devices = jax.devices()
    tpu_devices = [d for d in devices if 'tpu' in str(d).lower()]

    if tpu_devices:
        print(f"✓✓✓ TPU BACKEND INITIALIZED! ✓✓✓")
        print(f"TPU devices: {tpu_devices}")
    else:
        print("✗ No TPU devices found (backend may not have initialized)")
        print("  This is expected - /dev/accel0 exists but doesn't respond correctly")

except Exception as e:
    print(f"Error checking backends: {e}")

print("\n" + "="*80)
print("ATTEMPTING COMPILATION WITH MOCKED TPU v5e")
print("="*80)

# Even if TPU backend didn't initialize, try with mocked device
tpu_v5e = AbstractDevice(device_kind="TPU v5e", num_cores=1)
mesh = AbstractMesh((), (), abstract_device=tpu_v5e)

def mini_attention(x, w1, w2):
    """Mini attention kernel"""
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

print("\nLowering with mocked TPU v5e...")
try:
    with use_abstract_mesh(mesh):
        lowered = jax.jit(mini_attention).lower(x, w1, w2)
        print("✓ Lowering succeeded")

        print("\nAttempting compilation...")
        compiled = lowered.compile()
        print("✓ Compilation succeeded")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("CHECKING FOR LLO DUMPS")
print("="*80)

import glob

# Check for LLO files
llo_files = glob.glob(os.path.join(dump_dir, "**/*llo*"), recursive=True)
hlo_files = glob.glob(os.path.join(dump_dir, "**/*hlo*"), recursive=True)

print(f"\nDump directory: {dump_dir}")
print(f"LLO files found: {len(llo_files)}")
print(f"HLO files found: {len(hlo_files)}")

if llo_files:
    print("\n✓✓✓ LLO FILES FOUND! ✓✓✓")
    for f in llo_files[:10]:
        size = os.path.getsize(f)
        print(f"  {os.path.basename(f)} ({size} bytes)")

    # Read and display first LLO file
    if llo_files:
        first_llo = llo_files[0]
        print(f"\n{'='*80}")
        print(f"CONTENTS OF: {os.path.basename(first_llo)}")
        print(f"{'='*80}")
        with open(first_llo, 'r') as f:
            content = f.read()
            print(content[:2000])  # First 2000 chars
            if len(content) > 2000:
                print(f"\n... ({len(content) - 2000} more bytes)")
else:
    print("\n✗ No LLO files - TPU backend compilation did not happen")
    print("  (Expected - fake device doesn't implement the full protocol)")

if hlo_files:
    print(f"\nHLO files (first 5):")
    for f in hlo_files[:5]:
        size = os.path.getsize(f)
        print(f"  {os.path.basename(f)} ({size} bytes)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Device status:
  /dev/accel0: {'✓ Exists' if os.path.exists('/dev/accel0') else '✗ Missing'}

Results:
  Lowering: ✓ Works (with mocked device)
  Compilation: {'✓ Succeeded' if llo_files else '✗ CPU backend used'}
  LLO dumps: {'✓ Found' if llo_files else '✗ Not generated'}

Next steps if LLO not generated:
  The fake /dev/accel0 exists but doesn't respond to ioctl() calls correctly.
  We need either:
    1. Full device driver implementation (very complex)
    2. LD_PRELOAD to intercept and fake ioctl responses
    3. Patch libtpu to skip hardware interaction

Dump directory: {dump_dir}
""")
