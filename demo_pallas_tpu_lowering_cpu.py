"""
Demo: Lowering Pallas TPU code on CPU backend with mocked TPU devices

This demonstrates:
1. Lowering TPU Pallas code on CPU (previously blocked)
2. Using mocked TPU devices to target specific TPU versions (v5e, v5p, etc.)
3. Extracting and inspecting the generated StableHLO IR
4. Understanding compilation limitations (requires actual TPU backend)
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh

# Force CPU platform
jax.config.update('jax_platforms', 'cpu')

def add_kernel(x_ref, y_ref, o_ref):
  """Simple Pallas kernel that adds two arrays."""
  o_ref[...] = x_ref[...] + y_ref[...]

def demo_tpu_lowering(tpu_version: str = "TPU v5e", num_cores: int = 1):
  """
  Demo lowering TPU Pallas code on CPU for a specific TPU version.

  Args:
    tpu_version: TPU version string (e.g., "TPU v5e", "TPU v5p", "TPU v4")
    num_cores: Number of cores (1 for lite versions like v5e, v6e)
  """
  print(f"\n{'='*80}")
  print(f"Demo: Lowering Pallas TPU code for {tpu_version} on CPU backend")
  print(f"{'='*80}\n")

  # Create mocked TPU device
  tpu_device = AbstractDevice(device_kind=tpu_version, num_cores=num_cores)
  abstract_mesh = AbstractMesh((), (), abstract_device=tpu_device)

  print(f"1. Created mocked TPU device:")
  print(f"   Device: {tpu_device.device_kind}")
  print(f"   Cores: {tpu_device.num_cores}")

  # Create input data
  x = jnp.ones((8, 128), dtype=jnp.float32)
  y = jnp.ones((8, 128), dtype=jnp.float32)

  # Define Pallas function
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
        interpret=False,  # Use lowering, not interpretation
    )(x, y)

  # Lower with mocked TPU device
  print(f"\n2. Lowering code...")
  with use_abstract_mesh(abstract_mesh):
    lowered = jax.jit(add_matrices).lower(x, y)

    # Verify device was detected correctly
    from jax._src.pallas.mosaic import core as pallas_core
    detected_kind = pallas_core.get_device_kind()
    print(f"   ✓ Lowering succeeded!")
    print(f"   Detected device: {detected_kind}")

  # Extract and display IR
  print(f"\n3. Extracting StableHLO IR...")
  ir_text = lowered.as_text()
  print(f"   IR size: {len(ir_text)} characters")

  # Find the custom call in the IR
  for line in ir_text.split('\n'):
    if 'stablehlo.custom_call @tpu_custom_call' in line:
      print(f"   ✓ Found TPU custom call in IR")
      # Extract some info from the backend_config
      if 'backend_config' in line:
        print(f"   ✓ Contains TPU-specific backend configuration")
      break

  # Show first few lines of IR
  print(f"\n4. IR Preview (first 10 lines):")
  print("   " + "-" * 76)
  for i, line in enumerate(ir_text.split('\n')[:10], 1):
    print(f"   {line}")
  print("   " + "-" * 76)

  # Try compilation (will fail, but demonstrates the limitation)
  print(f"\n5. Compilation attempt:")
  try:
    compiled = lowered.compile()
    print(f"   ✓✓✓ Compilation succeeded! (unexpected)")
    return True
  except Exception as e:
    if "tpu_custom_call" in str(e):
      print(f"   ✗ Compilation failed (expected)")
      print(f"   Reason: CPU backend doesn't support tpu_custom_call")
      print(f"   Note: Compilation requires actual TPU hardware/backend")
      return False
    else:
      print(f"   ✗ Unexpected error: {e}")
      return False

def main():
  """Run demos for different TPU versions."""
  print("\n" + "="*80)
  print("PALLAS TPU LOWERING ON CPU DEMO")
  print("="*80)
  print("\nThis demonstrates the ability to lower (but not compile) Pallas TPU")
  print("code on CPU backend with mocked TPU device configurations.")
  print("\nUse cases:")
  print("  • Inspect generated IR without TPU hardware")
  print("  • Test lowering pipeline for different TPU versions")
  print("  • Debug Pallas kernel code structure")
  print("  • Develop/test on CPU before deploying to TPU")

  # Demo different TPU versions
  tpu_configs = [
      ("TPU v5e", 1),  # Lite version
      ("TPU v5p", 2),  # Performance version (2 cores per chip)
      ("TPU v4", 2),   # Previous generation
  ]

  results = []
  for tpu_version, num_cores in tpu_configs:
    success = demo_tpu_lowering(tpu_version, num_cores)
    results.append((tpu_version, success))

  # Summary
  print(f"\n{'='*80}")
  print("SUMMARY")
  print(f"{'='*80}\n")

  for tpu_version, success in results:
    status = "✓ Lowering works" if not success else "✓ Full compilation works"
    print(f"  {tpu_version:15s}: {status}")

  print(f"\n{'='*80}")
  print("KEY FINDINGS:")
  print(f"{'='*80}")
  print("  ✓ TPU code can be lowered on CPU backend (with our fix)")
  print("  ✓ Mocked TPU devices allow targeting specific TPU versions")
  print("  ✓ StableHLO IR can be extracted and inspected")
  print("  ✗ Full compilation requires actual TPU backend (libtpu)")
  print(f"{'='*80}\n")

if __name__ == "__main__":
  main()
