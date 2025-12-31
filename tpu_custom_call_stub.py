"""
Stub implementation of tpu_custom_call for CPU backend.

This allows TPU Pallas code to compile on CPU by registering a stub
handler for tpu_custom_call. The stub doesn't actually execute TPU code,
but allows compilation to succeed for testing/inspection purposes.
"""
import ctypes
import jax
from jax._src.lib import xla_client


def create_tpu_custom_call_stub():
  """
  Create and register a stub for tpu_custom_call on CPU backend.

  This allows compilation to succeed, but the compiled code won't
  actually execute correctly - it's meant for IR inspection and
  testing the compilation pipeline, not for execution.
  """

  # The custom call signature for XLA custom call API v0 (untyped)
  # void custom_call(void* out, const void** in, const char* opaque, size_t opaque_len)

  def tpu_custom_call_stub_fn(out_ptr, in_ptrs, opaque_ptr, opaque_len):
    """
    Stub function that would be called for TPU custom calls on CPU.

    This is a no-op stub - it doesn't actually execute TPU code.
    In a real scenario, this would need to parse the opaque data
    and execute the TPU kernel, but that requires the full TPU runtime.
    """
    # For now, just print a message if this ever gets called
    print("WARNING: tpu_custom_call stub was invoked - this is a no-op!")
    print("  This code cannot execute without a real TPU backend.")
    # Don't touch the output - leave it uninitialized
    # In practice, this stub is only for compilation, not execution
    return None

  try:
    # Try to create a C function pointer (PyCapsule) for the stub
    # Note: This is a simplified version - a real implementation would
    # need proper C/C++ code compiled and wrapped

    # For Python-based registration, we need to check if jaxlib supports it
    # Most custom calls need C/C++ implementations, but let's try

    # First check if we already have a registration
    try:
      # Try to register with a None handler - this will fail gracefully
      # and tell us if registration is supported
      xla_client.register_custom_call_target(
          "tpu_custom_call",
          None,  # Will fail, but we're just checking
          "cpu",
          api_version=0  # Untyped custom call API
      )
    except Exception as e:
      print(f"Note: Cannot register Python-based custom call handler: {e}")
      print("\nTo enable compilation on CPU, you would need to:")
      print("1. Implement tpu_custom_call in C/C++")
      print("2. Compile it as a shared library")
      print("3. Register it using PyCapsule from the shared library")
      print("\nAlternatively, use interpret mode for execution:")
      print("  pl.pallas_call(..., interpret=True)")
      return False

  except Exception as e:
    print(f"Error attempting to register stub: {e}")
    return False

  return True


def check_custom_call_registration():
  """Check if tpu_custom_call is registered for CPU."""
  try:
    # Try to get information about registered custom calls
    # This is platform-specific
    backend = xla_client.get_backend("cpu")
    print(f"CPU backend: {backend}")
    print(f"Backend platform: {backend.platform}")

    # Unfortunately, there's no easy way to query registered custom calls
    # But we can try to compile something and see if it fails

    return True
  except Exception as e:
    print(f"Error checking registration: {e}")
    return False


if __name__ == "__main__":
  print("="*80)
  print("TPU Custom Call Stub Registration Attempt")
  print("="*80)
  print()

  print("Checking CPU backend...")
  check_custom_call_registration()

  print("\nAttempting to register stub...")
  success = create_tpu_custom_call_stub()

  if success:
    print("\n✓ Stub registered successfully!")
  else:
    print("\n✗ Stub registration not possible with pure Python")
    print("\nWhy compilation fails:")
    print("-" * 80)
    print("The issue is that tpu_custom_call is an 'untyped' custom call that")
    print("requires a C/C++ implementation. Python-based handlers aren't supported")
    print("for this API version.")
    print()
    print("The TPU runtime (libtpu) provides the real implementation, but on CPU")
    print("we don't have access to it.")
    print()
    print("Options:")
    print("  1. Use interpret mode: interpret=True (works on CPU)")
    print("  2. Use actual TPU hardware (requires TPU backend)")
    print("  3. Create C++ stub (complex, requires building native extension)")
    print("  4. Stick with lowering only (IR inspection, no execution)")
    print("=" * 80)
