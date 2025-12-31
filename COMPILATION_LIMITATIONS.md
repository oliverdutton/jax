# TPU Code Compilation on CPU: Limitations and Analysis

## Summary

**✅ What Works:**
- Lowering TPU Pallas code on CPU backend (with our fix)
- Using mocked TPU devices to target specific TPU versions
- Extracting and inspecting generated StableHLO IR

**❌ What Doesn't Work:**
- Full compilation to executable code on CPU
- Running compiled TPU code on CPU (without interpret mode)

## Why Compilation Fails

### The Technical Issue

When we try to compile lowered TPU code on CPU, we get:
```
JaxRuntimeError: NOT_FOUND: No registered implementation for
untyped custom call to tpu_custom_call for Host
```

This error occurs because:

1. **Lowering Phase** (✅ Works)
   - Converts Pallas code → StableHLO IR
   - Generates `tpu_custom_call` operations
   - Uses mocked TPU device info (v5e, v5p, etc.)
   - **Our fix enables this on CPU**

2. **Compilation Phase** (❌ Fails)
   - Converts StableHLO IR → Executable code
   - Needs backend-specific implementation of `tpu_custom_call`
   - CPU backend doesn't have `tpu_custom_call` handler
   - **This is where it fails**

### Why libtpu Doesn't Help on CPU

We have libtpu installed (`/usr/local/lib/python3.11/dist-packages/libtpu/libtpu.so`),
but it cannot initialize without actual TPU hardware:

```python
>>> xla_bridge.get_backend('tpu')
RuntimeError: Backend 'tpu' failed to initialize:
UNKNOWN: TPU initialization failed: No jellyfish device found
```

**Why this happens:**
- libtpu requires physical TPU hardware (PCIe device)
- It looks for "jellyfish" device (TPU hardware identifier)
- Without hardware, the TPU backend cannot initialize
- Without backend initialization, `tpu_custom_call` isn't registered

### The Chicken-and-Egg Problem

```
Need TPU backend   →   Requires libtpu init   →   Needs TPU hardware
      ↑                                                    ↓
      └────────────────────────────────────────────────────┘
                    We don't have hardware!
```

## What's Actually Possible

### ✅ Lowering + IR Inspection (Current State)

```python
from jax._src.mesh import AbstractDevice, AbstractMesh, use_abstract_mesh

# Mock a TPU v5e device
tpu_v5e = AbstractDevice(device_kind="TPU v5e", num_cores=1)
mesh = AbstractMesh((), (), abstract_device=tpu_v5e)

with use_abstract_mesh(mesh):
    lowered = jax.jit(pallas_fn).lower(x, y)
    ir = lowered.as_text()  # ✓ Works!
    # compiled = lowered.compile()  # ✗ Fails
```

**Use cases:**
- Inspect generated IR for different TPU versions
- Debug kernel lowering issues
- Understand TPU compilation pipeline
- Test lowering logic without hardware

### ✅ Execution via Interpret Mode

```python
# Use interpret mode to run TPU kernels on CPU
result = pl.pallas_call(
    kernel,
    ...,
    backend="mosaic_tpu",
    interpret=True,  # Simulates TPU on CPU
)(x, y)
```

**How it works:**
- Doesn't use compilation
- Simulates TPU memory (VMEM, HBM, etc.)
- Emulates DMA operations
- Runs kernel logic on CPU
- **This actually executes the kernel!**

### ❌ What Would Be Needed for Compilation

To enable full compilation on CPU, we would need ONE of:

**Option 1: Stub TPU Runtime (Complex)**
- Implement `tpu_custom_call` handler in C/C++
- Register it for CPU backend
- Handle all TPU operations (VMEM, DMA, etc.)
- Essentially reimplementing the TPU runtime on CPU
- **Very complex, not practical**

**Option 2: Offline Compilation Mode (Would need XLA changes)**
- Modify XLA to allow compilation without runtime
- Generate code that can't execute but validates
- Useful for testing compiler pipeline
- **Would require changes to XLA/libtpu**

**Option 3: Actual TPU Hardware**
- Use real TPU (v4, v5e, v5p, etc.)
- libtpu initializes properly
- Full compilation and execution work
- **The intended use case**

## Workarounds and Best Practices

### For Development/Testing

1. **Use interpret mode for functionality testing:**
   ```python
   pl.pallas_call(..., interpret=True)
   ```

2. **Use lowering for IR inspection:**
   ```python
   lowered = jax.jit(fn).lower(x, y)
   print(lowered.as_text())
   ```

3. **Use mocked devices to test different TPU versions:**
   ```python
   for tpu_type in ["TPU v5e", "TPU v5p", "TPU v4"]:
       device = AbstractDevice(device_kind=tpu_type, ...)
       # ... test lowering for this TPU type
   ```

### For Production

1. **Deploy to actual TPU hardware**
2. **Use JAX's Cloud TPU integration**
3. **Test with interpret mode locally, verify on TPU**

## Comparison with GPU

**Why does GPU work differently?**

JAX has mock GPU support because:
- GPU backends (CUDA) can run on CPU in simulator mode for some ops
- JAX provides `jax_mock_gpu_topology` config
- GPU custom calls have CPU fallbacks for some operations

TPU is different because:
- TPU hardware is more specialized
- No CPU fallback for TPU-specific operations
- libtpu requires actual hardware for initialization

## References

- JAX Pallas documentation: https://jax.readthedocs.io/en/latest/pallas/index.html
- TPU device types: `jax/_src/pallas/mosaic/tpu_info.py`
- Interpret mode: `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py`
- Our fix: `jax/_src/pallas/pallas_call.py:1119-1126`

## Conclusion

Our fix successfully enables:
- ✅ **Lowering** TPU code on CPU
- ✅ **IR inspection** for all TPU versions
- ✅ **Development/testing** workflow improvements

But fundamental limitations prevent:
- ❌ **Full compilation** without TPU hardware
- ❌ **Execution** of compiled code on CPU

**For execution on CPU, use `interpret=True` instead.**
