# Custom Call Analysis and Implementation Plan

## Overview
Custom calls are JAX's mechanism for invoking platform-specific or external code that isn't expressible in pure StableHLO operations. They're critical for:
- Linear algebra (LAPACK/BLAS on CPU, cuSOLVER on CUDA)
- Random number generation
- Custom Pallas/Mosaic kernels
- FFI (Foreign Function Interface) calls

## Common Custom Call Targets

### CPU (LAPACK)
- `lapack_spotrf` / `lapack_dpotrf` → Cholesky decomposition
- `lapack_sgeqrf` / `lapack_dgeqrf` → QR decomposition
- `lapack_sgesdd` / `lapack_dgesdd` → SVD
- `lapack_ssyev` / `lapack_dsyevd` → Eigenvalue decomposition
- `lapack_sgeev` / `lapack_dgeev` → General eigenvalue decomposition
- `lapack_sgtsv` / `lapack_dgtsv` → Tridiagonal solve
- `lapack_strsm` / `lapack_dtrsm` → Triangular solve

### GPU (cuSOLVER/cuBLAS)
- `cusolver_potrf` → Cholesky (GPU)
- `cusolver_geqrf` → QR (GPU)
- `cusolver_gesvd` / `cusolver_gesvdj` → SVD (GPU)
- `cusolver_syev` / `cusolver_syevd` → Eigh (GPU)
- `cusolver_getrf` → LU decomposition (GPU)
- `cuda_threefry2x32` → Random number generation (GPU)
- `lu_pivots_to_permutation` → Convert LU pivots

### TPU
- `Sharding` → TPU sharding operations
- `Eigh`, `Lu`, `Qr` → Linear algebra primitives
- `ApproxTopK` → Approximate top-k

### Pallas/Mosaic
- `__gpu$...` → Mosaic GPU kernels
- `tpu_custom_call` → TPU custom operations
- Triton kernels compiled to custom calls

## Current Status

**Not Implemented** - The decompiler currently has no custom_call support. When it encounters a custom_call operation, it will fail with "unsupported operation" error.

## Implementation Strategy

### Phase 1: Basic Custom Call Detection
```python
elif op_name_str == 'stablehlo.custom_call':
    call_target = str(attrs.get('call_target_name', ''))
    print(f"Warning: Encountered custom_call to '{call_target}'")
    print(f"  Custom calls are platform-specific and may not decompile correctly")

    # For now, just pass through as a warning
    # Return a placeholder or try to map to known targets
    result = None
```

### Phase 2: Map Common Linear Algebra Calls
```python
# Map LAPACK/cuSOLVER calls to lax.linalg operations
LINALG_CUSTOM_CALLS = {
    'lapack_spotrf': lambda *args: lax.linalg.cholesky(args[0]),
    'lapack_dpotrf': lambda *args: lax.linalg.cholesky(args[0]),
    'cusolver_potrf': lambda *args: lax.linalg.cholesky(args[0]),

    'lapack_sgeqrf': lambda *args: lax.linalg.qr(args[0]),
    'lapack_dgeqrf': lambda *args: lax.linalg.qr(args[0]),
    'cusolver_geqrf': lambda *args: lax.linalg.qr(args[0]),

    'lapack_sgesdd': lambda *args: lax.linalg.svd(args[0], full_matrices=False),
    'lapack_dgesdd': lambda *args: lax.linalg.svd(args[0], full_matrices=False),
    'cusolver_gesvd': lambda *args: lax.linalg.svd(args[0]),

    'lapack_ssyev': lambda *args: lax.linalg.eigh(args[0]),
    'lapack_dsyev': lambda *args: lax.linalg.eigh(args[0]),
    'cusolver_syevd': lambda *args: lax.linalg.eigh(args[0]),

    'cusolver_getrf': lambda *args: lax.linalg.lu(args[0]),
}

elif op_name_str == 'stablehlo.custom_call':
    call_target = str(attrs.get('call_target_name', ''))

    if call_target in LINALG_CUSTOM_CALLS:
        result = LINALG_CUSTOM_CALLS[call_target](*operands)
    else:
        # Unknown custom call
        raise NotImplementedError(f"Custom call '{call_target}' not supported")
```

### Phase 3: Handle Custom Call Attributes
Custom calls have complex attributes that need parsing:
- `operand_layouts` - Layout of input arrays
- `result_layouts` - Layout of output arrays
- `api_version` - Custom call API version
- `backend_config` - Platform-specific configuration (often a serialized protobuf)

```python
# Parse backend_config for additional metadata
backend_config = attrs.get('backend_config', '')
# May need to decode/deserialize this
```

### Phase 4: Pallas Kernel Handling
Pallas kernels are custom calls with kernel code embedded. These can't be directly decompiled, but we can:
1. Extract the kernel signature
2. Create a placeholder function
3. Warn the user that kernel semantics are lost

```python
if call_target.startswith('__gpu$') or 'triton' in call_target:
    # Pallas/Triton kernel
    print(f"Warning: Pallas kernel '{call_target}' cannot be fully decompiled")
    # Return shape-preserving placeholder
    result = operands[0]  # or create zeros with correct shape
```

## Challenges

### 1. Platform Dependencies
Custom calls are platform-specific. A function compiled for GPU won't decompile correctly for CPU if it uses cuSOLVER.

**Solution**:
- Detect platform from call target
- Map to platform-agnostic lax operations where possible
- Warn when decompiled code may not match original platform

### 2. Multiple Return Values
Many custom calls return tuples (e.g., LU returns `(lu, pivots)`):
```mlir
%lu, %pivots, %info = stablehlo.custom_call @cusolver_getrf(...)
```

**Solution**:
- Parse the result tuple structure from MLIR types
- Return appropriate tuple from mapped lax operation
- Handle result indexing (e.g., `%lu` vs `%pivots`)

### 3. Backend Config Parsing
Backend config contains critical information but is often a serialized protobuf:
```mlir
backend_config = "\n\x02\x10\x00\x12\x04\x02\x01\x02\x01"
```

**Solution**:
- For common operations, reverse-engineer the config format
- For unknown formats, use default parameters
- Log warnings when config is ignored

### 4. Side Effects and Ordering
Custom calls may have side effects (e.g., writing to external memory). StableHLO uses tokens to order these.

**Solution**:
- Track token threading through custom calls
- Preserve call ordering in decompiled code
- May need to use `lax.stop_gradient` or explicit sequencing

## Testing Strategy

### Unit Tests
```python
def test_custom_call_cholesky():
    """Test that Cholesky custom_call decompiles correctly."""
    def f(x):
        return jnp.linalg.cholesky(x)

    x = jnp.array([[4.0, 2.0], [2.0, 3.0]])
    # This should compile to lapack_spotrf on CPU
    # Decompile and verify it maps back to lax.linalg.cholesky
```

### Integration Tests
Test with real workloads:
- Neural network inference
- Scientific computing libraries
- Custom Pallas kernels

### Edge Cases
- Custom calls with no operands
- Custom calls with only token operands
- Nested custom calls
- Custom calls in control flow

## Recommended Implementation

### Minimal Viable Implementation
```python
elif op_name_str == 'stablehlo.custom_call':
    call_target = str(attrs.get('call_target_name', '')).strip('"')

    # Map common linear algebra operations
    if 'potrf' in call_target:  # Cholesky
        result = jnp.linalg.cholesky(operands[0])
    elif 'geqrf' in call_target:  # QR
        result = jnp.linalg.qr(operands[0])
    elif 'gesdd' in call_target or 'gesvd' in call_target:  # SVD
        result = jnp.linalg.svd(operands[0], full_matrices=False)
    elif 'syev' in call_target or 'syevd' in call_target:  # Eigh
        result = jnp.linalg.eigh(operands[0])
    elif 'getrf' in call_target:  # LU
        # LU returns tuple, need to handle multi-value result
        result = jnp.linalg.lu(operands[0])
    elif 'threefry' in call_target:  # RNG
        # Can't decompile RNG, use placeholder
        print(f"Warning: RNG custom_call cannot be decompiled exactly")
        result = operands[0]  # Return key as-is
    else:
        # Unknown custom call
        print(f"Warning: Unknown custom_call '{call_target}'")
        # Try to infer shape from MLIR and return zeros
        if hasattr(op, 'results') and len(list(op.results)) > 0:
            shape, dtype = self._parse_tensor_type(list(op.results)[0].type)
            result = jnp.zeros(shape, dtype=dtype)
        else:
            result = operands[0] if operands else None
```

## Conclusion

Custom call support is **critical for production usage** but **complex to implement fully**. The recommended approach is:

1. ✅ **Implement basic detection and warnings** (easy)
2. ✅ **Map common linear algebra operations** (medium)
3. ⚠️  **Handle Pallas kernels with placeholders** (medium)
4. ⚠️  **Parse backend configs** (hard)
5. ⚠️  **Handle all edge cases** (very hard)

For a 90% solution, implementing steps 1-2 would cover the most common cases (linear algebra on CPU/GPU). Steps 3-5 require deep platform knowledge and may have diminishing returns.

## Priority for Implementation
**HIGH** - Linear algebra custom calls (Cholesky, QR, SVD, Eigh, LU)
**MEDIUM** - RNG custom calls (with appropriate warnings)
**LOW** - Pallas/Mosaic kernels (just warn and use placeholders)
**OPTIONAL** - Backend config parsing, token handling
