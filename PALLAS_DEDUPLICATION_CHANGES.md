# Pallas Deduplication Pass Implementation

## Summary

Added CSE (Common Subexpression Elimination) and canonicalize passes to the Pallas compilation pipeline for both TPU (Mosaic) and GPU (Mosaic GPU) backends. This ensures that duplicate computations are eliminated during compilation, similar to normal JAX compilation.

## Problem

Previously, Pallas compilation did not include deduplication passes in the MLIR optimization pipeline. This meant that duplicate computations (e.g., from rematerialization or repeated operations) were not being eliminated before serialization, potentially leading to inefficient compiled code.

Normal JAX compilation includes HLO CSE passes in the XLA compiler pipeline (as seen in `xla/service/gpu/gpu_compiler.cc`), but Pallas was only running serialization passes without optimization.

## Solution

### Changes Made

1. **TPU/Mosaic Backend** (`jax/_src/tpu_custom_call.py`):
   - Modified `_lower_mosaic_module_to_asm()` to run CSE and canonicalize passes before the mosaic-serde serialization pass
   - Added optimization pipeline: `"builtin.module(cse,canonicalize)"`

2. **GPU/Mosaic GPU Backend** (`jax/experimental/mosaic/gpu/core.py`):
   - Modified `_run_serde_pass()` to run CSE and canonicalize passes before the mosaic_gpu-serde serialization pass
   - Added optimization pipeline: `"builtin.module(cse,canonicalize)"`

### Technical Details

#### MLIR Passes Applied

- **CSE (Common Subexpression Elimination)**: Eliminates duplicate computations by identifying expressions with the same operands and replacing them with a single computation
- **Canonicalize**: Simplifies and normalizes IR patterns, includes some CSE-like optimizations and simplifications (e.g., `x + 0 -> x`, `x * 1 -> x`)

#### Compilation Pipeline

**Before:**
```
JAXPR -> MLIR Mosaic -> mosaic-serde -> Serialized Module -> XLA Backend
```

**After:**
```
JAXPR -> MLIR Mosaic -> CSE -> Canonicalize -> mosaic-serde -> Serialized Module -> XLA Backend
```

## Impact

- **Performance**: Reduced redundant computations in compiled kernels
- **Code Size**: Smaller compiled kernels due to deduplication
- **Compatibility**: Aligns Pallas compilation behavior with normal JAX compilation
- **Breaking Changes**: None - this is a pure optimization that doesn't change semantics

## Testing

Created `test_pallas_deduplication.py` with tests to verify:
- CSE pass eliminates duplicate computations
- Canonicalize pass simplifies expressions
- Multiple duplicate operations are correctly deduplicated

## References

- XLA HLO CSE implementation: `xla/service/hlo_cse.h`, `xla/service/hlo_cse.cc`
- XLA GPU compiler pipeline: `xla/service/gpu/gpu_compiler.cc` (lines 760, 1791, 1911)
- MLIR standard passes documentation

## Future Work

- Consider adding additional optimization passes (e.g., DCE - Dead Code Elimination)
- Add performance benchmarks to measure the impact of deduplication
- Investigate if other Pallas backends (e.g., Triton) need similar changes
