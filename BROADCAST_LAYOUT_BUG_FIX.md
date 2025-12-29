# Broadcast Layout Bug - Fix Proposal

## Bug Summary

**Issue**: `vector.broadcast` operations fail with "Invalid input layout" when:
- dim0 is a multiple of 128 (no padding)
- dim1 is a multiple of 128 but NOT 128 (e.g., 256, 384)

**Error**:
```
MosaicError: INTERNAL: Mosaic failed to compile TPU kernel: Invalid input layout

The MLIR operation involved:
  %1379 = "vector.broadcast"(%77) {in_layout = [#tpu.vpad<"32,{*,128},(8,128)">], out_layout = [#tpu.vpad<"32,{*,128},(8,128)">]} : (vector<1x128xi32>) -> vector<8x128xi32>
```

## Root Cause Analysis

### Layout Specification
The layout `#tpu.vpad<"32,{*,128},(8,128)">` means:
- bitwidth: 32
- offsets: {*, 128} (replicated in dim 0, **offset 128 in dim 1**)
- tiling: (8, 128)

### Why It's Invalid

For input shape `vector<1x128xi32>` with this layout:

1. Calculate `vregSlice[1]`:
   ```
   tilesPerVreg = vreg_capacity / tile_elems
                = (1024) / (8 * 128)
                = 1
   vregSlice[1] = tilesPerVreg * tiling[1]
                = 1 * 128
                = 128
   ```

2. Validation check (layout.h:506):
   ```cpp
   if (o.has_value() && (*o < 0 || vs <= *o)) {
       return false;  // FAILS: 128 <= 128 is TRUE
   }
   ```

3. **Offset 128 is out of bounds** for valid range [0, 128)

### Why This Offset Is Assigned

When `dim1 = 256`:
1. Array is tiled into 2 tiles of 128 width
2. Second tile has offset 128
3. Operations on the second tile have layouts with offset 128
4. **Layout inference propagates this offset backward to broadcast input**
5. **But for broadcast input of shape `(1, 128)`, offset 128 is invalid**

## The Real Problem

The layout inference pass in the Mosaic compiler does not properly validate or adjust layouts when propagating them backward through broadcast operations.

**Expected behavior**: When propagating a layout backward through a broadcast:
- Adjust offsets to be valid for the input shape
- OR use replicated offsets (*) for dimensions that don't fit
- OR set offset to 0 for the first tile

**Current behavior**: Blindly propagates offset 128 to an input where it's invalid

## Proposed Fix

### Location
The fix should be in the Mosaic layout inference pass (likely in XLA or compiled jaxlib code).

### Fix Strategy

When inferring or propagating layouts for `vector.broadcast` operations:

```cpp
// Pseudo-code for the fix
Layout inferBroadcastInputLayout(BroadcastOp op,  Layout output_layout) {
  auto input_shape = op.getInput().getType().getShape();
  auto output_shape = op.getOutput().getType().getShape();

  Layout input_layout = output_layout;  // Start with output layout

  // Adjust offsets for input shape
  for (int dim = 0; dim < input_layout.rank(); ++dim) {
    if (input_layout.hasOffset(dim)) {
      int64_t offset = input_layout.getOffset(dim);
      int64_t vregSlice = input_layout.getVregSlice(dim, input_shape);

      // If offset is out of bounds for input shape
      if (offset >= vregSlice) {
        // Option 1: Use replicated offset
        input_layout.setOffset(dim, REPLICATED);

        // Option 2: Reset to 0 (first tile)
        // input_layout.setOffset(dim, 0);

        // Option 3: Modulo into valid range
        // input_layout.setOffset(dim, offset % vregSlice);
      }
    }
  }

  return input_layout;
}
```

### Alternative Workarounds

If the layout inference cannot be easily fixed, alternative workarounds:

1. **In JAX lowering** (`jax/_src/pallas/mosaic/lowering.py`):
   - Detect problematic broadcast patterns
   - Decompose into simpler operations
   - Add explicit layout conversion operations

2. **In user code**:
   - Avoid exact multiples of 128 in dimension 1
   - Add padding that changes tiling
   - Restructure to avoid broadcast after slice

## Testing

Test cases to verify the fix:

```python
# Should all pass after fix
test_case(128, 256)  # Currently fails
test_case(128, 384)  # Currently fails
test_case(256, 256)  # Currently fails
test_case(384, 384)  # Currently fails

# Should continue to pass
test_case(128, 128)  # Currently passes
test_case(128, 193)  # Currently passes (not aligned)
test_case(137, 256)  # Currently passes (padded)
```

## Impact

This bug affects any Pallas TPU kernel that:
- Uses broadcast operations
- Works with arrays where dim1 is a multiple of 128 (but not 128)
- Includes common patterns like:
  - `jnp.broadcast_to()`
  - Broadcasting in arithmetic operations
  - Implicit broadcasts in indexing

## Related Code Files

- `jaxlib/mosaic/dialect/tpu/layout.h:502-517` - Layout validation
- `jaxlib/mosaic/dialect/tpu/util.cc:227-248` - `layoutIsValidForValue()`
- `jax/_src/pallas/mosaic/lowering.py:2064-2103` - Broadcast lowering
- Layout inference pass (location TBD - likely in XLA or compiled jaxlib)

## Recommended Action

1. Identify the layout inference pass code (likely in Mosaic compiler)
2. Add special handling for broadcast operations
3. Ensure offsets are validated and adjusted for input shapes
4. Add regression tests for the failing cases

## Temporary User Workaround

Until fixed, users can:
- Avoid shapes where dim1 is exactly a multiple of 128 (except 128 itself)
- Add explicit padding: `jnp.pad(arr, ((0,0), (0,1)))`
- Use dim1 = 127, 129, 255, 257, etc. instead of 128, 256, 384
