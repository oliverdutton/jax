# Broadcast Layout Bug Analysis

## Problem Summary

When broadcasting from `vector<1x128xi32>` to `vector<8x128xi32>`, the Mosaic compiler assigns an invalid layout to the broadcast input:

```mlir
%1379 = "vector.broadcast"(%77) {
  in_layout = [#tpu.vpad<"32,{*,128},(8,128)">],
  out_layout = [#tpu.vpad<"32,{*,128},(8,128)">]
} : (vector<1x128xi32>) -> (vector<8x128xi32>)
```

This fails with:
```
MosaicError: INTERNAL: Mosaic failed to compile TPU kernel: Invalid input layout
```

## Root Cause

The layout `#tpu.vpad<"32,{*,128},(8,128)">` means:
- bitwidth: 32
- offsets: {*, 128} (replicated in dim 0, offset 128 in dim 1)
- tiling: (8, 128)

For an input vector of shape `(1, 128)`, this layout is **invalid** because:

1. The `vregSlice` for dimension 1 is calculated as:
   - `tilesPerVreg = vreg_capacity / tile_elems = 1024 / 1024 = 1`
   - `vregSlice[1] = tilesPerVreg * tiling[1] = 1 * 128 = 128`

2. The validation check in `layout.h:506` is:
   ```cpp
   if (o.has_value() && (*o < 0 || vs <= *o)) {
       return false;
   }
   ```

3. With offset[1] = 128 and vregSlice[1] = 128:
   - `128 <= 128` → TRUE, so validation FAILS

## Why This Only Happens for Specific Shapes

The bug only occurs when:
- dim0 is a multiple of 128 (no padding)
- dim1 is a multiple of 128 but NOT 128 (e.g., 256, 384)

This is because:
- When dim1 = 256, it's split into 2 tiles of 128
- The second tile has offset 128
- This offset gets propagated back to the broadcast input
- For smaller dim1 values (like 193), padding is added, which changes the layout inference

## The Offset 128 Origin

The offset 128 comes from the fact that when processing `take_along_axis` with a shape like `(n, 256)`:
1. The array is tiled into two 128-wide tiles
2. The second tile has offset 128 in dimension 1
3. This layout is used for intermediate operations
4. The layout inference pass propagates this backward to the broadcast input
5. But for the broadcast input of shape `(1, 128)`, offset 128 is out of bounds

## Solution Options

### Option 1: Fix Layout Inference Pass (Preferred)
Modify the Mosaic layout inference to ensure that for `vector.broadcast` operations:
- Input layouts should have offsets adjusted for the input shape
- For dimensions being broadcast (size changes), use replicated offset (*)
- Never assign an offset that exceeds the input's vregSlice

### Option 2: Fix JAX Lowering
In `jax/_src/pallas/mosaic/lowering.py`, explicitly set layout attributes on broadcast operations:
- Calculate appropriate input layout based on input shape
- Ensure offsets are valid for input dimensions

### Option 3: Insert Layout Conversion
Add explicit layout conversion operations before/after broadcasts to ensure compatibility.

## Recommended Fix

The fix should be in the Mosaic layout inference pass. The pass should check:
1. When propagating layouts backward through a broadcast:
   - Adjust offsets to be valid for the input shape
   - Or use replicated offsets for broadcast dimensions
2. Validate that assigned layouts are valid for the operation's input shapes

## Temporary Workaround

Users can avoid this bug by:
- Using shapes that aren't exact multiples of 128 in the second dimension
- Adding explicit padding that changes the tiling pattern
- Restructuring code to avoid broadcast followed by take_along_axis on aligned shapes
