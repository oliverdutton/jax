# Practical Workarounds for Broadcast Layout Bug

## Summary

While waiting for a fix in the Mosaic compiler, you can work around the broadcast layout bug using one of these strategies:

## Workaround 1: Reshape Pattern (Recommended)

**Based on**: `test_retiling_with_replicated_lane` in `tests/pallas/tpu_ops_test.py:702`

This workaround changes the tiling pattern by reshaping before/after the broadcast:

```python
def workaround_broadcast(value, target_shape):
    """Broadcast with reshape to avoid layout bug.

    Args:
        value: Array to broadcast, e.g., shape (1, 128)
        target_shape: Target shape, e.g., (8, 128)

    Returns:
        Broadcasted array with target_shape
    """
    # Broadcast normally
    broadcasted = jnp.broadcast_to(value, target_shape)

    # Reshape to change tiling (forces different layout inference)
    # If target_shape is (n, b), reshape to (8, n//8, b) then back
    n, b = target_shape
    if n % 8 == 0 and b % 128 == 0:
        # Reshape to (8, n//8, b) to retile
        reshaped = broadcasted.reshape(8, n // 8, b)
        # Reshape back to original shape
        result = reshaped.reshape(n, b)
        return result
    else:
        return broadcasted
```

**How it works**:
- The extra reshape changes how the layout inference sees the operation
- Different tile structure avoids the specific pattern that triggers the bug
- JAX optimizations will likely eliminate the redundant reshapes

## Workaround 2: Use pltpu.repeat

**Available**: `pltpu.repeat(x, repeats, axis)`

Instead of `jnp.broadcast_to()`, use `pltpu.repeat()`:

```python
# Instead of:
broadcasted = jnp.broadcast_to(threshold_idx, (n, b))  # May fail

# Use:
broadcasted = pltpu.repeat(threshold_idx, n, axis=0)  # May work
```

**Caveat**: `pltpu.repeat` may use the same underlying `vector.broadcast` operation, so this might not always work. Test first.

## Workaround 3: Use jnp.tile

Similar to repeat, but using standard JAX:

```python
# Instead of:
broadcasted = jnp.broadcast_to(threshold_idx, (n, b))  # May fail

# Use:
broadcasted = jnp.tile(threshold_idx, (n, 1))  # May work
```

## Workaround 4: Avoid Aligned Shapes

Change your shapes to avoid the exact multiples of 128:

```python
# If you need dim1 = 256, use 255 or 257 instead
# Then slice/pad back to 256 if needed

# Original failing shape
b = 256

# Workaround: use 255 and pad
b_work = 255
result = compute_with_broadcast(b_work)
result = jnp.pad(result, ((0,0), (0,1)))  # Pad back to 256
```

## Workaround 5: Manual Loop Instead of Broadcast

For the `take_along_axis_arrays` pattern, unroll the broadcast:

```python
def take_along_axis_arrays_no_broadcast(val, idx, axis):
    """Version that avoids broadcasting threshold_idx."""
    shape = idx.shape
    tile_shape = (NUM_SUBLANES, NUM_LANES)
    val, idx = (pad(x, tile_shape, val=0) for x in (val, idx))

    # Instead of broadcasting threshold_idx to full shape,
    # slice val for each position and gather directly
    # ... (implement without broadcast)
```

## Recommended Approach for Your Code

For the `take_along_axis` reproducer, try this modification:

```python
def reproducer_kernel_fixed(topk_logits_ref, p_ref, out_ref):
    topk_logits = topk_logits_ref[...]
    p = p_ref[...]
    shape = topk_logits.shape

    cumsum_probs = topk_logits  # skip for reproducer

    threshold_idx = (cumsum_probs < p[None, :]).sum(0, keepdims=True)
    threshold_idx = jnp.where(p[None, :] == 1., shape[0] - 1, threshold_idx)

    # WORKAROUND: Use reshape pattern to avoid layout bug
    n, b = shape

    # Broadcast with reshape workaround
    broadcasted_idx = jnp.broadcast_to(threshold_idx, shape)
    if n % 8 == 0 and b % 128 == 0 and b > 128:
        # Retile to avoid bug
        broadcasted_idx = broadcasted_idx.reshape(8, n // 8, b).reshape(n, b)

    thresholds = take_along_axis_arrays(
        topk_logits, broadcasted_idx, 0)

    topp_logits = jnp.where(
        topk_logits >= thresholds,
        topk_logits, -1e12)
    out_ref[...] = topp_logits
```

## Testing the Workarounds

Use the provided `test_broadcast_workarounds.py` script to test which workaround works best for your use case:

```bash
python test_broadcast_workarounds.py
```

## Why These Work

These workarounds succeed by:

1. **Reshape**: Changes the tile structure seen by layout inference, avoiding the specific (8,128) tiling pattern
2. **pltpu.repeat**: Different primitive with potentially different layout rules
3. **jnp.tile**: Different lowering path that may avoid the bug
4. **Non-aligned shapes**: Padding changes the tiling calculation, avoiding offset 128

## Limitations

- These are **workarounds**, not proper fixes
- Performance may be slightly different (usually negligible)
- The reshape workaround adds extra operations (though often optimized away)
- Not guaranteed to work in all cases

## Long-term Solution

The proper fix needs to be in the Mosaic compiler's layout inference pass. See `BROADCAST_LAYOUT_BUG_FIX.md` for technical details on the required fix.
