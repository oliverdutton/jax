# pltpu.repeat Lowering Analysis

## Question
What does `pltpu.repeat(threshold_idx, n, axis=0)` lower down to? What does it compile to?

## Answer

`pltpu.repeat` has **two different lowering paths** depending on the context:

### Path 1: Default Lowering (Non-TPU or unregistered contexts)

**File**: `jax/_src/pallas/mosaic/primitives.py:67-78`

```python
@repeat_p.def_impl
def repeat_impl(x: jax.Array, *, repeats: int, axis: int):
    reps = [repeats if i == axis else 1 for i in range(x.ndim)]
    return jnp.tile(x, reps)  # Uses tile!

def _repeat_lowering_rule(ctx: mlir.LoweringRuleContext, x, *, repeats, axis):
    return mlir.lower_fun(
        functools.partial(repeat_impl, repeats=repeats, axis=axis),
        multiple_results=False,
    )(ctx, x)
mlir.register_lowering(repeat_p, _repeat_lowering_rule)
```

**Lowering chain:**
1. `pltpu.repeat(x, n, axis)`
2. → `repeat_impl` which calls `jnp.tile(x, reps)`
3. → `jnp.tile` implementation (jax/_src/numpy/lax_numpy.py:4528-4530):
   ```python
   result = broadcast_to(reshape(A, [j for i in A_shape for j in [1, i]]),
                         [k for pair in zip(reps_tup, A_shape) for k in pair])
   return reshape(result, tuple(np.multiply(A_shape, reps_tup)))
   ```
4. → **Uses `broadcast_to` internally!**
5. → `vector.broadcast` MLIR operation
6. → **HITS THE SAME LAYOUT BUG** ❌

### Path 2: TPU-Specific Lowering (Pallas TPU kernels)

**File**: `jax/_src/pallas/mosaic/lowering.py:3474-3484`

```python
@register_lowering_rule(tpu_primitives.repeat_p)
def _repeat_lowering_rule(ctx: LoweringRuleContext, x, *, repeats, axis):
    (out_aval,) = ctx.avals_out
    return tpu.repeat(  # TPU dialect operation
        aval_to_ir_type(
            ctx.lowering_context.dynamic_shape_replacement_fn, out_aval
        ),
        x,
        axis,
        repeats,
    )
```

**Lowering chain:**
1. `pltpu.repeat(x, n, axis)`
2. → `tpu.repeat` MLIR operation (TPU dialect)
3. → **Canonicalized to `tpu.concatenate`** (see note below)
4. → Concatenates multiple copies of the input vector
5. → **Does NOT use `vector.broadcast`** ✅

**Evidence from MLIR dialect definition** (`jaxlib/mosaic/dialect/tpu/tpu_ops.td:621-631`):
```tablegen
// TODO(mvoz): deprecated - use concat. Canonicalization will do so automatically.
// b/376295711
def TPU_RepeatOp : TPU_Op<"repeat", [Pure]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$source,
    I32Attr:$dimension,
    I32Attr:$times
  );
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{ $source `,` $dimension `x` $times attr-dict `:` type($source) `->` type($output) }];
}
```

The comment explicitly states: **"deprecated - use concat. Canonicalization will do so automatically."**

## Which Path is Used in Pallas TPU Kernels?

For Pallas TPU kernels, **Path 2 (TPU-specific)** is used because:

1. The `@register_lowering_rule` decorator in `lowering.py` registers a TPU-specific lowering rule
2. This overrides the default lowering from `primitives.py`
3. The Pallas TPU lowering context uses these TPU-specific rules

## Concatenate vs Broadcast

**Why concatenate avoids the bug:**

- **Broadcast**: Replicates data within a single vector by adjusting layout
  - `broadcast([a, b], (3, 2))` → `[[a, b], [a, b], [a, b]]`
  - Requires layout inference to handle replication
  - Subject to the layout offset bug

- **Concatenate**: Joins multiple copies end-to-end
  - `concatenate([a, b], [a, b], [a, b], axis=0)` → `[a, b, a, b, a, b]`
  - Just memory rearrangement, no layout propagation issues
  - Does NOT use `vector.broadcast` operation

## Conclusion

**For the workarounds:**

- ✅ **Method 2 (`pltpu.repeat`)**: Should work! Uses `tpu.repeat` → `tpu.concatenate`, avoids broadcast bug
- ❌ **Method 3 (`jnp.tile`)**: Will likely fail! Uses `broadcast_to` internally, hits the same bug
- ✅ **Method 1 (reshape)**: Should work! Changes tiling pattern to avoid specific bug condition

## Verification Needed

To definitively confirm this analysis:
1. Run the test with method=2 (`pltpu.repeat`) on actual TPU hardware
2. Check if it successfully avoids the layout error
3. Optionally: dump MLIR with `PALLAS_FLAGS=--pallas_dump_ir=true` to see actual lowering

If `pltpu.repeat` still fails, it might indicate:
- The canonicalization to concatenate isn't happening
- Or there's another issue in the TPU backend
- But based on the code, it *should* work
