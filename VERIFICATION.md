# Verification: categorical_simple.py matches JAX's implementation

## Summary
The implementation in `categorical_simple.py` produces **identical output** to `jax.random.categorical` for default parameters (shape=None, replace=True, mode=None).

## JAX's Implementation (jax/_src/random.py)

For **default parameters**, JAX's categorical simplifies to:

```python
# Line 1800-1803 (with shape=None, replace=True, mode=None)
return jnp.argmax(
    gumbel(key, logits.shape, logits.dtype, mode=None) + logits,
    axis=axis
)
```

Where `gumbel()` with mode=None uses "low" mode (line 1739-1740):

```python
return -jnp.log(-jnp.log(
    _uniform(key, minval=info.tiny, maxval=1., shape=shape, dtype=dtype)
))
```

## My Implementation (categorical_simple.py)

```python
def categorical(key, logits, axis=-1):
    tiny = jnp.finfo(logits.dtype).tiny
    u = jax.random.uniform(key, logits.shape, dtype=logits.dtype,
                          minval=tiny, maxval=1.0)
    gumbel = -jnp.log(-jnp.log(u))
    return jnp.argmax(logits + gumbel, axis=axis)
```

## Equivalence Proof

Both implementations use:

1. **Same PRNG key** → Same random stream
2. **Same shape**: `logits.shape`
3. **Same dtype**: `logits.dtype`
4. **Same uniform bounds**: `minval=finfo(dtype).tiny, maxval=1.0`
5. **Same Gumbel transform**: `-log(-log(u))`
6. **Same argmax operation**: `argmax(..., axis=axis)`

Since `jax.random.uniform()` internally calls `_uniform()` (the same function used by JAX's gumbel), and all parameters are identical, **the outputs are guaranteed to be identical**.

## Trace Through Example

**Example**: `jax.random.categorical(key, jnp.array([0.0, 1.0, 2.0]))`

### JAX's path:
1. `batch_shape = tuple(np.delete((3,), -1)) = ()`
2. `shape = None → shape = ()`
3. `shape_prefix = ()[:0] = ()`
4. `logits_shape` = `[3]` (reconstructed)
5. `gumbel(key, (3,), float32, None)`
   - Samples `u ~ Uniform(tiny, 1)` with shape `(3,)`
   - Returns `-log(-log(u))`
6. `argmax(gumbel + logits, axis=-1)`

### My path:
1. `tiny = finfo(float32).tiny`
2. `u = uniform(key, (3,), float32, minval=tiny, maxval=1.0)`
3. `gumbel = -log(-log(u))`
4. `argmax(logits + gumbel, axis=-1)`

**Result**: IDENTICAL ✓

## Tested Cases

- ✓ 1D logits: `(n,)`
- ✓ 2D logits: `(batch, n)`
- ✓ Multi-dimensional: `(d1, d2, d3)` with various axes
- ✓ All float dtypes (float16, float32, float64)

## Limitations

This simplified implementation only supports **default parameters**:
- `shape=None` (single sample per distribution)
- `replace=True` (sampling with replacement)
- `mode=None` (low precision Gumbel, sufficient for most cases)

For other cases, use the full `jax.random.categorical` function.
