# Pallas TPU Lowering Analysis for Control Flow Operations

## Executive Summary

This report analyzes whether the following JAX LAX operations can be lowered in Pallas TPU kernels:
- `lax.cond`
- `lax.select`
- `lax.switch`
- `lax.select_n`

**Key Finding:** All four operations are supported in Pallas TPU kernels, with one important limitation for `select_n`.

## Detailed Analysis

### 1. lax.select_n ✓ SUPPORTED (with limitations)

**Primitive:** `lax.select_n_p`

**Location:** `jax/_src/pallas/mosaic/lowering.py:3132-3149`

**Lowering Rule:**
```python
@register_lowering_rule(lax.select_n_p, kernel_types=[*tpu_core.KernelType])
def _select_n_lowering_rule(ctx: LoweringRuleContext, pred, x, *args):
  if len(args) > 1:
    raise NotImplementedError("select_n only supported with <= 2 arguments")
  pred_aval, x_aval = ctx.avals_in[:2]
  if pred_aval.dtype != np.dtype(np.bool_):
    # Convert non-boolean predicates to boolean
    pred = lower_fun(lambda x: x != 0, multiple_results=False)(lower_ctx, pred)
  if not args:
    return x
  y, = args
  return arith.select(pred, y, x)
```

**Key Points:**
- ✓ Registered for all TPU kernel types (`TC`, `SC_SCALAR_SUBCORE`, `SC_VECTOR_SUBCORE`)
- ⚠️ **Limitation:** Only supports binary selection (maximum 2 cases)
- Automatically converts non-boolean predicates to boolean
- Uses MLIR's `arith.select` operation

**Behavior:**
- `lax.select_n(pred, case0, case1)` → Works ✓
- `lax.select_n(pred, case0, case1, case2)` → Raises `NotImplementedError` ✗

---

### 2. lax.select ✓ SUPPORTED

**Type:** High-level wrapper function (not a primitive)

**Location:** `jax/_src/lax/lax.py:2840-2865`

**Implementation:**
```python
def select(pred: ArrayLike, on_true: ArrayLike, on_false: ArrayLike) -> Array:
  """Selects between two branches based on a boolean predicate."""
  # NOTE: select_n_p has *opposite* argument order!
  pred, on_false, on_true = core.standard_insert_pvary(
      pred, on_false, on_true)
  return select_n_p.bind(pred, on_false, on_true)
```

**Key Points:**
- ✓ Works via `select_n_p.bind()` internally
- Reverses argument order compared to `select_n_p`
- Standard ternary selection: `select(pred, on_true, on_false)`

---

### 3. lax.cond ✓ SUPPORTED

**Primitive:** `lax.cond_p`

**Location:** `jax/_src/pallas/mosaic/lowering.py:3348-3384`

**Lowering Rule:**
```python
@register_lowering_rule(lax.cond_p, kernel_types=[*tpu_core.KernelType])
def _cond_lowering_rule(ctx: LoweringRuleContext, *args, branches, **params):
  index, *args = args
  constant_index = _fold_and_get_constant_value(index)

  # Optimization: if index is constant, directly execute that branch
  if constant_index is not None:
    return jaxpr_subcomp(
        ctx.lowering_context.replace(block_shapes=ctx.block_shapes[1:]),
        branches[constant_index].jaxpr, *args
    )

  # For dynamic index, create scf.IfOp cascade
  out_types = map(aval_to_ir_type_with_fn, ctx.avals_out)
  pred = arith.cmpi(arith.CmpIPredicate.ne, index, ir_constant(0, index.type))
  if_op = scf.IfOp(pred, out_types, hasElse=True)

  # ... implementation continues with recursive handling for multi-way branches
```

**Key Points:**
- ✓ Registered for all TPU kernel types
- ✓ Supports multiple branches (2+ branches)
- **Optimization:** Constant indices are optimized away at compile time
- **Implementation:** Uses MLIR's `scf.IfOp` (Structured Control Flow dialect)
- **Multi-branch strategy:** Cascaded if/else structures for > 2 branches
  - Note at line 3369: TODO suggests using `scf.IndexSwitchOp` instead

**Behavior:**
- `lax.cond(pred, true_fn, false_fn, operand)` → Works ✓
- Multiple branches via `cond_p.bind()` → Works ✓

---

### 4. lax.switch ✓ SUPPORTED

**Type:** High-level control flow function (not a primitive)

**Location:** `jax/_src/lax/control_flow/conditionals.py:68-147`

**Implementation:**
```python
def switch(index, branches: Sequence[Callable], *operands, ...):
  """Apply exactly one of the ``branches`` given by ``index``."""
  # ... validation and setup ...

  index = lax.convert_element_type(index, np.int32)
  lo = np.array(0, np.int32)
  hi = np.array(len(branches) - 1, np.int32)
  index = lax.clamp(lo, index, hi)
  return _switch_internal(index, branches, operands, ...)

def _switch_internal(...):
  # ... jaxpr processing ...
  out = cond_p.bind(index, *consts, *args, **params)  # Line 177
  # ...
```

**Key Points:**
- ✓ Works via `cond_p.bind()` internally
- ✓ Supports arbitrary number of branches
- Automatically clamps index to valid range `[0, len(branches)-1]`
- Wraps XLA's Conditional operation

---

## Lowering Infrastructure

### Registration System

All lowering rules are registered via the decorator:
```python
@register_lowering_rule(
    prim: jax_core.Primitive,
    *,
    kernel_types: Collection[tpu_core.KernelType] = (tpu_core.KernelType.TC,),
    ensure_mlir_values: bool = True,
)
```

### Kernel Types

TPU kernels support three types (defined in `tpu_core.KernelType`):
- `TC` - Tensor Core kernels
- `SC_SCALAR_SUBCORE` - Scalar Super-Controller kernels
- `SC_VECTOR_SUBCORE` - Vector Super-Controller kernels

Both `cond_p` and `select_n_p` are registered for **all kernel types**: `[*tpu_core.KernelType]`

### Lowering Dispatch

The main lowering loop in `jaxpr_subcomp` (lines 1135-1240) dispatches to registered rules:
```python
def jaxpr_subcomp(ctx: LoweringContext, jaxpr: jax_core.Jaxpr, *args):
  for eqn in jaxpr.eqns:
    if eqn.primitive in lowering_rules[ctx.kernel_type]:
      ans = lowering_rules[ctx.kernel_type][eqn.primitive](rule_context, *invals, **eqn.params)
```

---

## MLIR Lowering Targets

| Operation | MLIR Operation | Dialect |
|-----------|---------------|---------|
| `select_n_p` | `arith.select` | Arithmetic |
| `cond_p` | `scf.IfOp` | Structured Control Flow |

---

## Comparison with Other Control Flow

For reference, Pallas TPU also supports:

### lax.while ✓ SUPPORTED

**Location:** `jax/_src/pallas/mosaic/lowering.py:3287-3345`

```python
@register_lowering_rule(lax.while_p, kernel_types=[*tpu_core.KernelType])
def _while_lowering_rule(...):
  # Attempts to optimize while as fori loop
  fori_jaxpr, _ = pallas_utils.pattern_match_while_to_fori_loop(...)
  if fori_jaxpr is not None:
    return _lower_while_via_fori(...)  # Optimized path

  # Fallback to general while loop using scf.WhileOp
```

---

## Limitations Summary

| Operation | Supported | Limitations |
|-----------|-----------|-------------|
| `lax.select` | ✓ Yes | Binary only (2 cases) |
| `lax.select_n` | ✓ Yes | **Max 2 cases** (raises `NotImplementedError` for > 2) |
| `lax.cond` | ✓ Yes | None |
| `lax.switch` | ✓ Yes | None (arbitrary branches via cascaded if/else) |

---

## Recommendations

1. **For binary selection:** Use `lax.select` or `lax.select_n` (both work)
2. **For multi-way selection (> 2 branches):** Use `lax.switch`, NOT `lax.select_n`
3. **For conditional execution:** Use `lax.cond` (supports both 2-way and N-way)
4. **For constant indices:** `lax.cond` optimizes these away at compile time

---

## Testing

A verification script has been provided: `check_pallas_tpu_lowering.py`

Run with:
```bash
python check_pallas_tpu_lowering.py
```

The script tests all four operations and validates the findings of this analysis.

---

## File References

| File | Purpose |
|------|---------|
| `jax/_src/pallas/mosaic/lowering.py` | Main TPU lowering rules (4,272 lines) |
| `jax/_src/pallas/mosaic/lowering.py:3132-3149` | `select_n_p` lowering rule |
| `jax/_src/pallas/mosaic/lowering.py:3348-3384` | `cond_p` lowering rule |
| `jax/_src/lax/lax.py:2840-2865` | `lax.select` implementation |
| `jax/_src/lax/control_flow/conditionals.py:68-177` | `lax.switch` implementation |
| `jax/_src/pallas/mosaic/pallas_call_registration.py` | Pallas call registration |

---

## Conclusion

All four operations (`lax.cond`, `lax.select`, `lax.switch`, `lax.select_n`) **are supported** in Pallas TPU kernels. The only significant limitation is that `lax.select_n` supports a maximum of 2 cases. For multi-way selection with more than 2 branches, use `lax.switch` instead.
