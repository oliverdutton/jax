# Pallas TPU Interpret Mode Performance Analysis

## Summary

**Why is interpret mode so slow?**

Pallas TPU interpret mode is slow because it uses **hybrid execution**: while the outer structure goes through MLIR lowering and uses JIT compilation infrastructure, the actual kernel operations execute **eagerly via synchronous callbacks** (`io_callback` with `ordered=True`). This prevents compiler optimizations and forces sequential execution.

## Execution Model

### Overall Flow

```
pallas_call
    ↓
mlir.lower_fun(interpret_pallas_call, ...)  ← JIT infrastructure
    ↓
interpret_pallas_call()  ← Executed via callbacks
    ↓
Multiple io_callback(ordered=True) calls  ← EAGER EXECUTION
    ↓
Simulates TPU state in CPU memory
```

### Key Code Locations

1. **Entry point**: `/home/user/jax/jax/_src/pallas/pallas_call.py:1407-1416`
   - Wraps interpret implementation with `mlir.lower_fun()`
   - Still uses JIT infrastructure but actual execution is eager

2. **Main implementation**: `/home/user/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py`
   - Line 1886: `get_interpret_effects()` returns `{callback._OrderedIOEffect}`
   - Line 1943-2500: `interpret_pallas_call()` - main interpret entry point

## Why It's Slow: The Callback Problem

### Callbacks Everywhere

The interpret mode uses `io_callback(ordered=True)` extensively, which forces **synchronous, eager execution**:

#### 1. **Initialization** (called once)
- Line 1988-1997: Initialize shared memory
- Line 2023+: Allocate input buffers
- Line 2058-2067: Allocate output buffers
- Line 2078-2087: Allocate scalar buffers
- Line 2096-2104: Allocate semaphores
- Line 2118-2129: Allocate kernel argument buffers
- Line 2134: Global barrier before kernel execution

#### 2. **Per Grid Point** (called for EVERY iteration)
- Line 2296-2309: Read slice from input buffer (**io_callback**)
- Line 2310-2322: Store slice to kernel input buffer (**io_callback**)
- Line 2339-2349: Execute kernel via `_interpret_jaxpr()`
- Line 2353-2363: Get kernel output (**io_callback**)
- Line 2365-2379: Store to output buffer (**io_callback**)

#### 3. **Inside Kernel Body** (called for EVERY operation)
In `_interpret_jaxpr()` (line 1242-1640):
- Line 1305-1314: Every `load_p` operation (**io_callback**)
- Line 1322-1333: Every `swap_p` operation (**io_callback**)
- Line 1505-1520: Every `dma_start_p` operation (**io_callback**)
- Line 1545-1568: Every `dma_wait_p` operation (**io_callback**)
- Line 1579-1596: Every `semaphore_signal_p` operation (**io_callback**)
- Line 1597-1611: Every `semaphore_wait_p` operation (**io_callback**)

### Impact

With a grid of size `(N, M)` and `K` memory operations per kernel:
- **Total callbacks** = O(1) init + O(N×M) × (4 boundary callbacks + K kernel callbacks)
- **Each callback** executes synchronously and cannot be optimized by the compiler
- **No batching** or fusion possible across callbacks

## Is It Being JIT-Compiled?

**Yes and No:**

### What IS JIT-Compiled
1. Grid iteration loop structure (`lax.while_loop` at line 2455)
2. Thread mapping jaxpr (`_run_jaxpr` at line 1900-1905 compiles the jaxpr)
3. Control flow within kernel body (`lax.cond`, `lax.scan`, `lax.while_loop` at lines 1360-1393)
4. Standard JAX operations that don't need special handling

### What Runs EAGERLY
1. **All memory operations** (load, store, swap) - via io_callback
2. **All DMA operations** (dma_start, dma_wait) - via io_callback
3. **All synchronization** (semaphore_signal, semaphore_wait, barriers) - via io_callback
4. **Buffer allocation and initialization** - via io_callback
5. **Thread mapping execution** - via io_callback (line 1936-1941)

The problem is that the **critical path** (memory operations, synchronization) runs eagerly, while only the **control flow structure** is JIT-compiled.

## Design Intent

From line 63-67 of `interpret_pallas_call.py`:

```python
"""TPU interpret mode is a way run Pallas TPU kernels on CPU, while simulating
a TPU's shared memory (HBM, VMEM, etc.), communication (remote and local
DMAs), and synchronization operations (semaphores, barriers, etc.).  This mode
is intended for debugging and testing.
```

**This is intentional** - interpret mode prioritizes:
- ✅ Correctness and debugging capability
- ✅ Simulating TPU semantics (memory hierarchy, synchronization)
- ✅ Race detection (optional, via vector clocks)
- ❌ NOT performance

## Known Optimization Opportunities

There's a TODO comment at line 1932-1935:

```python
# TODO(jburnim): Would it be worth trying to lower/compile the jaxpr at
# lowering/compilation time?  E.g., by using a custom primitive here, could
# we lower/compile jaxpr at lowering time, and then pass the compiled
# function to the callback?
```

This suggests the team knows the current approach is suboptimal.

## Configuration Options

`InterpretParams` provides some optimization flags (line 59-141):

| Flag | Effect | Performance Impact |
|------|--------|-------------------|
| `skip_floating_point_ops=True` | Skip FP computation, use sentinels | Can significantly reduce overhead |
| `detect_races=False` | Disable race detection | Reduces synchronization overhead |
| `dma_execution_mode="eager"` | Execute DMAs immediately | May reduce latency vs "on_wait" |
| `num_cores_per_device=1` | Simulate fewer cores | Reduces threading overhead |

## Recommendations

1. **For debugging/testing**: Use interpret mode as-is - it's designed for this
2. **For performance**: Don't use interpret mode - compile to actual TPU backend
3. **Potential speedup**: Set `skip_floating_point_ops=True` if you only care about control flow
4. **Future improvement**: The codebase would benefit from compiling more of the kernel body instead of using eager callbacks

## Conclusion

Pallas TPU interpret mode is **intentionally slow** because:
- It runs on CPU (not TPU hardware)
- It simulates TPU memory hierarchy and synchronization
- It uses eager execution via `io_callback(ordered=True)` for all critical operations
- It prioritizes correctness and debuggability over performance

The JIT infrastructure is present but mostly for control flow - the actual memory and synchronization operations that dominate execution time are all eager callbacks that cannot be optimized by the compiler.
