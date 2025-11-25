# Pallas TPU Interpret Mode: Optimization Strategies

## Current Performance Problems

From analyzing `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py`, the main bottlenecks are:

1. **~33+ io_callback calls per grid iteration** (ordered, synchronous, blocking)
2. **Callback overhead dominates execution time** (context switches, Python function calls)
3. **No batching** of operations - each memory access is a separate callback
4. **Simulation runs in Python** (numpy arrays, threading primitives) - can't be JIT-compiled
5. **All critical path operations are eager** (load, store, DMA, synchronization)

For a kernel with grid `(N, M)` and `K` operations per iteration:
```
Total overhead = O(N × M × K × callback_cost)
```

Where `callback_cost` >> compiled operation cost.

## Optimization Strategies (Ranked by Impact)

### Strategy 1: Pre-compile Kernel Body (HIGH IMPACT)

**Current code** (line 1900-1941):
```python
def _run_jaxpr(jaxpr, consts, *args):
  def _run(jaxpr, consts, *args):
    jax_core.eval_jaxpr(jaxpr, consts, *args)
  traced = jax.jit(_run, static_argnums=(0,)).trace(jaxpr, consts, *args)
  traced.lower().compile()(consts, *args)  # Compiled at RUNTIME
  return

def _call_threadmap_callback(jaxpr, num_threads, *consts):
  # TODO(jburnim): Would it be worth trying to lower/compile the jaxpr at
  # lowering/compilation time?  E.g., by using a custom primitive here, could
  # we lower/compile jaxpr at lowering time, and then pass the compiled
  # function to the callback?
  return callback.io_callback(
      functools.partial(_thread_map_callback, jaxpr),
      (),
      num_threads,
      consts,
      ordered=True)
```

**Proposed optimization:**

Compile the kernel jaxpr at **lowering time** instead of runtime, then pass the compiled executable to the callback.

```python
# New approach:
def _pallas_interpret_primitive_impl(
    compiled_kernel_fn,  # Pre-compiled function
    *args,
    grid_size,
    ...
):
    """Single callback that executes pre-compiled kernel."""
    for grid_point in range(grid_size):
        # Execute compiled kernel, not interpreted
        compiled_kernel_fn(grid_point, *args)
    return results

# At lowering time:
def interpret_lowering_rule(ctx, *in_nodes, jaxpr, ...):
    # Compile the kernel jaxpr HERE during lowering
    compiled_fn = jax.jit(lambda *args: _execute_kernel(jaxpr, *args)).lower().compile()

    # Single callback with compiled function
    return mlir.lower_callback(
        functools.partial(_pallas_interpret_primitive_impl, compiled_fn),
        ctx,
        *in_nodes,
        ...
    )
```

**Expected speedup:** 5-20x (eliminates most callbacks)

**Challenges:**
- Memory simulation still needs callbacks for race detection
- Need to handle stateful operations differently

---

### Strategy 2: Batch Memory Operations (HIGH IMPACT)

**Current code** (line 2296-2379):
Every grid iteration does 4+ callbacks:
```python
# Callback 1: Read input slice
sliced_val = callback.io_callback(get, ..., ordered=True)

# Callback 2: Store to kernel input
callback.io_callback(store, ..., ordered=True)

# Callback 3-N: Every load/store in _interpret_jaxpr
out = callback.io_callback(get, ..., ordered=True)

# Callback N+1: Get kernel output
kernel_output_val = callback.io_callback(get, ..., ordered=True)

# Callback N+2: Store to output buffer
callback.io_callback(store, ..., ordered=True)
```

**Proposed optimization:**

Collect all memory operations for a grid iteration and execute them in a **single batch callback**.

```python
@dataclasses.dataclass
class MemoryOperation:
    op_type: Literal['load', 'store', 'dma_start', 'dma_wait']
    buffer_id: int
    indices: tuple
    value: Array | None

class BatchedMemoryOps:
    def __init__(self):
        self.ops: list[MemoryOperation] = []

    def add_load(self, buffer_id, indices):
        self.ops.append(MemoryOperation('load', buffer_id, indices, None))

    def add_store(self, buffer_id, indices, value):
        self.ops.append(MemoryOperation('store', buffer_id, indices, value))

    def execute_batch(self, shared_memory):
        """Single callback that executes all operations."""
        results = []
        for op in self.ops:
            if op.op_type == 'load':
                results.append(shared_memory.get(op.buffer_id, op.indices))
            elif op.op_type == 'store':
                shared_memory.store(op.buffer_id, op.indices, op.value)
        return results

# In _interpret_jaxpr:
def _interpret_jaxpr_batched(jaxpr, *args, ...):
    batch = BatchedMemoryOps()

    # First pass: collect all operations
    for eqn in jaxpr.eqns:
        if eqn.primitive is primitives.load_p:
            batch.add_load(...)
        elif eqn.primitive is primitives.store_p:
            batch.add_store(...)

    # Second pass: execute in single callback
    results = callback.io_callback(
        batch.execute_batch,
        result_shape,
        shared_memory,
        ordered=True  # Still ordered, but only ONE callback
    )

    # Third pass: use results
    ...
```

**Expected speedup:** 3-10x (reduces callbacks per iteration from ~33 to ~3)

**Challenges:**
- Requires two passes over jaxpr (collect, execute, use results)
- Harder to debug
- Doesn't work well with control flow (cond, while)

---

### Strategy 3: Replace Simulation with Direct Computation (VERY HIGH IMPACT)

**Current approach:**
Simulates TPU memory hierarchy (HBM, VMEM, SMEM) using Python data structures with threading primitives.

**Proposed optimization:**

For performance testing (not debugging), skip simulation entirely and execute kernel body directly on CPU.

```python
@dataclasses.dataclass(frozen=True, kw_only=True)
class FastInterpretParams(InterpretParams):
    """Fast interpret mode - trades simulation fidelity for speed."""

    fast_mode: bool = True  # Skip memory simulation
    skip_race_detection: bool = True  # No vector clocks
    skip_dma_simulation: bool = True  # DMAs are instant
    skip_semaphore_simulation: bool = True  # No synchronization overhead

def interpret_pallas_call_fast(*args, interpret_params: FastInterpretParams, ...):
    """Fast interpret mode: direct execution without simulation."""

    if not interpret_params.fast_mode:
        return interpret_pallas_call(*args, ...)  # Fall back to full simulation

    # No shared memory allocation - use direct JAX arrays
    # No callbacks - just execute kernel body directly

    def execute_kernel_direct(grid_point, *inputs):
        # Map kernel inputs directly to JAX arrays (no HBM simulation)
        kernel_args = slice_inputs(inputs, grid_point)

        # Execute kernel body using normal JAX evaluation (JIT-compiled!)
        outputs = jax_core.eval_jaxpr(jaxpr, *kernel_args)

        return outputs

    # Vectorize over grid (can be JIT-compiled!)
    results = jax.vmap(
        jax.vmap(execute_kernel_direct, in_axes=(0, None))
    )(grid_points, inputs)

    return results
```

**Expected speedup:** 50-1000x (eliminates all simulation overhead)

**Tradeoffs:**
- ✅ Massively faster
- ✅ Still tests kernel logic
- ❌ Doesn't test memory access patterns
- ❌ Doesn't detect races
- ❌ Doesn't simulate TPU-specific behavior

**Use case:** Performance profiling, correctness testing (logic only)

---

### Strategy 4: Use JAX Arrays for Memory Simulation (MEDIUM IMPACT)

**Current code** (`shared_memory.py`):
Uses `numpy` arrays and Python `threading` primitives - can't be JIT-compiled.

```python
class Memory:
    def __init__(self, shape, dtype):
        self.data = np.zeros(shape, dtype)  # NumPy, not JAX
        self.lock = threading.Lock()  # Python threading
```

**Proposed optimization:**

Replace numpy/threading with JAX arrays and functional updates.

```python
import jax.numpy as jnp
from jax import lax

class MemoryState:
    """Immutable memory state - can be JIT-compiled."""
    buffers: dict[int, jnp.ndarray]  # JAX arrays
    semaphore_counts: jnp.ndarray

def memory_load(state: MemoryState, buffer_id: int, indices):
    """Pure function - can be JIT-compiled."""
    return state.buffers[buffer_id][indices]

def memory_store(state: MemoryState, buffer_id: int, indices, value):
    """Pure function - returns new state - can be JIT-compiled."""
    new_buffer = state.buffers[buffer_id].at[indices].set(value)
    return dataclasses.replace(
        state,
        buffers={**state.buffers, buffer_id: new_buffer}
    )

# Then JIT-compile the entire simulation loop:
@jax.jit
def execute_grid(initial_memory_state, grid_size, kernel_fn):
    def body(i, state):
        state = kernel_fn(state, grid_point=i)
        return state

    final_state = lax.fori_loop(0, grid_size, body, initial_memory_state)
    return final_state
```

**Expected speedup:** 2-5x (JIT-compiles the simulation)

**Challenges:**
- Hard to implement semaphores/synchronization functionally
- Memory overhead (immutable updates create copies)
- Race detection becomes harder

---

### Strategy 5: Use pure_callback Instead of io_callback (LOW IMPACT)

**Current code:**
```python
callback.io_callback(operation, result_shape, *args, ordered=True)
```

`io_callback` with `ordered=True` forces strict sequencing and prevents optimizations.

**Proposed optimization:**

Use `pure_callback` where operations don't have side effects:

```python
# For loads (no side effects):
callback.pure_callback(
    lambda args: memory.load(*args),
    result_shape,
    *args,
    vmap_method='legacy'  # Allow vectorization
)

# For stores (has side effects - still need io_callback):
callback.io_callback(
    lambda args: memory.store(*args),
    (),
    *args,
    ordered=True
)
```

**Expected speedup:** 1.2-2x (XLA can optimize pure callbacks)

**Limitations:**
- Only helps with loads, not stores
- Still requires callbacks

---

### Strategy 6: Ahead-of-Time Grid Analysis (MEDIUM IMPACT)

**Current code:**
Recomputes index maps and start indices for every grid iteration (line 2257-2271).

**Proposed optimization:**

Pre-compute all grid indices at lowering time if grid is static.

```python
def precompute_grid_indices(grid_mapping, grid):
    """Compute all block indices ahead of time."""
    all_indices = []
    for grid_point in itertools.product(*[range(dim) for dim in grid]):
        indices = compute_start_indices(grid_mapping, grid_point)
        all_indices.append(indices)
    return jnp.array(all_indices)  # JAX array, can be passed to compiled code

# Then in kernel execution:
@jax.jit
def execute_with_precomputed_indices(inputs, outputs, all_indices, kernel_fn):
    def body(i, state):
        indices = all_indices[i]  # Just index, don't recompute
        state = kernel_fn(state, indices)
        return state
    ...
```

**Expected speedup:** 1.5-3x (eliminates redundant computation)

**Limitations:**
- Only works for static grids
- Increases memory usage

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (1-2 days)
1. ✅ Add `FastInterpretParams.skip_race_detection=True` by default
2. ✅ Pre-compute grid indices for static grids
3. ✅ Use `pure_callback` for loads

**Expected speedup:** 2-3x

### Phase 2: Structural Changes (1-2 weeks)
1. ✅ Implement batched memory operations
2. ✅ Pre-compile kernel jaxpr at lowering time (as suggested in TODO)

**Expected speedup:** 10-30x (cumulative)

### Phase 3: Complete Redesign (1-2 months)
1. ✅ Implement `FastInterpretParams.fast_mode` with direct execution
2. ✅ Replace numpy/threading with JAX arrays in memory simulation
3. ✅ JIT-compile entire simulation loop

**Expected speedup:** 50-100x+ (cumulative)

---

## Code Locations for Changes

| File | Lines | Change |
|------|-------|--------|
| `interpret_pallas_call.py` | 1900-1941 | Pre-compile kernel jaxpr |
| `interpret_pallas_call.py` | 2296-2379 | Batch memory operations |
| `interpret_pallas_call.py` | 1305-1333 | Use pure_callback for loads |
| `interpret_pallas_call.py` | 2257-2271 | Pre-compute grid indices |
| `shared_memory.py` | 1-300 | Replace numpy with JAX arrays |
| `interpret_pallas_call.py` | 59-141 | Add `FastInterpretParams` |

---

## Alternative: CPU Backend with Real Compilation

Instead of optimizing interpret mode, consider adding a **CPU backend** that actually compiles kernels:

```python
# New file: jax/_src/pallas/cpu/lowering.py
def pallas_call_cpu_lowering_rule(ctx, *in_nodes, jaxpr, grid_mapping, ...):
    """Lower Pallas kernel to CPU MLIR (like GPU/TPU backends)."""

    # Generate MLIR for CPU execution (like triton/mosaic backends do)
    module = lower_jaxpr_to_cpu_mlir(jaxpr, grid_mapping)

    # Return compiled CPU kernel (no callbacks!)
    return compile_cpu_module(module)
```

This would:
- ✅ Be as fast as native execution
- ✅ Actually JIT-compile the kernel
- ❌ Require significant engineering effort
- ❌ No memory simulation or race detection

---

## Conclusion

The current interpret mode is slow because it uses **synchronous callbacks for every operation**. To make it fast:

1. **Short term:** Reduce callbacks (batch operations, pre-compute indices)
2. **Medium term:** Pre-compile kernel body (as suggested in TODO)
3. **Long term:** Add "fast mode" that skips simulation, or build real CPU backend

The fundamental tradeoff is **simulation fidelity vs speed**. For debugging, keep current mode. For performance testing, add a fast mode that sacrifices simulation accuracy.
