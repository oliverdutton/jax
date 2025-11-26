# Pallas NumPy Interpreter - Debug Status

## Current State: BUG NOT YET FIXED

The NumPy interpreter produces incorrect sorting results. After extensive debugging, the bug has been isolated but not resolved.

## What Works ✅

1. **60+ Primitive Operations** implemented and working:
   - Arithmetic: add, sub, mul, div, rem, etc.
   - Comparison: eq, ne, lt, gt, le, ge
   - Logical: and, or, not, xor (with correct int32 return types)
   - Bitwise: and, or, xor, not, shifts
   - Array ops: reshape, transpose, slice, pad, concatenate
   - State ops: get, swap, addupdate (for refs)
   - Control flow: scan, while, cond, jit
   - Advanced: gather (with batched support), select_n, iota

2. **Gather Implementation** - Batched gather tested independently and works correctly
   - `test_batched_gather.py` passes all tests
   - Handles complex dimension mappings

3. **Performance Isolation** - Successfully measured component costs:
   - Baseline (HLO): 28.33s
   - Gather → dummy: 10.99s (XLA gather costs ~17.3s)
   - Gather → NumPy: 12.42s (NumPy gather only ~1.4s, 12x faster!)
   - *Note: Full NumPy timing invalid due to incorrect results*

## The Bug 🐛

**Symptom:**
```python
Input:    [3.0, 1.0, 2.0]
Expected: [1.0, 2.0, 3.0]  (sorted)
Actual:   [1.0, 1.0, 1.0]  (all minimum value)
```

**What We Know:**

1. **Location**: Bug occurs inside first `scan` operation (EQN 4 @ depth 1)
2. **Mechanism**:
   - Input ref (8x128 array) passed as const to scan
   - Scan body executes: `concatenate → slice → transpose → swap`
   - `slice` already receives incorrect input (all 1.0)
   - Result: input ref gets overwritten with all 1.0 values

3. **Architecture**:
   - 4 refs: input (data), output (result), 2 scratch
   - Input ref sorted in-place, then copied to output
   - Jaxpr has 0 outvars (all communication via ref mutations)

4. **Attempted Fixes**:
   - ❌ Copy refs when passing to nested jaxprs → No sorting happens
   - ❌ Make refs read-only → Prevents required mutations
   - The sorting algorithm REQUIRES in-place ref mutation

## Debug Infrastructure 🔧

Added comprehensive logging (all disabled by default, enable in code):

1. **GET/SWAP Tracking** (`pallas_numpy_interpreter.py:215, 232`)
   - Logs all ref reads and writes
   - Shows indexer, shape, old/new values

2. **Scan Detail** (`pallas_numpy_interpreter.py:425`)
   - Logs num_consts, num_carry, length
   - Shows which inputs are refs vs values
   - Tracks writable status

3. **Equation Tracing** (`pallas_numpy_interpreter.py:184`)
   - Logs each equation with recursion depth
   - Helps understand nested jaxpr execution

4. **Gather Debugging** (`pallas_numpy_interpreter.py:300`)
   - Detailed operand/indices logging
   - Batch dimension tracking

5. **Recursion Depth** (`pallas_numpy_interpreter.py:137-144`)
   - Tracks nesting level of jaxpr execution
   - Helps isolate bugs in nested computations

## Test Files 📁

- `verify_numpy_sort.py` - Shows NumPy interpreter produces wrong results
- `verify_hlo_sort.py` - Confirms HLO interpreter works correctly
- `debug_simple_sort.py` - Minimal test case (3 elements)
- `test_batched_gather.py` - Unit test for gather (passes)
- `gather_only_numpy.py` - Isolated gather performance test
- `gather_dummy.py` - Measure XLA overhead without gather

## Next Steps 🎯

### Immediate (Required to Fix Bug)

1. **Implement Per-Operation Comparison**
   - Run each primitive with both HLO and NumPy
   - Compare results automatically
   - Identify first divergence point

2. **Focus Investigation On**:
   - Operations before EQN 1338 (slice)
   - Likely suspects: gather (144x), select_n (371x), iota, broadcast_in_dim
   - Check for: incorrect indexing, wrong broadcasting, dtype issues

3. **Specific Checks**:
   ```python
   # Check if gather produces constant output from varied input
   # Check if select_n always chooses same case
   # Check if broadcast creates shared memory views
   ```

### Medium Term (After Bug Fix)

1. Verify performance claims with correct results
2. Extend to other Pallas kernels (beyond sort)
3. Add comprehensive test suite
4. Performance optimization

## How to Debug 👨‍💻

1. **Enable logging** in `pallas_numpy_interpreter.py`:
   ```python
   DEBUG_GET = True
   DEBUG_SWAP = True
   DEBUG_SCAN_DETAIL = True
   ```

2. **Run minimal test**:
   ```bash
   python debug_simple_sort.py
   ```

3. **Look for**:
   - SWAP operations writing constant values
   - Operations at depth 2+ producing wrong results
   - Memory address reuse (shared views)

## Key Insight 💡

The bug is NOT in:
- Ref semantics (get/swap work correctly)
- Gather implementation (tested independently)
- Scan loop structure (executes correct number of iterations)
- Dtype handling (int32 bitcast is intentional)

The bug IS in:
- Some primitive operation producing a broadcasted constant
- Likely in a complex operation like gather with specific parameters
- Or in combination of operations (e.g., iota + broadcast + gather)

## Performance Note ⚡

Even with the bug, we've proven that:
- NumPy gather is 12x faster than XLA gather (1.4s vs 17.3s)
- io_callback overhead is minimal (~4ms per call)
- Pure NumPy execution has potential for significant speedup

Once the bug is fixed, the full NumPy interpreter should provide substantial performance improvements for Pallas kernels on CPU.
