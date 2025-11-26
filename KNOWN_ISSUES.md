# Known Issues in Pallas NumPy Interpreter

## Critical Bug: Incorrect Sorting Results

**Status:** Under Investigation
**Severity:** High
**Affects:** Bitonic sort implementation with NumPy interpreter

### Symptom
When running the bitonic sort with the NumPy interpreter, the output is incorrect:
- Input: `[3.0, 1.0, 2.0]`
- Expected output: `[1.0, 2.0, 3.0]`
- Actual output: `[1.0, 1.0, 1.0]` (all values become the minimum value)

### Root Cause Analysis

Through extensive debugging, the following was discovered:

1. **Ref Mutation in Nested Jaxprs**
   - Input refs are passed as `consts` to `scan` primitives
   - Inside the scan body (at depth 2+), these refs get modified via `swap` operations
   - The modifications produce incorrect values (all elements become 1.0 = 1065353216 in int32)

2. **Architecture**
   - The sort algorithm uses in-place mutation of an input ref
   - 4 refs total: input ref, output ref, and 2 scratch refs
   - Input ref is meant to be sorted in-place, then copied to output ref

3. **Failed Fix Attempts**
   - **Attempt 1:** Copy refs when passing to nested jaxprs
     - Result: Sorting doesn't happen, output is unsorted original input
   - **Attempt 2:** Make refs read-only when passed as consts
     - Result: Prevents mutation, but sorting logic requires mutation

4. **Debug Findings**
   - The bug occurs during equation execution at depth 2
   - Sequence: `concatenate → slice → transpose → swap`
   - The `slice` operation already receives incorrect input (all 1.0)
   - This suggests a primitive operation earlier in the chain produces wrong results

### Hypothesis

One of the following primitives is likely producing incorrect results:
- `gather` (144 operations) - complex batched indexing
- `select_n` (371 operations) - element-wise selection
- `iota` + `broadcast_in_dim` combination - array generation/broadcasting

The bug manifests as broadcasting or reduction producing a constant value instead of varied values.

### Next Steps

1. **Implement Per-Operation Comparison**
   - Run both HLO and NumPy interpreters in parallel
   - Compare results of each primitive operation
   - Identify which specific primitive produces different results

2. **Focus Areas**
   - Batched gather with dimension (8, 128) and indices (8, 128, 1)
   - Operations inside nested jaxpr at depth 2-3
   - Check for dtype mismatches or shape broadcasting errors

### Debugging Commands

Enable debugging flags in `pallas_numpy_interpreter.py`:
```python
DEBUG_GET = True      # Line ~215
DEBUG_SWAP = True     # Line ~232
DEBUG_SCAN_DETAIL = True  # Line ~425
DEBUG_GATHER = True   # Line ~300
```

Run verification:
```bash
python verify_numpy_sort.py
python debug_simple_sort.py
```

### Files Involved

- `pallas_numpy_interpreter.py` - Main interpreter implementation
- `verify_numpy_sort.py` - Correctness verification
- `debug_simple_sort.py` - Simple test case for debugging
- `test_batched_gather.py` - Unit test for gather (passes independently)

### Impact

- NumPy interpreter produces incorrect results for sorting
- Performance measurements are invalid (claimed 7x speedup was based on wrong output)
- All other operations tested work correctly

### Workaround

Use HLO interpreter (default) for correct results:
```python
sort(data, num_keys=1, interpret=True)  # Uses HLO interpreter by default
```
