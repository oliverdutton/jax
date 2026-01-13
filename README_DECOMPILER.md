# StableHLO to Jaxpr Decompiler

A complete decompiler that converts StableHLO IR (from JAX compiled objects) back to executable jaxpr using JAX lax primitives.

## Features

- ✅ **Basic Operations**: Arithmetic, unary, comparison operations
- ✅ **Shape Operations**: Reshape, transpose, broadcast, slicing
- ✅ **Matrix Operations**: dot_general (matrix multiplication)
- ✅ **Control Flow**: while loops, conditionals (cond/case)
- ✅ **Constants**: Proper handling of constant values
- ✅ **Type Conversions**: Element type conversions
- ✅ **Verification**: Decompiled jaxpr produces identical results to original

## Usage

```python
from hlo_to_jaxpr import StableHLOToJaxpr
from jax._src import core
import jax
import jax.numpy as jnp

# Define and compile a function
def my_function(x, y):
    return jnp.tanh(x * y + 1.0)

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

# Compile and lower to StableHLO
lowered = jax.jit(my_function).lower(x, y)
mlir_module = lowered.compiler_ir(dialect='stablehlo')

# Decompile to jaxpr
decompiler = StableHLOToJaxpr()
functions = decompiler.decompile_module(mlir_module)

# Get the main function
main_func = functions['"main"']
print("Decompiled jaxpr:")
print(main_func.jaxpr)

# Execute the decompiled jaxpr
result = core.eval_jaxpr(main_func.jaxpr, decompiler.constvals, x, y)
print(f"Result: {result}")

# Verify it matches the original
expected = my_function(x, y)
print(f"Matches original: {np.allclose(result, expected)}")
```

## How It Works

1. **MLIR Traversal**: Programmatically walks through the StableHLO MLIR module (no string parsing)
2. **Operation Mapping**: Maps each StableHLO operation to its corresponding JAX lax primitive
3. **Variable Tracking**: Maintains a mapping from MLIR SSA values to jaxpr variables
4. **Constant Handling**: Extracts constant values and represents them as jaxpr constvars
5. **Control Flow**: Recursively decompiles nested regions for while/cond operations
6. **Jaxpr Construction**: Builds valid jaxpr equations that can be executed

## Supported Operations

### Arithmetic
- add, subtract, multiply, divide, remainder
- maximum, minimum

### Unary
- abs, negate, exp, log, sqrt, tanh, sin, cos

### Comparison
- lt, le, gt, ge, eq, ne

### Shape Operations
- reshape, transpose, broadcast_in_dim
- dynamic_slice, dynamic_update_slice

### Matrix Operations
- dot_general (generalized matrix multiplication)

### Control Flow
- while_loop (condition + body)
- cond (multi-branch conditionals)

### Type Conversions
- convert_element_type

## Test Results

All comprehensive tests pass:
- ✓ Arithmetic Operations
- ✓ Unary Operations
- ✓ Matrix Multiplication
- ✓ While Loop
- ✓ Conditional (cond)
- ✓ Broadcasting
- ✓ Reshape and Transpose

## Implementation Details

### Variable Mapping
Uses string representation of MLIR values as stable keys (not Python object IDs, which change across uses).

### Control Flow Handling
- **While loops**: Decompiles condition and body regions separately, creates ClosedJaxpr for each
- **Conditionals**: Identifies captured variables from outer scope, passes them as explicit operands

### Constant Handling
Constants are extracted as constvars in the jaxpr and passed separately during evaluation.

## Limitations

- Scan operations are lowered to while loops in StableHLO, so decompiled code uses while_p instead of scan_p
- Some advanced operations (reduce with custom functions, etc.) may not be fully supported
- Distributed operations (all-reduce, psum, etc.) can be added following the same pattern

## Files

- `hlo_to_jaxpr.py` - Main decompiler implementation
- `test_comprehensive.py` - Comprehensive test suite
- `test_control_flow.py` - Control flow specific tests
- `explore_hlo.py` / `explore_control_flow.py` - Exploration scripts

## Next Steps

To add distributed operations (all-reduce, psum, etc.):

1. Identify the StableHLO operation name (e.g., `stablehlo.all_reduce`)
2. Add a mapping in `_stablehlo_to_primitive` to the corresponding lax primitive
3. Parse any required parameters from MLIR attributes
4. Add the `sharding` or other required parameters

Example:
```python
elif op_name_str == 'stablehlo.all_reduce':
    params = {
        'axis_name': ...,  # Parse from attributes
        'reducer': ...,     # Parse reducer function
    }
    return lax.psum_p, params, operand_vars, result_vars
```
