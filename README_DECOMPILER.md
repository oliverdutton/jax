# StableHLO to JAX Function Decompiler (Refactored)

A decompiler that converts StableHLO IR (from JAX compiled objects) back to executable Python functions using `jax.lax` operations.

## Philosophy

This refactored version takes a fundamentally different approach from traditional decompilers:

**OLD APPROACH (misguided):**
- Built jaxpr equations directly using primitives (`lax.add_p`, etc.)
- Created jaxpr Vars, equations, and manually constructed jaxpr structures
- Required understanding of jaxpr internals

**NEW APPROACH (better):**
- Creates executable Python functions using pure JAX Arrays
- Uses high-level `jax.lax` operations (like `lax.add`, `lax.mul`, not primitives)
- Maintains a dictionary mapping StableHLO value names to computed results
- Progressively builds the value dictionary by parsing the compiled object
- Returns actual callable functions that work with JAX Arrays

## Features

- ✅ **Arithmetic Operations**: add, subtract, multiply, divide, remainder, max, min
- ✅ **Unary Operations**: abs, negate, exp, log, sqrt, tanh, sin, cos, rsqrt, pow
- ✅ **Comparison Operations**: lt, le, gt, ge, eq, ne
- ✅ **Shape Operations**: reshape, transpose, broadcast_in_dim, slicing
- ✅ **Matrix Operations**: dot_general (generalized matrix multiplication)
- ✅ **Control Flow**: while loops, conditionals (case/switch)
- ✅ **Constants**: Proper handling of constant values
- ✅ **Type Conversions**: Element type conversions
- ✅ **Reduction Operations**: reduce_sum (and others)

## Usage

```python
from hlo_to_jaxpr import StableHLOToJaxpr
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

# Decompile to callable Python function
decompiler = StableHLOToJaxpr()
functions = decompiler.decompile_module(mlir_module)

# Get the main function
main_func = functions['"main"']

# Execute the decompiled function with JAX Arrays
result = main_func.callable_fn(x, y)
print(f"Result: {result}")

# Verify it matches the original
expected = my_function(x, y)
print(f"Matches original: {jnp.allclose(result, expected)}")
```

## How It Works

1. **MLIR Traversal**: Programmatically walks through the StableHLO MLIR module
2. **Operation Execution**: For each operation, looks up operands in a value dictionary and executes the corresponding `jax.lax` operation
3. **Value Dictionary**: Maintains a mapping from StableHLO value names to their computed JAX Array values
4. **Progressive Building**: As each operation executes, stores its result in the dictionary for later operations
5. **Callable Creation**: Returns a Python function that can be called with JAX Arrays as inputs

### Key Differences from Old Approach

| Aspect | Old Approach | New Approach |
|--------|--------------|--------------|
| **Output** | jaxpr structure | Python callable function |
| **Operations** | Primitives (`lax.add_p`) | High-level API (`lax.add`) |
| **Execution** | `core.eval_jaxpr` | Direct function call |
| **Complexity** | High (manual jaxpr building) | Low (progressive execution) |
| **Debugging** | Difficult (opaque jaxpr) | Easy (normal Python functions) |

## Example: Value Dictionary Flow

```python
# Given: x + y * 2.0

# StableHLO IR:
#   %0 = stablehlo.constant dense<2.0>
#   %1 = stablehlo.multiply %y, %0
#   %2 = stablehlo.add %x, %1
#   return %2

# Value Dictionary Evolution:
# Step 0: {"%arg0": x_array, "%arg1": y_array}
# Step 1: {"%arg0": x_array, "%arg1": y_array, "%0": 2.0}
# Step 2: {"%arg0": x_array, "%arg1": y_array, "%0": 2.0, "%1": y * 2.0}
# Step 3: {"%arg0": x_array, "%arg1": y_array, "%0": 2.0, "%1": y * 2.0, "%2": x + (y * 2.0)}
# Return: value_dict["%2"]
```

## Control Flow Support

### While Loops

```python
def countdown(n):
    def cond_fn(i):
        return i > 0

    def body_fn(i):
        return i - 1

    return lax.while_loop(cond_fn, body_fn, n)
```

The decompiler creates the condition and body functions dynamically by decompiling their respective blocks.

### Conditionals

```python
def abs_value(x):
    return lax.cond(x < 0, lambda: -x, lambda: x)
```

The decompiler handles multi-way branching using `lax.switch` for more than 2 branches.

## Implementation Details

### Operation Mapping

Each StableHLO operation is mapped directly to its JAX equivalent:

```python
'stablehlo.add' -> lax.add(a, b)
'stablehlo.multiply' -> lax.mul(a, b)
'stablehlo.tanh' -> lax.tanh(a)
'stablehlo.dot_general' -> lax.dot_general(a, b, dimension_numbers)
```

### Constants

Constants are parsed from dense attributes and converted to JAX arrays:

```python
'dense<2.0> : tensor<f32>' -> jnp.array(2.0, dtype=jnp.float32)
```

### Shape Operations

Shape and dtype information is extracted from the result type:

```python
'stablehlo.reshape' -> lax.reshape(operand, new_shape)
```

## Benefits of This Approach

1. **Simplicity**: No need to understand jaxpr internals
2. **Transparency**: The decompiled code is just Python functions using JAX
3. **Debuggability**: Can print intermediate values, step through with a debugger
4. **Composability**: The returned functions can be used like any other JAX function
5. **Maintainability**: Much easier to add new operations - just map them to `jax.lax` calls

## Limitations

- Assumes well-formed StableHLO IR (which JAX always produces)
- Some advanced operations may need additional implementation
- Control flow assumes standard patterns (which JAX produces)

## Files

- `hlo_to_jaxpr.py` - Main decompiler implementation (refactored)
- `test_*.py` - Test files (may need updates for new API)

## Migration from Old API

Old API:
```python
func = decompiler.decompile_function(func_op)
result = core.eval_jaxpr(func.jaxpr, decompiler.constvals, *inputs)
```

New API:
```python
func = decompiler.decompile_function(func_op)
result = func.callable_fn(*inputs)
```

Much simpler!
