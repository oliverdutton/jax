#!/usr/bin/env python3
"""StableHLO to Jaxpr decompiler."""

from typing import Any, Dict, List, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import core, dtypes
import numpy as np
from collections import defaultdict


class HLODecompiler:
    """Decompiles StableHLO IR to JAX jaxpr."""

    def __init__(self, mlir_module):
        self.module = mlir_module
        self.value_map = {}  # Maps MLIR values to jaxpr variables
        self.constants = {}  # Maps constant values
        self.var_counter = 0

    def _parse_mlir_type(self, mlir_type_str):
        """Parse MLIR tensor type to JAX abstract value."""
        type_str = str(mlir_type_str)

        # Parse shape and dtype from strings like "tensor<3xf32>"
        if 'tensor<' in type_str:
            inner = type_str.split('tensor<')[1].rstrip('>')

            # Handle scalar tensors
            if 'x' not in inner:
                # Just dtype, scalar
                dtype_str = inner
                shape = ()
            else:
                parts = inner.split('x')
                # Last part is dtype
                dtype_str = parts[-1]
                # Everything else is shape
                shape = tuple(int(p) if p.isdigit() else None for p in parts[:-1])

            # Map dtype
            dtype_map = {
                'f32': np.float32,
                'f64': np.float64,
                'i32': np.int32,
                'i64': np.int64,
                'i1': np.bool_,
                'i8': np.int8,
                'i16': np.int16,
            }
            dtype = dtype_map.get(dtype_str, np.float32)

            return core.ShapedArray(shape, dtype)

        # Fallback
        return core.ShapedArray((), np.float32)

    def _fresh_var(self, aval):
        """Create a fresh jaxpr variable."""
        var = core.Var(aval)
        self.var_counter += 1
        return var

    def _get_or_create_var(self, mlir_value, aval=None):
        """Get existing var for MLIR value or create new one."""
        # Use a unique identifier for the value
        value_id = id(mlir_value)

        if value_id not in self.value_map:
            if aval is None:
                aval = self._parse_mlir_type(mlir_value.type)
            var = self._fresh_var(aval)
            self.value_map[value_id] = var

        return self.value_map[value_id]

    def _parse_dense_elements_attr(self, attr_str):
        """Parse dense<...> attributes to numpy arrays."""
        attr_str = str(attr_str)

        # Handle dense<value> : tensor<...>
        if 'dense<' in attr_str:
            # Extract the value part
            value_part = attr_str.split('dense<')[1].split('>')[0]

            # Extract type part
            type_part = attr_str.split('tensor<')[1].rstrip('>')

            # Parse type to get shape and dtype
            if 'x' in type_part:
                parts = type_part.split('x')
                dtype_str = parts[-1]
                shape = tuple(int(p) for p in parts[:-1] if p.isdigit())
            else:
                dtype_str = type_part
                shape = ()

            # Map dtype
            dtype_map = {
                'f32': np.float32,
                'f64': np.float64,
                'i32': np.int32,
                'i64': np.int64,
                'i1': np.bool_,
            }
            dtype = dtype_map.get(dtype_str, np.float32)

            # Parse value
            try:
                # Handle scientific notation and special floats
                if 'e' in value_part.lower() or '.' in value_part:
                    value = float(value_part)
                else:
                    value = int(value_part)

                # Create array
                arr = np.full(shape, value, dtype=dtype)
                return arr
            except:
                # Fallback for complex dense values - return zero array
                return np.zeros(shape, dtype=dtype)

        return None

    def decompile_operation(self, op, inputs: List[core.Var]) -> Tuple[Optional[str], List[core.Var], Dict]:
        """
        Decompile a single operation.

        Returns:
            (primitive_name, output_vars, params)
        """
        op_name = str(op.operation.name)

        # Get operands (inputs)
        operand_vars = []
        if hasattr(op, 'operands'):
            for operand in op.operands:
                operand_id = id(operand)
                if operand_id in self.value_map:
                    operand_vars.append(self.value_map[operand_id])

        # Get results (outputs)
        result_vars = []
        if hasattr(op, 'results'):
            for result in op.results:
                aval = self._parse_mlir_type(result.type)
                result_var = self._get_or_create_var(result, aval)
                result_vars.append(result_var)

        # Get attributes
        params = {}
        if hasattr(op, 'attributes'):
            for name, value in dict(op.attributes).items():
                params[str(name)] = str(value)

        # Map StableHLO operations to lax primitives
        primitive_name = None

        if op_name == 'stablehlo.add':
            primitive_name = 'add'
        elif op_name == 'stablehlo.subtract':
            primitive_name = 'sub'
        elif op_name == 'stablehlo.multiply':
            primitive_name = 'mul'
        elif op_name == 'stablehlo.divide':
            primitive_name = 'div'
        elif op_name == 'stablehlo.dot':
            primitive_name = 'dot_general'
        elif op_name == 'stablehlo.dot_general':
            primitive_name = 'dot_general'
        elif op_name == 'stablehlo.compare':
            # Map comparison direction
            direction = params.get('comparison_direction', '')
            if 'LT' in direction:
                primitive_name = 'lt'
            elif 'LE' in direction:
                primitive_name = 'le'
            elif 'GT' in direction:
                primitive_name = 'gt'
            elif 'GE' in direction:
                primitive_name = 'ge'
            elif 'EQ' in direction:
                primitive_name = 'eq'
            elif 'NE' in direction:
                primitive_name = 'ne'
        elif op_name == 'stablehlo.convert':
            primitive_name = 'convert_element_type'
            # Get target dtype from result type
            if result_vars:
                params['new_dtype'] = result_vars[0].aval.dtype
        elif op_name == 'stablehlo.constant':
            primitive_name = 'constant'
            # Parse the constant value
            if 'value' in params:
                const_val = self._parse_dense_elements_attr(params['value'])
                params['value'] = const_val
        elif op_name == 'stablehlo.broadcast_in_dim':
            primitive_name = 'broadcast_in_dim'
            # Parse broadcast_dimensions
            if 'broadcast_dimensions' in params:
                # Extract dimensions from array<i64: ...>
                dims_str = params['broadcast_dimensions']
                params['broadcast_dimensions'] = ()  # Will need proper parsing
            if result_vars:
                params['shape'] = result_vars[0].aval.shape
        elif op_name == 'stablehlo.reshape':
            primitive_name = 'reshape'
            if result_vars:
                params['new_sizes'] = result_vars[0].aval.shape
        elif op_name == 'stablehlo.dynamic_slice':
            primitive_name = 'dynamic_slice'
            if 'slice_sizes' in params:
                # Parse slice sizes
                params['slice_sizes'] = result_vars[0].aval.shape if result_vars else ()
        elif op_name == 'stablehlo.dynamic_update_slice':
            primitive_name = 'dynamic_update_slice'
        elif op_name == 'stablehlo.negate':
            primitive_name = 'neg'
        elif op_name == 'stablehlo.tanh':
            primitive_name = 'tanh'
        elif op_name == 'stablehlo.exp':
            primitive_name = 'exp'
        elif op_name == 'stablehlo.log':
            primitive_name = 'log'
        elif op_name == 'stablehlo.reduce':
            primitive_name = 'reduce'
        elif op_name in ('func.return', 'stablehlo.return'):
            # Terminal operations - don't map to primitives
            return None, operand_vars, params

        return primitive_name, result_vars, params, operand_vars

    def decompile_function(self, func_op):
        """Decompile a function operation to a callable."""
        # Get function signature
        func_type = func_op.function_type
        print(f"\nDecompiling function: {func_op.name}")
        print(f"Type: {func_type}")

        # Create input variables
        input_vars = []
        for region in func_op.regions:
            for block in region.blocks:
                for arg in block.arguments:
                    aval = self._parse_mlir_type(arg.type)
                    var = self._get_or_create_var(arg, aval)
                    input_vars.append(var)

                # Process operations
                equations = []
                output_vars = []

                for op in block.operations:
                    result = self.decompile_operation(op, input_vars)
                    if result[0] is not None:  # Not a return operation
                        prim_name, out_vars, params, in_vars = result

                        print(f"  {prim_name}: {[v.aval for v in in_vars]} -> {[v.aval for v in out_vars]}")

                        # Create equation (simplified - would need proper primitive lookup)
                        # equations.append((prim_name, in_vars, out_vars, params))
                    else:
                        # Return operation
                        _, return_vars, _ = result
                        output_vars = return_vars

                return input_vars, output_vars, equations

        return None


def test_decompiler():
    """Test the decompiler on simple functions."""
    print("=" * 80)
    print("TESTING STABLEHLO DECOMPILER")
    print("=" * 80)

    # Test 1: Simple add
    def simple_add(x, y):
        return x + y

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])

    lowered = jax.jit(simple_add).lower(x, y)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    decompiler = HLODecompiler(mlir_module)

    for op in mlir_module.body.operations:
        if hasattr(op, 'function_type'):
            decompiler.decompile_function(op)


if __name__ == '__main__':
    test_decompiler()
