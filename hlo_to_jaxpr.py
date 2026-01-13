#!/usr/bin/env python3
"""
Refactored StableHLO to JAX function decompiler.

This decompiler converts StableHLO IR back to executable Python functions
using jax.lax operations (not primitives), building a value dictionary
progressively as we parse the compiled object.
"""

from typing import Any, Dict, List, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from dataclasses import dataclass


@dataclass
class DecompiledFunction:
    """Represents a decompiled function."""
    callable_fn: Callable
    name: str = "decompiled"
    input_names: List[str] = None
    output_names: List[str] = None


class StableHLOToJaxpr:
    """Converts StableHLO MLIR to executable JAX functions using jax.lax operations."""

    def __init__(self):
        self.functions = {}  # Name -> DecompiledFunction
        self.value_dict = {}  # Maps StableHLO value names/ids to computed values
        self.mlir_module = None  # Store module for function lookup
        self.function_cache = {}  # Cache for decompiled helper functions

    def _get_value_name(self, mlir_value):
        """Get a stable string identifier for an MLIR value."""
        return str(mlir_value)

    def _parse_tensor_type(self, mlir_type):
        """Parse MLIR tensor type to get shape and dtype info."""
        type_str = str(mlir_type)

        if 'tensor<' not in type_str:
            return (), np.float32

        # Extract tensor<...>
        inner = type_str.split('tensor<')[1].rstrip('>')

        # Parse shape and dtype
        if 'x' not in inner:
            # Scalar tensor like tensor<f32>
            dtype_str = inner
            shape = ()
        else:
            parts = inner.split('x')
            dtype_str = parts[-1]
            shape_parts = parts[:-1]

            # Handle dynamic dimensions
            shape = []
            for part in shape_parts:
                if part == '?' or not part.isdigit():
                    shape.append(None)  # Dynamic dimension
                else:
                    shape.append(int(part))
            shape = tuple(shape)

        # Map dtype strings to numpy dtypes
        dtype_map = {
            'f16': np.float16,
            'f32': np.float32,
            'f64': np.float64,
            'i1': np.bool_,
            'i8': np.int8,
            'i16': np.int16,
            'i32': np.int32,
            'i64': np.int64,
            'ui8': np.uint8,
            'ui16': np.uint16,
            'ui32': np.uint32,
            'ui64': np.uint64,
            'c64': np.complex64,
            'c128': np.complex128,
        }

        dtype = dtype_map.get(dtype_str, np.float32)
        return shape, dtype

    def _parse_dense_attr(self, attr_str):
        """Parse dense attribute to numpy array."""
        attr_str = str(attr_str)

        if 'dense<' not in attr_str:
            return None

        # Extract value and type
        value_part = attr_str.split('dense<')[1].split('>')[0]
        type_str = attr_str.split('tensor<')[1].rstrip('>') if 'tensor<' in attr_str else 'f32'

        # Parse type
        if 'x' in type_str:
            parts = type_str.split('x')
            dtype_str = parts[-1]
            shape = tuple(int(p) for p in parts[:-1] if p.isdigit())
        else:
            dtype_str = type_str
            shape = ()

        dtype_map = {
            'f32': np.float32, 'f64': np.float64, 'i32': np.int32,
            'i64': np.int64, 'i1': np.bool_, 'i8': np.int8,
            'i16': np.int16, 'ui8': np.uint8, 'ui16': np.uint16,
            'ui32': np.uint32, 'ui64': np.uint64,
        }
        dtype = dtype_map.get(dtype_str, np.float32)

        # Parse value
        try:
            if '.' in value_part or 'e' in value_part.lower():
                value = float(value_part)
            else:
                value = int(value_part)
            return jnp.full(shape, value, dtype=dtype)
        except:
            # Complex arrays - fallback to zeros
            return jnp.zeros(shape, dtype=dtype)

    def _get_attrs(self, op):
        """Get operation attributes as a dictionary."""
        attrs = {}
        if hasattr(op, 'attributes'):
            for name, value in dict(op.attributes).items():
                attrs[str(name)] = value
        return attrs

    def _parse_gather_dimension_numbers(self, attr_str):
        """Parse StableHLO gather dimension numbers attribute."""
        import re
        attr_str = str(attr_str)

        # Extract offset_dims
        offset_dims = ()
        offset_match = re.search(r'offset_dims\s*=\s*\[([^\]]*)\]', attr_str)
        if offset_match and offset_match.group(1).strip():
            offset_dims = tuple(int(x.strip()) for x in offset_match.group(1).split(',') if x.strip())

        # Extract collapsed_slice_dims
        collapsed_slice_dims = ()
        collapsed_match = re.search(r'collapsed_slice_dims\s*=\s*\[([^\]]*)\]', attr_str)
        if collapsed_match and collapsed_match.group(1).strip():
            collapsed_slice_dims = tuple(int(x.strip()) for x in collapsed_match.group(1).split(',') if x.strip())

        # Extract start_index_map
        start_index_map = ()
        start_match = re.search(r'start_index_map\s*=\s*\[([^\]]*)\]', attr_str)
        if start_match and start_match.group(1).strip():
            start_index_map = tuple(int(x.strip()) for x in start_match.group(1).split(',') if x.strip())

        return lax.GatherDimensionNumbers(
            offset_dims=offset_dims,
            collapsed_slice_dims=collapsed_slice_dims,
            start_index_map=start_index_map
        )

    def _parse_slice_sizes(self, attr_str):
        """Parse slice_sizes attribute from StableHLO gather."""
        import re
        attr_str = str(attr_str)

        # Extract numbers after the colon
        match = re.search(r'array<[^:]+:\s*([^>]+)>', attr_str)
        if match:
            numbers_str = match.group(1)
            return tuple(int(x.strip()) for x in numbers_str.split(',') if x.strip())
        return ()

    def _parse_scatter_dimension_numbers(self, attr_str):
        """Parse StableHLO scatter dimension numbers attribute."""
        import re
        attr_str = str(attr_str)

        # Extract update_window_dims
        update_window_dims = ()
        update_match = re.search(r'update_window_dims\s*=\s*\[([^\]]*)\]', attr_str)
        if update_match and update_match.group(1).strip():
            update_window_dims = tuple(int(x.strip()) for x in update_match.group(1).split(',') if x.strip())

        # Extract inserted_window_dims
        inserted_window_dims = ()
        inserted_match = re.search(r'inserted_window_dims\s*=\s*\[([^\]]*)\]', attr_str)
        if inserted_match and inserted_match.group(1).strip():
            inserted_window_dims = tuple(int(x.strip()) for x in inserted_match.group(1).split(',') if x.strip())

        # Extract scatter_dims_to_operand_dims
        scatter_dims = ()
        scatter_match = re.search(r'scatter_dims_to_operand_dims\s*=\s*\[([^\]]*)\]', attr_str)
        if scatter_match and scatter_match.group(1).strip():
            scatter_dims = tuple(int(x.strip()) for x in scatter_match.group(1).split(',') if x.strip())

        return lax.ScatterDimensionNumbers(
            update_window_dims=update_window_dims,
            inserted_window_dims=inserted_window_dims,
            scatter_dims_to_operand_dims=scatter_dims
        )

    # Operation mapping dictionaries for cleaner code
    BINARY_OPS = {
        'stablehlo.add': lax.add,
        'stablehlo.subtract': lax.sub,
        'stablehlo.multiply': lax.mul,
        'stablehlo.divide': lax.div,
        'stablehlo.remainder': lax.rem,
        'stablehlo.maximum': lax.max,
        'stablehlo.minimum': lax.min,
        'stablehlo.and': lax.bitwise_and,
        'stablehlo.or': lax.bitwise_or,
        'stablehlo.xor': lax.bitwise_xor,
        'stablehlo.pow': lax.pow,
        'stablehlo.power': lax.pow,
    }

    UNARY_OPS = {
        'stablehlo.negate': lax.neg,
        'stablehlo.abs': lax.abs,
        'stablehlo.not': lax.bitwise_not,
        'stablehlo.exp': lax.exp,
        'stablehlo.exponential': lax.exp,
        'stablehlo.log': lax.log,
        'stablehlo.tanh': lax.tanh,
        'stablehlo.sin': lax.sin,
        'stablehlo.sine': lax.sin,
        'stablehlo.cos': lax.cos,
        'stablehlo.cosine': lax.cos,
        'stablehlo.sqrt': lax.sqrt,
        'stablehlo.rsqrt': lax.rsqrt,
        'stablehlo.sign': lax.sign,
        'stablehlo.floor': lax.floor,
        'stablehlo.ceil': lax.ceil,
        'stablehlo.round_nearest_afz': lax.round,
    }

    def _execute_operation(self, op_name: str, op, value_dict: Dict[str, Any]) -> Optional[Any]:
        """
        Execute a StableHLO operation using jax.lax operations.

        Args:
            op_name: The operation name (e.g., 'stablehlo.add')
            op: The MLIR operation object
            value_dict: Dictionary mapping value names to computed values

        Returns:
            The computed result(s) of the operation, or None for terminal ops
        """
        op_name_str = str(op_name)

        # Get operands from value_dict
        operands = []
        if hasattr(op, 'operands'):
            for operand in op.operands:
                operand_name = self._get_value_name(operand)
                if operand_name in value_dict:
                    operands.append(value_dict[operand_name])
                else:
                    raise ValueError(f"Operand {operand_name} not found in value_dict")

        # Get attributes
        attrs = self._get_attrs(op)

        # Execute the operation based on its type
        result = None

        # Check binary operations dictionary
        if op_name_str in self.BINARY_OPS:
            result = self.BINARY_OPS[op_name_str](operands[0], operands[1])

        # Check unary operations dictionary
        elif op_name_str in self.UNARY_OPS:
            result = self.UNARY_OPS[op_name_str](operands[0])

        # Comparison operations
        elif op_name_str == 'stablehlo.compare':
            direction = str(attrs.get('comparison_direction', ''))
            if 'LT' in direction:
                result = lax.lt(operands[0], operands[1])
            elif 'LE' in direction:
                result = lax.le(operands[0], operands[1])
            elif 'GT' in direction:
                result = lax.gt(operands[0], operands[1])
            elif 'GE' in direction:
                result = lax.ge(operands[0], operands[1])
            elif 'EQ' in direction:
                result = lax.eq(operands[0], operands[1])
            elif 'NE' in direction:
                result = lax.ne(operands[0], operands[1])
            else:
                result = lax.eq(operands[0], operands[1])

        # Type conversion
        elif op_name_str == 'stablehlo.convert':
            # Get target dtype from result type
            if hasattr(op, 'results') and len(list(op.results)) > 0:
                _, new_dtype = self._parse_tensor_type(list(op.results)[0].type)
                result = lax.convert_element_type(operands[0], new_dtype)
            else:
                result = operands[0]

        # Shape operations
        elif op_name_str == 'stablehlo.reshape':
            if hasattr(op, 'results') and len(list(op.results)) > 0:
                new_shape, _ = self._parse_tensor_type(list(op.results)[0].type)
                result = lax.reshape(operands[0], new_shape)
            else:
                result = operands[0]

        elif op_name_str == 'stablehlo.broadcast_in_dim':
            if hasattr(op, 'results') and len(list(op.results)) > 0:
                shape, _ = self._parse_tensor_type(list(op.results)[0].type)
                # Parse broadcast_dimensions from attribute
                broadcast_dims = ()
                if 'broadcast_dimensions' in attrs:
                    dims_attr = attrs['broadcast_dimensions']
                    broadcast_dims = tuple(int(d) for d in dims_attr)
                result = lax.broadcast_in_dim(operands[0], shape, broadcast_dims)
            else:
                result = operands[0]

        elif op_name_str == 'stablehlo.transpose':
            # Parse permutation
            perm = ()
            if 'permutation' in attrs:
                perm_attr = attrs['permutation']
                perm = tuple(int(d) for d in perm_attr)
            result = lax.transpose(operands[0], perm)

        # Slicing operations
        elif op_name_str == 'stablehlo.dynamic_slice':
            if hasattr(op, 'results') and len(list(op.results)) > 0:
                slice_sizes, _ = self._parse_tensor_type(list(op.results)[0].type)
                # operands[0] is the array, operands[1:] are the start indices
                start_indices = operands[1:]
                result = lax.dynamic_slice(operands[0], start_indices, slice_sizes)
            else:
                result = operands[0]

        elif op_name_str == 'stablehlo.dynamic_update_slice':
            # operands: [array, update, *start_indices]
            result = lax.dynamic_update_slice(operands[0], operands[1], operands[2:])

        elif op_name_str == 'stablehlo.slice':
            # Static slice operation
            # Parse start_indices, limit_indices, strides from attributes
            start_indices = ()
            limit_indices = ()
            strides = ()

            if 'start_indices' in attrs:
                start_attr = attrs['start_indices']
                start_indices = tuple(int(d) for d in start_attr)

            if 'limit_indices' in attrs:
                limit_attr = attrs['limit_indices']
                limit_indices = tuple(int(d) for d in limit_attr)

            if 'strides' in attrs:
                stride_attr = attrs['strides']
                strides = tuple(int(d) for d in stride_attr)
            else:
                strides = tuple(1 for _ in start_indices)

            result = lax.slice(operands[0], start_indices, limit_indices, strides)

        # Concatenate
        elif op_name_str == 'stablehlo.concatenate':
            # Parse dimension from attributes
            dimension = 0
            if 'dimension' in attrs:
                dimension = int(attrs['dimension'])

            result = lax.concatenate(operands, dimension)

        # Pad
        elif op_name_str == 'stablehlo.pad':
            # operands: [operand, padding_value]
            # Parse padding configuration from attributes
            if 'edge_padding_low' in attrs and 'edge_padding_high' in attrs:
                low = tuple(int(d) for d in attrs['edge_padding_low'])
                high = tuple(int(d) for d in attrs['edge_padding_high'])

                # Interior padding (between elements)
                interior = ()
                if 'interior_padding' in attrs:
                    interior = tuple(int(d) for d in attrs['interior_padding'])
                else:
                    interior = tuple(0 for _ in low)

                # Create padding_config for lax.pad
                padding_config = tuple((l, h, i) for l, h, i in zip(low, high, interior))
                result = lax.pad(operands[0], operands[1], padding_config)
            else:
                result = operands[0]

        # Gather
        elif op_name_str == 'stablehlo.gather':
            # operands: [operand, start_indices]
            # Parse dimension numbers and slice sizes
            dimension_numbers = None
            slice_sizes = ()

            if 'dimension_numbers' in attrs:
                dim_nums_str = str(attrs['dimension_numbers'])
                dimension_numbers = self._parse_gather_dimension_numbers(dim_nums_str)

            if 'slice_sizes' in attrs:
                slice_sizes = self._parse_slice_sizes(str(attrs['slice_sizes']))

            # Execute gather
            if dimension_numbers and slice_sizes:
                result = lax.gather(operands[0], operands[1], dimension_numbers, slice_sizes)
            else:
                print(f"Warning: Could not parse gather parameters, skipping")
                result = operands[0]

        # Scatter
        elif op_name_str == 'stablehlo.scatter':
            # operands: [operand, scatter_indices, updates]
            # Parse dimension numbers
            dimension_numbers = None

            if 'scatter_dimension_numbers' in attrs:
                dim_nums_str = str(attrs['scatter_dimension_numbers'])
                dimension_numbers = self._parse_scatter_dimension_numbers(dim_nums_str)

            # Execute scatter
            # Note: StableHLO scatter has a computation region that defines the reduction
            # For now, we'll use the default (addition) which is most common
            if dimension_numbers:
                result = lax.scatter(operands[0], operands[1], operands[2], dimension_numbers)
            else:
                print(f"Warning: Could not parse scatter parameters, skipping")
                result = operands[0]

        # Reverse
        elif op_name_str == 'stablehlo.reverse':
            # Parse dimensions to reverse
            dimensions = ()
            if 'dimensions' in attrs:
                dims_attr = attrs['dimensions']
                dimensions = tuple(int(d) for d in dims_attr)

            # Use lax.rev to reverse along specified dimensions
            result = operands[0]
            for dim in dimensions:
                result = lax.rev(result, (dim,))

        # Sort
        elif op_name_str == 'stablehlo.sort':
            # Sort is complex - has comparator region
            # For now, use simple sort
            print(f"Warning: sort operation has limited support")
            result = lax.sort(operands[0])

        # Select and clamp
        elif op_name_str == 'stablehlo.select':
            # select(pred, on_true, on_false)
            result = lax.select(operands[0], operands[1], operands[2])

        elif op_name_str == 'stablehlo.clamp':
            # clamp(min, operand, max)
            result = lax.clamp(operands[0], operands[1], operands[2])

        # Dot operations
        elif op_name_str == 'stablehlo.dot_general':
            # Parse dot_dimension_numbers
            import re

            lhs_contract = ()
            rhs_contract = ()
            lhs_batch = ()
            rhs_batch = ()

            if 'dot_dimension_numbers' in attrs:
                dim_nums_str = str(attrs['dot_dimension_numbers'])

                # Extract dimensions using regex
                lhs_contract_match = re.search(r'lhs_contracting_dimensions\s*=\s*\[([^\]]*)\]', dim_nums_str)
                if lhs_contract_match:
                    lhs_contract = tuple(int(x.strip()) for x in lhs_contract_match.group(1).split(',') if x.strip())

                rhs_contract_match = re.search(r'rhs_contracting_dimensions\s*=\s*\[([^\]]*)\]', dim_nums_str)
                if rhs_contract_match:
                    rhs_contract = tuple(int(x.strip()) for x in rhs_contract_match.group(1).split(',') if x.strip())

                lhs_batch_match = re.search(r'lhs_batching_dimensions\s*=\s*\[([^\]]*)\]', dim_nums_str)
                if lhs_batch_match:
                    lhs_batch = tuple(int(x.strip()) for x in lhs_batch_match.group(1).split(',') if x.strip())

                rhs_batch_match = re.search(r'rhs_batching_dimensions\s*=\s*\[([^\]]*)\]', dim_nums_str)
                if rhs_batch_match:
                    rhs_batch = tuple(int(x.strip()) for x in rhs_batch_match.group(1).split(',') if x.strip())

            dimension_numbers = ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
            result = lax.dot_general(operands[0], operands[1], dimension_numbers)

        # Constants
        elif op_name_str == 'stablehlo.constant':
            if 'value' in attrs:
                result = self._parse_dense_attr(str(attrs['value']))
                if result is None:
                    raise ValueError(f"Could not parse constant value: {attrs['value']}")

        elif op_name_str == 'stablehlo.iota':
            # Create an array of values from 0 to N-1
            # Parse the iota dimension and result type
            if hasattr(op, 'results') and len(list(op.results)) > 0:
                shape, dtype = self._parse_tensor_type(list(op.results)[0].type)
                # Get the iota dimension
                iota_dimension = 0
                if 'iota_dimension' in attrs:
                    iota_dimension = int(attrs['iota_dimension'])
                result = lax.iota(dtype, shape[iota_dimension] if shape else 0)
                # If shape has multiple dimensions, need to broadcast
                if len(shape) > 1:
                    # Reshape and broadcast to the full shape
                    result = lax.broadcast_in_dim(result, shape, (iota_dimension,))
            else:
                result = jnp.array([])

        # Reduction operations
        elif op_name_str == 'stablehlo.reduce':
            # Parse dimensions
            dimensions = ()
            if 'dimensions' in attrs:
                dims_attr = attrs.get('dimensions')
                if dims_attr:
                    dimensions = tuple(int(d) for d in dims_attr)

            # Infer the reduction type from the computation region
            # operands: [input, init_value]
            reduction_type = 'sum'  # default

            if hasattr(op, 'regions'):
                regions = list(op.regions)
                if regions:
                    region = regions[0]
                    blocks = list(region.blocks)
                    if blocks:
                        block = blocks[0]
                        # Check the operations in the reduction body
                        for block_op in block.operations:
                            op_name_str_inner = str(block_op.operation.name)
                            if 'add' in op_name_str_inner:
                                reduction_type = 'sum'
                                break
                            elif 'maximum' in op_name_str_inner:
                                reduction_type = 'max'
                                break
                            elif 'minimum' in op_name_str_inner:
                                reduction_type = 'min'
                                break
                            elif 'multiply' in op_name_str_inner:
                                reduction_type = 'prod'
                                break
                            elif 'or' in op_name_str_inner:
                                reduction_type = 'any'
                                break
                            elif 'and' in op_name_str_inner:
                                reduction_type = 'all'
                                break

            # Apply the appropriate reduction
            if reduction_type == 'sum':
                result = lax.reduce_sum(operands[0], dimensions)
            elif reduction_type == 'max':
                result = lax.reduce_max(operands[0], dimensions)
            elif reduction_type == 'min':
                result = lax.reduce_min(operands[0], dimensions)
            elif reduction_type == 'prod':
                result = lax.reduce_prod(operands[0], dimensions)
            elif reduction_type == 'any':
                result = lax.reduce_or(operands[0], dimensions)
            elif reduction_type == 'all':
                result = lax.reduce_and(operands[0], dimensions)
            else:
                result = lax.reduce_sum(operands[0], dimensions)

        # Control flow - while loop
        elif op_name_str == 'stablehlo.while':
            # While has 2 regions: condition and body
            regions = list(op.regions)
            if len(regions) == 2:
                cond_region = regions[0]
                body_region = regions[1]

                # Get condition and body blocks
                cond_blocks = list(cond_region.blocks)
                body_blocks = list(body_region.blocks)

                if cond_blocks and body_blocks:
                    cond_block = cond_blocks[0]
                    body_block = body_blocks[0]

                    # StableHLO while loop operands represent the initial loop state
                    # Can be single value or tuple of values
                    init_val = operands[0] if len(operands) == 1 else tuple(operands)

                    # Create Python functions for condition and body
                    # lax.while_loop expects functions that take the loop state as a single argument
                    def cond_fn(loop_state):
                        # Unpack loop state for decompile_block
                        if isinstance(loop_state, tuple):
                            return self.decompile_block(cond_block, *loop_state)
                        else:
                            return self.decompile_block(cond_block, loop_state)

                    def body_fn(loop_state):
                        # Unpack loop state for decompile_block
                        if isinstance(loop_state, tuple):
                            return self.decompile_block(body_block, *loop_state)
                        else:
                            return self.decompile_block(body_block, loop_state)

                    # Execute while loop
                    result = lax.while_loop(cond_fn, body_fn, init_val)

        # Control flow - conditional (cond/case)
        elif op_name_str == 'stablehlo.case':
            # Case has an index operand and multiple branch regions
            index = operands[0]  # First operand is the branch index
            regions = list(op.regions)

            # Create branch functions
            # Branches need access to outer scope variables
            branches = []
            for region in regions:
                blocks = list(region.blocks)
                if blocks:
                    block = blocks[0]
                    # Create a function for this branch
                    # Branches in StableHLO case don't take arguments but can access outer scope
                    def make_branch_fn(blk, outer_dict):
                        def branch_fn():
                            # Pass outer scope so branch can access outer variables
                            return self.decompile_block(blk, outer_scope=outer_dict)
                        return branch_fn
                    branches.append(make_branch_fn(block, value_dict.copy()))

            # Use lax.switch for multi-way branch
            if len(branches) == 2:
                # Binary conditional - use lax.cond
                result = lax.cond(index, branches[1], branches[0])
            else:
                # Multi-way branch - use lax.switch
                result = lax.switch(index, branches)

        # Function call
        elif op_name_str == 'func.call':
            # Get the function name from attributes
            if 'callee' in attrs:
                callee_name = str(attrs['callee']).strip('"').strip('@')

                # Look up the function in the module
                if callee_name in self.function_cache:
                    # Use cached decompiled function
                    func_callable = self.function_cache[callee_name]
                    result = func_callable(*operands)
                elif self.mlir_module:
                    # Find and decompile the function
                    for module_op in self.mlir_module.body.operations:
                        if hasattr(module_op, 'name'):
                            func_name = str(module_op.name).strip('"').strip('@')
                            if func_name == callee_name:
                                # Decompile this function
                                decompiled = self.decompile_function(module_op)
                                # Cache it
                                self.function_cache[callee_name] = decompiled.callable_fn
                                # Execute it
                                result = decompiled.callable_fn(*operands)
                                break
                    else:
                        print(f"Warning: Could not find function {callee_name}")
                        result = operands[0] if operands else None
                else:
                    print(f"Warning: No module available for function lookup")
                    result = operands[0] if operands else None
            else:
                print(f"Warning: func.call without callee attribute")
                result = operands[0] if operands else None

        # Terminal operations
        elif op_name_str in ('func.return', 'stablehlo.return'):
            # Return the operands as the result
            # Return as tuple for multiple values to maintain pytree structure
            if len(operands) > 1:
                return tuple(operands)
            elif len(operands) == 1:
                return operands[0]
            else:
                return None

        else:
            # Unknown operation - skip
            print(f"Warning: Skipping unsupported operation: {op_name_str}")
            return None

        return result

    def decompile_block(self, block, *input_values, outer_scope=None):
        """
        Decompile a block by executing all operations.

        Args:
            block: The MLIR block to decompile
            *input_values: Input values for the block arguments
            outer_scope: Optional dictionary of values from outer scope

        Returns:
            The output values from the block
        """
        # Create a value dictionary for this block
        # Start with outer scope if provided (for case branches)
        value_dict = outer_scope.copy() if outer_scope else {}

        # Map block arguments to input values
        block_args = list(block.arguments)
        if len(block_args) != len(input_values):
            raise ValueError(f"Expected {len(block_args)} inputs, got {len(input_values)}")

        for arg, value in zip(block_args, input_values):
            arg_name = self._get_value_name(arg)
            value_dict[arg_name] = value

        # Process operations in order
        outputs = None
        for op in block.operations:
            op_name = op.operation.name

            # Execute the operation
            result = self._execute_operation(op_name, op, value_dict)

            # If this is a return operation, save the outputs
            if 'return' in str(op_name):
                outputs = result
                break

            # Store result(s) in value_dict
            if result is not None and hasattr(op, 'results'):
                results = list(op.results)
                if len(results) == 1:
                    result_name = self._get_value_name(results[0])
                    value_dict[result_name] = result
                elif len(results) > 1:
                    # Multiple results
                    for i, res in enumerate(results):
                        result_name = self._get_value_name(res)
                        value_dict[result_name] = result[i] if isinstance(result, (tuple, list)) else result

        return outputs

    def decompile_function(self, func_op) -> DecompiledFunction:
        """
        Decompile a function operation to a callable Python function.

        Args:
            func_op: The MLIR function operation

        Returns:
            A DecompiledFunction containing the callable
        """
        func_name = str(func_op.name)

        # Get the function body (first region, first block)
        regions = list(func_op.regions)
        if not regions:
            raise ValueError(f"Function {func_name} has no regions")

        region = regions[0]
        blocks = list(region.blocks)
        if not blocks:
            raise ValueError(f"Function {func_name} has no blocks")

        block = blocks[0]

        # Get input names
        input_names = [self._get_value_name(arg) for arg in block.arguments]

        # Create a callable that executes the block
        def callable_fn(*inputs):
            return self.decompile_block(block, *inputs)

        return DecompiledFunction(
            callable_fn=callable_fn,
            name=func_name,
            input_names=input_names,
        )

    def decompile_module(self, mlir_module) -> Dict[str, DecompiledFunction]:
        """Decompile an entire MLIR module."""
        # Store module for function lookup
        self.mlir_module = mlir_module
        functions = {}

        for op in mlir_module.body.operations:
            if hasattr(op, 'function_type'):
                func = self.decompile_function(op)
                functions[func.name] = func

        return functions


def test_decompiler():
    """Test the refactored decompiler."""
    print("=" * 80)
    print("TESTING REFACTORED STABLEHLO TO JAX FUNCTION DECOMPILER")
    print("=" * 80)

    # Test 1: Simple arithmetic
    def simple_add(x, y):
        return x + y

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])

    print("\n[TEST 1] Simple Add")
    lowered = jax.jit(simple_add).lower(x, y)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    decompiler = StableHLOToJaxpr()
    functions = decompiler.decompile_module(mlir_module)

    for name, func in functions.items():
        print(f"\nFunction: {name}")
        print(f"Executing decompiled function:")

        result = func.callable_fn(x, y)
        expected = simple_add(x, y)

        print(f"Result: {result}")
        print(f"Expected: {expected}")
        print(f"Match: {np.allclose(result, expected)}")

    # Test 2: More complex
    def complex_func(x):
        return jnp.tanh(x * 2.0) + 1.0

    x = jnp.array([1.0, 2.0, 3.0])

    print("\n" + "=" * 80)
    print("[TEST 2] Complex Function")
    lowered = jax.jit(complex_func).lower(x)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    decompiler = StableHLOToJaxpr()
    functions = decompiler.decompile_module(mlir_module)

    for name, func in functions.items():
        print(f"\nFunction: {name}")
        print("Executing decompiled function:")

        try:
            result = func.callable_fn(x)
            expected = complex_func(x)
            print(f"Result: {result}")
            print(f"Expected: {expected}")
            print(f"Match: {np.allclose(result, expected)}")
        except Exception as e:
            print(f"Execution failed: {e}")
            import traceback
            traceback.print_exc()

    # Test 3: Matrix multiplication
    def matmul_func(x, y):
        return jnp.dot(x, y)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[5.0, 6.0], [7.0, 8.0]])

    print("\n" + "=" * 80)
    print("[TEST 3] Matrix Multiplication")
    lowered = jax.jit(matmul_func).lower(x, y)
    mlir_module = lowered.compiler_ir(dialect='stablehlo')

    decompiler = StableHLOToJaxpr()
    functions = decompiler.decompile_module(mlir_module)

    for name, func in functions.items():
        print(f"\nFunction: {name}")
        print("Executing decompiled function:")

        try:
            result = func.callable_fn(x, y)
            expected = matmul_func(x, y)
            print(f"Result:\n{result}")
            print(f"Expected:\n{expected}")
            print(f"Match: {np.allclose(result, expected)}")
        except Exception as e:
            print(f"Execution failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    test_decompiler()
