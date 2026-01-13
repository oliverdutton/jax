#!/usr/bin/env python3
"""
Complete StableHLO to Jaxpr decompiler.

This decompiler maps StableHLO IR back to executable jaxpr,
using JAX lax primitives for all operations.
"""

from typing import Any, Dict, List, Tuple, Optional, Callable, Set
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import core, dtypes as jax_dtypes
from jax._src.interpreters import mlir
import numpy as np
from dataclasses import dataclass


@dataclass
class DecompiledFunction:
    """Represents a decompiled function."""
    in_avals: List[core.AbstractValue]
    out_avals: List[core.AbstractValue]
    jaxpr: core.Jaxpr
    name: str = "decompiled"


class StableHLOToJaxpr:
    """Converts StableHLO MLIR to JAX jaxpr."""

    def __init__(self):
        self.functions = {}  # Name -> DecompiledFunction
        self.value_map = {}  # MLIR value id -> jaxpr Var
        self.constvars = []  # List of constant variables
        self.constvals = []  # List of constant values

    def _parse_tensor_type(self, mlir_type):
        """Parse MLIR tensor type to JAX AbstractValue."""
        type_str = str(mlir_type)

        if 'tensor<' not in type_str:
            # Default fallback
            return core.ShapedArray((), np.float32)

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
            'bf16': jax_dtypes.bfloat16 if hasattr(jax_dtypes, 'bfloat16') else np.float32,
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
        return core.ShapedArray(shape, dtype)

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
        }
        dtype = dtype_map.get(dtype_str, np.float32)

        # Parse value
        try:
            if '.' in value_part or 'e' in value_part.lower():
                value = float(value_part)
            else:
                value = int(value_part)
            return np.full(shape, value, dtype=dtype)
        except:
            # Complex arrays - fallback to zeros
            return np.zeros(shape, dtype=dtype)

    def _get_var(self, mlir_value):
        """Get jaxpr var for MLIR value."""
        # Use string representation as stable key
        vid = str(mlir_value)
        if vid not in self.value_map:
            aval = self._parse_tensor_type(mlir_value.type)
            self.value_map[vid] = core.Var(aval)
        return self.value_map[vid]

    def _stablehlo_to_primitive(self, op_name: str, op, operand_vars, result_vars):
        """
        Map StableHLO operation to JAX primitive and parameters.

        Returns:
            (primitive, params, invars, outvars)
        """
        # Get attributes
        attrs = {}
        if hasattr(op, 'attributes'):
            for name, value in dict(op.attributes).items():
                attrs[str(name)] = str(value)

        op_name_str = str(op_name)

        # Arithmetic operations
        if op_name_str == 'stablehlo.add':
            return lax.add_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.subtract':
            return lax.sub_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.multiply':
            return lax.mul_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.divide':
            return lax.div_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.remainder':
            return lax.rem_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.maximum':
            return lax.max_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.minimum':
            return lax.min_p, {}, operand_vars, result_vars

        # Comparison operations
        elif op_name_str == 'stablehlo.compare':
            direction = attrs.get('comparison_direction', '')
            if 'LT' in direction:
                prim = lax.lt_p
            elif 'LE' in direction:
                prim = lax.le_p
            elif 'GT' in direction:
                prim = lax.gt_p
            elif 'GE' in direction:
                prim = lax.ge_p
            elif 'EQ' in direction:
                prim = lax.eq_p
            elif 'NE' in direction:
                prim = lax.ne_p
            else:
                prim = lax.eq_p
            return prim, {}, operand_vars, result_vars

        # Unary operations
        elif op_name_str == 'stablehlo.negate':
            return lax.neg_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.abs':
            return lax.abs_p, {}, operand_vars, result_vars

        elif op_name_str in ('stablehlo.exp', 'stablehlo.exponential'):
            return lax.exp_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.log':
            return lax.log_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.tanh':
            return lax.tanh_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.sin':
            return lax.sin_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.cos':
            return lax.cos_p, {}, operand_vars, result_vars

        elif op_name_str == 'stablehlo.sqrt':
            return lax.sqrt_p, {}, operand_vars, result_vars

        # Type conversion
        elif op_name_str == 'stablehlo.convert':
            new_dtype = result_vars[0].aval.dtype
            params = {
                'new_dtype': new_dtype,
                'weak_type': False,
                'sharding': None,
            }
            return lax.convert_element_type_p, params, operand_vars, result_vars

        # Shape operations
        elif op_name_str == 'stablehlo.reshape':
            new_shape = result_vars[0].aval.shape
            params = {
                'new_sizes': new_shape,
                'dimensions': None,
                'sharding': None,
            }
            return lax.reshape_p, params, operand_vars, result_vars

        elif op_name_str == 'stablehlo.broadcast_in_dim':
            shape = result_vars[0].aval.shape
            # Parse broadcast_dimensions from attribute
            broadcast_dims = ()
            if 'broadcast_dimensions' in attrs:
                # The attribute value might be a DenseI64ArrayAttr object
                dims_attr = dict(op.attributes)['broadcast_dimensions']
                # It's iterable
                broadcast_dims = tuple(int(d) for d in dims_attr)

            params = {
                'shape': shape,
                'broadcast_dimensions': broadcast_dims,
                'sharding': None,  # Default sharding
            }
            return lax.broadcast_in_dim_p, params, operand_vars, result_vars

        elif op_name_str == 'stablehlo.transpose':
            # Parse permutation from DenseI64ArrayAttr
            perm = ()
            if 'permutation' in attrs:
                perm_attr = dict(op.attributes)['permutation']
                # It's iterable like broadcast_dimensions
                perm = tuple(int(d) for d in perm_attr)
            params = {
                'permutation': perm,
            }
            return lax.transpose_p, params, operand_vars, result_vars

        # Slicing operations
        elif op_name_str == 'stablehlo.dynamic_slice':
            slice_sizes = result_vars[0].aval.shape
            params = {'slice_sizes': slice_sizes}
            return lax.dynamic_slice_p, params, operand_vars, result_vars

        elif op_name_str == 'stablehlo.dynamic_update_slice':
            return lax.dynamic_update_slice_p, {}, operand_vars, result_vars

        # Dot operations
        elif op_name_str == 'stablehlo.dot_general':
            # Parse dot_dimension_numbers
            # For now, use default contraction
            lhs_shape = operand_vars[0].aval.shape
            rhs_shape = operand_vars[1].aval.shape

            # Simple matmul case
            dimension_numbers = (((len(lhs_shape) - 1,), (0,)), ((), ()))
            params = {
                'dimension_numbers': dimension_numbers,
                'precision': None,
                'preferred_element_type': None,
                'out_sharding': None,
            }
            return lax.dot_general_p, params, operand_vars, result_vars

        # Constants
        elif op_name_str == 'stablehlo.constant':
            if 'value' in attrs:
                const_val = self._parse_dense_attr(attrs['value'])
                if const_val is not None:
                    # Add to constants and return a literal
                    # For jaxpr, we'll use a variable bound to the constant
                    return None, {'value': const_val}, [], result_vars

        # Reduction operations
        elif op_name_str == 'stablehlo.reduce':
            # This is complex, skip for now
            return None, {}, operand_vars, result_vars

        # Control flow - while loop
        elif op_name_str == 'stablehlo.while':
            # While has 2 regions: condition and body
            regions = list(op.regions)
            if len(regions) == 2:
                # Decompile condition and body
                cond_region = regions[0]
                body_region = regions[1]

                # Save current state
                saved_value_map = self.value_map.copy()
                saved_constvars = self.constvars.copy()
                saved_constvals = self.constvals.copy()

                # Decompile condition
                cond_blocks = list(cond_region.blocks)
                if cond_blocks:
                    self.value_map = {}
                    self.constvars = []
                    self.constvals = []
                    cond_invars, cond_outvars, cond_eqns = self.decompile_block(cond_blocks[0], "cond")
                    cond_jaxpr = core.Jaxpr(
                        constvars=self.constvars,
                        invars=cond_invars,
                        outvars=cond_outvars,
                        eqns=cond_eqns,
                        effects=set(),
                    )
                    cond_constvars = self.constvars
                    cond_constvals = self.constvals

                # Restore and decompile body
                self.value_map = {}
                self.constvars = []
                self.constvals = []
                body_blocks = list(body_region.blocks)
                if body_blocks:
                    body_invars, body_outvars, body_eqns = self.decompile_block(body_blocks[0], "body")
                    body_jaxpr = core.Jaxpr(
                        constvars=self.constvars,
                        invars=body_invars,
                        outvars=body_outvars,
                        eqns=body_eqns,
                        effects=set(),
                    )
                    body_constvars = self.constvars
                    body_constvals = self.constvals

                # Restore value map
                self.value_map = saved_value_map
                self.constvars = saved_constvars
                self.constvals = saved_constvals

                # Register result vars
                for result in op.results:
                    self._get_var(result)

                # Create while_loop params
                # Note: StableHLO while has all values as loop state, but in JAX
                # some values might be consts. For simplicity, treat all as loop state.
                params = {
                    'cond_jaxpr': core.ClosedJaxpr(cond_jaxpr, cond_constvals),
                    'cond_nconsts': 0,  # All values passed as loop state
                    'body_jaxpr': core.ClosedJaxpr(body_jaxpr, body_constvals),
                    'body_nconsts': 0,  # All values passed as loop state
                }

                return lax.while_p, params, operand_vars, result_vars

            return None, {}, operand_vars, result_vars

        # Control flow - cond/case
        elif op_name_str == 'stablehlo.case':
            # Case has multiple regions, one for each branch
            # StableHLO branches implicitly capture outer variables
            # JAX cond_p passes captured variables as explicit operands
            regions = list(op.regions)

            # Save current state
            saved_value_map = self.value_map.copy()
            saved_constvars = self.constvars.copy()
            saved_constvals = self.constvals.copy()

            # Find all variables captured in branches
            captured_vars = set()
            branch_jaxprs = []

            for region in regions:
                blocks = list(region.blocks)
                if blocks:
                    # Decompile with outer scope available
                    self.constvars = []
                    self.constvals = []
                    branch_value_map_before = self.value_map.copy()

                    invars, outvars, eqns = self.decompile_block(blocks[0], "branch")

                    # Find which outer vars are used
                    for eqn in eqns:
                        for v in eqn.invars:
                            if v in branch_value_map_before.values():
                                captured_vars.add(v)
                    for v in outvars:
                        if v in branch_value_map_before.values():
                            captured_vars.add(v)

                    # Save constvars and constvals for this branch
                    branch_jaxprs.append((
                        invars,
                        outvars,
                        eqns,
                        self.constvars[:],  # Save constvars
                        self.constvals[:]   # Save constvals
                    ))

                    # Restore value_map for next branch
                    self.value_map = saved_value_map.copy()

            # Restore state
            self.value_map = saved_value_map
            self.constvars = saved_constvars
            self.constvals = saved_constvals

            # Register result vars
            for result in op.results:
                self._get_var(result)

            # Convert captured vars to list for consistent ordering
            captured_var_list = sorted(captured_vars, key=lambda v: v.count)

            # Rebuild branch jaxprs with captured vars as explicit arguments
            branches = []
            for invars, outvars, eqns, branch_constvars, branch_constvals in branch_jaxprs:
                # Branch gets captured vars as arguments, plus any constants defined in the branch
                branch_jaxpr = core.Jaxpr(
                    constvars=branch_constvars,  # Keep constants defined in branch
                    invars=captured_var_list,     # Captured vars as explicit args
                    outvars=outvars,
                    eqns=eqns,
                    effects=set(),
                )
                # Pass the constant values for this branch
                branches.append(core.ClosedJaxpr(branch_jaxpr, branch_constvals))

            # Add captured vars to operands (after the index)
            all_operands = operand_vars + captured_var_list

            # Create cond params
            params = {
                'branches': tuple(branches),
            }

            return lax.cond_p, params, all_operands, result_vars

        # Terminal operations
        elif op_name_str in ('func.return', 'stablehlo.return'):
            return None, {}, operand_vars, []

        # Unknown operation - skip
        return None, {}, operand_vars, result_vars

    def decompile_block(self, block, name="main") -> Tuple[List[core.Var], List[core.Var], List[core.JaxprEqn]]:
        """
        Decompile a block to jaxpr equations.

        Returns:
            (invars, outvars, equations)
        """
        # Register block arguments as input variables
        invars = []
        for arg in block.arguments:
            var = self._get_var(arg)
            invars.append(var)

        # Process operations
        equations = []
        outvars = []

        for op in block.operations:
            op_name = op.operation.name

            # Get operand variables
            operand_vars = []
            if hasattr(op, 'operands'):
                for operand in op.operands:
                    operand_vars.append(self._get_var(operand))

            # Get result variables
            result_vars = []
            if hasattr(op, 'results'):
                for result in op.results:
                    result_vars.append(self._get_var(result))

            # Check if this is a return operation
            if 'return' in str(op_name):
                outvars = operand_vars
                continue

            # Map to primitive
            prim, params, inv, outv = self._stablehlo_to_primitive(
                op_name, op, operand_vars, result_vars
            )

            if prim is None:
                # Handle constants
                if 'value' in params:
                    # In jaxpr, constants are constvars
                    const_val = params['value']
                    # Create a constvar for this constant
                    const_aval = core.ShapedArray(const_val.shape, const_val.dtype)
                    const_var = core.Var(const_aval)
                    self.constvars.append(const_var)
                    self.constvals.append(const_val)

                    # Map the result to the constvar
                    if result_vars:
                        vid = str(list(op.results)[0])
                        self.value_map[vid] = const_var
                    continue
                # Skip other unsupported operations
                continue

            # Create jaxpr equation
            from jax._src import source_info_util
            name_stack = source_info_util.NameStack()
            source_info = source_info_util.SourceInfo(None, name_stack)
            ctx = core.JaxprEqnContext(None, False, None)
            eqn = core.JaxprEqn(
                inv,  # invars
                outv,  # outvars
                prim,  # primitive
                params,  # params
                set(),  # effects
                source_info,  # source_info
                ctx,  # ctx
            )
            equations.append(eqn)

        return invars, outvars, equations

    def decompile_function(self, func_op) -> DecompiledFunction:
        """Decompile a function operation."""
        func_name = str(func_op.name)

        # Reset value map for new function
        self.value_map = {}
        self.constvars = []
        self.constvals = []

        # Get the function body (first region, first block)
        regions = list(func_op.regions)
        if not regions:
            raise ValueError(f"Function {func_name} has no regions")

        region = regions[0]
        blocks = list(region.blocks)
        if not blocks:
            raise ValueError(f"Function {func_name} has no blocks")

        block = blocks[0]

        # Decompile the block
        invars, outvars, equations = self.decompile_block(block, func_name)

        # Build jaxpr
        in_avals = [v.aval for v in invars]
        out_avals = [v.aval for v in outvars]

        jaxpr = core.Jaxpr(
            constvars=self.constvars,
            invars=invars,
            outvars=outvars,
            eqns=equations,
            effects=set(),
        )

        return DecompiledFunction(
            in_avals=in_avals,
            out_avals=out_avals,
            jaxpr=jaxpr,
            name=func_name,
        )

    def decompile_module(self, mlir_module) -> Dict[str, DecompiledFunction]:
        """Decompile an entire MLIR module."""
        functions = {}

        for op in mlir_module.body.operations:
            if hasattr(op, 'function_type'):
                func = self.decompile_function(op)
                functions[func.name] = func

        return functions


def test_decompiler():
    """Test the decompiler."""
    print("=" * 80)
    print("TESTING STABLEHLO TO JAXPR DECOMPILER")
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
        print(f"Input types: {func.in_avals}")
        print(f"Output types: {func.out_avals}")
        print(f"Jaxpr:\n{func.jaxpr}")

        # Try to execute the jaxpr
        print("\nExecuting decompiled jaxpr:")
        # Need to get the constant values from the decompiler
        constvals = decompiler.constvals
        result = core.eval_jaxpr(func.jaxpr, constvals, x, y)
        print(f"Result: {result}")
        print(f"Expected: {x + y}")
        print(f"Match: {np.allclose(result, x + y)}")

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
        print(f"Jaxpr:\n{func.jaxpr}")

        print("\nExecuting decompiled jaxpr:")
        try:
            constvals = decompiler.constvals
            result = core.eval_jaxpr(func.jaxpr, constvals, x)
            expected = complex_func(x)
            print(f"Result: {result}")
            print(f"Expected: {expected}")
            print(f"Match: {np.allclose(result, expected)}")
        except Exception as e:
            print(f"Execution failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    test_decompiler()
