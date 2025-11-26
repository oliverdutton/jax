"""Pallas NumPy interpreter using io_callback."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax._src import core as jax_core
from jax._src import lax_reference
from jax._src.lax import slicing as lax_slicing
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives as pallas_primitives
from jax._src.pallas import pallas_call
from jax._src import state
from jax.experimental import io_callback


# Numpy implementations for common JAX primitives
# Most of these come from lax_reference.py
_NUMPY_IMPL = {
    # Arithmetic
    'add': np.add,
    'sub': np.subtract,
    'mul': np.multiply,
    'div': lax_reference.div,
    'rem': lax_reference.rem,
    'max': np.maximum,
    'min': np.minimum,
    'neg': np.negative,
    'sign': np.sign,
    'abs': np.absolute,
    'pow': np.power,
    'sqrt': np.sqrt,
    'rsqrt': lax_reference.rsqrt,
    'cbrt': np.cbrt,
    'square': np.square,
    'reciprocal': np.reciprocal,

    # Trigonometric
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'asin': np.arcsin,
    'acos': np.arccos,
    'atan': np.arctan,
    'atan2': np.arctan2,
    'sinh': np.sinh,
    'cosh': np.cosh,
    'tanh': np.tanh,
    'asinh': np.arcsinh,
    'acosh': np.arccosh,
    'atanh': np.arctanh,

    # Exponential/logarithmic
    'exp': np.exp,
    'exp2': np.exp2,
    'expm1': np.expm1,
    'log': np.log,
    'log1p': np.log1p,

    # Comparison
    'eq': np.equal,
    'ne': np.not_equal,
    'ge': np.greater_equal,
    'gt': np.greater,
    'le': np.less_equal,
    'lt': np.less,

    # Logical (JAX returns int32, not bool)
    'and': lambda x, y: np.logical_and(x, y).astype(np.int32),
    'or': lambda x, y: np.logical_or(x, y).astype(np.int32),
    'not': lambda x: np.logical_not(x).astype(np.int32),
    'xor': lambda x, y: np.logical_xor(x, y).astype(np.int32),

    # Bitwise
    'bitwise_not': np.bitwise_not,
    'bitwise_and': np.bitwise_and,
    'bitwise_or': np.bitwise_or,
    'bitwise_xor': np.bitwise_xor,
    'shift_left': np.left_shift,
    'shift_right_arithmetic': np.right_shift,
    'population_count': lax_reference.population_count,

    # Array manipulation
    'convert_element_type': lax_reference.convert_element_type,
    'bitcast_convert_type': lax_reference.bitcast_convert_type,
    'clamp': lax_reference.clamp,
    'concatenate': lax_reference.concatenate,
    'reshape': lax_reference.reshape,
    'transpose': np.transpose,
    'broadcast_in_dim': lambda operand, *, shape, broadcast_dimensions, sharding=None: (
        lambda op, shp, dims: (
            print(f"[BROADCAST_IN_DIM] operand.shape={np.array(op).shape}, target_shape={shp}, broadcast_dims={dims}") if False else None,
            lax_reference.broadcast_in_dim(op, shape=shp, broadcast_dimensions=dims)
        )[1]
    )(operand, shape, broadcast_dimensions),
    'squeeze': np.squeeze,
    'slice': lax_reference.slice,
    'dynamic_slice': lax_reference.dynamic_slice,
    'dynamic_update_slice': lax_reference.dynamic_update_slice,
    'pad': lax_reference.pad,
    'rev': lax_reference.rev,
    'select': np.where,

    # Reduction
    'reduce_sum': np.sum,
    'reduce_max': np.max,
    'reduce_min': np.min,

    # Linear algebra
    'dot': np.dot,
    'dot_general': lax_reference.dot_general,
}


def eval_jaxpr_numpy(
    jaxpr: jax_core.Jaxpr,
    consts: Sequence[Any],
    *args: Any,
    grid_env: dict[int, tuple[int, int]] | None = None,
) -> list[Any]:
  """Evaluate a Jaxpr using NumPy arrays.

  Args:
    jaxpr: The Jaxpr to evaluate
    consts: Constants used in the jaxpr
    *args: Input arguments (as NumPy arrays)
    grid_env: Dictionary mapping axis to (index, size) for program_id/num_programs

  Returns:
    List of output values
  """
  # Track recursion depth
  if not hasattr(eval_jaxpr_numpy, '_depth'):
    eval_jaxpr_numpy._depth = 0
  eval_jaxpr_numpy._depth += 1
  try:
    return _eval_jaxpr_numpy_impl(jaxpr, consts, *args, grid_env=grid_env)
  finally:
    eval_jaxpr_numpy._depth -= 1

def _eval_jaxpr_numpy_impl(
    jaxpr: jax_core.Jaxpr,
    consts: Sequence[Any],
    *args: Any,
    grid_env: dict[int, tuple[int, int]] | None = None,
) -> list[Any]:
  """Implementation of eval_jaxpr_numpy. as NumPy arrays
  """
  if grid_env is None:
    grid_env = {}

  def read(v: jax_core.Atom) -> Any:
    if isinstance(v, jax_core.Literal):
      return np.asarray(v.val)
    return env[v]

  def write(v: jax_core.Var, val: Any) -> None:
    # Ensure refs are writable (not read-only views)
    if hasattr(v.aval, 'inner_aval') and isinstance(val, np.ndarray):
      # This is a ref - ensure it's writable
      if not val.flags.writeable:
        val = val.copy()
    env[v] = val

  env: dict[jax_core.Var, Any] = {}

  # Initialize environment with consts and args
  for v, c in zip(jaxpr.constvars, consts):
    write(v, np.asarray(c))
  for v, a in zip(jaxpr.invars, args):
    write(v, np.asarray(a))

  # Evaluate each equation
  for eqn_idx, eqn in enumerate(jaxpr.eqns):
    in_vals = [read(v) for v in eqn.invars]
    prim = eqn.primitive

    # Debug: Log equation execution - with depth info
    DEBUG_EQN = False
    if DEBUG_EQN and eval_jaxpr_numpy._depth <= 3:  # Show a few levels
      indent = "  " * (eval_jaxpr_numpy._depth - 1)
      print(f"{indent}[EQN {eqn_idx} @depth{eval_jaxpr_numpy._depth}] {prim.name}")

    # Handle Pallas-specific primitives
    if prim.name == 'program_id':
      axis = eqn.params['axis']
      if axis in grid_env:
        result = np.int32(grid_env[axis][0])
      else:
        result = np.int32(0)

    elif prim.name == 'num_programs':
      axis = eqn.params['axis']
      if axis in grid_env:
        result = np.int32(grid_env[axis][1])
      else:
        result = np.int32(1)

    # Handle state primitives (refs)
    elif prim.name == 'get':
      # Reading from a ref
      ref_val = in_vals[0]
      indexer = eqn.params.get('indexer', None)
      if indexer is None:
        result = ref_val
      else:
        # Apply indexing
        result = ref_val[indexer.indices]

      # Debug logging
      DEBUG = True
      if DEBUG:
        result_arr = np.array(result)
        if indexer is not None:
          print(f"[GET] indexer={indexer.indices if hasattr(indexer, 'indices') else indexer} → shape={result_arr.shape}, values={result_arr.flatten()[:5]}")
        else:
          print(f"[GET] FULL → shape={result_arr.shape}, values={result_arr.flatten()[:5]}")


    elif prim.name == 'swap':
      # Writing to a ref
      ref_val = in_vals[0]
      new_val = in_vals[1]
      indexer = eqn.params.get('indexer', None)
      old_val = ref_val.copy() if indexer is None else ref_val[indexer.indices].copy()

      # Debug logging
      DEBUG = True
      if DEBUG:
        old_arr = np.array(old_val)
        new_arr = np.array(new_val)
        if indexer is not None:
          print(f"[SWAP] idx={indexer.indices if hasattr(indexer, 'indices') else indexer}, shape={old_arr.shape}, old={old_arr.flatten()[:5]}, new={new_arr.flatten()[:5]}")
        else:
          print(f"[SWAP] FULL, shape={old_arr.shape}, old={old_arr.flatten()[:5]}, new={new_arr.flatten()[:5]}")

      if indexer is None:
        # Full update
        ref_val[:] = new_val
      else:
        # Partial update
        ref_val[indexer.indices] = new_val

      result = old_val

    elif prim.name == 'addupdate':
      # Atomic add
      ref_val = in_vals[0]
      update_val = in_vals[1]
      indexer = eqn.params.get('indexer', None)

      if indexer is None:
        old_val = ref_val.copy()
        ref_val[:] = ref_val + update_val
      else:
        old_val = ref_val[indexer.indices].copy()
        ref_val[indexer.indices] = ref_val[indexer.indices] + update_val

      result = old_val

    elif prim.name == 'select_n':
      # Select from multiple options based on index
      # First arg is the index, rest are the options
      which = in_vals[0]
      cases = in_vals[1:]
      # Use np.choose for element-wise selection
      # np.choose requires which to be int32 and in range [0, len(cases))
      result = np.choose(which, cases)

    elif prim.name == 'iota':
      # Create array with incrementing values along a dimension
      dtype = eqn.params['dtype']
      shape = eqn.params['shape']
      dimension = eqn.params['dimension']
      # Create array of indices along the specified dimension
      result = np.arange(shape[dimension], dtype=dtype)
      # Broadcast to full shape
      new_shape = [1] * len(shape)
      new_shape[dimension] = shape[dimension]
      result = result.reshape(new_shape)
      result = np.broadcast_to(result, shape).copy()

    elif prim.name == 'gather':
      # Gather using advanced NumPy indexing
      operand = in_vals[0]
      start_indices = in_vals[1]
      dimension_numbers = eqn.params['dimension_numbers']
      slice_sizes = tuple(eqn.params['slice_sizes'])

      operand_batching_dims = getattr(dimension_numbers, 'operand_batching_dims', ())
      start_indices_batching_dims = getattr(dimension_numbers, 'start_indices_batching_dims', ())
      collapsed_slice_dims = tuple(dimension_numbers.collapsed_slice_dims)
      start_index_map = tuple(dimension_numbers.start_index_map)

      DEBUG_GATHER = False
      if DEBUG_GATHER and eval_jaxpr_numpy._depth >= 2:
        print(f"[GATHER] operand.shape={operand.shape}, indices.shape={start_indices.shape}")
        print(f"  operand={operand.flatten()[:10]}")
        print(f"  indices={start_indices.flatten()[:10]}")
        print(f"  batch_dims={operand_batching_dims}, collapsed={collapsed_slice_dims}")

      # Handle batched gather
      if operand_batching_dims and len(start_index_map) == 1 and len(collapsed_slice_dims) == 1:
        # Batched gather: for each batch element, gather from the specified dimension
        # Example: operand (8, 128), indices (8, 128, 1) with batch dim 1
        # Result should be (8, 128) where result[i,j] = operand[indices[i,j,0], j]

        batch_dim = operand_batching_dims[0]
        gather_dim = start_index_map[0]

        # Squeeze trailing dimension from indices
        indices = start_indices.squeeze(-1) if start_indices.shape[-1] == 1 else start_indices

        # Build advanced indexing arrays
        # For operand[indices, batch_idx] where we gather from gather_dim and keep batch_dim
        if gather_dim == 0 and batch_dim == 1:
          # operand[indices[i,j], j] for all i,j
          batch_idx = np.arange(operand.shape[batch_dim])[None, :]
          result = operand[indices.astype(np.intp), batch_idx]
        elif gather_dim == 1 and batch_dim == 0:
          # operand[i, indices[i,j]] for all i,j
          batch_idx = np.arange(operand.shape[batch_dim])[:, None]
          result = operand[batch_idx, indices.astype(np.intp)]
        else:
          raise NotImplementedError(f"Batched gather with gather_dim={gather_dim}, batch_dim={batch_dim}")
      elif len(start_index_map) == 1 and len(collapsed_slice_dims) == 1 and not operand_batching_dims:
        # Simple non-batched gather
        axis = start_index_map[0]
        indices = start_indices.squeeze(-1) if start_indices.shape[-1] == 1 else start_indices
        result = np.take_along_axis(operand, indices.astype(np.intp), axis=axis)
      else:
        raise NotImplementedError(f"Complex gather pattern not yet supported: dimension_numbers={dimension_numbers}")

      if DEBUG_GATHER and eval_jaxpr_numpy._depth >= 2:
        print(f"  result={result.flatten()[:10]}")

    # Handle standard JAX primitives
    elif prim.name in _NUMPY_IMPL:
      impl = _NUMPY_IMPL[prim.name]
      # Handle parameter name mapping and filtering
      params = dict(eqn.params)
      if prim.name == 'convert_element_type':
        if 'new_dtype' in params:
          params['dtype'] = params.pop('new_dtype')
        # Filter out parameters that lax_reference doesn't accept
        params = {k: v for k, v in params.items() if k in ['dtype']}
      elif prim.name == 'transpose':
        # NumPy uses 'axes' instead of 'permutation'
        if 'permutation' in params:
          params['axes'] = params.pop('permutation')
      elif prim.name == 'reshape':
        # Filter out JAX-specific parameters
        params = {k: v for k, v in params.items() if k in ['new_sizes', 'dimensions']}
      elif prim.name == 'concatenate':
        # concatenate takes (operands, dimension) as positional args
        dimension = params.get('dimension', 0)
        result = impl(in_vals, dimension)
        params = {}  # Don't pass params again

      if params or prim.name != 'concatenate':
        # Log operation for debugging - only key operations
        DEBUG_OPS = {'get', 'swap', 'select_n', 'gather', 'transpose', 'slice', 'concatenate'}  # Focus on these
        if prim.name in DEBUG_OPS and eval_jaxpr_numpy._depth == 2:
          print(f"[{eqn_idx:3d}] {prim.name:20s}", end=' ')
          for i, val in enumerate(in_vals):
            arr = np.array(val)
            if arr.size <= 5:
              print(f"in{i}={arr.flatten()} ", end='')
            else:
              print(f"in{i}=[{arr.flatten()[0]:.2f}...] ", end='')
          if params:
            print(f"params={params} ", end='')

        result = impl(*in_vals, **params)

        if prim.name in DEBUG_OPS and eval_jaxpr_numpy._depth == 2:
          arr = np.array(result) if hasattr(result, '__array__') or isinstance(result, np.ndarray) else result
          if isinstance(arr, np.ndarray) and arr.size <= 5:
            print(f"→ {arr.flatten()}")
          elif isinstance(arr, np.ndarray):
            print(f"→ [{arr.flatten()[0]:.2f}...]")
          else:
            print(f"→ {arr}")

    # Handle higher-order primitives
    elif prim.name == 'cond':
      # Conditional
      pred = in_vals[0]
      branches = eqn.params['branches']
      # For simplicity, just handle true/false branches
      if len(branches) == 2:
        branch_jaxpr = branches[1 if pred else 0].jaxpr
        branch_consts = branches[1 if pred else 0].consts
        result = eval_jaxpr_numpy(branch_jaxpr, branch_consts, *in_vals[1:], grid_env=grid_env)
        result = result[0] if len(result) == 1 else result
      else:
        raise NotImplementedError("Only binary cond supported")

    elif prim.name == 'while':
      # While loop
      cond_jaxpr = eqn.params['cond_jaxpr']
      body_jaxpr = eqn.params['body_jaxpr']
      carry = list(in_vals)

      while True:
        cond_result = eval_jaxpr_numpy(cond_jaxpr.jaxpr, cond_jaxpr.consts, *carry, grid_env=grid_env)
        if not cond_result[0]:
          break
        carry = eval_jaxpr_numpy(body_jaxpr.jaxpr, body_jaxpr.consts, *carry, grid_env=grid_env)

      result = carry

    elif prim.name == 'scan':
      # Scan loop
      scan_jaxpr = eqn.params['jaxpr']
      num_consts = eqn.params['num_consts']
      num_carry = eqn.params['num_carry']
      length = eqn.params['length']

      DEBUG_SCAN_DETAIL = False
      if DEBUG_SCAN_DETAIL and eval_jaxpr_numpy._depth == 1:
        print(f"[SCAN] num_consts={num_consts}, num_carry={num_carry}, length={length}")
        print(f"[SCAN] in_vals: {len(in_vals)} values")
        # Check which invars correspond to input vs output refs
        print(f"[SCAN] jaxpr invars: {len(scan_jaxpr.jaxpr.invars)}")
        for i, (val, invar) in enumerate(zip(in_vals[:num_consts], scan_jaxpr.jaxpr.invars[:num_consts])):
          arr = np.array(val)
          if isinstance(arr, np.ndarray):
            is_writable = arr.flags.writeable if hasattr(arr, 'flags') else 'N/A'
            is_ref = hasattr(invar.aval, 'inner_aval')
            print(f"  const[{i}]: shape={arr.shape}, dtype={arr.dtype}, writable={is_writable}, is_ref={is_ref}, values={arr.flatten()[:5]}")

      consts = in_vals[:num_consts]
      init_carry = in_vals[num_consts:num_consts + num_carry]
      xs = in_vals[num_consts + num_carry:]

      # Note: Disabling read-only protection for now
      # The real bug is in the computation producing wrong results

      carry = init_carry
      ys = []

      for i in range(length):
        x_i = [x[i] for x in xs] if xs else []
        scan_in = [*consts, *carry, *x_i]
        scan_out = eval_jaxpr_numpy(scan_jaxpr.jaxpr, scan_jaxpr.consts, *scan_in, grid_env=grid_env)
        carry = scan_out[:num_carry]
        y_i = scan_out[num_carry:]
        if y_i:
          ys.append(y_i)

      if ys:
        ys = [np.stack([y[i] for y in ys]) for i in range(len(ys[0]))]
        result = [*carry, *ys]
      else:
        result = carry

      # Debug scan result
      DEBUG_SCAN = False
      if DEBUG_SCAN:
        for i, r in enumerate(result if isinstance(result, (list, tuple)) else [result]):
          arr = np.array(r)
          if arr.size > 0:
            print(f"[SCAN RESULT {i}] shape={arr.shape}, dtype={arr.dtype}, values={arr.flatten()[:5]}")

    elif prim.name == 'jit':
      # JIT primitive - in interpret mode, just evaluate the jaxpr directly
      jit_jaxpr = eqn.params['jaxpr']
      result = eval_jaxpr_numpy(jit_jaxpr.jaxpr, jit_jaxpr.consts, *in_vals, grid_env=grid_env)
      # Don't unwrap - jit primitive uses multiple_results to determine how to handle result

    else:
      raise NotImplementedError(f"Primitive {prim.name} not implemented in NumPy interpreter")

    # Write results
    if prim.multiple_results:
      if not isinstance(result, (list, tuple)):
        raise RuntimeError(f"Primitive {prim.name} has multiple_results=True but returned {type(result)}: {result}. Expected list/tuple. outvars: {len(eqn.outvars)}")
      for v, r in zip(eqn.outvars, result):
        write(v, r)
    else:
      # Check dtype matching but only auto-fix for non-refs
      if isinstance(result, np.ndarray) and hasattr(eqn.outvars[0].aval, 'dtype'):
        expected_dtype = eqn.outvars[0].aval.dtype
        if result.dtype != expected_dtype:
          # Only auto-convert if this is NOT a ref (refs must maintain identity for mutations)
          is_ref = hasattr(eqn.outvars[0].aval, 'inner_aval')
          if not is_ref:
            result = result.astype(expected_dtype)
      write(eqn.outvars[0], result)

  # Return outputs
  return [read(v) for v in jaxpr.outvars]


def pallas_call_numpy_interpret(
    *args,
    jaxpr: jax_core.Jaxpr,
    grid_mapping,
    out_avals,
    **kwargs
):
  """Interpret a Pallas call using NumPy via io_callback.

  This function intercepts Pallas calls and executes them using pure NumPy,
  leveraging the stateful nature of NumPy arrays to implement memory references.
  """
  from jax._src.pallas import hlo_interpreter

  print(f"[NumPy Interpreter] Intercepted pallas_call with grid={grid_mapping.grid}")

  # First, let's see what primitives are in the jaxpr
  print(f"[NumPy Interpreter] Analyzing jaxpr with {len(jaxpr.eqns)} equations...")

  # Collect all primitives used (recursively scan nested jaxprs)
  def collect_primitives(jaxpr, prims_dict):
    """Recursively collect all primitives from jaxpr including nested ones."""
    for eqn in jaxpr.eqns:
      prim_name = eqn.primitive.name
      prims_dict[prim_name] = prims_dict.get(prim_name, 0) + 1

      # Check for nested jaxprs in parameters
      if prim_name in ['scan', 'while', 'cond', 'jit']:
        if 'jaxpr' in eqn.params:
          nested = eqn.params['jaxpr']
          if hasattr(nested, 'jaxpr'):
            collect_primitives(nested.jaxpr, prims_dict)
        if 'cond_jaxpr' in eqn.params:
          collect_primitives(eqn.params['cond_jaxpr'].jaxpr, prims_dict)
        if 'body_jaxpr' in eqn.params:
          collect_primitives(eqn.params['body_jaxpr'].jaxpr, prims_dict)
        if 'branches' in eqn.params:
          for branch in eqn.params['branches']:
            if hasattr(branch, 'jaxpr'):
              collect_primitives(branch.jaxpr, prims_dict)

  primitives_used = {}
  collect_primitives(jaxpr, primitives_used)

  # Primitives we can handle
  supported_prims = set(_NUMPY_IMPL.keys()) | {
      'program_id', 'num_programs', 'get', 'swap', 'addupdate',
      'scan', 'while', 'cond', 'jit', 'select_n', 'iota', 'gather'
  }

  print(f"[NumPy Interpreter] Primitives used:")
  for prim_name, count in sorted(primitives_used.items()):
    implemented = prim_name in supported_prims
    status = "✓" if implemented else "✗"
    print(f"  {status} {prim_name}: {count}")

  # Check if all primitives are supported
  unsupported = [p for p in primitives_used.keys() if p not in supported_prims]

  if unsupported:
    print(f"[NumPy Interpreter] Unsupported primitives: {unsupported}")
    print(f"[NumPy Interpreter] Falling back to HLO interpreter...")
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'interpret'}
    return hlo_interpreter.pallas_call_hlo_interpret(
        *args,
        jaxpr=jaxpr,
        grid_mapping=grid_mapping,
        out_avals=out_avals,
        **filtered_kwargs
    )

  # All primitives supported - use NumPy interpreter!
  print(f"[NumPy Interpreter] Using pure NumPy interpretation!")

  # Debug: print jaxpr structure
  print(f"[NumPy Interpreter] Jaxpr invars: {len(jaxpr.invars)}")
  print(f"[NumPy Interpreter] Jaxpr outvars: {len(jaxpr.outvars)}")
  print(f"[NumPy Interpreter] Input args: {len(args)}")
  print(f"[NumPy Interpreter] Expected output avals: {len(out_avals)}")

  # Create scratch refs - these are the extra invars beyond the input args
  # The jaxpr expects: [*input_refs, *output_refs, *scratch_refs]
  num_input_refs = len(args)
  num_output_refs = len(out_avals)
  num_scratch_refs = len(jaxpr.invars) - num_input_refs - num_output_refs

  print(f"[NumPy Interpreter] Creating {num_scratch_refs} scratch refs...")

  # Use io_callback to execute in NumPy
  def numpy_kernel(*jax_args):
    # Convert JAX arrays to NumPy with correct dtypes from jaxpr
    input_refs = []
    for i, arg in enumerate(jax_args):
      if i < num_input_refs:
        invar = jaxpr.invars[i]
        if hasattr(invar.aval, 'inner_aval'):
          # This is a ref - get dtype from inner_aval
          expected_dtype = invar.aval.inner_aval.dtype
          input_refs.append(np.asarray(arg, dtype=expected_dtype))
        else:
          input_refs.append(np.asarray(arg))
      else:
        input_refs.append(np.asarray(arg) if hasattr(arg, '__array__') else arg)

    # Create output refs as mutable NumPy arrays
    output_refs = [np.zeros(aval.shape, dtype=aval.dtype) for aval in out_avals]

    # Create scratch refs from the jaxpr invars
    scratch_refs = []
    for i in range(num_scratch_refs):
      invar_idx = num_input_refs + num_output_refs + i
      scratch_aval = jaxpr.invars[invar_idx].aval
      # Extract inner aval from Ref type
      if hasattr(scratch_aval, 'inner_aval'):
        inner_aval = scratch_aval.inner_aval
        scratch_shape = inner_aval.shape
        scratch_dtype = inner_aval.dtype
      else:
        scratch_shape = scratch_aval.shape
        scratch_dtype = scratch_aval.dtype
      scratch_refs.append(np.zeros(scratch_shape, dtype=scratch_dtype))

    # Combine all refs in the order the jaxpr expects
    all_refs = input_refs + output_refs + scratch_refs

    print(f"[NumPy Interpreter] Executing with {len(all_refs)} refs (NumPy arrays)")
    print(f"[DEBUG] Ref dtypes: {[r.dtype if isinstance(r, np.ndarray) else type(r) for r in all_refs]}")

    # Debug: Check initial values
    DEBUG = False
    if DEBUG:
      print(f"[NumPy Interpreter] Initial refs:")
      for i, ref in enumerate(all_refs[:2]):  # Just check first 2 refs
        arr = np.array(ref)
        print(f"  ref[{i}]: shape={arr.shape}, dtype={arr.dtype}, values={arr.flatten()[:10]}")

    # Get grid
    grid = grid_mapping.grid

    # Iterate over grid
    if not grid:
      # No grid, run once
      grid_env = {}
      eval_jaxpr_numpy(jaxpr, [], *all_refs, grid_env=grid_env)
    else:
      # Multi-dimensional grid
      import itertools
      for indices in itertools.product(*[range(g) for g in grid]):
        grid_env = {i: (idx, size) for i, (idx, size) in enumerate(zip(indices, grid))}
        eval_jaxpr_numpy(jaxpr, [], *all_refs, grid_env=grid_env)

    # Debug: Check what the output refs contain
    DEBUG = False
    if DEBUG:
      print(f"[NumPy Interpreter] Final output refs:")
      for i, ref in enumerate(output_refs):
        arr = np.array(ref)
        print(f"  output_ref[{i}]: shape={arr.shape}, dtype={arr.dtype}, values={arr.flatten()[:10]}")

    # Return the output refs (which have been modified in place)
    return output_refs

  # Build result shape
  result_shapes = [jax.ShapeDtypeStruct(aval.shape, aval.dtype) for aval in out_avals]

  # Use io_callback to execute
  return io_callback(numpy_kernel, result_shapes, *args)


# Monkey-patch to intercept Pallas calls
_original_pallas_call_bind = None

def install_numpy_interpreter():
  """Install the NumPy interpreter for Pallas calls."""
  global _original_pallas_call_bind

  if _original_pallas_call_bind is not None:
    return  # Already installed

  _original_pallas_call_bind = pallas_call.pallas_call_p.bind

  def numpy_pallas_call_bind(*args, **params):
    # Check if we should use numpy interpreter
    interpret = params.get('interpret', False)

    if interpret:
      # Use NumPy interpreter
      return pallas_call_numpy_interpret(*args, **params)
    else:
      # Use original implementation
      return _original_pallas_call_bind(*args, **params)

  pallas_call.pallas_call_p.bind = numpy_pallas_call_bind
  print("NumPy interpreter installed for Pallas calls")


def uninstall_numpy_interpreter():
  """Uninstall the NumPy interpreter for Pallas calls."""
  global _original_pallas_call_bind

  if _original_pallas_call_bind is None:
    return  # Not installed

  pallas_call.pallas_call_p.bind = _original_pallas_call_bind
  _original_pallas_call_bind = None
  print("NumPy interpreter uninstalled")
