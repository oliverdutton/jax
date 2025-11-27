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
    DEBUG_EQN = False  # Disable for now
    if DEBUG_EQN and eval_jaxpr_numpy._depth == 2 and eqn_idx >= 1200 and eqn_idx <= 1340:
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
      DEBUG = False
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
      DEBUG = False
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

      DEBUG_SELECT_N = True
      if DEBUG_SELECT_N and eval_jaxpr_numpy._depth == 3:
        print(f"[SELECT_N depth3 EQN {eqn_idx}]")
        if isinstance(which, np.ndarray):
          which_unique = np.unique(which.flatten())
          print(f"  which: shape={which.shape}, unique={len(which_unique)}, values={which_unique[:5]}")
        else:
          print(f"  which: scalar={which}")
        for i, case in enumerate(cases):
          if isinstance(case, np.ndarray) and case.size >= 128:
            case_int32 = case.view(np.int32) if case.dtype == np.float32 else case
            case_unique = np.unique(case_int32.flatten())
            print(f"  case[{i}]: shape={case.shape}, unique={len(case_unique)}, values={case_unique[:5]}")

      # Use np.choose for element-wise selection
      # np.choose requires which to be int32 and in range [0, len(cases))
      result = np.choose(which, cases)

      if DEBUG_SELECT_N and eval_jaxpr_numpy._depth == 3 and isinstance(result, np.ndarray) and result.size >= 128:
        result_int32 = result.view(np.int32) if result.dtype == np.float32 else result
        result_unique = np.unique(result_int32.flatten())
        print(f"  result: shape={result.shape}, unique={len(result_unique)}, values={result_unique[:5]}")

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

      DEBUG_GATHER = True
      if DEBUG_GATHER and eval_jaxpr_numpy._depth == 3:
        print(f"[GATHER] operand.shape={operand.shape}, indices.shape={start_indices.shape}")
        print(f"  operand={operand.flatten()[:10]}")
        print(f"  indices={start_indices.flatten()[:10]}")
        print(f"  slice_sizes={slice_sizes}")
        print(f"  operand_batching_dims={operand_batching_dims}")
        print(f"  start_indices_batching_dims={start_indices_batching_dims}")
        print(f"  collapsed_slice_dims={collapsed_slice_dims}")
        print(f"  start_index_map={start_index_map}")

      # DEBUG: Try using fallback (non-batched) gather to see if it works correctly
      USE_FALLBACK_GATHER = True
      if USE_FALLBACK_GATHER and eval_jaxpr_numpy._depth == 3:
        # Temporarily skip batched path to use fallback
        saved_operand_batching_dims = operand_batching_dims
        operand_batching_dims = ()  # Disable batched path

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

          DEBUG_GATHER_DETAIL = True
          if DEBUG_GATHER_DETAIL and eval_jaxpr_numpy._depth == 3:
            print(f"[BATCHED GATHER] gather_dim=0, batch_dim=1")
            print(f"  operand.shape={operand.shape}")
            print(f"  indices.shape={indices.shape}, batch_idx.shape={batch_idx.shape}")
            print(f"  indices[0,:5]={indices[0,:5]}")
            print(f"  batch_idx[0,:5]={batch_idx[0,:5]}")
            print(f"  indices[:,0] = {indices[:,0]}")  # All indices for column 0
            operand_int32 = operand.view(np.int32) if operand.dtype == np.float32 else operand
            print(f"  operand[0,:5]={operand_int32[0,:5]}")
            print(f"  operand[1,:5]={operand_int32[1,:5]}")
            print(f"  operand[2,:5]={operand_int32[2,:5]}")
            print(f"  operand[3,:5]={operand_int32[3,:5]}")
            # Show which rows have which values at column 0
            print(f"  operand[:,0] = {operand_int32[:,0]}")  # All values at column 0

          result = operand[indices.astype(np.intp), batch_idx]

          if DEBUG_GATHER_DETAIL and eval_jaxpr_numpy._depth == 3:
            print(f"  result.shape={result.shape}")
            print(f"  result[0,:5]={result[0,:5]}")
            result_int32 = result.view(np.int32) if result.dtype == np.float32 else result
            operand_int32 = operand.view(np.int32) if operand.dtype == np.float32 else operand
            print(f"  operand unique values: {np.unique(operand_int32.flatten())[:6]}")
            print(f"  result unique values: {np.unique(result_int32.flatten())[:6]}")
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
        if DEBUG_GATHER and eval_jaxpr_numpy._depth == 3:
          print(f"[FALLBACK GATHER] axis={axis}, indices.shape={indices.shape}")
          print(f"  indices[:,0] = {indices[:,0]}")
        result = np.take_along_axis(operand, indices.astype(np.intp), axis=axis)
        if DEBUG_GATHER and eval_jaxpr_numpy._depth == 3 and result.size >= 128:
          result_int32 = result.view(np.int32) if result.dtype == np.float32 else result
          print(f"  result unique values (fallback): {np.unique(result_int32.flatten())[:6]}")
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

        # Log concatenate operation
        DEBUG_OPS = {'transpose', 'slice', 'concatenate', 'gather'}
        if prim.name in DEBUG_OPS and eval_jaxpr_numpy._depth == 2 and eqn_idx >= 1320 and eqn_idx <= 1345:
          print(f"\n[{eqn_idx:3d}] concatenate  dimension={dimension}")
          print(f"  in_vals is list with {len(in_vals)} arrays:")
          for i, val in enumerate(in_vals):
            arr = np.array(val)
            if arr.size > 0:
              unique = len(np.unique(arr.flatten()[:20]))
              print(f"    [{i}]: shape={arr.shape}, unique={unique}, sample={arr.flatten()[:5]}")

        result = impl(in_vals, dimension)

        if prim.name in DEBUG_OPS and eval_jaxpr_numpy._depth == 2 and eqn_idx >= 1320 and eqn_idx <= 1345:
          arr = np.array(result)
          unique = len(np.unique(arr.flatten()[:20]))
          print(f"  → shape={arr.shape}, unique={unique}, sample={arr.flatten()[:5]}")

        params = {}  # Don't pass params again

      if params or prim.name != 'concatenate':
        # Log operation for debugging - only key operations
        DEBUG_OPS = {'transpose', 'slice', 'concatenate', 'gather'}  # Focus on these
        if prim.name in DEBUG_OPS and eval_jaxpr_numpy._depth == 2 and eqn_idx >= 1320 and eqn_idx <= 1345:
          print(f"\n[{eqn_idx:3d}] {prim.name:20s}", end=' ')
          for i, val in enumerate(in_vals):
            arr = np.array(val)
            if arr.size <= 10:
              print(f"\n  in{i}={arr.flatten()} ", end='')
            else:
              unique = len(np.unique(arr.flatten()[:20]))
              print(f"\n  in{i}: shape={arr.shape}, unique={unique}, sample={arr.flatten()[:5]} ", end='')
          if params:
            print(f"\n  params={params} ", end='')

        result = impl(*in_vals, **params)

        if prim.name in DEBUG_OPS and eval_jaxpr_numpy._depth == 2 and eqn_idx >= 1320 and eqn_idx <= 1345:
          arr = np.array(result) if hasattr(result, '__array__') or isinstance(result, np.ndarray) else result
          if isinstance(arr, np.ndarray):
            unique = len(np.unique(arr.flatten()[:20]))
            print(f"\n  → shape={arr.shape}, unique={unique}, sample={arr.flatten()[:5]}")
          else:
            print(f"\n  → {arr}")

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

      DEBUG_JIT = True
      jit_indices = [38, 67, 1186, 1196, 1206, 1216, 1226, 1236, 1246, 1256, 1266, 1276, 1286, 1296, 1306, 1316, 1326, 1336]
      if DEBUG_JIT and eval_jaxpr_numpy._depth == 2 and eqn_idx in jit_indices:
        print(f"\n[JIT {eqn_idx}] Inputs:")
        for i, inv in enumerate(in_vals[:5]):
          if isinstance(inv, np.ndarray):
            inv_int32 = inv.view(np.int32) if inv.dtype == np.float32 else inv
            print(f"  in[{i}]: shape={inv.shape}, unique={len(np.unique(inv_int32.flatten()[:50]))}, sample={inv_int32.flatten()[:10]}")
        # Show what operations are in this JIT
        if eqn_idx in [38, 67]:
          print(f"  JIT contains {len(jit_jaxpr.jaxpr.eqns)} equations:")
          prim_counts = {}
          for jeqn in jit_jaxpr.jaxpr.eqns:
            pname = jeqn.primitive.name
            prim_counts[pname] = prim_counts.get(pname, 0) + 1
          for pname, count in sorted(prim_counts.items()):
            print(f"    {pname}: {count}")

      result = eval_jaxpr_numpy(jit_jaxpr.jaxpr, jit_jaxpr.consts, *in_vals, grid_env=grid_env)

      if DEBUG_JIT and eval_jaxpr_numpy._depth == 2 and eqn_idx in jit_indices:
        print(f"[JIT {eqn_idx}] Result:")
        if isinstance(result, (list, tuple)):
          for i, r in enumerate(result[:5]):
            if isinstance(r, np.ndarray):
              r_int32 = r.view(np.int32) if r.dtype == np.float32 else r
              print(f"  out[{i}]: shape={r.shape}, unique={len(np.unique(r_int32.flatten()[:50]))}, sample={r_int32.flatten()[:10]}")
        elif isinstance(result, np.ndarray):
          r_int32 = result.view(np.int32) if result.dtype == np.float32 else result
          print(f"  result: shape={result.shape}, unique={len(np.unique(r_int32.flatten()[:50]))}, sample={r_int32.flatten()[:10]}")

      # Don't unwrap - jit primitive uses multiple_results to determine how to handle result

    else:
      raise NotImplementedError(f"Primitive {prim.name} not implemented in NumPy interpreter")

    # Debug: Check for suspicious results (constant arrays from varied inputs)
    DEBUG_SUSPICIOUS = True
    if DEBUG_SUSPICIOUS and isinstance(result, np.ndarray) and result.size > 10:
        # Check if result is suspiciously constant
        if result.size >= 128:  # Only check large arrays
            unique_vals = len(np.unique(result.flatten()[:100]))
            if unique_vals == 1:
                # Check if inputs were varied
                input_varied = False
                for inv in in_vals:
                    if isinstance(inv, np.ndarray) and inv.size > 10:
                        inv_unique = len(np.unique(inv.flatten()[:100]))
                        if inv_unique > 1:
                            input_varied = True
                            break

                if input_varied:
                    print(f"\n⚠️  SUSPICIOUS: {prim.name} @ depth {eval_jaxpr_numpy._depth}")
                    print(f"   Varied input → Constant output (all {result.flatten()[0]})")
                    print(f"   Result shape: {result.shape}")
                    for i, inv in enumerate(in_vals[:3]):  # Show first 3 inputs
                        if isinstance(inv, np.ndarray) and inv.size > 0:
                            inv_arr = np.array(inv)
                            print(f"   in[{i}]: shape={inv_arr.shape}, unique={len(np.unique(inv_arr.flatten()[:20]))}, sample={inv_arr.flatten()[:5]}")

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

    # Debug: Track data loss - when arrays transition from 3+ unique values to 2
    DEBUG_TRACK_LOSS = True
    # Log operations at depth 3 (inside JIT 38) to find exact operation that loses 2.0
    if DEBUG_TRACK_LOSS and eval_jaxpr_numpy._depth == 3:
      results_to_check = []
      if prim.multiple_results and isinstance(result, (list, tuple)):
        results_to_check = result
      else:
        results_to_check = [result]
      for res_idx, res in enumerate(results_to_check):
        if isinstance(res, np.ndarray) and res.size >= 128:
          res_int32 = res.view(np.int32) if res.dtype == np.float32 else res
          unique_vals = np.unique(res_int32.flatten())
          print(f"  [depth3 EQN {eqn_idx}] {prim.name:15s} → shape={res.shape}, unique={len(unique_vals)}, values={unique_vals[:6]}")
          # Show inputs too
          for i, inv in enumerate(in_vals[:3]):
            if isinstance(inv, np.ndarray) and inv.size > 0:
              inv_int32 = inv.view(np.int32) if inv.dtype == np.float32 else inv
              inv_unique = np.unique(inv_int32.flatten())
              print(f"       in[{i}]: shape={inv.shape}, unique={len(inv_unique)}, values={inv_unique[:6]}")

    # Also log first 50 equations to see initial data state
    if DEBUG_TRACK_LOSS and eval_jaxpr_numpy._depth == 2 and eqn_idx < 50:
      results_to_check = []
      if prim.multiple_results and isinstance(result, (list, tuple)):
        results_to_check = result
      else:
        results_to_check = [result]
      for res_idx, res in enumerate(results_to_check):
        if isinstance(res, np.ndarray) and res.size >= 128:
          res_int32 = res.view(np.int32) if res.dtype == np.float32 else res
          unique_count = len(np.unique(res_int32.flatten()))  # Check ALL values, not just first 50
          if res.shape == (8, 128):
            unique_vals = np.unique(res_int32.flatten())
            print(f"[EQN {eqn_idx:3d}] {prim.name:20s} → shape={res.shape}, unique={unique_count}, values={unique_vals[:5]}")

    if DEBUG_TRACK_LOSS and eval_jaxpr_numpy._depth == 2 and eqn_idx < 1186:
      # Check all results that are large arrays
      results_to_check = []
      if prim.multiple_results and isinstance(result, (list, tuple)):
        results_to_check = result
      else:
        results_to_check = [result]

      for res_idx, res in enumerate(results_to_check):
        if isinstance(res, np.ndarray) and res.size >= 128:
          # Count unique values (use view as int32 to match debug output)
          res_int32 = res.view(np.int32) if res.dtype == np.float32 else res
          unique_count = len(np.unique(res_int32.flatten()[:50]))

          # Flag if we see only 2 unique values in what should be data array
          if unique_count == 2 and res.shape == (8, 128):
            sample_vals = res_int32.flatten()[:10]
            # Check if it's the specific corruption pattern (1.0 = 1065353216 and NaN = 2147483647)
            if 1065353216 in sample_vals and 2147483647 in sample_vals:
              print(f"\n🔴 DATA LOSS at EQN {eqn_idx}: {prim.name}")
              print(f"   Result has only 2 unique values (should have 3+ data values)")
              print(f"   shape={res.shape}, unique={unique_count}, sample={sample_vals}")
              print(f"   Inputs:")
              for i, inv in enumerate(in_vals[:5]):
                if isinstance(inv, np.ndarray) and inv.size >= 128:
                  inv_int32 = inv.view(np.int32) if inv.dtype == np.float32 else inv
                  inv_unique = len(np.unique(inv_int32.flatten()[:50]))
                  inv_sample = inv_int32.flatten()[:10]
                  print(f"     in[{i}]: shape={inv.shape}, unique={inv_unique}, sample={inv_sample}")

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
