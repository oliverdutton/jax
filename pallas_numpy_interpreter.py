"""Pallas NumPy interpreter using io_callback."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax._src import lax_reference
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
    'broadcast_in_dim': lax_reference.broadcast_in_dim,
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
    List of output values as NumPy arrays
  """
  if grid_env is None:
    grid_env = {}

  def read(v: jax_core.Atom) -> Any:
    if isinstance(v, jax_core.Literal):
      return np.asarray(v.val)
    return env[v]

  def write(v: jax_core.Var, val: Any) -> None:
    env[v] = val

  env: dict[jax_core.Var, Any] = {}

  # Initialize environment with consts and args
  for v, c in zip(jaxpr.constvars, consts):
    write(v, np.asarray(c))
  for v, a in zip(jaxpr.invars, args):
    write(v, np.asarray(a))

  # Evaluate each equation
  for eqn in jaxpr.eqns:
    in_vals = [read(v) for v in eqn.invars]
    prim = eqn.primitive

    # Handle Pallas-specific primitives
    if prim is pallas_primitives.program_id_p:
      axis = eqn.params['axis']
      if axis in grid_env:
        result = np.int32(grid_env[axis][0])
      else:
        result = np.int32(0)

    elif prim is pallas_primitives.num_programs_p:
      axis = eqn.params['axis']
      if axis in grid_env:
        result = np.int32(grid_env[axis][1])
      else:
        result = np.int32(1)

    # Handle state primitives (refs)
    elif isinstance(prim, state.GetPrim):
      # Reading from a ref
      ref_val = in_vals[0]
      indexer = eqn.params.get('indexer', None)
      if indexer is None:
        result = ref_val
      else:
        # Apply indexing
        result = ref_val[indexer.indices]

    elif isinstance(prim, state.SwapPrim):
      # Writing to a ref
      ref_val = in_vals[0]
      new_val = in_vals[1]
      indexer = eqn.params.get('indexer', None)
      old_val = ref_val.copy() if indexer is None else ref_val[indexer.indices].copy()

      if indexer is None:
        # Full update
        ref_val[:] = new_val
      else:
        # Partial update
        ref_val[indexer.indices] = new_val

      result = old_val

    elif isinstance(prim, state.AddUpdatePrim):
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

    # Handle standard JAX primitives
    elif prim.name in _NUMPY_IMPL:
      impl = _NUMPY_IMPL[prim.name]
      result = impl(*in_vals, **eqn.params)

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

    else:
      raise NotImplementedError(f"Primitive {prim.name} not implemented in NumPy interpreter")

    # Write results
    if prim.multiple_results:
      for v, r in zip(eqn.outvars, result):
        write(v, r)
    else:
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

  # For now, fall back to the standard HLO interpreter
  # This already does what we want - interprets the jaxpr
  # TODO: Progressively replace operations with pure NumPy via io_callback
  print(f"[NumPy Interpreter] Intercepted pallas_call with grid={grid_mapping.grid}")

  # Filter out the 'interpret' kwarg since HLO interpreter doesn't expect it
  filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'interpret'}

  return hlo_interpreter.pallas_call_hlo_interpret(
      *args,
      jaxpr=jaxpr,
      grid_mapping=grid_mapping,
      out_avals=out_avals,
      **filtered_kwargs
  )


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
