
"""
Optimized Bitonic Sort, Top-K and Gather Implementation for TPU using JAX/Pallas.

This module provides efficient sorting and top-k operations optimized for TPU hardware,
utilizing bitonic sorting algorithms with tile-based processing.
"""

import functools
from functools import lru_cache
import math
import gzip
import json
import os
from glob import glob
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental import checkify
from jax.experimental.pallas import tpu as pltpu
import pandas as pd
from jax._src.pallas.mosaic.interpret import interpret_pallas_call
import time


# ============================================================================
# Hardware Constants
# ============================================================================

NUM_SUBLANES = 8
NUM_LANES = 128
LOG_LANES = 7  # log2(NUM_LANES)


# ============================================================================
# Performance Statistics
# ============================================================================

def calculate_performance_stats(num_stages: int, num_tokens: int, time_us: float):
  """Calculate and print performance statistics for sorting operations."""
  num_tiles = ((2**num_stages) // NUM_LANES)* (num_tokens // NUM_SUBLANES)
  stage_sequence = (1, 2, 3, 4, 5, 6)
  permutes_per_tile = (
      sum(stage_sequence[:num_stages]) +
      7 * max(0, num_stages - len(stage_sequence))
  )

  total_permutes = permutes_per_tile * num_tiles
  total_cycles = (time_us * 1e-6) * (1.75e9)
  cycles_per_permute = total_cycles / total_permutes

  print(f"Permutes: {total_permutes}, Cycles: {total_cycles}, "
        f"Cycles/Permute: {cycles_per_permute}")


# ============================================================================
# Utility Functions
# ============================================================================

@lru_cache
def _log2(value: int) -> int:
  """Calculate log base 2 of an integer."""
  log_result = 0
  n = value
  while n > 1:
    n = n // 2
    log_result += 1
  return log_result

# colab version of JAX lacks pl.loop, so reimplement
def loop(
    lower: jax.typing.ArrayLike,
    upper: jax.typing.ArrayLike,
    *,
    step: jax.typing.ArrayLike = 1,
    unroll: int | bool | None = None,
) -> Callable[[Callable[[jax.Array], None]], None]:
  """Returns a decorator that calls the decorated function in a loop."""
  zero: jax.typing.ArrayLike
  if not all(map(lambda v: type(v) == int, (lower, upper, step))):
    idx_type = jnp.result_type(lower, upper, step)
    lower = jax.lax.convert_element_type(lower, idx_type)
    upper = jax.lax.convert_element_type(upper, idx_type)
    step = jax.lax.convert_element_type(step, idx_type)
    zero = jnp.array(0, dtype=idx_type)
  else:
    zero = 0

  def decorator(body_fn):
    jax.lax.fori_loop(
        zero,
        pl.cdiv(upper - lower, step),
        lambda idx, _: body_fn(lower + idx * step),
        init_val=None,
        unroll=unroll,
    )
  return decorator


# JAX lowering for Pallas doesnt support integer unroll
def unrolled_fori_loop(length: int, body_fn, init_val, unroll: int):
  """Execute a for loop with manual unrolling for better performance."""
  unroll = min(length, unroll)

  def unrolled_body(i, carry):
    i *= unroll
    for j in range(unroll):
      carry = body_fn(i + j, carry)
    return carry

  carry = jax.lax.fori_loop(0, length // unroll, unrolled_body, init_val)
  for j in range(length % unroll):
    carry = body_fn((length // unroll) * unroll + j, carry)
  return carry


gather_2d = jax.vmap(lambda x, index: x[index])


# ============================================================================
# Value Packing/Unpacking for Index Preservation
# ============================================================================

def pack_value_with_index(val, index):
  """
  Pack bfloat16 value and int32 index into a single float32.
  This allows sorting while preserving original indices.
  """
  assert index.dtype == jnp.int32
  # BF16 values in F32 have empty lower 16 bits in mantissa where we pack the index
  return lax.bitcast_convert_type(
      lax.bitcast_convert_type(val.astype(jnp.float32), jnp.int32) | index,
      jnp.float32,
  )


def unpack_value_and_index(packed):
  """Extract the original value and index from packed representation."""
  val = lax.bitcast_convert_type(
      lax.bitcast_convert_type(packed, jnp.int32) & ~0xFFFF, jnp.float32
  ).astype(jnp.bfloat16)
  index = lax.bitcast_convert_type(packed, jnp.int32) & 0xFFFF
  return val, index


# ============================================================================
# Tile Management
# ============================================================================

def split_array_to_tiles(array_ref, use_dslice=True):
  """Split a 2D array into a flat list of tiles."""
  num_rows, num_cols = array_ref.shape
  tile_rows = num_rows // NUM_SUBLANES
  tile_cols = num_cols // NUM_LANES

  tiles = []
  for row in range(tile_rows):
    for col in range(tile_cols):
      if use_dslice:
        tile = array_ref[
            pl.dslice(row * NUM_SUBLANES, NUM_SUBLANES),
            pl.dslice(col * NUM_LANES, NUM_LANES)
        ]
      else:
        tile = array_ref[
            row * NUM_SUBLANES: (row + 1) * NUM_SUBLANES,
            col * NUM_LANES: (col + 1) * NUM_LANES,
        ]
      tiles.append(tile)
  return tiles


def join_tiles_to_array(target_shape, tiles):
  """Reconstruct a 2D array from a flat list of tiles."""
  num_rows, num_cols = target_shape
  tile_rows, tile_cols = tiles[0].shape
  grid_rows = num_rows // tile_rows
  grid_cols = num_cols // tile_cols

  rows = []
  for i in range(pl.cdiv(len(tiles), grid_cols)):
    row_tiles = tiles[i * grid_cols: (i + 1) * grid_cols]
    rows.append(jnp.concatenate(row_tiles, axis=-1))

  return jnp.concatenate(rows, axis=-2)

def create_bit_indicator(bit_position: int, index=None):
  """Create a boolean mask indicating which elements have a specific bit set."""
  if index is None:
    index = lax.broadcasted_iota(jnp.int32, (NUM_SUBLANES, NUM_LANES), 1)
  return (index & (1 << bit_position)) > 0


def compute_crosstile_substage(
    array_ref,
    substage: int,
    stage: int,
    sort_order: int,
    aux_refs=(),
    unroll: int = 16,
    concat_before_writeout: bool = False,
    read_then_split: bool = False,
    dim1_offset: int = 0,
    dim1_length: int = None
):
  """
  Perform a substage of sort involving comparisons between tiles

  Args:
      array_ref: Reference to array being sorted
      aux_ref: array to be sorted according to value in array_ref (e.g. array indices)
      substage: Current substage within the stage
      stage: Current sorting stage
      sort_order: 0=ascending, 1=descending, 2=bitonic
      unroll: Loop unrolling factor
  """
  assert (unroll % 2) == 0, 'Static sort order requires even unroll factor'

  num_pairs = array_ref.shape[-1] // 2 ** (substage + 1)
  unroll = min(unroll, num_pairs)

  if dim1_length is not None:
    raise NotImplementedError("Dynamic array size not yet supported")

  @loop(0, pl.cdiv(num_pairs, unroll))
  def process_pairs(loop_idx):
    outputs = []
    aux_outputs = [[] for _ in aux_refs]
    is_descending = sort_order == 1
    pair_length = 2 ** (substage + 1)
    slice_length = unroll * pair_length
    array_slice, *aux_slices = (ref.at[:, pl.dslice(loop_idx * slice_length, slice_length)] for ref in (array_ref, *aux_refs))

    data, *aux_datas = (v[...] if read_then_split else v for v in (array_slice, *aux_slices))

    for i in range(unroll):
      pair_offset = (loop_idx * unroll + i) * pair_length
      half_length = 2 ** substage

      # Slice subarrays to be compared
      left, *aux_lefts  = (v[:, i * pair_length: i * pair_length + half_length] for v in (data, *aux_datas))
      right, *aux_rights = (v[:, i * pair_length + half_length: i * pair_length + 2 * half_length] for v in (data, *aux_datas))

      # Determine the swap mask based on the main array's values
      if sort_order != 2:
        mask = (left > right) if is_descending else (left < right)
      else:
        is_descending_bit = create_bit_indicator(stage, dim1_offset + pair_offset)
        mask = jnp.bitwise_xor(is_descending_bit, left < right)

      # Store the sorted pairs for the main array
      outputs.append([jnp.where(m, left, right) for m in (mask, ~mask)])

      # Apply the *same mask* to the auxiliary arrays and store the results
      for i_aux, (l, r) in enumerate(zip(aux_lefts, aux_rights)):
        aux_outputs[i_aux].append([jnp.where(m, l, r) for m in (mask, ~mask)])

    if concat_before_writeout:
      # Concatenate all sorted pairs and write the entire slice back at once
      array_slice[...] = jnp.concatenate(jax.tree.leaves(outputs), axis=-1)
      for i_aux, aux_slice in enumerate(aux_slices):
        aux_slice[...] = jnp.concatenate(jax.tree.leaves(aux_outputs[i_aux]), axis=-1)
    else:
      # Write back each sorted pair individually
      for i in range(unroll):
        pair_offset = (loop_idx * unroll + i) * pair_length
        half_length = 2 ** substage

        # Write back main array pairs
        array_ref[:, pl.dslice(pl.multiple_of(pair_offset, half_length), half_length)] = outputs[i][0]
        array_ref[:, pl.dslice(pl.multiple_of(pair_offset + half_length, half_length), half_length)] = outputs[i][1]

        # Write back auxiliary array pairs
        for i_aux, aux_ref_i in enumerate(aux_refs):
          aux_ref_i[:, pl.dslice(pl.multiple_of(pair_offset, half_length), half_length)] = aux_outputs[i_aux][i][0]
          aux_ref_i[:, pl.dslice(pl.multiple_of(pair_offset + half_length, half_length), half_length)] = aux_outputs[i_aux][i][1]


def _compute_subtile_substages(
    array_ref,
    num_substages: int,
    stage: int,
    sort_order: int,
    aux_refs=(),
    dim1_offset: int = 0,
):
  """Execute multiple substages of bitonic sort where compared values are from the same tile."""
  index = lax.broadcasted_iota(jnp.int32, (NUM_SUBLANES, NUM_LANES), 1)
  tile_independent = num_substages < LOG_LANES

  def compute_substage(substage, all_tiles):
    tiles, *auxs_tiles = all_tiles
    is_right_half = create_bit_indicator(substage)

    if sort_order == 0:
      base_should_swap = is_right_half
    elif sort_order == 1:
      base_should_swap = ~is_right_half
    elif sort_order == 2:
      if tile_independent:
        base_should_swap = jnp.bitwise_xor(is_right_half, create_bit_indicator(stage))
      else:
        base_should_swap = None

    permutation = jnp.bitwise_xor(index, 1 << substage)
    permuted_tiles, *permuted_auxs_tiles = ([gather_2d(tile, permutation) for tile in t] for t in (tiles, *auxs_tiles))

    output_tiles = []
    output_auxs_tiles = [[] for _ in auxs_tiles]
    flip_frequency = 1 << stage

    for tile_idx, (tile, permuted_tile) in enumerate(zip(tiles, permuted_tiles, strict=True)):
      if sort_order == 2 and not tile_independent:
        tile_offset = dim1_offset + (tile_idx * NUM_LANES)
        # only i32,i32->i1 lowers, not i1,i1->i1 so we upcast and then downcast
        should_swap_local = jnp.where(
            (tile_offset & flip_frequency) > 0,
            (~is_right_half).astype(jnp.int32),
            is_right_half.astype(jnp.int32)
        ).astype(bool)
      else:
        should_swap_local = base_should_swap

      condition = (tile > permuted_tile) == should_swap_local
      output_tiles.append(
          jnp.where(condition, tile, permuted_tile)
      )
      for aux_tiles, permuted_aux_tiles, output_aux_tiles in zip(auxs_tiles, permuted_auxs_tiles, output_auxs_tiles):
        output_aux_tiles.append(
          jnp.where(condition, aux_tiles[tile_idx], permuted_aux_tiles[tile_idx])
        )
    return (output_tiles, *output_auxs_tiles)

  assert num_substages <= LOG_LANES
  all_tiles = [split_array_to_tiles(ref) for ref in (array_ref, *aux_refs)]

  for i in range(num_substages):
    substage = num_substages - 1 - i
    all_tiles = compute_substage(substage, all_tiles)

  return (join_tiles_to_array(array_ref.shape, t) for t in all_tiles)

def compute_subtile_substages(
    array_ref,
    *,
    num_substages: int,
    stage: int,
    sort_order: int,
    aux_refs=(),
    unroll: int = 256,
    dim1_offset: int = 0,
    dim1_length: jax.Array = None,
    slice_dim1: int = None
):
  """Orchestrate subtile sorting operations with proper blocking."""
  if slice_dim1 is None:
    slice_dim1 = min(unroll * NUM_LANES, array_ref.shape[1])
  if dim1_length is None:
    dim1_length = array_ref.shape[1]

  unroll_dim0 = (unroll * NUM_LANES) // slice_dim1
  slice_dim0 = min(unroll_dim0 * NUM_SUBLANES, array_ref.shape[0])
  unroll = (slice_dim0 * slice_dim1) // (NUM_SUBLANES * NUM_LANES)

  grid_dim0 = array_ref.shape[0] // slice_dim0
  grid_dim1 = dim1_length // slice_dim1

  @loop(0, grid_dim0 * grid_dim1)
  def process_block(loop_idx):
    block_row = loop_idx // grid_dim1
    block_col = loop_idx % grid_dim1

    array_ref_slice, *aux_ref_slices= (ref.at[
        pl.dslice(block_row * slice_dim0, slice_dim0),
        pl.dslice(block_col * slice_dim1, slice_dim1)
    ] for ref in (array_ref, *aux_refs))

    outputs = _compute_subtile_substages(
        array_ref_slice,
        aux_refs=aux_ref_slices,
        num_substages=num_substages,
        stage=stage,
        sort_order=sort_order,
        dim1_offset=dim1_offset + (block_col * slice_dim1)
    )
    for ref, output in zip((array_ref_slice, *aux_ref_slices), outputs):
      ref[...] = output


# ============================================================================
# Bitonic Sort - Stage Orchestration
# ============================================================================

def compute_stages(
    start_stage: int,
    end_stage: int,
    array_ref,
    sort_order: int,
    aux_refs=(),
    unroll_crosstile: int = 64,
    unroll_subtile: int = 64,
    dim1_offset: int = 0
):
  """Execute a range of bitonic sorting stages."""
  log_n = _log2(array_ref.shape[-1])

  @loop(start_stage, end_stage)
  def run_stage(stage):
    for i in range(log_n):
      substage = log_n - 1 - i
      if (substage >= LOG_LANES):
        @pl.when(stage > substage)
        def _():
          compute_crosstile_substage(
              array_ref,
              aux_refs=aux_refs,
              substage=substage,
              stage=stage,
              sort_order=sort_order,
              unroll=unroll_crosstile,
              dim1_offset=dim1_offset
          )
      elif substage == (LOG_LANES - 1):
        @pl.when(stage > substage)
        def _():
          compute_subtile_substages(
              array_ref,
              aux_refs=aux_refs,
              num_substages=substage + 1,
              stage=stage, sort_order=sort_order,
              dim1_offset=dim1_offset,
              unroll=unroll_subtile,
          )
      else:
        @pl.when(stage  == (substage+1))
        def _():
          compute_subtile_substages(
              array_ref,
              aux_refs=aux_refs,
              num_substages=substage + 1,
              stage=stage, sort_order=sort_order,
              dim1_offset=dim1_offset,
              unroll=unroll_subtile,
          )


def bitonic_sort(
    array_ref,
    aux_refs = (),
    k: int = None,
    dim: int = None,
    descending: bool = False
):
  """Core bitonic sort implementation."""
  if dim is None:
    dim = len(array_ref.shape) - 1
  if dim != len(array_ref.shape) - 1:
    raise ValueError("Only sorting along the last dimension is supported")

  if k is None:
    k = array_ref.shape[-1]

  log_n = _log2(array_ref.shape[dim])
  if 2**log_n != array_ref.shape[dim]:
    raise ValueError("Size along sort dimension must be a power of 2")

  log_k = _log2(k)
  if 2**log_k != k:
    raise ValueError("k must be a power of 2")

  if array_ref.shape[0] > 2**15:
    raise ValueError("Index packing requires shape[0] <= 32768")

  # Execute bitonic stages
  compute_stages(
    1, log_n, array_ref,
    aux_refs=aux_refs,
    sort_order=2)
  compute_stages(
    log_n, log_n + 1, array_ref,
    aux_refs=aux_refs,
    sort_order=(1 if descending else 0)
  )


def sort_kernel(
    input_ref,
    output_ref,
    output_index_ref,
    *,
    descending: bool
):
  """Pallas kernel for sorting."""
  return_indices = output_index_ref is not None
  pack_indices = (input_ref.dtype==jnp.bfloat16 and input_ref.shape[-1]<2**16)
  use_index_scratch=return_indices and not pack_indices and not (output_index_ref.shape==input_ref.shape and output_index_ref.dtype==jnp.int32)
  @functools.partial(pl.run_scoped,
    float32_scratch_ref=pltpu.VMEM(input_ref.shape, jnp.float32) if input_ref.dtype != jnp.float32 else None,
    index_scratch_ref=pltpu.VMEM(input_ref.shape, jnp.int32) if use_index_scratch else None,
  )
  def _(float32_scratch_ref, index_scratch_ref):
    if float32_scratch_ref is None:
      working_ref = input_ref
    else:
      working_ref = float32_scratch_ref
      working_ref[...] = input_ref[...].astype(jnp.float32)

    aux_refs = []
    if return_indices:
      indices = jax.lax.broadcasted_iota(jnp.int32, input_ref.shape, 1)
      if pack_indices:
        working_ref[...] = pack_value_with_index(working_ref[...], indices)
        assert index_scratch_ref is None
      else:
        if index_scratch_ref is not None:
          index_ref = index_scratch_ref
        else:
          index_ref = output_index_ref
        index_ref[...] = indices
        aux_refs.append(index_ref)

    k = output_ref.shape[-1]
    bitonic_sort(
      working_ref,
      aux_refs=aux_refs,
      k=max(k, NUM_LANES),
      descending=descending)

    if return_indices:
      if pack_indices:
        values, indices = unpack_value_and_index(working_ref[...])
      else:
        indices = index_ref
        values = working_ref
      output_index_ref[...] = indices[..., :k].astype(output_index_ref.dtype)
    else:
      values = working_ref

    output_ref[...] = values[..., :k].astype(output_ref.dtype)


@functools.partial(
    jit,
    static_argnames=("k", "block_size", "output_dtype", "return_indices", "inplace", "descending")
)
def sort_pallas(
    x,
    k=None,
    block_size=None,
    return_indices=False,
    output_dtype=None,
    inplace=False,
    descending=False
):
  """
  High-level interface for Pallas-based sorting on TPU.

  Args:
      x: Input array to sort (2D)
      k: Number of top elements to return (default: all)
      block_size: Token blocking size for memory efficiency
      return_indices: Whether to return original indices
      output_dtype: Output data type
      inplace: Whether to modify input array
      descending: Sort in descending order
  """
  if x.ndim != 2:
    raise ValueError('Only 2D inputs supported')

  if k is None:
    k = x.shape[-1]

  if inplace and k!=x.shape[-1]:
    raise ValueError('Cannot reuse input buffer if topk requested')

  if block_size is None:
    block_size = min(max(NUM_SUBLANES, (2**14) // x.shape[-1]), x.shape[0])

  if x.dtype not in (jnp.bfloat16, jnp.float32):
    raise NotImplementedError('Only f32 and bf16 inputs supported')

  if output_dtype is None:
    output_dtype = x.dtype

  output_shapes = (
      jax.ShapeDtypeStruct((x.shape[0], k), output_dtype),
      jax.ShapeDtypeStruct((x.shape[0], k), jnp.int32) if return_indices else None,
  )

  input_spec = pl.BlockSpec((block_size, x.shape[-1]), lambda i: (i, 0))
  output_specs = (
      pl.BlockSpec((block_size, k), lambda i: (i, 0)),
      pl.BlockSpec((block_size, k), lambda i: (i, 0)) if return_indices else None,
  )

  val, index = pl.pallas_call(
      functools.partial(sort_kernel, descending=descending),
      out_shape=output_shapes,
      in_specs=(input_spec,),
      out_specs=output_specs,
      grid=(x.shape[0] // block_size,),
      input_output_aliases={0: 0} if inplace else {}
  )(x)
  if return_indices:
    return val, index
  return val


# ============================================================================
# Large Array Sorting (Multi-Stage HBM)
# ============================================================================

class AsyncCopyAggregator:
  """Bundles multiple async copy operations as a single copy operation."""

  def __init__(self, copy_descriptors):
    self.copy_descriptors = tuple(copy_descriptors)

  def wait(self):
    """Wait for all copy operations to complete."""
    for descriptor in self.copy_descriptors:
      descriptor.wait()


def _substage_hbm_kernel(
    input_hbm_ref,
    aux_input_hbm_refs,
    substage_ref,
    stage_ref,
    output_hbm_ref,
    aux_output_hbm_refs,
    input_semaphores,
    output_semaphores,
    input_vmem_ref,
    aux_input_vmem_refs,
    output_vmem_ref,
    aux_output_vmem_refs,
    sort_order: int
):
  """Kernel for running a substage which do not fit in VMEM."""
  # Handle sublane dimension indexing
  sublane_block = input_vmem_ref.shape[-2]
  sublane_slice = pl.dslice(pl.program_id(0) * sublane_block, sublane_block)
  input_hbm_ref = input_hbm_ref.at[sublane_slice]
  output_hbm_ref = output_hbm_ref.at[sublane_slice]

  substage = substage_ref[0]
  stage = stage_ref[0]
  slice_length = input_vmem_ref.shape[-1]
  array_length = input_hbm_ref.shape[-1]
  pair_length = 2 ** (substage + 1)
  num_pairs = array_length // pair_length
  slices_per_pair = (pair_length // 2) // slice_length

  def compute_start_index(i):
    pair_idx = i // slices_per_pair
    pair_subslice_idx = i % slices_per_pair
    return pair_idx * pair_length + pair_subslice_idx * slice_length

  def perform_dma(i, is_load):
    """Perform DMA operation (load or store)."""
    buffer_slot = lax.rem(i, 2)
    left_start = compute_start_index(i)
    right_start = left_start + (pair_length // 2)
    input_dma_refs = (
      (input_hbm_ref, *aux_input_hbm_refs),
      (input_vmem_ref, *aux_input_vmem_refs)
    )
    output_dma_refs = (
      (output_hbm_ref, *aux_output_hbm_refs),
      (output_vmem_ref, *aux_output_vmem_refs)
    )
    copies = []
    for i_ref, (hbm_ref, vmem_ref) in enumerate(zip(
        input_dma_refs if is_load else output_dma_refs,
        strict=True
      )):
        for vmem_slot, start in enumerate((left_start, right_start)):
          # Compiler fails to recognize start indices are multiples of num_lanes, so we tell the compiler explicitly
          start = pl.multiple_of(start, NUM_LANES)
          hbm_ref_slice = hbm_ref.at[:, pl.dslice(start, slice_length)]
          vmem_ref_slice = vmem_ref.at[buffer_slot, vmem_slot]
          sem = input_semaphore.at[buffer_slot, vmem_slot, i_ref]
          src, dst = (hbm_ref_slice, vmem_ref_slice) if is_load else (hbm_ref_slice, vmem_ref_slice)
          copies.append(
            pltpu.async_copy(
              src_ref=src,
              dst_ref=dst,
              sem=sem,
          ))
    return AsyncCopyAggregator(copies)

  load_dma = functools.partial(perform_dma, is_load=True)
  store_dma = functools.partial(perform_dma, is_load=False)

  def compute_comparison(loop_idx):
    """Perform comparison and swap logic."""
    start_idx = compute_start_index(loop_idx)
    slot = lax.rem(loop_idx, 2)
    left, right = input_vmem_ref[slot]

    if sort_order != 2:
      is_descending = (sort_order == 1)
      mask = (left > right) if is_descending else (left < right)
    else:
      is_descending = create_bit_indicator(stage, start_idx)
      mask = jnp.bitwise_xor(is_descending, left < right)

    for i, m in enumerate((mask, ~mask)):
      output_vmem_ref[slot, i] = jnp.where(m, left, right)

      for (aux_input_ref, aux_output_ref) in zip(aux_input_refs, aux_output_refs):
        aux_output_vmem_ref[slot, i] = jnp.where(m, *aux_input_vmem_ref[slot])

  num_iterations = input_hbm_ref.shape[-1] // (2 * slice_length)
  assert num_iterations > 0

  # Pipeline: Load -> Compute -> Store
  initial_load = load_dma(0)
  if num_iterations > 1:
    next_load = load_dma(1)

  initial_load.wait()
  compute_comparison(0)

  if num_iterations == 1:
    store_dma(0).wait()
    return

  next_load.wait()

  @loop(1, num_iterations - 1)
  def pipeline_iteration(loop_idx):
    store_op = store_dma(loop_idx - 1)
    load_op = load_dma(loop_idx + 1)
    compute_comparison(loop_idx)
    store_op.wait()
    load_op.wait()

  store_op = store_dma(num_iterations - 2)
  compute_comparison(num_iterations - 1)
  store_op.wait()
  store_dma(num_iterations - 1).wait()


@functools.partial(
    jax.jit,
    static_argnames=('sort_order', 'block_shape', 'inplace')
)
def compute_substage_vmem_efficient(
    x,
    substage: int,
    stage: int,
    sort_order: int,
    auxs=(),
    block_shape=None,
    inplace=True,
):
  """Runs a substage without loading the full lane dimension into VMEM."""
  if block_shape is None:
    block_shape = (NUM_SUBLANES, 2**16)

  checkify.check(substage >= LOG_LANES, 'Intra tile comparisons not supported')
  slice_length = block_shape[-1]
  checkify.check(slice_length <= 2**substage, 'invalid slice length, sections of length {} (2**substage) sliced into chunks of size {}', 2**substage, slice_length)
  checkify.check(substage < stage, 'substage greater than stage is not valid, substage={}, stage={}', substage, stage)

  # HBM-VMEM transfers handled manually as loading and storing two blocks from the same array (inplace) is not expressible in BlockSpecs
  input_specs = (
      pl.BlockSpec(memory_space=pltpu.ANY),
      (pl.BlockSpec(memory_space=pltpu.ANY),)*len(auxs),
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(memory_space=pltpu.SMEM),
  )

  output_shape = (
    jax.ShapeDtypeStruct(x.shape, x.dtype),
    jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), auxs)
  )
  num_refs = 1+len(auxs)
  aux_vmems = jax.tree.map(lambda x: pltpu.VMEM((2, 2, *block_shape), x.dtype), auxs)

  return pl.pallas_call(
      functools.partial(
          _substage_hbm_kernel,
          sort_order=sort_order
      ),
      # indexing in outer loop over sublane dimension is handled inside the kernel, as pltpu.ANY memory space doesnt support block specs
      grid=(x.shape[0] // block_shape[0],),
      out_shape=output_shape,
      in_specs=input_specs,
      out_specs=input_specs[:2],
      input_output_aliases={0: 0} if inplace else {},
      # (2,2) = (slot, left/right array for swap)
      scratch_shapes=(
          pltpu.SemaphoreType.DMA((2, 2, num_refs)),
          pltpu.SemaphoreType.DMA((2, 2, num_refs)),
          pltpu.VMEM((2, 2, *block_shape), jnp.float32),
          aux_vmems,
          pltpu.VMEM((2, 2, *block_shape), jnp.float32),
          aux_vmems,
      )
  )(x, auxs, substage[None], stage[None])


def subsort_kernel(
    input_ref,
    aux_refs,
    stage_ref,
    output_ref,
    float32_scratch_ref,
    *,
    sort_order: int
):
  """Kernel for sorting subsequences of input for substages which fit in VMEM."""
  if float32_scratch_ref is None:
    assert input_ref.dtype == jnp.float32
    working_ref = input_ref
  else:
    working_ref = float32_scratch_ref
    working_ref[...] = input_ref[...].astype(jnp.float32)

  # to keep track of global index for bitonic sort order (based off stage)
  dim1_offset = pl.program_id(1) * input_ref.shape[-1]

  if stage_ref is None:
    # Run all stages from 1 to log2(length)
    # All stages run with songle sort order (a 'normal' sort is bitonic order until the last stage which is ascending/descending)
    compute_stages(
      1, _log2(input_ref.shape[-1]),
      working_ref,
      aux_refs=aux_refs,
      sort_order=sort_order,
      dim1_offset=dim1_offset
    )
  else:
    # Run a single stage
    stage = stage_ref[0]
    compute_stages(
        stage, stage + 1,
        working_ref,
        aux_refs=aux_refs,
        sort_order=sort_order,
        dim1_offset=dim1_offset
    )
  output_ref[...] = working_ref[...].astype(output_ref.dtype)


@functools.partial(
    jit,
    static_argnames=("block_size", "num_substages", "sort_order")
)
def compute_substages(
    x,
    stage,
    num_substages: int,
    sort_order: int,
    auxs=(),
    block_size=None
):
  """
  Runs substages from num_substages-1 down to 0 as part of a stage

  Args:
      x: Input array
      stage: Specific stage to run (or None to run stages 1 to 'num_substages')
      num_substages: how many substages to run
      sort_order: 0=ascending, 1=descending, 2=bitonic
      block_size: Token blocking size
  """
  if x.ndim != 2:
    raise ValueError('Only 2D inputs supported')

  if block_token is None:
    block_token = NUM_SUBLANES

  subsequence_length = 2**num_substages
  if subsequence_length > 2**19:
    raise ValueError('block size exceeds VMEM limits, max subsequence length is 524288')
  x_block_spec = pl.BlockSpec((block_token, subsequence_length), lambda i, j: (i, j))
  in_specs=(
      x_block_spec,
      (x_block_spec,)*len(auxs),
      pl.BlockSpec(memory_space=pltpu.SMEM) if stage is not None else None,
  )
  return pl.pallas_call(
      functools.partial(subsort_kernel, sort_order=sort_order),
      out_shape=(
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), auxs),
      ),
      in_specs=in_specs,
      out_specs=in_specs[:2],
      scratch_shapes=(
          pltpu.VMEM((block_token, subsequence_length), jnp.float32)
          if x.dtype != jnp.float32 else None,
      ),
      grid=(x.shape[0] // block_token, x.shape[-1] // subsequence_length)
  )(x, auxs, stage[None] if stage is not None else None)


@functools.partial(jax.jit, static_argnames=('num_vmem_substages', 'descending', 'return_indices'))
def sort_pallas_vmem_efficient(x, num_vmem_substages=19, descending=False, return_indices=False):
  """
  Sort large arrays using a hybrid HBM-VMEM approach.

  This function handles arrays larger than VMEM by breaking them into
  subsections, sorting in VMEM, then merging with HBM-based operations.

  Args:
      x: Input array to sort
      num_vmem_substages: log2 of max size that fits in VMEM (default: 2^19)
      descending: Sort in descending order
  """
  num_stages = _log2(x.shape[-1])

  # If array fits in VMEM, use simple sort
  if num_stages <= num_vmem_substages:
    return sort_pallas(x, descending=descending, return_indices=return_indices)

  auxs = (jax.lax.broadcasted_iota(jnp.int32, x.shape, 1),) if return_indices else ()

  def run_stage(stage, carry, *, order):
    """Execute a complete sorting stage."""
    def _compute_substage_vmem_efficient_body(i, carry):
      substage = stage - 1 - i
      x, auxs = carry
      return compute_substage_vmem_efficient(
          x, substage, stage, auxs=auxs, sort_order=order
      )

    # First: HBM-based substages for cross-VMEM-block operations
    x, auxs = jax.lax.fori_loop(0, stage - num_vmem_substages, _compute_substage_vmem_efficient_body, carry)

    # Then: VMEM-based substages for within-block operations
    return compute_substages(x, stage, num_vmem_substages, auxs=auxs, order=order)

  # Initial bitonic sorting of VMEM-sized blocks up to VMEM-sized subsequences
  carry = compute_substages(x, auxs=auxs, stage=None, num_vmem_substages=num_vmem_substages, order=2)

  # Merge blocks through successive stages
  carry = jax.lax.fori_loop(
      num_vmem_substages, num_stages,
      functools.partial(run_stage, order=2), carry
  )

  # Final stage determines overall sort direction
  x, auxs = run_stage(num_stages, carry, order=int(descending))
  if return_indices:
    return x, auxs[0]
  return x


# ============================================================================
# Top-K Operations
# ============================================================================

def blockwise_topk(
    logits,
    k: int,
    block_topk_values=None,
    block_topk_indices=None,
    start_k: int = 0,
    num_blocks: int = NUM_LANES,
    mode: str = "jax",
):
  """
  Compute blockwise top-k using a sinking sort approach.

  Args:
      logits: Input logits to find top-k from
      k: Number of top elements to find
      block_topk_values: Pre-allocated buffers for values
      block_topk_indices: Pre-allocated buffers for indices
      start_k: Starting position (for incremental top-k)
      num_blocks: Number of blocks to process
      mode: "jax" or "pallas" execution mode
  """
  num_tokens = logits.shape[0]

  if start_k != 0 and (block_topk_values is None or block_topk_indices is None):
    raise ValueError(
        "start_k > 0 requires pre-computed buffers in "
        "block_topk_values and block_topk_indices"
    )

  if mode == "jax":
    block_topk_values = [
        jnp.full(
            (num_tokens, num_blocks),
            jnp.finfo(logits.dtype).min,
            dtype=logits.dtype
        )
        for _ in range(k)
    ]
    block_topk_indices = [
        jnp.full((num_tokens, num_blocks), 0, dtype=jnp.int32)
        for _ in range(k)
    ]
  elif mode == "pallas":
    if block_topk_values is None or block_topk_indices is None:
      raise ValueError(
          "Pallas mode requires pre-allocated buffers"
      )

  def process_block(block_idx, carry):
    """Process a single tile with sinking sort."""
    values_list, indices_list = carry

    # Extract current block
    if mode == "pallas":
      current_values = logits[..., pl.dslice(num_blocks * block_idx, num_blocks)]
    elif mode == "jax":
      current_values = jax.lax.dynamic_slice_in_dim(
          logits, block_idx * num_blocks, num_blocks, axis=1
      )
    else:
      raise ValueError("mode must be 'pallas' or 'jax'")

    current_indices = jnp.full((num_tokens, num_blocks), block_idx, jnp.int32)

    # Sinking sort: compare and swap through k levels
    for level in range(k):
      if level < start_k:
        # Invalidate already-found elements
        current_values = jnp.where(
            current_indices == indices_list[level],
            float("-inf"),
            current_values
        )
      else:
        # Exchange with stored top-k
        mask = current_values > values_list[level]

        values_list[level], current_values = (
            jnp.where(m, current_values, values_list[level])
            for m in (mask, ~mask)
        )
        indices_list[level], current_indices = (
            jnp.where(m, current_indices, indices_list[level])
            for m in (mask, ~mask)
        )

    return (values_list, indices_list)

  return unrolled_fori_loop(
      logits.shape[-1] // num_blocks,
      process_block,
      (block_topk_values, block_topk_indices),
      unroll=16,
  )


def dense_gather_kernel(values_ref, indices_ref, output_ref):
  """Gather values by indexing in to all of value with a mask, rather than a single gather per index."""
  # TODO: consider fori_loop and unroll for large shapes
  for token_offset in range(0, values_ref.shape[0], NUM_SUBLANES):
    token_slice = pl.dslice(token_offset, NUM_SUBLANES)
    output = jnp.zeros((NUM_SUBLANES, NUM_LANES), values_ref.dtype)
    indices = indices_ref[token_offset: token_offset + NUM_SUBLANES]

    for block_offset in range(0, values_ref.shape[1], NUM_LANES):
      mask = (indices >= block_offset) & (indices < block_offset + NUM_LANES)
      output = jnp.where(
          mask,
          gather_2d(
              values_ref[
                  token_offset: token_offset + NUM_SUBLANES,
                  block_offset: block_offset + NUM_LANES
              ],
              indices % NUM_LANES
          ),
          output,
      )

    output_ref[token_slice] = output[:, :output_ref.shape[1]].astype(output_ref.dtype)


def topk_from_packed(x_ref, k: int):
  """
  Extract top-k from packed float32 array.

  Args:
      x_ref: Reference to packed array (bfloat16 values + uint16 indices)
      k: Number of top elements to extract

  Returns:
      Tuple of (values, indices) for top-k elements
  """
  assert x_ref.dtype == jnp.float32

  iota = jax.lax.broadcasted_iota(jnp.int32, x_ref.shape, 1)
  assert x_ref.shape[-1] < 2**16, \
      'Packing requires vocab size < 65536 for uint16 indices'
  x_ref[...] = pack_value_with_index(x_ref[...], iota)
  bitonic_sort(x_ref, k=max(k, NUM_LANES), descending=True)
  values, indices = unpack_value_and_index(x_ref[...])
  return values[:, :k], indices[:, :k]


def topk_blockwise_superset_kernel(
    logits_ref,
    topk_values_ref,
    topk_indices_ref,
    max_depth_ref,
    block_topm_values_ref,
    block_topm_indices_ref,
    termination_flag_ref,
    k: int = 64,
    block_topk_schedule: tuple[int] | None = None,
    topk_schedule: tuple[int] | None = None,
):
  """
  Compute blockwise top-k supersets until global top-k is guaranteed.

  This uses an adaptive algorithm that incrementally increases m until
  the blockwise top-m's provably contain the global top-k.
  """
  # Initialize buffers
  block_size = logits_ref.shape[0]
  shape = (block_size, block_topm_values_ref.shape[1])

  token_slice = pl.dslice(pl.program_id(0) * block_size, block_size)

  block_topm_values_ref[token_slice] = jnp.full(
      shape, jnp.finfo(jnp.float32).min, dtype=jnp.float32
  )
  block_topm_indices_ref[token_slice] = jnp.full(shape, 0, dtype=jnp.int32)

  for i in range(block_size):
    max_depth_ref[pl.program_id(0) * block_size + i] = k

  termination_flag_ref[0] = 0

  # Schedule of progressively larger m values
  if block_topk_schedule is None:
    block_topk_schedule = (5, 7, 9, 12)
  block_topk_schedule = (0,) + block_topk_schedule + (k,)

  # Incremental blockwise top-k computation
  for completed_m, target_m in zip(block_topk_schedule, block_topk_schedule[1:]):

    @pl.when(termination_flag_ref[0] == 0)
    def _():
      # Compute blockwise top-m
      topk_vals, topk_idxs = blockwise_topk(
          logits_ref,
          block_topk_values=[
              block_topm_values_ref[
                  token_slice, pl.dslice(i * NUM_LANES, NUM_LANES)
              ].astype(jnp.float32)
              for i in range(target_m)
          ],
          block_topk_indices=[
              block_topm_indices_ref[
                  token_slice, pl.dslice(i * NUM_LANES, NUM_LANES)
              ]
              for i in range(target_m)
          ],
          k=target_m,
          num_blocks=NUM_LANES,
          start_k=completed_m,
          mode="pallas",
      )

      # Store results
      for i in range(completed_m, target_m):
        block_topm_values_ref[
            token_slice, pl.dslice(i * NUM_LANES, NUM_LANES)
        ] = topk_vals[i].astype(block_topm_values_ref.dtype)
        block_topm_indices_ref[
            token_slice, pl.dslice(i * NUM_LANES, NUM_LANES)
        ] = topk_idxs[i].astype(block_topm_indices_ref.dtype)

      # Termination criterion:
      # If top-(m-1) blocks contain >= k values larger than
      # the m-th largest value, then top-k is guaranteed to be in top-(m-1)
      pivot = topk_vals[target_m - 1].max(-1, keepdims=True)
      num_larger = (
          sum([(v >= pivot) for v in topk_vals[:target_m - 1]])
          .astype(jnp.float32)
          .sum(-1)
      )

      termination_flag_ref[0] = 0
      for i in range(block_size):
        contains_topk = num_larger[i] >= k
        termination_flag_ref[0] += contains_topk

        # Record depth when criterion was met
        token_idx = pl.program_id(0) * block_size + i
        current_max = max_depth_ref[token_idx]
        max_depth_ref[token_idx] = jnp.where(
            contains_topk & (current_max == k),
            target_m - 1,
            current_max
        )

      # Check if all tokens converged
      @pl.when(termination_flag_ref[0] != block_size)
      def _():
        termination_flag_ref[0] = 0

  # Final top-k extraction (done by last program)
  @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
  def _():
    # Find maximum depth across all tokens
    max_depth = jnp.array(0)
    for i in range(max_depth_ref.shape[0]):
      max_depth = jnp.maximum(max_depth, max_depth_ref[i])

    # Use appropriate sorting depth based on max_depth
    for depth_lower, depth_upper in zip(topk_schedule, topk_schedule[1:]):

      @pl.when((max_depth > depth_lower) & (max_depth <= depth_upper))
      def _():
        # Sort the blockwise superset
        values, block_local_indices = topk_from_packed(
            block_topm_values_ref.at[:, :depth_upper * NUM_LANES],
            k=NUM_LANES
        )
        topk_values_ref[...] = values.astype(topk_values_ref.dtype)

        # Reconstruct global indices
        global_indices = (
            block_topm_indices_ref[:, :depth_upper * NUM_LANES] * NUM_LANES
        ) + (
            jax.lax.broadcasted_iota(
                jnp.int32,
                block_topm_indices_ref[:, :depth_upper * NUM_LANES].shape,
                1
            ) % NUM_LANES
        )

        dense_gather_kernel(
            global_indices, block_local_indices, topk_indices_ref
        )


@functools.partial(
    jit,
    static_argnames=("k", "block_size", "block_topk_schedule", "topk_schedule"),
)
def topk_pallas(
    logits,
    k: int,
    block_size: int = 8,
    block_topk_schedule=None,
    topk_schedule=None,
):
  """
  High-level interface for adaptive blockwise top-k on TPU.

  Args:
      logits: Input logits [num_tokens, vocab_size]
      k: Number of top elements to find
      block_size: Token blocking size
      block_topk_schedule: Schedule of m values for blockwise top-m
      topk_schedule: Schedule for final sorting depth

  Returns:
      Tuple of (values, indices) for top-k elements
  """
  num_tokens, vocab_size = logits.shape

  if num_tokens % block_size != 0:
    raise ValueError("num_tokens must be divisible by block_size")

  if topk_schedule is None:
    topk_schedule = (0, 8, k)

  if k > NUM_LANES:
    raise ValueError(f"k cannot exceed {NUM_LANES}")

  output_shapes = (
      jax.ShapeDtypeStruct((num_tokens, NUM_LANES), logits.dtype),
      jax.ShapeDtypeStruct((num_tokens, NUM_LANES), jnp.int32),
      jax.ShapeDtypeStruct((num_tokens,), jnp.int32),
  )

  output_specs = (
      pl.BlockSpec(),
      pl.BlockSpec(),
      pl.BlockSpec(memory_space=pltpu.SMEM),
  )

  topk_vals, topk_idxs, depths = pl.pallas_call(
      functools.partial(
          topk_blockwise_superset_kernel,
          k=k,
          block_topk_schedule=block_topk_schedule,
          topk_schedule=topk_schedule,
      ),
      in_specs=(
          pl.BlockSpec((block_size, vocab_size), lambda i: (i, 0)),
      ),
      out_shape=output_shapes,
      scratch_shapes=(
          pltpu.VMEM((num_tokens, k * NUM_LANES), jnp.float32),
          pltpu.VMEM((num_tokens, k * NUM_LANES), jnp.int32),
          pltpu.SMEM((1,), jnp.int32),
      ),
      grid=(num_tokens // block_size,),
      out_specs=output_specs,
  )(logits)

  return topk_vals[:, :k], topk_idxs[:, :k]


# ============================================================================
#


import gzip
from glob import glob
import json
import pandas as pd
import os


k = 64
num_queries = 32
vocab_size = 2048
hidden_dim = 2880

logit_key, key_act, key_weight = jax.random.split(jax.random.key(0), 3)
x = jax.random.normal(key_act, (num_queries, hidden_dim), dtype=jnp.float32)
w = jax.random.normal(key_weight, (hidden_dim, vocab_size), dtype=jnp.float32)
logits = jax.random.normal(
    key_weight, (num_queries, vocab_size), dtype=jnp.float32
)

topk_xla = jax.jit(jax.lax.top_k, static_argnames=("k",))
approx_topk_xla = jax.jit(jax.lax.approx_max_k, static_argnames=("k",))
sort_xla = jax.jit(jnp.sort)
argsort_xla = jax.jit(jnp.argsort)
@jax.jit
def add_one(x):
  return x+1


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, None))
def matmul_and_topk_xla(x, w, k=k):
  logits = x @ w
  return jax.lax.top_k(logits, k)

def benchmark(_run, in_place: bool):
  def run():
    return jax.block_until_ready(_run(in_place))
  run()
  with jax.profiler.trace("/tmp/"):
    run()

  path = sorted(glob("/tmp/plugins/profile/*/**.json.gz"), key=os.path.getmtime)[-1]
  trace = json.load(gzip.open(path))
  df = pd.DataFrame(trace["traceEvents"])
  df = df[~df.name.isna()]
  print(df[df.name.str.contains("jit_")][['name', 'dur']])

for n in (2**13, 2**15, 2**17, 2**20):
  y = jax.random.normal(jax.random.key(0), (2**5, n), jnp.float32)
  print('y shape ', y.shape )
  def _run(in_place: bool):
    with interpret_pallas_call.force_tpu_interpret_mode(
        interpret_pallas_call.InterpretParams(in_place=in_place)
    ):
      return topk_pallas(logits, k=k, block_size=8)

  benchmark(_run, in_place=False)
  check_sort = True
  if check_sort:
    a, b,c, d, *_ = _run(in_place=False)
    print('sort ', y.shape, y.dtype)
    print("xla", a)
    print("pallas", b)
    print('match: ', (a==b).mean())
    print('argsort ', y.shape, y.dtype)
    print("xla", c)
    print("pallas", d[1])
    print('match: ', (c==d[1]).mean())

check = True

def _run(in_place: bool):
  with interpret_pallas_call.force_tpu_interpret_mode(
      interpret_pallas_call.InterpretParams(in_place=in_place)
  ):
    return (
      add_one(logits),
      topk_xla(logits, k=k),
      topk_pallas(logits, k=k, block_size=8),
      topk_pallas(logits, k=k, block_size=16),
      # Not exact. Runtime varies with recall, here run with default 0.95
      approx_topk_xla(logits, k=k),
    )

if __name__ == "__main__":
  if check:
    print("Running benchmark with in_place=False")
    benchmark(_run, in_place=False)
    print("Running benchmark with in_place=True")
    benchmark(_run, in_place=True)
    print('topk', logits.shape, logits.dtype, k)
    print("XLA: ", topk_xla(logits, k=k))
    print("\nPallas:", topk_pallas(logits, k=k))
    print(
    [
    (topk_xla(logits, k=k)[i] == topk_pallas(logits, k=k)[i]).mean() for i in range(2)
    ]
    )
