import gzip
import json
import os
from glob import glob
import jax
import pandas as pd
import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# This is a standalone script, so we are copying all the dependencies here.
# Normally, this would be handled by imports.

def benchmark(_run):
  """Benchmark function and print timing from profiler trace."""
  def run():
    return jax.block_until_ready(_run())

  # Warmup
  run()

  tmpdir = "."
  with jax.profiler.trace(tmpdir):
    run()

  # Find trace file
  files = glob(f"{tmpdir}/plugins/profile/*/**.json.gz", recursive=True)
  if not files:
    print("No trace file generated.")
    return

  path = sorted(files, key=os.path.getmtime)[-1]
  try:
    with gzip.open(path, 'rb') as f:
      trace = json.load(f)
  except Exception as e:
    print(f"Failed to load trace: {e}")
    return

  if "traceEvents" not in trace:
    print("No traceEvents in trace.")
    return

  df = pd.DataFrame(trace["traceEvents"])
  if df.empty or 'name' not in df.columns:
    print("Trace dataframe empty or no name column.")
    return

  df = df[~df.name.isna()]
  df['name'] = df.name.apply(lambda s: s.split('(')[0])

  # Look for JIT compiled functions
  mask = df.name.str.contains("jit_")
  res = df[mask][['name', 'dur']]

  if not res.empty:
    print(res.to_string(index=False))
  else:
    print("No jit functions found in trace.")

def run_benchmark():
  ntoken = 8
  n = 128
  interpret = jax.default_backend() == "cpu"
  
  # For now, just create a simple test
  operands = [jax.random.normal(jax.random.PRNGKey(0), (ntoken, n), dtype=jnp.float32)]
  
  print(f'\n{(operands[0].shape, operands[0].dtype)}')
  
  # Simple test function instead of full sort
  def simple_kernel(x_ref, o_ref):
    # Just copy input to output for now
    o_ref[...] = x_ref[...]
  
  def _run():
    out_shape = jax.ShapeDtypeStruct(operands[0].shape, operands[0].dtype)
    result = pl.pallas_call(
        simple_kernel,
        out_shape=out_shape,
        interpret=interpret,
    )(operands[0])
    return result

  benchmark(_run)

if __name__ == "__main__":
  run_benchmark()
