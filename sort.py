import functools
import os
import time

# Set platform to CPU
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
from tallax import tax
from tallax.utils import is_cpu_platform

def run_benchmarks():
  ntoken = 8
  interpret = True
  for num_operands in range(1,2):
    for num_keys in range(1, num_operands+1):
      for n in (
          512,  # Changed from 128 to 512
      ):
        for dtype in (
            jnp.float32,
            #jnp.bfloat16,
            #jnp.int32,
        ):
          operands = list(jax.random.randint(jax.random.key(0), (num_operands, ntoken,n), jnp.iinfo(jnp.int32).min, jnp.iinfo(jnp.int32).max, jnp.int32).view(dtype)[...,:n])
          for kwargs in (
              dict(),
          ):
            x = operands[0]
            print(f'\n{(x.shape, x.dtype)}\n{num_operands=} {num_keys=} {kwargs=}')

            # Compile first (JIT compilation happens here)
            print("Compiling...")
            compile_start = time.time()
            def _run():
              return (
                  tax.sort(operands, num_keys=num_keys, interpret=interpret, **kwargs),
              )
            result = _run()  # First call triggers compilation
            jax.block_until_ready(result)
            compile_time = time.time() - compile_start
            print(f"Compilation + first run: {compile_time:.3f}s")

            # Run again to measure execution only
            print("Running (execution only)...")
            exec_start = time.time()
            result = _run()
            jax.block_until_ready(result)
            exec_time = time.time() - exec_start
            print(f"Execution time: {exec_time:.3f}s")
            print(result)

if __name__ == "__main__":
  run_benchmarks()
