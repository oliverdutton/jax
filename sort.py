import functools
import os
import time

# Set platform to CPU and optimize XLA compilation
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = (
    '--xla_cpu_enable_fast_math=true '
    '--xla_cpu_fast_math_honor_nans=false '
    '--xla_cpu_fast_math_honor_infs=false '
    '--xla_cpu_fast_math_honor_division=false '
    '--xla_cpu_fast_math_honor_functions=false '
    '--xla_cpu_enable_fast_min_max=true '
    '--xla_force_host_platform_device_count=1 '
    '--xla_cpu_use_thunk_runtime=false'
)

import jax
import jax.numpy as jnp
from tallax import tax
from tallax.utils import is_cpu_platform

# Enable compilation cache
jax.config.update('jax_enable_compilation_cache', True)
jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')

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

            # Enable verbose logging
            jax.config.update('jax_log_compiles', True)

            # Measure tracing vs compilation time
            print("Running with timing breakdown...")

            # First run: tracing + compilation + execution
            overall_start = time.time()
            def _run():
              return (
                  tax.sort(operands, num_keys=num_keys, interpret=interpret, **kwargs),
              )

            # Use jax.make_jaxpr to measure pure tracing time
            trace_start = time.time()
            traced_fn = jax.jit(_run).lower(*[])
            trace_time = time.time() - trace_start
            print(f"Tracing time: {trace_time:.3f}s")

            # Now compile it
            compile_start = time.time()
            compiled_fn = traced_fn.compile()
            compile_time = time.time() - compile_start
            print(f"Compilation time (XLA): {compile_time:.3f}s")

            # Execute
            exec_start = time.time()
            result = compiled_fn()
            jax.block_until_ready(result)
            exec_time = time.time() - exec_start
            overall_time = time.time() - overall_start
            print(f"Execution time: {exec_time:.3f}s")
            print(f"Total time: {overall_time:.3f}s")

            # Run again to measure execution only (cached)
            print("\nSecond run (cached)...")
            exec_start2 = time.time()
            result = _run()
            jax.block_until_ready(result)
            exec_time2 = time.time() - exec_start2
            print(f"Cached execution time: {exec_time2:.3f}s")
            print(result)

if __name__ == "__main__":
  run_benchmarks()
