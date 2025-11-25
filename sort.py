import functools
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
          128,
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
            def _run():
              return (
                  tax.sort(operands, num_keys=num_keys, interpret=interpret, **kwargs),
              )
            print(jax.block_until_ready(_run()))

if __name__ == "__main__":
  run_benchmarks()
