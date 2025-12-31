import time
import os
import jax
import jax.numpy as jnp

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("Strategy: Unfuse compilation stages for smaller compilation units")
print("=" * 60)

# Best XLA flags
os.environ['XLA_FLAGS'] = (
    '--xla_backend_optimization_level=0 '
    '--xla_llvm_disable_expensive_passes=true'
)

# Import and patch tallax to disable stage fusion
import tallax.tax.sort as sort_module
from tallax import tax
from tallax.tax.sort import _sort_pallas_vmem

# Store original
_original_sort_pallas_vmem = _sort_pallas_vmem

def unfused_sort_pallas_vmem(
    operand,
    num_keys,
    k=None,
    block_token=None,
    block_seq=None,
    return_argsort=False,
    descending=False,
    is_stable=False,
    stage=None,
    log_n=None,
    interpret=False,
):
    """Modified version that runs stages separately instead of fused."""
    from tallax.tax.sort import canonicalize_operand, log2

    # If stage is None and we're in interpret mode, unfuse the stages
    # by calling _sort_pallas_vmem multiple times with individual stages
    if stage is None and interpret and log_n is not None and log_n <= 7:
        # Run stages 1 through log_n separately
        operands = operand
        for s in range(1, int(log_n) + 1):
            operands = _original_sort_pallas_vmem(
                operands,
                num_keys=num_keys,
                k=k,
                block_token=block_token,
                block_seq=block_seq,
                return_argsort=return_argsort,
                descending=descending,
                is_stable=is_stable,
                stage=s,  # Run one stage at a time
                log_n=log_n,
                interpret=interpret,
            )
        return operands
    else:
        # Use original for other cases
        return _original_sort_pallas_vmem(
            operand,
            num_keys,
            k,
            block_token,
            block_seq,
            return_argsort,
            descending,
            is_stable,
            stage,
            log_n,
            interpret,
        )

# Monkey-patch
sort_module._sort_pallas_vmem = unfused_sort_pallas_vmem

# Need to also patch the reference in the sort function
# This is tricky because sort has already imported it
# Let me reload the module

import importlib
import sys

# Remove from cache
if 'tallax.tax.sort' in sys.modules:
    del sys.modules['tallax.tax.sort']
if 'tallax.tax' in sys.modules:
    del sys.modules['tallax.tax']

# Re-import with our patch in place
import tallax.tax.sort
tallax.tax.sort._sort_pallas_vmem = unfused_sort_pallas_vmem

from tallax import tax

print("\n🔥 STRATEGY 18: Unfused Compilation Stages\n")
print("Running each stage separately for smaller compilation units...")

try:
    # Define benchmark inline to use patched version
    from tallax.utils import is_cpu_platform

    def run_benchmarks():
        ntoken = 8
        interpret = is_cpu_platform()
        for num_operands in range(1,2):
            for num_keys in range(1, num_operands+1):
                for n in (128,):
                    for dtype in (jnp.float32,):
                        operands = list(jax.random.randint(jax.random.key(0), (num_operands, ntoken,n), jnp.iinfo(jnp.int32).min, jnp.iinfo(jnp.int32).max, jnp.int32).view(dtype)[...,:n])
                        for kwargs in (dict(),):
                            x = operands[0]
                            print(f'\n{(x.shape, x.dtype)}\n{num_operands=} {num_keys=} {kwargs=}')
                            def _run():
                                return (
                                    tax.sort(operands, num_keys=num_keys, interpret=interpret, **kwargs),
                                )
                            print(jax.block_until_ready(_run()))

    start_time = time.time()
    run_benchmarks()
    end_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print(f"{'=' * 60}")

finally:
    # Restore
    sort_module._sort_pallas_vmem = _original_sort_pallas_vmem
