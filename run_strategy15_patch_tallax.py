import time
import os
import jax
import jax.numpy as jnp

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("Strategy: Patch tallax to use Python control flow")
print("=" * 60)

# Minimal XLA optimization
os.environ['XLA_FLAGS'] = (
    '--xla_backend_optimization_level=0 '
    '--xla_llvm_disable_expensive_passes=true'
)

# Import tallax first
import tallax.tax as tax

# Store original sort function
_original_sort = tax.sort

# We need to access the internal functions
from tallax.tax.sort import (
    _sort_pallas_vmem,
    _compute_substage_hbm,
    canonicalize_operand,
    log2,
    NUM_SUBLANES,
    pad,
    float_to_sortable_int,
    sortable_int_to_float,
    unpack_bf16_u16_from_i32,
    pack_bf16_u16_to_i32,
    is_32bit,
)

def patched_sort(
    operand,
    num_keys,
    is_stable=False,
    return_argsort=False,
    descending=False,
    num_vmem_substages=None,
    block_token=None,
    interpret=False,
):
    """Modified sort that uses Python loops instead of jax.lax.fori_loop.

    This version removes the top-level @jax.jit and uses eager Python control flow
    for stage iteration, while JIT-compiling each stage individually for better
    compilation granularity and caching.
    """
    operands, shape = canonicalize_operand(operand)
    num_stages = log2(shape[1])

    if (shape[1] != 2**num_stages and
        any(not jnp.issubdtype(x.dtype, jnp.floating) for x in operands)):
        is_stable = True

    use_indices = return_argsort or is_stable
    if use_indices:
        indices = jax.lax.broadcasted_iota(jnp.int32, operands[0].shape, 1)
        if descending and is_stable:
            indices = shape[1] - indices
        indices_index = num_keys
        operands.insert(num_keys, indices)
        if is_stable:
            num_keys += 1

    if num_vmem_substages is None:
        num_vmem_substages = 18 - log2(
            len(operands) + sum(not is_32bit(x) for x in operands) * 0.5
        )

    dtypes = [x.dtype for x in operands]

    use_packed_bf16_u16 = (
        operands[0].dtype == jnp.bfloat16 and len(operands) == 2 and
        (operands[1].dtype == jnp.uint16 or
         (use_indices and shape[1] <= 2**16))
    )
    if use_packed_bf16_u16:
        operands = [pack_bf16_u16_to_i32(*operands)]
        num_keys = 1

    operands = [
        float_to_sortable_int(x)
        if jnp.issubdtype(x.dtype, jnp.floating) and i < num_keys
        else x
        for i, x in enumerate(operands)
    ]

    operands = [
        pad(x, block_shape=(NUM_SUBLANES, 'power_of_2_lanes'), prepend=(False, descending))
        for x in operands
    ]

    # The key change: handle stages with Python loop if possible
    if num_stages <= num_vmem_substages:
        # Array fits in VMEM - use original
        operands = _sort_pallas_vmem(
            operands,
            descending=descending,
            num_keys=num_keys,
            is_stable=False,
            return_argsort=False,
            block_token=block_token,
            log_n=num_stages,
            interpret=interpret
        )
    else:
        # Multi-stage sort - this is where we can optimize
        # Initial bitonic sorting of VMEM-sized blocks
        operands = _sort_pallas_vmem(
            tuple(operands),
            block_seq=2**num_vmem_substages,
            stage=None,
            descending=descending,
            num_keys=num_keys,
            is_stable=False,
            interpret=interpret
        )

        # Instead of jax.lax.fori_loop, use Python loop with per-stage JIT
        # This allows better compilation granularity and caching
        import functools

        @functools.lru_cache(maxsize=32)
        def get_stage_runner(stage, num_vmem_substages, num_keys, descending, interpret):
            """Get cached JIT-compiled stage runner."""
            @jax.jit
            def run_single_stage(operands):
                def _compute_substages_hbm_body(i, operands):
                    substage = stage - 1 - i
                    return _compute_substage_hbm(
                        operands, substage, stage, num_keys=num_keys,
                        descending=descending, interpret=interpret
                    )

                # HBM substages
                operands = jax.lax.fori_loop(
                    0, stage - num_vmem_substages, _compute_substages_hbm_body, operands
                )

                # VMEM substages
                return _sort_pallas_vmem(
                    operands,
                    block_seq=2**num_vmem_substages,
                    stage=stage,
                    descending=descending,
                    num_keys=num_keys,
                    is_stable=False,
                    interpret=interpret
                )
            return run_single_stage

        # Use Python loop for stages (eager evaluation of stage iteration)
        # Each stage gets its own JIT-compiled function with caching
        for stage in range(int(num_vmem_substages), int(num_stages) + 1):
            stage_runner = get_stage_runner(stage, num_vmem_substages, num_keys, descending, interpret)
            operands = stage_runner(operands)

    # Unpad
    if not descending:
        operands = tuple(x[:shape[0], :shape[1]] for x in operands)
    else:
        operands = tuple(x[:shape[0], -shape[1]:] for x in operands)

    if use_packed_bf16_u16:
        operands = unpack_bf16_u16_from_i32(operands[0])

    operands = tuple(
        sortable_int_to_float(x)
        if (jnp.issubdtype(dtype, jnp.floating) and
            jnp.issubdtype(x.dtype, jnp.integer))
        else x
        for x, dtype in zip(operands, dtypes)
    )

    operands = list(operands)
    if use_indices:
        indices = operands.pop(indices_index)
        if return_argsort:
            if descending and is_stable:
                indices = shape[1] - indices
            operands.append(indices)

    return tuple(operands)

# Monkey-patch the sort function BEFORE importing benchmark
tax.sort = patched_sort

print("\n🔥 STRATEGY 15: Patched tallax with Python control flow\n")
print("Monkey-patched tax.sort to use Python loops for stages")

try:
    # Now import benchmark which will use our patched version
    import functools
    import jax.numpy as jnp
    from tallax import tax
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
    # Restore original
    tax.sort = _original_sort
