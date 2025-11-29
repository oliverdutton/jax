#!/usr/bin/env python3
"""
Simple categorical sampling using Gumbel max trick with jax.random.uniform.

This replicates jax.random.categorical with default parameters (replace=True, mode=None)
but as a single simple function.
"""

import jax
import jax.numpy as jnp


def categorical(key, logits, axis=-1):
    """Sample from categorical distribution using Gumbel max trick.

    This produces the same output as jax.random.categorical(key, logits, axis=axis)
    with default parameters, using only jax.random.uniform.

    Args:
        key: PRNG key
        logits: Unnormalized log probabilities (any shape)
        axis: Axis along which to sample (default: -1)

    Returns:
        Integer array of sampled indices
    """
    # Get dtype info for numerical stability (avoid log(0))
    dtype = logits.dtype
    tiny = jnp.finfo(dtype).tiny

    # Sample uniform random numbers in (tiny, 1) to avoid log(0)
    u = jax.random.uniform(key, logits.shape, dtype=dtype, minval=tiny, maxval=1.0)

    # Gumbel max trick: -log(-log(u)) + logits, then argmax
    # This is equivalent to sampling from categorical distribution
    gumbel = -jnp.log(-jnp.log(u))

    return jnp.argmax(logits + gumbel, axis=axis)


# Example usage and verification
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    # Example 1: Simple 1D logits
    logits = jnp.array([0.0, 1.0, 2.0])
    sample = categorical(key, logits)
    print(f"Example 1 - Single sample from {logits.shape} logits:")
    print(f"  Sample: {sample}")
    print()

    # Example 2: Batch of distributions
    key = jax.random.PRNGKey(43)
    logits_batch = jnp.array([
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 1.0]
    ])
    samples = categorical(key, logits_batch, axis=-1)
    print(f"Example 2 - Batch sampling from {logits_batch.shape} logits:")
    print(f"  Samples: {samples}")
    print()

    # Example 3: Verify distribution
    key = jax.random.PRNGKey(44)
    logits = jnp.array([0.0, 1.0, 2.0])
    probs = jax.nn.softmax(logits)

    # Generate many samples
    n_samples = 10000
    keys = jax.random.split(key, n_samples)
    samples = jax.vmap(lambda k: categorical(k, logits))(keys)

    # Count occurrences
    counts = jnp.array([jnp.sum(samples == i) for i in range(3)])
    empirical_probs = counts / n_samples

    print(f"Example 3 - Distribution check ({n_samples} samples):")
    print(f"  True probabilities:      {probs}")
    print(f"  Empirical probabilities: {empirical_probs}")
    print(f"  Difference:              {jnp.abs(probs - empirical_probs)}")
    print()

    # Example 4: Compare with jax.random.categorical
    print("Example 4 - Verify same output as jax.random.categorical:")
    key = jax.random.PRNGKey(100)
    logits = jnp.array([0.5, 1.0, 1.5, 0.2])

    # Our implementation
    our_sample = categorical(key, logits)

    # JAX's implementation
    jax_sample = jax.random.categorical(key, logits)

    print(f"  Our sample: {our_sample}")
    print(f"  JAX sample: {jax_sample}")
    print(f"  Match: {our_sample == jax_sample}")
