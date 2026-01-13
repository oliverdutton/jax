#!/usr/bin/env python3
"""Debug top_k decompilation with verbose output."""

import jax
import jax.numpy as jnp
from jax import lax

def simple_topk(x):
    vals, idxs = lax.top_k(x, k=3)
    print(f"Inside simple_topk - vals type: {type(vals)}, vals: {vals}")
    print(f"Inside simple_topk - idxs type: {type(idxs)}, idxs: {idxs}")
    return vals, idxs

x = jnp.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0])

# Test lax.top_k directly
print("Testing lax.top_k directly:")
result = lax.top_k(x, k=3)
print(f"Result type: {type(result)}")
print(f"Result: {result}")
if isinstance(result, tuple):
    print(f"Result[0]: {result[0]}")
    print(f"Result[1]: {result[1]}")

# Test with function
print("\n\nTesting with function:")
vals, idxs = simple_topk(x)
print(f"vals: {vals}")
print(f"idxs: {idxs}")
