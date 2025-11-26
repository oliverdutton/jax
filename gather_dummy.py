"""Replace gather with a cheap dummy operation to test XLA performance without gather."""

import jax
import jax.numpy as jnp
from jax._src.lax import slicing as lax_slicing

# Store original gather implementation
_original_gather_p_bind = lax_slicing.gather_p.bind

def dummy_gather(operand, start_indices, *, dimension_numbers, slice_sizes, **kwargs):
    """Replace gather with a dummy operation that returns correct shape."""

    # Calculate what the output shape would be
    collapsed_slice_dims = tuple(dimension_numbers.collapsed_slice_dims)
    start_index_map = tuple(dimension_numbers.start_index_map)

    # For the simple batched gather case
    if len(start_index_map) == 1 and len(collapsed_slice_dims) == 1:
        # Output has same shape as indices (without trailing dim)
        if start_indices.shape[-1] == 1:
            result_shape = start_indices.shape[:-1]
        else:
            result_shape = start_indices.shape

        # Return dummy array with correct shape
        # Use zeros or ones - should be cheap
        return jnp.zeros(result_shape, dtype=operand.dtype)
    else:
        # For complex cases, use original implementation
        return _original_gather_p_bind(
            operand, start_indices,
            dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes,
            **kwargs
        )

def install_dummy_gather_hook():
    """Install the dummy gather hook."""
    lax_slicing.gather_p.bind = dummy_gather
    print("Dummy gather hook installed (gather replaced with jnp.zeros)")

def uninstall_dummy_gather_hook():
    """Restore original gather implementation."""
    lax_slicing.gather_p.bind = _original_gather_p_bind
    print("Dummy gather hook uninstalled")

if __name__ == "__main__":
    install_dummy_gather_hook()
