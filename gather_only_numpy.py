"""Replace only gather primitive with NumPy io_callback, leave rest as XLA."""

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.lax import slicing as lax_slicing
from jax.experimental import io_callback

# Store original gather implementation
_original_gather_p_bind = lax_slicing.gather_p.bind

def numpy_gather_via_callback(operand, start_indices, *, dimension_numbers, slice_sizes, **kwargs):
    """Gather implementation using NumPy via io_callback."""

    def numpy_gather_impl(operand, start_indices):
        """NumPy implementation of gather using take_along_axis."""
        collapsed_slice_dims = tuple(dimension_numbers.collapsed_slice_dims)
        start_index_map = tuple(dimension_numbers.start_index_map)

        # Simple case: batched gather along one dimension
        if len(start_index_map) == 1 and len(collapsed_slice_dims) == 1:
            axis = start_index_map[0]
            # Remove the trailing dimension from indices if present
            indices = start_indices.squeeze(-1) if start_indices.shape[-1] == 1 else start_indices
            result = np.take_along_axis(operand, indices.astype(np.intp), axis=axis)
            return result
        else:
            # Fallback to original for complex cases
            print(f"[Gather NumPy] Complex pattern, using original implementation")
            return _original_gather_p_bind(
                operand, start_indices,
                dimension_numbers=dimension_numbers,
                slice_sizes=slice_sizes,
                **kwargs
            )

    # Determine output shape
    # For the simple case we're handling
    collapsed_slice_dims = tuple(dimension_numbers.collapsed_slice_dims)
    start_index_map = tuple(dimension_numbers.start_index_map)

    if len(start_index_map) == 1 and len(collapsed_slice_dims) == 1:
        # Simple batched gather - output has same shape as indices (without trailing dim)
        indices_shape = start_indices.shape[:-1] if start_indices.shape[-1] == 1 else start_indices.shape
        result_shape = indices_shape
        result_dtype = operand.dtype

        return io_callback(
            numpy_gather_impl,
            jax.ShapeDtypeStruct(result_shape, result_dtype),
            operand,
            start_indices
        )
    else:
        # Use original implementation for complex cases
        return _original_gather_p_bind(
            operand, start_indices,
            dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes,
            **kwargs
        )

def install_gather_numpy_hook():
    """Install the gather NumPy hook."""
    lax_slicing.gather_p.bind = numpy_gather_via_callback
    print("Gather NumPy hook installed (XLA for everything else)")

def uninstall_gather_numpy_hook():
    """Restore original gather implementation."""
    lax_slicing.gather_p.bind = _original_gather_p_bind
    print("Gather NumPy hook uninstalled")

if __name__ == "__main__":
    install_gather_numpy_hook()
