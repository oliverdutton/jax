import time
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import inspect

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("Strategy: Full Pallas NumPy Interpreter")
print("=" * 60)

# Store original
_original_pallas_call = pl.pallas_call

class NumpyRef:
    """NumPy array reference (mutable, stateful)."""
    def __init__(self, array, parent=None, slices=None):
        self.array = array
        self.parent = parent
        self.slices = slices

    @property
    def shape(self):
        if self.slices:
            return self._get_view().shape
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def _get_view(self):
        """Get the actual view into the array."""
        if self.slices:
            return self.array[self.slices]
        return self.array

    def at(self, *slices, **kwargs):
        """Return a sliced reference."""
        # Handle pl.dslice
        actual_slices = []
        for s in slices:
            if hasattr(s, '__call__'):
                # It's a function, call it
                s = s()
            if hasattr(s, 'start') and hasattr(s, 'size'):
                # It's a dslice-like object
                actual_slices.append(slice(s.start, s.start + s.size))
            else:
                actual_slices.append(s)

        combined_slices = tuple(actual_slices) if actual_slices else self.slices
        return NumpyRef(self.array, parent=self, slices=combined_slices)

    def __getitem__(self, key):
        """Get values."""
        view = self._get_view()
        result = view[key]
        # Return as regular array
        if isinstance(result, np.ndarray):
            return result
        return result

    def __setitem__(self, key, value):
        """Set values (mutates in place!)."""
        view = self._get_view()
        view[key] = np.asarray(value) if hasattr(value, '__array__') else value

    def __repr__(self):
        return f"NumpyRef(shape={self.shape}, dtype={self.dtype})"

class PallasContext:
    """Context for executing Pallas kernel with NumPy."""
    def __init__(self, grid_idx, grid_shape):
        self.grid_idx = grid_idx
        self.grid_shape = grid_shape

    def program_id(self, axis):
        """Return current position in grid."""
        return self.grid_idx[axis] if axis < len(self.grid_idx) else 0

    def num_programs(self, axis):
        """Return grid size."""
        return self.grid_shape[axis] if axis < len(self.grid_shape) else 1

# Global context
_pallas_ctx = None

def pl_program_id(axis):
    """Implementation of pl.program_id."""
    if _pallas_ctx:
        return _pallas_ctx.program_id(axis)
    return 0

def pl_num_programs(axis):
    """Implementation of pl.num_programs."""
    if _pallas_ctx:
        return _pallas_ctx.num_programs(axis)
    return 1

def pl_dslice(start, size):
    """Implementation of pl.dslice."""
    # Return a slice object with metadata
    class DSlice:
        def __init__(self, start, size):
            self.start = start
            self.size = size

        def __call__(self):
            return slice(self.start, self.start + self.size)

    return DSlice(start, size)()  # Return actual slice

def pl_loop(start, end, unroll=1):
    """Decorator for pl.loop - converts to Python for loop."""
    def decorator(body_fn):
        def wrapper(*args, **kwargs):
            # Execute body in a Python loop
            for i in range(start, end):
                body_fn(i, *args, **kwargs)
        return wrapper
    return decorator

def pl_when(condition):
    """Decorator for pl.when - converts to Python if."""
    def decorator(body_fn):
        def wrapper(*args, **kwargs):
            if condition:
                body_fn(*args, **kwargs)
        return wrapper
    return decorator

def numpy_pallas_call(kernel, out_shape, *,
                     grid=None,
                     in_specs=None,
                     out_specs=None,
                     scratch_shapes=None,
                     interpret=False,
                     **kwargs):
    """NumPy implementation of pallas_call."""

    # Only intercept interpret mode
    if not interpret:
        print("  [Not interpret mode, using original pallas_call]")
        return _original_pallas_call(kernel, out_shape, grid=grid,
                                    in_specs=in_specs, out_specs=out_specs,
                                    scratch_shapes=scratch_shapes,
                                    interpret=interpret, **kwargs)

    print(f"  [NumPy Pallas Interpreter: grid={grid}]")

    # Normalize grid
    if grid is None:
        grid = (1,)
    elif isinstance(grid, int):
        grid = (grid,)

    # Normalize out_shape
    if not isinstance(out_shape, tuple):
        out_shape = (out_shape,)
    if isinstance(out_shape[0], (list, tuple)):
        out_shape = out_shape[0]

    def numpy_kernel_wrapper(*args):
        """Execute kernel across grid with NumPy."""
        global _pallas_ctx

        # Convert inputs to NumPy
        numpy_args = []
        for arg in args:
            if hasattr(arg, '__array__'):
                numpy_args.append(np.asarray(arg))
            else:
                numpy_args.append(arg)

        # Determine number of inputs and outputs from specs
        num_inputs = len(numpy_args)
        num_outputs = len(out_shape)

        # Initialize outputs
        outputs = []
        for out_spec in out_shape:
            if hasattr(out_spec, 'shape'):
                outputs.append(np.zeros(out_spec.shape, dtype=out_spec.dtype))
            else:
                outputs.append(np.zeros((1,), dtype=np.float32))

        # Iterate over grid
        import itertools
        for grid_idx in itertools.product(*[range(g) for g in grid]):
            # Set up context
            _pallas_ctx = PallasContext(grid_idx, grid)

            # Monkey-patch Pallas functions
            pl.program_id = pl_program_id
            pl.num_programs = pl_num_programs
            pl.dslice = pl_dslice
            pl.loop = pl_loop
            pl.when = pl_when

            # Get blocks for this grid cell
            input_refs = []
            for i, arg in enumerate(numpy_args):
                # Apply blocking based on in_specs
                if in_specs and i < len(in_specs) and in_specs[i] is not None:
                    spec = in_specs[i]
                    if hasattr(spec, 'block_shape') and hasattr(spec, 'index_map'):
                        block_shape = spec.block_shape
                        index_map = spec.index_map

                        # Compute block indices
                        block_indices = index_map(*grid_idx)

                        # Extract block
                        slices = tuple(slice(idx * bs, (idx + 1) * bs)
                                     for idx, bs in zip(block_indices, block_shape))

                        # Create ref to this block
                        block = arg[slices].copy()  # Copy to allow mutation
                        input_refs.append(NumpyRef(block))
                    else:
                        input_refs.append(NumpyRef(arg))
                else:
                    input_refs.append(NumpyRef(arg))

            # Output refs
            output_refs = []
            for i, out_arr in enumerate(outputs):
                if out_specs and i < len(out_specs) and out_specs[i] is not None:
                    spec = out_specs[i]
                    if hasattr(spec, 'block_shape') and hasattr(spec, 'index_map'):
                        block_shape = spec.block_shape
                        index_map = spec.index_map

                        # Compute block indices
                        block_indices = index_map(*grid_idx)

                        # Extract block
                        slices = tuple(slice(idx * bs, (idx + 1) * bs)
                                     for idx, bs in zip(block_indices, block_shape))

                        # Create ref to this block (view into output)
                        output_refs.append(NumpyRef(out_arr, slices=slices))
                    else:
                        output_refs.append(NumpyRef(out_arr))
                else:
                    output_refs.append(NumpyRef(out_arr))

            # Scratch refs
            scratch_refs = []
            if scratch_shapes:
                for scratch_spec in scratch_shapes:
                    if isinstance(scratch_spec, (list, tuple)) and hasattr(scratch_spec[0] if len(scratch_spec) > 0 else None, 'inner_aval'):
                        # It's a list of MemoryRef objects
                        for mem_ref in scratch_spec:
                            if hasattr(mem_ref, 'inner_aval'):
                                aval = mem_ref.inner_aval
                                scratch_arr = np.zeros(aval.shape, dtype=aval.dtype)
                                scratch_refs.append(NumpyRef(scratch_arr))
                    elif hasattr(scratch_spec, 'inner_aval'):
                        # Single MemoryRef object
                        aval = scratch_spec.inner_aval
                        scratch_arr = np.zeros(aval.shape, dtype=aval.dtype)
                        scratch_refs.append(NumpyRef(scratch_arr))
                    elif hasattr(scratch_spec, 'shape'):
                        # Has shape directly
                        scratch_arr = np.zeros(scratch_spec.shape, dtype=scratch_spec.dtype)
                        scratch_refs.append(NumpyRef(scratch_arr))
                    elif isinstance(scratch_spec, tuple):
                        # (shape, dtype) tuple
                        shape, dtype = scratch_spec[0], scratch_spec[1] if len(scratch_spec) > 1 else np.float32
                        scratch_arr = np.zeros(shape, dtype=dtype)
                        scratch_refs.append(NumpyRef(scratch_arr))

            # Call kernel with proper argument structure
            try:
                # Parse the actual structure from in_specs/out_specs/scratch_shapes
                # in_specs is typically: ([spec, spec, ...], stage_spec or None)
                # out_specs is typically: ((spec, spec, ...),)
                # scratch_shapes is typically: ([shape, ...], shape, ...)

                args_to_pass = []

                # Handle in_specs
                # in_specs structure: ([spec, spec, ...], stage_spec or None)
                # Should pass as: (list of refs, stage_ref)
                if in_specs:
                    if isinstance(in_specs, (list, tuple)) and len(in_specs) > 0:
                        # First element: list of input BlockSpecs -> pass as list of refs
                        if isinstance(in_specs[0], (list, tuple)):
                            in_refs_list = []
                            for i, spec in enumerate(in_specs[0]):
                                if i < len(input_refs):
                                    in_refs_list.append(input_refs[i])
                            args_to_pass.append(in_refs_list)
                        else:
                            # Single spec - still pass as list for consistency
                            args_to_pass.append(input_refs)

                        # Second element: stage_ref (if present)
                        if len(in_specs) > 1:
                            stage_ref = in_specs[1]
                            args_to_pass.append(None)  # stage_ref is usually None
                    else:
                        args_to_pass.append(input_refs)
                else:
                    args_to_pass.append(input_refs)

                # Handle out_specs
                # out_specs structure: ((spec, spec, ...),)
                # Should pass as: list of refs
                if out_specs:
                    if isinstance(out_specs, (list, tuple)) and len(out_specs) > 0:
                        # Usually: ((spec1, spec2, ...),)
                        out_specs_inner = out_specs[0] if isinstance(out_specs[0], (list, tuple)) else out_specs
                        out_refs_list = []
                        for i, spec in enumerate(out_specs_inner):
                            if i < len(output_refs):
                                out_refs_list.append(output_refs[i])
                        args_to_pass.append(out_refs_list)
                    else:
                        args_to_pass.append(output_refs)
                else:
                    args_to_pass.append(output_refs)

                # Handle scratch_shapes
                # scratch_shapes structure: ([ref, ref, ...], ref, ref, ...)
                # where lists should be passed as lists, not unpacked
                if scratch_shapes:
                    if isinstance(scratch_shapes, (list, tuple)):
                        for i, scratch_spec in enumerate(scratch_shapes):
                            if isinstance(scratch_spec, (list, tuple)):
                                # It's a list of scratch refs - pass as a list
                                refs_list = []
                                for j, _ in enumerate(scratch_spec):
                                    if len(scratch_refs) > 0:
                                        refs_list.append(scratch_refs.pop(0))
                                args_to_pass.append(refs_list)
                            else:
                                # Single scratch ref
                                if len(scratch_refs) > 0:
                                    args_to_pass.append(scratch_refs.pop(0))

                # Call kernel
                kernel(*args_to_pass)

                # Copy output blocks back
                for out_ref, out_arr in zip(output_refs, outputs):
                    if out_ref.slices:
                        out_arr[out_ref.slices] = out_ref.array
                    # Already in place if no slices

            except Exception as e:
                print(f"    Error executing kernel: {e}")
                print(f"    Kernel: {kernel}")
                print(f"    in_specs: {in_specs}")
                print(f"    out_specs: {out_specs}")
                print(f"    scratch_shapes: {scratch_shapes}")
                print(f"    num args passed: {len(args_to_pass)}")
                print(f"    args_to_pass types: {[type(a).__name__ for a in args_to_pass]}")

                import traceback
                traceback.print_exc()

                # Fall back to original
                print("    Falling back to compiled Pallas...")
                return _original_pallas_call(kernel, out_shape, grid=grid,
                                           in_specs=in_specs, out_specs=out_specs,
                                           scratch_shapes=scratch_shapes,
                                           interpret=True, **kwargs)(*args)

        # Return outputs
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    return numpy_kernel_wrapper

# Install the interpreter
pl.pallas_call = numpy_pallas_call

# Also need to patch it in the pallas module
import jax.experimental.pallas as pallas_module
pallas_module.pallas_call = numpy_pallas_call

print("\n🔥 STRATEGY 25: Full Pallas NumPy Interpreter\n")
print("Interpreting Pallas kernels with NumPy (stateful arrays + Python control flow)")

try:
    from benchmark_sort import run_benchmarks

    start_time = time.time()
    run_benchmarks()
    end_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print(f"{'=' * 60}")

finally:
    # Restore
    pl.pallas_call = _original_pallas_call
    pallas_module.pallas_call = _original_pallas_call
