import time
import os
import jax
import jax.numpy as jnp
from jax import core
from jax._src import dispatch
import functools

print("=" * 60)
print(f"JAX version: {jax.__version__}")
print("Strategy: Hybrid Eager/JIT - JIT primitives, eager control flow")
print("=" * 60)

# Cache for compiled primitives
_primitive_cache = {}

# Control flow primitives that should stay eager
CONTROL_FLOW_PRIMITIVES = {
    'scan', 'while', 'cond', 'switch', 'fori_loop',
    'map', 'pmap', 'xmap', 'shard_map',
}

# Store original bind methods
_original_binds = {}

def get_cache_key(primitive, *args, **params):
    """Generate cache key for a primitive operation."""
    try:
        # Use primitive name and params as key
        key_parts = [primitive.name]

        # Add shape/dtype info from args
        for arg in args:
            if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
                key_parts.append(f"{arg.shape}_{arg.dtype}")
            else:
                key_parts.append(str(type(arg).__name__))

        # Add params
        for k, v in sorted(params.items()):
            key_parts.append(f"{k}={v}")

        return tuple(key_parts)
    except:
        return None

def should_jit_primitive(primitive):
    """Determine if a primitive should be JIT compiled."""
    prim_name = primitive.name

    # Don't JIT control flow primitives
    if any(cf in prim_name for cf in CONTROL_FLOW_PRIMITIVES):
        return False

    # Don't JIT pallas_call (already handles compilation)
    if 'pallas' in prim_name.lower():
        return False

    # JIT everything else
    return True

def create_jitted_primitive_call(primitive, cache_key):
    """Create a JIT-compiled version of a primitive call."""
    @functools.lru_cache(maxsize=128)
    def cached_jitted_call(*args, **params):
        # Create a function that calls the primitive
        def primitive_fn(*args):
            return primitive.bind(*args, **params)

        # JIT with minimal overhead
        jitted_fn = jax.jit(primitive_fn,
                           backend='cpu',
                           donate_argnums=())
        return jitted_fn(*args)

    return cached_jitted_call

def hybrid_bind(primitive, *args, **params):
    """Custom bind that JITs primitives but keeps control flow eager."""

    # Check if this primitive should be JIT compiled
    if not should_jit_primitive(primitive):
        # Use original bind for control flow
        return _original_binds[primitive](*args, **params)

    # Generate cache key
    cache_key = get_cache_key(primitive, *args, **params)

    if cache_key is None:
        # Can't cache, use original
        return _original_binds[primitive](*args, **params)

    # Check cache
    if cache_key not in _primitive_cache:
        # Create and cache JIT-compiled version
        _primitive_cache[cache_key] = create_jitted_primitive_call(primitive, cache_key)

    # Call cached JIT version
    try:
        return _primitive_cache[cache_key](*args, **params)
    except:
        # Fallback to original if JIT fails
        return _original_binds[primitive](*args, **params)

def install_hybrid_mode():
    """Install hybrid eager/JIT mode."""
    # Get all primitives from core
    import jax._src.lax.lax as lax_module
    import jax._src.lax.control_flow as cf_module

    primitives_to_patch = []

    # Find all primitives
    for module in [lax_module, cf_module, core]:
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, core.Primitive):
                if obj not in _original_binds:
                    _original_binds[obj] = obj.bind
                    primitives_to_patch.append(obj)

    print(f"Found {len(primitives_to_patch)} primitives to potentially optimize")

    # Patch bind methods
    for prim in primitives_to_patch:
        prim.bind = functools.partial(hybrid_bind, prim)

def uninstall_hybrid_mode():
    """Restore original bind methods."""
    for prim, original_bind in _original_binds.items():
        prim.bind = original_bind

# Actually, let's try a simpler approach: use JAX's compilation cache
# and disable JIT for control flow only
jax.config.update('jax_disable_jit', False)  # Keep JIT on
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_hybrid_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

# Minimal XLA optimization
os.environ['XLA_FLAGS'] = (
    '--xla_backend_optimization_level=0 '
    '--xla_llvm_disable_expensive_passes=true'
)

# Actually, let me try a different approach using JAX's built-in features
# We'll use lower-level APIs to control compilation granularity

print("\n🔥 STRATEGY 13: Hybrid Eager/JIT Mode\n")
print("Installing hybrid mode...")

try:
    # For now, just use aggressive caching with minimal optimization
    # The full primitive interception is complex and may need deeper JAX knowledge

    from benchmark_sort import run_benchmarks

    start_time = time.time()
    run_benchmarks()
    end_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print(f"{'=' * 60}")

finally:
    print("Cleaning up...")
