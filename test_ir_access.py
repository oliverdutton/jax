#!/usr/bin/env python3
"""
Test accessing StableHLO from both lowered and compiled objects.
"""

import jax
import jax.numpy as jnp
from jax._src import xla_bridge

def test_access_patterns():
    """Test different ways to access HLO/StableHLO."""

    def f(x):
        return x + 1

    x = jnp.array([1.0, 2.0, 3.0])

    # Pattern 1: From lowered object (current approach)
    print("="*80)
    print("PATTERN 1: From Lowered Object")
    print("="*80)
    lowered = jax.jit(f).lower(x)
    print(f"Lowered type: {type(lowered)}")

    # Get StableHLO MLIR
    stablehlo_mlir = lowered.compiler_ir(dialect='stablehlo')
    print(f"StableHLO MLIR type: {type(stablehlo_mlir)}")
    print(f"StableHLO text (first 200 chars):\n{str(stablehlo_mlir)[:200]}")

    # Pattern 2: From compiled object
    print("\n" + "="*80)
    print("PATTERN 2: From Compiled Object")
    print("="*80)
    compiled = lowered.compile()
    print(f"Compiled type: {type(compiled)}")

    # Get HLO text
    hlo_text = compiled.as_text()
    print(f"HLO text (first 200 chars):\n{hlo_text[:200] if hlo_text else 'None'}")

    # Get from runtime executable
    runtime_exec = compiled.runtime_executable()
    print(f"\nRuntime executable type: {type(runtime_exec)}")

    hlo_text_2 = runtime_exec.get_hlo_text()
    print(f"HLO text from runtime_exec (first 200 chars):\n{hlo_text_2[:200]}")

    # Get HLO modules
    hlo_modules = runtime_exec.hlo_modules()
    print(f"\nNumber of HLO modules: {len(hlo_modules)}")
    if hlo_modules:
        module = hlo_modules[0]
        print(f"Module name: {module.name()}")
        module_str = module.to_string()
        print(f"Module string (first 200 chars):\n{module_str[:200]}")

    # Pattern 3: Directly from jit
    print("\n" + "="*80)
    print("PATTERN 3: Direct from jit().lower()")
    print("="*80)

    # This is the correct way
    lowered2 = jax.jit(f).lower(x)
    stablehlo_mlir2 = lowered2.compiler_ir(dialect='stablehlo')
    print(f"Got StableHLO MLIR: {type(stablehlo_mlir2)}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("- For decompilation, we need StableHLO MLIR")
    print("- StableHLO is available from lowered.compiler_ir(dialect='stablehlo')")
    print("- Compiled objects only have optimized HLO (post-XLA)")
    print("- We should use: jax.jit(f).lower(args).compiler_ir(dialect='stablehlo')")
    print("- The compiled object can be obtained via: lowered.compile()")


if __name__ == '__main__':
    test_access_patterns()
