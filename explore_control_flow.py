#!/usr/bin/env python3
"""Deep dive into control flow structures in StableHLO."""

import jax
import jax.numpy as jnp
from jax import lax

def while_sum(n):
    def cond_fun(val):
        i, total = val
        return i < n
    def body_fun(val):
        i, total = val
        return (i + 1, total + i)
    _, result = lax.while_loop(cond_fun, body_fun, (0, 0))
    return result

def scan_cumsum(arr):
    def body(carry, x):
        new_carry = carry + x
        return new_carry, new_carry
    _, result = lax.scan(body, 0.0, arr)
    return result

def cond_abs(x):
    return lax.cond(x >= 0, lambda x: x, lambda x: -x, x)


def explore_operation_deep(op, indent=0):
    """Deep exploration of an operation including all nested regions."""
    prefix = "  " * indent
    op_name = str(op.operation.name)
    print(f"{prefix}Operation: {op_name}")

    # Operands
    if hasattr(op, 'operands'):
        operands = list(op.operands)
        if operands:
            print(f"{prefix}  Operands: {len(operands)}")
            for i, operand in enumerate(operands):
                print(f"{prefix}    [{i}] {operand.type}")

    # Results
    if hasattr(op, 'results'):
        results = list(op.results)
        if results:
            print(f"{prefix}  Results: {len(results)}")
            for i, result in enumerate(results):
                print(f"{prefix}    [{i}] {result.type}")

    # Attributes
    if hasattr(op, 'attributes'):
        attrs = dict(op.attributes)
        if attrs:
            print(f"{prefix}  Attributes:")
            for name, value in attrs.items():
                print(f"{prefix}    {name} = {value}")

    # Nested regions (for control flow)
    if hasattr(op, 'regions'):
        regions = list(op.regions)
        if regions:
            print(f"{prefix}  Regions: {len(regions)}")
            for region_idx, region in enumerate(regions):
                print(f"{prefix}    Region [{region_idx}]:")
                for block_idx, block in enumerate(region.blocks):
                    print(f"{prefix}      Block [{block_idx}]:")
                    block_args = list(block.arguments)
                    if block_args:
                        print(f"{prefix}        Arguments: {len(block_args)}")
                        for arg_idx, arg in enumerate(block_args):
                            print(f"{prefix}          [{arg_idx}] {arg.type}")

                    print(f"{prefix}        Operations:")
                    for nested_op in block.operations:
                        explore_operation_deep(nested_op, indent + 4)


print("=" * 80)
print("CONTROL FLOW: WHILE LOOP")
print("=" * 80)

n = 5
lowered_while = jax.jit(while_sum).lower(n)
mlir_while = lowered_while.compiler_ir(dialect='stablehlo')

for op in mlir_while.body.operations:
    if hasattr(op, 'function_type'):
        print(f"\nFunction: {op.name}")
        print(f"Type: {op.function_type}")
        for region in op.regions:
            for block in region.blocks:
                print(f"\nMain function body:")
                for block_op in block.operations:
                    explore_operation_deep(block_op)


print("\n" + "=" * 80)
print("CONTROL FLOW: SCAN")
print("=" * 80)

arr = jnp.array([1.0, 2.0, 3.0, 4.0])
lowered_scan = jax.jit(scan_cumsum).lower(arr)
mlir_scan = lowered_scan.compiler_ir(dialect='stablehlo')

for op in mlir_scan.body.operations:
    if hasattr(op, 'function_type'):
        print(f"\nFunction: {op.name}")
        print(f"Type: {op.function_type}")
        for region in op.regions:
            for block in region.blocks:
                print(f"\nMain function body:")
                for block_op in block.operations:
                    explore_operation_deep(block_op)


print("\n" + "=" * 80)
print("CONTROL FLOW: COND")
print("=" * 80)

x = jnp.array(5.0)
lowered_cond = jax.jit(cond_abs).lower(x)
mlir_cond = lowered_cond.compiler_ir(dialect='stablehlo')

for op in mlir_cond.body.operations:
    if hasattr(op, 'function_type'):
        print(f"\nFunction: {op.name}")
        print(f"Type: {op.function_type}")
        for region in op.regions:
            for block in region.blocks:
                print(f"\nMain function body:")
                for block_op in block.operations:
                    explore_operation_deep(block_op)


print("\n" + "=" * 80)
print("ACCESSING COMPILED OBJECTS")
print("=" * 80)

# Can we get HLO from compiled?
compiled_while = lowered_while.compile()
print(f"\nCompiled type: {type(compiled_while)}")
print(f"Compiled attributes: {[a for a in dir(compiled_while) if not a.startswith('_')]}")

# Check if we can get HLO from compiled
print("\nTrying to get HLO from compiled object...")
try:
    hlo_text = compiled_while.as_text()
    print("Success! HLO text available from compiled object")
    print("Length:", len(hlo_text))
    print("\nFirst 500 chars:")
    print(hlo_text[:500])
except Exception as e:
    print(f"Failed: {e}")

# Try runtime executable
print(f"\nCompiled runtime_executable: {type(compiled_while.runtime_executable())}")
exec_obj = compiled_while.runtime_executable()
print(f"Executable attributes: {[a for a in dir(exec_obj) if not a.startswith('_')]}")

# Try to get modules
try:
    hlo_modules = exec_obj.hlo_modules()
    print(f"\nHLO modules available: {len(hlo_modules)}")
    for i, module in enumerate(hlo_modules):
        print(f"Module {i}: {type(module)}")
        print(f"Module {i} attributes: {[a for a in dir(module) if not a.startswith('_')]}")
except Exception as e:
    print(f"Failed to get hlo_modules: {e}")
