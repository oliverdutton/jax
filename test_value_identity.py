#!/usr/bin/env python3
"""Test how to identify MLIR values across uses."""

import jax
import jax.numpy as jnp

def simple_add(x, y):
    return x + y

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

lowered = jax.jit(simple_add).lower(x, y)
mlir_module = lowered.compiler_ir(dialect='stablehlo')

# Walk the IR and store all values
all_values = {}

for op in mlir_module.body.operations:
    if hasattr(op, 'function_type'):
        for region in op.regions:
            for block in region.blocks:
                # Block arguments
                args = list(block.arguments)
                print(f"Block has {len(args)} arguments")
                for i, arg in enumerate(args):
                    print(f"  arg[{i}]: id={id(arg)}, type={arg.type}")
                    print(f"    dir: {[a for a in dir(arg) if not a.startswith('_')]}")
                    print(f"    hasattr owner: {hasattr(arg, 'owner')}")
                    print(f"    str: {str(arg)}")
                    print(f"    repr: {repr(arg)}")
                    print()

                    # Try to get a stable reference
                    if hasattr(arg, '_CAPIPtr'):
                        print(f"    C pointer: {arg._CAPIPtr}")

                # Check if the operands in the next op reference the same values
                for block_op in block.operations:
                    print(f"\nOp: {block_op.operation.name}")
                    if hasattr(block_op, 'operands'):
                        operands = list(block_op.operands)
                        for i, operand in enumerate(operands):
                            print(f"  operand[{i}]: id={id(operand)}, str={str(operand)}, repr={repr(operand)}")
                            if hasattr(operand, '_CAPIPtr'):
                                print(f"    C pointer: {operand._CAPIPtr}")

                            # Check if it's equal to any of the args
                            for j, arg in enumerate(args):
                                if operand == arg:
                                    print(f"    -> EQUAL to arg[{j}] (using ==)")
                                if str(operand) == str(arg):
                                    print(f"    -> STR EQUAL to arg[{j}]")
                                if hasattr(operand, '_CAPIPtr') and hasattr(arg, '_CAPIPtr'):
                                    if operand._CAPIPtr == arg._CAPIPtr:
                                        print(f"    -> C POINTER EQUAL to arg[{j}]")
