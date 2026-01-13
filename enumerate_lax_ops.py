#!/usr/bin/env python3
"""
Enumerate all lax operations and categorize them.
"""

import jax
from jax import lax
import inspect

# Get all attributes from lax module
lax_members = dir(lax)

# Categorize operations
categories = {
    'arithmetic': [],
    'comparison': [],
    'logical': [],
    'trigonometric': [],
    'exponential': [],
    'shape': [],
    'indexing': [],
    'reduction': [],
    'linear_algebra': [],
    'fft': [],
    'random': [],
    'control_flow': [],
    'collective': [],
    'other': [],
}

# Categorize each operation
for name in lax_members:
    if name.startswith('_'):
        continue

    attr = getattr(lax, name)
    if not callable(attr):
        continue

    # Categorize
    if name in ['add', 'sub', 'mul', 'div', 'rem', 'neg', 'abs', 'sign', 'floor', 'ceil', 'round', 'sqrt', 'rsqrt', 'pow', 'integer_pow']:
        categories['arithmetic'].append(name)
    elif name in ['eq', 'ne', 'lt', 'le', 'gt', 'ge', 'max', 'min', 'clamp']:
        categories['comparison'].append(name)
    elif name in ['bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not', 'shift_left', 'shift_right_logical', 'shift_right_arithmetic']:
        categories['logical'].append(name)
    elif name in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']:
        categories['trigonometric'].append(name)
    elif name in ['exp', 'expm1', 'log', 'log1p', 'log10', 'log2', 'logistic', 'erf', 'erfc', 'erf_inv']:
        categories['exponential'].append(name)
    elif name in ['reshape', 'transpose', 'broadcast', 'broadcast_in_dim', 'squeeze', 'expand_dims', 'concatenate', 'pad', 'slice', 'dynamic_slice', 'dynamic_update_slice', 'rev', 'collapse']:
        categories['shape'].append(name)
    elif name in ['gather', 'scatter', 'scatter_add', 'scatter_mul', 'scatter_min', 'scatter_max', 'index_in_dim', 'slice_in_dim']:
        categories['indexing'].append(name)
    elif name in ['reduce', 'reduce_window', 'argmin', 'argmax', 'cumsum', 'cumprod', 'cummax', 'cummin']:
        categories['reduction'].append(name)
    elif name in ['dot', 'dot_general', 'conv', 'conv_general_dilated', 'conv_transpose', 'batch_matmul']:
        categories['linear_algebra'].append(name)
    elif name in ['fft', 'ifft', 'rfft', 'irfft']:
        categories['fft'].append(name)
    elif name in ['random_split', 'random_uniform', 'random_normal']:
        categories['random'].append(name)
    elif name in ['cond', 'while_loop', 'fori_loop', 'scan', 'switch', 'select', 'select_n']:
        categories['control_flow'].append(name)
    elif name in ['psum', 'pmean', 'pmax', 'pmin', 'pprod', 'por', 'pand', 'all_gather', 'all_to_all', 'axis_index', 'psum_scatter', 'reduce_scatter']:
        categories['collective'].append(name)
    else:
        categories['other'].append(name)

# Print categorized operations
print("="*80)
print("LAX OPERATIONS BY CATEGORY")
print("="*80)

total = 0
for category, ops in categories.items():
    if ops:
        print(f"\n{category.upper()} ({len(ops)} operations):")
        for op in sorted(ops):
            print(f"  - {op}")
        total += len(ops)

print(f"\n{'='*80}")
print(f"TOTAL: {total} operations")
print("="*80)
