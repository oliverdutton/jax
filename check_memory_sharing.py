"""Check if arrays are inadvertently sharing memory."""

import numpy as np
import pallas_numpy_interpreter

# Patch SWAP to check for memory sharing
original_eval = pallas_numpy_interpreter._eval_jaxpr_numpy_impl

swaps_seen = []

def tracking_eval(jaxpr, consts, *args, **kwargs):
    # Track arrays passed in
    for arg in args:
        if isinstance(arg, np.ndarray):
            swaps_seen.append({
                'id': id(arg),
                'data_ptr': arg.__array_interface__['data'][0],
                'shape': arg.shape,
                'values': arg.flatten()[:3].tolist() if arg.size > 0 else []
            })
    return original_eval(jaxpr, consts, *args, **kwargs)

pallas_numpy_interpreter._eval_jaxpr_numpy_impl = tracking_eval

# Run sort
pallas_numpy_interpreter.install_numpy_interpreter()

from sort import sort
test_data = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)

result = sort(test_data, num_keys=1, interpret=True)

# Check for duplicate data pointers
ptrs = [s['data_ptr'] for s in swaps_seen if s['shape'] == (8, 128)]
print(f"Total (8, 128) arrays seen: {len(ptrs)}")
print(f"Unique data pointers: {len(set(ptrs))}")

if len(ptrs) != len(set(ptrs)):
    print("\n⚠️  Memory sharing detected!")
    # Find duplicates
    from collections import Counter
    counts = Counter(ptrs)
    for ptr, count in counts.items():
        if count > 1:
            print(f"  Pointer {ptr:x} used {count} times")
            # Show which arrays share this pointer
            for s in swaps_seen:
                if s['data_ptr'] == ptr and s['shape'] == (8, 128):
                    print(f"    id={s['id']}, values={s['values']}")
