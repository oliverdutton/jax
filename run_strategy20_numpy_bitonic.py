import time
import numpy as np

print("=" * 60)
print("Strategy: Pure NumPy Bitonic Sort - NO compilation")
print("=" * 60)

def bitonic_compare_and_swap(arr, i, j, ascending=True):
    """Compare and swap elements at indices i and j."""
    if ascending:
        mask = arr[:, i] > arr[:, j]
    else:
        mask = arr[:, i] < arr[:, j]

    # Swap where needed
    temp = arr[mask, i].copy()
    arr[mask, i] = arr[mask, j]
    arr[mask, j] = temp

def bitonic_merge(arr, low, cnt, ascending=True):
    """Recursively merge a bitonic sequence."""
    if cnt > 1:
        k = cnt // 2
        for i in range(low, low + k):
            bitonic_compare_and_swap(arr, i, i + k, ascending)

        bitonic_merge(arr, low, k, ascending)
        bitonic_merge(arr, low + k, k, ascending)

def bitonic_sort_recursive(arr, low, cnt, ascending=True):
    """Recursively apply bitonic sort."""
    if cnt > 1:
        k = cnt // 2

        # Sort first half ascending
        bitonic_sort_recursive(arr, low, k, True)

        # Sort second half descending
        bitonic_sort_recursive(arr, low + k, k, False)

        # Merge the whole sequence
        bitonic_merge(arr, low, cnt, ascending)

def numpy_bitonic_sort(operands, num_keys=1, interpret=True, **kwargs):
    """Pure NumPy implementation of bitonic sort."""
    # Get the array to sort
    arr = operands[0].copy()

    # Bitonic sort requires power-of-2 length
    n = arr.shape[1]
    if n & (n - 1) != 0:  # Not power of 2
        # Pad to next power of 2
        next_pow2 = 2 ** int(np.ceil(np.log2(n)))
        padded = np.full((arr.shape[0], next_pow2), np.inf, dtype=arr.dtype)
        padded[:, :n] = arr
        arr = padded
        was_padded = True
    else:
        was_padded = False

    # Apply bitonic sort to each row
    for row_idx in range(arr.shape[0]):
        row = arr[row_idx:row_idx+1, :]
        bitonic_sort_recursive(row, 0, arr.shape[1], ascending=True)

    # Unpad if needed
    if was_padded:
        arr = arr[:, :n]

    return (arr,)

def numpy_bitonic_sort_vectorized(operands, num_keys=1, interpret=True, **kwargs):
    """Vectorized pure NumPy implementation of bitonic sort (all rows at once)."""
    arr = operands[0].copy()

    # Ensure power of 2
    n = arr.shape[1]
    if n & (n - 1) != 0:
        next_pow2 = 2 ** int(np.ceil(np.log2(n)))
        padded = np.full((arr.shape[0], next_pow2), np.inf, dtype=arr.dtype)
        padded[:, :n] = arr
        arr = padded
        was_padded = True
    else:
        was_padded = False

    # Iterative bitonic sort (vectorized across all rows)
    num_stages = int(np.log2(arr.shape[1]))

    for stage in range(1, num_stages + 1):
        for substage in range(stage - 1, -1, -1):
            distance = 2 ** substage
            for i in range(0, arr.shape[1], 2 * distance):
                for j in range(i, i + distance):
                    comp_idx = j + distance

                    # Determine sort direction based on bitonic sequence
                    block = j // (2 ** stage)
                    ascending = (block % 2) == 0

                    # Compare and swap
                    if ascending:
                        mask = arr[:, j] > arr[:, comp_idx]
                    else:
                        mask = arr[:, j] < arr[:, comp_idx]

                    # Swap
                    temp = arr[mask, j].copy()
                    arr[mask, j] = arr[mask, comp_idx]
                    arr[mask, comp_idx] = temp

    if was_padded:
        arr = arr[:, :n]

    return (arr,)

print("\n🔥 STRATEGY 20: Pure NumPy Bitonic Sort (Zero Compilation)\n")
print("Implementing actual bitonic sort algorithm in NumPy")

def run_benchmarks_bitonic():
    ntoken = 8

    for num_operands in range(1, 2):
        for num_keys in range(1, num_operands + 1):
            for n in (128,):
                rng = np.random.RandomState(0)

                for dtype in (np.float32,):
                    # Generate same random data as JAX version
                    operands = [
                        rng.randint(
                            np.iinfo(np.int32).min,
                            np.iinfo(np.int32).max,
                            size=(ntoken, n),
                            dtype=np.int32
                        ).view(dtype)
                        for _ in range(num_operands)
                    ]

                    x = operands[0]
                    print(f'\n{(x.shape, x.dtype)}\n{num_operands=} {num_keys=}')

                    # Use vectorized version for better performance
                    result = numpy_bitonic_sort_vectorized(operands, num_keys=num_keys)

                    print(f"Result shape: {result[0].shape}, dtype: {result[0].dtype}")
                    print(f"First row sorted: {np.all(np.diff(result[0][0]) >= 0)}")
                    print(f"Sample values: {result[0][0, :5]}")

start_time = time.time()
run_benchmarks_bitonic()
end_time = time.time()

print(f"\n{'=' * 60}")
print(f"Total execution time: {end_time - start_time:.4f} seconds")
print(f"{'=' * 60}")
print("\nThis implements the actual bitonic sort algorithm!")
print("Pure NumPy, Python control flow, ZERO compilation overhead.")
