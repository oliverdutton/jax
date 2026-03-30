"""Test batched gather in isolation."""

import numpy as np

# Simulate the batched gather scenario from bitonic sort
operand = np.array([
    [10, 20, 30],
    [40, 50, 60],
], dtype=np.int32)

indices = np.array([
    [1, 0, 1],
    [0, 1, 0],
], dtype=np.int32)

print("Operand (8 x 3):")
print(operand)
print("\nIndices (8 x 3):")
print(indices)

# Expected result: for each position (i,j), gather operand[indices[i,j], j]
# result[0,0] = operand[1, 0] = 40
# result[0,1] = operand[0, 1] = 20
# result[0,2] = operand[1, 2] = 60
# result[1,0] = operand[0, 0] = 10
# result[1,1] = operand[1, 1] = 50
# result[1,2] = operand[0, 2] = 30

print("\nExpected result:")
expected = np.array([
    [40, 20, 60],
    [10, 50, 30],
], dtype=np.int32)
print(expected)

# Our implementation
batch_dim = 1
gather_dim = 0
batch_idx = np.arange(operand.shape[batch_dim])[None, :]  # [[0, 1, 2]]
result = operand[indices.astype(np.intp), batch_idx]

print("\nOur gather result:")
print(result)
print("\nMatch:", np.array_equal(result, expected))

# Manual verification
print("\nManual verification:")
for i in range(2):
    for j in range(3):
        manual = operand[indices[i,j], j]
        actual = result[i,j]
        print(f"  result[{i},{j}] = operand[{indices[i,j]}, {j}] = {manual}, got {actual}, {'✓' if manual == actual else '✗'}")
