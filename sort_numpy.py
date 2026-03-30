"""
Run bitonic sort with NumPy interpreter.

This version uses the io_callback-based NumPy interpreter instead of
the standard Pallas interpret mode.
"""

import sys
sys.path.insert(0, '.')

import pallas_numpy_interpreter
import sort

# Install the NumPy interpreter
print("Installing NumPy interpreter for Pallas calls...")
pallas_numpy_interpreter.install_numpy_interpreter()

# Run the benchmark
print("\nRunning bitonic sort with NumPy interpreter...\n")
sort.run_benchmark()
