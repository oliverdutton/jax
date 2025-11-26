"""Run sort.py with NumPy interpreter installed."""
import sys
sys.path.insert(0, '.')

# Install the numpy interpreter BEFORE importing sort
import pallas_numpy_interpreter
pallas_numpy_interpreter.install_numpy_interpreter()

# Now import and run sort
import sort
sort.run_benchmark()
