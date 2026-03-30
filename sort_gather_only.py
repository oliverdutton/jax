"""Run bitonic sort with only gather replaced by NumPy io_callback."""

# Install the gather-only NumPy hook before importing sort
import gather_only_numpy
gather_only_numpy.install_gather_numpy_hook()

# Now run the benchmark
import sort
sort.run_benchmark()
