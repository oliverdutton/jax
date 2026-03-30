"""Run bitonic sort with gather replaced by dummy operation (jnp.zeros)."""

# Install the dummy gather hook before importing sort
import gather_dummy
gather_dummy.install_dummy_gather_hook()

# Now run the benchmark
import sort
sort.run_benchmark()
