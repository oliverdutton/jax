# Pallas TPU Compilation Pipeline Timing Instrumentation

This directory contains comprehensive timing instrumentation for the Pallas TPU lowering and compilation pipeline. These tools allow you to measure and analyze every stage of the compilation process with fine-grained detail.

## Files

1. **pallas_tpu_timing_instrumentation.py** - Core instrumentation framework
2. **pallas_tpu_timing_examples.py** - Usage examples for various kernel types
3. **advanced_pallas_timing.py** - Advanced analysis including HLO inspection and memory tracking

## What Gets Measured

The instrumentation captures timing for these stages:

### High-Level Stages
- **0_total_jit_compilation** - Complete JIT compilation time
- **0a_jit_lower** - Lowering to HLO
- **0b_compile_from_lowered** - Backend compilation

### Pallas-Specific Stages
- **1_total_pallas_call_tpu_lowering** - Complete Pallas TPU lowering
- **2_tensorcore_lower_jaxpr_to_module** - TensorCore: Jaxpr → Mosaic IR
- **2_sparsecore_lower_jaxpr_to_module** - SparseCore: Jaxpr → Mosaic IR
- **3_lower_mosaic_module_to_asm** - Mosaic IR serialization
- **3a_check_has_communication** - Detect collective operations
- **3b_clone_module** - MLIR module cloning
- **3c_mosaic_serde_pass** - Serialization pass (Mosaic IR → bytecode)
- **3d_write_bytecode** - Write MLIR bytecode
- **3x_lower_to_custom_call_config** - Create backend config
- **3x1_get_device_type** - Determine device type (TC/SC)
- **3x2_get_active_core_count** - Determine core count
- **4_tpu_custom_call_lowering** - Create custom call op
- **4a_backend_config_to_json** - Serialize backend config to JSON
- **4b_create_custom_call_op** - Create the MLIR custom call

### Execution
- **5_first_execution** - First kernel execution (may include lazy init)

## Quick Start

### Basic Usage

```python
from pallas_tpu_timing_instrumentation import time_pallas_compilation, TIMING_STATS
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Define your kernel
def my_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0

# Wrap compilation with timing
with time_pallas_compilation():
    def my_function(x):
        return pl.pallas_call(
            my_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        )(x)

    x = jnp.ones(1024, dtype=jnp.float32)
    compiled = jax.jit(my_function).lower(x).compile()
    result = compiled(x)
    result.block_until_ready()

# Print timing results
TIMING_STATS.print_summary()
```

### Running Examples

```bash
# Run all examples
python pallas_tpu_timing_examples.py

# Run a specific example (1-6)
python pallas_tpu_timing_examples.py 3

# Run advanced analysis
python advanced_pallas_timing.py
```

## Example Output

```
================================================================================
PALLAS TPU COMPILATION PIPELINE TIMING SUMMARY
================================================================================

1_total_pallas_call_tpu_lowering:
  Total:  45.234 ms
  Mean:   45.234 ms
  Min:    45.234 ms
  Max:    45.234 ms
  Count:  1

3c_mosaic_serde_pass:
  Total:  23.456 ms
  Mean:   23.456 ms
  Min:    23.456 ms
  Max:    23.456 ms
  Count:  1

2_tensorcore_lower_jaxpr_to_module:
  Total:  15.678 ms
  Mean:   15.678 ms
  Min:    15.678 ms
  Max:    15.678 ms
  Count:  1

...

================================================================================
Total measured time: 123.456 ms
================================================================================
```

## Advanced Features

### Memory Tracking

```python
from advanced_pallas_timing import detailed_compilation_analysis

results = detailed_compilation_analysis(
    jax.jit(my_function),
    x,
    kernel_name="my_custom_kernel"
)
```

### HLO Analysis

```python
from advanced_pallas_timing import analyze_hlo

lowered = jax.jit(my_function).lower(x)
hlo_stats = analyze_hlo(lowered)
```

### Representation Comparison

```python
from advanced_pallas_timing import compare_kernel_representations

representations = compare_kernel_representations(
    jax.jit(my_function),
    x
)
```

## Understanding the Pipeline

The Pallas TPU compilation pipeline:

```
Python Kernel
    ↓
Jaxpr (JAX IR)
    ↓
Mosaic IR Module (MLIR with TPU dialect)
    ↓
Serialized Mosaic IR (via mosaic-serde pass)
    ↓
MLIR Bytecode
    ↓
Base64-encoded in backend_config JSON
    ↓
StableHLO Custom Call (api_version=1)
    ↓
XLA/TPU Compiler
    ↓
TPU Executable
```

## Key Insights from Timing

1. **mosaic-serde pass** - Often the most expensive single operation
2. **lower_jaxpr_to_module** - Varies based on kernel complexity
3. **backend_config_to_json** - Usually fast, but can be significant for large kernels
4. **First execution** - May include lazy initialization overhead

## Export Results

Timing data is automatically exported to JSON:

```python
# Automatically saved during examples
/tmp/pallas_tpu_timing.json
/tmp/pallas_detailed_analysis.json
```

## Use Cases

1. **Performance Optimization** - Identify compilation bottlenecks
2. **Regression Testing** - Track compilation time changes
3. **Kernel Comparison** - Compare different implementations
4. **Research** - Study compiler behavior
5. **Debugging** - Understand where compilation time is spent

## Limitations

- Instrumentation adds overhead (typically <5%)
- Some internal passes may not be captured
- Memory tracking uses `tracemalloc` (Python-level only)
- Timing resolution depends on system performance

## Requirements

- JAX with TPU support
- jaxlib with Mosaic support
- Access to TPU hardware or TPU simulator

## Notes

- The instrumentation uses monkey-patching and is automatically restored
- Thread-safe for single-threaded compilation
- Reset `TIMING_STATS` between measurements for clean results
- All times are in seconds (displayed as milliseconds in output)
